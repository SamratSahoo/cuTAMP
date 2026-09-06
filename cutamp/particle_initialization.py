# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
from typing import Optional

import roma
import torch
from curobo.geom.types import Cuboid
from curobo.types.math import Pose

from cutamp.config import TAMPConfiguration
from cutamp.costs import sphere_to_sphere_overlap
from cutamp.posture_prior import DEFAULT_POSTURE_REF, load_posture_prior, posture_penalty
from cutamp.robots.bimanual_yam import bimanual_yam_neutral_joint_positions
from cutamp.samplers import (
    grasp_4dof_sampler,
    grasp_6dof_sampler,
    place_4dof_sampler,
    sample_yaw,
)
from cutamp.tamp_domain import (
    Handover,
    MoveFree,
    MoveHolding,
    MoveHoldingBoth,
    MoveHoldingGiver,
    MoveHoldingTaker,
    Pick,
    PickBoth,
    PickGiver,
    Place,
    PlaceBoth,
    PlaceTaker,
    Push,
    PushStick,
)
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.utils.common import (
    Particles,
    action_4dof_to_mat4x4,
    action_6dof_to_mat4x4,
    pose_list_to_mat4x4,
    sample_between_bounds,
    transform_spheres,
)
from cutamp.utils.shapes import MultiSphere

_log = logging.getLogger(__name__)


class NoGraspsError(RuntimeError):
    """Perception proposed no grasps for an object that must be picked.

    Raised only when ``TAMPConfiguration.require_m2t2_grasps`` is set; otherwise the heuristic
    sampler stands in and a warning is logged. See ``_sample_grasps``.
    """



# --------------------------------------------------------------------------------------------- #
# Teleop-posture IK branch selection                                                             #
# --------------------------------------------------------------------------------------------- #
_posture_warned = False
_ROLL_CACHE: dict = {}


def parallel_jaw_roll(device, dtype):
    """Rz(pi) as a 4x4: the parallel-jaw grasp symmetry.

    Verified for the FR3 that this commutes with ``tool_from_ee`` --
    ``tool_from_ee @ Rz @ tool_from_ee^-1 == Rz`` exactly -- so the SAME matrix rolls the
    end-effector pose and the stored ``obj_from_grasp``, and the two stay consistent.
    """
    key = (str(device), str(dtype))
    if key not in _ROLL_CACHE:
        r = torch.eye(4, device=device, dtype=dtype)
        r[0, 0] = -1.0
        r[1, 1] = -1.0
        _ROLL_CACHE[key] = r
    return _ROLL_CACHE[key]


def _branch_scores(ik_result, goal_pose: Pose, ik_solver, config, pack):
    """(solutions [n,k,dof], penalty [n,k] with invalid branches at +inf)."""
    sol = ik_result.solution
    n, k, dof = sol.shape
    # cuRobo's IKResult.success comes back all-False in this batched path, so validate by FK.
    fk = ik_solver.fk(sol.reshape(-1, dof)).ee_pose
    d_pos = (fk.position.reshape(n, k, 3) - goal_pose.position.unsqueeze(1)).norm(dim=-1)
    dot = (fk.quaternion.reshape(n, k, 4) * goal_pose.quaternion.unsqueeze(1)).sum(-1).abs().clamp(max=1.0)
    valid = (d_pos < config.posture_pos_tol) & (2 * torch.acos(dot) < config.posture_rot_tol)
    return sol, posture_penalty(sol, pack).masked_fill(~valid, float("inf"))


def _posture_pack(sol, config):
    """Load the baked prior for this arm, or None (with one warning) if it cannot be used."""
    global _posture_warned
    if config.posture_selection_seeds <= 1 or sol.shape[1] <= 1:
        return None
    try:
        pack = load_posture_prior(config.posture_ref or DEFAULT_POSTURE_REF, sol.device, sol.dtype)
    except (OSError, KeyError) as exc:
        if not _posture_warned:
            _log.warning(f"posture selection disabled: could not load the posture prior ({exc})")
            _posture_warned = True
        return None
    if pack["n_joints"] > sol.shape[-1]:
        # e.g. a prior baked for a 7-DOF arm against a 6-DOF one, which has no redundancy anyway.
        if not _posture_warned:
            _log.warning(
                f"posture selection disabled: prior covers {pack['n_joints']} joints but this arm "
                f"has {sol.shape[-1]} DOF -- re-bake with cutamp/scripts/bake_posture_ref.py"
            )
            _posture_warned = True
        return None
    return pack


def _argmin_branch(sol, penalty, fallback):
    """Lowest-penalty branch per particle; ``fallback`` where no branch is FK-valid."""
    n = sol.shape[0]
    chosen = sol[torch.arange(n, device=sol.device), penalty.argmin(dim=1)]
    none_valid = torch.isinf(penalty).all(dim=1)
    if none_valid.any():
        chosen = torch.where(none_valid.unsqueeze(1), fallback, chosen)
    return chosen


def solve_with_grasp_roll(world, world_from_ee, config, obj_from_grasp_mat):
    """Solve IK for a grasp AND its pi-rolled twin, select across both, roll the grasp to match.

    A parallel jaw is invariant to a pi roll about its approach axis, so each grasp has two equally
    valid end-effector poses. cuTAMP only ever builds one -- whichever representative M2T2 emitted --
    and about half the time that is the wrist branch teleoperation never uses. Both IK branches of a
    single pose share that pose's wrist roll, so branch selection alone cannot fix it; the twin pose
    has to be in the pool.

    Returns ``(q [n, dof], obj_from_grasp [n, 4, 4] or None, ik_result)``. The returned grasp is the
    input with the roll applied wherever the twin won, so the recorded grasp always matches the
    configuration; ``None`` means nothing was rolled and the caller keeps its existing grasp tensor.

    IMPORTANT -- each pool is scored IMMEDIATELY after its own solve. ``solve_batch`` normalises its
    goal quaternion in place and the Pose objects alias cuRobo's internal goal buffer, so a goal
    held across a second solve is silently overwritten (measured drift 1.41 on a unit quaternion --
    the first goal literally becomes the second). Scoring eagerly sidesteps that entirely; deferring
    it made every branch of the first pool fail its FK check and the twin win 100% of the time.
    """
    seeds = max(1, config.posture_selection_seeds)
    ik_result = world.ik_solver.solve_batch(world_from_ee, seed_config=None, return_seeds=seeds)
    pack = _posture_pack(ik_result.solution, config)
    roll_on = pack is not None and config.posture_grasp_roll and obj_from_grasp_mat is not None
    if pack is None:
        return ik_result.solution[:, 0], None, ik_result

    # Score pool 1 NOW, while its goal is still intact.
    sol, penalty = _branch_scores(ik_result, world_from_ee, world.ik_solver, config, pack)
    if not roll_on:
        return _argmin_branch(sol, penalty, ik_result.solution[:, 0]), None, ik_result

    roll = parallel_jaw_roll(world_from_ee.position.device, world_from_ee.position.dtype)
    # Separate same-size call: doubling the batch trips cuRobo's cuda graph ("changing goal type").
    flipped = Pose.from_matrix(world_from_ee.get_matrix() @ roll)
    alt_result = world.ik_solver.solve_batch(flipped, seed_config=None, return_seeds=seeds)
    sol_a, pen_a = _branch_scores(alt_result, flipped, world.ik_solver, config, pack)

    used_alt = pen_a.min(dim=1).values < penalty.min(dim=1).values
    q = _argmin_branch(
        torch.cat([sol, sol_a], dim=1), torch.cat([penalty, pen_a], dim=1), ik_result.solution[:, 0]
    )
    used_alt = used_alt & ~torch.isinf(pen_a).all(dim=1)
    if not used_alt.any():
        return q, None, ik_result
    # Same roll applies on the grasp side: tool_from_ee @ Rz @ tool_from_ee^-1 == Rz for this arm.
    rolled = obj_from_grasp_mat @ roll.to(obj_from_grasp_mat.dtype)
    return q, torch.where(used_alt.view(-1, 1, 1), rolled, obj_from_grasp_mat), ik_result


def select_posture_branch(ik_result, goal_pose: Pose, ik_solver, config: TAMPConfiguration):
    """Pick, per particle, the IK branch with the lowest data-derived posture penalty.

    cuRobo ranks the seeds it returns by ``pose_error + null_space_error``, and
    ``null_space_cfg.weight`` is 0.001 against a generic home retract -- so with the default
    ``return_seeds=1`` the redundant arm's branch is effectively chosen by pose error alone, and
    the arm's redundancy lands wherever, independently at every endpoint. This scores every
    returned branch with ``posture_prior.posture_penalty`` and keeps the best.

    Returns ``[num_particles, dof]``. Falls back to cuRobo's own top seed (``solution[:, 0]``, the
    previous behaviour) for any particle with no FK-valid branch, and whenever the baked prior does
    not fit this arm -- so enabling it can never lose a solution that would otherwise be found.
    """
    pack = _posture_pack(ik_result.solution, config)
    if pack is None:
        return ik_result.solution[:, 0]
    sol, penalty = _branch_scores(ik_result, goal_pose, ik_solver, config, pack)
    return _argmin_branch(sol, penalty, ik_result.solution[:, 0])


class ParticleInitializer:
    def __init__(self, world: TAMPWorld, config: TAMPConfiguration, grasps: dict[str, dict] | None = None):
        if config.enable_traj:
            raise NotImplementedError("Trajectory initialization not yet supported")
        if config.place_dof != 4:
            raise NotImplementedError(f"Only 4-DOF grasp and placement supported for now, not {config.place_dof}")
        if config.grasp_dof != 4 and config.grasp_dof != 6:
            raise NotImplementedError(f"Only 4-DOF or 6-DOF grasp supported for now, not {config.grasp_dof}")
        if not config.m2t2_grasps and grasps:
            raise ValueError("M2T2 grasps is not enabled but got grasps")
        self.world = world
        self.config = config
        self.q_init = world.q_init.repeat(config.num_particles, 1)
        if grasps is not None:
            _log.info(f"Using provided grasps instead of built-in cuTAMP samplers")
        self.grasps = grasps

        # Sampler caching
        self._arm_sphere_rows = None
        # NOTE: these caches are per-run -- run_cutamp builds a ParticleInitializer alongside one
        # TAMPConfiguration (algorithm.py) and drops it with that run. "q_posture" entries below
        # therefore cannot outlive the posture prior/seed count that produced them. If this object is
        # ever hoisted to live across configs, key or invalidate those entries on the prior.
        self.pick_cache = {}
        self.place_cache = {}
        self.push_button_cache = {}
        self.push_stick_cache = {}
        self.failed_push = set()

    def _cross_arm_sphere_rows(self):
        """Row indices into the link-sphere tensor for each arm, cached.

        Used to score how close two independently-solved arm configurations come to each other.
        """
        if getattr(self, "_arm_sphere_rows", None) is None:
            kc = self.world.kin_model.kinematics_config
            idx_to_link = {v: k for k, v in kc.link_name_to_idx_map.items()}
            link_of_sphere = [idx_to_link[int(i)] for i in kc.link_sphere_idx_map.cpu().numpy()]
            self._arm_sphere_rows = [
                torch.tensor(
                    [r for r, link in enumerate(link_of_sphere) if link.startswith(f"{spec.name}_")],
                    dtype=torch.long, device=self.world.tensor_args.device,
                )
                for spec in self.world.arms
            ]
        return self._arm_sphere_rows

    def _cross_arm_gap(self, q):
        """Smallest gap between any sphere of one arm and any sphere of another, per configuration.

        Negative means the arms interpenetrate. Only CROSS-arm pairs are scored: within one arm the
        neighbouring links always touch, which is what the cuRobo config's self_collision_ignore is
        for.
        """
        spheres = self.world.kin_model.get_state(q).get_link_spheres()  # (b, n, 4)
        rows_a, rows_b = self._cross_arm_sphere_rows()[0], self._cross_arm_sphere_rows()[1]
        a, b = spheres[:, rows_a], spheres[:, rows_b]
        active = (a[..., 3] > 0).unsqueeze(2) & (b[..., 3] > 0).unsqueeze(1)
        dist = torch.cdist(a[..., :3], b[..., :3]) - a[..., 3].unsqueeze(2) - b[..., 3].unsqueeze(1)
        return dist.masked_fill(~active, float("inf")).amin(dim=(1, 2))

    def _seed_dual_conf(self, arm_to_world_from_ee: dict, num_particles: int, header: str, log_debug,
                        num_branches: int = 4):
        """Seed a multi-arm configuration from per-arm IK, choosing branches that do not collide.

        cuRobo has no simultaneous multi-pose IK, and cuTAMP does not need one: these are only the
        INITIAL particles, which the optimizer then refines against the dual kinematic cost. Because
        the arms share no joints, solving each against its own single-arm config and writing the
        results into that arm's column slice reproduces every target pose exactly through the shared
        forward kinematics.

        The catch is that each arm's IK runs with the OTHER arm parked at its neutral pose, so two
        independently-chosen elbow branches routinely intersect -- and once the hands are pinned by
        the kinematic constraint, the optimizer has no easy path out of that. So several IK branches
        are requested per arm and the combination with the largest cross-arm clearance is kept.
        With 2 arms and k branches that is k**2 candidates, evaluated batched.

        Returns ``(q, both_solved_mask)``.
        """
        world = self.world
        dof = world.robot_container.joint_limits.shape[1]
        device = world.tensor_args.device

        # An arm with no target is IDLE at this timestep (asymmetric operators like PickGiver), and is
        # parked at the retract posture -- the same one cuRobo's per-arm configs lock it at, so the
        # single-arm IK solves are consistent with where the idle arm actually is.
        parked = torch.tensor(bimanual_yam_neutral_joint_positions, dtype=torch.float32, device=device)

        per_arm = []
        both_ok = torch.ones(num_particles, dtype=torch.bool, device=device)
        for spec in world.arms:
            target = arm_to_world_from_ee.get(spec.name)
            if target is None:
                per_arm.append(parked.view(1, 1, -1).expand(num_particles, num_branches, -1))
                continue
            res = world.ik_solvers[spec.name].solve_batch(
                Pose.from_matrix(target), seed_config=None, return_seeds=num_branches,
            )
            per_arm.append(res.solution)                      # (b, k, 6)
            both_ok &= res.success.any(dim=-1).flatten()
            log_debug(f"{header}. {spec.name} IK success: {int(res.success.any(dim=-1).sum())}/{num_particles}")

        k = min(num_branches, *[sol.shape[1] for sol in per_arm])
        best_q = torch.zeros((num_particles, dof), device=device)
        best_gap = torch.full((num_particles,), -float("inf"), device=device)
        for i in range(k):
            for j in range(k):
                cand = torch.zeros((num_particles, dof), device=device)
                cand[:, world.arms[0].joint_slice] = per_arm[0][:, i]
                cand[:, world.arms[1].joint_slice] = per_arm[1][:, j]
                gap = self._cross_arm_gap(cand)
                better = gap > best_gap
                best_gap = torch.where(better, gap, best_gap)
                best_q[better] = cand[better]

        clear = best_gap > 0
        n_ok = int((both_ok & clear).sum())
        log_debug(
            f"{header}. both arms solved: {int(both_ok.sum())}/{num_particles}; "
            f"of those, {n_ok} with the arms clear of each other "
            f"(best cross-arm gap {float(best_gap.max()) * 1000:.1f} mm)"
        )
        usable = both_ok & clear
        n_usable = int(usable.sum())
        if 0 < n_usable < num_particles:
            # Replace unusable particles with copies of usable ones, so the optimizer starts from a
            # full population rather than one padded with self-colliding configurations it cannot
            # escape once the kinematic constraint pins the hands.
            ok_idx = torch.nonzero(usable, as_tuple=False).flatten()
            fill = ok_idx[torch.randint(0, n_usable, (int((~usable).sum()),), device=ok_idx.device)]
            best_q[~usable] = best_q[fill]
        return best_q, usable

    def _sample_grasps(self, obj: str, num_particles: int):
        """Sample, collision-filter and rank candidate grasps for one object.

        Extracted verbatim out of the ``Pick`` branch so ``PickBoth`` can call it once per hand.
        The single-arm path calls it exactly where the inline code used to run, so its behaviour is
        unchanged. Returns ``(selected_grasps, selected_confs, is_mat4x4)``.
        """
        config, world = self.config, self.world

        n_m2t2 = len(self.grasps.get(obj, {}).get("grasps_obj", ())) if self.grasps else 0
        if config.m2t2_grasps and n_m2t2 == 0:
            # Perception proposed NOTHING for this object and the heuristic samplers below will
            # quietly stand in -- sampling grasps off the object's COLLISION-SPHERE approximation,
            # with no perception and no confidence behind them. On a plush toy that reconstructs as
            # a lumpy blob of spheres, that is a grasp pose with no evidence any graspable geometry
            # is there, and it is a direct mechanism for closing on empty air.
            #
            # Measured over the shipped runs' scene_objects.json
            # (analysis_dataset_diff/TELEOP_VS_APEX.md, Finding 5): 43% of objects in
            # 1_pp_toys_plate_vae_style_timing_apex_posture and 18% in ..._learned_posture had ZERO
            # M2T2 candidates, and every one of them was picked anyway on these fallback grasps.
            # It was invisible because the only trace was a _log.debug.
            #
            # The warning is unconditional -- a silent substitution of a different grasp source is
            # never what a caller wants to not be told about. The raise is opt-in via
            # `require_m2t2_grasps`, because failing here changes planning outcomes for every
            # existing config and some of them legitimately rely on the heuristic path.
            _log.warning(
                f"No M2T2 grasps for {obj}: falling back to {config.grasp_dof}-DOF HEURISTIC grasps "
                f"sampled from its collision spheres, not from perception. These miss far more "
                f"often. Raise perception.m2t2.num_runs (tamp_overrides `m2t2_num_runs`) to give "
                f"this object candidates; set `require_m2t2_grasps` to fail instead of guessing."
            )
            if config.require_m2t2_grasps:
                raise NoGraspsError(
                    f"no M2T2 grasp candidates for {obj!r} and require_m2t2_grasps is set. The "
                    f"heuristic fallback would sample grasps from this object's collision spheres "
                    f"rather than from perception."
                )

        # Sample grasps
        obj_curobo = world.get_object(obj)
        num_faces = 4 if isinstance(obj_curobo, Cuboid) else None
        obj_spheres = world.get_collision_spheres(obj)
        num_samples = num_particles * 2  # oversample at first
        sampled_confs = None
        is_mat4x4 = False

        if config.m2t2_grasps and n_m2t2 > 0:
            _log.debug(f"Using M2T2 grasps for {obj}")
            provided_grasps = self.grasps[obj]["grasps_obj"]
            confs = self.grasps[obj]["confidences_pt"]
            # Sample randomly with replacement if needed
            sample_idxs = torch.randint(
                0, provided_grasps.shape[0], (num_samples,), device=provided_grasps.device
            )
            sampled_grasps = provided_grasps[sample_idxs]
            sampled_confs = confs[sample_idxs]
            obj_from_grasp = sampled_grasps
            is_mat4x4 = True
        elif config.grasp_dof == 4:
            _log.debug(f"Falling back to 4-DOF heuristic grasps for {obj}")
            sampled_grasps = grasp_4dof_sampler(num_samples, obj_curobo, obj_spheres, num_faces=num_faces)
            obj_from_grasp = action_4dof_to_mat4x4(sampled_grasps)
        else:
            _log.debug(f"Falling back to 6-DOF heuristic grasps for {obj}")
            sampled_grasps = grasp_6dof_sampler(num_samples, obj_curobo, num_faces=num_faces)
            obj_from_grasp = action_6dof_to_mat4x4(sampled_grasps)

        # Filter grasps by collision with object
        grasp_spheres = transform_spheres(world.robot_container.gripper_spheres, obj_from_grasp)
        grasp_coll = sphere_to_sphere_overlap(obj_spheres, grasp_spheres, activation_distance=0.0)

        # Select grasps: use collision-free only, or fall back to lowest collision
        collision_free_mask = grasp_coll <= 1e-2
        if collision_free_mask.any():
            # Use only collision-free grasps, sorted by confidence if available
            cfree_grasps = sampled_grasps[collision_free_mask]

            if sampled_confs is not None:
                cfree_confs = sampled_confs[collision_free_mask]
                sort_idxs = torch.argsort(cfree_confs, descending=True)[:num_particles]
                selected_grasps = cfree_grasps[sort_idxs]
                selected_confs = cfree_confs[sort_idxs]
            else:
                selected_grasps = cfree_grasps[:num_particles]
                selected_confs = None
        else:
            # No collision-free grasps, take lowest collision scores
            best_idxs = grasp_coll.topk(num_particles, largest=False).indices
            selected_grasps = sampled_grasps[best_idxs]
            selected_confs = sampled_confs[best_idxs] if sampled_confs is not None else None

        # Sample with replacement if we don't have enough grasps
        if selected_grasps.shape[0] < num_particles:
            sample_idxs = torch.randint(
                0, selected_grasps.shape[0], (num_particles,), device=selected_grasps.device
            )
            selected_grasps = selected_grasps[sample_idxs]
            if selected_confs is not None:
                selected_confs = selected_confs[sample_idxs]

        return selected_grasps, selected_confs, is_mat4x4

    def _slide_grasps_to_end(self, grasps, is_mat4x4: bool, obj: str, sign: float):
        """Slide grasps toward one END of the object's longest horizontal axis, in the object frame.

        A handover needs the two hands on OPPOSITE ends of the object -- two 10 cm YAM grippers meeting
        at the same point leave ~1 mm of arm clearance, while a 22 cm bar held at +/-8 cm leaves
        ~127 mm. The heuristic 4-DOF sampler cannot express that: it pins every grasp to the object's
        vertical axis (``translation[:, :2] = 0``), so every candidate it emits is at the same point and
        no amount of "pick the farthest candidate" can separate them. So the offset is applied here.

        ``sign`` selects the end (+1 / -1); grasps are NOT optimized, so this pairing is final.
        """
        spheres = self.world.get_collision_spheres(obj)  # (n, 4), object frame
        lo = (spheres[:, :3] - spheres[:, 3:4]).min(dim=0).values
        hi = (spheres[:, :3] + spheres[:, 3:4]).max(dim=0).values
        extents = (hi - lo)[:2]
        axis = int(extents.argmax())
        # Back off by the pad's own half-width so the jaws still close on solid material.
        reach = max(float(extents[axis]) / 2 - _HANDOVER_END_MARGIN, 0.0)
        n = grasps.shape[0]
        offset = sign * reach * torch.empty(
            n, device=grasps.device, dtype=grasps.dtype
        ).uniform_(*_HANDOVER_END_FRACTION)
        grasps = grasps.clone()
        if is_mat4x4:
            grasps[:, axis, 3] += offset
        else:
            grasps[:, axis] += offset
        return grasps, axis, reach

    def __call__(self, plan_skeleton: PlanSkeleton, verbose: bool = True) -> Optional[Particles]:
        config = self.config
        num_particles = self.config.num_particles
        world = self.world
        particles = {"q0": self.q_init.clone()}
        deferred_params = set()
        log_debug = _log.debug if verbose else lambda *args, **kwargs: None

        # Note: we don't consider state after executing earlier samples
        # Iterate through each ground operator in the plan skeleton and initialize and build up particles
        for idx, ground_op in enumerate(plan_skeleton):
            op_name = ground_op.operator.name
            params = ground_op.values
            header = f"{idx + 1}. {ground_op}"

            # MoveFree
            if op_name == MoveFree.name:
                q_start, _traj, q_end = params
                if q_start not in particles:
                    raise ValueError(f"{q_start=} should already be bound")
                deferred_params.add(q_end)
                log_debug(f"{header}. Deferred {q_end}")

            # MoveHolding
            elif op_name == MoveHolding.name:
                obj, grasp, q_start, _traj, q_end = params
                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                if grasp not in particles:
                    raise ValueError(f"{grasp=} should already be bound")
                if q_start not in particles:
                    raise ValueError(f"{q_start=} should already be bound")
                deferred_params.add(q_end)
                log_debug(f"{header}. Deferred {q_end}")

            # Pick
            elif op_name == Pick.name:
                obj, grasp, q = params
                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                if grasp in particles:
                    raise ValueError(f"{grasp=} shouldn't already be bound")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                # Note: pick cache currently assumes object is at same pose as when sampled
                if obj in self.pick_cache:
                    # important, we need to clone here
                    particles[grasp] = self.pick_cache[obj]["sampled_grasps"].clone()
                    _cache_entry = self.pick_cache[obj]
                    ik_result = _cache_entry["ik_result"]
                    # posture-selected branch if this cache entry has one (see
                    # select_posture_branch); otherwise cuRobo's top seed, as before.
                    _cached_q = _cache_entry.get("q_posture")
                    particles[q] = (ik_result.solution[:, 0] if _cached_q is None else _cached_q).clone()
                    deferred_params.remove(q)
                    log_debug(
                        f"{header}. Using cached grasp poses for {obj}. {ik_result.success.sum()}/{num_particles} success"
                    )
                    particles[f"{grasp}_confidences"] = self.pick_cache[obj]["confidences"]
                    continue

                obj_curobo = world.get_object(obj)
                selected_grasps, selected_confs, is_mat4x4 = self._sample_grasps(obj, num_particles)
                particles[grasp] = selected_grasps
                particles[f"{grasp}_confidences"] = selected_confs

                # Transform grasps to hand frame
                if config.random_init:
                    q_sample = sample_between_bounds(num_particles, world.robot_container.joint_limits)
                    particles[q] = q_sample
                else:
                    obj_from_grasp = particles[grasp]
                    if not is_mat4x4:
                        obj_from_grasp = (
                            action_4dof_to_mat4x4(obj_from_grasp)
                            if config.grasp_dof == 4
                            else action_6dof_to_mat4x4(obj_from_grasp)
                        )
                    world_from_obj = pose_list_to_mat4x4(obj_curobo.pose).to(world.tensor_args.device)
                    world_from_grasp = world_from_obj @ obj_from_grasp
                    world_from_ee = world_from_grasp @ world.tool_from_ee

                    # Solve IK with cuRobo. Rolling the grasp is only expressible when it is
                    # stored as a 4x4 (M2T2); the 4/6-DOF parameterisations cannot carry it.
                    world_from_ee = Pose.from_matrix(world_from_ee)
                    particles[q], rolled_grasp, ik_result = solve_with_grasp_roll(
                        world, world_from_ee, config, obj_from_grasp if is_mat4x4 else None
                    )
                    if rolled_grasp is not None:
                        particles[grasp] = rolled_grasp
                    log_debug(
                        f"{header}. IK success: {ik_result.success.sum()}/{num_particles}, took {ik_result.solve_time:.2f}s"
                    )
                deferred_params.remove(q)

                # Store in cache
                if config.cache_subgraphs:
                    self.pick_cache[obj] = {
                        "sampled_grasps": particles[grasp],
                        "ik_result": ik_result,
                        # posture-selected branch (see select_posture_branch); solution[:, 0] otherwise
                        "q_posture": particles[q],
                        "confidences": particles[f"{grasp}_confidences"],
                    }

            # Place
            elif op_name == Place.name:
                obj, grasp, placement, surface, q = params
                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                if grasp not in particles:
                    raise ValueError(f"{grasp=} should already be bound")
                if placement in particles:
                    raise ValueError(f"{placement=} shouldn't already be bound")
                if not world.has_object(surface):
                    raise ValueError(f"{surface=} not found in world")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                if (obj, surface) in self.place_cache:
                    # need to make sure the grasps match what is cached
                    actual_grasp = particles[grasp]
                    cached_grasp = self.place_cache[(obj, surface)]["grasp"]
                    if not (actual_grasp == cached_grasp).all():
                        raise RuntimeError(f"Grasps don't match for {obj} on {surface}")

                    # important, we need to clone here
                    sampled_placements = self.place_cache[(obj, surface)]["sampled_placements"].clone()
                    particles[placement] = sampled_placements
                    _cache_entry = self.place_cache[(obj, surface)]
                    ik_result = _cache_entry["ik_result"]
                    # posture-selected branch if this cache entry has one (see
                    # select_posture_branch); otherwise cuRobo's top seed, as before.
                    _cached_q = _cache_entry.get("q_posture")
                    particles[q] = (ik_result.solution[:, 0] if _cached_q is None else _cached_q).clone()
                    deferred_params.remove(q)
                    log_debug(
                        f"{header}. Using cached placement poses for {obj}. {ik_result.success.sum()}/{num_particles} success"
                    )
                    continue

                # Sample placements pose of object (in world frame)
                obj_curobo = world.get_object(obj)
                obj_spheres = world.get_collision_spheres(obj)
                if config.random_init:
                    yaw = sample_yaw(num_particles * 2, None, self.world.tensor_args.device)
                    aabb = world.world_aabb.clone()
                    aabb[0, 2] = 0.0
                    aabb[1, 2] = max(aabb[1, 2], 0.2)
                    xyz = sample_between_bounds(num_particles * 2, aabb)
                    sampled_placements = torch.cat([xyz, yaw.unsqueeze(-1)], dim=1)
                else:
                    surface_curobo = world.get_object(surface)
                    sampled_placements = place_4dof_sampler(
                        num_particles * 2,
                        obj_curobo,
                        obj_spheres,
                        surface_curobo,
                        surface_rep=self.config.placement_check,
                        shrink_dist=self.config.placement_shrink_dist,
                        collision_activation_dist=self.config.world_activation_distance,
                    )

                # Select the placements that are not in collision with the object
                world_from_obj = action_4dof_to_mat4x4(sampled_placements)  # desired placement pose
                obj_place_spheres = transform_spheres(obj_spheres, world_from_obj)
                place_coll = world.collision_fn(obj_place_spheres[:, None].contiguous())[:, 0]
                best_idxs = place_coll.topk(num_particles, largest=False).indices
                sampled_placements = sampled_placements[best_idxs]
                world_from_obj = world_from_obj[best_idxs]

                # Set particles and then solve for robot configurations
                particles[placement] = sampled_placements
                if config.random_init:
                    q_sample = sample_between_bounds(num_particles, world.robot_container.joint_limits)
                    particles[q] = q_sample
                else:
                    # Get the hand pose given the placement pose in world frame.
                    # Need to take grasp into account to transform into hand frame.
                    if particles[grasp].shape[1:3] == (4, 4):  # (n, 4, 4) likely from M2T2
                        obj_from_grasp = particles[grasp]
                    elif config.grasp_dof == 4:
                        obj_from_grasp = action_4dof_to_mat4x4(particles[grasp])
                    else:
                        obj_from_grasp = action_6dof_to_mat4x4(particles[grasp])
                    world_from_grasp = world_from_obj @ obj_from_grasp
                    world_from_ee = world_from_grasp @ world.tool_from_ee

                    # Solve IK. The grasp is already bound by the preceding Pick, so a roll here
                    # would contradict it -- only the branch is selected, the pose is fixed.
                    world_from_ee = Pose.from_matrix(world_from_ee)
                    ik_result = world.ik_solver.solve_batch(
                        world_from_ee, seed_config=None,
                        return_seeds=max(1, config.posture_selection_seeds),
                    )
                    log_debug(
                        f"{header}. IK success: {ik_result.success.sum()}/{num_particles}, took {ik_result.solve_time:.2f}s"
                    )
                    particles[q] = select_posture_branch(
                        ik_result, world_from_ee, world.ik_solver, config
                    )
                deferred_params.remove(q)

                # Store in cache
                if config.cache_subgraphs:
                    self.place_cache[(obj, surface)] = {
                        "sampled_placements": sampled_placements,
                        "ik_result": ik_result,
                        "q_posture": particles[q],
                        "grasp": particles[grasp],
                    }

            # Push Button (without stick)
            elif op_name == Push.name:
                button, push_pose, q = params
                assert not config.random_init, "Random initialization not supported for pushing"
                if not world.has_object(button):
                    raise ValueError(f"{button=} not found in world")
                if push_pose in particles:
                    raise ValueError(f"{push_pose=} shouldn't already be bound")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                # Pruning failed subgraphs (i.e., we couldn't push this button at all)
                if button in self.failed_push and config.skip_failed_subgraphs:
                    return None

                if button in self.push_button_cache:
                    # important, we need to clone here
                    sampled_push = self.push_button_cache[button]["sampled_push"].clone()
                    particles[push_pose] = sampled_push
                    _cache_entry = self.push_button_cache[button]
                    ik_result = _cache_entry["ik_result"]
                    # posture-selected branch if this cache entry has one (see
                    # select_posture_branch); otherwise cuRobo's top seed, as before.
                    _cached_q = _cache_entry.get("q_posture")
                    particles[q] = (ik_result.solution[:, 0] if _cached_q is None else _cached_q).clone()
                    deferred_params.remove(q)
                    log_debug(
                        f"{header}. Using cached push poses for {button}. {ik_result.success.sum()}/{num_particles} success"
                    )
                    continue

                # Sample 4-DOF push poses for the button
                aabb = world.get_aabb(button).clone()
                surface_z = aabb[1, 2]  # top of the button aabb
                # add 1cm buffer
                lower_xy, upper_xy = aabb[:, :2]
                lower_xy += 0.01
                upper_xy -= 0.01

                sampled_xy = lower_xy + torch.rand(num_particles, 2, device=world.tensor_args.device) * (
                    upper_xy - lower_xy
                )
                sampled_z = (
                    surface_z.expand(num_particles) + 0.02 + world.collision_activation_distance
                )  # 2cm above button for now
                sampled_yaw = sample_yaw(num_particles, num_faces=None, device=world.tensor_args.device)
                sampled_push = torch.cat([sampled_xy, sampled_z[:, None], sampled_yaw[:, None]], dim=1)
                particles[push_pose] = sampled_push

                # Transform from tool to hand frame
                world_from_push = action_4dof_to_mat4x4(sampled_push)
                world_from_ee = world_from_push @ world.tool_from_ee

                # Solve IK with cuRobo
                world_from_ee = Pose.from_matrix(world_from_ee)
                ik_result = world.ik_solver.solve_batch(
                    world_from_ee, seed_config=None,
                    return_seeds=max(1, config.posture_selection_seeds),
                )
                log_debug(
                    f"{header}. IK success: {ik_result.success.sum()}/{num_particles}, took {ik_result.solve_time:.2f}s"
                )
                particles[q] = select_posture_branch(
                    ik_result, world_from_ee, world.ik_solver, config
                )
                deferred_params.remove(q)

                # Failed subgraph!
                if not ik_result.success.any():
                    self.failed_push.add(button)

                # Cache the push poses
                if config.cache_subgraphs:
                    self.push_button_cache[button] = {
                        "sampled_push": sampled_push,
                        "ik_result": ik_result,
                        "q_posture": particles[q],
                    }

            # Push Button with Stick
            elif op_name == PushStick.name:
                button, stick_name, grasp, push_pose, q = params
                assert not config.random_init, "Random initialization not supported for pushing"
                if not world.has_object(button):
                    raise ValueError(f"{button=} not found in world")
                if not world.has_object(stick_name):
                    raise ValueError(f"{stick_name=} not found in world")
                if grasp not in particles:
                    raise ValueError(f"{grasp=} should already be binded")
                if push_pose in particles:
                    raise ValueError(f"{push_pose=} shouldn't already be binded")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be binded")

                if (button, stick_name) in self.push_stick_cache:
                    # need to make sure the grasps match what is cached
                    actual_grasp = particles[grasp]
                    cached_grasp = self.push_stick_cache[(button, stick_name)]["grasp"]
                    if not (actual_grasp == cached_grasp).all():
                        raise RuntimeError(f"Grasps don't match for {button} with {stick_name}")

                    # important, we need to clone here
                    sampled_push = self.push_stick_cache[(button, stick_name)]["sampled_push"].clone()
                    particles[push_pose] = sampled_push
                    _cache_entry = self.push_stick_cache[(button, stick_name)]
                    ik_result = _cache_entry["ik_result"]
                    # posture-selected branch if this cache entry has one (see
                    # select_posture_branch); otherwise cuRobo's top seed, as before.
                    _cached_q = _cache_entry.get("q_posture")
                    particles[q] = (ik_result.solution[:, 0] if _cached_q is None else _cached_q).clone()
                    deferred_params.remove(q)

                    log_debug(
                        f"{header}. Using cached push for {button} with {stick_name}. "
                        f"{ik_result.success.sum()}/{num_particles} success"
                    )
                    continue

                # Sample pushes for the button, this will be in the stick frame
                aabb = world.get_aabb(button).clone()
                surface_z = aabb[1, 2]  # top of the button aabb
                # add 1cm buffer
                lower_xy, upper_xy = aabb[:, :2]
                lower_xy += 0.01
                upper_xy -= 0.01

                sampled_xy = lower_xy + torch.rand(num_particles, 2, device=world.tensor_args.device) * (
                    upper_xy - lower_xy
                )
                sampled_z = (
                    surface_z.expand(num_particles) + 0.02 + world.collision_activation_distance
                )  # 2cm above button for now
                sampled_yaw = sample_yaw(num_particles, num_faces=None, device=world.tensor_args.device)
                sampled_push = torch.cat([sampled_xy, sampled_z[:, None], sampled_yaw[:, None]], dim=1)

                # Sample somewhere along the stick for the push
                stick: MultiSphere = world.get_object("stick")
                spheres = stick.spheres
                if not (spheres[:, 1:3] == 0.0).all():
                    raise ValueError("Expected stick spheres to have y and z positions of 0")
                sphere_x = spheres[:, 0]
                x_idxs = torch.randint(0, len(sphere_x), (num_particles,), device=spheres.device)
                sampled_x = sphere_x[x_idxs]

                stick_from_tip = torch.eye(4, device=world.tensor_args.device).repeat(num_particles, 1, 1)
                stick_from_tip[:, 0, 3] = -sampled_x

                # Where we are pushing the button with the stick - i.e., the pose of the stick
                world_from_push = action_4dof_to_mat4x4(sampled_push)
                world_from_stick = world_from_push @ stick_from_tip

                # Push pose is pose of stick in world frame
                rpy = roma.rotmat_to_euler("XYZ", world_from_stick[:, :3, :3])
                assert (rpy[:, :2] == 0.0).all(), "roll and pitch should be 0"
                pos = world_from_stick[:, :3, 3]
                yaw = rpy[:, 2]
                action_4dof = torch.cat([pos, yaw[:, None]], dim=1)
                particles[push_pose] = action_4dof

                # Convert to tool frame
                if config.grasp_dof == 4:
                    obj_from_grasp = action_4dof_to_mat4x4(particles[grasp])
                else:
                    obj_from_grasp = action_6dof_to_mat4x4(particles[grasp])
                world_from_grasp = world_from_stick @ obj_from_grasp
                world_from_ee = world_from_grasp @ world.tool_from_ee

                # Solve IK with cuRobo
                world_from_ee = Pose.from_matrix(world_from_ee)
                ik_result = world.ik_solver.solve_batch(
                    world_from_ee, seed_config=None,
                    return_seeds=max(1, config.posture_selection_seeds),
                )
                log_debug(
                    f"{header}. IK success: {ik_result.success.sum()}/{num_particles}, took {ik_result.solve_time:.2f}s"
                )
                particles[q] = select_posture_branch(
                    ik_result, world_from_ee, world.ik_solver, config
                )
                deferred_params.remove(q)

                # Store in cache
                if config.cache_subgraphs:
                    self.push_stick_cache[(button, stick_name)] = {
                        "sampled_push": sampled_push,
                        "ik_result": ik_result,
                        "q_posture": particles[q],
                        "grasp": particles[grasp],
                    }

            # Unknown
            # PickBoth -- both hands grasp a different object at ONE shared configuration
            elif op_name == PickBoth.name:
                obj_a, grasp_a, obj_b, grasp_b, q = params
                arm_to_ee = {}
                for spec, obj, grasp in zip(world.arms, (obj_a, obj_b), (grasp_a, grasp_b)):
                    if not world.has_object(obj):
                        raise ValueError(f"{obj=} not found in world")
                    if grasp in particles:
                        raise ValueError(f"{grasp=} shouldn't already be bound")
                    sel, confs, is_mat4x4 = self._sample_grasps(obj, num_particles)
                    particles[grasp] = sel
                    particles[f"{grasp}_confidences"] = confs

                    obj_from_grasp = sel
                    if not is_mat4x4:
                        obj_from_grasp = (
                            action_4dof_to_mat4x4(obj_from_grasp)
                            if config.grasp_dof == 4
                            else action_6dof_to_mat4x4(obj_from_grasp)
                        )
                    world_from_obj = pose_list_to_mat4x4(world.get_object(obj).pose).to(world.tensor_args.device)
                    arm_to_ee[spec.name] = world_from_obj @ obj_from_grasp @ spec.tool_from_ee
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")
                particles[q], _ = self._seed_dual_conf(arm_to_ee, num_particles, header, log_debug)
                deferred_params.remove(q)

            # MoveHoldingBoth -- same deferral as MoveFree/MoveHolding, for two held objects
            elif op_name == MoveHoldingBoth.name:
                obj_a, grasp_a, obj_b, grasp_b, q_start, _traj, q_end = params
                for obj, grasp in ((obj_a, grasp_a), (obj_b, grasp_b)):
                    if not world.has_object(obj):
                        raise ValueError(f"{obj=} not found in world")
                    if grasp not in particles:
                        raise ValueError(f"{grasp=} should already be bound")
                if q_start not in particles:
                    raise ValueError(f"{q_start=} should already be bound")
                deferred_params.add(q_end)
                log_debug(f"{header}. Deferred {q_end}")

            # PlaceBoth -- both hands release at ONE shared configuration
            elif op_name == PlaceBoth.name:
                obj_a, grasp_a, place_a, surf_a, obj_b, grasp_b, place_b, surf_b, q = params
                arm_to_ee = {}
                for spec, obj, grasp, placement, surface in zip(
                    world.arms, (obj_a, obj_b), (grasp_a, grasp_b), (place_a, place_b), (surf_a, surf_b)
                ):
                    if grasp not in particles:
                        raise ValueError(f"{grasp=} should already be bound")
                    if placement in particles:
                        raise ValueError(f"{placement=} shouldn't already be bound")
                    if not world.has_object(surface):
                        raise ValueError(f"{surface=} not found in world")

                    obj_spheres = world.get_collision_spheres(obj)
                    sampled_placements = place_4dof_sampler(
                        num_particles * 2,
                        world.get_object(obj),
                        obj_spheres,
                        world.get_object(surface),
                        surface_rep=config.placement_check,
                        shrink_dist=config.placement_shrink_dist,
                        collision_activation_dist=config.world_activation_distance,
                    )
                    world_from_obj = action_4dof_to_mat4x4(sampled_placements)
                    place_coll = world.collision_fn(
                        transform_spheres(obj_spheres, world_from_obj)[:, None].contiguous()
                    )[:, 0]
                    best = place_coll.topk(num_particles, largest=False).indices
                    particles[placement] = sampled_placements[best]

                    obj_from_grasp = particles[grasp]
                    if obj_from_grasp.shape[1:3] != (4, 4):
                        obj_from_grasp = (
                            action_4dof_to_mat4x4(obj_from_grasp)
                            if config.grasp_dof == 4
                            else action_6dof_to_mat4x4(obj_from_grasp)
                        )
                    arm_to_ee[spec.name] = world_from_obj[best] @ obj_from_grasp @ spec.tool_from_ee
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")
                particles[q], _ = self._seed_dual_conf(arm_to_ee, num_particles, header, log_debug)
                deferred_params.remove(q)

            # PickGiver -- only the giver arm reaches; the taker parks.
            elif op_name == PickGiver.name:
                obj, grasp, q = params
                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                giver = world.arms[0]
                sel, confs_, is_mat4x4 = self._sample_grasps(obj, num_particles)
                # The giver takes the +end of the object so the taker can have the -end.
                sel, _axis, _reach = self._slide_grasps_to_end(sel, is_mat4x4, obj, +1.0)
                particles[grasp] = sel
                particles[f"{grasp}_confidences"] = confs_
                obj_from_grasp = sel if is_mat4x4 else (
                    action_4dof_to_mat4x4(sel) if config.grasp_dof == 4 else action_6dof_to_mat4x4(sel)
                )
                world_from_obj = pose_list_to_mat4x4(world.get_object(obj).pose).to(world.tensor_args.device)
                particles[q], _ = self._seed_dual_conf(
                    {giver.name: world_from_obj @ obj_from_grasp @ giver.tool_from_ee},
                    num_particles, header, log_debug,
                )
                deferred_params.remove(q)

            elif op_name in (MoveHoldingGiver.name, MoveHoldingTaker.name):
                obj, grasp, q_start, _traj, q_end = params
                if grasp not in particles:
                    raise ValueError(f"{grasp=} should already be bound")
                if q_start not in particles:
                    raise ValueError(f"{q_start=} should already be bound")
                deferred_params.add(q_end)
                log_debug(f"{header}. Deferred {q_end}")

            # Handover -- both hands meet on one object at a free mid-air pose.
            elif op_name == Handover.name:
                obj, grasp_g, grasp_t, hand_pose, q = params
                if grasp_g not in particles:
                    raise ValueError(f"{grasp_g=} should already be bound")
                giver, taker = world.arms[0], world.arms[1]

                # The taker's grasp must be FAR ALONG THE OBJECT from the giver's, or the two hands
                # occupy the same space: two 10 cm YAM grippers on one 5 cm cube leave ~1 mm of arm
                # clearance, while a 22 cm bar grasped at opposite ends leaves ~127 mm. Grasps are
                # NOT optimized (only Pose and Conf are), so a badly-paired sample can never be
                # repaired later -- hence choosing the separation here. Every candidate is first slid
                # to the object's -end (the giver holds the +end), then the farthest one is kept, which
                # is what actually separates real M2T2 grasps.
                cand, cand_confs, cand_mat = self._sample_grasps(obj, num_particles * 8)
                cand, axis, reach = self._slide_grasps_to_end(cand, cand_mat, obj, -1.0)
                cand_mat4 = cand if cand_mat else (
                    action_4dof_to_mat4x4(cand) if config.grasp_dof == 4 else action_6dof_to_mat4x4(cand)
                )
                giver_mat4 = particles[grasp_g]
                if giver_mat4.shape[1:3] != (4, 4):
                    giver_mat4 = (action_4dof_to_mat4x4(giver_mat4) if config.grasp_dof == 4
                                  else action_6dof_to_mat4x4(giver_mat4))
                # distance between grasp origins in the OBJECT frame, per (particle, candidate)
                sep = torch.cdist(giver_mat4[:, :3, 3], cand_mat4[:, :3, 3])       # (n, 8n)
                best = sep.argmax(dim=1)
                particles[grasp_t] = cand[best]
                particles[f"{grasp_t}_confidences"] = None if cand_confs is None else cand_confs[best]
                taker_mat4 = cand_mat4[best]
                log_debug(f"{header}. hands on object axis {axis} (usable half-length "
                          f"{reach * 100:.1f} cm). taker grasp separation: "
                          f"min {float(sep.gather(1, best[:, None]).min()) * 100:.1f} cm, "
                          f"median {float(sep.gather(1, best[:, None]).median()) * 100:.1f} cm")

                # The handover pose is a free 4-DOF object pose in MID-AIR: sampled in a box above
                # the table between the two shoulders, where the feasibility sweep found both arms can
                # reach without their upper arms intersecting.
                if hand_pose in particles:
                    raise ValueError(f"{hand_pose=} shouldn't already be bound")
                lo, hi = world.handover_region
                xyz = sample_between_bounds(num_particles, torch.stack([lo, hi]))
                yaw = sample_yaw(num_particles, None, world.tensor_args.device)
                # The giver holds the object's +axis end, so in mid-air that end has to point to the
                # giver's OWN side of the robot -- otherwise the arms must cross, which is never
                # collision-free. A uniform yaw gets this wrong half the time, and adding pi fixes
                # exactly those samples without changing the object's tilt.
                giver_side = world.arm_side_signs[giver.name]
                axis_world_y = torch.sin(yaw) if axis == 0 else torch.cos(yaw)
                yaw = torch.where(axis_world_y * giver_side < 0, yaw + torch.pi, yaw)
                particles[hand_pose] = torch.cat([xyz, yaw.unsqueeze(-1)], dim=1)
                world_from_obj = action_4dof_to_mat4x4(particles[hand_pose])

                particles[q], _ = self._seed_dual_conf(
                    {giver.name: world_from_obj @ giver_mat4 @ giver.tool_from_ee,
                     taker.name: world_from_obj @ taker_mat4 @ taker.tool_from_ee},
                    num_particles, header, log_debug,
                )
                deferred_params.remove(q)

            # PlaceTaker -- only the taker arm reaches; the giver parks.
            elif op_name == PlaceTaker.name:
                obj, grasp, placement, surface, q = params
                if grasp not in particles:
                    raise ValueError(f"{grasp=} should already be bound")
                if not world.has_object(surface):
                    raise ValueError(f"{surface=} not found in world")
                taker = world.arms[1]
                obj_spheres = world.get_collision_spheres(obj)
                sampled = place_4dof_sampler(
                    num_particles * 2, world.get_object(obj), obj_spheres, world.get_object(surface),
                    surface_rep=config.placement_check, shrink_dist=config.placement_shrink_dist,
                    collision_activation_dist=config.world_activation_distance,
                )
                world_from_obj = action_4dof_to_mat4x4(sampled)
                coll = world.collision_fn(
                    transform_spheres(obj_spheres, world_from_obj)[:, None].contiguous()
                )[:, 0]
                best = coll.topk(num_particles, largest=False).indices
                particles[placement] = sampled[best]

                obj_from_grasp = particles[grasp]
                if obj_from_grasp.shape[1:3] != (4, 4):
                    obj_from_grasp = (action_4dof_to_mat4x4(obj_from_grasp) if config.grasp_dof == 4
                                      else action_6dof_to_mat4x4(obj_from_grasp))
                particles[q], _ = self._seed_dual_conf(
                    {taker.name: world_from_obj[best] @ obj_from_grasp @ taker.tool_from_ee},
                    num_particles, header, log_debug,
                )
                deferred_params.remove(q)

            else:
                raise NotImplementedError(f"Unsupported operator {op_name}")

        # There should not be any deferred parameters left
        if deferred_params:
            raise RuntimeError(f"Deferred parameters not resolved: {deferred_params}")

        return particles
