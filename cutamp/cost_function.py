# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import itertools
from functools import reduce
import logging
from collections import defaultdict
from typing import Dict, Union

import roma
import torch
from einops import rearrange
from jaxtyping import Float

from curobo.rollout.cost.self_collision_cost import SelfCollisionCost, SelfCollisionCostConfig
from curobo.types.math import Pose
from cutamp.config import TAMPConfiguration
from cutamp.costs import dist_from_bounds_jit, sphere_to_sphere_overlap, trajectory_length
from cutamp.rollout import Rollout
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.task_planning.constraints import (
    Collision,
    CollisionFree,
    CollisionFreeGrasp,
    CollisionFreeHolding,
    CollisionFreePlacement,
    DualKinematicConstraint,
    KinematicConstraint,
    Motion,
    StablePlacement,
    ValidPush,
    ValidPushStick,
)
from cutamp.task_planning.costs import GraspCost, TrajectoryLength
from cutamp.utils.common import transform_spheres
from cutamp.utils.support import Footprint, footprint_from_object

_log = logging.getLogger(__name__)


class CostFunction:
    """
    Cost Function for a given plan skeleton. Given a rollout, we compute the constraints and costs.
    The __init__ function caches constraints and costs for quicker indexing during rollout evaluation.
    """

    def __init__(self, plan_skeleton: PlanSkeleton, world: TAMPWorld, config: TAMPConfiguration):
        if config.enable_traj:
            raise NotImplementedError("Trajectories not supported in cost function yet")

        self.plan_skeleton = plan_skeleton
        self.world = world
        self.config = config
        self._rollout_validated = False

        # Accumulate the constraints, so we can batch them up when computing the costs
        self.cfree_constraints = []
        self.kinematic_constraints = []
        self.dual_kinematic_constraints = []
        self._active_slots = None  # cached (timestep, arm) index pair; see dual_kinematic_costs
        self.motion_constraints = []
        self.stable_placement_constraints = []
        self.valid_push_constraints = []
        self.valid_push_stick_constraints = []
        self.traj_length_costs = []
        self.grasp_costs = []

        type_to_list = {
            KinematicConstraint.type: self.kinematic_constraints,
            DualKinematicConstraint.type: self.dual_kinematic_constraints,
            Motion.type: self.motion_constraints,
            CollisionFree.type: self.cfree_constraints,
            CollisionFreeHolding.type: self.cfree_constraints,
            CollisionFreeGrasp.type: self.cfree_constraints,
            CollisionFreePlacement.type: self.cfree_constraints,
            StablePlacement.type: self.stable_placement_constraints,
            ValidPush.type: self.valid_push_constraints,
            ValidPushStick.type: self.valid_push_stick_constraints,
            TrajectoryLength.type: self.traj_length_costs,
            GraspCost.type: self.grasp_costs,
        }
        for ground_op in plan_skeleton:
            for co in [*ground_op.constraints, *ground_op.costs]:
                if co.type not in type_to_list:
                    raise NotImplementedError(f"Unhandled constraint or cost: {co}")
                else:
                    type_to_list[co.type].append(co)

        # GraspCost(obj, grasp): soft costs scoring the grasp itself, each independently opt-in (see
        # grasp_soft_costs). `grasp_rot_change` charges the geodesic angle between the grasp's
        # end-effector orientation and the robot's INITIAL EE orientation (FK of world.q_init), so the
        # planner prefers grasps that reorient the wrist least; it indexes world_from_ee_desired at the
        # grasp's timestep, which is why the initial orientation is cached below. `grasp_center_offset`
        # charges how far off-center the grasp sits on the object and needs no such setup -- it reads
        # obj_from_grasp straight off the rollout. grasp is the action param (params[1]).
        self.grasp_cost_action_names = [cost.params[1] for cost in self.grasp_costs]
        self.init_ee_rotmat = None
        self.init_ee_rotmat_arms = None
        if self.grasp_costs and self.config.grasp_orientation_cost:
            q_init = self.world.q_init.view(1, -1)
            init_state = self.world.kin_model.get_state(q_init)
            init_ee_mat = init_state.ee_pose.get_matrix()  # (1, 4, 4)
            self.init_ee_rotmat = init_ee_mat[:, :3, :3].detach()  # (1, 3, 3), constant reference
            if self.world.arms:
                # One reference orientation per hand. Cloned because cuRobo hands back views into
                # buffers it overwrites on the next get_state() call.
                self.init_ee_rotmat_arms = torch.stack([
                    init_state.link_pose[spec.ee_link].get_matrix()[0, :3, :3]
                    for spec in self.world.arms
                ]).detach().clone()  # (A, 3, 3)

        # All conf parameters for motion constraints, we check rollout is subset
        self.motion_conf_params = set(
            iter(itertools.chain.from_iterable(con.params for con in self.motion_constraints))
        )

        # Setup self-collision cost
        self_collision_config = SelfCollisionCostConfig(
            self.world.tensor_args.to_device([1.0]),
            self.world.tensor_args,
            return_loss=True,
            self_collision_kin_config=self.world.kin_model.get_self_collision_config(),
        )
        self.self_collision_cost_fn = SelfCollisionCost(self_collision_config)
        # Should be using experimental kernel by default
        if not self.self_collision_cost_fn.self_collision_kin_config.experimental_kernel:
            raise ValueError("Expected self-collision cost to use experimental kernel")

        # Conf parameters for kinematic constraints, order in rollout should match. A skeleton uses
        # either the single-arm or the dual-arm operator set, never both, so exactly one of these two
        # lists is populated; zip(*[]) would otherwise fail to unpack into two names.
        if self.kinematic_constraints and self.dual_kinematic_constraints:
            raise NotImplementedError("Mixed single-arm and dual-arm kinematic constraints in one skeleton")
        if self.kinematic_constraints:
            self.kinematic_confs, self.kinematic_actions = map(
                list, zip(*(con.params for con in self.kinematic_constraints))
            )
        else:
            self.kinematic_confs, self.kinematic_actions = [], []
        # conf per timestep, and the per-arm action names at that timestep (T x A)
        self.dual_kinematic_confs = [con.params[0] for con in self.dual_kinematic_constraints]
        self.dual_kinematic_actions = [list(con.params[1:]) for con in self.dual_kinematic_constraints]
        self.action_to_arm_idx = {
            name: i for row in self.dual_kinematic_actions for i, name in enumerate(row)
        }

        # Compute the AABB and surface z-position for the placement surfaces
        self.surface_to_aabb = {}
        self.surface_to_obb = {}
        # Placement yaw a surface's support region requires, per surface; None where any yaw works.
        self.surface_to_yaw = {}
        self.surface_to_target_z = {}
        self.surface_to_objs = defaultdict(list)
        # Collect the objects per surface BEFORE fitting any region: a "support" region has to fit
        # the largest object that lands on the surface, which is not knowable from the first
        # constraint mentioning it.
        for con in self.stable_placement_constraints:
            obj, _, _, surface = con.params
            self.surface_to_objs[surface].append(obj)

        for surface in self.surface_to_objs:
            if self.config.placement_check == "aabb":
                aabb = world.get_aabb(surface)  # includes xyz
                aabb_xy = aabb[:, :2]
                self.surface_to_aabb[surface] = aabb_xy
                surface_z = aabb[1, 2]
            else:
                assert self.config.placement_check in ("obb", "support")
                # Through the world, not get_object_obb directly, so this is the SAME region the
                # placement sampler drew from -- and so a "support" region is fitted to an object
                # that actually has to fit on it.
                # Several objects may share a surface; the region has to hold all of them, so the
                # footprint fitted for is the one that contains every footprint.
                footprint = reduce(
                    Footprint.union,
                    (
                        footprint_from_object(world.get_object(obj), world.get_collision_spheres(obj))
                        for obj in self.surface_to_objs[surface]
                    ),
                )
                # A "support" region is already the set of valid object CENTRES, which is exactly
                # what the cost below constrains (it measures the object's origin against the
                # region), so it goes in as fitted.
                obb, object_yaw = world.placement_obb(surface, footprint, self.config)
                self.surface_to_obb[surface] = obb
                self.surface_to_yaw[surface] = object_yaw
                surface_z = obb.surface_z

            # For target z, need to take collision activation distance into account
            target_z = surface_z + world.collision_activation_distance + 2e-3  # add some buffer
            self.surface_to_target_z[surface] = target_z

        # Store the button AABBs for ValidPush
        self.button_to_action = {}
        self.button_aabbs = []
        for con in self.valid_push_constraints:
            button, action = con.params
            if not self.world.has_object(button):
                raise ValueError(f"{button=} not found in world")
            if button in self.button_to_action:
                raise NotImplementedError(f"We only support pushing a button once right now")

            # Set z to be 2cm buffer above surface of the button
            aabb = self.world.get_aabb(button).clone()
            aabb[0, 2] = aabb[1, 2] + world.collision_activation_distance + 2e-3
            aabb[1, 2] = aabb[1, 2] + world.collision_activation_distance + 0.02
            self.button_aabbs.append(aabb)
            self.button_to_action[button] = action
        self.button_aabbs = torch.stack(self.button_aabbs) if self.button_aabbs else None

        # Store the buttons and sticks for ValidPushStick
        self.button_stick_actions = {}
        self.button_stick_aabbs = []
        self.button_to_stick = {}
        for con in self.valid_push_stick_constraints:
            button, stick, action = con.params
            if not self.world.has_object(button):
                raise ValueError(f"{button=} not found in world")
            if not self.world.has_object(stick):
                raise ValueError(f"{stick=} not found in world")
            if button in self.button_to_stick:
                raise NotImplementedError(f"We only support pushing a button once right now")

            # Set z to be 2cm buffer above surface of the button
            button_aabb = self.world.get_aabb(button).clone()
            button_aabb[0, 2] = button_aabb[1, 2] + world.collision_activation_distance + 2e-3
            button_aabb[1, 2] = button_aabb[1, 2] + world.collision_activation_distance + 0.02
            self.button_stick_aabbs.append(button_aabb)

            self.button_to_stick[button] = stick
            self.button_stick_actions[button] = action
        self.button_stick_aabbs = torch.stack(self.button_stick_aabbs) if self.button_stick_aabbs else None

        # All conf parameters for trajectory length costs, we check rollout matches
        # No support for trajectories right now
        self.traj_length_confs = []
        for cost in self.traj_length_costs:
            q_start, traj, q_end = cost.params
            self.traj_length_confs.append(q_start)
            self.traj_length_confs.append(q_end)
        self.traj_length_confs = list(dict.fromkeys(self.traj_length_confs))  # remove duplicates
        if self.traj_length_confs[0] != "q0":
            raise ValueError("Expected q0 to be the first conf")
        self.traj_length_confs = self.traj_length_confs[1:]

        # Identify which objects are manipulated in the plan
        self.activated_obj = set()
        self.obj_to_first_place = {}
        for ground_op in plan_skeleton:
            op_name = ground_op.operator.name
            if op_name == "Pick":
                obj = ground_op.values[0]
                self.activated_obj.add(obj)
            elif op_name == "Place":
                obj = ground_op.values[0]
                pose = ground_op.values[2]
                self.activated_obj.add(obj)
                if obj not in self.obj_to_first_place:
                    self.obj_to_first_place[obj] = pose
            elif op_name == "PickBoth":
                v = ground_op.values
                self.activated_obj.update({v[0], v[2]})
            elif op_name == "PlaceBoth":
                v = ground_op.values
                for obj, pose in ((v[0], v[2]), (v[4], v[6])):
                    self.activated_obj.add(obj)
                    self.obj_to_first_place.setdefault(obj, pose)
            elif op_name == "PickGiver":
                self.activated_obj.add(ground_op.values[0])
            elif op_name == "Handover":
                # The object moves to the handover pose, so that pose activates it exactly like a
                # placement does -- which is what makes it collision-checked against the world and so
                # keeps the exchange off the table.
                obj, hand_pose = ground_op.values[0], ground_op.values[3]
                self.activated_obj.add(obj)
                self.obj_to_first_place.setdefault(obj, hand_pose)
            elif op_name == "PlaceTaker":
                obj, pose = ground_op.values[0], ground_op.values[2]
                self.activated_obj.add(obj)
                self.obj_to_first_place.setdefault(obj, pose)
            elif op_name == "PushStick" or op_name == "Push":
                raise NotImplementedError(f"Haven't handled {op_name}")
            else:
                assert op_name in ("MoveFree", "MoveHolding", "MoveHoldingBoth",
                                   "MoveHoldingGiver", "MoveHoldingTaker")

        # Pre-compute movable object pairs for collision checking.
        # Only include pairs where at least one object is manipulated
        movable_names = [m.name for m in self.world.movables]
        all_movable_pairs = list(itertools.combinations(movable_names, 2)) if len(movable_names) > 1 else []
        self.movable_obj_pairs = [
            (obj_1, obj_2)
            for obj_1, obj_2 in all_movable_pairs
            if obj_1 in self.activated_obj or obj_2 in self.activated_obj
        ]
        pairs_msg = ", ".join(f"({o_1}, {o_2})" for o_1, o_2 in self.movable_obj_pairs)
        _log.debug(f"Checking movable collisions between {pairs_msg}")

        # Populated upon validating the rollout
        self.obj_to_first_pose_ts = {}
        self.pair_to_first_pose_ts = {}
        self._activated_objs = sorted(self.activated_obj)  # deterministic ordering for torch.stack
        self._movable_world_mask = None  # lazily built in collision_costs
        self._target_surface_mask_cache = None  # lazily built in _target_surface_masks
        self._all_pose_ts = None

    def _validate_rollout(self, rollout: Rollout):
        """Checks structure of the rollout conforms to the assumptions we make in the cost function implementation."""
        if self._rollout_validated:
            return

        # Configurations should be subset of parameters involved in motion constraints
        if not set(rollout["conf_params"]).issubset(self.motion_conf_params):
            raise RuntimeError(
                f"Missing conf params in motion constraints: {rollout['conf_params'] - self.motion_conf_params}"
            )

        # Kinematic configuration and actions (i.e., poses) should match
        if self.dual_kinematic_constraints:
            if self.dual_kinematic_confs != rollout["conf_params"]:
                raise RuntimeError(
                    f"Expected conf params {self.dual_kinematic_confs} but got {rollout['conf_params']}"
                )
            if self.dual_kinematic_actions != rollout["action_params_arms"]:
                raise RuntimeError(
                    f"Expected per-arm action params {self.dual_kinematic_actions} "
                    f"but got {rollout['action_params_arms']}"
                )
        else:
            if self.kinematic_confs != rollout["conf_params"]:
                raise RuntimeError(f"Expected conf params {self.kinematic_confs} but got {rollout['conf_params']}")
            if self.kinematic_actions != rollout["action_params"]:
                raise RuntimeError(
                    f"Expected action params {self.kinematic_actions} but got {rollout['action_params']}"
                )

        # Trajectory length parameters should match
        if self.traj_length_confs != rollout["conf_params"]:
            raise RuntimeError(f"Expected conf params {self.traj_length_confs} but got {rollout['conf_params']}")

        # Bad-ish hack to get the first point at which the object is activated
        for obj, action in self.obj_to_first_place.items():
            self.obj_to_first_pose_ts[obj] = rollout["action_to_pose_ts"][action]
        for pair in self.movable_obj_pairs:
            obj_1, obj_2 = pair
            if obj_1 in self.obj_to_first_pose_ts and obj_2 in self.obj_to_first_pose_ts:
                self.pair_to_first_pose_ts[pair] = min(
                    self.obj_to_first_pose_ts[obj_1], self.obj_to_first_pose_ts[obj_2]
                )
            elif obj_1 in self.obj_to_first_pose_ts:
                self.pair_to_first_pose_ts[pair] = self.obj_to_first_pose_ts[obj_1]
            elif obj_2 in self.obj_to_first_place:
                self.pair_to_first_pose_ts[pair] = self.obj_to_first_pose_ts[obj_2]
            else:
                _log.warning(f"{pair} is never activated in a 'Place' action")

        self._all_pose_ts = list(rollout["ts_to_pose_ts"].values())

        self._rollout_validated = True

    def dual_kinematic_costs(self, rollout: Rollout) -> Union[dict, None]:
        """Pose error at EVERY arm, for a configuration constrained at both hands at once.

        The arm axis is folded into the time axis: values come out as ``(b, T*A)``. That is exactly
        what the downstream reducers already expect -- ``CostReducer`` sums ``dim=1`` (charge every
        (timestep, arm)) and ``ConstraintChecker`` does ``.all(dim=1)`` (satisfy at every (timestep,
        arm)) -- so neither needs to learn about arms. Emitting under ``KinematicConstraint.type``
        with the same ``pos_err``/``rot_err`` names also reuses the registered multipliers and
        tolerances, instead of silently defaulting to weight 1.0 and tolerance 0.0.
        """
        if not self.dual_kinematic_constraints:
            return None
        # Gather ONLY the (timestep, arm) pairs this skeleton actually constrains. For the lockstep
        # operators that is every pair, but a handover leaves one hand idle at the pick and the other
        # idle at the place, and charging an idle hand against a meaningless desired pose would make
        # every particle infeasible.
        if self._active_slots is None:
            active = rollout["arm_active"]
            pairs = [(t, a) for t, row in enumerate(active) for a, on in enumerate(row) if on]
            dev = rollout["ee_position_arms"].device
            self._active_slots = (
                torch.tensor([t for t, _ in pairs], dtype=torch.long, device=dev),
                torch.tensor([a for _, a in pairs], dtype=torch.long, device=dev),
            )
        t_idx, a_idx = self._active_slots
        ee_pos = rollout["ee_position_arms"][:, t_idx, a_idx]        # (b, n_active, 3)
        ee_quat = rollout["ee_quaternion_arms"][:, t_idx, a_idx]
        desired = rollout["world_from_ee_desired_arms"][:, t_idx, a_idx]
        b, n = ee_pos.shape[:2]
        ee_pose = Pose(position=ee_pos.reshape(-1, 3), quaternion=ee_quat.reshape(-1, 4),
                       normalize_rotation=False)
        p_dist, quat_dist = ee_pose.distance(Pose.from_matrix(desired.reshape(-1, 4, 4)))
        # Flattening (timestep, arm) into one axis is what lets CostReducer's sum over dim=1 and
        # ConstraintChecker's .all(dim=1) mean "charge every constrained hand" / "satisfy at every
        # constrained hand" without either of them knowing about arms.
        return {
            "type": "constraint",
            "constraints": self.dual_kinematic_constraints,
            "values": {"pos_err": p_dist.reshape(b, n), "rot_err": quat_dist.reshape(b, n)},
        }

    def kinematic_costs(self, rollout: Rollout) -> Union[dict, None]:
        """Kinematic constraints - i.e., pose error between actual and desired end-effector poses."""
        if not self.kinematic_constraints:
            return None
        # FK side: build a Pose from stored position+quaternion to skip the matrix round-trip.
        # Desired side is a 4x4 built from matrix multiplication upstream, so it still needs from_matrix.
        ee_pose = Pose(
            position=rollout["ee_position"].view(-1, 3),
            quaternion=rollout["ee_quaternion"].view(-1, 4),
            normalize_rotation=False,
        )
        desired_pose = Pose.from_matrix(rollout["world_from_ee_desired"].view(-1, 4, 4))
        p_dist_flat, quat_dist_flat = ee_pose.distance(desired_pose)
        batch_shape = rollout["ee_position"].shape[:-1]
        pos_errs = p_dist_flat.view(batch_shape)
        rot_errs = quat_dist_flat.view(batch_shape)
        kinematic_cost = {
            "type": "constraint",
            "constraints": self.kinematic_constraints,
            "values": {"pos_err": pos_errs, "rot_err": rot_errs},
        }
        return kinematic_cost

    def motion_costs(self, rollout: Rollout) -> Union[dict, None]:
        """Motion constraints - valid motions don't exceed joint limits or self-collide."""
        # Joint limits
        confs = rollout["confs"]
        dist_from_joint_lims = dist_from_bounds_jit(
            confs, self.world.robot_container.joint_limits[0], self.world.robot_container.joint_limits[1]
        )

        # Self collisions
        robot_spheres = rollout["robot_spheres"]
        with torch.profiler.record_function("coll::self_collision"):
            self_coll_vals = self.self_collision_cost_fn(robot_spheres)

        motion_cost = {
            "type": "constraint",
            "constraints": self.motion_constraints,
            "values": {"joint_limit": dist_from_joint_lims, "self_collision": self_coll_vals},
        }
        return motion_cost

    def valid_push_costs(
        self, rollout: Rollout, obj_to_spheres: Dict[str, Float[torch.Tensor, "b t n 4"]]
    ) -> Union[dict, None]:
        """Valid Push constraints - i.e., distance from the button for push actions."""
        if not (self.valid_push_constraints or self.valid_push_stick_constraints):
            return None

        valid_push_cost = {"type": "constraint", "constraints": [], "values": {}}

        if self.valid_push_constraints:
            ts_idxs = []  # get timestep for the button push actions
            for button, action in self.button_to_action.items():
                ts = rollout["action_to_ts"][action]
                ts_idxs.append(ts)
            tool_poses = rollout["world_from_tool_desired"][:, ts_idxs]
            tool_xyz = tool_poses[:, :, :3, 3]
            dist_from_button = dist_from_bounds_jit(tool_xyz, self.button_aabbs[:, 0], self.button_aabbs[:, 1])
            valid_push_cost["constraints"].extend(self.valid_push_constraints)
            valid_push_cost["values"]["dist_from_button"] = dist_from_button

        if self.valid_push_stick_constraints:
            # Get the stick spheres for each button push action
            all_stick_spheres = []
            for button, action in self.button_stick_actions.items():
                pose_ts = rollout["action_to_pose_ts"][action]
                stick_name = self.button_to_stick[button]
                stick_spheres = obj_to_spheres[stick_name][:, pose_ts]
                all_stick_spheres.append(stick_spheres)
            all_stick_spheres = torch.stack(all_stick_spheres, dim=1)

            # We'll consider bottom of spheres, so subtract the radius
            stick_xyz = all_stick_spheres[..., :3].clone()  # important! clone so we don't modify original tensor
            stick_xyz[..., 2] -= all_stick_spheres[..., 3]

            # Compute distance between all stick spheres and button AABB, take the minimum
            push_stick_cost = dist_from_bounds_jit(
                stick_xyz, self.button_stick_aabbs[:, None, 0], self.button_stick_aabbs[:, None, 1]
            )
            push_stick_cost = push_stick_cost.min(-1).values

            valid_push_cost["constraints"].extend(self.valid_push_stick_constraints)
            valid_push_cost["values"]["stick_dist_from_button"] = push_stick_cost

        return valid_push_cost

    def stable_placement_costs(
        self, rollout: Rollout, obj_to_spheres: Dict[str, Float[torch.Tensor, "b t n 4"]]
    ) -> Union[dict, None]:
        """
        Stable Placement constraints. Converted to two costs:
            1. Object sphere xy positions within surface AABB
            2. Minimum object sphere is supported by the surface (compute distance between surface and bottom of sphere)
        """
        if not self.stable_placement_constraints:
            return None

        # First collate the objects by placement surface.
        surface_to_obj = defaultdict(list)
        surface_to_spheres = defaultdict(list)
        surface_to_origins = defaultdict(list)
        surface_to_placements = defaultdict(list)
        for con in self.stable_placement_constraints:
            obj, _, placement, surface = con.params
            pose_ts = rollout["action_to_pose_ts"][placement]
            obj_spheres = obj_to_spheres[obj][:, pose_ts]
            surface_to_obj[surface].append(obj)
            surface_to_spheres[surface].append(obj_spheres)
            # The object's own origin at the placement, which is what a "support" region is a region
            # OF -- see the xy branch below.
            surface_to_origins[surface].append(rollout["obj_to_pose"][obj][:, pose_ts, :3, 3])
            surface_to_placements[surface].append((obj, placement))

        num_particles = rollout["num_particles"]
        support_vals = {}
        for surface, objs in surface_to_obj.items():
            # Create map of sphere index to object index
            sphere_idx_map = []
            for o_idx, (obj, spheres) in enumerate(zip(objs, surface_to_spheres[surface])):
                num_spheres = spheres.shape[1]
                sph_idxs = [o_idx] * num_spheres
                sphere_idx_map.extend(sph_idxs)
            sphere_idx_map = torch.tensor(sphere_idx_map, dtype=torch.int64, device=self.world.device)
            sphere_idx_map_expand = sphere_idx_map[None].expand(num_particles, -1)  # expand by batch size

            # Since objects can have different numbers of spheres, we need to concatenate instead of stack
            spheres = torch.cat(surface_to_spheres[surface], dim=1)
            spheres_xy = spheres[..., :2]

            # Within goal xy bounds, need to gather by the spheres for each object
            if self.config.placement_check == "aabb":
                in_goal_xy = dist_from_bounds_jit(spheres_xy, *self.surface_to_aabb[surface])
                obj_in_goal_xy = torch.zeros(
                    (num_particles, len(objs)), dtype=in_goal_xy.dtype, device=in_goal_xy.device
                )
                obj_in_goal_xy.scatter_add_(1, sphere_idx_map_expand, in_goal_xy)
                support_vals[f"{surface}_in_xy"] = obj_in_goal_xy
            else:
                obb = self.surface_to_obb[surface]
                obb_xy_lower = -obb.half_extents[:2]
                obb_xy_upper = obb.half_extents[:2]

                if self.config.placement_check == "support":
                    # A support region is the set of valid object ORIGINS: it was fitted for this
                    # object's footprint, so an origin inside it already means every part of the
                    # object is over surface that holds it. Charging each collision SPHERE against it
                    # instead asks the whole object to fit inside a region that is the surface minus
                    # the object -- unsatisfiable for anything but a point, and the constraint no
                    # placement could clear.
                    points = torch.stack(surface_to_origins[surface], dim=1)  # (b, objs, 3)
                else:
                    # Check spheres are within the OBB's xy plane by transforming to OBB local frame
                    points = spheres[..., :3]  # (b, n, 3)

                # Transform to the OBB's local frame using its cached rotation matrix
                points_local = (points - obb.center) @ obb.rot_matrix_inv.T
                dist = dist_from_bounds_jit(points_local[..., :2], obb_xy_lower, obb_xy_upper)

                if self.config.placement_check == "support":
                    support_vals[f"{surface}_in_xy"] = dist  # already (b, objs)
                    target_yaw = self.surface_to_yaw.get(surface)
                    if target_yaw is not None:
                        # The region only holds the object at THIS orientation (see
                        # cutamp.utils.support.Footprint), and yaw is one of the placement's
                        # optimized parameters -- without a term here the optimizer is free to turn
                        # the object out of the pose the region was fitted for. |sin| rather than an
                        # angle difference because a rectangle is the same either way up: it is zero
                        # at the pinned yaw AND its half-turn twin, which is exactly the symmetry,
                        # and reads as radians for the small deviations the tolerance cares about.
                        rot = torch.stack(
                            [rollout["obj_to_pose"][obj][:, rollout["action_to_pose_ts"][placement]]
                             for obj, placement in surface_to_placements[surface]], dim=1
                        )[..., :3, :3]
                        yaw = torch.atan2(rot[..., 1, 0], rot[..., 0, 0])  # (b, objs)
                        support_vals[f"{surface}_yaw"] = torch.abs(torch.sin(yaw - target_yaw))
                else:
                    # Accumulate per-object distances
                    obj_in_goal_xy = torch.zeros(
                        (num_particles, len(objs)), dtype=dist.dtype, device=dist.device
                    )
                    obj_in_goal_xy.scatter_add_(1, sphere_idx_map_expand, dist)
                    support_vals[f"{surface}_in_xy"] = obj_in_goal_xy

            # Distance between bottom of spheres and z-position of the surface
            spheres_bottom = spheres[..., 2] - spheres[..., 3]
            obj_bottom = torch.full(
                (num_particles, len(objs)), float("inf"), dtype=spheres_bottom.dtype, device=spheres_bottom.device
            )
            obj_bottom.scatter_reduce_(1, sphere_idx_map_expand, spheres_bottom, reduce="amin")
            target_z = self.surface_to_target_z[surface]
            support_vals[f"{surface}_support"] = torch.abs(obj_bottom - target_z)

        stable_placement_cost = {
            "type": "constraint",
            "constraints": self.stable_placement_constraints,
            "values": support_vals,
        }
        return stable_placement_cost

    def trajectory_costs(self, rollout: Rollout) -> dict:
        """Trajectory costs, just joint space distance between configurations for now."""
        traj_cost = {
            "type": "cost",
            "costs": self.traj_length_costs,
            "values": {"traj_length": trajectory_length(rollout["confs"], p=self.config.traj_length_norm)},
        }
        return traj_cost

    def grasp_soft_costs(self, rollout: Rollout) -> Union[dict, None]:
        """All GraspCost terms, emitted together (cost_dict is keyed by cost TYPE, so one dict per type).

        Each term is independently gated by a config flag, because the reducer treats an ABSENT
        multiplier as weight 1.0 -- an always-emitted value would silently change every caller's
        objective. Returns None when the skeleton has no grasps or no term is enabled.

          - ``grasp_rot_change``    (config.grasp_orientation_cost) -- see grasp_orientation_costs
          - ``grasp_center_offset`` (config.grasp_center_cost)      -- see grasp_center_costs
        """
        if not self.grasp_costs:
            return None
        values = {}
        rot = self.grasp_orientation_costs(rollout)
        if rot is not None:
            values.update(rot)
        center = self.grasp_center_costs(rollout)
        if center is not None:
            values.update(center)
        if not values:
            return None
        return {"type": "cost", "costs": self.grasp_costs, "values": values}

    def grasp_center_costs(self, rollout: Rollout) -> Union[dict, None]:
        """``grasp_center_offset`` - horizontal distance from the object's origin to the grasp's TCP.

        Incentivizes grasps nearer the middle of the object instead of out at an edge, where the lever
        arm between the contact and the object's center of mass makes the hold prone to slipping or
        pivoting. Charged in METERS, per grasp parameter.

        The grasp parameter IS ``obj_from_grasp``, so its translation is already the TCP expressed in
        the object frame -- no timestep or arm indexing is needed, and the value is identical at every
        rollout timestep (the grasp is a rigid offset). Grasp parameters are not optimized
        (``types_to_optimize = {Pose, Conf}``), so this term acts purely as a selection signal over the
        discrete candidates the particles were initialized with.

        Only the xy components are charged. For TiPToP's perceived meshes the object frame origin is
        the mean of the reconstructed vertices (``perception/utils.py::convert_trimesh_to_curobo_mesh``),
        which comes from a single-view point cloud: the z component of that centroid is biased toward
        the camera-facing surface, while xy is comparatively unbiased. xy is also the axis the
        instability actually lives on -- an off-center grasp in the table plane is what lets gravity
        pivot the object out of the fingers.

        Two caveats, both about WHICH objects this is a stability signal for:

        - Hollow objects (bowl, plate, ring). The origin sits in empty space, so every reachable grasp
          is on the rim -- but they are NOT equidistant from it: measured over saved TiPToP runs, the
          rim grasps on one bowl span 2-5 cm of offset, the same range as the toys this term is meant
          to fix. Because a single-view reconstruction pulls the vertex centroid toward the
          camera-facing wall, what the term then ranks is proximity to a view-biased centroid, which
          has nothing to do with stability. Prefer leaving it off for bowl/plate tasks.
        - Large objects. The charge is raw meters with no normalization by extent, deliberately: that
          is what makes the END of a long object expensive, which normalizing would undo. The cost is
          therefore a tiebreaker on a small toy (3-5 cm of spread across its candidates) and decisive
          on a big one (a potted plant spans ~14 cm), at one and the same weight.
        """
        if not self.grasp_costs or not self.config.grasp_center_cost:
            return None
        obj_from_grasp = rollout["grasp_to_obj_from_grasp"]
        offsets = torch.stack(
            [obj_from_grasp[name][:, :2, 3] for name in self.grasp_cost_action_names], dim=1
        )  # (b, k, 2)
        return {"grasp_center_offset": torch.linalg.norm(offsets, dim=-1)}  # (b, k)

    def grasp_orientation_costs(self, rollout: Rollout) -> Union[dict, None]:
        """``grasp_rot_change`` - geodesic angle between each grasp's EE orientation and the initial one.

        Incentivizes grasps that require the least change in end-effector orientation from the robot's
        starting pose. Returns None unless the skeleton has grasps AND config.grasp_orientation_cost is
        set (the gate: an absent reducer multiplier defaults to weight 1.0, so we must not emit this
        value unless the caller opted in).
        """
        if not self.grasp_costs or not self.config.grasp_orientation_cost:
            return None
        # EE orientation desired at each grasp's timestep (same "ee" frame as kinematic_costs uses).
        ts_idxs = [rollout["action_to_ts"][name] for name in self.grasp_cost_action_names]
        if self.init_ee_rotmat_arms is not None:
            # Dual-arm: index the (t, arm) slot each grasp belongs to, and compare against THAT
            # hand's initial orientation -- the two hands start mirrored, so one shared reference
            # would charge the right arm for the left arm's starting pose.
            arm_idxs = [self.action_to_arm_idx[name] for name in self.grasp_cost_action_names]
            grasp_rotmat = rollout["world_from_ee_desired_arms"][:, ts_idxs, arm_idxs, :3, :3]
            init_rotmat = self.init_ee_rotmat_arms[arm_idxs].unsqueeze(0).expand_as(grasp_rotmat)
        else:
            grasp_rotmat = rollout["world_from_ee_desired"][:, ts_idxs, :3, :3]  # (b, k, 3, 3)
            init_rotmat = self.init_ee_rotmat.view(1, 1, 3, 3).expand_as(grasp_rotmat)
        # Geodesic distance on SO(3) in radians; roma clamps internally to keep gradients stable.
        grasp_rot_change = roma.rotmat_geodesic_distance(grasp_rotmat, init_rotmat)  # (b, k)
        return {"grasp_rot_change": grasp_rot_change}

    def _target_surface_masks(self, rollout: Rollout) -> Dict[str, torch.Tensor]:
        """Per placement surface, an (objs, 1, t) mask of when an object is resting ON it.

        An object resting on a surface is exempt from colliding with THAT surface (see
        ``TAMPConfiguration.placement_ignores_target_surface``). "Resting on" runs from the object's
        placement there until its next placement somewhere else, so an object moved plate -> box is
        exempt from the plate only over the leg where it is on the plate. Cached: the timesteps come
        from the skeleton, not from the particles.

        The mask is on the OBJECT-POSE timeline -- ``rollout["obj_to_pose"]``, one entry per Place
        plus the initial pose -- because that is what ``obj_to_spheres`` and therefore the collision
        values it selects are indexed by. Not the robot timeline (``robot_spheres``), which has an
        entry per Pick AND per Place: the two are the same length only for a single-placement
        skeleton, so sizing this from the robot's worked for one pick-and-place and mismatched 4
        against 3 on the first plan that placed twice.
        """
        if self._target_surface_mask_cache is not None:
            return self._target_surface_mask_cache

        obj_idx = {obj: i for i, obj in enumerate(self._activated_objs)}
        # obj -> [(pose timestep, surface)], in execution order
        placements = defaultdict(list)
        for con in self.stable_placement_constraints:
            obj, _, placement, surface = con.params
            if obj in obj_idx:
                placements[obj].append((rollout["action_to_pose_ts"][placement], surface))

        if not placements:
            self._target_surface_mask_cache = {}
            return self._target_surface_mask_cache

        # Every movable's pose list is accumulated in lockstep, so any one of them gives the length.
        num_t = next(iter(rollout["obj_to_pose"].values())).shape[1]
        masks: Dict[str, torch.Tensor] = {}
        for obj, entries in placements.items():
            entries.sort()
            for i, (ts, surface) in enumerate(entries):
                end = entries[i + 1][0] if i + 1 < len(entries) else num_t
                if surface not in masks:
                    masks[surface] = torch.zeros(
                        (len(self._activated_objs), 1, num_t), dtype=torch.bool, device=self.world.device
                    )
                masks[surface][obj_idx[obj], :, ts:end] = True
        self._target_surface_mask_cache = masks
        return masks

    def collision_costs(self, rollout: Rollout, obj_to_spheres: Dict[str, Float[torch.Tensor, "b t n 4"]]) -> dict:
        """Collision costs."""
        # Robot to world
        robot_spheres = rollout["robot_spheres"]
        with torch.profiler.record_function("coll::robot_to_world"):
            # robot_collision_fn, not collision_fn: the arm is allowed to reach INTO the surfaces
            # listed in world.pick_transparent (see TAMPWorld). movable_to_world below keeps the
            # full checker, so a PLACED object is still screened against those surfaces.
            coll_values = {"robot_to_world": self.world.robot_collision_fn(robot_spheres)}

        # Collision between movables and world — batch all activated objects in one collision_fn call,
        # then mask out timesteps before each object's first placement. The motion solver handles this
        # analogously by temporarily detaching the object from the robot when the grasped object's
        # spheres cause an invalid start state during retract planning.
        with torch.profiler.record_function("coll::movable_to_world"):
            stacked = torch.stack([obj_to_spheres[obj] for obj in self._activated_objs])
            flat = rearrange(stacked, "objs b t n d -> (objs b) t n d")
            num_objs = len(self._activated_objs)
            coll = self.world.collision_fn(flat)
            coll = rearrange(coll, "(objs b) t -> objs b t", objs=num_objs)

            # Let each placed object overlap the surface it was placed ON, from that placement
            # onwards. An open container reconstructs as its convex hull, so the inside of a box and
            # the dish of a plate read as solid and NO placement there can satisfy this constraint --
            # which is why placements used to end up on top of the hull, at the height of a box's
            # lid. One extra collision call per placement surface (a plan has one or two), each
            # differing from the full checker by exactly that one obstacle; every other obstacle, and
            # this object before it is placed, is still screened by `coll` above.
            if self.config.placement_ignores_target_surface:
                for surface, mask in self._target_surface_masks(rollout).items():
                    exempt_fn = self.world.collision_fn_for_placement(exclude=surface)
                    coll_exempt = rearrange(
                        exempt_fn(flat), "(objs b) t -> objs b t", objs=num_objs
                    )
                    coll = torch.where(mask, coll_exempt, coll)
            if self.config.mask_initial_movable_world_collision:
                if self._movable_world_mask is None:
                    num_objs, t = coll.shape[0], coll.shape[2]
                    mask = torch.ones(num_objs, 1, t, device=coll.device)
                    for i, obj in enumerate(self._activated_objs):
                        if obj not in self.obj_to_first_pose_ts:
                            # If an object has no pose ts then it is only ever picked, so we don't need to check collision
                            # between it and the world. Hence, we set the mask to 0.0
                            mask[i] = 0.0
                        else:
                            first_ts = self.obj_to_first_pose_ts[obj]
                            if first_ts > 0:
                                mask[i, :, :first_ts] = 0.0
                    self._movable_world_mask = mask
                coll = coll * self._movable_world_mask
            coll_values["movable_to_world"] = coll.sum(dim=0)

        with torch.profiler.record_function("coll::robot_to_movables"):
            # Concatenate all movable spheres into one kernel launch — faster than per-object
            # launches at our sphere counts (~50/object) where launch overhead dominates.
            all_obj_spheres = torch.cat(
                [obj_s[:, self._all_pose_ts] for obj_s in obj_to_spheres.values()], dim=-2
            )
            coll_values["robot_to_movables"] = sphere_to_sphere_overlap(
                robot_spheres, all_obj_spheres, activation_distance=self.config.gripper_activation_distance
            )

        # Collision between movable objects
        if self.movable_obj_pairs:
            with torch.profiler.record_function("coll::movable_to_movable"):
                # Stack into (num_pairs, b, t, n_spheres, 4)
                obj_1_spheres_list = [obj_to_spheres[name1] for name1, _ in self.movable_obj_pairs]
                obj_2_spheres_list = [obj_to_spheres[name2] for _, name2 in self.movable_obj_pairs]
                obj_1_spheres_batched = torch.stack(obj_1_spheres_list, dim=0)
                obj_2_spheres_batched = torch.stack(obj_2_spheres_list, dim=0)

                collision_results = sphere_to_sphere_overlap(
                    obj_1_spheres_batched,
                    obj_2_spheres_batched,
                    activation_distance=self.config.movable_activation_distance,
                    use_aabb_check=True,
                )  # (num_pairs, b, t)

            for idx, pair in enumerate(self.movable_obj_pairs):
                if pair not in self.pair_to_first_pose_ts:
                    # Neither object was ever 'activated' (i.e., placed), so no need to collision check this object pair
                    continue
                pair_cost = collision_results[idx]
                pose_ts = self.pair_to_first_pose_ts[pair]
                # Only consider costs from when the action was activated. This allows us to handle objects that
                # are initially in-collision (perhaps due to bad perception)
                pair_cost_filtered = pair_cost[:, pose_ts:]
                name1, name2 = pair
                coll_values[f"{name1}_to_{name2}"] = pair_cost_filtered

        coll_cost = {
            "type": "constraint",
            "constraints": self.cfree_constraints,
            "values": coll_values,
        }
        return coll_cost

    def soft_costs(self, rollout: Rollout) -> dict:
        """Soft costs defined on the goal state."""
        # last object pose
        last_obj_position = [v[:, -1, :3, 3] for v in rollout["obj_to_pose"].values()]
        last_obj_position = torch.stack(last_obj_position, dim=1)

        if self.config.soft_cost == "dist_from_origin":
            dist_from_origin = last_obj_position.norm(dim=-1)
            dist_from_origin = -dist_from_origin.sum(dim=-1)
            values = {"dist_from_origin": dist_from_origin}
        elif self.config.soft_cost == "max_obj_dist" or self.config.soft_cost == "min_obj_dist":
            all_obj_dists = torch.cdist(last_obj_position, last_obj_position, p=2)  # (b, n, n)
            mask = torch.triu(torch.ones_like(all_obj_dists), diagonal=1) == 1
            obj_dists = all_obj_dists[mask].view(mask.shape[0], -1)  # reshape into num pairs
            dists_sum = obj_dists.sum(-1)
            if self.config.soft_cost == "max_obj_dist":
                values = {"max_obj_dist": -dists_sum}
            else:
                values = {"min_obj_dist": dists_sum}
        elif self.config.soft_cost == "min_y" or self.config.soft_cost == "max_y":
            last_obj_y = last_obj_position[..., 1]
            last_y = last_obj_y.sum(dim=-1)
            if self.config.soft_cost == "min_y":
                values = {"min_y": last_y}
            else:
                values = {"max_y": -last_y}
        elif self.config.soft_cost == "align_yaw":
            last_obj_mat3x3 = [v[:, -1, :3, :3] for v in rollout["obj_to_pose"].values()]
            last_obj_mat3x3 = torch.stack(last_obj_mat3x3, dim=1)
            last_obj_rpy = roma.rotmat_to_euler("XYZ", last_obj_mat3x3)
            last_obj_yaw = last_obj_rpy[..., 2]

            # Compute pairwise yaw differences and normalize to be between -pi and pi
            yaw_diffs = last_obj_yaw[:, :, None] - last_obj_yaw[:, None, :]
            yaw_diffs = torch.atan2(torch.sin(yaw_diffs), torch.cos(yaw_diffs)).abs()
            mask = torch.triu(torch.ones_like(yaw_diffs), diagonal=1) == 1
            yaw_diffs = yaw_diffs[mask].view(mask.shape[0], -1)  # reshape into num pairs
            yaw_diffs = yaw_diffs.sum(-1)
            values = {"align_yaw": yaw_diffs}
        else:
            raise ValueError(f"Unsupported soft cost: {self.config.soft_cost}")

        return {"type": "cost", "constraints": [], "values": values}

    def __call__(self, rollout: Rollout) -> Dict[str, dict]:
        self._validate_rollout(rollout)
        cost_dict = {}

        def add_cost(k_, v_):
            if v_ is not None:
                cost_dict[k_] = v_

        # Trajectory cost
        with torch.profiler.record_function("cost::trajectory"):
            traj_cost = self.trajectory_costs(rollout)
        add_cost(TrajectoryLength.type, traj_cost)

        # Grasp soft costs -- orientation change and/or off-center offset. Both are opt-in, and only
        # reduced when the matching GraspCost multiplier is set.
        with torch.profiler.record_function("cost::grasp"):
            grasp_cost = self.grasp_soft_costs(rollout)
        add_cost(GraspCost.type, grasp_cost)

        # Get collision spheres for movable objects
        with torch.profiler.record_function("cost::transform_spheres"):
            obj_to_spheres = {}
            for idx, obj in enumerate(self.world.movables):
                if obj.name in obj_to_spheres:
                    raise RuntimeError(f"Object {obj.name} already in obj_to_spheres")
                obj_pose = rollout["obj_to_pose"][obj.name]
                obj_spheres = transform_spheres(self.world.get_collision_spheres(obj), obj_pose)
                obj_to_spheres[obj.name] = obj_spheres

        # Collision costs
        with torch.profiler.record_function("cost::collision"):
            collision_cost = self.collision_costs(rollout, obj_to_spheres)
        add_cost(Collision.type, collision_cost)

        # Valid Push constraints
        with torch.profiler.record_function("cost::valid_push"):
            valid_push_cost = self.valid_push_costs(rollout, obj_to_spheres)
        add_cost(ValidPush.type, valid_push_cost)

        # Stable placement cost
        with torch.profiler.record_function("cost::stable_placement"):
            stable_placement_cost = self.stable_placement_costs(rollout, obj_to_spheres)
        add_cost(StablePlacement.type, stable_placement_cost)

        # Valid motions don't exceed joint limits
        with torch.profiler.record_function("cost::motion"):
            motion_cost = self.motion_costs(rollout)
        add_cost(Motion.type, motion_cost)

        # Kinematic costs. A skeleton is single-arm or dual-arm, never both, so exactly one of these
        # returns a value; both emit under KinematicConstraint.type so the registered multipliers and
        # tolerances apply either way.
        with torch.profiler.record_function("cost::kinematic"):
            kinematic_cost = self.kinematic_costs(rollout)
            if kinematic_cost is None:
                kinematic_cost = self.dual_kinematic_costs(rollout)
        add_cost(KinematicConstraint.type, kinematic_cost)

        # Soft costs
        if self.config.soft_cost is not None:
            with torch.profiler.record_function("cost::soft"):
                soft_cost = self.soft_costs(rollout)
            add_cost("soft", soft_cost)

        return cost_dict
