# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Solving motions with cuRobo."""

import logging
import time
from typing import List, Optional

import torch

from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import Sphere
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig, MotionGen
from cutamp.config import TAMPConfiguration
from cutamp.optimize_plan import PlanContainer
from cutamp.tamp_domain import MoveHolding, Push, PushStick, MoveFree, Place, Pick
from cutamp.tamp_world import TAMPWorld
from cutamp.utils.common import Particles, action_6dof_to_mat4x4, action_4dof_to_mat4x4
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import Visualizer

_log = logging.getLogger(__name__)


class MotionPlanningError(RuntimeError):
    pass


def solve_curobo(
    plan_info: PlanContainer,
    best_particle: Particles,
    world: TAMPWorld,
    config: TAMPConfiguration,
    timer: TorchTimer,
    visualizer: Visualizer,
    obj_to_initial_pose: dict[str, torch.Tensor],
    timeline: str = "curobo",
    motion_gen: Optional[MotionGen] = None,
):
    """
    Solve for full motion plan given a plan skeleton and optimized particles.
    Note that visualization adds non-trivial overhead.
    """
    plan_skeleton = plan_info["plan_skeleton"]
    if motion_gen is None:
        motion_gen = world.get_motion_gen(collision_activation_distance=config.world_activation_distance)
    if config.warmup_motion_gen:
        with timer.time(f"{timeline}_motion_gen_warmup", log_callback=_log.debug):
            motion_gen.warmup()

    plan_config = MotionGenPlanConfig(
        timeout=0.25, enable_finetune_trajopt=False, time_dilation_factor=config.time_dilation_factor
    )

    # Log initial state
    ts = 0.0
    obj_to_current_pose = {k: v.clone() for k, v in obj_to_initial_pose.items()}
    # obj_to_current_pose = {obj.name: world.get_object_pose(obj) for obj in world.movables}

    # Reset motion gen, clear attachments and reset pose of all objects
    motion_gen.detach_object_from_robot("attached_object")
    for obj, obj_pose in obj_to_current_pose.items():
        motion_gen.world_coll_checker.enable_obstacle(enable=True, name=obj)
        obj_pose = obj_to_current_pose[obj]
        motion_gen.world_collision.update_obstacle_pose(obj, Pose.from_matrix(obj_pose))

    visualizer.set_time_seconds(timeline, ts)
    visualizer.set_joint_positions(best_particle["q0"])
    for obj, pose in obj_to_current_pose.items():
        visualizer.log_mat4x4(f"world/{obj}", pose)

    last_js = JointState.from_position(best_particle["q0"][None].clone())
    last_q_name = "q0"

    # Fixed approach offset. This could be something we eventually optimize too
    approach_offset = torch.eye(4, device=world.device)
    approach_offset[2, 3] = -0.05

    approach_offsets = torch.eye(4, device=world.device).repeat(4, 1, 1)
    approach_offsets[:, 2, 3] = torch.tensor([-0.05, -0.1, -0.15, -0.2], device=world.device)

    constrained_motion_cost_metric = PoseCostMetric(
        hold_partial_pose=True,
        hold_vec_weight=world.tensor_args.to_device([0.1, 0.1, 0.1, 0.1, 0.1, 0.0]),
        project_to_goal_frame=True,
    )
    constrained_plan_config = plan_config.clone()
    constrained_plan_config.pose_cost_metric = constrained_motion_cost_metric

    # Accumulated plans we return that the real robot can actually execute
    accum_plans = []

    # Iterate through skeleton and motion plan
    for idx, ground_op in enumerate(plan_skeleton):
        op_name = ground_op.operator.name
        print(f"{idx + 1}. {ground_op.name}")

        # MoveFree, defer motion planning to pick to use object pose instead of planning from q_start to q_end.
        # This works more reliably and gives higher quality motions.
        if op_name == MoveFree.name:
            q_start, traj, q_end = ground_op.values
            if traj in best_particle:
                raise NotImplementedError("Trajectories not supported yet")
            last_q_name = q_start

        # MoveHolding
        elif op_name == MoveHolding.name:
            obj, grasp, q_start, traj, q_end = ground_op.values
            if traj in best_particle:
                raise NotImplementedError("Trajectories not supported yet")
            last_q_name = q_start

        # Pick
        elif op_name == Pick.name:
            obj, grasp, q = ground_op.values
            assert last_js is not None

            with timer.time(f"{timeline}_planning"):
                start_js = last_js

                # Get the retract pose and plan to it if it's not q0
                if last_q_name != "q0":
                    world_from_ee = world.kin_model.get_state(start_js.position).ee_pose.get_matrix()[0]
                    world_from_retract = world_from_ee @ approach_offset
                    retract_result = motion_gen.plan_single(
                        start_js, Pose.from_matrix(world_from_retract), constrained_plan_config
                    )
                    if not retract_result.success:
                        raise MotionPlanningError(
                            f"Failed to plan for retract for {ground_op.name}. Status: {retract_result.status}"
                        )
                    retract_js = JointState.from_position(retract_result.get_interpolated_plan().position[-1:])
                else:
                    retract_result = None
                    retract_js = start_js

                # Get the approach pose and plan to it
                world_from_obj = obj_to_current_pose[obj]
                if best_particle[grasp].shape == (4, 4):  # already a 4x4, probably came from M2T2
                    obj_from_grasp = best_particle[grasp].clone()
                elif config.grasp_dof == 4:
                    obj_from_grasp = action_4dof_to_mat4x4(best_particle[grasp].clone())
                else:
                    obj_from_grasp = action_6dof_to_mat4x4(best_particle[grasp].clone())
                world_from_grasp = world_from_obj @ obj_from_grasp
                world_from_ee = world_from_grasp @ world.tool_from_ee

                world_from_approach = world_from_ee @ approach_offset
                approach_result = motion_gen.plan_single(retract_js, Pose.from_matrix(world_from_approach), plan_config)
                if not approach_result.success:
                    raise MotionPlanningError(
                        f"Failed to plan for approach for {ground_op.name}. Status: {approach_result.status}"
                    )

                # Plan to from approach to target EE pose for grasp
                approach_js = JointState.from_position(approach_result.get_interpolated_plan().position[-1:])
                end_result = motion_gen.plan_single(
                    approach_js, Pose.from_matrix(world_from_ee), constrained_plan_config
                )
                if not end_result.success:
                    raise MotionPlanningError(
                        f"Failed to plan from approach to end for {ground_op.name}. Status: {end_result.status}"
                    )

            for result in [retract_result, approach_result, end_result]:
                if result is None:
                    continue
                dt = result.interpolation_dt
                plan = result.get_interpolated_plan()
                accum_plans.append(
                    {
                        "type": "trajectory",
                        "plan": plan,
                        "dt": dt,
                        "optimized_plan": result.optimized_plan,
                        "optimized_dt": result.optimized_dt,
                        "label": ground_op.name,
                    }
                )
                last_js = JointState.from_position(plan[-1:].position)
                ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)

            # Temporarily monkey patch get_bounding_spheres to return the spheres we sampled
            obstacle = motion_gen.world_model.get_obstacle(obj)
            obstacle.old_get_bounding_spheres = obstacle.get_bounding_spheres

            def get_bounding_spheres(self, *args, **kwargs) -> List[Sphere]:
                spheres = world.get_collision_spheres(obj)
                pts = spheres[:, :3].cpu().numpy()
                n_radius = spheres[:, 3].cpu().numpy()

                obj_pose = Pose.from_matrix(obj_to_current_pose[obj])
                pre_transform_pose = kwargs["pre_transform_pose"]
                if pre_transform_pose is not None:
                    obj_pose = pre_transform_pose.multiply(obj_pose)  # convert object pose to another frame

                if pts is None or len(pts) == 0:
                    raise ValueError("No points found from the spheres")

                points_cuda = self.tensor_args.to_device(pts)
                pts = obj_pose.transform_points(points_cuda).cpu().view(-1, 3).numpy()

                new_spheres = [
                    Sphere(
                        name=f"{self.name}_sph_{i}",
                        pose=[pts[i, 0], pts[i, 1], pts[i, 2], 1, 0, 0, 0],
                        radius=n_radius[i],
                    )
                    for i in range(pts.shape[0])
                ]
                return new_spheres

            obstacle.get_bounding_spheres = get_bounding_spheres.__get__(obstacle)

            # Attach the object to the robot
            with timer.time(f"{timeline}_planning"):
                motion_gen.attach_objects_to_robot(
                    last_js,
                    object_names=[obj],
                    surface_sphere_radius=0.005,
                    sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
                    voxelize_method="subdivide",
                )

            obstacle.get_bounding_spheres = obstacle.old_get_bounding_spheres
            del obstacle.old_get_bounding_spheres

            # Close the gripper in the visualization
            if config.robot == "ur5" or config.robot == "fr3_robotiq":
                end_val = 0.4
                interp = torch.linspace(0.0, end_val, 20)
                interp = interp[:, None]
            else:
                end_val = 0.02
                interp = torch.linspace(0.04, end_val, 20)[:, None]
                interp = interp.repeat(1, 2)
            dt = 0.02
            accum_plans.append({"type": "gripper", "action": "close", "label": ground_op.name})

            all_pos = last_js.position.expand(interp.shape[0], -1).cpu()
            all_pos = torch.cat([all_pos, interp], dim=1)
            ts = visualizer.log_joint_trajectory(all_pos, timeline=timeline, start_time=ts, dt=dt)

        # Place
        elif op_name == Place.name:
            obj, grasp, placement, surface, q = ground_op.values
            assert last_js is not None

            with timer.time(f"{timeline}_planning"):
                start_js = last_js

                # Plan to retract
                world_from_ee = world.kin_model.get_state(start_js.position).ee_pose.get_matrix()[0]
                world_from_ee_start = world_from_ee
                world_from_retract = world_from_ee @ approach_offset
                retract_result = motion_gen.plan_single(
                    start_js, Pose.from_matrix(world_from_retract), constrained_plan_config
                )
                if not retract_result.success:
                    if (
                        retract_result.status is not None
                        and retract_result.status.name == "INVALID_START_STATE_WORLD_COLLISION"
                    ):
                        kin_config = motion_gen.kinematics.kinematics_config
                        link_name = "attached_object"
                        curr_obj_sphs = kin_config.get_link_spheres(link_name).clone()
                        kin_config.detach_object(link_name)
                        retract_result = motion_gen.plan_single(
                            start_js, Pose.from_matrix(world_from_retract), plan_config
                        )
                        kin_config.attach_object(sphere_tensor=curr_obj_sphs, link_name=link_name)
                        if not retract_result.success:
                            raise MotionPlanningError(
                                f"Failed to plan for retract for {ground_op.name}. Status: {retract_result.status}"
                            )
                    else:
                        raise MotionPlanningError(
                            f"Failed to plan for retract for {ground_op.name}. Status: {retract_result.status}"
                        )

                # Plan from retract to approach
                retract_js = JointState.from_position(retract_result.get_interpolated_plan().position[-1:])
                world_from_obj = action_4dof_to_mat4x4(best_particle[placement].clone())
                if best_particle[grasp].shape == (4, 4):  # already a 4x4, probably came from M2T2
                    obj_from_grasp = best_particle[grasp].clone()
                elif config.grasp_dof == 4:
                    obj_from_grasp = action_4dof_to_mat4x4(best_particle[grasp].clone())
                else:
                    obj_from_grasp = action_6dof_to_mat4x4(best_particle[grasp].clone())
                world_from_grasp = world_from_obj @ obj_from_grasp
                world_from_ee = world_from_grasp @ world.tool_from_ee
                world_from_approaches = world_from_ee @ approach_offsets
                for app_idx, world_from_approach in enumerate(world_from_approaches):
                    approach_result = motion_gen.plan_single(
                        retract_js, Pose.from_matrix(world_from_approach), plan_config
                    )
                    _log.debug(
                        f"Approach attempt {app_idx + 1}/{len(world_from_approaches)}. {approach_result.success}"
                    )
                    if approach_result.success:
                        break

                if not approach_result.success:
                    raise MotionPlanningError(
                        f"Failed to plan for approach for {ground_op.name}. Status: {approach_result.status}"
                    )

                # Plan from approach to end js
                approach_js = JointState.from_position(approach_result.get_interpolated_plan().position[-1:])
                end_result = motion_gen.plan_single(
                    approach_js, Pose.from_matrix(world_from_ee), constrained_plan_config
                )
                if not end_result.success:
                    raise MotionPlanningError(
                        f"Failed to plan from approach to end for {ground_op.name}. Status: {end_result.status}"
                    )

            # Compute the offset between the object and end-effector at start of plan
            obj_from_ee = torch.inverse(obj_to_current_pose[obj]) @ world_from_ee_start
            ee_from_obj = torch.inverse(obj_from_ee)

            for result in [retract_result, approach_result, end_result]:
                dt = result.interpolation_dt
                plan = result.get_interpolated_plan()
                accum_plans.append(
                    {
                        "type": "trajectory",
                        "plan": plan,
                        "dt": dt,
                        "optimized_plan": result.optimized_plan,
                        "optimized_dt": result.optimized_dt,
                        "label": ground_op.name,
                    }
                )
                last_js = JointState.from_position(plan[-1:].position)

                # Forward kinematics to get end-effector pose
                robot_state = world.kin_model.get_state(plan.position)
                world_from_ee = robot_state.ee_pose.get_matrix()
                world_from_obj = world_from_ee @ ee_from_obj
                ts = visualizer.log_joint_trajectory_with_mat4x4(
                    traj=plan.position,
                    mat4x4_key=f"world/{obj}",
                    mat4x4=world_from_obj,
                    timeline=timeline,
                    start_time=ts,
                    dt=dt,
                )

                # Updated pose is the last pose
                obj_to_current_pose[obj] = world_from_obj[-1]

            # Detach object from robot and enable it again
            with timer.time(f"{timeline}_planning"):
                motion_gen.detach_object_from_robot("attached_object")
                motion_gen.world_coll_checker.enable_obstacle(enable=True, name=obj)
                obj_pose = obj_to_current_pose[obj]
                motion_gen.world_collision.update_obstacle_pose(obj, Pose.from_matrix(obj_pose))

            # Open the gripper for visualization purposes
            if config.robot == "ur5" or config.robot == "fr3_robotiq":
                end_val = 0.0
                interp = torch.linspace(0.4, end_val, 20)
                interp = interp[:, None]
            else:
                end_val = 0.04
                interp = torch.linspace(0.02, end_val, 20)[:, None]
                interp = interp.repeat(1, 2)
            dt = 0.02
            accum_plans.append({"type": "gripper", "action": "open", "label": ground_op.name})

            all_pos = last_js.position.expand(interp.shape[0], -1).cpu()
            all_pos = torch.cat([all_pos, interp], dim=1)
            ts = visualizer.log_joint_trajectory(all_pos, timeline=timeline, start_time=ts, dt=dt)

        # Push and PushStick
        elif op_name == Push.name or op_name == PushStick.name:
            # TODO: implement motion solving for these operators
            raise NotImplementedError("Push and PushStick operations are not yet supported in cuRobo motion planning.")

        # Unsupported
        else:
            raise NotImplementedError(f"Unsupported operator {op_name}")

    start_js = last_js

    # Plan to retract
    world_from_ee = world.kin_model.get_state(start_js.position).ee_pose.get_matrix()[0]
    world_from_retract = world_from_ee @ approach_offset
    retract_result = motion_gen.plan_single(start_js, Pose.from_matrix(world_from_retract), constrained_plan_config)
    if not retract_result.success:
        raise MotionPlanningError(f"Failed to plan for retract. Status: {retract_result.status}")
    dt = retract_result.interpolation_dt
    plan = retract_result.get_interpolated_plan()
    accum_plans.append(
        {
            "type": "trajectory",
            "plan": plan,
            "dt": dt,
            "optimized_plan": result.optimized_plan,
            "optimized_dt": result.optimized_dt,
            "label": "GoToInitial(q0)",
        }
    )
    last_js = JointState.from_position(plan[-1:].position)
    ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)

    # Plan to go home at the end which we'll assume is q0
    q_last = last_js.position[0]
    q_home = best_particle["q0"].clone()
    js_last = JointState.from_position(q_last[None])
    js_home = JointState.from_position(q_home[None])
    with timer.time(f"{timeline}_planning"):
        result = motion_gen.plan_single_js(js_last, js_home, plan_config)
    if not result.success:
        raise MotionPlanningError("Failed to plan for going home")

    dt = result.interpolation_dt
    plan = result.get_interpolated_plan()
    accum_plans.append(
        {
            "type": "trajectory",
            "plan": plan,
            "dt": dt,
            "optimized_plan": result.optimized_plan,
            "optimized_dt": result.optimized_dt,
            "label": "GoToInitial(q0)",
        }
    )
    _ = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)
    _log.debug("Planned to go home")

    _log.info(f"Motion planning metrics: {timer.get_summary(f'{timeline}_planning')}")
    return accum_plans


# ------------------------------------------------------------------------------------------------
# Phase 2: batched motion refinement (solve_curobo_batched)
#
# Collapses B independent serial solve_curobo runs into one sweep where each skeleton step is a
# single cuRobo plan_batch_env call over B envs (built via build_batched_curobo_solvers). The three
# helpers below are the batched analogues of the per-env operations solve_curobo does inline.
#
# HARD INVARIANT (verified against cuRobo): plan_batch_env has NO env_query_idx -- batch position i
# maps to collision env i POSITIONALLY. So every plan_batch_env call must pass the FULL B batch (a
# dummy goal for dead/inactive envs) and mask result.success[B] afterward; never compact to a subset.
# ------------------------------------------------------------------------------------------------


def batched_grasp_to_mat4x4(grasp_batch: torch.Tensor, grasp_dof: int) -> torch.Tensor:
    """Batched form of solve_curobo's per-particle grasp->4x4 dispatch (motion_solver.py:148-153).

    grasp_batch is [B, 4, 4] (already a matrix, e.g. from M2T2), [B, 4] (4-DOF), or [B, 6] (6-DOF);
    returns world-agnostic obj_from_grasp as [B, 4, 4]. action_{4,6}dof_to_mat4x4 vectorize over
    leading dims, so the batch axis passes straight through.
    """
    if grasp_batch.shape[1:] == (4, 4):  # already a 4x4 (serial checks shape == (4, 4) per particle)
        return grasp_batch.clone()
    elif grasp_dof == 4:
        return action_4dof_to_mat4x4(grasp_batch.clone())
    else:
        return action_6dof_to_mat4x4(grasp_batch.clone())


def batched_world_from_ee(world: TAMPWorld, positions: torch.Tensor) -> torch.Tensor:
    """Batched FK: joint positions [B, dof] -> world_from_ee [B, 4, 4].

    Serial solve_curobo takes get_matrix()[0] to pick the single env (e.g. motion_solver.py:132,262);
    the batched form keeps all B rows.
    """
    return world.kin_model.get_state(positions).ee_pose.get_matrix()


def reset_batched_world(motion_gen: MotionGen, obj_to_current_pose_list: list[dict[str, torch.Tensor]]) -> None:
    """Batched analogue of solve_curobo's per-run world reset (motion_solver.py:70-74).

    Detaches the single shared attached_object once, then for EACH env re-enables and reseats every
    movable at that env's pose. obj_to_current_pose_list has one {name: [4,4]} dict per env. Every
    movable name must already exist in every env's world_cfg (allocation is fixed at build time), so
    these are in-place, CUDA-graph/alloc-stable pose swaps.
    """
    motion_gen.detach_object_from_robot("attached_object")  # shared kinematics: once, no env_idx
    for env_idx, obj_to_pose in enumerate(obj_to_current_pose_list):
        for obj, pose in obj_to_pose.items():
            motion_gen.world_coll_checker.enable_obstacle(enable=True, name=obj, env_idx=env_idx)
            motion_gen.world_collision.update_obstacle_pose(obj, Pose.from_matrix(pose), env_idx=env_idx)


def _append_batched_traj(accum_list: list, result, paths: list, env_idx: int, label: str) -> None:
    """Append one env's trajectory dict from a batched MotionGenResult (same schema as serial
    solve_curobo, motion_solver.py:179-188). interpolation_dt is a shared scalar; optimized_plan
    (JointState) and optimized_dt are batched [B,...], so index them per env -- EXCEPT cuRobo squeezes
    the batch axis when the plan_batch_env batch is size 1 (optimized_plan -> [T,dof], optimized_dt ->
    scalar), matching the interpolated_plan squeeze handled in _masked_plan. Index only when the batch
    axis is present, else use the tensor directly.
    """
    opt_plan = result.optimized_plan
    opt_dt = result.optimized_dt
    opt_plan_squeezed = opt_plan is not None and getattr(opt_plan, "position", None) is not None and opt_plan.position.ndim == 2
    opt_dt_scalar = opt_dt is not None and getattr(opt_dt, "ndim", 1) == 0
    accum_list.append(
        {
            "type": "trajectory",
            "plan": paths[env_idx],
            "dt": result.interpolation_dt,
            "optimized_plan": None if opt_plan is None else (opt_plan if opt_plan_squeezed else opt_plan[env_idx]),
            "optimized_dt": None if opt_dt is None else (opt_dt if opt_dt_scalar else opt_dt[env_idx]),
            "label": label,
        }
    )


def _cover_spheres(pts: torch.Tensor, radii: torch.Tensor, k: int):
    """Reduce M spheres to k CONSERVATIVE cover spheres: greedy farthest-point pick of k centers,
    then each center's radius grows to bound its assigned members (center distance + member radius).
    Every input sphere is contained in some output sphere."""
    sel = [int(torch.argmin(((pts - pts.mean(0)) ** 2).sum(-1)))]
    d = torch.linalg.norm(pts - pts[sel[0]], dim=-1)
    for _ in range(1, min(k, pts.shape[0])):
        nxt = int(torch.argmax(d))
        sel.append(nxt)
        d = torch.minimum(d, torch.linalg.norm(pts - pts[nxt], dim=-1))
    centers = pts[sel]  # [k,3]
    dist = torch.cdist(pts, centers)  # [M,k]
    assign = dist.argmin(-1)
    need = dist[torch.arange(pts.shape[0], device=pts.device), assign] + radii
    cover_r = torch.zeros(len(sel), device=pts.device, dtype=pts.dtype)
    cover_r.scatter_reduce_(0, assign, need, reduce="amax")
    return centers, cover_r


def _attach_shared_blob(
    motion_gen: MotionGen, world: TAMPWorld, obj: str,
    obj_to_current_pose_list: list[dict[str, torch.Tensor]], positions: torch.Tensor,
    live_mask: torch.Tensor, env_obj_spheres: Optional[dict] = None,
) -> None:
    """Attach ONE conservative sphere blob for a held object, shared across all envs.

    cuRobo's attached_object is a single link_spheres buffer (no env axis), so B envs cannot each hold
    their own object. The blob is the UNION of every live env's held-object occupancy in ITS OWN
    gripper (EE) frame -- env b's true sphere set composed with env b's object pose and grasp joint
    config -- voxel re-fit by attach_objects_to_robot into the fixed attached_object budget. This is
    shape-correct for every env (conservative where envs disagree); sizing from one representative
    env's REP-WORLD spheres (the old behavior, kept as fallback when env_obj_spheres is None) gave 15
    of 16 envs a wrong-shaped blob and was the dominant refine-yield cap. The EE frame here matches
    attach_objects_to_robot's pre_transform (compute_kinematics(js).ee_pose.inverse(), same ee_link
    as world.kin_model). Finally disables the object in EVERY env (attach only disables env 0).

    env_obj_spheres: optional {name: [B, n, 4]} per-env OBJECT-FRAME sphere sets (each env's true
    geometry, refine-env indexed). positions: [B, dof] grasp configs. live_mask: [B] bool -- only
    live envs holding the real object contribute to the union.
    """
    obstacle = motion_gen.world_model.get_obstacle(obj)
    obstacle.old_get_bounding_spheres = obstacle.get_bounding_spheres
    live_idxs = [int(b) for b in torch.nonzero(live_mask).flatten()]
    world_from_ee = batched_world_from_ee(world, positions)  # [B,4,4]

    def get_bounding_spheres(self, *args, **kwargs) -> List[Sphere]:
        n_budget = int(args[0]) if args else 50  # attach passes its per-object sphere budget first
        pts_list, radii_list = [], []
        for b in live_idxs[:1]:  # ANCHOR env only: exact spheres, no cover bulge, no budget pressure.
            # Multi-env unions were tried and abandoned: any cover within the 50-sphere budget bulges
            # ~1-3cm below the true toy surface and collides with the table at the grasp start state,
            # while raising the budget OOMs shared GPUs. Grasp-aligned rank selection (_aligned_rows)
            # makes the anchor's blob approximately correct for the other envs instead.
            if env_obj_spheres is not None and obj in env_obj_spheres:
                sph = env_obj_spheres[obj][b]
            else:
                sph = world.get_collision_spheres(obj)
            ee_from_obj = torch.inverse(world_from_ee[b]) @ obj_to_current_pose_list[b][obj]
            pts_h = torch.cat([sph[:, :3], torch.ones_like(sph[:, :1])], dim=1)
            pts_list.append((pts_h @ ee_from_obj.T)[:, :3])
            radii_list.append(sph[:, 3])
        pts_t, radii_t = torch.cat(pts_list), torch.cat(radii_list)
        if pts_t.shape[0] > n_budget:
            # attach_objects_to_robot copies the returned spheres VERBATIM into a [budget,4] buffer
            # (the fit normally happens inside the real get_bounding_spheres), so reduce the union to
            # the budget with a conservative global cover. This stays tight ONLY because refine rank
            # selection grasp-ALIGNS the envs (_aligned_rows): the per-env clouds then coincide in the
            # gripper frame. Unaligned, a global cover bridges different orientations into giant
            # spheres (measured: swallowed the fingers, killed every held segment).
            pts_t, radii_t = _cover_spheres(pts_t, radii_t, n_budget)
        pts = pts_t.cpu().numpy()
        radii = radii_t.cpu().numpy()
        return [
            Sphere(name=f"{self.name}_sph_{i}", pose=[pts[i, 0], pts[i, 1], pts[i, 2], 1, 0, 0, 0],
                   radius=radii[i])
            for i in range(pts.shape[0])
        ]

    obstacle.get_bounding_spheres = get_bounding_spheres.__get__(obstacle)
    rep_js = JointState.from_position(positions[live_idxs[0] : live_idxs[0] + 1])
    motion_gen.attach_objects_to_robot(
        rep_js, object_names=[obj], surface_sphere_radius=0.005,
        sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE, voxelize_method="subdivide",
    )
    obstacle.get_bounding_spheres = obstacle.old_get_bounding_spheres
    del obstacle.old_get_bounding_spheres
    for b in range(len(obj_to_current_pose_list)):  # attach disabled env 0 only; disable in every env
        motion_gen.world_coll_checker.enable_obstacle(enable=False, name=obj, env_idx=b)


def solve_curobo_batched(
    plan_info: PlanContainer,
    batched_particles: Particles,
    world: TAMPWorld,
    config: TAMPConfiguration,
    motion_gen: MotionGen,
    obj_to_initial_pose_list: list[dict[str, torch.Tensor]],
    B: int,
    timeline: str = "curobo",
    initial_live: Optional[torch.Tensor] = None,
    phantom_objs_per_env: Optional[List[set]] = None,
    batched_obj_spheres: Optional[dict] = None,
):
    """Batched analogue of solve_curobo (Phase 2): refine B scenes' plans in one plan_batch_env sweep.

    Supports MoveFree/MoveHolding bookkeeping, Pick (retract/approach/grasp + shared-blob attach),
    Place (retract/approach/place + detach/reseat), the final retract, and a serial-over-survivors
    go-home. HELD segments use ONE conservative shared attached blob (per-env attach is unsupported).
    Not yet: the Place 4-offset approach retry (#5) and the INVALID_START detach-replan fallback (#6)
    -- the happy path uses approach offset 0 and lets envs that miss die (masked). Push/PushStick raise.

    Requires a batched solver from build_batched_curobo_solvers (n_envs == B). All B scenes share ONE
    plan_skeleton; only continuous particle values, object poses, and per-env collision worlds differ.
    batched_particles maps key -> [B, ...] (q0 [B,dof]; grasp [B,4,4]/[B,4]/[B,6]); obj_to_initial_pose_
    list has one {name: [4,4]} dict per env.

    INVARIANT: every plan_batch_env call passes the full B batch (a no-op goal = current EE pose for
    dead/inactive envs) and masks result.success[B] after -- never a compacted subset (plan_batch_env
    maps batch position i to collision env i positionally, with no env_query_idx to remap).

    Returns (accum_plans_per_env, live): a length-B list of per-env accum_plans, and the [B] bool mask
    of envs that produced a complete plan.

    initial_live ([B] bool, optional): envs that start dead (False) are PADDING -- they ride along in
    every full-B plan_batch_env call with no-op goals (free: wall is flat in B) but skip the attach-rep
    choice, the per-env serial go-home, and never produce results. Used by run_cutamp_batched to pad
    groups to a fixed env count so the batched-solver cache hits across batches.

    phantom_objs_per_env (list of B name sets, optional): objects that are PHANTOMS (count-padding
    ghosts) in a given env. A Pick/Place op whose object is phantom for env b is SKIPPED for b (b is
    inactive in the op's segments, appends nothing, keeps its pose/live state) while envs holding the
    real object plan normally. Phantom cycles are always a SUFFIX of an env's real ops (canonical
    pick-order naming), so a masked env simply coasts to the final retract. The phantom cuboids
    themselves stay ordinary world obstacles in every env, so name-scoped attach/reset/reseat calls
    resolve everywhere.

    batched_obj_spheres (optional {name: [B, n, 4]}): per-env object-frame sphere sets used to build
    the held-object attach blob as a union over live envs' true geometry (see _attach_shared_blob).
    """
    plan_skeleton = plan_info["plan_skeleton"]

    plan_config = MotionGenPlanConfig(
        timeout=0.25, enable_finetune_trajopt=False, time_dilation_factor=config.time_dilation_factor
    )
    constrained_motion_cost_metric = PoseCostMetric(
        hold_partial_pose=True,
        hold_vec_weight=world.tensor_args.to_device([0.1, 0.1, 0.1, 0.1, 0.1, 0.0]),
        project_to_goal_frame=True,
    )
    constrained_plan_config = plan_config.clone()
    constrained_plan_config.pose_cost_metric = constrained_motion_cost_metric

    # Per-env carried state as [B]-leading tensors / lists.
    q0_batch = batched_particles["q0"]  # [B, dof]
    device = q0_batch.device
    last_js = JointState.from_position(q0_batch.clone())  # [B, dof]
    last_q_name = ["q0"] * B  # per-env; gates the Pick retract (uniform under a shared skeleton)
    obj_to_current_pose_list = [{k: v.clone() for k, v in d.items()} for d in obj_to_initial_pose_list]
    live = (torch.ones(B, dtype=torch.bool, device=device) if initial_live is None
            else initial_live.to(device=device, dtype=torch.bool).clone())
    _all_envs = torch.ones(B, dtype=torch.bool, device=device)

    def _op_env_mask(obj_: str) -> torch.Tensor:
        """[B] bool, False where obj_ is a phantom for that env (the op is skipped for it)."""
        if phantom_objs_per_env is None:
            return _all_envs
        return torch.tensor([obj_ not in s for s in phantom_objs_per_env], dtype=torch.bool, device=device)
    accum_plans_per_env: list[list] = [[] for _ in range(B)]

    reset_batched_world(motion_gen, obj_to_current_pose_list)

    approach_offset = torch.eye(4, device=world.device)
    approach_offset[2, 3] = -0.05
    held_attach = None  # (obj, grasp-time {obj: pose} per env, grasp-time positions, live mask)

    def _masked_plan(start_js_: JointState, goal_mats: torch.Tensor, active_: torch.Tensor, cfg):
        """One full-B plan_batch_env with a no-op goal (= current EE pose) for inactive/dead envs, so
        the batch stays size B and collision env indices stay aligned. Returns (result, paths, success)."""
        ee_now = batched_world_from_ee(world, start_js_.position)  # [B,4,4]
        goal = torch.where(active_.view(B, 1, 1), goal_mats, ee_now)
        # [scale-diag] time the batched solve keyed by B to measure sub-linear scaling on real geometry:
        # if plan_batch_env at B=3 costs ~= B=1, batching parallelizes (canonicalization pays off);
        # if it costs ~3x, it doesn't. Sync CUDA so we time the GPU solve, not just the async launch.
        torch.cuda.synchronize()
        _t0 = time.perf_counter()
        result = motion_gen.plan_batch_env(start_js_, Pose.from_matrix(goal), cfg)
        torch.cuda.synchronize()
        _plan_ms = (time.perf_counter() - _t0) * 1e3
        # result.get_paths() raises "JointState does not have horizon" for any env whose plan_batch_env
        # produced no real trajectory -- notably the trivial no-op goal fed to INACTIVE envs (start->start
        # has zero horizon), and hard failures on cluttered real scenes. Extract per-env robustly instead;
        # inactive/failed envs -> None and are never used (callers gate on active/success masks).
        ip, lt = result.interpolated_plan, result.path_buffer_last_tstep
        paths = [None] * B
        if ip is not None and lt is not None:
            # cuRobo squeezes the batch axis when the plan_batch_env batch is size 1: interpolated_plan
            # comes back [T,dof] (2D) instead of [B,T,dof] (3D). Its own API mirrors this split --
            # get_interpolated_plan() trims the whole tensor directly for len==1, get_paths() indexes
            # [b] for len>1. Indexing ip[b] on a 2D plan grabs a single timestep (no horizon) and
            # trim_trajectory raises "JointState does not have horizon" -> None -> usable=0 for EVERY
            # active B=1 segment (real scene-6 is all singleton groups). Discriminate on ndim so both
            # the squeezed single-env and the true-batch case extract a real trajectory.
            squeezed = getattr(ip, "position", None) is not None and ip.position.ndim == 2
            if squeezed:
                try:
                    paths[0] = ip.trim_trajectory(0, lt[0])
                except (ValueError, IndexError, TypeError):
                    paths[0] = None
            else:
                for b in range(B):
                    try:
                        paths[b] = ip[b].trim_trajectory(0, lt[b])
                    except (ValueError, IndexError, TypeError):
                        paths[b] = None
        # plan_batch_env can report success=True for an env with no usable interpolated trajectory
        # (e.g. a degenerate 0-horizon plan). Mask those to failed so success[b] implies paths[b] is a
        # real trajectory -- callers dereference paths[b].position only when success[b] is True.
        succ = result.success.clone()
        for b in range(B):
            if paths[b] is None:
                succ[b] = False
        _log.info(f"    [seg-diag] plan_batch_env: active={int(active_.sum().item())}/{B} "
                  f"-> raw_succ={int(result.success.sum().item())} usable={int(succ.sum().item())} "
                  f"wall={_plan_ms:.0f}ms status={getattr(result, 'status', None)}")
        return result, paths, succ

    for idx, ground_op in enumerate(plan_skeleton):
        op_name = ground_op.operator.name

        if op_name == MoveFree.name:
            q_start, traj, q_end = ground_op.values
            if traj in batched_particles:
                raise NotImplementedError("Trajectories not supported yet")
            last_q_name = [q_start] * B  # pure bookkeeping, no planning

        elif op_name == Pick.name:
            obj, grasp, q = ground_op.values
            start_js = last_js
            opm = _op_env_mask(obj)  # False where obj is a phantom for that env -> op skipped there

            # --- retract (only where live, op applies, and last_q_name != q0) ---
            active = live & opm
            for b in range(B):
                if last_q_name[b] == "q0":
                    active[b] = False
            world_from_ee = batched_world_from_ee(world, start_js.position)
            world_from_retract = world_from_ee @ approach_offset
            r_res, r_paths, r_succ = _masked_plan(start_js, world_from_retract, active, constrained_plan_config)
            retract_pos = start_js.position.clone()
            for b in range(B):
                if active[b] and r_succ[b]:
                    retract_pos[b] = r_paths[b].position[-1]
                    _append_batched_traj(accum_plans_per_env[b], r_res, r_paths, b, ground_op.name)
            live = live & (r_succ | ~active)  # active-but-failed envs die
            retract_js = JointState.from_position(retract_pos)

            # --- approach (live envs the op applies to) ---
            obj_from_grasp = batched_grasp_to_mat4x4(batched_particles[grasp], config.grasp_dof)  # [B,4,4]
            world_from_obj = torch.stack([obj_to_current_pose_list[b][obj] for b in range(B)])  # [B,4,4]
            world_from_grasp = world_from_obj @ obj_from_grasp
            world_from_ee_grasp = world_from_grasp @ world.tool_from_ee
            world_from_approach = world_from_ee_grasp @ approach_offset
            a_res, a_paths, a_succ = _masked_plan(retract_js, world_from_approach, live & opm, plan_config)
            approach_pos = retract_js.position.clone()
            for b in range(B):
                if live[b] and opm[b] and a_succ[b]:
                    approach_pos[b] = a_paths[b].position[-1]
                    _append_batched_traj(accum_plans_per_env[b], a_res, a_paths, b, ground_op.name)
            live = live & (a_succ | ~opm)  # op-masked envs coast; only applied-and-failed envs die
            approach_js = JointState.from_position(approach_pos)

            # --- approach -> grasp EE pose (constrained) ---
            e_res, e_paths, e_succ = _masked_plan(approach_js, world_from_ee_grasp, live & opm, constrained_plan_config)
            end_pos = approach_js.position.clone()
            for b in range(B):
                if live[b] and opm[b] and e_succ[b]:
                    end_pos[b] = e_paths[b].position[-1]
                    _append_batched_traj(accum_plans_per_env[b], e_res, e_paths, b, ground_op.name)
            live = live & (e_succ | ~opm)
            last_js = JointState.from_position(end_pos)

            # gripper close (bookkeeping, per live env the op applies to)
            for b in range(B):
                if live[b] and opm[b]:
                    accum_plans_per_env[b].append({"type": "gripper", "action": "close", "label": ground_op.name})

            # Attach the picked object as a shared conservative blob (held through subsequent Place):
            # the union of all live real-holding envs' gripper-frame occupancy (their true geometry).
            # Remember the grasp-time state: the Place lift-off replans with the blob detached and
            # reattaches from this exact state (gripper-frame offsets are invariant while held).
            if bool((live & opm).any()):
                _attach_shared_blob(motion_gen, world, obj, obj_to_current_pose_list, last_js.position,
                                    live & opm, env_obj_spheres=batched_obj_spheres)
                held_attach = (obj, [{obj: d[obj].clone()} for d in obj_to_current_pose_list],
                               last_js.position.clone(), (live & opm).clone())

        elif op_name == MoveHolding.name:
            obj, grasp, q_start, traj, q_end = ground_op.values
            if traj in batched_particles:
                raise NotImplementedError("Trajectories not supported yet")
            last_q_name = [q_start] * B  # pure bookkeeping while holding

        elif op_name == Place.name:
            obj, grasp, placement, surface, q = ground_op.values
            start_js = last_js
            opm = _op_env_mask(obj)

            # --- retract (constrained), planned with the blob DETACHED: at the grasp start state the
            # held object touches the table, so an attached blob makes waypoint 0 invalid and the
            # lift-off fails wholesale. The serial pipeline hits the same wall and recovers via its
            # INVALID_START detach-replan fallback; this is the batched analog, applied preemptively
            # to the one segment where the held object is guaranteed table-adjacent. The blob is
            # reattached from grasp-time state for the approach/place segments below. ---
            world_from_ee_start = batched_world_from_ee(world, start_js.position)  # [B,4,4], saved for ee_from_obj
            world_from_retract = world_from_ee_start @ approach_offset
            motion_gen.detach_object_from_robot("attached_object")
            r_res, r_paths, r_succ = _masked_plan(start_js, world_from_retract, live & opm, constrained_plan_config)
            retract_pos = start_js.position.clone()
            for b in range(B):
                if live[b] and opm[b] and r_succ[b]:
                    retract_pos[b] = r_paths[b].position[-1]
                    _append_batched_traj(accum_plans_per_env[b], r_res, r_paths, b, ground_op.name)
            live = live & (r_succ | ~opm)
            retract_js = JointState.from_position(retract_pos)
            if held_attach is not None:  # restore the shared blob for the held approach/place
                a_obj, a_poses, a_pos, a_mask = held_attach
                _attach_shared_blob(motion_gen, world, a_obj, a_poses, a_pos, a_mask,
                                    env_obj_spheres=batched_obj_spheres)

            # --- approach (offset 0 only for happy path; the 4-offset retry is #5) ---
            world_from_obj_place = action_4dof_to_mat4x4(batched_particles[placement].clone())  # [B,4,4]
            obj_from_grasp = batched_grasp_to_mat4x4(batched_particles[grasp], config.grasp_dof)  # [B,4,4]
            world_from_grasp = world_from_obj_place @ obj_from_grasp
            world_from_ee_place = world_from_grasp @ world.tool_from_ee  # [B,4,4]
            world_from_approach = world_from_ee_place @ approach_offset
            a_res, a_paths, a_succ = _masked_plan(retract_js, world_from_approach, live & opm, plan_config)
            approach_pos = retract_js.position.clone()
            for b in range(B):
                if live[b] and opm[b] and a_succ[b]:
                    approach_pos[b] = a_paths[b].position[-1]
                    _append_batched_traj(accum_plans_per_env[b], a_res, a_paths, b, ground_op.name)
            live = live & (a_succ | ~opm)
            approach_js = JointState.from_position(approach_pos)

            # --- approach -> place EE pose (constrained) ---
            e_res, e_paths, e_succ = _masked_plan(approach_js, world_from_ee_place, live & opm, constrained_plan_config)
            end_pos = approach_js.position.clone()
            for b in range(B):
                if live[b] and opm[b] and e_succ[b]:
                    end_pos[b] = e_paths[b].position[-1]
                    _append_batched_traj(accum_plans_per_env[b], e_res, e_paths, b, ground_op.name)
            live = live & (e_succ | ~opm)
            last_js = JointState.from_position(end_pos)

            # rigid object<-EE offset (constant while held), then FK the final placed object pose from the
            # end waypoint (mirror serial motion_solver.py:327-360, final value only; viz skipped).
            world_from_obj_now = torch.stack([obj_to_current_pose_list[b][obj] for b in range(B)])  # [B,4,4]
            ee_from_obj = torch.inverse(torch.inverse(world_from_obj_now) @ world_from_ee_start)  # [B,4,4]
            world_from_ee_final = batched_world_from_ee(world, last_js.position)  # [B,4,4]
            world_from_obj_final = world_from_ee_final @ ee_from_obj  # [B,4,4]
            for b in range(B):
                if live[b] and opm[b]:
                    obj_to_current_pose_list[b][obj] = world_from_obj_final[b]

            # detach the shared blob (global), re-enable + reseat the object per env at its NEW placed pose
            motion_gen.detach_object_from_robot("attached_object")
            held_attach = None
            for b in range(B):
                motion_gen.world_coll_checker.enable_obstacle(enable=True, name=obj, env_idx=b)
                motion_gen.world_collision.update_obstacle_pose(
                    obj, Pose.from_matrix(obj_to_current_pose_list[b][obj]), env_idx=b
                )

            for b in range(B):  # gripper open (bookkeeping, per live env the op applies to)
                if live[b] and opm[b]:
                    accum_plans_per_env[b].append({"type": "gripper", "action": "open", "label": ground_op.name})

        elif op_name == Push.name or op_name == PushStick.name:
            raise NotImplementedError("Push and PushStick are not supported in cuRobo motion planning.")
        else:
            raise NotImplementedError(f"Unsupported operator {op_name}")

    # --- final retract (all live) ---
    start_js = last_js
    world_from_ee = batched_world_from_ee(world, start_js.position)
    world_from_retract = world_from_ee @ approach_offset
    fr_res, fr_paths, fr_succ = _masked_plan(start_js, world_from_retract, live, constrained_plan_config)
    fr_pos = start_js.position.clone()
    for b in range(B):
        if live[b] and fr_succ[b]:
            fr_pos[b] = fr_paths[b].position[-1]
            # Bug fix vs serial L409 (which appended a stale `result.optimized_plan`): use fr_res's own.
            _append_batched_traj(accum_plans_per_env[b], fr_res, fr_paths, b, "GoToInitial(q0)")
    live = live & fr_succ
    last_js = JointState.from_position(fr_pos)

    # --- go home: ONE batched EE-pose plan to FK(q0) per env. The old per-env serial plan_single_js
    # collision-checked env 0's world for EVERY env (a single-env query on a batched checker), so
    # envs b>0 planned against the wrong scene; the full-B masked pose-goal plan checks each env
    # against its own world. The end configuration may differ from q0 by IK multiplicity, but the EE
    # returns to the home pose, which is what episode replay needs.
    world_from_home = batched_world_from_ee(world, q0_batch)  # [B,4,4]
    gh_res, gh_paths, gh_succ = _masked_plan(last_js, world_from_home, live, plan_config)
    for b in range(B):
        if live[b] and gh_succ[b]:
            _append_batched_traj(accum_plans_per_env[b], gh_res, gh_paths, b, "GoToInitial(q0)")
    live = live & gh_succ

    return accum_plans_per_env, live
