# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Solving motions with cuRobo."""

import contextlib
import logging
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
from cutamp.robots.bimanual_yam import YAM_FINGER_CLOSED, YAM_FINGER_OPEN
from cutamp.tamp_domain import (Handover, MoveHolding, MoveHoldingBoth, MoveHoldingGiver,
                                MoveHoldingTaker, Push, PushStick, MoveFree, Place, PlaceBoth,
                                PlaceTaker, Pick, PickBoth, PickGiver)
from cutamp.tamp_world import TAMPWorld
from cutamp.utils.common import Particles, action_6dof_to_mat4x4, action_4dof_to_mat4x4
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import Visualizer

_log = logging.getLogger(__name__)


class MotionPlanningError(RuntimeError):
    pass


def _start_state_in_world_collision(result) -> bool:
    """Whether cuRobo rejected a plan because its START configuration is in world collision."""
    return result.status is not None and result.status.name == "INVALID_START_STATE_WORLD_COLLISION"


@contextlib.contextmanager
def _obstacles_hidden(motion_gen, names):
    """Hide named world obstacles for one planning segment, then restore them.

    For the reach-into surfaces (``TAMPWorld.pick_transparent``): an open container reconstructs as
    a filled solid, so the legs that deliberately enter it -- the grasp approach, and the retract
    that lifts the held object back out -- are unplannable while it is enabled. Every other segment,
    transit above all, keeps it as a full obstacle.
    """
    names = tuple(names)
    for name in names:
        motion_gen.world_coll_checker.enable_obstacle(name=name, enable=False)
    try:
        yield
    finally:
        for name in names:
            motion_gen.world_coll_checker.enable_obstacle(name=name, enable=True)


def _plan_transit(
    motion_gen,
    start_js: JointState,
    world_from_goal: torch.Tensor,
    plan_config: MotionGenPlanConfig,
    world: TAMPWorld,
    apex_height: float,
    apex_min_dist: float,
):
    """Plan one free-space transit leg, optionally arcing over an explicit APEX waypoint.

    This is the long unconstrained leg of a Pick or Place -- retract -> pre-grasp/pre-place. Planned
    directly it is a joint-space geodesic, which the end-effector traces as a low lateral sweep across
    the table, because nothing in cuRobo's trajopt cost reads the EE's Cartesian path (see
    ``TAMPConfiguration.transit_apex_height``). Splitting it at an apex above the straight line turns
    the same leg into lift -> traverse -> descend.

    The apex pose is the horizontal midpoint of the start and goal EE positions, ``apex_height`` above
    the higher of the two, carrying the GOAL orientation so the second leg is a near-straight descent
    with the wrist already aligned. It is planned with the same ``plan_config`` (and, at the call
    site, the same hidden-obstacle context) as the direct plan it replaces.

    Returns ``(results, last_result)``. ``results`` is the list of successful MotionGenResults to
    append to the plan -- two when the apex was used, one for a direct plan -- or None if the transit
    failed outright. ``last_result`` is the final cuRobo result either way, so callers can inspect
    ``.status`` (the Place ladder keys off INVALID_START_STATE_WORLD_COLLISION).

    Any apex failure falls back to the direct plan rather than failing the transit: the apex is a
    preference about path shape, not a constraint, and an apex that happens to be unreachable (near a
    joint limit, under a ceiling, inside an obstacle) must not cost us a plan we would otherwise find.
    """
    goal_pose = Pose.from_matrix(world_from_goal)
    if apex_height > 0.0:
        world_from_start = world.kin_model.get_state(start_js.position).ee_pose.get_matrix()[0]
        start_pos, goal_pos = world_from_start[:3, 3], world_from_goal[:3, 3]
        horizontal_dist = torch.linalg.norm(goal_pos[:2] - start_pos[:2]).item()
        if horizontal_dist < apex_min_dist:
            _log.debug(
                f"Transit is {horizontal_dist:.3f}m horizontally (< {apex_min_dist}m), skipping the apex"
            )
        else:
            world_from_apex = world_from_goal.clone()  # goal orientation, apex position
            world_from_apex[:2, 3] = 0.5 * (start_pos[:2] + goal_pos[:2])
            world_from_apex[2, 3] = torch.maximum(start_pos[2], goal_pos[2]) + apex_height
            apex_result = motion_gen.plan_single(start_js, Pose.from_matrix(world_from_apex), plan_config)
            if apex_result.success:
                apex_js = JointState.from_position(apex_result.get_interpolated_plan().position[-1:])
                descend_result = motion_gen.plan_single(apex_js, goal_pose, plan_config)
                if descend_result.success:
                    _log.debug(f"Transit planned via apex {apex_height}m above the straight line")
                    return [apex_result, descend_result], descend_result
                _log.debug(f"Apex -> goal leg failed ({descend_result.status}); falling back to direct")
            else:
                _log.debug(f"Start -> apex leg failed ({apex_result.status}); falling back to direct")

    direct_result = motion_gen.plan_single(start_js, goal_pose, plan_config)
    return ([direct_result] if direct_result.success else None), direct_result


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
    q_return: Optional[torch.Tensor] = None,
):
    """
    Solve for full motion plan given a plan skeleton and optimized particles.
    Note that visualization adds non-trivial overhead.

    ``q_return`` overrides the configuration the closing GoToInitial drives to. It defaults to q0 --
    the configuration this plan started from -- which is what a standalone plan wants. A caller that
    CONCATENATES plans needs the override: the second plan starts wherever the first one handed over,
    so its q0 is a mid-episode pose, and defaulting would end the episode by driving back to it.
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
                with _obstacles_hidden(motion_gen, world.pick_transparent):
                    # Free-space transit to the pre-grasp pose, optionally arcing over an apex
                    # waypoint instead of sweeping across the table (transit_apex_height).
                    approach_results, last_approach = _plan_transit(
                        motion_gen, retract_js, world_from_approach, plan_config, world,
                        config.transit_apex_height, config.transit_apex_min_dist,
                    )
                    if approach_results is None:
                        raise MotionPlanningError(
                            f"Failed to plan for approach for {ground_op.name}. Status: {last_approach.status}"
                        )

                    # Plan to from approach to target EE pose for grasp
                    approach_js = JointState.from_position(last_approach.get_interpolated_plan().position[-1:])
                    end_result = motion_gen.plan_single(
                        approach_js, Pose.from_matrix(world_from_ee), constrained_plan_config
                    )
                if not end_result.success:
                    raise MotionPlanningError(
                        f"Failed to plan from approach to end for {ground_op.name}. Status: {end_result.status}"
                    )

            for result in [retract_result, *approach_results, end_result]:
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
            if config.robot.startswith("bimanual_yam_"):
                # Two prismatic fingers, open at YAM_FINGER_OPEN (negative) and closed at 0.
                interp = torch.linspace(YAM_FINGER_OPEN, YAM_FINGER_CLOSED, 20)[:, None].repeat(1, 2)
            elif config.robot == "ur5" or config.robot == "fr3_robotiq":
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

                # Where the object has to end up. Independent of the retract, so compute it once.
                world_from_ee_start = world.kin_model.get_state(start_js.position).ee_pose.get_matrix()[0]
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

                # Lift the held object clear of whatever it came out of, THEN transit to the place.
                #
                # The retract runs with the reach-into surfaces hidden (it starts inside one) and as
                # a LADDER rather than a fixed 5 cm, because the approach that follows starts where
                # the retract ended and runs with those surfaces back ON. If the retract is too
                # short the held object is still inside the container's solid proxy and cuRobo
                # rejects the approach with INVALID_START_STATE_WORLD_COLLISION -- identically for
                # every approach offset, since those vary only the GOAL. Measured on a scene reset:
                # a plate rim topping out at z = +0.042 against toy bottoms at z = -0.011 needs
                # 5.1-5.4 cm of lift, and the grasps are near-vertical so a 5 cm tool-axis retract
                # supplies at most 5.0 cm -- 0/56 grasps cleared, and every one of those picks
                # failed. Climbing the ladder is what keeps the approach itself honest: it is an
                # unconstrained free-space plan across the table, and planning THAT with the
                # container hidden would let cuRobo sweep the held object straight through it.
                retract_result = approach_results = None
                retract_status = approach_status = None
                for ret_idx, world_from_retract in enumerate(world_from_ee_start @ approach_offsets):
                    with _obstacles_hidden(motion_gen, world.pick_transparent):
                        ret_result = motion_gen.plan_single(
                            start_js, Pose.from_matrix(world_from_retract), constrained_plan_config
                        )
                        if not ret_result.success and _start_state_in_world_collision(ret_result):
                            # Start state still invalid: the held object's own spheres. Hide it for
                            # this one short constrained move, then put it straight back.
                            kin_config = motion_gen.kinematics.kinematics_config
                            link_name = "attached_object"
                            curr_obj_sphs = kin_config.get_link_spheres(link_name).clone()
                            kin_config.detach_object(link_name)
                            ret_result = motion_gen.plan_single(
                                start_js, Pose.from_matrix(world_from_retract), plan_config
                            )
                            kin_config.attach_object(sphere_tensor=curr_obj_sphs, link_name=link_name)
                    if not ret_result.success:
                        retract_status = ret_result.status
                        continue

                    retract_js = JointState.from_position(ret_result.get_interpolated_plan().position[-1:])
                    for app_idx, world_from_approach in enumerate(world_from_approaches):
                        # Free-space transit to the pre-place pose, optionally arcing over an apex
                        # waypoint instead of sweeping across the table (transit_apex_height).
                        app_results, app_result = _plan_transit(
                            motion_gen, retract_js, world_from_approach, plan_config, world,
                            config.transit_apex_height, config.transit_apex_min_dist,
                        )
                        _log.debug(
                            f"Retract attempt {ret_idx + 1}/{len(approach_offsets)}, approach attempt "
                            f"{app_idx + 1}/{len(world_from_approaches)}. {app_results is not None}"
                        )
                        if app_results is not None:
                            break
                    approach_status = app_result.status
                    if app_results is not None:
                        retract_result, approach_results = ret_result, app_results
                        break
                    if not _start_state_in_world_collision(app_result):
                        # A longer retract only moves the approach's START state, so a failure that
                        # is not about the start state will not be fixed by climbing the ladder.
                        break

                if approach_results is None:
                    if retract_status is not None and approach_status is None:
                        raise MotionPlanningError(
                            f"Failed to plan for retract for {ground_op.name}. Status: {retract_status}"
                        )
                    raise MotionPlanningError(
                        f"Failed to plan for approach for {ground_op.name}. Status: {approach_status}"
                    )

                # Plan from approach to end js
                approach_js = JointState.from_position(approach_results[-1].get_interpolated_plan().position[-1:])
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

            for result in [retract_result, *approach_results, end_result]:
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
            if config.robot.startswith("bimanual_yam_"):
                interp = torch.linspace(YAM_FINGER_CLOSED, YAM_FINGER_OPEN, 20)[:, None].repeat(1, 2)
            elif config.robot == "ur5" or config.robot == "fr3_robotiq":
                end_val = 0.0
                interp = torch.linspace(0.4, end_val, 20)
                interp = interp[:, None]
            else:
                end_val = 0.04
                interp = torch.linspace(0.02, end_val, 20)[:, None]
                interp = interp.repeat(1, 2)
            dt = 0.02
            accum_plans.append(
                {
                    "type": "gripper",
                    "action": "open",
                    "label": ground_op.name,
                    # Where this Place leaves the object, in world frame. Recorded on the step because
                    # this is the only point it is known exactly: it is what the collision world is
                    # updated to on the line above, and it accounts for the approach offsets and the
                    # attachment transform that a caller reconstructing it from the trajectory's
                    # kinematics would have to re-derive (and get wrong -- ee_from_obj above is taken
                    # at the START of this operator, not at the grasp). A consumer that plans a
                    # FOLLOW-ON problem against the post-plan scene needs exactly this.
                    "placed_object": obj,
                    "world_from_obj": obj_pose.detach().cpu().numpy(),
                }
            )

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

    # Plan to go home at the end, which is q0 unless the caller named somewhere else (see q_return).
    q_last = last_js.position[0]
    q0 = best_particle["q0"]
    q_home = q0.clone() if q_return is None else torch.as_tensor(q_return).to(q0).clone()
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


def solve_curobo_dual(
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
    """Motion-plan a LOCKSTEP dual-arm skeleton, where both hands act at the same timestep.

    Kept separate from :func:`solve_curobo` rather than folded into it, because the two differ in the
    one place that matters: ``solve_curobo`` drives every segment with ``plan_single`` against a
    single ``ee_link`` pose target, which on a 12-DOF dual chain would constrain ONE hand and leave
    the other arm's six joints floating in the null space. There is no cuRobo API for giving two
    hands a pose target at once.

    So this plans in CONFIGURATION space instead: cuTAMP has already optimized a full 12-DOF
    configuration for each timestep, satisfying both hands' kinematic constraints simultaneously, so
    ``plan_single_js`` between consecutive configurations is both sufficient and exactly right -- it
    is the only cuRobo motion call that is DOF-agnostic and needs no pose target. The trade-off is
    that a joint-space goal cannot express "approach along the tool's -z", so the cartesian
    approach/retract segments of the single-arm solver have no counterpart here.

    Gripper steps carry an ``"arm"`` key so the two hands can be actuated independently downstream;
    a grasped object is attached to that arm's own ``<arm>_attached_object`` link.
    """
    plan_skeleton = plan_info["plan_skeleton"]
    arms = world.arms
    if not arms:
        raise ValueError("solve_curobo_dual requires a multi-arm robot container")

    if motion_gen is None:
        motion_gen = world.get_motion_gen(collision_activation_distance=config.world_activation_distance)
    if config.warmup_motion_gen:
        with timer.time(f"{timeline}_motion_gen_warmup", log_callback=_log.debug):
            motion_gen.warmup()

    # Far more generous than the single-arm pose-target config. A joint-space goal over 12 DOF is a
    # much longer motion than the single-arm solver's short cartesian hops (which chain
    # retract -> approach -> grasp), so trajopt alone reliably fails on it; the graph planner finds a
    # collision-free seed and finetuning cleans it up.
    plan_config = MotionGenPlanConfig(
        timeout=15.0,
        enable_finetune_trajopt=True,
        enable_graph=True,
        enable_graph_attempt=1,
        max_attempts=8,
        time_dilation_factor=config.time_dilation_factor,
    )

    ts = 0.0
    obj_to_current_pose = {k: v.clone() for k, v in obj_to_initial_pose.items()}

    # Reset: clear BOTH hands' attachments and restore every object's pose.
    for spec in arms:
        motion_gen.detach_object_from_robot(spec.attached_link)
    for obj, obj_pose in obj_to_current_pose.items():
        motion_gen.world_coll_checker.enable_obstacle(enable=True, name=obj)
        motion_gen.world_collision.update_obstacle_pose(obj, Pose.from_matrix(obj_pose))

    visualizer.set_time_seconds(timeline, ts)
    visualizer.set_joint_positions(best_particle["q0"])
    for obj, pose in obj_to_current_pose.items():
        visualizer.log_mat4x4(f"world/{obj}", pose)

    last_js = JointState.from_position(best_particle["q0"][None].clone())
    accum_plans = []

    @contextlib.contextmanager
    def temporarily_detached(specs):
        """Hide the given hands' attached-object spheres from the collision model for one solve.

        Nesting is safe: an inner block stashes whatever the outer one left (already-empty spheres)
        and restores that, and the outer block restores the real ones.
        """
        kin_config = motion_gen.kinematics.kinematics_config
        stashed = {}
        for spec in specs:
            try:
                stashed[spec.attached_link] = kin_config.get_link_spheres(spec.attached_link).clone()
                kin_config.detach_object(spec.attached_link)
            except Exception:  # nothing attached to this hand
                pass
        try:
            yield
        finally:
            for link_name, spheres in stashed.items():
                kin_config.attach_object(sphere_tensor=spheres, link_name=link_name)

    def plan_to(q_name, label: str):
        """Joint-space segment from the current state to a configuration (name or tensor)."""
        nonlocal last_js, ts
        goal = q_name if torch.is_tensor(q_name) else best_particle[q_name]
        goal_js = JointState.from_position(goal[None].clone())
        with timer.time(f"{timeline}_planning"):
            result = motion_gen.plan_single_js(last_js, goal_js, plan_config)
        if (
            not result.success
            and result.status is not None
            and result.status.name == "INVALID_START_STATE_WORLD_COLLISION"
        ):
            # Right after a grasp the held objects are attached bodies still resting on the table, so
            # the start state reads as in-collision and nothing can be planned OUT of it. Detach the
            # attached spheres for the duration of this solve and restore them after -- the same
            # fallback solve_curobo uses for its cartesian retract, extended to every hand.
            with temporarily_detached(arms), timer.time(f"{timeline}_planning"):
                result = motion_gen.plan_single_js(last_js, goal_js, plan_config)
        if not result.success:
            name = "a lifted configuration" if torch.is_tensor(q_name) else q_name
            raise MotionPlanningError(f"Failed to plan to {name} for {label}. Status: {result.status}")
        plan = result.get_interpolated_plan()
        accum_plans.append({
            "type": "trajectory",
            "plan": plan,
            "dt": result.interpolation_dt,
            "optimized_plan": result.optimized_plan,
            "optimized_dt": result.optimized_dt,
            "label": label,
        })
        last_js = JointState.from_position(plan[-1:].position)
        ts = visualizer.log_joint_trajectory(
            plan.position, timeline=timeline, start_time=ts, dt=result.interpolation_dt
        )

    def lift_configuration(q_name_or_tensor, dz: float = 0.10):
        """A configuration with both hands raised ``dz`` in world z, via per-arm IK.

        The joint-space counterpart of ``solve_curobo``'s cartesian retract, and just as necessary:
        immediately after a grasp the held object is still sitting on the table, so once it becomes
        an attached body every subsequent plan starts in world collision. ``plan_single_js`` cannot
        express "move along the tool axis", so the lifted pose is turned back into a configuration by
        solving each arm's IK for its own raised end-effector pose -- the same trick used to seed
        dual-arm particles. Returns None if either arm's IK fails, letting the caller skip the lift.
        """
        q = q_name_or_tensor if torch.is_tensor(q_name_or_tensor) else best_particle[q_name_or_tensor]
        state = world.kin_model.get_state(q.view(1, -1))
        lifted = torch.zeros_like(q.view(1, -1))
        for spec in arms:
            world_from_ee = state.link_pose[spec.ee_link].get_matrix().clone()  # clone: cuRobo reuses buffers
            world_from_ee[:, 2, 3] += dz
            res = world.ik_solvers[spec.name].solve_batch(Pose.from_matrix(world_from_ee))
            if not bool(res.success.flatten()[0]):
                return None
            lifted[:, spec.joint_slice] = res.solution[:, 0]
        return lifted[0]

    def back_off_configuration(q_name_or_tensor, specs, dist: float = 0.12):
        """``q`` with the given arms' end-effectors pulled back ``dist`` along their own approach axes.

        Serves both ends of a grasp. As a PRE-grasp it substitutes for the cartesian approach that
        ``solve_curobo`` gets from ``plan_single`` and joint-space planning cannot express: without it
        the hand may arrive at the grasp from any direction at all, and since the target object is
        necessarily disabled as an obstacle for that segment, a long object gets swept aside instead of
        gripped. As a RETRACT it withdraws the giver after a handover, whose open jaws still surround
        an object that now belongs to the taker -- lifting both arms cannot fix that, since they would
        rise together and stay overlapped.

        A shorter stand-off is tried when the full one is unreachable -- a mid-air handover pose sits
        near the edge of both arms' workspaces, where 12 cm back along the approach axis is often
        outside it while 4 cm is not, and a short approach is still far better than none. Returns None
        only if every distance fails, letting the caller fall back to planning straight to ``q``.
        """
        q0 = (q_name_or_tensor if torch.is_tensor(q_name_or_tensor) else best_particle[q_name_or_tensor])
        for scale in (1.0, 0.6, 0.35):
            q = q0.view(1, -1).clone()
            state = world.kin_model.get_state(q)
            ee_from_back = torch.eye(4, device=q.device, dtype=q.dtype)[None].clone()
            ee_from_back[:, 2, 3] = -dist * scale
            ok = True
            for spec in specs:
                world_from_ee = state.link_pose[spec.ee_link].get_matrix().clone()  # buffers are reused
                res = world.ik_solvers[spec.name].solve_batch(
                    Pose.from_matrix(world_from_ee @ ee_from_back)
                )
                if not bool(res.success.flatten()[0]):
                    ok = False
                    break
                q[:, spec.joint_slice] = res.solution[:, 0]
            if ok:
                return q[0]
        return None

    def approach_and_plan_to(q_name, specs, label: str, disable: tuple = (), dist: float = 0.10):
        """Plan to ``q_name`` via a pre-grasp waypoint, with ``disable``d obstacles only on the last hop.

        The objects being grasped (or the surface being placed on) have to be disabled to reach a
        configuration that is in contact with them by construction -- but disabling them for the WHOLE
        segment lets the arm travel through them. Splitting the segment confines that blindness to a
        short straight-in descent, while the long transit still sees every obstacle.
        """
        pre = back_off_configuration(q_name, specs, dist)
        if pre is not None:
            plan_to(pre, f"Approach({label})")
        for name in disable:
            motion_gen.world_coll_checker.enable_obstacle(enable=False, name=name)
        try:
            plan_to(q_name, label)
        finally:
            for name in disable:
                motion_gen.world_coll_checker.enable_obstacle(enable=True, name=name)

    def attach(obj: str, spec):
        """Attach one object to one hand, using cuTAMP's sampled spheres for that object."""
        obstacle = motion_gen.world_model.get_obstacle(obj)
        obstacle.old_get_bounding_spheres = obstacle.get_bounding_spheres

        def get_bounding_spheres(self, *args, **kwargs) -> List[Sphere]:
            spheres = world.get_collision_spheres(obj)
            pts = spheres[:, :3].cpu().numpy()
            n_radius = spheres[:, 3].cpu().numpy()
            obj_pose = Pose.from_matrix(obj_to_current_pose[obj])
            pre_transform_pose = kwargs["pre_transform_pose"]
            if pre_transform_pose is not None:
                obj_pose = pre_transform_pose.multiply(obj_pose)
            points_cuda = self.tensor_args.to_device(pts)
            pts = obj_pose.transform_points(points_cuda).cpu().view(-1, 3).numpy()
            return [
                Sphere(name=f"{self.name}_sph_{i}",
                       pose=[pts[i, 0], pts[i, 1], pts[i, 2], 1, 0, 0, 0], radius=n_radius[i])
                for i in range(pts.shape[0])
            ]

        obstacle.get_bounding_spheres = get_bounding_spheres.__get__(obstacle)
        with timer.time(f"{timeline}_planning"):
            motion_gen.attach_objects_to_robot(
                last_js,
                object_names=[obj],
                link_name=spec.attached_link,
                surface_sphere_radius=0.005,
                sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
                voxelize_method="subdivide",
            )
        obstacle.get_bounding_spheres = obstacle.old_get_bounding_spheres
        del obstacle.old_get_bounding_spheres

    def gripper_step(action: str, specs, label: str):
        """One gripper event for one OR MORE hands.

        ``specs`` is a list because a lockstep operator means the hands act at the same instant:
        ``PickBoth`` is one closure of two jaws, not two closures that happen to be adjacent. Emitting
        it as a single step is what lets a consumer actuate them together -- two consecutive per-hand
        steps are indistinguishable from a deliberate sequence, which is exactly what ``Handover``
        needs (its taker must close BEFORE its giver opens, or the object is unsupported), so the two
        cases have to be distinguishable in the plan itself rather than by a downstream heuristic.

        ``arm`` is kept alongside ``arms`` for the single-hand case so existing consumers still read.
        """
        if not isinstance(specs, (list, tuple)):
            specs = [specs]
        step = {"type": "gripper", "action": action, "arms": [s.name for s in specs], "label": label}
        if len(specs) == 1:
            step["arm"] = specs[0].name
        accum_plans.append(step)

    for idx, ground_op in enumerate(plan_skeleton):
        op_name = ground_op.operator.name
        _log.info(f"{idx + 1}. {ground_op.name}")

        # The Move operators only name the configuration the next action lands on; planning is done
        # by that action, exactly as in solve_curobo.
        if op_name in (MoveFree.name, MoveHoldingBoth.name,
                       MoveHoldingGiver.name, MoveHoldingTaker.name):
            continue

        elif op_name == PickBoth.name:
            obj_a, grasp_a, obj_b, grasp_b, q = ground_op.values
            # The grasp configuration is, by construction, in contact with the objects being
            # grasped: cuTAMP has already checked the gripper against them using their sampled
            # surface spheres, but cuRobo's world model treats them as SOLID obstacles, so leaving
            # them enabled makes the grasp itself read as a collision and motion planning into it
            # can never succeed. Disable exactly the two targets, and only for the final descent from
            # the pre-grasp; they come back as attached bodies immediately afterwards, and every other
            # object stays an obstacle throughout.
            approach_and_plan_to(q, arms, ground_op.name, disable=(obj_a, obj_b))
            # ONE closure of both jaws -- see gripper_step. The attachments are planner-side only, so
            # doing them after the event does not change what executes.
            gripper_step("close", arms, ground_op.name)
            for spec, obj in zip(arms, (obj_a, obj_b)):
                attach(obj, spec)
                motion_gen.world_coll_checker.enable_obstacle(enable=False, name=obj)
            # Lift before moving: the objects are now attached bodies still resting on the table.
            lifted = lift_configuration(q)
            if lifted is not None:
                plan_to(lifted, f"Retract({ground_op.name})")

        elif op_name == PlaceBoth.name:
            obj_a, grasp_a, place_a, surf_a, obj_b, grasp_b, place_b, surf_b, q = ground_op.values
            plan_to(q, ground_op.name)
            gripper_step("open", arms, ground_op.name)  # both jaws release together
            for spec, obj, placement in zip(arms, (obj_a, obj_b), (place_a, place_b)):
                motion_gen.detach_object_from_robot(spec.attached_link)
                # The object simply IS at its optimized placement now, so unlike solve_curobo there
                # is no need to recover its pose through the end-effector transform.
                obj_to_current_pose[obj] = action_4dof_to_mat4x4(best_particle[placement][None].clone())[0]
                motion_gen.world_coll_checker.enable_obstacle(enable=True, name=obj)
                motion_gen.world_collision.update_obstacle_pose(
                    obj, Pose.from_matrix(obj_to_current_pose[obj])
                )
                visualizer.log_mat4x4(f"world/{obj}", obj_to_current_pose[obj])
            # Lift clear of the objects just released before heading home.
            lifted = lift_configuration(q)
            if lifted is not None:
                plan_to(lifted, f"Retract({ground_op.name})")

        elif op_name == PickGiver.name:
            obj, grasp, q = ground_op.values
            giver = arms[0]
            approach_and_plan_to(q, [giver], ground_op.name, disable=(obj,))
            gripper_step("close", giver, ground_op.name)
            attach(obj, giver)
            motion_gen.world_coll_checker.enable_obstacle(enable=False, name=obj)
            lifted = lift_configuration(q)
            if lifted is not None:
                plan_to(lifted, f"Retract({ground_op.name})")

        elif op_name == Handover.name:
            obj, grasp_g, grasp_t, hand_pose, q = ground_op.values
            giver, taker = arms[0], arms[1]
            # The taker has to drive its jaws ONTO the object, which is an attached body on the giver
            # -- so to cuRobo the approach is a self-collision and no trajectory exists. cuTAMP has
            # already checked this configuration's gripper-vs-object clearance under its own sphere
            # model (CollisionFreeGrasp), so the attachment is hidden for this one solve, exactly as
            # the world obstacle is hidden for the pick.
            # The pre-grasp hop still sees the object on the giver's hand, so the taker cannot swipe
            # through it on the way in; only the short descent is blind to it.
            pre = back_off_configuration(q, [taker])
            if pre is not None:
                plan_to(pre, f"Approach({ground_op.name})")
            with temporarily_detached([giver]):
                plan_to(q, ground_op.name)
            # Both hands hold the object for an instant: the taker closes first, THEN the giver
            # releases, so the object is never unsupported. cuRobo can only model it as attached to
            # one link, so the handoff is a detach/attach pair between the two hands.
            gripper_step("close", taker, ground_op.name)
            obj_to_current_pose[obj] = action_4dof_to_mat4x4(best_particle[hand_pose][None].clone())[0]
            motion_gen.world_collision.update_obstacle_pose(obj, Pose.from_matrix(obj_to_current_pose[obj]))
            motion_gen.detach_object_from_robot(giver.attached_link)
            attach(obj, taker)
            gripper_step("open", giver, ground_op.name)
            visualizer.log_mat4x4(f"world/{obj}", obj_to_current_pose[obj])
            # Withdraw the giver before the taker carries the object anywhere, or every later segment
            # starts with the giver's fingers inside the taker's attached object.
            backed = back_off_configuration(q, [giver])
            if backed is not None:
                with temporarily_detached([taker]):
                    plan_to(backed, f"Retract({ground_op.name})")

        elif op_name == PlaceTaker.name:
            obj, grasp, placement, surface, q = ground_op.values
            taker = arms[1]
            # A placement RESTS ON the surface, so the held object's spheres touch the tray -- solid to
            # cuRobo, in contact by construction to cuTAMP, which already checked the placement is
            # stable and collision-free. Same treatment as the grasp target at pick time: the surface
            # is disabled for this one segment.
            approach_and_plan_to(q, [taker], ground_op.name, disable=(surface,))
            gripper_step("open", taker, ground_op.name)
            motion_gen.detach_object_from_robot(taker.attached_link)
            obj_to_current_pose[obj] = action_4dof_to_mat4x4(best_particle[placement][None].clone())[0]
            motion_gen.world_collision.update_obstacle_pose(obj, Pose.from_matrix(obj_to_current_pose[obj]))
            visualizer.log_mat4x4(f"world/{obj}", obj_to_current_pose[obj])
            # The jaws are still around the object, so it stays disabled until the arm has withdrawn;
            # re-enabling it first makes the retract's own start state read as a world collision.
            lifted = lift_configuration(q)
            if lifted is not None:
                plan_to(lifted, f"Retract({ground_op.name})")
            motion_gen.world_coll_checker.enable_obstacle(enable=True, name=obj)

        else:
            raise NotImplementedError(f"Unsupported operator for dual-arm motion planning: {op_name}")

    # Return home
    plan_to("q0", "GoToInitial(q0)")
    _log.info(f"Motion planning metrics: {timer.get_summary(f'{timeline}_planning')}")
    return accum_plans
