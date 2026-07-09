# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Core cuTAMP algorithm implementation."""

import dataclasses
import logging
import math
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import List, Union, Optional, Tuple
from unittest.mock import Mock

import torch

from curobo.geom.types import Cuboid
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.wrap.reacher.ik_solver import IKSolver
from curobo.wrap.reacher.motion_gen import MotionGen
from cutamp.config import TAMPConfiguration, validate_tamp_config
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_function import CostFunction
from cutamp.cost_reduction import CostReducer
from cutamp.envs.utils import TAMPEnvironment
from cutamp.experiment_logger import ExperimentLogger
from cutamp.motion_solver import solve_curobo, solve_curobo_batched, MotionPlanningError
from cutamp.optimize_plan import ParticleOptimizer
from cutamp.particle_initialization import ParticleInitializer
from cutamp.robots import get_q_home, load_robot_container
from cutamp.rollout import RolloutFunction
from cutamp.tamp_domain import all_tamp_operators, MoveFree, MoveHolding, Pick, Place
from cutamp.tamp_world import TAMPWorld, check_tamp_world_not_in_collision
from cutamp.task_planning import PlanSkeleton, task_plan_generator
from cutamp.utils.collision import get_batched_world_collision_cost
from cutamp.utils.common import (
    action_4dof_to_mat4x4,
    action_6dof_to_mat4x4,
    get_world_cfg,
    pose_list_to_mat4x4,
)
from cutamp.utils.obb import get_object_obb
from cutamp.utils.shapes import sample_greedy_surface_spheres
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import RerunVisualizer, MockVisualizer, Visualizer

_log = logging.getLogger(__name__)

# Optional in-process capture for the Phase-2 equivalence harness. No-op unless CUTAMP_PHASE2_CAPTURE
# is set; then it stashes the exact solve_curobo inputs + the serial reference plan so the harness can
# run solve_curobo_batched on identical inputs and compare. Kept in-process (GPU tensors, not pickled).
_PHASE2_CAPTURE = None


def heuristic_fn(
    plan_skeleton: PlanSkeleton, cost_dict: dict, constraint_checker: ConstraintChecker, verbose: bool = True
) -> float:
    """
    Get a single heuristic value for a cost dict corresponding to a rollout.

    We first compute the success rate of each constraint. If the constraint has zero success, we assign it a penalty
    of -num_particles. We then compute the mean success rate across all constraints, and use the failure rate as the
    heuristic (lower the better).
    """
    full_mask = constraint_checker.get_full_mask(cost_dict)
    successes = []
    num_particles = None
    for con_type, con_info in full_mask.items():
        for name, mask in con_info.items():
            if mask.ndim == 2:
                satisfying = mask.sum(0)
            else:
                satisfying = mask.sum()

            if num_particles is None:
                num_particles = mask.shape[0]
            else:
                assert num_particles == mask.shape[0]

            # replace zeros with -num_particles
            satisfying[satisfying == 0] = -num_particles
            successes.extend(satisfying.tolist())
            if verbose:
                _log.debug(f"{con_type} {name} {satisfying.tolist()}")
    success_mean = sum(successes) / len(successes)
    success_rate = success_mean / num_particles
    failure_rate = 1 - success_rate
    heuristic = 100 * failure_rate

    # We have a preference for shorter plans
    heuristic += len(plan_skeleton)
    return heuristic


def get_best_particle(
    plan_info: dict, config: TAMPConfiguration, constraint_checker: ConstraintChecker, cost_reducer: CostReducer
) -> dict:
    """Get the particle that satisfies the constraints and has the best soft cost."""
    particles, rollout_fn, cost_fn = plan_info["particles"], plan_info["rollout_fn"], plan_info["cost_fn"]
    with torch.no_grad():
        rollout = rollout_fn(particles)
        cost_dict = cost_fn(rollout)

    # Take the best particle that is satisfying and has the best soft cost
    satisfying_mask = constraint_checker.get_mask(cost_dict, verbose=False)
    if not satisfying_mask.any():
        raise RuntimeError("No satisfying particles found")

    soft_costs = cost_reducer.soft_costs(cost_dict)
    satisfying_costs = soft_costs[satisfying_mask]
    best_satisfying_idx = satisfying_costs.argmin()
    indices = torch.arange(config.num_particles, device=satisfying_costs.device)
    best_idx = indices[satisfying_mask][best_satisfying_idx]
    best_particle = {k: v[best_idx].detach().clone() for k, v in particles.items() if v is not None}
    return best_particle


def _visualize_best_particle(
    visualizer: Visualizer,
    rollout: dict,
    best_idx: int,
    world: TAMPWorld,
    config: TAMPConfiguration,
) -> None:
    """Visualize the rollout for the best ranked particle."""
    visualizer.set_time_sequence("rollout_best", 0)
    visualizer.set_joint_positions(world.q_init.tolist())
    for obj in world.movables:
        obj_pose = world.get_object_pose(obj).cpu()
        visualizer.log_mat4x4(f"world/{obj.name}", obj_pose)

    for ts in range(len(rollout["conf_params"])):
        visualizer.set_time_sequence("rollout_best", ts + 1)
        q = rollout["confs"][best_idx, ts]

        gripper_close = rollout["gripper_close"][ts]
        if config.robot == "ur5":
            gripper_joints = [0.4] if gripper_close else [0.0]
        elif config.robot == "panda":
            gripper_joints = [0.01, 0.01] if gripper_close else [0.04, 0.04]
        else:
            gripper_joints = []
        visualizer.set_joint_positions(q.tolist() + gripper_joints)

        ee_pose = Pose(
            position=rollout["ee_position"][best_idx, ts][None],
            quaternion=rollout["ee_quaternion"][best_idx, ts][None],
        )
        visualizer.log_mat4x4("rollout/ee_pose", ee_pose.get_matrix()[0].cpu())

        robot_spheres = rollout["robot_spheres"][best_idx, ts].cpu()
        visualizer.log_spheres("rollout/robot_spheres", robot_spheres)

        pose_ts = rollout["ts_to_pose_ts"][ts]
        for obj in world.movables:
            obj_pose = rollout["obj_to_pose"][obj.name][best_idx, pose_ts].cpu()
            visualizer.log_mat4x4(f"world/{obj.name}", obj_pose)


def get_ranked_satisfying_particles(
    plan_info: dict,
    config: TAMPConfiguration,
    constraint_checker: ConstraintChecker,
    cost_reducer: CostReducer,
    visualizer: Visualizer | None = None,
) -> dict[str, torch.Tensor]:
    """Get the satisfying particles ranked by grasp confidence (if available) or soft costs."""
    particles, rollout_fn, cost_fn = plan_info["particles"], plan_info["rollout_fn"], plan_info["cost_fn"]
    with torch.no_grad():
        rollout = rollout_fn(particles)
        cost_dict = cost_fn(rollout)

    # Get all satisfying particles
    satisfying_mask = constraint_checker.get_mask(cost_dict, verbose=False)
    if not satisfying_mask.any():
        raise RuntimeError("No satisfying particles found")

    soft_costs = cost_reducer.soft_costs(cost_dict)
    satisfying_costs = soft_costs[satisfying_mask]

    # Sum grasp confidences across all grasp parameters
    grasp_keys = [k for k in particles if k.startswith("grasp") and k.endswith("_confidences")]
    grasp_confs = None
    for grasp_key in grasp_keys:
        if (conf := particles[grasp_key]) is not None:
            grasp_confs = conf if grasp_confs is None else (grasp_confs + conf)
    satisfying_grasp_confs = grasp_confs[satisfying_mask] if grasp_confs is not None else None

    # Rank satisfying particles by grasp confidence (if available) or soft costs
    indices = torch.arange(config.num_particles, device=satisfying_costs.device)
    satisfying_idxs = indices[satisfying_mask]
    use_grasp_ranking = satisfying_grasp_confs is not None
    if use_grasp_ranking:
        sorted_idxs = satisfying_grasp_confs.argsort(descending=True)  # Higher confidence is better
    else:
        sorted_idxs = satisfying_costs.argsort()  # Lower cost is better
    ranked_idxs = satisfying_idxs[sorted_idxs]
    ranked_particles = {k: v[ranked_idxs].detach().clone() for k, v in particles.items() if v is not None}

    # Visualize the best particle
    if visualizer is not None:
        best_idx = ranked_idxs[0]
        _visualize_best_particle(visualizer, rollout, best_idx, rollout_fn.world, config)

    return ranked_particles


def sample_plan_skeleton(
    plan_gen,
    world: TAMPWorld,
    config: TAMPConfiguration,
    timer: TorchTimer,
    plan_count: int,
    constraint_checker: ConstraintChecker,
    cost_reducer: CostReducer,
    particle_initializer: ParticleInitializer,
) -> Tuple[Union[dict, None], bool]:
    """
    Try sampling a plan skeleton (if any remain), then its particles and compute the heuristic.
    Returns the plan_info dict and whether any satisfying particles were found upon initialization.
    """
    with timer.time("sample_task_plan"):
        plan_skeleton = next(plan_gen)
    if not plan_skeleton:
        return None, False

    plan_str = [op.name for op in plan_skeleton]
    _log.debug(f"[Plan {plan_count + 1}] Sampled plan {plan_str}")

    # Sample particles
    with timer.time("initialize_particles"):
        plan_particles = particle_initializer(plan_skeleton)
    if plan_particles is None:  # failed subgraph
        return None, False

    # Rollout particles and compute costs
    rollout_fn = RolloutFunction(plan_skeleton, world, config)
    cost_fn = CostFunction(plan_skeleton, world, config)
    with timer.time("measure_heuristic"), torch.no_grad():
        rollout = rollout_fn(plan_particles)
        cost_dict = cost_fn(rollout)
        heuristic = heuristic_fn(plan_skeleton, cost_dict, constraint_checker)

    # Number of satisfying particles
    with timer.time("get_satisfying_mask"):
        satisfying_mask = constraint_checker.get_mask(cost_dict)
    num_satisfying = satisfying_mask.sum().item()

    if config.stick_button_experiment and num_satisfying > 0:
        # Custom logic in stick button for breaking early for sampling baseline
        heuristic -= 100
        print(f"Found satisfying plan: {plan_str} heuristic -= 100")

    # Best cost initially
    with timer.time("compute_best_cost"):
        consider_types = {"constraint"}
        if config.optimize_soft_costs:
            consider_types.add("cost")
        costs = cost_reducer(cost_dict, consider_types=consider_types)
        if satisfying_mask.any():
            best_cost = costs[satisfying_mask].min().item()
            best_soft_cost = cost_reducer.soft_costs(cost_dict)[satisfying_mask].min().item()
        else:
            best_cost, best_soft_cost = float("inf"), float("inf")

    plan_info = {
        "idx": plan_count,
        "plan_skeleton": plan_skeleton,
        "particles": plan_particles,
        "rollout_fn": rollout_fn,
        "cost_fn": cost_fn,
        "heuristic": heuristic,
        "num_satisfying": num_satisfying,
        "best_cost": best_cost,
        "best_soft_cost": best_soft_cost,
    }

    _log.debug(
        f"[Plan {plan_count + 1}] {plan_info['num_satisfying']}/{config.num_particles} satisfying, "
        f"heuristic = {plan_info['heuristic']}"
    )
    return plan_info, num_satisfying > 0


def resample_plan_info(
    plan_info: dict,
    world: TAMPWorld,
    config: TAMPConfiguration,
    timer: TorchTimer,
    cost_reducer: CostReducer,
    constraint_checker: ConstraintChecker,
    particle_initializer: ParticleInitializer,
) -> int:
    """
    Sample particles again in-place for a plan info container with a plan skeleton. This can be used for rejection
    sampling strategy (for the sampling baseline), or for random restarts.

    Returns number of satisfying particles after re-sampling.
    """
    with timer.time("initialize_particles"), timer.time("resample_particles"):
        plan_particles = particle_initializer(plan_info["plan_skeleton"], verbose=False)

    # Rollout new particles and compute costs
    with timer.time("measure_heuristic"), torch.no_grad():
        rollout = plan_info["rollout_fn"](plan_particles)
        cost_dict = plan_info["cost_fn"](rollout)
        heuristic = heuristic_fn(plan_info["plan_skeleton"], cost_dict, constraint_checker, verbose=False)

    # Number of satisfying particles
    with timer.time("get_satisfying_mask"):
        satisfying_mask = constraint_checker.get_mask(cost_dict, verbose=False)
    num_satisfying = satisfying_mask.sum().item()

    # Best cost
    with timer.time("compute_best_cost"):
        consider_types = {"constraint"}
        if config.optimize_soft_costs:
            consider_types.add("cost")
        costs = cost_reducer(cost_dict, consider_types=consider_types)
        if satisfying_mask.any():
            best_cost = costs[satisfying_mask].min().item()  # note: should consider satisfying mask?
            soft_costs = cost_reducer.soft_costs(cost_dict)
            best_soft_cost = soft_costs[satisfying_mask].min().item()
            indices = torch.arange(config.num_particles, device=soft_costs.device)
            best_idx = indices[satisfying_mask][costs[satisfying_mask].argmin()]
            best_soft_idx = indices[satisfying_mask][soft_costs[satisfying_mask].argmin()]
        else:
            best_cost, best_soft_cost = float("inf"), float("inf")
            best_idx = None
            best_soft_idx = None

    # Update plan info
    plan_info["particles"] = plan_particles
    plan_info["heuristic"] = heuristic
    plan_info["num_satisfying"] = num_satisfying
    plan_info["best_cost"] = best_cost
    plan_info["best_soft_cost"] = best_soft_cost
    plan_info["rollout"] = rollout
    plan_info["best_idx"] = best_idx
    plan_info["best_soft_idx"] = best_soft_idx
    return num_satisfying


def setup_cutamp(
    env: TAMPEnvironment,
    config: TAMPConfiguration,
    q_init: Optional[List[float]] = None,
    experiment_id: Optional[str] = None,
    ik_solver: Optional[IKSolver] = None,
    experiment_dir: Optional[Path] = None,
):
    # Validate args and setup experiment logger
    validate_tamp_config(config)
    if experiment_id is None:
        # Microseconds + PID keep this unique across concurrently-running planners (e.g. multiple
        # tiptop servers sharing experiment_root). The old per-second id made two plans that started
        # in the same second share an experiment dir and collide ("File .../opt_0001.json already
        # exists") in ExperimentLogger.log_dict.
        experiment_id = f"{datetime.now().isoformat()}_{os.getpid()}"

    exp_logger = (
        ExperimentLogger(name=experiment_id, config=config, experiment_dir=experiment_dir)
        if config.enable_experiment_logging
        else Mock()
    )
    exp_logger.save_env(env)

    # Loading robot can be done offline, so doesn't count towards timing
    tensor_args = TensorDeviceType()
    robot_container = load_robot_container(config.robot, tensor_args)
    if q_init is None:
        q_init = get_q_home(config.robot)
    q_init = tensor_args.to_device(q_init)

    # Load TAMP world and warmup IK solver
    timer = TorchTimer()
    with timer.time("load_tamp_world", log_callback=_log.info):
        world = TAMPWorld(
            env,
            tensor_args,
            robot=robot_container,
            q_init=q_init,
            collision_activation_distance=config.world_activation_distance,
            coll_n_spheres=config.coll_n_spheres,
            coll_sphere_radius=config.coll_sphere_radius,
            ik_solver=ik_solver,
        )
        check_tamp_world_not_in_collision(world, movable_activation_dist=config.movable_activation_distance)

    if config.warmup_ik:
        with timer.time("warmup_ik_solver", log_callback=_log.info):
            world.warmup_ik_solver(config.num_particles)

    # Setup visualizer (doesn't count towards timing)
    visualizer = (
        RerunVisualizer(config, q_init, application_id=env.name, recording_id=experiment_id, spawn=config.rr_spawn)
        if config.enable_visualizer
        else MockVisualizer()
    )
    visualizer.log_tamp_world(world)
    return exp_logger, visualizer, timer, world


def _save_final_plan_graph(plan_skeleton, plan_info, world, config, constraint_checker, exp_logger, found_solution):
    """Build and save a factor-graph representation (JSON + DOT) of a plan skeleton.

    Optionally annotates the constraint/cost factors with concrete values by rolling out the
    skeleton's current particles and selecting a satisfying particle when one exists.
    """
    from cutamp.plan_graph import build_plan_graph, save_plan_graph

    exp_dir = getattr(exp_logger, "exp_dir", None)
    if not isinstance(exp_dir, Path):
        _log.warning("save_plan_graph requested but no experiment directory is available (logging disabled?)")
        return

    cost_dict = None
    particle_idx = None
    if plan_info is not None:
        try:
            with torch.no_grad():
                rollout = plan_info["rollout_fn"](plan_info["particles"])
                cost_dict = plan_info["cost_fn"](rollout)
                mask = constraint_checker.get_mask(cost_dict, verbose=False)
                if bool(mask.any()):
                    particle_idx = int(torch.nonzero(mask, as_tuple=False)[0].item())
        except Exception as e:
            _log.warning(f"Failed to compute cost dict for plan graph enrichment: {e}")
            cost_dict = None

    graph = build_plan_graph(
        plan_skeleton,
        name=world.env.name,
        env=world.env,
        initial_state=world.initial_state,
        goal_state=world.goal_state,
        cost_dict=cost_dict,
        constraint_checker=constraint_checker,
        particle_idx=particle_idx,
        solved=found_solution,
    )
    paths = save_plan_graph(graph, exp_dir)
    _log.info("Saved plan graph: " + ", ".join(f"{k}={v}" for k, v in paths.items()))


def run_cutamp(
    env: TAMPEnvironment,
    config: TAMPConfiguration,
    cost_reducer: CostReducer,
    constraint_checker: ConstraintChecker,
    q_init: Optional[List[float]] = None,
    experiment_id: Optional[str] = None,
    ik_solver: Optional[IKSolver] = None,
    grasps: Optional[dict] = None,
    motion_gen: Optional[MotionGen] = None,
    experiment_dir: Optional[Path] = None,
):
    """Overall cuTAMP algorithm implementation."""
    if config.m2t2_grasps and not grasps:
        _log.warning(f"M2T2 grasps enabled but no grasps provided! Falling back to grasp_dof={config.grasp_dof}")

    # Setup all the things and load the world
    exp_logger, visualizer, timer, world = setup_cutamp(env, config, q_init, experiment_id, ik_solver, experiment_dir)
    particle_initializer = ParticleInitializer(world, config, grasps)

    # Task plan generator
    _log.info(f"Initial State: {world.initial_state}")
    _log.info(f"Goal State: {world.goal_state}")
    with timer.time("get_plan_generator", log_callback=_log.info):
        plan_gen = task_plan_generator(
            world.initial_state,
            world.goal_state,
            operators=all_tamp_operators,
            explored_state_check=config.explored_state_check,
        )

    # Sample initial plans and particles
    found_solution_initially = False
    num_skipped_plans = 0
    with timer.time("sample_initial_plans", log_callback=_log.info):
        plan_queue: List[dict] = []
        plan_count = 0
        for idx in range(config.num_initial_plans):
            try:
                plan_info, has_solution = sample_plan_skeleton(
                    plan_gen, world, config, timer, idx, constraint_checker, cost_reducer, particle_initializer
                )
                if plan_info is None:
                    _log.debug("failed subgraph, skipping...")
                    num_skipped_plans += 1
                    continue
            except StopIteration:
                _log.info("Ran out of plans to sample")
                break
            plan_queue.append(plan_info)
            if has_solution:
                found_solution_initially = True
                break
            plan_count += 1

    # Sort plans by heuristic
    def sort_plans():
        with timer.time("sort_plans"):
            plan_queue.sort(key=lambda x: x["heuristic"])

    sort_plans()
    _log.info(f"Num plans: {len(plan_queue)}, num skipped: {num_skipped_plans}")

    curobo_plan = None
    failure_reason = None
    overall_metrics = {
        "num_optimized_plans": 0,
        "num_initial_plans": plan_count,
        "num_skipped_plans": num_skipped_plans,
        "num_satisfying_final": 0,
        "num_particles": config.num_particles,
        "best_cost": float("inf"),
        "best_soft_cost": float("inf"),
    }
    found_solution = False
    # Track the final skeleton (and its particle container) for the plan-graph export.
    final_plan_skeleton = None
    final_plan_info = None
    particle_optimizer = ParticleOptimizer(config, cost_reducer, constraint_checker)
    timer.start("first_solution")
    if found_solution_initially:
        found_solution = True
        timer.stop("first_solution")

    # Optimization loop for each skeleton and its particles
    timer.start("start_optimization")
    for idx, plan_info in enumerate(plan_queue):
        opt_iter = idx + 1
        should_break = False
        plan_skeleton = plan_info["plan_skeleton"]
        _log.info(f"[Opt {opt_iter}] Optimizing plan {[op.name for op in plan_skeleton]}")
        _log.info(
            f"[Opt {opt_iter}] plan idx = {plan_info['idx']}, heuristic = {plan_info['heuristic']:.2f}, "
            f"num satisfying = {plan_info['num_satisfying']}"
        )
        best_particle = None

        if config.approach == "optimization":
            has_satisfying, metrics, time_exceeded = particle_optimizer(plan_info, timer, visualizer)

            # For the sake of printing out debug info
            with torch.no_grad():
                rollout = plan_info["rollout_fn"](plan_info["particles"])
                cost_dict = plan_info["cost_fn"](rollout)
                _ = heuristic_fn(plan_skeleton, cost_dict, constraint_checker, verbose=True)

            if metrics["best_cost"] is not None:
                overall_metrics["best_cost"] = min(overall_metrics["best_cost"], metrics["best_cost"])
            if metrics["best_soft_cost"] is not None:
                overall_metrics["best_soft_cost"] = min(overall_metrics["best_soft_cost"], metrics["best_soft_cost"])
            if time_exceeded:
                _log.info(f"Max loop duration reached, stopping optimization")
                should_break = True
            exp_logger.log_dict(f"optimization/opt_{opt_iter:04d}", metrics)
            if has_satisfying:
                best_particle = get_best_particle(plan_info, config, constraint_checker, cost_reducer)
        else:
            # This is the parallelized sampling baseline
            assert config.approach == "sampling"
            num_resample_attempts = 0
            resample_dur = 0.0
            has_satisfying = plan_info["num_satisfying"] > 0
            total_num_satisfying = plan_info["num_satisfying"]
            best_particle = None
            best_soft_costs = []
            elapsed = []

            if not has_satisfying or not config.break_on_satisfying:
                timer.start("resample_duration")
                for resample_idx in range(config.num_resampling_attempts):
                    if config.max_loop_dur is not None and timer.elapsed("start_optimization") >= config.max_loop_dur:
                        _log.info(f"Max loop duration reached, stopping resampling")
                        should_break = True
                        break
                    timer.start("resample_plan_info")
                    num_satisfying = resample_plan_info(
                        plan_info,
                        world,
                        config,
                        timer,
                        cost_reducer,
                        constraint_checker,
                        particle_initializer,
                    )
                    total_num_satisfying += num_satisfying
                    if plan_info["best_soft_cost"] < overall_metrics["best_soft_cost"]:
                        best_soft_idx = plan_info["best_soft_idx"]
                        best_particle = {
                            k: v[best_soft_idx].detach().clone() for k, v in plan_info["particles"].items()
                        }

                    overall_metrics["best_cost"] = min(overall_metrics["best_cost"], plan_info["best_cost"])
                    overall_metrics["best_soft_cost"] = min(
                        overall_metrics["best_soft_cost"], plan_info["best_soft_cost"]
                    )

                    # Keep track of the best soft cost since start of resampling
                    best_soft_costs.append(overall_metrics["best_soft_cost"])
                    elapsed.append(timer.elapsed("start_optimization"))

                    resample_plan_info_dur = timer.stop("resample_plan_info")
                    _log.debug(
                        f"[Plan {plan_info['idx'] + 1}] Resample attempt {resample_idx + 1}/{config.num_resampling_attempts}, "
                        f"{num_satisfying}/{config.num_particles} satisfying particles. Total satisfying {total_num_satisfying}. "
                        f"Took {resample_plan_info_dur:.2f}s"
                    )
                    has_satisfying = num_satisfying > 0
                    num_resample_attempts += 1

                    # Visualize best particle rollout state
                    rollout = plan_info["rollout"]
                    best_soft_idx = plan_info["best_soft_idx"]
                    if best_soft_idx is None:
                        best_soft_idx = 0
                    visualizer.set_time_sequence(f"samp", num_resample_attempts)
                    q_last = rollout["confs"][best_soft_idx, -1].tolist()
                    visualizer.set_joint_positions(q_last)
                    for obj in rollout["obj_to_pose"]:
                        mat4x4_last = rollout["obj_to_pose"][obj][best_soft_idx, -1]
                        visualizer.log_mat4x4(f"world/{obj}", mat4x4_last)

                    if has_satisfying:
                        if timer.has_timer("first_solution"):
                            time_to_first_sol = timer.stop("first_solution")
                            _log.info(f"Found first solution in {time_to_first_sol:.2f}s after sampling plans")
                        if config.break_on_satisfying:
                            should_break = True
                            break
                resample_dur = timer.stop("resample_duration")
                _log.info(f"Total resample duration: {resample_dur:.2f}s")
            else:
                _log.info("Already has satisfying particles, skipping resampling")
                overall_metrics["best_cost"] = min(overall_metrics["best_cost"], plan_info["best_cost"])
                overall_metrics["best_soft_cost"] = min(overall_metrics["best_soft_cost"], plan_info["best_soft_cost"])
                if config.break_on_satisfying:
                    should_break = True

            metrics = {
                "plan_skeleton": [str(op) for op in plan_skeleton],
                "num_particles": config.num_particles,
                "num_resample_attempts": num_resample_attempts,
                "resample_duration": resample_dur,
                "num_satisfying_final": total_num_satisfying,
                "total_num_particles": config.num_particles * (num_resample_attempts + 1),
                "best_cost": overall_metrics["best_cost"],
                "best_soft_cost": overall_metrics["best_soft_cost"],
                "best_soft_costs": best_soft_costs,
                "elapsed": elapsed,
            }
            exp_logger.log_dict(f"sampling/samp_{opt_iter:04d}", metrics)
            has_satisfying = total_num_satisfying > 0
            overall_metrics["num_satisfying_final"] = total_num_satisfying

            # Log best particle as last
            if best_particle is not None:
                rollout = plan_info["rollout_fn"]({k: v[None] for k, v in best_particle.items()})
                visualizer.set_time_sequence(f"samp", num_resample_attempts)
                q_last = rollout["confs"][0, -1].tolist()
                visualizer.set_joint_positions(q_last)

                for obj in rollout["obj_to_pose"]:
                    mat4x4_last = rollout["obj_to_pose"][obj][0, -1]
                    visualizer.log_mat4x4(f"world/{obj}", mat4x4_last)

        # Now we've either optimized or resampled
        overall_metrics["num_optimized_plans"] += 1
        if has_satisfying:
            found_solution = True
            ranked_particles = get_ranked_satisfying_particles(
                plan_info, config, constraint_checker, cost_reducer, visualizer
            )
            if config.curobo_plan:
                # Need to cache initial pose as cuRobo dynamically updates during planning which sucks ass
                obj_to_initial_pose = {obj.name: world.get_object_pose(obj) for obj in world.movables}

                # IMPORTANT: this line comes after the previous as it messes with cuRobo internal memory,
                # also call it only once!
                if motion_gen is not None:
                    all_world_cfg = get_world_cfg(world.env, include_movables=True)
                    motion_gen.update_world(all_world_cfg)
                    _log.info(f"Updated motion gen with world cfg")

                num_satisfying = ranked_particles["q0"].shape[0]
                max_attempts = min(config.max_motion_refine_attempts or num_satisfying, num_satisfying)
                for curr_idx in range(max_attempts):
                    _log.info(f"Trying cuRobo planning with satisfying particle {curr_idx + 1}/{max_attempts} ({num_satisfying} total satisfying)")
                    curr_particle = {k: v[curr_idx] for k, v in ranked_particles.items()}
                    try:
                        curobo_plan = solve_curobo(
                            plan_info,
                            curr_particle,
                            world,
                            config,
                            timer,
                            visualizer,
                            obj_to_initial_pose=obj_to_initial_pose,
                            timeline=f"curobo_{curr_idx}",
                            motion_gen=motion_gen,
                        )
                        _log.info("Successful plan found!")
                        failure_reason = None
                        if os.environ.get("CUTAMP_PHASE2_CAPTURE"):
                            global _PHASE2_CAPTURE
                            _PHASE2_CAPTURE = {
                                "plan_info": plan_info,
                                "best_particle": curr_particle,
                                "world": world,
                                "obj_to_initial_pose": obj_to_initial_pose,
                                "serial_plan": curobo_plan,
                            }
                        break
                    except MotionPlanningError as e:
                        _log.warning(f"Failed to motion plan: {e}")
                else:
                    # All attempted particles failed motion planning
                    if curobo_plan is None:
                        max_reached = " (max attempts reached)" if max_attempts < num_satisfying else ""
                        failure_reason = (
                            f"Motion planning failed for {max_attempts}/{num_satisfying} satisfying particle(s){max_reached}"
                        )

            overall_metrics["num_satisfying_final"] = metrics["num_satisfying_final"]
            overall_metrics["final_plan_skeleton"] = [str(op) for op in plan_skeleton]
            final_plan_skeleton = plan_skeleton
            final_plan_info = plan_info
            _log.debug(f"Total num satisfying {metrics['num_satisfying_final']}")
            if config.curobo_plan and curobo_plan is None:
                # Motion refinement failed, try next skeleton. Intentionally overrides should_break
                # set by break_on_satisfying during resampling — we don't want to stop on a skeleton
                # where motion planning failed. The max_loop_dur timeout will still be checked at the
                # start of the next skeleton's resampling loop.
                _log.info(f"Motion refinement failed for skeleton {[op.name for op in plan_skeleton]}, trying next")
                should_break = False
            elif config.break_on_satisfying:
                should_break = True

        if should_break:
            break

        # TODO: complete version of our algorithm that adds additional skeletons to the queue, resorts, revisits
        #  skeletons, etc.
        # new_plan_info = sample_plan_skeleton()
        # if new_plan_info is not None:
        #     plan_queue.append(new_plan_info)
        #     sort_plans()

    opt_elapsed = timer.stop("start_optimization")
    _log.debug(f"Optimization loop took roughly {opt_elapsed:.2f}s")
    if found_solution and config.curobo_plan and curobo_plan is None:
        found_solution = False
        if failure_reason is None:
            failure_reason = "Motion planning failed for all skeletons with satisfying particles"
    if not found_solution:
        if len(plan_queue) == 0:
            if num_skipped_plans > 0:
                failure_reason = f"All {num_skipped_plans} plan skeleton(s) failed particle initialization"
            else:
                failure_reason = "No valid plan skeletons found for the given goal"
        elif failure_reason is None:
            # Had plans but no satisfying particles (or timed out)
            optimized = overall_metrics["num_optimized_plans"]
            total = len(plan_queue)
            if optimized < total:
                failure_reason = (
                    f"No satisfying particles found after optimizing "
                    f"{optimized}/{total} plan(s) (time budget {config.max_loop_dur}s exceeded)"
                )
            else:
                failure_reason = (
                    f"No satisfying particles found after optimizing all {total} plan(s)"
                )
        _log.warning(failure_reason)
    _log.debug(f"Best cost: {overall_metrics['best_cost']:.4f}, soft cost: {overall_metrics['best_soft_cost']:.4f}")

    # Save a factor-graph representation of the final plan skeleton, if requested. Falls back to
    # the best-heuristic skeleton (marked unsolved) when no satisfying skeleton was found.
    if config.save_plan_graph:
        graph_skeleton = final_plan_skeleton
        graph_plan_info = final_plan_info
        if graph_skeleton is None and plan_queue:
            graph_plan_info = plan_queue[0]
            graph_skeleton = graph_plan_info["plan_skeleton"]
        if graph_skeleton is not None:
            try:
                _save_final_plan_graph(
                    graph_skeleton, graph_plan_info, world, config, constraint_checker, exp_logger, found_solution
                )
            except Exception as e:
                _log.warning(f"Failed to save plan graph: {e}")
        else:
            _log.warning("save_plan_graph requested but no plan skeleton was available")

    # Dump metrics out
    overall_metrics["found_solution"] = found_solution
    exp_logger.log_dict("overall_metrics", overall_metrics)
    exp_logger.log_dict("timer_metrics", timer.get_summaries())

    # Log constraint and cost multipliers
    exp_logger.log_dict("multipliers", cost_reducer.cost_config)
    exp_logger.log_dict("tolerances", constraint_checker.constraint_config)
    return curobo_plan, overall_metrics["num_satisfying_final"], failure_reason


def _rename_world_movables(world, role_map: dict) -> None:
    """Rename movable objects in a TAMPWorld in place: obstacle.name + every name-keyed index.

    role_map: {old_name: new_role}. Obstacle.name is mutable (curobo Obstacle is a non-frozen dataclass),
    and world.env.movables / .statics / .type_to_objects all hold the SAME Obstacle references, so setting
    obj.name propagates to them + to any WorldConfig built AFTER this. Only TAMPWorld's snapshot dicts
    (built in __init__ from the old names) must be rebuilt. Solvers built before this keep stale names, so
    the caller must build the batched motion_gen AFTER canonicalizing (get_batched_motion_gen reads live
    env names)."""
    for obj in world.env.movables:
        if obj.name in role_map:
            obj.name = role_map[obj.name]
    world._name_to_obj = {obj.name: obj for obj in world.env.movables + world.env.statics}
    world._movable_names = {obj.name for obj in world.env.movables}
    world._obj_to_spheres = {role_map.get(k, k): v for k, v in world._obj_to_spheres.items()}
    world._obj_to_aabb = {role_map.get(k, k): v for k, v in world._obj_to_aabb.items()}


def _canonicalize_scene_by_pick_order(scene: dict) -> None:
    """Relabel a scene's movables to positional roles (toy0, toy1, ...) so structurally-identical scenes
    with different per-perception object NAMES collapse into ONE batchable group (the only name-dependent
    part of the grouping _sig is the movable name set).

    Roles are assigned BY PICK ORDER (first Pick -> toy0, ...). This is what makes correctness hold when
    _solve_group drives every scene in a group with the group REPRESENTATIVE's single skeleton: after this
    relabel every scene's skeleton is byte-identical (Pick(toy0),Place(toy0),Pick(toy1),...), and the
    particle<->role binding is consistent by construction (the planner fabricates grasp1/pose1/q1 for the
    first-picked object, which is now toy0, in every scene). Movables never picked (extra clutter) get the
    remaining roles by sorted name so initial_poses keys stay uniform across identical scenes. Surfaces are
    left as-is (assumed uniform across scene-6, and renaming one would need matching
    constraint_to_tol[StablePlacement][{surface}_in_xy/_support] keys). Mutates scene in place.
    """
    skeleton = scene["skeleton"]
    ordered: list = []
    for op in skeleton:
        if op.operator.name == "Pick":
            obj = op.values[0]  # Pick.values = (obj, grasp, q)
            if obj not in ordered:
                ordered.append(obj)
    # any movable not on the pick path -> stable role suffix by sorted name (keeps initial_poses uniform)
    movable_names = [obj.name for obj in scene["world"].env.movables]
    for name in sorted(movable_names):
        if name not in ordered:
            ordered.append(name)
    role_map = {name: f"toy{i}" for i, name in enumerate(ordered)}
    if all(name == role for name, role in role_map.items()):
        return  # already canonical (idempotent)

    _rename_world_movables(scene["world"], role_map)
    new_skeleton = []
    for op in skeleton:
        subs = {p.name: role_map.get(v, v) for p, v in zip(op.operator.parameters, op.values)}
        new_skeleton.append(op.operator.ground(subs))
    scene["skeleton"] = new_skeleton
    scene["initial_poses"] = {role_map.get(k, k): v for k, v in scene["initial_poses"].items()}
    # scene["ops"] (operator names) + scene["particles"] (fabricated keys grasp1/pose1/q1) are already
    # name-independent, so they are unchanged.


def _rename_world_surface(world, old: str, new: str) -> None:
    """Rename a static (placement surface) in place: Obstacle.name + name-keyed indexes. Mirrors
    _rename_world_movables for statics (surfaces are never in _movable_names/_obj_to_spheres)."""
    for obj in world.env.statics:
        if obj.name == old:
            obj.name = new
    world._name_to_obj = {obj.name: obj for obj in world.env.movables + world.env.statics}
    if old in world._obj_to_aabb:
        world._obj_to_aabb[new] = world._obj_to_aabb.pop(old)


def _canonicalize_scene_surface(scene: dict, canon: str) -> None:
    """Rename the scene's (single) placement surface to the canonical name so scenes whose perception
    labeled the same physical surface differently (plate vs white_plate) share one group signature.

    Only applies when the scene has exactly ONE distinct Place surface (scene-6 always does) and the
    canonical name is free; otherwise the scene is left alone and simply groups separately. The
    constraint tolerances f"{canon}_in_xy"/"_support" must exist wherever the ConstraintChecker is
    built -- the tiptop batch coordinator aliases its loosened per-surface tolerances under the
    canonical name for this reason."""
    surfaces = {op.values[3] for op in scene["skeleton"] if op.operator.name == "Place"}
    if len(surfaces) != 1:
        return
    old = next(iter(surfaces))
    if old == canon:
        return
    world = scene["world"]
    if world.has_object(canon):
        _log.warning(f"surface-canon: world already has an object named {canon!r}; leaving {old!r} as-is")
        return
    _rename_world_surface(world, old, canon)
    sub = {old: canon}
    scene["skeleton"] = [
        op.operator.ground({p.name: sub.get(v, v) for p, v in zip(op.operator.parameters, op.values)})
        for op in scene["skeleton"]
    ]


def _normalize_scene_grasps(scene: dict, config) -> None:
    """Normalize every grasp particle block to FIXED [pps,4,4] matrices + a confidences tensor.

    M2T2 scenes already carry 4x4 grasp blocks; scenes (or single objects) that fell back to the
    4/6-DOF heuristic sampler carry [pps,4]/[pps,6] blocks, which shatters the group signature.
    Converting is LOSSLESS for optimization: ParticleOptimizer only steps {Pose, Conf} types, so grasp
    blocks are frozen after initialization in either representation. Missing confidences become ones
    (a constant, so confidence ranking degenerates to sample order only for scenes with no M2T2
    grasps at all)."""
    parts = scene["particles"]
    for key in list(parts):
        if key.startswith("grasp") and not key.endswith("_confidences"):
            g = parts[key]
            if g is not None and g.dim() == 2:
                parts[key] = action_4dof_to_mat4x4(g) if g.shape[1] == 4 else action_6dof_to_mat4x4(g)
            ck = f"{key}_confidences"
            if parts.get(ck) is None:
                parts[ck] = torch.ones(parts[key].shape[0], device=parts[key].device)


_CYCLE_OPS = ("MoveFree", "Pick", "MoveHolding", "Place")


def _pad_scene_to_canonical(scene: dict, config, target_count: int) -> None:
    """Pad a scene to the canonical toy count with PHANTOM pick-place cycles so every scene-6 scene
    shares ONE batchable signature (single group), closing the toy-count fragmentation axis.

    A scene with k < target_count real toys gets target_count - k phantoms:
      - a small template Cuboid per phantom, parked at a free spot near the toys and registered in the
        world like a real movable (env.movables + name/sphere/pose indexes). It participates in the
        optimizer and the refinement worlds as ordinary geometry (tiny and out of the way).
      - a MoveFree/Pick/MoveHolding/Place cycle appended to the skeleton with the task planner's
        positional naming (q{2j+1}/q{2j+2}, traj{2j+1}/{2j+2}, grasp{j+1}, pose{j+1}), making the
        padded skeleton structurally identical to a natural target_count-toy canonical skeleton.
      - fabricated particle blocks: grasp = clone of grasp1 (fixed 4x4s, never optimized), placement
        initialized on the scene's OWN surface, q's seeded from q0 (Adam refines them).
    The pre-padding skeleton is kept as scene["skeleton_rank"]: per-scene satisfying-particle ranking
    uses ONLY the real ops, so phantom constraints never gate a scene's yield. Refinement masks
    phantom ops per env (solve_curobo_batched phantom_objs_per_env), so emitted plans contain no
    phantom motions. Requires _normalize_scene_grasps to have run first (grasp1 is 4x4 + confidences).
    """
    skeleton = scene["skeleton"]
    scene.setdefault("phantoms", set())
    scene["skeleton_rank"] = list(skeleton)
    picked: list = []
    for op in skeleton:
        if op.operator.name == "Pick" and op.values[0] not in picked:
            picked.append(op.values[0])
    k = len(picked)
    if k == 0 or k >= target_count:
        return
    ops = tuple(op.operator.name for op in skeleton)
    surfaces = {op.values[3] for op in skeleton if op.operator.name == "Place"}
    if ops != _CYCLE_OPS * k or len(surfaces) != 1:
        _log.warning(f"pad-to-canonical: unexpected skeleton pattern {ops} / surfaces {surfaces}; not padding")
        return
    surf = next(iter(surfaces))
    world, parts, pps = scene["world"], scene["particles"], config.num_particles
    device = parts["q0"].device

    # Free spots for the phantom cuboids: fixed offsets around the first-picked toy, away from every
    # real movable and the surface center (this is soft-cost geometry only -- phantom motions are
    # masked out of the emitted plans).
    anchor = world.get_object_pose(picked[0])[:3, 3]
    surf_aabb = world.get_aabb(surf)
    surf_top = float(surf_aabb[1, 2])
    z_rest = float(anchor[2])
    taken = torch.stack([world.get_object_pose(m)[:2, 3] for m in world.movables] + [surf_aabb.mean(0)[:2]])
    offsets = [(-0.16, -0.12), (-0.16, 0.12), (0.18, -0.12), (0.18, 0.12), (-0.24, 0.0), (0.24, 0.0),
               (0.0, -0.24), (0.0, 0.24)]
    for j in range(k, target_count):
        name = f"toy{j}"
        spot = None
        for ox, oy in offsets:
            cand = torch.tensor([float(anchor[0]) + ox, float(anchor[1]) + oy], device=taken.device)
            if torch.linalg.norm(taken - cand, dim=-1).min() >= 0.11:
                spot = cand
                break
        if spot is None:
            spot = torch.tensor([float(anchor[0]), float(anchor[1]) - 0.30 * (j - k + 1)], device=taken.device)
        taken = torch.cat([taken, spot[None]])

        cub = Cuboid(name=name, pose=[float(spot[0]), float(spot[1]), z_rest, 1.0, 0.0, 0.0, 0.0],
                     dims=[0.04, 0.04, 0.04])
        world.env.movables.append(cub)
        world._movable_names.add(name)
        world._name_to_obj[name] = cub
        world._obj_to_spheres[name] = sample_greedy_surface_spheres(
            cub, n_spheres=config.coll_n_spheres, sphere_radius=config.coll_sphere_radius).to(device)
        scene["initial_poses"][name] = pose_list_to_mat4x4(list(cub.pose)).to(device)
        scene["phantoms"].add(name)

        i1 = j + 1
        qa, qb = f"q{2 * j + 1}", f"q{2 * j + 2}"
        skeleton.append(MoveFree.ground({"q_start": f"q{2 * j}", "traj": f"traj{2 * j + 1}", "q_end": qa}))
        skeleton.append(Pick.ground({"obj": name, "grasp": f"grasp{i1}", "q": qa}))
        skeleton.append(MoveHolding.ground(
            {"obj": name, "grasp": f"grasp{i1}", "q_start": qa, "traj": f"traj{2 * j + 2}", "q_end": qb}))
        skeleton.append(Place.ground(
            {"obj": name, "grasp": f"grasp{i1}", "placement": f"pose{i1}", "surface": surf, "q": qb}))

        parts[f"grasp{i1}"] = parts["grasp1"].clone()
        parts[f"grasp{i1}_confidences"] = parts["grasp1_confidences"].clone()
        pose4 = torch.empty(pps, 4, device=device)
        surf_center = surf_aabb.mean(0)
        pose4[:, 0] = surf_center[0] + (torch.rand(pps, device=device) - 0.5) * 0.10
        pose4[:, 1] = surf_center[1] + (torch.rand(pps, device=device) - 0.5) * 0.10
        pose4[:, 2] = surf_top + 0.02 + world.collision_activation_distance + 2e-3
        pose4[:, 3] = (torch.rand(pps, device=device) - 0.5) * (2.0 * math.pi)
        parts[f"pose{i1}"] = pose4
        parts[qa] = parts["q0"].clone()
        parts[qb] = parts["q0"].clone()

    scene["ops"] = tuple(op.operator.name for op in skeleton)


def _batched_object_spheres(group: list, rep) -> dict:
    """Per-scene collision spheres for each canonical movable, {name: [S, n_max, 4]}. Sphere counts
    can differ per scene (MultiSphere toys keep their perception sphere count), so shorter sets are
    padded by repeating their first sphere -- duplicates only inflate summed violation terms
    (conservative), never relax them. Gives the batched optimizer each scene's TRUE object geometry;
    without it, costs evaluated every scene with the REP's toy shapes (same leakage class as the
    surface targets, and the residual reason non-rep scenes under-satisfied)."""
    out = {}
    for obj in rep.movables:
        per = [sc["world"].get_collision_spheres(obj.name) for sc in group]
        n_max = max(p.shape[0] for p in per)
        per = [p if p.shape[0] == n_max else torch.cat([p, p[:1].expand(n_max - p.shape[0], -1)]) for p in per]
        out[obj.name] = torch.stack(per).to(rep.device)
    return out


def _batched_surface_targets(group: list, skel, config, rep) -> dict:
    """Per-scene placement-surface geometry for the batched CostFunction. Without this, the rep
    world's cached surface AABB/OBB/target_z silently pulls every scene's placements toward the REP's
    surface pose (the scene-6 plate is randomized per scene; measured on real data: non-rep group
    members got ~0 satisfying particles, e.g. [87, 0, 0])."""
    surfaces = {op.values[3] for op in skel if op.operator.name == "Place"}
    out = {}
    for surf in surfaces:
        if config.placement_check == "aabb":
            aabbs = torch.stack([sc["world"].get_aabb(surf) for sc in group]).to(rep.device)  # [S,2,3]
            out[surf] = {
                "xy_lower": aabbs[:, 0, :2].contiguous(),
                "xy_upper": aabbs[:, 1, :2].contiguous(),
                "target_z": aabbs[:, 1, 2] + rep.collision_activation_distance + 2e-3,
            }
        else:
            obbs = [get_object_obb(sc["world"].get_object(surf), shrink_dist=config.placement_shrink_dist)
                    for sc in group]
            out[surf] = {
                "center": torch.stack([torch.as_tensor(o.center) for o in obbs]).to(rep.device),
                "rot_inv": torch.stack([torch.as_tensor(o.rot_matrix_inv) for o in obbs]).to(rep.device),
                "half_xy": torch.stack([torch.as_tensor(o.half_extents[:2]) for o in obbs]).to(rep.device),
                "target_z": torch.stack([torch.as_tensor(o.surface_z) for o in obbs]).to(rep.device)
                + rep.collision_activation_distance + 2e-3,
            }
    return out


# Cache of batched MotionGens keyed by (n_envs, group signature), REUSED across batches as a pure
# THROUGHPUT optimization: a fresh get_batched_motion_gen per group costs seconds of solver build +
# batched warmup, which canonicalized recurring (n_envs, sig) groups can skip entirely. (The CUDA
# illegal-memory-access this cache was originally built to dodge was NOT build-count related: it was a
# warp-mesh use-after-free in curobo's WorldMeshCollision -- the fork keyed its warp cache by mesh NAME
# only, so batched builds loading the same names per env freed earlier envs' meshes while their ids were
# still referenced by the collision tensors. Fixed in curobo world_mesh.py by keying on (env_idx, name);
# fresh per-group builds are safe again.) LRU-bounded so the resident solver set fits GPU memory
# alongside Isaac/M2T2. Module-level -> persists for the tiptop server process lifetime.
_BATCHED_MG_CACHE: "OrderedDict" = OrderedDict()
_BATCHED_MG_CACHE_MAX = int(os.environ.get("CUTAMP_MG_CACHE", "12"))


def _update_batched_world(mg, world_cfgs: list) -> None:
    """Reload each env's obstacles into a REUSED batched MotionGen via load_collision_model(env_idx=i).

    A full per-env RELOAD (not pose/dim rewrites) is required for correctness: toys are MESHES whose
    per-scene geometry differs (the canonical 'toy0' is a physically different toy in every scene), so
    the warp meshes must be re-uploaded, not just re-posed. Safe with the env-keyed warp-mesh cache fix:
    reloading env i frees only env i's replaced meshes. The same call rewrites cuboid dims/poses and
    enable masks, and disables slots past the new counts. Buffer capacities are fixed at build time
    (obb=48/mesh=24) and scene-6 worlds stay far below them, so reuse never reallocates."""
    wc = mg.world_coll_checker
    for i, wcfg in enumerate(world_cfgs):
        # Zero this env's rows first so a world with 0 cuboids or 0 meshes cannot leave stale obstacles
        # enabled (load_collision_model early-outs on empty categories instead of clearing them).
        if wc._cube_tensor_list is not None:
            wc._cube_tensor_list[2][i, :] = 0
            wc._env_n_obbs[i] = 0
        if wc._mesh_tensor_list is not None:
            wc._mesh_tensor_list[2][i, :] = 0
            wc._env_n_mesh[i] = 0
        wc.load_collision_model(wcfg, env_idx=i)
        # Drop warp meshes this env no longer uses; their tensor slots were just disabled/overwritten,
        # so freeing them cannot dangle. Bounds cache growth across group swaps.
        names = {m.name for m in wcfg.mesh}
        for key in [k for k in wc._wp_mesh_cache if k[0] == i and k[1] not in names]:
            del wc._wp_mesh_cache[key]


def _get_cached_batched_mg(rep, n_envs: int, sig, world_cfgs: list, config):
    """Return a batched MotionGen for (n_envs, sig): build+cache on miss, REUSE on hit with a full
    per-env world reload. Reuse skips the per-group solver build + batched warmup (seconds per group).
    LRU-evict (with free + empty_cache) when the cache is full."""
    key = (n_envs, sig)
    cached = _BATCHED_MG_CACHE.get(key)
    if cached is not None:
        _BATCHED_MG_CACHE.move_to_end(key)
        try:
            cached.detach_object_from_robot("attached_object")  # clear any shared blob a prior solve left
        except Exception:
            pass
        _update_batched_world(cached, world_cfgs)
        _log.info(f"  [mg-cache] REUSE (n_envs={n_envs}); cache={len(_BATCHED_MG_CACHE)}/{_BATCHED_MG_CACHE_MAX}")
        return cached
    while len(_BATCHED_MG_CACHE) >= _BATCHED_MG_CACHE_MAX:
        _, old = _BATCHED_MG_CACHE.popitem(last=False)
        del old
        torch.cuda.empty_cache()
    mg = rep.get_batched_motion_gen(
        n_envs=n_envs, collision_activation_distance=config.world_activation_distance,
        world_cfgs=world_cfgs, num_batch_trajopt_seeds=1)
    _BATCHED_MG_CACHE[key] = mg
    _log.info(f"  [mg-cache] BUILD (n_envs={n_envs}); cache={len(_BATCHED_MG_CACHE)}/{_BATCHED_MG_CACHE_MAX}")
    return mg


def run_cutamp_batched(
    envs: List[TAMPEnvironment],
    config: TAMPConfiguration,
    cost_reducer: CostReducer,
    constraint_checker: ConstraintChecker,
    q_inits: Optional[list] = None,
    grasps_list: Optional[list] = None,
    motion_gen_batched: Optional[MotionGen] = None,
) -> List[Optional[list]]:
    """Batched cuTAMP (Phase 2 + 2b): plan for N scenes together.

    Scenes are GROUPED by plan skeleton (real perception yields varying detected object sets -> varying
    skeletons), and each homogeneous group is solved with ONE batched particle optimization
    ([g*num_particles] particles sharing an n_envs=g collision checker via env_query_idx) followed by a
    batched cuRobo refinement that RETRIES across ranked satisfying particles (the batched analog of the
    serial max_motion_refine_attempts loop). Returns a length-N list of plans (accum_plans, same schema
    as run_cutamp/solve_curobo); None for a scene whose group found no plan for it. (motion_gen_batched
    is currently unused -- each group builds its own solver -- kept for API compatibility.)
    """
    n = len(envs)
    if n == 0:
        return []
    pps = config.num_particles
    q_inits = q_inits if q_inits is not None else [None] * n
    grasps_list = grasps_list if grasps_list is not None else [None] * n
    # Batched analog of the serial max_motion_refine_attempts particle retries. Live data (job
    # 3507724): rank 0 produces nearly all plans, ranks 1+ produce ~none but cost a full skeleton
    # sweep each (~8s at B=8) -- tune down via CUTAMP_REFINE_RANKS for throughput.
    max_refine_ranks = int(os.environ.get("CUTAMP_REFINE_RANKS", "8"))

    # Debug: capture the FIRST real batch's inputs to disk (env-guarded) for fast OFFLINE replay of the
    # exact perception-built scenes -- avoids the ~20-min containerized datagen startup per debug cycle.
    _cap = os.environ.get("CUTAMP_BATCH_CAPTURE")
    if _cap and not os.path.exists(_cap):
        try:
            torch.save({"grasps_list": grasps_list, "q_inits": q_inits, "envs": envs, "config": config,
                        "cost_reducer": cost_reducer, "constraint_checker": constraint_checker}, _cap)
            _log.info(f"run_cutamp_batched: captured {n} real perception envs -> {_cap}")
        except Exception as _e:
            _log.warning(f"run_cutamp_batched capture failed: {_e}")

    def _solve_group(group: list, sig) -> List[Optional[list]]:
        """Batched particle-opt + refinement (retry across ranked particles) for scenes SHARING a
        skeleton. One batched optimization -> per-scene ranked satisfying particles -> refine with
        rank-0 for all, then rank-1 for the ones still unsolved, ... (max_refine_ranks)."""
        g = len(group)
        skel = group[0]["skeleton"]
        rep = group[0]["world"]
        opt_particles = {
            k: (torch.cat([sc["particles"][k].clone() for sc in group], dim=0) if group[0]["particles"][k] is not None else None)
            for k in group[0]["particles"]
        }
        binit = {name: torch.stack([sc["initial_poses"][name] for sc in group]) for name in group[0]["initial_poses"]}
        bcfg = dataclasses.replace(config, num_particles=g * pps, prop_satisfying_break=None)
        rf = RolloutFunction(skel, rep, bcfg)
        rf.batched_initial_poses, rf.particles_per_scene = binit, pps
        cf = CostFunction(skel, rep, bcfg)
        cf.batched_collision_fn = get_batched_world_collision_cost(
            [get_world_cfg(sc["world"].env, include_movables=False) for sc in group],
            rep.tensor_args, config.world_activation_distance)
        cf.scene_env_idx = torch.arange(g, dtype=torch.int32, device=rep.device).repeat_interleave(pps)
        cf.batched_surfaces = _batched_surface_targets(group, skel, config, rep)
        group_obj_spheres = _batched_object_spheres(group, rep)
        cf.batched_obj_spheres = group_obj_spheres
        pinfo = dict(plan_skeleton=skel, particles=opt_particles, rollout_fn=rf, cost_fn=cf, idx=0,
                     heuristic=0.0, num_satisfying=0, best_cost=float("inf"), best_soft_cost=float("inf"))
        bt = TorchTimer()
        bt.start("start_optimization")
        ParticleOptimizer(bcfg, cost_reducer, constraint_checker)(pinfo, bt, group[0]["vis"])

        ranked_per = []
        for i, sc in enumerate(group):
            block = {k: (v[i * pps:(i + 1) * pps] if v is not None else None) for k, v in opt_particles.items()}
            # Rank with the scene's OWN pre-padding skeleton (skeleton_rank): a count-padded scene's
            # phantom-op constraints must not gate which of its particles count as satisfying.
            skel_rank = sc.get("skeleton_rank") or skel
            pinfo_i = dict(plan_skeleton=skel_rank, particles=block,
                           rollout_fn=RolloutFunction(skel_rank, sc["world"], config),
                           cost_fn=CostFunction(skel_rank, sc["world"], config))
            try:
                ranked_per.append(get_ranked_satisfying_particles(pinfo_i, config, constraint_checker, cost_reducer, sc["vis"]))
            except RuntimeError:
                ranked_per.append(None)

        solvable = [i for i in range(g) if ranked_per[i] is not None]
        n_sat = [(ranked_per[i]["q0"].shape[0] if ranked_per[i] is not None else 0) for i in range(g)]
        _log.info(f"  [batch-diag] group size={g}: {len(solvable)}/{g} scenes have satisfying particles "
                  f"(#satisfying per scene: {n_sat})")
        gp: List[Optional[list]] = [None] * g
        if not solvable:
            return gp
        # Get a batched solver from the (n_envs, sig) cache -- built once, REUSED across batches with a
        # full per-env world reload (skips the seconds-long per-group build+warmup). The historical
        # illegal-memory-access crashes here were a warp-mesh use-after-free in curobo's batched world
        # loading (name-keyed warp cache freed earlier envs' meshes; fixed by (env_idx, name) keying in
        # world_mesh.py), not a rebuild-count effect. num_batch_trajopt_seeds=1 kept as the
        # offline-validated config (the seed=4 'misaligned address' fault was the same use-after-free).
        world_cfgs = [get_world_cfg(group[i]["world"].env, include_movables=True) for i in solvable]
        # Pad the refine batch to a FIXED env count (default 8 = the server batch size): plan_batch_env
        # wall time is FLAT in B (7.74x/scene at B=8, job 3505678) so padded no-op envs are ~free,
        # while keying the mg-cache on the exact solvable count made keys almost never recur (33 builds
        # / 0 reuses in job 3507060) at ~8-9s per build. With a fixed count the key is effectively the
        # signature alone -> one build per sig per server lifetime. Padded envs clone scene 0's world/
        # particles/poses and start DEAD (initial_live) so they skip the per-env serial go-home and
        # never surface in results. CUTAMP_BATCH_PAD=1 restores exact-count batches.
        k_solv = len(solvable)
        B_pad = max(int(os.environ.get("CUTAMP_BATCH_PAD", "8")), k_solv)
        n_pad = B_pad - k_solv
        if n_pad:
            world_cfgs = world_cfgs + [world_cfgs[0]] * n_pad
        mg = _get_cached_batched_mg(rep, B_pad, sig, world_cfgs, config)
        obj_sub = [{name: group[i]["initial_poses"][name].clone() for name in group[i]["initial_poses"]} for i in solvable]
        obj_sub += [{name: t.clone() for name, t in obj_sub[0].items()} for _ in range(n_pad)]
        initial_live = torch.zeros(B_pad, dtype=torch.bool, device=rep.device)
        initial_live[:k_solv] = True
        # Count-padding phantoms per env: refinement skips Pick/Place ops on an env's phantom toys
        # (masked in solve_curobo_batched), so emitted plans contain only that scene's real motions.
        phantoms = [set(group[i].get("phantoms") or ()) for i in solvable]
        phantoms += [set(phantoms[0])] * n_pad
        # Refine-env-indexed object spheres (solvable order + padding) for the union attach blob.
        refine_obj_spheres = {
            name: torch.stack([t[i] for i in solvable] + [t[solvable[0]]] * n_pad)
            for name, t in group_obj_spheres.items()
        }
        keys = list(ranked_per[solvable[0]].keys())
        n_ranked = [ranked_per[i]["q0"].shape[0] for i in solvable]
        grasp_keys = [k for k in keys if k.startswith("grasp") and not k.endswith("_confidences")]
        used_rows = [set() for _ in solvable]

        def _aligned_rows(rank: int) -> list:
            """Pick one ranked-particle row per scene for this refine attempt. Scene 0 walks its
            confidence-ranked list; every OTHER scene picks its best-ranked UNUSED row whose grasp
            ROTATIONS best match scene 0's row. The held-object attach blob is shared across envs and
            built from the first live env's grasp (scene 0), so grasp-aligned rows make every env hold
            its toy in a similar orientation -> the union blob stays at single-toy scale AND is
            shape-correct for everyone. (Unaligned selection measured: the blob's union over 16
            different grasp orientations killed every held segment.)"""
            rows = [min(rank, n_ranked[0] - 1)]
            for j in range(1, len(solvable)):
                i = solvable[j]
                cand = [m for m in range(n_ranked[j]) if m not in used_rows[j]] or [n_ranked[j] - 1]
                cand_t = torch.tensor(cand, device=rep.device, dtype=torch.long)
                score = torch.zeros(len(cand), device=rep.device)
                for gk in grasp_keys:
                    anchor_rot = ranked_per[solvable[0]][gk][rows[0]][:3, :3]
                    rots = ranked_per[i][gk][cand_t][:, :3, :3]
                    tr = (rots * anchor_rot).sum(dim=(-2, -1))  # trace(R^T A) per candidate
                    score = score + torch.arccos(torch.clamp((tr - 1.0) / 2.0, -1.0, 1.0))
                rows.append(cand[int(torch.argmin(score))])
            for j, m in enumerate(rows):
                used_rows[j].add(m)
            return rows

        for rank in range(max_refine_ranks):
            # Deaden envs that are already refined or out of unused ranked particles: they would
            # otherwise re-solve a stale particle every remaining rank. The loop exits once nobody
            # needs another attempt.
            live_rank = initial_live.clone()
            for j, i in enumerate(solvable):
                if gp[i] is not None or len(used_rows[j]) >= n_ranked[j]:
                    live_rank[j] = False
            if not bool(live_rank.any()):
                break
            rows = _aligned_rows(rank)
            parts = {k: torch.stack([ranked_per[i][k][rows[j]] for j, i in enumerate(solvable)]) for k in keys}
            if n_pad:
                parts = {k: torch.cat([v, v[:1].repeat(n_pad, *([1] * (v.dim() - 1)))]) for k, v in parts.items()}
            accum, live = solve_curobo_batched(dict(plan_skeleton=skel), parts, rep, config, mg, obj_sub, B_pad,
                                               initial_live=live_rank, phantom_objs_per_env=phantoms,
                                               batched_obj_spheres=refine_obj_spheres)
            for j, i in enumerate(solvable):
                if gp[i] is None and bool(live[j]):
                    gp[i] = accum[j]
            _log.info(f"  [batch-diag] refine rank {rank}: live={int(live[:k_solv].sum().item())}/{k_solv}, "
                      f"total refined so far={sum(1 for p in gp if p is not None)}/{len(solvable)}")
        # Free the per-group optimizer tensors (NOT mg -- it is cached and reused). empty_cache returns
        # those blocks so they don't fragment across groups.
        del rf, cf, opt_particles
        torch.cuda.empty_cache()
        return gp

    # per-scene setup + task plan + sample (independent; a failed scene -> None, excluded from batching)
    scenes: list = [None] * n
    for i, env in enumerate(envs):
        try:
            _, vis, _timer, world = setup_cutamp(env, config, q_inits[i])
            plan_gen = task_plan_generator(
                world.initial_state, world.goal_state, operators=all_tamp_operators,
                explored_state_check=config.explored_state_check)
            skeleton = next(plan_gen)
            particles = ParticleInitializer(world, config, grasps_list[i])(skeleton)
            if particles is None:
                continue
            scenes[i] = dict(world=world, vis=vis, skeleton=skeleton,
                             ops=tuple(op.operator.name for op in skeleton), particles=particles,
                             initial_poses={obj.name: world.get_object_pose(obj) for obj in world.movables})
            # Canonicalize toward ONE batchable group: positional toy roles, a canonical surface name,
            # a uniform grasp representation, and toy-count padding with masked phantom cycles (the
            # four fragmentation axes of real perception). Default on; CUTAMP_CANONICALIZE=0 restores
            # per-name grouping; CUTAMP_CANON_SURFACE='' / CUTAMP_CANON_COUNT=0 disable those axes
            # individually.
            if os.environ.get("CUTAMP_CANONICALIZE", "1").lower() not in ("0", "false"):
                _canonicalize_scene_by_pick_order(scenes[i])
                canon_surf = os.environ.get("CUTAMP_CANON_SURFACE", "plate")
                if canon_surf:
                    _canonicalize_scene_surface(scenes[i], canon_surf)
                _normalize_scene_grasps(scenes[i], config)
                canon_count = int(os.environ.get("CUTAMP_CANON_COUNT", "3"))
                if canon_count > 0:
                    _pad_scene_to_canonical(scenes[i], config, canon_count)
        except Exception:
            _log.exception(f"run_cutamp_batched: scene {i} setup/sample failed")

    # Group scenes so each group is truly BATCHABLE: batching stacks per-key particle tensors and
    # per-name object poses, so a group must share not just the operator-name sequence but also the
    # exact particle keys+shapes (M2T2 4x4 grasp vs 4-dof fallback differ!) and movable object names
    # (real perception detects varying object sets). Grouping only by ops was too coarse and tripped
    # KeyError / cat dimension-mismatch. Singleton groups (unique signatures) still solve as batch-of-1.
    def _sig(sc):
        parts = tuple(sorted((k, tuple(v.shape[1:])) for k, v in sc["particles"].items() if v is not None))
        # Include the placement-surface set. After toy canonicalization the movable names are uniform, so
        # WITHOUT this two scenes that place on differently-labelled surfaces (real perception calls the
        # plate 'plate' in some scenes, 'white_plate' in others) would group together -- but _solve_group
        # drives the whole group with group[0]'s skeleton, whose surface name would not resolve in the
        # other scenes' worlds/constraint tolerances. Grouping by surface set keeps every group's rep
        # skeleton valid for all its members. (Surfaces are left un-canonicalized in v1; unifying
        # plate/white_plate for even bigger groups is a v2 follow-on that also needs matching
        # constraint_to_tol[StablePlacement] keys.)
        surfaces = tuple(sorted({op.values[3] for op in sc["skeleton"] if op.operator.name == "Place"}))
        return (sc["ops"], parts, tuple(sorted(sc["initial_poses"].keys())), surfaces)

    groups: dict = {}
    for i in range(n):
        if scenes[i] is not None:
            groups.setdefault(_sig(scenes[i]), []).append(i)
    _log.info(f"run_cutamp_batched: {n} scenes -> {len(groups)} batchable group(s) "
              f"(sizes {sorted((len(v) for v in groups.values()), reverse=True)})")

    plans: List[Optional[list]] = [None] * n
    for sig, idxs in groups.items():
        try:
            for i, p in zip(idxs, _solve_group([scenes[i] for i in idxs], sig)):
                plans[i] = p
        except Exception:
            _log.exception(f"run_cutamp_batched: group (size {len(idxs)}) failed")
    return plans
