# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class TAMPConfiguration:
    # Number of particles to initialize and optimize over
    num_particles: int = 1024

    # Robot embodiment to use. The bimanual YAM appears once per arm because cuTAMP plans a single
    # kinematic chain: the active arm is fixed by that config's ee_link + lock_joints.
    robot: Literal[
        "panda", "fr3_robotiq", "ur5", "panda_robotiq", "fr3_franka",
        "bimanual_yam_left", "bimanual_yam_right", "bimanual_yam_dual",
    ] = "panda"

    # "single" = one kinematic chain, the original domain (Pick/Place/MoveFree/MoveHolding).
    # "dual"   = both arms as one chain acting in LOCKSTEP, using dual_tamp_operators: each timestep
    #            has one shared configuration constrained at BOTH hands, so the two arms pick and
    #            place simultaneously. Requires a multi-arm robot; see cutamp/robots ArmSpec.
    arm_mode: Literal["single", "dual"] = "single"

    # Which dual-arm operator set to plan with (ignored unless arm_mode == "dual").
    #   "parallel" -- lockstep: both hands pick, and both place, at the same configuration.
    #   "handover" -- asymmetric: one hand picks, both hands meet on the object, the other carries on.
    dual_task: Literal["parallel", "handover"] = "parallel"

    # Grasp and Placements
    grasp_dof: Literal[4, 6] = 4
    place_dof: Literal[4] = 4

    # M2T2 Grasps which will be used first, and then grasp_dof fallback
    m2t2_grasps: bool = False

    # Approach to use. Note: optimization includes particle initialization (i.e., sampling)
    approach: Literal["optimization", "sampling"] = "optimization"

    # Number of resampling attempts per plan skeleton if the approach is sampling
    num_resampling_attempts: int = 0

    # Optimization hyperparams
    num_opt_steps: int = 1_000
    lr: float = 7e-3  # default LR for optimizer
    conf_lr: float = 2.226e-2  # LR for robot configurations

    ## Advanced args - for soft cost experiments. Warning! Might cause unexpected behavior if not used correctly.
    # Maximum time for optimization or sampling in seconds before breaking
    max_loop_dur: Optional[float] = None
    # Proportion satisfying to break for optimization
    prop_satisfying_break: Optional[float] = None
    # Whether to break upon finding a satisfying particle
    break_on_satisfying: bool = True
    # Whether we're running stick button experiment. Modifies heuristic for comparing baselines
    stick_button_experiment: bool = False

    ## Experimental Stuff
    # Whether to check placements using AABB or OBB formulation
    placement_check: Literal["aabb", "obb"] = "aabb"
    # Distance to shrink the placement region check on all sides, only supported for OBB right now
    placement_shrink_dist: Optional[float] = None

    ## Soft Costs
    optimize_soft_costs: bool = False
    # Supported: dist_from_origin, max_obj_dist, min_obj_dist, min_y, max_y, align_yaw
    soft_cost: Optional[str] = None
    # Enable the GraspCost soft cost: geodesic angle between each grasp's end-effector orientation and
    # the robot's initial EE orientation (FK of q_init), steering the planner toward grasps that
    # reorient the wrist least. Gated here (default off) because the cost reducer treats an ABSENT
    # multiplier as weight 1.0, so an always-emitted value would change every caller's objective; the
    # weight is set separately via constraint_to_mult[GraspCost.type]["grasp_rot_change"].
    grasp_orientation_cost: bool = False

    ## Task Planning and subgraph caching
    # Number of initial plans to sample
    num_initial_plans: int = 30
    # Whether to check if state has been explored before adding to search tree for task planning
    explored_state_check: bool = True
    # Cache particles for reuse - this is coupled to our current domains so the implementation is not general
    # Warning: don't use this with the sampling approach, as the samplers will just return the same samples
    cache_subgraphs: bool = False
    skip_failed_subgraphs: bool = False
    # Random particle initialization for placements and robot configurations. Not supported for all domains.
    random_init: bool = False

    ## Collision Checking
    # Movable object collision sphere representation
    coll_n_spheres: int = 50
    coll_sphere_radius: float = 0.005
    # Distance at which collision checking is activated between the world (in cuRobo)
    world_activation_distance: float = 0.0
    # Mask out movable-to-world collision costs at timesteps before each object's first placement.
    # Objects may initially be in collision with surfaces they rest on due to perception noise.
    # Set to False in simulation or when debugging to surface genuine environment setup issues.
    mask_initial_movable_world_collision: bool = True
    # Distance at which collision checking is activated between gripper and movables
    gripper_activation_distance: float = 0.0
    # Distance at which collision checking is activated between movables and movables
    movable_activation_distance: float = 0.0

    ## Trajectories and cuRobo
    # Norm p used for the per-move joint-space distance ||q_start - q_end||_p in the TrajectoryLength
    # cost (see cutamp/costs.py::trajectory_length). 2.0 is the Euclidean straight-line distance;
    # float("inf") is the max joint displacement (the infinity-norm). Both lower-bound the shortest
    # collision-free path length between the two configurations.
    traj_length_norm: float = 2.0
    # Height (metres) of an explicit APEX waypoint inserted into each free-space transit of a Pick or
    # Place -- the long unconstrained leg between the retract and the pre-grasp/pre-place pose. The
    # transit is planned as start -> apex -> goal instead of start -> goal, where the apex sits at the
    # horizontal midpoint of the two end-effector positions, `transit_apex_height` above the HIGHER of
    # them, carrying the goal orientation.
    #
    # Why an explicit waypoint rather than a cost: cuRobo's trajopt objective is entirely joint-space
    # (bound_cfg's L2 on acceleration/jerk plus the limit hinges) and has no term reading the
    # end-effector's Cartesian path, so with fixed endpoints its optimum is the joint-space geodesic --
    # the EE skims across the table in a low lateral sweep. Splitting the transit at an apex changes
    # the endpoints themselves, which is the only lever that shapes the Cartesian path without adding
    # a new cost term.
    #
    # 0.0 (default) disables it: the transit is planned exactly as before, one plan_single call.
    # Applies to the single-arm solver (solve_curobo) only. If either apex leg fails to plan, the
    # solver silently falls back to the direct transit, so enabling this can never lose a plan that
    # would otherwise have been found.
    transit_apex_height: float = 0.0
    # Minimum horizontal (xy) distance between the transit's start and goal end-effector positions for
    # the apex to be inserted. Short transits -- an object placed right next to where it was picked --
    # get a pointless up-and-down from an apex, so they keep the direct plan. Ignored when
    # transit_apex_height is 0.
    transit_apex_min_dist: float = 0.10
    # Whether to also optimize full trajectories (not supported right now)
    enable_traj: bool = False
    # Motion plan with cuRobo after optimization
    curobo_plan: bool = False
    # Max satisfying particles to try motion refinement on per skeleton (None = try all)
    max_motion_refine_attempts: Optional[int] = None
    # For slowing down cuRobo motion plans (0.5 is safe on the real robot)
    time_dilation_factor: Optional[float] = None
    # Whether to warmup IK solver
    warmup_ik: bool = True
    # Whether to warmup motion generator
    warmup_motion_gen: bool = True

    ## Visualizer Args
    # Whether to use visualizer, if set to False a Mock is used
    enable_visualizer: bool = True
    # Number of steps between visualizations of optimization state (note: visualization takes non-trivial time)
    opt_viz_interval: int = 10
    # Whether to visualize the robot mesh, set to False if you want to save network bandwidth (~10MB)
    viz_robot_mesh: bool = True
    # Spawn the rerun visualizer
    rr_spawn: bool = True

    ## Logging Args
    enable_experiment_logging: bool = True
    # Root directory for logging experiments
    experiment_root: str = "/tmp/cutamp-experiments"
    # Save a factor-graph representation (JSON + Graphviz DOT) of the final solved plan skeleton
    # into the experiment directory. See cutamp/plan_graph.py.
    save_plan_graph: bool = False


def validate_tamp_config(config: TAMPConfiguration):
    if config.num_particles <= 0:
        raise ValueError(f"num_particles must be positive, not {config.num_particles}")
    if config.robot not in {
        "panda", "fr3_robotiq", "ur5", "panda_robotiq", "fr3_franka",
        "bimanual_yam_left", "bimanual_yam_right", "bimanual_yam_dual",
    }:
        raise ValueError(f"Invalid embodiment: {config.robot}")
    if config.arm_mode not in {"single", "dual"}:
        raise ValueError(f"Invalid arm_mode: {config.arm_mode}")
    if config.dual_task not in {"parallel", "handover"}:
        raise ValueError(f"Invalid dual_task: {config.dual_task}")
    if (config.arm_mode == "dual") != (config.robot == "bimanual_yam_dual"):
        raise ValueError(
            f"arm_mode={config.arm_mode!r} does not match robot={config.robot!r}; "
            "dual-arm planning needs the 12-DOF 'bimanual_yam_dual' embodiment and vice versa"
        )
    if config.grasp_dof not in {4, 6}:
        raise ValueError(f"Invalid grasp_dof: {config.grasp_dof}")
    if config.place_dof not in {4}:
        raise ValueError(f"Invalid place_dof: {config.place_dof}")
    if config.approach not in {"optimization", "sampling"}:
        raise ValueError(f"Invalid approach: {config.approach}")
    if config.num_resampling_attempts < 0:
        raise ValueError(f"num_resampling_attempts must be non-negative, not {config.num_resampling_attempts}")

    # Optimization hyperparams
    if config.num_opt_steps <= 0:
        raise ValueError(f"num_opt_steps must be positive, not {config.num_opt_steps}")
    if config.lr <= 0:
        raise ValueError(f"Learning rate (lr) must be positive, not {config.lr}")
    if config.conf_lr <= 0:
        raise ValueError(f"Configuration learning rate (conf_lr) must be positive, not {config.conf_lr}")

    # Advanced args
    if config.max_loop_dur is not None and config.max_loop_dur <= 0:
        raise ValueError(f"max_loop_dur must be positive or None, not {config.max_loop_dur}")

    # Task Planning and subgraph caching
    if config.num_initial_plans <= 0:
        raise ValueError(f"num_initial_plans must be positive, not {config.num_initial_plans}")
    if config.cache_subgraphs and config.approach == "sampling":
        raise ValueError("cache_subgraphs is not compatible with sampling approach")

    # Collision checking
    if config.coll_n_spheres <= 0:
        raise ValueError(f"coll_n_spheres must be positive, not {config.coll_n_spheres}")
    if config.coll_sphere_radius <= 0:
        raise ValueError(f"coll_sphere_radius must be positive, not {config.coll_sphere_radius}")
    if config.world_activation_distance < 0:
        raise ValueError(f"world_activation_distance must be non-negative, not {config.world_activation_distance}")
    if config.movable_activation_distance < 0:
        raise ValueError(f"movable_activation_distance must be non-negative, not {config.movable_activation_distance}")

    # Trajectory length norm (torch.norm requires p >= 1; inf is allowed for the max-norm)
    if config.traj_length_norm < 1:
        raise ValueError(f"traj_length_norm must be >= 1 (or inf), not {config.traj_length_norm}")

    # Transit apex waypoint
    if config.transit_apex_height < 0:
        raise ValueError(f"transit_apex_height must be non-negative, not {config.transit_apex_height}")
    if config.transit_apex_min_dist < 0:
        raise ValueError(f"transit_apex_min_dist must be non-negative, not {config.transit_apex_min_dist}")

    # Motion refinement
    if config.max_motion_refine_attempts is not None and config.max_motion_refine_attempts <= 0:
        raise ValueError(f"max_motion_refine_attempts must be positive or None, not {config.max_motion_refine_attempts}")

    # Placement region checks
    if config.placement_check != "obb" and config.placement_shrink_dist is not None:
        raise NotImplementedError(
            f"placement_shrink_dist only supported with placement_check = obb, not {config.placement_check}"
        )
