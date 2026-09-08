# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


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

    # Fail instead of silently falling back to heuristic grasps when M2T2 proposes NOTHING for an
    # object that has to be picked. With m2t2_grasps on, a zero-candidate object is currently NOT
    # dropped: `_sample_grasps` falls through to grasp_4dof_sampler/grasp_6dof_sampler, which sample
    # grasps from the object's collision-sphere approximation rather than from perception. Measured
    # over shipped runs, 18-43% of picked objects took that path, and it is a direct mechanism for
    # closing on empty air (analysis_dataset_diff/TELEOP_VS_APEX.md, Finding 5).
    #
    # Default False so every existing config keeps its current planning outcomes; the fallback is
    # warned about either way. Turn on for data collection, where a guessed grasp that misses costs
    # a whole mislabelled episode. Opt in from cfg/tamp with `require_m2t2_grasps: true`.
    require_m2t2_grasps: bool = False

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
    # How the region an object may be placed on is derived from the surface.
    #   aabb    -- axis-aligned bounding box of the surface, object bottom at its top face
    #   obb     -- minimum-area oriented box, object bottom at the surface's highest vertex
    #   support -- the largest LEVEL, OBSERVED patch of the surface that the object's footprint fits
    #              inside with margin, object bottom at that patch's own height. Needs the surface's
    #              raw point cloud (TAMPEnvironment.support_points); see cutamp/utils/support.py for
    #              why the bounding box is the wrong region for anything that is not a slab.
    placement_check: Literal["aabb", "obb", "support"] = "aabb"
    # Distance to shrink the placement region check on all sides, only supported for OBB right now
    placement_shrink_dist: Optional[float] = None
    # placement_check="support" knobs -- see cutamp.utils.support.SupportConfig for what each does.
    support_resolution: float = 0.005
    support_flatness_tol: float = 0.008
    support_margin: float = 0.005
    # Treat the unobserved cells enclosed by a surface's outline as floor, so an object can be placed
    # in a container whose interior the camera could not see into. Off by default: it is the one knob
    # here that places onto surface that was never observed. See cutamp.utils.support._fill_occluded.
    support_fill_occluded: bool = False
    # With support_fill_occluded, the fraction of a footprint that must be genuinely observed.
    support_min_seen_frac: float = 0.25
    # What to do when no level patch of a surface is big enough for the object. True (the default)
    # rejects the placement, so the plan fails with a reason instead of releasing the object over
    # whatever the bounding box happened to span. False falls back to the OBB region and logs.
    placement_support_required: bool = True
    # Let a PLACED object overlap the surface it was placed on in the movable-to-world collision
    # cost, from that placement onwards. Perception reconstructs an open container as its convex
    # hull -- a filled solid -- so the inside of a box, and the dish of a plate, are "in collision"
    # with the container itself and no placement there can ever satisfy the constraint. Only the
    # object's own target surface is exempted, and only from its placement on; every other obstacle,
    # and the same object before it is placed, keeps the full checker. Placing INTO a container
    # needs this; it is only sound alongside placement_check="support", which is what then keeps the
    # object on real geometry rather than inside it.
    placement_ignores_target_surface: bool = False

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
    # Enable the GraspCost soft cost: horizontal (in-object-frame xy) distance between the grasp's TCP
    # and the object's frame origin, steering the planner toward grasps nearer the middle of the object
    # rather than out at an edge, where the lever arm about the contact makes the hold unstable. Gated
    # here (default off) for the same weight-1.0 reason as grasp_orientation_cost above; the weight is
    # set separately via constraint_to_mult[GraspCost.type]["grasp_center_offset"].
    grasp_center_cost: bool = False
    # Weight on summed M2T2 grasp confidence when RANKING the satisfying particles that motion
    # refinement is attempted on (get_ranked_satisfying_particles), scoring each particle
    # `soft_cost - weight * summed_confidence`, lower first.
    #
    # None (the default, and the historical behaviour) ranks on confidence ALONE whenever M2T2
    # confidences are present, which is every TiPToP run. That ranking ignores every soft cost, and
    # since cuRobo almost always succeeds on the first-ranked particle, the EXECUTED grasp is just
    # the argmax-confidence candidate -- so grasp_center_cost / grasp_orientation_cost influence
    # nothing but the logged breakdown. Measured over the 22 shipped runs of
    # 4_pack_toys_..._learned_posture_v2, the executed grasp was confidence rank 0 in 48/66 picks
    # and within the top 4 in all 66, sitting at a uniformly random offset percentile of the
    # candidate pool (banana 58th, toys 41st) despite grasp_center_weight=30.
    #
    # Set this to fold the soft costs back into that ranking. Confidence is kept in the score rather
    # than dropped: it is the only signal that the grasp is on real graspable geometry, and ranking
    # on soft cost alone would happily pick a centred grasp on a reconstruction artifact. 0.0 means
    # rank on soft cost alone.
    grasp_rank_conf_weight: Optional[float] = None

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

    # --- Teleop-posture IK branch selection -------------------------------------------------- #
    # Every plan endpoint comes from ik_solver.solve_batch(..., seed_config=None) with the default
    # return_seeds=1, and cuRobo ranks its seeds by pose_error + null_space_error where
    # null_space_cfg.weight is 0.001 against a generic home retract -- i.e. the redundant arm's
    # branch is picked by pose error alone, independently for every endpoint. The arm's redundancy
    # then lands wherever, which is the bulk of why TAMP trajectories visit joint configurations
    # teleoperation never does.
    #
    # With this set to k > 1, each endpoint's IK is solved with return_seeds=k and the branch with
    # the lowest posture penalty is kept instead of cuRobo's top seed. The penalty is entirely
    # data-derived -- see cutamp/posture_prior.py and the baked posture_ref.npz: a per-joint-pair
    # human band weighted by how much that pair's direction is free to move (Jacobian null space).
    # There is nothing to tune per scene; k is the only knob.
    #
    # Measured on 512-particle batches this costs nothing (~31 ms at k=12 against multi-second
    # plans). 0 or 1 (the default) disables it: IK is solved and read exactly as before.
    posture_selection_seeds: int = 0
    # Also consider the pi-ROLLED twin of each grasp. A parallel jaw is invariant to a pi roll about
    # its approach axis, so every grasp has two equally valid end-effector poses -- and cuTAMP only
    # ever builds one of them, chosen by whichever representative M2T2 happened to emit. Roughly half
    # the time that is the one teleoperation never uses, and NO amount of IK branch selection can fix
    # it, because both branches of a single pose share that pose's wrist roll.
    #
    # This is the dominant residual once branch selection is on: measured over 512 endpoints, the
    # best achievable branch of the given pose alone still leaves 37-42% of configurations outside
    # held-out DROID's 99th percentile, while adding the twin pose takes the SELECTED configuration
    # to 5% (fruits) / 7% (plate). The twin wins about half the time, as expected.
    #
    # Costs one extra solve_batch per endpoint (they must be separate same-size calls -- doubling
    # the batch trips cuRobo's cuda graph with "changing goal type"). When the twin wins, the stored
    # obj_from_grasp is rolled to match, so the recorded grasp always agrees with the configuration.
    # Only applies where the grasp is a 4x4 matrix (M2T2 grasps); 4/6-DOF sampled grasps are skipped
    # because rolling them cannot be expressed in their parameterisation.
    posture_grasp_roll: bool = True
    # Path to the baked prior. None -> cutamp/posture_ref.npz (or $CUTAMP_POSTURE_REF).
    posture_ref: Optional[str] = None
    # Tolerances a returned seed must meet, by forward kinematics, to be considered at all.
    # cuRobo's IKResult.success comes back all-False in this batched path, so the branches are
    # validated by FK rather than trusted.
    posture_pos_tol: float = 5e-3
    posture_rot_tol: float = 0.05

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

    # Teleop-posture IK branch selection
    if config.posture_selection_seeds < 0:
        raise ValueError(f"posture_selection_seeds must be non-negative, not {config.posture_selection_seeds}")
    if config.posture_selection_seeds < 0:
        raise ValueError(
            f"posture_selection_seeds must be non-negative, not {config.posture_selection_seeds}"
        )

    # Motion refinement
    if config.max_motion_refine_attempts is not None and config.max_motion_refine_attempts <= 0:
        raise ValueError(f"max_motion_refine_attempts must be positive or None, not {config.max_motion_refine_attempts}")
    # Negative would rank LOW-confidence grasps first, which is never what a caller means. 0.0 is
    # allowed and means rank on soft cost alone.
    if config.grasp_rank_conf_weight is not None and config.grasp_rank_conf_weight < 0:
        raise ValueError(
            f"grasp_rank_conf_weight must be non-negative or None, not {config.grasp_rank_conf_weight}"
        )

    # Placement region checks
    if config.placement_check != "obb" and config.placement_shrink_dist is not None:
        # "support" has its own clearance knob (support_margin), applied against the object's actual
        # footprint rather than as a blanket inset, so accepting both would shrink twice.
        raise NotImplementedError(
            f"placement_shrink_dist only supported with placement_check = obb, not {config.placement_check}"
            + (" -- use support_margin instead" if config.placement_check == "support" else "")
        )
    if config.support_resolution <= 0.0:
        raise ValueError(f"support_resolution must be positive, not {config.support_resolution}")
    if config.support_flatness_tol <= 0.0:
        raise ValueError(f"support_flatness_tol must be positive, not {config.support_flatness_tol}")
    if config.support_margin < 0.0:
        raise ValueError(f"support_margin must be non-negative, not {config.support_margin}")
    if not 0.0 <= config.support_min_seen_frac <= 1.0:
        raise ValueError(
            f"support_min_seen_frac must be in [0, 1], not {config.support_min_seen_frac}"
        )
    if config.placement_ignores_target_surface and config.placement_check != "support":
        # Without the support region there is nothing keeping the object ON the surface once the
        # surface stops rejecting it -- the bounding-box region would happily drop it through.
        raise ValueError(
            "placement_ignores_target_surface requires placement_check = support, not "
            f"{config.placement_check}"
        )
