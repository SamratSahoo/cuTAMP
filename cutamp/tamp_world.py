# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import itertools
import logging
import warnings
from functools import cached_property
from typing import TYPE_CHECKING, List, Literal, Dict, Union, Optional

import numpy as np
import torch
from jaxtyping import Float

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.types import Obstacle
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.ik_solver import IKSolver
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from cutamp.costs import sphere_to_sphere_overlap
from cutamp.envs import TAMPEnvironment
from cutamp.robots import RobotContainer, load_robot_container
from cutamp.robots.franka_robotiq import get_fr3_robotiq_ik_solver, fr3_robotiq_curobo_cfg
from cutamp.robots.franka import franka_curobo_cfg, get_franka_ik_solver, get_fr3_franka_ik_solver, fr3_franka_curobo_cfg
from cutamp.robots.ur5 import ur5_curobo_cfg, get_ur5_ik_solver
from cutamp.robots.bimanual_yam import bimanual_yam_curobo_cfg, get_bimanual_yam_ik_solver
from cutamp.tamp_domain import get_initial_state
from cutamp.task_planning import State
from cutamp.utils.collision import get_world_collision_cost
from cutamp.utils.common import approximate_goal_aabb, transform_spheres
from cutamp.utils.common import sample_between_bounds, get_world_cfg, pose_list_to_mat4x4
from cutamp.utils.obb import OrientedBoundingBox, get_object_obb
from cutamp.utils.support import Footprint, NoSupportRegion, SupportConfig, fit_support_region
from cutamp.utils.shapes import sample_greedy_surface_spheres

if TYPE_CHECKING:
    from cutamp.config import TAMPConfiguration

_log = logging.getLogger(__name__)


class TAMPWorld:
    """
    Represents a TAMP world that wraps a static TAMPEnvironment with robot-specific logic,
    object indexing utilities, collision checking, IK solvers, and motion generation support.
    """

    def __init__(
        self,
        env: TAMPEnvironment,
        tensor_args: TensorDeviceType,
        robot: Union[Literal["panda", "ur5"], RobotContainer],
        q_init: Float[torch.Tensor, "dof"],
        collision_activation_distance: float = 0.0,
        coll_n_spheres: int = 50,
        coll_sphere_radius: float = 0.005,
        ik_solver: Optional[IKSolver] = None,
    ):
        self.env = env
        self.tensor_args = tensor_args
        # Raw observed points per surface, for the "support" placement region (see placement_obb).
        self.support_points = dict(getattr(env, "support_points", {}) or {})
        self._placement_obbs: Dict[tuple, tuple] = {}
        self._collision_fns_excluding: Dict[str, object] = {}

        # Dicts and sets for indexing
        self._movable_names = {obj.name for obj in env.movables}
        self._name_to_obj = {obj.name: obj for obj in env.movables + env.statics}

        # Setup collision function
        # Surfaces the arm is allowed to reach INTO. Perception reconstructs an open container
        # (a plate, a bowl) as a mesh whose collision proxy is a filled OBB spanning its full
        # height, so every object resting in it is embedded in an obstacle. Measured on a scene
        # reset: with the plate in the world, grasp IK for the three toys was 0/25, 0/1 and
        # 13/30; with it out, 25/25, 1/1 and 30/30 -- and cuTAMP's own robot_to_world rejected
        # the pick configurations too (90/256, 82/256, 22/256 against 253-256 elsewhere).
        #
        # They are dropped from `reach_world_cfg` only, which feeds the IK solvers and the
        # robot_to_world cost -- both of which merely CHOOSE WAYPOINTS. Everything that screens
        # geometry keeps the full `world_cfg`: movable_to_world, and the candidate-placement
        # filter in particle_initialization, so placing a toy half-on the plate rim is still
        # rejected on the task path. Placement HEIGHT is untouched either way -- it comes from
        # _name_to_obj, not from a collision checker. The motion generator also keeps them
        # (get_motion_gen / algorithm.py build their world off the untouched env), so the
        # planned path still avoids the container everywhere except the pick and place-retract
        # legs, where solve_curobo hides them explicitly via _obstacles_hidden.
        self.pick_transparent = {
            name for name in getattr(env, "pick_transparent", ()) if name in self._name_to_obj
        }
        if self.pick_transparent:
            _log.info(f"Reach-into surfaces hidden from IK/collision cost: {sorted(self.pick_transparent)}")
        # doesn't include movables
        self.world_cfg = get_world_cfg(env, include_movables=False)
        self.collision_fn = get_world_collision_cost(self.world_cfg, tensor_args, collision_activation_distance)
        # Reach-into variant. Identical to the above when nothing is pick_transparent.
        self.reach_world_cfg = self.world_cfg
        self.robot_collision_fn = self.collision_fn
        if self.pick_transparent:
            self.reach_world_cfg = get_world_cfg(
                env, include_movables=False, exclude=self.pick_transparent
            )
            self.robot_collision_fn = get_world_collision_cost(
                self.reach_world_cfg, tensor_args, collision_activation_distance
            )
        self.collision_activation_distance = collision_activation_distance

        # Setup robot container
        if isinstance(robot, str):
            warnings.warn(f"RobotContainer not provided, loading based on robot name {robot}")
            self.robot_container = load_robot_container(robot, tensor_args)
        else:
            self.robot_container = robot
        self.robot_name = self.robot_container.name
        self.q_init = q_init

        # Setup the IK solver, right now it needs WorldCfg and I don't know the behavior, can speed up later
        if ik_solver is not None:
            self.ik_solver = ik_solver
            self.ik_solver.update_world(self.reach_world_cfg)
        elif self.robot_name == "panda":
            self.ik_solver = get_franka_ik_solver(self.reach_world_cfg)
        elif self.robot_name == "panda_robotiq":
            self.ik_solver = get_franka_ik_solver(self.reach_world_cfg)
        elif self.robot_name == "fr3_robotiq":
            self.ik_solver = get_fr3_robotiq_ik_solver(self.reach_world_cfg)
        elif self.robot_name == "fr3_franka":
            self.ik_solver = get_fr3_franka_ik_solver(self.reach_world_cfg)
        elif self.robot_name == "ur5":
            self.ik_solver = get_ur5_ik_solver(self.reach_world_cfg)
        elif self.robot_name.startswith("bimanual_yam_"):
            self.ik_solver = get_bimanual_yam_ik_solver(self.reach_world_cfg, self.robot_name.rsplit("_", 1)[1])
        else:
            raise ValueError(f"Unsupported robot: {self.robot_name}")

        # Per-arm IK solvers, only for a multi-arm container. Dual-arm configurations are SEEDED by
        # solving each arm's 6-DOF IK independently against its own single-arm cuRobo config and
        # scattering the two solutions into the 12-vector's column slices -- verified to reproduce
        # both target poses through the dual forward kinematics to <1e-5. cuRobo has no API for a
        # simultaneous two-pose IK, and cuTAMP does not need one: these are only initial particles,
        # and the optimizer refines them against the dual kinematic cost.
        self.ik_solvers: Dict[str, IKSolver] = {}
        if self.robot_container.is_multi_arm:
            for spec in self.robot_container.arms:
                self.ik_solvers[spec.name] = get_bimanual_yam_ik_solver(self.reach_world_cfg, spec.name)

        # Sample collision spheres for all movables
        self._obj_to_spheres: Dict[str, Float[torch.Tensor, "n 4"]] = {}
        for obj in self.movables:
            spheres = sample_greedy_surface_spheres(obj, n_spheres=coll_n_spheres, sphere_radius=coll_sphere_radius)
            self._obj_to_spheres[obj.name] = spheres.to(tensor_args.device)

        # AABB cache
        self._obj_to_aabb = {}

    @property
    def movables(self) -> List[Obstacle]:
        return self.env.movables

    def is_movable(self, obj: Obstacle | str) -> bool:
        if isinstance(obj, Obstacle):
            obj = obj.name
        return obj in self._movable_names

    @property
    def statics(self) -> List[Obstacle]:
        return self.env.statics

    @property
    def kin_model(self) -> CudaRobotModel:
        return self.robot_container.kin_model

    @property
    def handover_region(self):
        """(lo, hi) xyz box, in the world frame, where a two-handed exchange may happen.

        Mid-air between the two shoulders and above the table: an IK sweep over this box found the
        two arms can both reach an object there without their upper arms intersecting. Derived from
        the arm mounts rather than hard-coded so it follows the robot if the base pose changes.
        """
        base = self.kin_model.get_state(
            torch.zeros((1, self.robot_container.joint_limits.shape[1]), device=self.device)
        )
        # Midpoint between the hands at the zero configuration fixes x and y; z spans a band above
        # the table that both arms can reach.
        mid = torch.stack([base.link_pose[a.ee_link].position[0] for a in self.arms]).mean(dim=0)
        lo = torch.tensor([float(mid[0]) - 0.08, float(mid[1]) - 0.05, 0.18], device=self.device)
        hi = torch.tensor([float(mid[0]) + 0.14, float(mid[1]) + 0.05, 0.34], device=self.device)
        return lo, hi

    @property
    def arm_side_signs(self) -> dict:
        """Sign of each arm's world y at the zero configuration: which side of the robot it works on."""
        base = self.kin_model.get_state(
            torch.zeros((1, self.robot_container.joint_limits.shape[1]), device=self.device)
        )
        return {
            a.name: 1.0 if float(base.link_pose[a.ee_link].position[0][1]) >= 0 else -1.0
            for a in self.arms
        }

    @property
    def arms(self) -> tuple:
        """Per-arm specs, or () for a single-arm robot. Non-empty selects every dual-arm code path."""
        return self.robot_container.arms

    @property
    def tool_from_ee(self) -> Float[torch.Tensor, "4 4"]:
        """Transformation from tool frame to end-effector frame used by kinematics model and IK solver."""
        return self.robot_container.tool_from_ee

    @property
    def device(self) -> torch.device:
        return self.tensor_args.device

    @property
    def initial_state(self) -> State:
        initial_state = get_initial_state(
            movables=self.get_objects_by_type("Movable", return_name=True),
            surfaces=self.get_objects_by_type("Surface", return_name=True),
            sticks=self.get_objects_by_type("Stick", return_name=True),
            buttons=self.get_objects_by_type("Button", return_name=True),
        )
        return initial_state

    @property
    def goal_state(self) -> State:
        return self.env.goal_state

    def get_objects_by_type(self, obj_type: str, return_name: bool = True) -> List[Union[Obstacle, str]]:
        if obj_type not in self.env.type_to_objects:
            return []
        objs = self.env.type_to_objects[obj_type]
        if return_name:
            objs = [obj.name for obj in objs]
        return objs

    def get_object(self, name: str) -> Obstacle:
        """Get cuRobo Obstacle for object with the given name."""
        if name not in self._name_to_obj:
            raise ValueError(f"Object '{name}' not found in environment")
        return self._name_to_obj[name]

    def has_object(self, name: str) -> bool:
        """Whether the object with the given name exists in the environment."""
        return name in self._name_to_obj

    def get_object_pose(self, obj: Union[Obstacle, str]) -> Float[torch.Tensor, "4 4"]:
        """Get the object initial pose."""
        obj = obj if isinstance(obj, Obstacle) else self.get_object(obj)
        mat4x4 = pose_list_to_mat4x4(obj.pose).to(self.device)
        return mat4x4

    def get_collision_spheres(self, obj: Union[Obstacle, str]) -> Float[torch.Tensor, "n 4"]:
        """Get the collision spheres for the object (by either name or the cuRobo Obstacle)."""
        obj_name = obj.name if isinstance(obj, Obstacle) else obj
        return self._obj_to_spheres[obj_name]

    def placement_obb(
        self, surface: str, footprint: "Footprint", config: "TAMPConfiguration"
    ) -> tuple[OrientedBoundingBox, Optional[float]]:
        """The region an object of this ``footprint`` may be placed on ``surface``, and its yaw.

        The yaw is None when the placement may be made at any orientation -- the normal case. It is
        set when the object only fits the surface at a committed orientation, and both the sampler
        and the placement cost then have to pin the placement's yaw to it (see cutamp.utils.support).

        One place, so the sampler that draws placements and the cost that scores them cannot
        disagree about where the surface is -- they did when each called ``get_object_obb`` itself.
        Cached per (surface, rounded footprint); the half extents are quantised to a millimetre so two
        objects of near-identical size share one fit.

        For ``placement_check="support"`` this is the largest level patch of the surface's observed
        point cloud that the footprint fits inside, at that patch's own height (see
        cutamp.utils.support). Falls back to the surface's oriented bounding box when the surface has
        no point cloud -- the table, and anything from a hand-built environment, are genuinely slabs
        and the box is right for them.

        Raises:
            NoSupportRegion: when the surface HAS a point cloud but no level patch of it is big
                enough, and ``config.placement_support_required``.
        """
        key = (surface, *(round(v, 3) for v in
                          (footprint.min_x, footprint.max_x, footprint.min_y, footprint.max_y)))
        if key in self._placement_obbs:
            return self._placement_obbs[key]

        surface_obj = self.get_object(surface)
        obb, object_yaw = None, None
        if config.placement_check == "support":
            points = self.support_points.get(surface)
            if points is None:
                _log.info(
                    f"Surface '{surface}' has no point cloud; using its bounding box as the "
                    "placement region"
                )
            else:
                region = fit_support_region(
                    points,
                    footprint,
                    SupportConfig(
                        resolution=config.support_resolution,
                        flatness_tol=config.support_flatness_tol,
                        margin=config.support_margin,
                        fill_occluded=config.support_fill_occluded,
                        min_seen_frac=config.support_min_seen_frac,
                    ),
                )
                if region is None:
                    msg = (
                        f"No level patch of '{surface}' is large enough to support an object of "
                        f"footprint {(footprint.max_x - footprint.min_x) * 100:.1f} x "
                        f"{(footprint.max_y - footprint.min_y) * 100:.1f} cm (swept disc "
                        f"{footprint.radius * 200:.1f} cm across) with "
                        f"{config.support_margin * 100:.1f} cm margin"
                    )
                    if config.placement_support_required:
                        raise NoSupportRegion(msg)
                    _log.warning(f"{msg}; falling back to the bounding box placement region")
                else:
                    _log.info(
                        f"Support region for '{surface}': "
                        f"{region.half_extents[0] * 200:.1f}x{region.half_extents[1] * 200:.1f} cm at "
                        f"z={region.surface_z:.3f} ({region.observed_frac:.0%} of the footprint "
                        f"genuinely observed{'' if region.object_yaw is None else f', at a pinned yaw of {np.degrees(region.object_yaw) % 180:.0f} deg'}), "
                        f"against a bounding box top at z={get_object_obb(surface_obj).surface_z:.3f}"
                    )
                    obb, object_yaw = region.to_obb(self.tensor_args), region.object_yaw

        if obb is None:
            # support_margin stands in for placement_shrink_dist on this path, which the "support"
            # configuration leaves unset -- otherwise the table, the one surface that never has a
            # point cloud, would be the only surface placements could run right up to the edge of.
            shrink = config.placement_shrink_dist
            if shrink is None and config.placement_check == "support":
                shrink = config.support_margin
            obb = get_object_obb(surface_obj, shrink_dist=shrink)
        self._placement_obbs[key] = (obb, object_yaw)
        return self._placement_obbs[key]

    def collision_fn_for_placement(self, exclude: Optional[str] = None):
        """``collision_fn``, optionally with one obstacle dropped.

        ``exclude`` is the surface an object is being placed ON. Perception reconstructs an open
        container as its convex hull, so the inside of a box and the dish of a plate are both solid
        to the checker and every placement there scores as a collision -- see
        ``TAMPConfiguration.placement_ignores_target_surface``. Cached per name; there are as many
        of these as there are placement surfaces in a plan, which is one or two.
        """
        if exclude is None:
            return self.collision_fn
        if exclude not in self._collision_fns_excluding:
            cfg = get_world_cfg(self.env, include_movables=False, exclude={exclude})
            self._collision_fns_excluding[exclude] = get_world_collision_cost(
                cfg, self.tensor_args, self.collision_activation_distance
            )
        return self._collision_fns_excluding[exclude]

    def get_aabb(self, obj: Union[Obstacle, str]) -> Float[torch.Tensor, "2 3"]:
        """Get AABB for the given object."""
        obj_name = obj.name if isinstance(obj, Obstacle) else obj
        # Compute AABB if not cached
        if obj_name not in self._obj_to_aabb:
            obj = self.get_object(obj_name)
            aabb = approximate_goal_aabb(obj).to(self.device)
            # aabb[0, :2] += 0.02
            # aabb[1, :2] -= 0.02
            self._obj_to_aabb[obj_name] = aabb
        return self._obj_to_aabb[obj_name]

    @cached_property
    def world_aabb(self) -> Float[torch.Tensor, "2 3"]:
        """Get AABB for the entire world (i.e., union of all objects)"""
        aabbs = [self.get_aabb(obj) for obj in self.movables] + [self.get_aabb(obj) for obj in self.statics]
        aabbs = torch.stack(aabbs)
        union_lower = aabbs[:, 0].min(dim=0).values
        union_upper = aabbs[:, 1].max(dim=0).values
        union_aabb = torch.stack([union_lower, union_upper])
        return union_aabb

    def warmup_ik_solver(self, num_particles: int):
        """Warmup cuRobo IK solver."""
        q = sample_between_bounds(num_particles, bounds=self.robot_container.joint_limits)
        goal_pose = self.kin_model.get_state(q).ee_pose
        _ = self.ik_solver.solve_batch(goal_pose)

    def get_motion_gen(self, collision_activation_distance: float, use_cuda_graph: bool = True) -> MotionGen:
        """
        Get the cuRobo motion generator for the robot. If you're debugging, you should set `use_cuda_graph=False`
        """
        if self.robot_name == "panda":
            robot_cfg = franka_curobo_cfg()
        elif self.robot_name == "panda_robotiq":
            robot_cfg = franka_curobo_cfg()
        elif self.robot_name == "fr3_robotiq":
            robot_cfg = fr3_robotiq_curobo_cfg()
        elif self.robot_name == "fr3_franka":
            robot_cfg = fr3_franka_curobo_cfg()
        elif self.robot_name == "ur5":
            robot_cfg = ur5_curobo_cfg()
        elif self.robot_name.startswith("bimanual_yam_"):
            robot_cfg = bimanual_yam_curobo_cfg(self.robot_name.rsplit("_", 1)[1])
        else:
            raise ValueError(f"Unsupported robot: {self.robot_name}")

        # Size EVERY attached-object slot to the largest object sphere set. A multi-arm config
        # declares one slot per hand (left_attached_object / right_attached_object), so writing only
        # the single-arm "attached_object" key would leave those at their declared placeholder size
        # and attach_objects_to_robot would fail trying to fit 50 spheres into 4.
        max_num_spheres = max([len(sphs) for sphs in self._obj_to_spheres.values()])
        extra_spheres = robot_cfg["robot_cfg"]["kinematics"]["extra_collision_spheres"]
        for link_name in extra_spheres:
            extra_spheres[link_name] = max_num_spheres
        _log.info(
            f"Setting number of spheres for attachments to {max_num_spheres} "
            f"on {sorted(extra_spheres)}"
        )

        # World config needs to include movables for cuRobo
        world_cfg = get_world_cfg(self.env, include_movables=True)
        motion_gen_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg=robot_cfg,
            world_model=world_cfg,
            use_cuda_graph=use_cuda_graph,
            collision_activation_distance=collision_activation_distance,
        )
        motion_gen = MotionGen(motion_gen_cfg)
        return motion_gen


def check_tamp_world_not_in_collision(
    world: TAMPWorld, collision_tol: float = 1e-6, movable_activation_dist: float = 0.0
):
    """Check that the initial state of the movable objects are not in collision."""
    for obj in world.movables:
        # Transform spheres to world frame
        mat4x4 = pose_list_to_mat4x4(obj.pose).to(world.device)
        spheres = transform_spheres(world.get_collision_spheres(obj), mat4x4)  # [n, 4]
        spheres = spheres[None, None].contiguous()  # [1, 1, n, 4]

        coll_cost = world.collision_fn(spheres).sum()
        if coll_cost > collision_tol:
            _log.warning(f"Initial state in collision for object '{obj.name}' with cost {coll_cost}")
            # raise ValueError(f"Initial state in collision for object '{obj.name}' with cost {coll_cost}")

    # Catch collisions between spheres for movable objects
    obj_to_spheres = {}
    for idx, obj in enumerate(world.movables):
        obj_spheres = transform_spheres(world.get_collision_spheres(obj), world.get_object_pose(obj))
        obj_to_spheres[obj.name] = obj_spheres

    for obj_1, obj_2 in itertools.combinations(world.movables, 2):
        obj_1_spheres = obj_to_spheres[obj_1.name]
        obj_2_spheres = obj_to_spheres[obj_2.name]
        coll_cost = sphere_to_sphere_overlap(
            obj_1_spheres,
            obj_2_spheres,
            activation_distance=movable_activation_dist,
            use_aabb_check=True,
        )
        if coll_cost > collision_tol:
            _log.warning(f"Initial state in collision between {obj_1.name} and {obj_2.name} with cost {coll_cost}")
            # raise ValueError(f"Initial state in collision between {obj_1.name} and {obj_2.name} with cost {coll_cost}")
