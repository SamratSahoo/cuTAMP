"""Tests for cutamp.cost_function."""

import os

import numpy as np
import pytest
import torch

from cutamp.algorithm import run_cutamp
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs.utils import get_env_dir, load_env
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def _run_blocks_activation_test(mask: bool) -> int:
    """Run planning on the blocks_activation_dist_test env and return num satisfying particles."""
    env = load_env(os.path.join(get_env_dir(), "blocks_activation_dist_test.yml"))
    config = TAMPConfiguration(
        num_particles=512,
        robot="fr3_robotiq",
        num_opt_steps=500,
        max_loop_dur=20.0,
        enable_visualizer=False,
        rr_spawn=False,
        enable_experiment_logging=False,
        world_activation_distance=0.0,
        mask_initial_movable_world_collision=mask,
    )
    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())
    _, num_satisfying, _ = run_cutamp(env, config, cost_reducer, constraint_checker)
    return num_satisfying


@gpu
def test_movable_to_world_masking():
    """Masking initial movable-to-world collisions should find satisfying particles when blocks
    slightly penetrate the floor (simulating perception noise), while disabling masking should not."""
    assert _run_blocks_activation_test(mask=False) == 0
    assert _run_blocks_activation_test(mask=True) > 0


class _FakeGraspCost:
    """Stand-in for a GraspCost instance; only ``params[1]`` (the grasp parameter name) is read."""

    def __init__(self, obj: str, grasp: str):
        self.params = (obj, grasp)


def _grasp_cost_host(grasp_names: list, **flags):
    """A CostFunction with just the attributes the grasp soft costs read.

    Built without __init__ so the terms can be exercised on CPU, without a TAMPWorld or a solve.
    """
    from cutamp.cost_function import CostFunction

    host = object.__new__(CostFunction)
    host.grasp_costs = [_FakeGraspCost(f"obj{i}", g) for i, g in enumerate(grasp_names)]
    host.grasp_cost_action_names = list(grasp_names)
    host.config = TAMPConfiguration(**flags)
    return host


def test_grasp_center_offset_is_horizontal_distance_from_the_object_origin():
    """grasp_center_offset charges ||xy|| of obj_from_grasp's translation, in meters, ignoring z."""
    host = _grasp_cost_host(["g0"], grasp_center_cost=True)
    mats = torch.eye(4).repeat(3, 1, 1)
    mats[0, :3, 3] = torch.tensor([0.00, 0.00, 0.05])  # dead center; the 5cm of z must not be charged
    mats[1, :3, 3] = torch.tensor([0.03, 0.04, 0.00])  # 5cm off-center horizontally (3-4-5)
    mats[2, :3, 3] = torch.tensor([-0.12, 0.00, 0.02])  # 12cm off, out at an edge
    out = host.grasp_soft_costs({"grasp_to_obj_from_grasp": {"g0": mats}})

    assert out["type"] == "cost" and out["costs"] is host.grasp_costs
    assert set(out["values"]) == {"grasp_center_offset"}  # orientation term stays off when not gated
    offsets = out["values"]["grasp_center_offset"]
    assert offsets.shape == (3, 1)
    assert torch.allclose(offsets.squeeze(1), torch.tensor([0.0, 0.05, 0.12]), atol=1e-6)


def test_grasp_center_offset_covers_every_grasp_in_the_skeleton():
    """One column per grasp parameter, in the order the GraspCosts were collected."""
    host = _grasp_cost_host(["g0", "g1"], grasp_center_cost=True)
    g0, g1 = torch.eye(4).repeat(2, 1, 1), torch.eye(4).repeat(2, 1, 1)
    g0[:, 0, 3] = torch.tensor([0.01, 0.02])
    g1[:, 1, 3] = torch.tensor([0.03, 0.04])
    offsets = host.grasp_soft_costs({"grasp_to_obj_from_grasp": {"g0": g0, "g1": g1}})["values"][
        "grasp_center_offset"
    ]
    assert offsets.shape == (2, 2)
    assert torch.allclose(offsets, torch.tensor([[0.01, 0.03], [0.02, 0.04]]), atol=1e-6)


def test_grasp_soft_costs_emit_nothing_unless_opted_in():
    """The gate matters: CostReducer charges an ABSENT multiplier at weight 1.0, so a value emitted
    without opt-in would silently change every caller's objective."""
    rollout = {"grasp_to_obj_from_grasp": {"g0": torch.eye(4).repeat(2, 1, 1)}}
    assert _grasp_cost_host(["g0"]).grasp_soft_costs(rollout) is None
    assert _grasp_cost_host(["g0"]).grasp_center_costs(rollout) is None
    # ...and a skeleton with no grasps at all emits nothing even when the flag is on.
    assert _grasp_cost_host([], grasp_center_cost=True).grasp_soft_costs(rollout) is None


class _FakePlacement:
    """Stand-in for a StablePlacement constraint: params are (obj, grasp, placement, surface)."""

    def __init__(self, obj: str, surface: str):
        self.params = (obj, "grasp1", "pose1", surface)


class _FakeWorld:
    def __init__(self, device):
        self.device = device


def _placement_host(config, obb, target_z=0.0):
    """A CostFunction with just the attributes stable_placement_costs reads, on CPU."""
    from cutamp.cost_function import CostFunction

    host = object.__new__(CostFunction)
    host.config = config
    host.world = _FakeWorld(obb.center.device)
    host.stable_placement_constraints = [_FakePlacement("block", "table")]
    host.surface_to_aabb = {}
    host.surface_to_obb = {"table": obb}
    host.surface_to_yaw = {"table": None}
    host.surface_to_target_z = {"table": target_z}
    return host


def _placement_rollout(origin_xy, device, num_particles=1, half=0.04, yaw=0.0):
    """A rollout placing `block` with its origin at ``origin_xy`` and spheres out to ``half``."""
    origins = torch.tensor(
        [[float(origin_xy[0]), float(origin_xy[1]), 0.0]], device=device
    ).repeat(num_particles, 1)
    pose = torch.eye(4, device=device).repeat(num_particles, 1, 1)
    pose[:, :3, 3] = origins
    cos_a, sin_a = float(np.cos(yaw)), float(np.sin(yaw))
    pose[:, :2, :2] = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=device)
    # Four spheres at the corners of the object's footprint, plus one at its origin.
    offsets = torch.tensor(
        [[0.0, 0.0], [half, half], [-half, half], [half, -half], [-half, -half]], device=device
    )
    spheres = torch.zeros(num_particles, 1, len(offsets), 4, device=device)
    spheres[..., :2] = origins[:, None, None, :2] + offsets
    spheres[..., 3] = 0.005
    rollout = {
        "num_particles": num_particles,
        "action_to_pose_ts": {"pose1": 0},
        "obj_to_pose": {"block": pose[:, None]},
    }
    return rollout, {"block": spheres}


def _placement_values(config, obb, origin_xy, yaw=0.0, target_yaw=None):
    host = _placement_host(config, obb)
    host.surface_to_yaw = {"table": target_yaw}
    rollout, obj_to_spheres = _placement_rollout(origin_xy, obb.center.device, yaw=yaw)
    return host.stable_placement_costs(rollout, obj_to_spheres)["values"]


def _xy_cost(config, obb, origin_xy):
    return float(_placement_values(config, obb, origin_xy)["table_in_xy"][0, 0])


def _region(half_x=0.05, half_y=0.05):
    from cutamp.utils.support import SupportRegion

    import numpy as np

    return SupportRegion(
        center=np.array([0.0, 0.0, 0.0]),
        half_extents=np.array([half_x, half_y]),
        yaw=0.0,
        surface_z=0.0,
        area=4 * half_x * half_y,
        observed_frac=1.0,
    ).to_obb()


def test_support_placement_scores_the_object_origin_not_every_sphere():
    """A support region is a region of valid object ORIGINS.

    Charging each collision sphere against it instead asks the whole object to fit inside a region
    that is already the surface minus that object -- which no placement can satisfy, and which took
    the constraint from satisfiable to permanently violated for every particle.
    """
    config = TAMPConfiguration(placement_check="support")
    obb = _region(0.05, 0.05)  # 10 x 10 cm of valid centres
    # The object is 8 cm across, so its corner spheres sit outside the region even when its origin
    # is dead centre. Scoring the origin, that is a perfect placement.
    assert _xy_cost(config, obb, (0.0, 0.0)) == pytest.approx(0.0)
    # Moving the origin out of the region does cost, by the distance it left by.
    assert _xy_cost(config, obb, (0.07, 0.0)) == pytest.approx(0.02, abs=1e-4)


def test_obb_placement_still_scores_every_sphere():
    """The bounding-box region is a region of surface AREA, so the object has to fit in it."""
    config = TAMPConfiguration(placement_check="obb")
    obb = _region(0.05, 0.05)
    # Origin centred: the four corner spheres at +/-4 cm are inside 5 cm, so nothing is charged.
    assert _xy_cost(config, obb, (0.0, 0.0)) == pytest.approx(0.0)
    # Shift by 2 cm and the two leading corners hang 1 cm over the edge each.
    assert _xy_cost(config, obb, (0.02, 0.0)) == pytest.approx(0.02, abs=1e-4)


def test_a_pinned_region_charges_the_placement_yaw():
    """A support region fitted at one orientation has to keep the placement there.

    Yaw is an optimized placement parameter, so without this term the optimizer is free to turn the
    object out of the pose the region was fitted for -- straight over the edge the fit was chosen to
    clear. |sin| is zero at the pinned yaw and at its half-turn twin, which is the symmetry a
    rectangle actually has.
    """
    config = TAMPConfiguration(placement_check="support")
    obb = _region()
    target = 0.4

    aligned = _placement_values(config, obb, (0.0, 0.0), yaw=target, target_yaw=target)
    assert float(aligned["table_yaw"][0, 0]) == pytest.approx(0.0, abs=1e-6)

    flipped = _placement_values(config, obb, (0.0, 0.0), yaw=target + np.pi, target_yaw=target)
    assert float(flipped["table_yaw"][0, 0]) == pytest.approx(0.0, abs=1e-6)

    # Small errors read as radians, so a tolerance in radians means what it looks like.
    off = _placement_values(config, obb, (0.0, 0.0), yaw=target + 0.05, target_yaw=target)
    assert float(off["table_yaw"][0, 0]) == pytest.approx(0.05, abs=1e-3)

    # A quarter turn is as wrong as it gets.
    square = _placement_values(config, obb, (0.0, 0.0), yaw=target + np.pi / 2, target_yaw=target)
    assert float(square["table_yaw"][0, 0]) == pytest.approx(1.0, abs=1e-6)


def test_an_unpinned_region_emits_no_yaw_term():
    """The normal case: the region holds the object at any yaw, so nothing constrains it."""
    values = _placement_values(TAMPConfiguration(placement_check="support"), _region(), (0.0, 0.0))
    assert "table_yaw" not in values
    assert "table_in_xy" in values
