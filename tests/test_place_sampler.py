"""Tests for place_4dof_sampler's handling of a fitted support region."""

import numpy as np
import pytest
import torch

from curobo.geom.types import Cuboid
from cutamp.samplers import _YAW_PIN_JITTER, place_4dof_sampler
from cutamp.utils.support import SupportRegion


def _region(object_yaw=None, half_extents=(0.05, 0.03), yaw=0.3):
    return SupportRegion(
        center=np.array([0.5, 0.0, 0.02]),
        half_extents=np.array(half_extents),
        yaw=yaw,
        surface_z=0.02,
        area=4 * half_extents[0] * half_extents[1],
        observed_frac=1.0,
        object_yaw=object_yaw,
    )


def _sample(region, num=512, sphere_radius=0.02):
    obb = region.to_obb()
    surface = Cuboid(name="s", dims=[0.4, 0.4, 0.02], pose=[0.5, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0])
    obj = Cuboid(name="o", dims=[0.06, 0.04, 0.05], pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    spheres = torch.tensor([[0.0, 0.0, 0.0, sphere_radius]], device=obb.center.device)
    return place_4dof_sampler(
        num, obj, spheres, surface, "support", None, 0.0, obb=obb, object_yaw=region.object_yaw
    )


def test_a_pinned_region_samples_only_its_own_orientation():
    target = 1.1
    placements = _sample(_region(object_yaw=target))
    # |sin| is the same measure the placement cost charges, and every particle starts well inside it.
    assert float(torch.abs(torch.sin(placements[:, 3] - target)).max()) <= _YAW_PIN_JITTER + 1e-6


def test_both_half_turns_of_a_pinned_orientation_are_offered():
    """A rectangle covers the same ground either way up, and the two differ for reach and collision."""
    target = 1.1
    yaw = _sample(_region(object_yaw=target))[:, 3]
    assert ((yaw - target).abs() < 0.5).any()
    assert ((yaw - target - np.pi).abs() < 0.5).any()


def test_an_unpinned_region_samples_yaw_freely():
    yaw = _sample(_region())[:, 3]
    assert float(yaw.max() - yaw.min()) > 6.0  # the full turn, as before


def test_placements_land_inside_the_region_without_a_second_inset():
    """The region is already the set of valid object CENTRES, so the sampler must not shrink it."""
    region = _region(half_extents=(0.05, 0.03))
    placements = _sample(region, sphere_radius=0.02)
    obb = region.to_obb()
    local = (placements[:, :3] - obb.center) @ obb.rot_matrix
    assert float(local[:, 0].abs().max()) <= region.half_extents[0] + 1e-6
    assert float(local[:, 1].abs().max()) <= region.half_extents[1] + 1e-6
    # And they use the region, rather than collapsing to its middle.
    assert float(local[:, 0].abs().max()) > 0.8 * region.half_extents[0]


def test_the_object_bottom_lands_on_the_regions_surface():
    region = _region()
    placements = _sample(region, sphere_radius=0.02)
    # z is the object's origin; its lowest sphere sits 0.02 below, at the support height.
    bottoms = placements[:, 2] - 0.02
    assert float(bottoms.min()) >= region.surface_z
    assert float(bottoms.max()) <= region.surface_z + 0.02


def test_support_without_a_region_is_refused():
    surface = Cuboid(name="s", dims=[0.4, 0.4, 0.02], pose=[0.5, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0])
    obj = Cuboid(name="o", dims=[0.06, 0.04, 0.05], pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    spheres = torch.tensor([[0.0, 0.0, 0.0, 0.02]])
    with pytest.raises(ValueError, match="needs the fitted region"):
        place_4dof_sampler(8, obj, spheres, surface, "support", None, 0.0)
