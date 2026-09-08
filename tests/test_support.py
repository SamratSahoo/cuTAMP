"""Tests for the placement support region (cutamp.utils.support).

The cases are the shapes the bounding-box placement region gets wrong: an open container, whose
bounding box top is the rim of its wall rather than its floor, and a surface too small for the object
it is asked to hold.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch

from curobo.geom.types import Cuboid, Mesh

from cutamp.utils.support import (
    Footprint,
    SupportConfig,
    _largest_rectangle,
    fit_support_region,
    footprint_from_object,
    footprint_from_spheres,
)


def _bowl_points(rim_z=0.06, radius=0.09, step=0.003):
    """A hemispherical bowl: no level patch anywhere, but an object still rests in it stably."""
    xs = np.arange(-radius, radius, step)
    grid_x, grid_y = np.meshgrid(xs, xs)
    r2 = grid_x**2 + grid_y**2
    inside = r2 <= radius**2
    # Paraboloid from the bowl's floor up to the rim.
    z = rim_z * r2[inside] / radius**2
    return np.stack([grid_x[inside], grid_y[inside], z], axis=1)


def _grid(x_range, y_range, z, step=0.002):
    """A dense level patch of points at height ``z``."""
    xs = np.arange(x_range[0], x_range[1], step)
    ys = np.arange(y_range[0], y_range[1], step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, z)], axis=1)


def open_box_points(floor_z=0.002, wall_z=0.05, half_x=0.15, half_y=0.06, wall=0.004):
    """A shallow open box: a floor, and four thin walls standing above it.

    This is the shape that fails under the bounding-box region -- its highest points are the wall
    tops, so an object is released at ``wall_z`` anywhere over the footprint, including over the
    open middle.
    """
    floor = _grid((-half_x, half_x), (-half_y, half_y), floor_z)
    walls = [
        _grid((-half_x, -half_x + wall), (-half_y, half_y), wall_z),
        _grid((half_x - wall, half_x), (-half_y, half_y), wall_z),
        _grid((-half_x, half_x), (-half_y, -half_y + wall), wall_z),
        _grid((-half_x, half_x), (half_y - wall, half_y), wall_z),
    ]
    return np.concatenate([floor, *walls])


class TestLargestRectangle:
    def test_finds_the_biggest_rectangle(self):
        mask = np.zeros((6, 8), dtype=bool)
        mask[1:5, 2:7] = True  # 4 rows x 5 cols
        assert _largest_rectangle(mask, 1, 1) == (4, 5, 1, 2)

    def test_respects_the_minimum_side_lengths(self):
        # A wide-but-thin bar and a smaller square. The bar has more area but is only 1 row tall.
        mask = np.zeros((6, 10), dtype=bool)
        mask[0, :] = True
        mask[2:5, 0:3] = True
        assert _largest_rectangle(mask, 1, 1) == (1, 10, 0, 0)
        assert _largest_rectangle(mask, 3, 3) == (3, 3, 2, 0)

    def test_none_when_nothing_fits(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        assert _largest_rectangle(mask, 3, 3) is None
        assert _largest_rectangle(mask, 9, 9) is None


class TestFitSupportRegion:
    def test_flat_slab_is_the_slab_inset_by_the_footprint(self):
        points = _grid((-0.1, 0.1), (-0.1, 0.1), 0.02)
        region = fit_support_region(points, 0.02, config=SupportConfig(margin=0.005))
        assert region is not None
        assert region.surface_z == pytest.approx(0.02, abs=1e-3)
        # The region holds valid object CENTRES, so it is the 20 cm square inset by the 2.5 cm
        # footprint-plus-margin on each side.
        assert region.half_extents == pytest.approx([0.075, 0.075], abs=0.01)
        assert region.center[:2] == pytest.approx([0.0, 0.0], abs=0.01)

    def test_open_box_supports_on_the_floor_not_the_walls(self):
        points = open_box_points()
        region = fit_support_region(points, 0.02)
        assert region is not None
        # The bounding-box region would put the object's bottom at the top of the wall.
        assert points[:, 2].max() == pytest.approx(0.05)
        assert region.surface_z == pytest.approx(0.002, abs=0.005)
        # And every centre in it clears the walls by the footprint.
        assert abs(region.center[0]) + region.half_extents[0] <= 0.15 - 0.02
        assert abs(region.center[1]) + region.half_extents[1] <= 0.06 - 0.02

    def test_a_bowl_is_supported_even_though_nothing_in_it_is_level(self):
        # The case a "largest level patch" rule gets wrong: a bowl has no flat area at all, but an
        # object lowered into it is held on all sides and cannot roll out.
        points = _bowl_points()
        region = fit_support_region(points, 0.03)
        assert region is not None
        assert region.center[:2] == pytest.approx([0.0, 0.0], abs=0.02)
        # It rests down in the bowl, well below the rim it would be dropped from otherwise.
        assert region.surface_z < 0.5 * points[:, 2].max()

    def test_a_slope_is_not_supported(self):
        # A ramp rising 20 cm over 30 cm -- the shape a folded-back box lid reconstructs as. Every
        # contact under the object is downhill of its centre, so it slides; nothing here qualifies.
        xs = np.arange(-0.15, 0.15, 0.003)
        grid_x, grid_y = np.meshgrid(xs, np.arange(-0.10, 0.10, 0.003))
        points = np.stack([grid_x.ravel(), grid_y.ravel(), (grid_x.ravel() + 0.15) * 0.66], axis=1)
        assert fit_support_region(points, 0.03) is None

    def test_rejects_an_object_too_big_for_the_floor(self):
        # The box's floor is 30 x 12 cm, so a 20 cm-radius footprint cannot fit inside it -- and
        # neither can it fit on the walls. There is no answer, and None says so rather than falling
        # back on the footprint of the whole box.
        assert fit_support_region(open_box_points(), 0.20) is None

    def test_margin_is_enforced(self):
        # A 10 x 10 cm patch holds a 6 x 6 cm object with 1 cm of clearance all round; asking for
        # 2.5 cm wants 11 cm of surface, which is more than there is.
        points = _grid((-0.05, 0.05), (-0.05, 0.05), 0.0)
        obj = Footprint.centered(0.03, 0.03)
        assert fit_support_region(points, obj, SupportConfig(margin=0.01)) is not None
        assert fit_support_region(points, obj, SupportConfig(margin=0.025)) is None

    def test_prefers_the_larger_patch_over_the_higher_one(self):
        # A big low table with a small block on it: the object goes on the table, because the block
        # is not big enough to hold it -- the opposite of "highest point wins".
        table = _grid((-0.2, 0.2), (-0.2, 0.2), 0.0)
        block = _grid((0.15, 0.19), (0.15, 0.19), 0.06)
        region = fit_support_region(np.concatenate([table, block]), 0.03)
        assert region is not None
        assert region.surface_z == pytest.approx(0.0, abs=0.005)

    def test_a_yawed_surface_keeps_its_own_frame(self):
        yaw = np.deg2rad(30.0)
        points = _grid((-0.15, 0.15), (-0.05, 0.05), 0.01)
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        points[:, :2] = points[:, :2] @ rot.T
        region = fit_support_region(points, 0.02)
        assert region is not None
        # The fitted frame is the surface's own, up to the rectangle's 90-degree symmetry.
        assert np.isclose((region.yaw - yaw) % (np.pi / 2), 0.0, atol=0.05) or np.isclose(
            (region.yaw - yaw) % (np.pi / 2), np.pi / 2, atol=0.05
        )
        # Inset by the 2.5 cm footprint-plus-margin on each side of the 30 x 10 cm patch.
        assert sorted(region.half_extents) == pytest.approx([0.025, 0.125], abs=0.015)

    def test_too_few_points(self):
        assert fit_support_region(np.zeros((2, 3)), 0.01) is None

    def test_non_finite_points_are_dropped(self):
        points = _grid((-0.1, 0.1), (-0.1, 0.1), 0.02)
        points = np.concatenate([points, np.full((5, 3), np.nan)])
        assert fit_support_region(points, 0.02) is not None

    def test_a_mostly_unobserved_patch_is_rejected(self):
        # A ring of floor points around a large hole, as an occluded container interior looks. The
        # closing must not invent a floor across the hole.
        outer = _grid((-0.15, 0.15), (-0.10, 0.10), 0.0)
        hole = (np.abs(outer[:, 0]) < 0.10) & (np.abs(outer[:, 1]) < 0.06)
        region = fit_support_region(outer[~hole], 0.05)
        assert region is None

    def test_to_obb_round_trips_the_region(self):
        region = fit_support_region(_grid((-0.1, 0.1), (-0.06, 0.06), 0.03), 0.01)
        assert region is not None
        obb = region.to_obb()
        assert obb.surface_z == pytest.approx(region.surface_z)
        assert obb.center.cpu().numpy() == pytest.approx(region.center, abs=1e-5)
        assert obb.half_extents[:2].cpu().numpy() == pytest.approx(region.half_extents, abs=1e-5)
        # Local -> world through the OBB's rotation agrees with the region's own yaw.
        local = np.array([[region.half_extents[0], 0.0, 0.0]], dtype=np.float32)
        world = local @ obb.rot_matrix.cpu().numpy().T + region.center
        expected = region.center + region.half_extents[0] * np.array(
            [np.cos(region.yaw), np.sin(region.yaw), 0.0]
        )
        assert world[0] == pytest.approx(expected, abs=1e-4)


def occluded_tray_points(floor_z=0.0, wall_z=0.05, half_x=0.15, half_y=0.09, wall=0.004, seen=0.03):
    """An open tray whose floor is only observed in a strip, as a camera looking ACROSS it sees it.

    The near wall hides the near half of the floor and the far wall the far half, so the returns are
    the four wall tops plus one band of floor -- which is what the 2026-09-07_14-56-40 box looked
    like, and why every placement in it "rested" on the wall.
    """
    walls = [
        _grid((-half_x, -half_x + wall), (-half_y, half_y), wall_z),
        _grid((half_x - wall, half_x), (-half_y, half_y), wall_z),
        _grid((-half_x, half_x), (-half_y, -half_y + wall), wall_z),
        _grid((-half_x, half_x), (half_y - wall, half_y), wall_z),
    ]
    strip = _grid((-half_x + wall, half_x - wall), (0.0, seen), floor_z)
    return np.concatenate([strip, *walls])


class TestFillOccluded:
    def test_an_occluded_tray_floor_has_no_region_without_the_fill(self):
        points = occluded_tray_points()
        assert fit_support_region(points, 0.03, SupportConfig(fill_occluded=False)) is None

    def test_filling_recovers_the_tray_floor(self):
        points = occluded_tray_points()
        region = fit_support_region(points, 0.03, SupportConfig(fill_occluded=True))
        assert region is not None
        # On the floor, not on the wall tops the bounding box would have used.
        assert region.surface_z == pytest.approx(0.0, abs=0.01)
        assert points[:, 2].max() == pytest.approx(0.05)
        # And it reports honestly how much of the footprint was really seen.
        assert 0.25 <= region.observed_frac < 1.0

    def test_min_seen_frac_rejects_a_placement_resting_only_on_filled_floor(self):
        points = occluded_tray_points(seen=0.008)  # barely any floor observed
        permissive = fit_support_region(points, 0.03, SupportConfig(fill_occluded=True, min_seen_frac=0.0))
        strict = fit_support_region(points, 0.03, SupportConfig(fill_occluded=True, min_seen_frac=0.9))
        assert permissive is not None  # the fill alone would place here
        assert strict is None  # but almost nothing under the object was actually seen

    def test_the_fill_does_not_invent_surface_outside_the_outline(self):
        # A plain slab: there is nothing enclosed, so filling changes nothing at all.
        points = _grid((-0.1, 0.1), (-0.1, 0.1), 0.02)
        plain = fit_support_region(points, 0.02, SupportConfig(fill_occluded=False))
        filled = fit_support_region(points, 0.02, SupportConfig(fill_occluded=True))
        assert plain is not None and filled is not None
        assert filled.half_extents == pytest.approx(plain.half_extents, abs=1e-9)
        assert filled.surface_z == pytest.approx(plain.surface_z)

    def test_the_fill_reaches_only_as_far_as_the_floor_it_can_see(self):
        """A filled cell takes the lowest height observed within the OBJECT's own window.

        So the fill extends a floor across a gap the object could span, and stops in the middle of
        one it could not -- there is no observation there to take a height from. That, and not the
        bridge, is what keeps it from paving over a hole the object would drop through.
        """
        def slab_with_hole(hole):
            slab = _grid((-0.16, 0.16), (-0.16, 0.16), 0.0)
            bite = (np.abs(slab[:, 0]) < hole / 2) & (np.abs(slab[:, 1]) < hole / 2)
            return slab[~bite]

        config = SupportConfig(fill_occluded=True, margin=0.005)
        # A 3 cm gap sits well inside the 5 cm footprint window, so the floor carries across it and
        # the slab is whole again.
        small = fit_support_region(slab_with_hole(0.03), 0.02, config)
        assert small is not None
        assert small.area == pytest.approx(
            fit_support_region(_grid((-0.16, 0.16), (-0.16, 0.16), 0.0), 0.02, config).area, rel=0.1
        )
        # A 16 cm gap does not: its middle is out of reach of any observation, stays unsupported,
        # and the region falls back to the ring of real floor around it.
        big = fit_support_region(slab_with_hole(0.16), 0.02, config)
        assert big is None or big.area < 0.5 * small.area


class TestCommittedYaw:
    """The second pass: an object that only fits a surface at one orientation."""

    @staticmethod
    def _narrow_tray(width=0.09, length=0.40, wall_z=0.06, wall=0.004):
        """A long shallow tray, narrower than a loaf's bounding disc but wider than the loaf."""
        floor = _grid((-length / 2, length / 2), (-width / 2, width / 2), 0.0, step=0.002)
        walls = [
            _grid((-length / 2, length / 2), (-width / 2 - wall, -width / 2), wall_z, step=0.002),
            _grid((-length / 2, length / 2), (width / 2, width / 2 + wall), wall_z, step=0.002),
        ]
        return np.concatenate([floor, *walls])

    def test_a_loaf_fits_a_tray_its_bounding_disc_does_not(self):
        points = self._narrow_tray()
        loaf = Footprint.centered(0.033, 0.023)  # 6.6 x 4.6 cm; swept disc 8.0 cm across
        config = SupportConfig(margin=0.005)

        # Held to orientations that work at EVERY yaw (num_yaws=0), the loaf has to fit as the 9.0
        # cm disc it sweeps, against a 9 cm tray -- no room.
        assert fit_support_region(points, loaf, replace(config, num_yaws=0)) is None
        # Allowed to commit to one, it lies along the tray and fits with centimetres to spare.
        region = fit_support_region(points, loaf, config)
        assert region is not None
        assert region.surface_z == pytest.approx(0.0, abs=0.005)
        # And the region says which way round: along the tray, which runs down the surface's x axis.
        assert region.object_yaw is not None
        assert np.isclose((region.object_yaw - region.yaw) % np.pi, 0.0, atol=0.4) or np.isclose(
            (region.object_yaw - region.yaw) % np.pi, np.pi, atol=0.4
        )

    def test_a_cramped_free_region_loses_to_a_roomy_pinned_one(self):
        """A free-yaw region is preferred, but not when it is a sliver.

        The disc pass can just barely succeed on a surface the object fits comfortably one way
        round. Returning that would hand the optimizer a region smaller than the object itself to
        move in, when committing to an orientation opens up far more.
        """
        loaf = Footprint.centered(0.033, 0.023)
        config = SupportConfig(margin=0.005)
        # Wide enough that the swept disc fits, barely; the loaf lying along it fits with room.
        points = self._narrow_tray(width=0.105)
        free = fit_support_region(points, loaf, replace(config, num_yaws=0))
        assert free is not None and free.object_yaw is None
        best = fit_support_region(points, loaf, config)
        assert best is not None
        assert best.area > free.area
        assert best.object_yaw is not None

    def test_yaw_is_left_free_when_the_disc_already_fits(self):
        # The same loaf on an open table: no reason to pin anything, so the planner keeps every
        # orientation to choose from.
        region = fit_support_region(_grid((-0.2, 0.2), (-0.2, 0.2), 0.0), Footprint.centered(0.033, 0.023))
        assert region is not None
        assert region.object_yaw is None

    def test_a_square_object_that_does_not_fit_stays_unplaced(self):
        # Turning a centred square does not make it smaller, so the yaw pass has nothing to find.
        assert fit_support_region(self._narrow_tray(width=0.03), Footprint.centered(0.03, 0.029)) is None

    def test_an_off_centre_footprint_is_tested_where_it_actually_is(self):
        """An object whose geometry sits to one side of its origin sweeps a disc much bigger than
        itself, so the yaw pass earns its keep even for a compact shape."""
        offset = Footprint(-0.045, 0.015, -0.020, 0.020)  # 6 x 4 cm, origin 1.5 cm off centre
        assert offset.radius == pytest.approx(np.hypot(0.045, 0.020))
        region = fit_support_region(self._narrow_tray(width=0.09), offset, SupportConfig(margin=0.005))
        assert region is not None
        assert region.object_yaw is not None

    def test_a_disc_footprint_reports_the_radius_it_was_built_from(self):
        assert Footprint.disc(0.05).radius == pytest.approx(0.05)
        assert Footprint.centered(0.03, 0.02).radius == pytest.approx(np.hypot(0.03, 0.02))
        assert Footprint.centered(0.03, 0.02).inradius == pytest.approx(0.02)


class TestFootprintKernels:
    def test_the_disc_kernel_is_the_rectangles_swept_union(self):
        from cutamp.utils.support import _disc_kernels, _footprint_kernels

        disc, _ = _disc_kernels(10)
        # Every yaw of a rectangle with half-diagonal 10 lies inside the disc of radius 10.
        for yaw in np.linspace(0.0, 2 * np.pi, 9):
            shape, _ = _footprint_kernels(Footprint.centered(8.0, 6.0), yaw)  # half-diagonal 10
            pad = (disc.shape[0] - shape.shape[0]) // 2
            padded = np.pad(shape, pad)
            assert not (padded.astype(bool) & ~disc.astype(bool)).any()

    def test_every_sector_of_a_rectangle_has_cells(self):
        from cutamp.utils.support import _footprint_kernels

        # An empty sector would make the stability test vacuously fail (nothing can reach into it).
        for bounds in (
            Footprint.centered(8.0, 6.0),
            Footprint.centered(12.0, 4.0),
            Footprint.centered(5.0, 5.0),
            Footprint(-14.0, 4.0, -6.0, 6.0),  # origin well off to one side
        ):
            _, sectors = _footprint_kernels(bounds, 0.3)
            assert all(s.sum() > 0 for s in sectors), bounds


class TestFootprintFromObject:
    """Where the footprint comes from. The 2026-09-07_16-01-00 box failed on this alone."""

    @staticmethod
    def _spheres(mesh_bounds, radius=0.008, n=400):
        """Collision spheres as cuTAMP samples them: centres ON the surface, so each bulges by r."""
        (min_x, max_x, min_y, max_y) = mesh_bounds
        rng = np.random.default_rng(0)
        x = rng.choice([min_x, max_x], n) + rng.uniform(-1e-4, 1e-4, n)
        y = rng.uniform(min_y, max_y, n)
        pts = np.stack([np.r_[x, rng.uniform(min_x, max_x, n)],
                        np.r_[y, rng.choice([min_y, max_y], n)],
                        np.zeros(2 * n)], axis=1)
        return torch.as_tensor(np.c_[pts, np.full(2 * n, radius)], dtype=torch.float32)

    def test_the_spheres_overstate_the_object_by_a_sphere_radius_each_side(self):
        bounds = (-0.035, 0.027, -0.015, 0.028)  # the 6.2 x 4.3 cm loaf, off its own centroid
        from_spheres = footprint_from_spheres(self._spheres(bounds))
        assert from_spheres.max_x - from_spheres.min_x == pytest.approx(0.062 + 0.016, abs=2e-3)
        assert from_spheres.max_y - from_spheres.min_y == pytest.approx(0.043 + 0.016, abs=2e-3)

    def test_a_mesh_object_is_measured_from_its_vertices(self):
        bounds = (-0.035, 0.027, -0.015, 0.028)
        mesh = Mesh(
            name="loaf",
            vertices=[[x, y, z] for x in bounds[:2] for y in bounds[2:] for z in (-0.02, 0.02)],
            faces=[[0, 1, 2]],
            pose=[0.5, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0],
        )
        footprint = footprint_from_object(mesh, self._spheres(bounds))
        assert (footprint.min_x, footprint.max_x) == pytest.approx(bounds[:2])
        assert (footprint.min_y, footprint.max_y) == pytest.approx(bounds[2:])
        # And the offset is kept, so the swept disc is the real one -- measuring half-extents about
        # the centroid instead would have called this 7.1 x 5.6 cm.
        assert footprint.radius == pytest.approx(np.hypot(0.035, 0.028))

    def test_a_cuboid_is_measured_from_its_dims(self):
        cuboid = Cuboid(name="b", dims=[0.06, 0.04, 0.05], pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        footprint = footprint_from_object(cuboid, self._spheres((-0.03, 0.03, -0.02, 0.02)))
        assert footprint.max_x == pytest.approx(0.03)
        assert footprint.max_y == pytest.approx(0.02)

    def test_the_sphere_footprint_can_cost_a_placement_the_object_would_fit(self):
        """The regression: a loaf that fits a tray, refused because its SPHERES do not."""
        bounds = (-0.035, 0.027, -0.015, 0.028)  # a 6.2 x 4.3 cm loaf
        # 7.5 cm of tray: room for the loaf lying along it with a centimetre either side, and not
        # for the 5.9 cm its spheres claim it is.
        tray = TestCommittedYaw._narrow_tray(width=0.075)
        config = SupportConfig(margin=0.01)
        assert fit_support_region(tray, footprint_from_spheres(self._spheres(bounds)), config) is None
        mesh_footprint = Footprint(*bounds)
        assert fit_support_region(tray, mesh_footprint, config) is not None
