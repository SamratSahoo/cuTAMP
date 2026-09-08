"""Placement support regions fitted to a surface's observed point cloud.

The default placement model (``get_object_obb``) treats a surface as ONE oriented box: an object
may be placed anywhere inside its xy footprint, with its bottom at ``z_max`` -- the highest vertex
of the surface's convex hull. That is right for a slab and wrong for anything with structure:

  - An open box reconstructs as a hull spanning floor, walls and a folded-back lid. ``z_max`` is
    the top of the LID, so the release height is the lid's, and the footprint is the whole box --
    so a placement is happily "stable" a hand's width above the tray, out over a 2 mm flap. That is
    the 2026-09-07_13-39-40 bread/box failure: the bread was released at z = 0.196 over the box's
    far wall and fell.
  - A plate's ``z_max`` is its RIM, ~1.5 cm above the flat middle the object actually lands on, so
    every placement onto a plate is a drop from rim height.

This module fits the region an object can actually be SUPPORTED on. For every candidate centre it
asks where a flat-bottomed object of that footprint would come to REST -- the highest surface under
it -- and whether what it rests on reaches that height on all four sides, which is the support
polygon containing the object's centre. That admits a level patch and the inside of a bowl the object
bridges, and rejects a rim, a wall lip and the ridge of a folded-back lid. A surface where nothing
qualifies has no placement, which is the answer we want when the only thing big enough to hold the
object is not there.

The object is modelled by the disc it sweeps over all yaws wherever that fits, so a placement's
orientation stays free. Where it does not fit, the object's actual rectangle is tried at a committed
yaw, and the region then carries that yaw for the sampler and the placement cost to pin -- see
:class:`Footprint`.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from jaxtyping import Float

from curobo.geom.transform import matrix_to_quaternion
from curobo.geom.types import Cuboid, Obstacle
from curobo.types.base import TensorDeviceType
from cutamp.utils.obb import OrientedBoundingBox


class NoSupportRegion(RuntimeError):
    """No level patch of a surface is large enough to hold the object.

    Raised rather than quietly widened to the bounding box: the whole point of the support region is
    that a placement it cannot justify is one the object would fall off.
    """


@dataclass(frozen=True)
class SupportConfig:
    """Knobs for :func:`fit_support_region`. Defaults are tuned for tabletop stereo point clouds."""

    # Height-map cell size. Smaller resolves thin rims; larger bridges depth dropouts on its own.
    resolution: float = 0.005
    # Cells within this of a candidate plane's height belong to it. Covers stereo noise and the
    # slight tilt of a real "flat" surface, and must stay well under the height of the features we
    # are trying to tell apart (a box wall, a plate rim).
    flatness_tol: float = 0.008
    # Fraction of an object's footprint that must be over surface for a placement centre to count.
    # This is what holds the object clear of the EDGE of the surface -- off the edge reads as
    # unobserved -- and what rejects a placement over a hole. Single-cell depth dropouts are closed
    # over first, so it does not have to absorb stereo speckle.
    min_observed_frac: float = 0.9
    # Treat the unobserved cells ENCLOSED by a surface's own outline as floor rather than as absent
    # (see _fill_occluded). Off by default: it is the one thing here that lets an object be placed
    # over surface the robot never saw. Worth turning on for containers, whose interior a camera
    # looking across them cannot see -- measured on the 2026-09-07_14-56-40 box, only a 3.5 cm strip
    # of a ~10 cm tray floor came back, and the bread needs ~9 cm.
    fill_occluded: bool = False
    # With fill_occluded, the fraction of the footprint that must be GENUINELY observed. Stops a
    # placement resting entirely on assumed floor -- something has to have been seen under it.
    min_seen_frac: float = 0.25
    # Largest gap in a surface's outline that fill_occluded will bridge before deciding what is
    # "inside" it. Sized to the gaps a thin wall leaves in a point cloud, NOT to the object: bridge
    # by the object's own size and the closing swallows the whole bounding box, which hands back the
    # very region the support fit exists to replace.
    fill_bridge: float = 0.02
    # Clearance added to the object's footprint radius, so a placement keeps this much surface
    # around it. This is what "the surface is bigger than the object" buys us.
    margin: float = 0.005
    # Grid is capped at this many cells per side; resolution is coarsened to fit. Bounds the cost of
    # the rectangle search on a table-sized surface.
    max_cells_per_side: int = 320
    # And the footprint radius is capped at this many cells, coarsening the resolution the same way.
    # The stability test dilates by a disc of that radius eight times, which is O(cells x disc area)
    # -- unbounded, a big object on a big surface would spend seconds here.
    max_footprint_cells: int = 12
    # Candidate planes evaluated per surface, taken by descending cell count.
    max_candidate_planes: int = 24
    # Orientations tried over a full turn when the object's swept disc does not fit and it has to be
    # placed at a committed yaw. Eight is 45-degree steps; each one costs another pass of the max
    # filters, and they only run when the disc pass has already failed. Zero disables the committed
    # yaw entirely, leaving only placements that hold at every orientation.
    num_yaws: int = 8


@dataclass(frozen=True)
class SupportRegion:
    """Where an object of a given size may be placed on a surface, in the world frame.

    The rectangle holds valid object CENTRES, not surface area: it is already inset by the object's
    footprint, so a placement anywhere inside it rests on observed surface that reaches its bottom in
    every direction, at ``surface_z`` (+/- ``flatness_tol``). Downstream must not shrink it again.
    Contrast the surface's own bounding box, whose interior may be over a hole, a wall or thin air.
    """

    center: np.ndarray  # (3,) world-frame centre of the rectangle; z is surface_z
    half_extents: np.ndarray  # (2,) half x/y size of the rectangle, in its own frame
    yaw: float  # rotation of the rectangle's frame about world z
    surface_z: float  # world z of the patch -- where the object's BOTTOM goes
    area: float  # rectangle area (m^2), what candidate regions were ranked by
    observed_frac: float  # mean fraction of the footprint GENUINELY observed, over the region
    # World yaw the object must be placed at for this region to hold, or None when it may be placed
    # at any yaw. Set only where the object's own bounding disc did not fit but its rectangle did --
    # the region is then valid for THAT orientation (and its 180-degree twin) and no other, so the
    # sampler and the placement cost both have to pin the yaw. See fit_support_region.
    object_yaw: Optional[float] = None

    def to_obb(self, tensor_args: Optional[TensorDeviceType] = None) -> OrientedBoundingBox:
        """As an ``OrientedBoundingBox``, so it drops into the existing placement sampler and cost.

        The z half-extent is zero: this is a surface patch, not a volume, and nothing downstream
        reads it (the sampler and the stable-placement cost use ``half_extents[:2]`` and
        ``surface_z``).
        """
        tensor_args = tensor_args or TensorDeviceType()
        cos_a, sin_a = np.cos(self.yaw), np.sin(self.yaw)
        mat3x3 = tensor_args.to_device(
            np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        )
        return OrientedBoundingBox(
            center=tensor_args.to_device(self.center),
            half_extents=tensor_args.to_device([*self.half_extents, 0.0]),
            quat_wxyz=matrix_to_quaternion(mat3x3[None])[0],
            surface_z=self.surface_z,
        )


@dataclass(frozen=True)
class Footprint:
    """What an object occupies in xy, as bounds in its OWN frame about its OWN origin.

    Bounds rather than half-extents because the two are not the same thing here. A perception mesh is
    centred on its centroid, which is not the middle of its bounding box, so a loaf spanning
    x in [-3.5, 2.7] cm is 6.2 cm wide but needs 7.1 cm of centred rectangle to contain it. Placement
    positions the ORIGIN, so the offset is real and carrying it costs nothing.

    Both the rectangle and the DISC it sweeps matter. The disc is what has to fit when the placement
    yaw is free, which is the normal case and the one that leaves the planner every orientation to
    choose from. The rectangle is smaller -- that loaf sweeps a 9.1 cm disc about its own origin --
    and fitting it means committing to an orientation, which is the difference between placing the
    loaf in a 9 cm tray and not placing it at all. :func:`fit_support_region` tries the disc first
    and only commits to a yaw when the disc does not fit.
    """

    min_x: float
    max_x: float
    min_y: float
    max_y: float

    @property
    def radius(self) -> float:
        """Radius of the disc the rectangle sweeps about the origin -- its furthest corner."""
        return float(np.hypot(max(abs(self.min_x), self.max_x), max(abs(self.min_y), self.max_y)))

    @property
    def inradius(self) -> float:
        """Distance from the origin to the nearest edge."""
        return float(min(abs(self.min_x), self.max_x, abs(self.min_y), self.max_y))

    def grown(self, margin: float) -> "Footprint":
        return Footprint(self.min_x - margin, self.max_x + margin,
                         self.min_y - margin, self.max_y + margin)

    def union(self, other: "Footprint") -> "Footprint":
        """The footprint that contains both -- what a surface shared by two objects has to hold."""
        return Footprint(min(self.min_x, other.min_x), max(self.max_x, other.max_x),
                         min(self.min_y, other.min_y), max(self.max_y, other.max_y))

    @classmethod
    def centered(cls, half_x: float, half_y: float) -> "Footprint":
        return cls(-half_x, half_x, -half_y, half_y)

    @classmethod
    def disc(cls, radius: float) -> "Footprint":
        """A round footprint whose swept disc has this radius, for callers that only have one."""
        half = radius / np.sqrt(2.0)
        return cls.centered(half, half)


def footprint_from_spheres(obj_spheres: Float[torch.Tensor, "n 4"]) -> Footprint:
    """The object's xy bounds about its own origin, from its collision spheres.

    The fallback. Prefer :func:`footprint_from_object`: the spheres are sampled ON the surface, so
    each one bulges a radius past the geometry and the footprint comes out roughly two sphere
    radii too big in every dimension -- measured on a 6.2 x 4.3 cm loaf, 8.1 x 7.2 cm, which is what
    stopped it fitting a tray it fits easily.
    """
    x, y, r = obj_spheres[:, 0], obj_spheres[:, 1], obj_spheres[:, 3]
    return Footprint(float((x - r).min()), float((x + r).max()),
                     float((y - r).min()), float((y + r).max()))


def footprint_from_object(obj: Obstacle, obj_spheres: Float[torch.Tensor, "n 4"]) -> Footprint:
    """The object's xy bounds about its own origin, from its geometry where it has any.

    The geometry, not the collision-sphere approximation: the spheres are the planner's conservative
    stand-in for hitting things, and inflating an object by a sphere radius on every side to decide
    whether it FITS somewhere is a different question with a much worse answer.
    """
    if isinstance(obj, Cuboid):
        return Footprint.centered(obj.dims[0] / 2.0, obj.dims[1] / 2.0)
    vertices = getattr(obj, "vertices", None)
    if vertices is None:
        return footprint_from_spheres(obj_spheres)
    xy = np.asarray(vertices, dtype=np.float64)[:, :2]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    return Footprint(float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))


def _obb_frame(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Minimum-area rectangle over the xy points: (centre, world-from-local 2x2 rotation, yaw)."""
    rect = cv2.minAreaRect(points_xy.astype(np.float32))
    box_points = cv2.boxPoints(rect)
    edge0 = box_points[1] - box_points[0]
    yaw = float(np.arctan2(edge0[1], edge0[0]))
    cos_a, sin_a = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return np.asarray(rect[0], dtype=np.float64), rot, yaw


def _height_map(local_xy: np.ndarray, z: np.ndarray, resolution: float, max_cells: int):
    """Top-down max-z height map over the surface's own frame.

    Max rather than mean: a cell straddling the lip of a box wall is a cell the object would land on
    the wall in, and the height that matters there is the wall's.
    """
    lo = local_xy.min(axis=0)
    span = local_xy.max(axis=0) - lo
    # Coarsen rather than allocate an enormous grid for a table-sized surface.
    resolution = max(resolution, float(span.max()) / max_cells if max_cells > 0 else resolution)
    n_x = max(1, int(np.ceil(span[0] / resolution)))
    n_y = max(1, int(np.ceil(span[1] / resolution)))
    idx_x = np.clip(((local_xy[:, 0] - lo[0]) / resolution).astype(np.int64), 0, n_x - 1)
    idx_y = np.clip(((local_xy[:, 1] - lo[1]) / resolution).astype(np.int64), 0, n_y - 1)
    heights = np.full((n_y, n_x), -np.inf)
    np.maximum.at(heights, (idx_y, idx_x), z)
    return heights, lo, resolution


_VERY_LOW = -1e6  # stands in for "no surface here" under max filters; never wins a max


def _footprint_kernels(bounds_cells: "Footprint", yaw: float, num_sectors: int = 8):
    """A footprint structuring element and the sectors of its outer ring, as uint8 kernels.

    ``bounds_cells`` is the object's rectangle in CELLS, in its own frame about its own origin, and
    the kernel is that rectangle rotated by ``yaw`` and anchored at the origin -- so an object whose
    geometry sits off to one side of its origin is tested where it actually is.

    The sectors are what turn "something is under the object" into "the object does not tip": an
    object whose contacts reach its resting height in EVERY direction around its centre has that
    centre inside the convex hull of the contacts, which is the support-polygon condition for it
    staying put. A contact set confined to one side -- the lip of a box wall -- leaves the sectors
    on the other side short, and so does a ridge, whose sectors across the crest fall away even
    though the ones along it do not.

    The ring excludes the middle of the footprint: with the centre cell in every sector, an object
    balanced on a single high point directly beneath it satisfies all of them trivially. Support has
    to be found OUT from the centre for the contacts to bracket it. Four sectors are not enough
    either -- each of the four quadrants contains a direction ALONG a ridge, so a ridge passes.
    """
    reach = int(np.ceil(bounds_cells.radius))
    yy, xx = np.mgrid[-reach : reach + 1, -reach : reach + 1]
    cos_a, sin_a = np.cos(yaw), np.sin(yaw)
    # The grid offset from the candidate centre, read in the OBJECT's frame, to test against its own
    # axis-aligned bounds.
    obj_x = xx * cos_a + yy * sin_a
    obj_y = -xx * sin_a + yy * cos_a
    shape = (
        (obj_x >= bounds_cells.min_x) & (obj_x <= bounds_cells.max_x)
        & (obj_y >= bounds_cells.min_y) & (obj_y <= bounds_cells.max_y)
    )
    r2 = xx**2 + yy**2
    # The ring has to stay inside the rectangle all the way round the origin, so it is bounded by
    # the NEAREST edge -- otherwise a sector on the short side is empty and can never be reached
    # into, which would read as "unstable" everywhere.
    ring = shape & (r2 >= max(1, int(bounds_cells.inradius) // 2) ** 2)
    return shape.astype(np.uint8), _ring_sectors(ring, xx, yy, num_sectors)


def _disc_kernels(radius_cells: int, num_sectors: int = 8):
    """The yaw-free footprint: the disc the object's rectangle sweeps as it turns.

    Exactly a circle of the rectangle's half-diagonal -- that IS the union of the rectangle over
    every yaw, so a placement that fits here fits at any orientation and needs none pinned.
    """
    yy, xx = np.mgrid[-radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1]
    r2 = xx**2 + yy**2
    disc = r2 <= radius_cells**2
    ring = disc & (r2 >= max(1, radius_cells // 2) ** 2)
    return disc.astype(np.uint8), _ring_sectors(ring, xx, yy, num_sectors)


def _ring_sectors(ring, xx, yy, num_sectors: int):
    """Split a footprint's outer ring into ``num_sectors`` angular wedges."""
    angle = np.arctan2(yy, xx) % (2 * np.pi)
    width = 2 * np.pi / num_sectors
    sectors = []
    for i in range(num_sectors):
        # Centred on i * width, so the sector straddles the axis rather than starting on it.
        offset = (angle - i * width + np.pi) % (2 * np.pi) - np.pi
        sectors.append((ring & (np.abs(offset) <= width / 2)).astype(np.uint8))
    return sectors


_VERY_HIGH = 1e6  # stands in for "no surface here" under min filters; never wins a min


def _fill_occluded(heights, observed, bridge_cells: int, window_cells: int):
    """Treat the unobserved cells inside a surface's own outline as floor. -> (heights, filled).

    A camera looking ACROSS a container cannot see its floor: the near wall hides the near half and
    a raised lid hides the far half, so the tray comes back as a narrow strip of returns with a hole
    where the rest of the floor is. That hole is occlusion, not absence -- the floor is there, it was
    simply not visible from the capture pose -- and read as absence it makes every placement in the
    container rest on the lid instead.

    "Inside the outline" is the observed mask closed by ``bridge_cells`` -- enough to seal the gaps a
    thin wall leaves in a point cloud, and no more -- and then hole-filled, so only cells the surface
    encloses are filled and a concavity open to the outside is left alone. The height assigned is the
    LOWEST surface observed within ``window_cells``: near a tray that is its floor, and it is the
    conservative choice for support, since a filled cell can then never raise where an object comes
    to rest -- only give it something to rest ON.

    What this cannot do is see an object hiding in the occluded region. ``min_seen_frac`` is the
    guard: some real fraction of every footprint has to be surface that was actually observed.
    """
    bridge = np.ones((2 * bridge_cells + 1,) * 2, np.uint8)
    closed = cv2.morphologyEx(observed.astype(np.uint8), cv2.MORPH_CLOSE, bridge)

    # Hole-fill: pad by one so the flood always starts OUTSIDE the silhouette, whatever sits in the
    # corner, then keep what the flood could not reach.
    padded = cv2.copyMakeBorder(closed, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(padded, np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8), (0, 0), 1)
    silhouette = closed.astype(bool) | (padded[1:-1, 1:-1] == 0)

    # Lowest observed height within a footprint of each cell (erode is a min filter). This window is
    # the object's, not the bridge's: the floor an object rests on is the floor under all of it.
    low = cv2.erode(
        np.where(observed, heights, _VERY_HIGH).astype(np.float32),
        np.ones((2 * window_cells + 1,) * 2, np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=_VERY_HIGH,
    )
    filled = silhouette & ~observed & (low < _VERY_HIGH / 2)
    return np.where(filled, low, heights), filled


def _resting_heights(heights, supported, observed, kernels, flatness_tol: float):
    """Where a flat-bottomed object of the given footprint radius would come to rest, per cell.

    ``supported`` is where there is surface to rest on and ``observed`` where it was actually seen.
    They differ only when :func:`_fill_occluded` has added a container's occluded floor. ``kernels``
    is the (footprint, sectors) pair from :func:`_disc_kernels` or :func:`_footprint_kernels`.

    For each candidate centre:

    ``resting``  the height the object's bottom stops at -- the HIGHEST surface anywhere under its
                 footprint, because that is what it lands on first.
    ``stable``   whether the surface it lands on reaches that height in every direction around the
                 footprint (see :func:`_disc_kernels`). True for a level patch, true for the inside
                 of a bowl the object bridges, false on a rim, a wall lip, a slope or a ridge.
    ``covered``  the fraction of the footprint with surface under it at all.
    ``seen``     the fraction of it that was genuinely observed rather than filled in.

    Everything is a max- or box-filter over a disc, so the whole grid is done in a handful of OpenCV
    calls rather than per candidate. Outside the grid counts as absent -- which is what keeps
    placements away from the edge of the surface without a separate margin test.
    """
    disc, sectors = kernels

    surface = np.where(supported, heights, _VERY_LOW).astype(np.float32)
    resting = cv2.dilate(surface, disc, borderType=cv2.BORDER_CONSTANT, borderValue=_VERY_LOW)
    stable = np.ones(heights.shape, dtype=bool)
    for sector in sectors:
        reach = cv2.dilate(surface, sector, borderType=cv2.BORDER_CONSTANT, borderValue=_VERY_LOW)
        stable &= reach >= resting - flatness_tol

    # Both fractions are measured with single-cell holes closed, so the speckle dropouts stereo
    # leaves all over a low-texture surface do not read as missing support. A real gap -- a hole in
    # the surface, or the footprint simply hanging off its edge, where the border pads with zeros --
    # survives the closing and shows up here.
    disc_area = float(disc.sum())
    weights = disc.astype(np.float32) / disc_area

    def _fraction(mask):
        solid = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return cv2.filter2D(
            solid.astype(np.float32), -1, weights, borderType=cv2.BORDER_CONSTANT
        )

    covered = _fraction(supported)
    seen = covered if supported is observed else _fraction(observed)
    # A centre with nothing at all under it rests nowhere.
    stable &= resting > _VERY_LOW / 2
    return resting, stable, covered, seen


def _largest_rectangle(mask: np.ndarray, min_rows: int, min_cols: int):
    """Largest all-True axis-aligned rectangle with at least ``min_rows`` x ``min_cols`` cells.

    Standard largest-rectangle-in-histogram sweep. It enumerates every MAXIMAL rectangle, and any
    rectangle meeting the minimum is contained in a maximal one whose sides are at least as long, so
    filtering the maximal ones on the minimum loses nothing.

    Returns ``(rows, cols, row0, col0)`` of the best rectangle, or None.
    """
    n_rows, n_cols = mask.shape
    if min_rows > n_rows or min_cols > n_cols:
        return None

    best = None
    best_area = 0
    run = np.zeros(n_cols, dtype=np.int64)  # consecutive True cells ending at this row, per column
    for row in range(n_rows):
        run = np.where(mask[row], run + 1, 0)
        stack: list[tuple[int, int]] = []  # (start column, bar height)
        for col in range(n_cols + 1):
            height = int(run[col]) if col < n_cols else 0
            start = col
            while stack and stack[-1][1] >= height:
                bar_start, bar_height = stack.pop()
                width = col - bar_start
                if bar_height >= min_rows and width >= min_cols and bar_height * width > best_area:
                    best_area = bar_height * width
                    best = (bar_height, width, row - bar_height + 1, bar_start)
                start = bar_start
            stack.append((start, height))
    return best


def fit_support_region(
    points: np.ndarray,
    footprint: Footprint | float,
    config: SupportConfig = SupportConfig(),
) -> Optional[SupportRegion]:
    """Where an object of this ``footprint`` can be supported on ``points``.

    Tries the object's bounding DISC first, which fits at every yaw and so leaves the placement's
    orientation free -- the normal case, and the one that keeps the planner's choice of pose wide.
    Only if the disc does not fit anywhere does it commit to an orientation and try the object's
    actual RECTANGLE at each of ``num_yaws`` yaws, returning the best along with the yaw it needs.
    That second pass is what places a 6.6 x 4.7 cm loaf in a 9 cm tray: the loaf fits lying along the
    tray, and its 8.8 cm bounding disc does not fit at all.

    Args:
        points: (N, 3) world-frame points of the SURFACE, as observed. These must be the raw
            observations: a convex hull has no concavity left to find, and a point cloud with the
            near-table band filtered out has lost the floor of every shallow container.
        footprint: the object's half-extents about its own origin (see
            :func:`footprint_from_spheres`). A bare float is read as a disc of that radius.
        config: see :class:`SupportConfig`.

    Returns:
        The region of valid object CENTRES (already inset by the footprint -- do not shrink it
        again), or None when nowhere on the surface would hold the object, for which the right
        response is to reject the placement rather than fall back on the bounding box.
    """
    if not isinstance(footprint, Footprint):
        footprint = Footprint.disc(float(footprint))
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return None

    grown = footprint.grown(config.margin)
    center_xy, rot, surface_yaw = _obb_frame(points[:, :2])
    local_xy = (points[:, :2] - center_xy) @ rot  # world -> local
    resolution = max(config.resolution, grown.radius / max(1, config.max_footprint_cells))
    heights, lo, resolution = _height_map(local_xy, points[:, 2], resolution, config.max_cells_per_side)
    observed = np.isfinite(heights)
    if not observed.any():
        return None

    supported = observed
    if config.fill_occluded:
        # A container's floor is occluded, not absent, from a camera looking across it. Filling it
        # is what lets an object be placed INSIDE one; min_seen_frac below is what stops the
        # placement from resting entirely on the part that was filled in.
        heights, filled = _fill_occluded(
            heights,
            observed,
            max(1, int(round(config.fill_bridge / resolution))),
            max(1, int(round(grown.radius / resolution))),
        )
        supported = observed | filled

    def _fit(kernels, object_yaw):
        resting, stable, covered, seen = _resting_heights(
            heights, supported, observed, kernels, config.flatness_tol
        )
        valid = stable & (covered >= config.min_observed_frac) & (seen >= config.min_seen_frac)
        if not valid.any():
            return None
        return _best_band(
            valid, resting, covered, seen, lo, resolution, center_xy, rot, surface_yaw, object_yaw, config
        )

    radius_cells = max(1, int(round(grown.radius / resolution)))
    region = _fit(_disc_kernels(radius_cells), None)
    # A free-yaw region is worth having, but not at any size: one that is smaller than the object
    # itself is a placement the optimizer has no room to move in, and the committed-yaw pass usually
    # finds far more space. So try it too whenever the free one is missing or cramped, and keep the
    # free one unless a pinned one is genuinely bigger.
    object_area = (grown.max_x - grown.min_x) * (grown.max_y - grown.min_y)
    if region is not None and region.area >= object_area:
        return region

    in_cells = Footprint(grown.min_x / resolution, grown.max_x / resolution,
                         grown.min_y / resolution, grown.max_y / resolution)
    best = region
    for i in range(config.num_yaws):
        # A full turn, not a half: the rectangle is offset from the origin, so turning it by pi moves
        # it as well as rotating it and the two are not the same placement. (For a centred footprint
        # they are, and the second half of the sweep simply reproduces the first.)
        local_yaw = 2 * np.pi * i / config.num_yaws
        candidate = _fit(_footprint_kernels(in_cells, local_yaw), float(local_yaw + surface_yaw))
        if candidate is not None and (best is None or candidate.area > best.area):
            best = candidate
    return best


def _best_band(valid, resting, covered, seen, lo, resolution, center_xy, rot, surface_yaw,
               object_yaw, config) -> Optional[SupportRegion]:
    """Largest rectangle of valid centres, among those that share a resting height.

    Grouping by resting height first is what stops one region spanning a bowl from its floor to its
    rim: inside a band the object's bottom lands within 2 * flatness_tol of ``surface_z`` wherever
    in the region it goes.
    """
    valid_heights = resting[valid]
    bin_width = config.flatness_tol
    bins = np.floor((valid_heights - valid_heights.min()) / bin_width).astype(np.int64)
    counts = np.bincount(bins)
    order = np.argsort(counts)[::-1][: config.max_candidate_planes]
    bands = [valid_heights.min() + (b + 0.5) * bin_width for b in order if counts[b] > 0]
    bands += [b + 0.5 * bin_width for b in bands]

    best: Optional[SupportRegion] = None
    for band_z in bands:
        in_band = valid & (np.abs(resting - band_z) <= config.flatness_tol)
        found = _largest_rectangle(in_band, 1, 1)
        if found is None:
            continue
        rows, cols, row0, col0 = found

        area = rows * cols * resolution * resolution
        window = (slice(row0, row0 + rows), slice(col0, col0 + cols))
        # The HIGHEST resting height anywhere in the region: release the object from there and it is
        # never dropped from more than the band's own spread, wherever inside the region it goes.
        surface_z = float(resting[window].max())
        # Rank by area, and among equals prefer the LOWER region -- the inside of a container over
        # its rim, which is both the shorter drop and what "in the box" means.
        if best is not None and (area, -surface_z) <= (best.area, -best.surface_z):
            continue

        half_extents = np.array([cols * resolution / 2.0, rows * resolution / 2.0])
        local_center = np.array([lo[0] + (col0 + cols / 2.0) * resolution, lo[1] + (row0 + rows / 2.0) * resolution])
        world_center_xy = rot @ local_center + center_xy
        best = SupportRegion(
            center=np.array([world_center_xy[0], world_center_xy[1], surface_z]),
            half_extents=half_extents,
            yaw=surface_yaw,
            surface_z=surface_z,
            area=area,
            observed_frac=float(seen[window].mean()),
            object_yaw=object_yaw,
        )
    return best
