"""A zero-candidate object must not quietly become a collision-sphere guess.

With ``m2t2_grasps`` on, an object perception proposed NOTHING for is not dropped: ``_sample_grasps``
falls through to the 4-/6-DOF heuristic sampler, which draws grasps from the object's COLLISION
SPHERES rather than from perception. Measured over shipped runs' scene_objects.json, 18-43% of picked
objects took that path with no trace but a _log.debug, and it is a direct mechanism for closing on
empty air (analysis_dataset_diff/TELEOP_VS_APEX.md, Finding 5).
"""

from __future__ import annotations

import logging

import pytest

from cutamp.config import TAMPConfiguration
from cutamp.particle_initialization import NoGraspsError, ParticleInitializer


class _Cfg:
    """Only the fields _sample_grasps reads before it would branch."""

    def __init__(self, require: bool):
        self.m2t2_grasps = True
        self.grasp_dof = 4
        self.require_m2t2_grasps = require


class _Init:
    """ParticleInitializer with just enough state to reach the guard."""

    _sample_grasps = ParticleInitializer._sample_grasps

    def __init__(self, grasps, require=False):
        self.grasps = grasps
        self.config = _Cfg(require)
        self.world = None  # only reached AFTER the guard, so a raise never touches it


def test_default_is_off_so_existing_configs_are_unchanged():
    assert TAMPConfiguration(robot="fr3_robotiq").require_m2t2_grasps is False


def test_raises_when_required_and_no_candidates():
    init = _Init({"pink_toy": {"grasps_obj": []}}, require=True)
    with pytest.raises(NoGraspsError) as exc:
        init._sample_grasps("pink_toy", 8)
    assert "pink_toy" in str(exc.value)
    assert "collision spheres" in str(exc.value)


def test_raises_when_the_object_is_absent_entirely():
    """An object missing from the grasp dict is zero candidates, not a KeyError."""
    init = _Init({"blue_toy": {"grasps_obj": [object()]}}, require=True)
    with pytest.raises(NoGraspsError):
        init._sample_grasps("pink_toy", 8)


def test_no_grasps_at_all_is_zero_candidates():
    init = _Init(None, require=True)
    with pytest.raises(NoGraspsError):
        init._sample_grasps("pink_toy", 8)


def test_warns_but_proceeds_when_not_required(caplog):
    """Default path: the substitution still happens, but it is no longer silent."""
    init = _Init({"pink_toy": {"grasps_obj": []}}, require=False)
    with caplog.at_level(logging.WARNING, logger="cutamp.particle_initialization"):
        # Falls through to the heuristic sampler, which needs a real world -- so it fails LATER,
        # on world access, not at the guard. That it gets past the guard is the assertion.
        with pytest.raises(Exception) as exc:
            init._sample_grasps("pink_toy", 8)
        assert not isinstance(exc.value, NoGraspsError)
    assert any("HEURISTIC" in r.message or "HEURISTIC" in r.getMessage() for r in caplog.records)


def test_object_with_candidates_is_untouched(caplog):
    """The guard must not fire, and must not warn, when perception did propose grasps."""
    init = _Init({"pink_toy": {"grasps_obj": [object(), object()]}}, require=True)
    with caplog.at_level(logging.WARNING, logger="cutamp.particle_initialization"):
        with pytest.raises(Exception) as exc:
            init._sample_grasps("pink_toy", 8)  # proceeds into the M2T2 branch, then needs a world
        assert not isinstance(exc.value, NoGraspsError)
    assert not [r for r in caplog.records if "HEURISTIC" in r.getMessage()]
