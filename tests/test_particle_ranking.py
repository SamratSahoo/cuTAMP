"""Tests for how satisfying particles are ranked for motion refinement.

This ranking decides the plan that is actually executed -- the caller keeps the first particle
cuRobo can motion plan -- so what it does and does not consider is the difference between a soft
cost shaping the robot's behaviour and merely being logged.
"""

import types

import pytest
import torch

from cutamp.algorithm import get_ranked_satisfying_particles
from cutamp.cost_reduction import CostReducer


class _StubChecker:
    """Stands in for ConstraintChecker: marks every particle satisfying."""

    def get_mask(self, cost_dict, verbose: bool = False):
        return torch.ones(4, dtype=torch.bool)


def _plan_info(offsets, confidences):
    """Four particles differing only in grasp offset (metres) and M2T2 confidence."""
    offsets = torch.tensor(offsets)
    confidences = torch.tensor(confidences)
    particles = {
        "grasp1": torch.arange(4, dtype=torch.float32),
        "grasp1_confidences": confidences,
    }
    cost_dict = {
        "GraspCost": {"type": "cost", "values": {"grasp_center_offset": offsets}},
    }
    return {
        "particles": particles,
        "rollout_fn": lambda p: {},
        "cost_fn": lambda rollout: cost_dict,
    }


def _rank(offsets, confidences, conf_weight):
    config = types.SimpleNamespace(num_particles=4, grasp_rank_conf_weight=conf_weight)
    reducer = CostReducer({"GraspCost": {"grasp_center_offset": 30.0}})
    ranked = get_ranked_satisfying_particles(
        _plan_info(offsets, confidences), config, _StubChecker(), reducer
    )
    return [int(i) for i in ranked["grasp1"].tolist()]


# Particle 0 is the M2T2 favourite but sits 10cm out at the tip; particle 3 is nearly centred with
# clearly lower confidence; 1 and 2 are in between. These are the real magnitudes: banana grasp
# offsets span ~1-13cm and M2T2 confidences on one object's candidates span ~0.05-0.20.
OFFSETS = [0.100, 0.070, 0.040, 0.010]
CONFS = [0.20, 0.15, 0.10, 0.06]


def test_default_ranks_on_confidence_alone():
    """Historical behaviour: the most-confident grasp wins however far off-centre it is."""
    assert _rank(OFFSETS, CONFS, conf_weight=None) == [0, 1, 2, 3]


def test_zero_weight_ranks_on_soft_cost_alone():
    assert _rank(OFFSETS, CONFS, conf_weight=0.0) == [3, 2, 1, 0]


def test_confidence_weight_trades_centering_against_confidence():
    """At w=5 the 9cm of centering is worth far more than the 0.14 of confidence given up."""
    assert _rank(OFFSETS, CONFS, conf_weight=5.0) == [3, 2, 1, 0]


def test_large_confidence_weight_restores_the_confidence_order():
    """The knob spans both regimes, so a caller can dial back rather than choose between two modes."""
    assert _rank(OFFSETS, CONFS, conf_weight=1000.0) == [0, 1, 2, 3]


def test_confidence_breaks_ties_between_equally_centred_grasps():
    """Confidence stays in the score: it is the only evidence the grasp is on real geometry."""
    assert _rank([0.05] * 4, [0.06, 0.10, 0.15, 0.20], conf_weight=5.0) == [3, 2, 1, 0]


def test_diagnostics_record_the_executed_ranking():
    """The per-rank numbers `log_executed_particle` reports are recorded in ranked order."""
    plan_info = _plan_info(OFFSETS, CONFS)
    config = types.SimpleNamespace(num_particles=4, grasp_rank_conf_weight=5.0)
    reducer = CostReducer({"GraspCost": {"grasp_center_offset": 30.0}})
    get_ranked_satisfying_particles(plan_info, config, _StubChecker(), reducer)

    diagnostics = plan_info["ranked_diagnostics"]
    assert diagnostics["particle_idx"] == [3, 2, 1, 0]
    assert diagnostics["grasp_rank_conf_weight"] == 5.0
    # Rank 0 is the nearly-centred grasp: 0.01m * 30 - 5.0 * 0.06.
    assert diagnostics["soft_cost"][0] == pytest.approx(0.3, abs=1e-5)
    assert diagnostics["rank_score"][0] == pytest.approx(0.0, abs=1e-5)
    assert diagnostics["rank_score"] == sorted(diagnostics["rank_score"])
