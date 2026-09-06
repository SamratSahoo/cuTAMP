"""Data-derived posture prior for IK branch selection.

A redundant arm reaches the same end-effector pose with a continuum of joint configurations.
cuRobo ranks its IK seeds by ``pose_error + null_space_error`` with ``null_space_cfg.weight`` at
0.001 against a generic home retract, so with the default ``return_seeds=1`` the branch is picked
by pose error alone -- and the arm's redundancy lands wherever, independently at every endpoint.
The result is a planner that visits joint configurations human teleoperation never does.

Given k IK branches for ONE pose, the principled choice is ``argmax_q p(q | pose)``. On the
manifold of configurations reaching that pose the normaliser is identical for every branch, so this
is just ``argmax p(q)`` -- a plain density over human configurations. That is what the primary model
is: a full-covariance Gaussian mixture fit to the reference corpus (K chosen by held-out
log-likelihood; evaluation here is pure torch, sklearn is only needed to bake).

    penalty(q) = -log p_GMM(q)

The earlier PAIRWISE model is kept as a fallback for references baked without a mixture:

    penalty(q) = sum_{i<j} w_ij * [ relu(d_ij - hi_ij) + relu(lo_ij - d_ij) ],  d_ij = q_i - q_j

with ``lo/hi`` the 5th/95th percentile of ``q_i - q_j`` over the corpus and
``w_ij = E[|| N(q) (e_i - e_j)/sqrt(2) ||^2]``, ``N = I - J^+ J`` the Jacobian null-space projector
-- the fraction of that pair's direction that is self-motion, so pairs that would fight the pose
goal contribute ~nothing.

Measured on real IK branches against a NON-PARAMETRIC held-out yardstick (5-NN distance to held-out
DROID episodes, so no mixture appears in the metric): stock 1.11, pairwise 0.92, **GMM 0.85**,
held-out DROID itself 0.19. The mixture also has no structural blind spot -- the pairwise tables are
provably invariant to adding a constant to every joint, since every difference is unchanged.

The reference is one self-contained artifact, ``posture_ref.npz`` next to this file, baked by
``cutamp/scripts/bake_posture_ref.py``. Override with the ``CUTAMP_POSTURE_REF`` env var.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

DEFAULT_POSTURE_REF = os.environ.get(
    "CUTAMP_POSTURE_REF", str(Path(__file__).resolve().parent / "posture_ref.npz")
)

_CACHE: dict = {}


def load_posture_prior(ref_path: str, device, dtype):
    """Load (and cache) the baked prior as tensors. Frozen, no grad."""
    key = (ref_path, str(device), str(dtype))
    if key in _CACHE:
        return _CACHE[key]
    blob = np.load(ref_path, allow_pickle=True)
    if "n_joints" not in blob:
        raise KeyError(f"{ref_path} has no n_joints; rebuild with cutamp/scripts/bake_posture_ref.py")

    def _t(x):
        return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)

    pack = {
        "n_joints": int(blob["n_joints"]),
        "provenance": str(blob["provenance"]) if "provenance" in blob else "",
        "model": str(blob["model"]) if "model" in blob else "pairwise",
    }
    if pack["model"] == "gmm":
        if "gmm_weights" not in blob:
            raise KeyError(f"{ref_path} says model=gmm but has no mixture; re-bake it")
        prec = _t(blob["gmm_prec_chol"])                    # [K, J, J]
        pack.update(
            g_mean=_t(blob["gmm_means"]),                   # [K, J]  (standardised space)
            g_prec=prec,
            g_mu=_t(blob["gmm_mu"]), g_scale=_t(blob["gmm_scale"]),
            # log w_k + sum(log diag(P_k)) folded into one constant per component
            g_const=_t(np.log(np.asarray(blob["gmm_weights"])))
            + torch.log(torch.diagonal(prec, dim1=-2, dim2=-1)).sum(-1),
        )
    else:
        for k in ("pair_lo", "pair_hi", "pair_weight"):
            if k not in blob:
                raise KeyError(f"{ref_path} has no {k}; rebuild with bake_posture_ref.py")
        pack.update(lo=_t(blob["pair_lo"]), hi=_t(blob["pair_hi"]), w=_t(blob["pair_weight"]))
    _CACHE[key] = pack
    return pack


def posture_penalty(q: torch.Tensor, pack: dict) -> torch.Tensor:
    """Posture penalty for each configuration. ``q``: [..., dof] -> [...]. Lower is more human.

    GMM model: ``-log p(q)``, up to the constant ``0.5 * J * log(2*pi)`` which is dropped because it
    is identical for every branch and cannot change an argmin.
    Pairwise model: weighted out-of-band distance summed over joint pairs, 0 when every pair
    difference sits inside the human range.
    """
    J = pack["n_joints"]
    x = q[..., :J]
    if pack["model"] == "gmm":
        z = (x - pack["g_mu"]) / pack["g_scale"]                       # [..., J]
        # y_k = (z - mean_k) @ P_k, then log N_k = const_k - 0.5 ||y_k||^2   (sklearn's form)
        y = torch.einsum("...j,kjm->...km", z, pack["g_prec"]) - torch.einsum(
            "kj,kjm->km", pack["g_mean"], pack["g_prec"]
        )
        log_comp = pack["g_const"] - 0.5 * (y * y).sum(-1)             # [..., K]
        return -torch.logsumexp(log_comp, dim=-1)
    d = x.unsqueeze(-1) - x.unsqueeze(-2)                              # [..., J, J]  d[i,j] = q_i - q_j
    out = (d - pack["hi"]).clamp(min=0.0) + (pack["lo"] - d).clamp(min=0.0)
    # w is zero on the diagonal, and (i,j)/(j,i) are counted twice -- symmetric, so a constant
    # factor of 2 on every term, which cannot change the ranking.
    return (out * pack["w"]).flatten(-2).sum(-1)


def posture_ref_summary(ref_path: Optional[str] = None, top: int = 4) -> str:
    """Human-readable summary of a baked reference, for logging."""
    pack = load_posture_prior(ref_path or DEFAULT_POSTURE_REF, "cpu", torch.float32)
    if pack["model"] == "gmm":
        return f"{pack['provenance']} | model=gmm K={pack['g_mean'].shape[0]} J={pack['n_joints']}"
    w = pack["w"].numpy()
    lo, hi = pack["lo"].numpy(), pack["hi"].numpy()
    iu = np.triu_indices(pack["n_joints"], 1)
    rows = ", ".join(
        f"q{iu[0][k]+1}-q{iu[1][k]+1} w={w[iu[0][k], iu[1][k]]:.2f} "
        f"[{lo[iu[0][k], iu[1][k]]:+.2f},{hi[iu[0][k], iu[1][k]]:+.2f}]"
        for k in np.argsort(w[iu])[::-1][:top]
    )
    return f"{pack['provenance']} | model=pairwise | top pairs: {rows}"
