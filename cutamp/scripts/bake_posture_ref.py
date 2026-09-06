"""Bake the posture prior (cutamp/posture_ref.npz) from a human-teleop corpus.

Self-contained: corpus loading, the Jacobian, and the table build all live in this file, so
regenerating the prior needs nothing but cuTAMP, numpy, and the corpus on disk.

Produces the two N x N tables that ``cutamp/posture_prior.py`` consumes:

    pair_lo[i, j], pair_hi[i, j]   percentile band of (q_i - q_j) over the corpus
    pair_weight[i, j]              E[|| N(q) (e_i - e_j)/sqrt(2) ||^2],  N = I - J^+ J

Nothing is hand-picked: WHICH joints matter falls out of ``pair_weight`` (a pair whose direction
moves the end-effector scores ~0 and is ignored), and HOW MUCH they may vary falls out of the
bands. Re-bake against a different robot or corpus and the whole prior moves with it.

The Jacobian is finite-differenced through cuRobo's own forward kinematics rather than a hardcoded
DH table, so this works for any robot cuTAMP can build an IK solver for.

Usage (from the tamp-vla root, in the tiptop pixi env):

    # DROID proprio shard cache (the default; what the shipped posture_ref.npz was baked from)
    python -m cutamp.scripts.bake_posture_ref

    # a LeRobot dataset directory instead
    python -m cutamp.scripts.bake_posture_ref \\
        --source lerobot --path ~/.cache/huggingface/lerobot/<user>/<repo> --robot fr3_franka
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

#: DROID and this lab's LeRobot datasets are all recorded at 15 Hz.
FPS = 15.0
#: Total joint speed (rad/s) above which a frame counts as moving. Human corpora contain long
#: stationary stretches; including them would concentrate the bands on wherever the arm was parked.
NONIDLE_SPEED = 0.05

DEFAULT_DROID_CACHE = Path("/home/prpl/tamp-vla/vae/data_cache/droid_full_proprio")


# --------------------------------------------------------------------------------------------- #
# corpus loading                                                                                 #
# --------------------------------------------------------------------------------------------- #
def _nonidle(q: np.ndarray) -> np.ndarray:
    """Frames whose total joint speed exceeds NONIDLE_SPEED, by central difference at FPS."""
    qd = np.gradient(q, 1.0 / FPS, axis=0, edge_order=1)
    return np.linalg.norm(qd, axis=1) > NONIDLE_SPEED


def load_droid_shards(cache_dir, n_joints, max_episodes=None, min_len=30):
    """Episodes from the DROID proprio shard cache (shard_*.npz with joints/episode_index)."""
    shards = sorted(Path(cache_dir).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shard_*.npz under {cache_dir}")
    out = []
    for sp in shards:
        z = np.load(sp, allow_pickle=True)
        for arr in z["joints"]:
            q = np.asarray(arr, dtype=np.float64)
            if q.ndim != 2 or q.shape[1] < n_joints or q.shape[0] < min_len:
                continue
            out.append(q[:, :n_joints])
            if max_episodes is not None and len(out) >= max_episodes:
                return out
    return out


def load_lerobot(root, n_joints, max_episodes=None, min_len=30):
    """Episodes from a LeRobot dataset directory (data/**/*.parquet + meta/info.json)."""
    import pyarrow.parquet as pq

    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info.get("fps", FPS))
    if abs(fps - FPS) > 1e-6:
        raise ValueError(f"{root} is {fps} Hz, not {FPS}; resampling would be required")
    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no data parquet under {root}")
    out = []
    for f in files:
        d = pq.read_table(
            f, columns=["observation.state.joint_position", "episode_index", "frame_index"]
        ).to_pydict()
        qs = d["observation.state.joint_position"]
        eidx = np.asarray(d["episode_index"])
        fidx = np.asarray(d["frame_index"])
        for e in np.unique(eidx):
            m = np.flatnonzero(eidx == e)
            m = m[np.argsort(fidx[m])]
            if len(m) < min_len:
                continue
            out.append(np.asarray([qs[i] for i in m], dtype=np.float64)[:, :n_joints])
            if max_episodes is not None and len(out) >= max_episodes:
                return out
    return out


# --------------------------------------------------------------------------------------------- #
# kinematics: finite-difference Jacobian through cuRobo FK (robot-agnostic)                       #
# --------------------------------------------------------------------------------------------- #
def make_jacobian_fn(robot: str, chunk: int = 512):
    """Return jac(q[N, dof]) -> J[N, 6, dof], finite-differenced through the robot's own FK.

    cuRobo's CudaRobotModel refuses to output an analytic Jacobian ("Outputting jacobian is not
    supported"), and hardcoding a DH table would tie this script to one arm -- so the Jacobian is
    central-differenced from FK instead. h=1e-4 rad is far above FP32 FK noise and far below the
    curvature scale, and the null-space projector below only needs a few digits.
    """
    import torch
    from curobo.geom.types import Cuboid, WorldConfig

    from cutamp.robots.franka import get_franka_ik_solver, get_fr3_franka_ik_solver
    from cutamp.robots.franka_robotiq import get_fr3_robotiq_ik_solver

    # A tiny far-away obstacle: cuRobo refuses an empty collision world, and we only want FK here.
    world = WorldConfig(cuboid=[Cuboid(name="far", pose=[5.0, 5.0, 5.0, 1, 0, 0, 0], dims=[0.01] * 3)])
    builders = {
        "fr3_franka": get_fr3_franka_ik_solver,
        "fr3_robotiq": get_fr3_robotiq_ik_solver,
        "panda": get_franka_ik_solver,
    }
    if robot not in builders:
        raise ValueError(f"unknown robot {robot!r}; known: {sorted(builders)}")
    ik = builders[robot](world)
    dev = "cuda"
    h = 1e-4

    def _mats(q):
        out = []
        for s in range(0, len(q), chunk):
            out.append(ik.fk(q[s : s + chunk]).ee_pose.get_matrix())
        return torch.cat(out, 0)

    def jac(q_np):
        q = torch.as_tensor(np.asarray(q_np), dtype=torch.float32, device=dev)
        n, dof = q.shape
        J = torch.zeros(n, 6, dof, device=dev)
        for i in range(dof):
            e = torch.zeros(dof, device=dev)
            e[i] = h
            Tp, Tm = _mats(q + e), _mats(q - e)
            J[:, :3, i] = (Tp[:, :3, 3] - Tm[:, :3, 3]) / (2 * h)
            # angular part: log(R+ R-^T) / 2h, via the skew-symmetric part of the rotation delta
            dR = Tp[:, :3, :3] @ Tm[:, :3, :3].transpose(1, 2)
            J[:, 3, i] = (dR[:, 2, 1] - dR[:, 1, 2]) / (4 * h)
            J[:, 4, i] = (dR[:, 0, 2] - dR[:, 2, 0]) / (4 * h)
            J[:, 5, i] = (dR[:, 1, 0] - dR[:, 0, 1]) / (4 * h)
        return J.cpu().numpy().astype(np.float64)

    return jac


# --------------------------------------------------------------------------------------------- #
# the bake                                                                                       #
# --------------------------------------------------------------------------------------------- #
def fit_gmm(qs, n_components, seed=0, max_fit=120_000):
    """Fit a full-covariance GMM to standardised configurations; return bakeable arrays.

    Selection among IK branches for one pose is argmax p(q | pose), and on the manifold of
    configurations reaching that pose p(q | pose) is proportional to p(q) -- the normaliser is the
    same for every branch. So a plain density over configurations IS the conditional MAP here, with
    no need to model the pose dependence.

    ``sklearn`` is needed only to FIT; evaluation at plan time is pure torch (posture_prior.py).
    """
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(seed)
    mu, sc = qs.mean(0), qs.std(0)
    x = (qs - mu) / sc
    if len(x) > max_fit:
        x = x[rng.choice(len(x), max_fit, replace=False)]
    g = GaussianMixture(
        n_components, covariance_type="full", random_state=seed, reg_covar=1e-4, max_iter=200
    ).fit(x)
    return dict(
        gmm_weights=g.weights_, gmm_means=g.means_,
        gmm_prec_chol=g.precisions_cholesky_,      # [K, J, J], upper-triangular factors of Sigma^-1
        gmm_mu=mu, gmm_scale=sc, gmm_k=n_components,
    )


def bake(qs, jac_fn, n_joints, pct=(5, 95), n_jac=4000, seed=0, provenance="", gmm_k=16):
    """qs: [N, dof] configurations. jac_fn: q[M, dof] -> J[M, 6, dof]. Returns a np.savez dict."""
    qs = np.asarray(qs, dtype=np.float64)[:, :n_joints]
    lo = np.zeros((n_joints, n_joints))
    hi = np.zeros((n_joints, n_joints))
    for i in range(n_joints):
        for j in range(n_joints):
            if i != j:
                lo[i, j], hi[i, j] = np.percentile(qs[:, i] - qs[:, j], pct)

    rng = np.random.default_rng(seed)
    sub = qs[rng.choice(len(qs), min(n_jac, len(qs)), replace=False)]
    A = np.zeros((n_joints, n_joints))
    for J in jac_fn(sub):
        A += np.eye(n_joints) - np.linalg.pinv(J) @ J
    A /= len(sub)

    w = np.zeros((n_joints, n_joints))
    for i in range(n_joints):
        for j in range(n_joints):
            if i == j:
                continue
            v = np.zeros(n_joints)
            v[i], v[j] = 1.0, -1.0
            v /= np.sqrt(2.0)
            w[i, j] = float(v @ A @ v)
    out = dict(
        pair_lo=lo, pair_hi=hi, pair_weight=w, n_joints=n_joints, n_frames=len(qs),
        n_jac_samples=len(sub), percentiles=np.asarray(pct), provenance=provenance,
        model="gmm" if gmm_k else "pairwise",
    )
    if gmm_k:
        out.update(fit_gmm(qs, gmm_k, seed=seed))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "posture_ref.npz"))
    ap.add_argument("--source", choices=("droid", "lerobot"), default="droid")
    ap.add_argument("--path", default=str(DEFAULT_DROID_CACHE), help="shard cache or LeRobot dir")
    ap.add_argument("--robot", default="fr3_franka")
    ap.add_argument("--n-joints", type=int, default=7)
    ap.add_argument("--max-episodes", type=int, default=4000)
    ap.add_argument("--gmm-k", type=int, default=16,
                    help="GMM components (0 = bake the pairwise tables only)")
    args = ap.parse_args()

    loader = load_droid_shards if args.source == "droid" else load_lerobot
    episodes = loader(args.path, args.n_joints, max_episodes=args.max_episodes)
    qs = np.concatenate([q[_nonidle(q)] for q in episodes if _nonidle(q).any()])
    prov = f"{args.source}:{Path(args.path).name} robot={args.robot} " \
           f"{len(episodes)} episodes, {len(qs)} non-idle frames"

    out = bake(qs, make_jacobian_fn(args.robot), n_joints=args.n_joints,
               provenance=prov, gmm_k=args.gmm_k)
    np.savez(args.out, **out)
    print(f"wrote {args.out}\n  {prov}\n  model={out['model']}"
          + (f" K={args.gmm_k}" if args.gmm_k else ""))
    iu = np.triu_indices(args.n_joints, 1)
    for k in np.argsort(out["pair_weight"][iu])[::-1][:6]:
        i, j = iu[0][k], iu[1][k]
        print(f"  q{i+1}-q{j+1}: w={out['pair_weight'][i,j]:.3f} "
              f"band [{out['pair_lo'][i,j]:+.2f},{out['pair_hi'][i,j]:+.2f}]")


if __name__ == "__main__":
    main()
