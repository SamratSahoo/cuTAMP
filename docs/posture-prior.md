# The posture prior: choosing human-like IK branches

How `cutamp/posture_prior.py` and `cutamp/posture_ref.npz` work, and the maths behind the two
tables. Implementation: `cutamp/posture_prior.py`, `cutamp/particle_initialization.py`
(`select_posture_branch`), `cutamp/scripts/bake_posture_ref.py`.

---

## 1. The problem

A 7-DOF arm needs only 6 degrees of freedom to place the gripper. The spare one means **infinitely
many joint configurations reach the same end-effector pose** — same hand, different elbow. Touch
your nose: your elbow can be high, low or out to the side and your fingertip never moves.

Humans teleoperating the arm pick one narrow family of those configurations, consistently. The
planner was picking essentially at random, which is why planned trajectories visited joint
configurations teleoperation never does.

## 2. Why it was random

Every plan waypoint starts with an IK solve. cuRobo optimises ~12 random seeds in parallel and each
converges to a *different* valid configuration — these are the **branches**. It then ranks them and
returns one:

```
rank = pose_error + null_space_error          # ik_solver.py:1099-1100
```

Every branch reaches the pose, so `pose_error` is ~0 for all of them, and `null_space_cfg.weight` is
**0.001** (`base_cfg.yml:66`) against a generic home retract. The tiebreaker is negligible, so the
winner is arbitrary — independently at every grasp, place and lift.

Measured consequence: **80–90% of the deviation from human configurations lives *between* plan
segments, not inside them.** That is exactly the part no trajopt cost can reach, because trajopt's
endpoints are already pinned by the time it runs. It is why `joint_density_weight` never moved it.

## 3. The fix

Two levers, both at the IK endpoint:

1. **Branch selection** — ask IK for `k` branches instead of 1 and keep the most probable under a
   density fit to human configurations (§4).
2. **Grasp roll** — also solve the pi-rolled twin of the grasp and keep the better of the two (§6).

Lever 1 alone gets the residual from ~47% to ~41%; the two together reach **7–14%** (§9). Lever 2 is
the larger one, because a wrist roll is a property of the *grasp*, and both branches of a single
pose share it — no amount of branch selection can reach it.

---

## 4. The model: a learned density over configurations

Given `k` IK branches for **one** pose, the principled choice is

$$\arg\max_q \, p(q \mid \text{pose})$$

On the manifold of configurations reaching that pose the normaliser is identical for every branch,
so this reduces exactly to $\arg\max p(q)$ — a plain density over human configurations. No pose
conditioning needs modelling.

The density is a **full-covariance Gaussian mixture** fit to the standardised reference corpus:

$$\text{penalty}(q) = -\log p_{\text{GMM}}(q)
= -\log \sum_{k=1}^{K} \pi_k \, \mathcal{N}\!\left(z; \mu_k, \Sigma_k\right),
\qquad z = \frac{q - \bar q}{\sigma_q}$$

Evaluated with the precision Cholesky factors $P_k$ ($\Sigma_k^{-1} = P_k P_k^\top$) for stability:

$$\log \mathcal{N}_k = \underbrace{\log \pi_k + \textstyle\sum_d \log (P_k)_{dd}}_{\text{baked constant}}
- \tfrac{1}{2}\big\|(z - \mu_k)^\top P_k\big\|^2$$

The additive constant $\tfrac12 J \log 2\pi$ is dropped — identical for every branch, so it cannot
change an argmin. Evaluation is pure torch; `sklearn` is needed only to bake. Verified against
`sklearn.score_samples` to **1e-14**.

### Choosing K

By held-out log-likelihood on an **episode-level** split (frames within an episode are highly
correlated, so a frame split leaks):

| K | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| held-out LL | −9.21 | −8.01 | −7.52 | −7.21 | **−7.06** | **−6.96** | −6.96 | −7.06 |

Flat between 16 and 32 and turning over by 64. K=16 is shipped: it matched K=32 on the downstream
selection metric while being half the size and less prone to overfitting a re-bake on a smaller
corpus. The artifact is 12.7 KB.

---

## 5. Fallback model: the pairwise tables

Retained for references baked with `--gmm-k 0`, and worth reading because it explains what the
density has to capture. It penalises joint-**pair** differences outside the human range, weighted by
how free each pair is to move:

$$\text{penalty}(q) = \sum_{i<j} w_{ij}\Big[\max(0, d_{ij} - \texttt{hi}_{ij}) +
\max(0, \texttt{lo}_{ij} - d_{ij})\Big], \qquad d_{ij} = q_i - q_j$$

### 5a. Table A — where humans put their joints

### Input

$$Q \in \mathbb{R}^{N \times J}, \qquad N = 987{,}631 \text{ frames}, \quad J = 7$$

Each row is one recorded frame of a human driving the arm (DROID, ~4,000 episodes). Frames are
filtered to non-idle only — $\|\dot q\|_2 > 0.05$ rad/s by central difference at 15 Hz. Without that
filter, long parked stretches dominate and the bands collapse onto wherever the arm happened to rest.

### The computation

For each ordered pair $(i,j)$, form the scalar difference across all frames and take two empirical
quantiles of that 1-D sample:

$$d_{ij}^{(n)} = Q_{n,i} - Q_{n,j}, \qquad
\texttt{lo}_{ij} = F^{-1}_{d_{ij}}(0.05), \qquad
\texttt{hi}_{ij} = F^{-1}_{d_{ij}}(0.95)$$

```python
for i in range(n_joints):
    for j in range(n_joints):
        if i != j:
            lo[i, j], hi[i, j] = np.percentile(qs[:, i] - qs[:, j], pct)   # pct = (5, 95)
```

$J^2 = 49$ independent 1-D quantile estimates. No fitting, no optimisation, no covariance.

### Why *differences* rather than each joint alone

This is where the signal is. In the planner's data joint 1 alone looked fine and joint 3 alone
looked fine — their **difference** was 4× too wide. A per-joint marginal is mathematically blind to
that, which is precisely why the earlier `joint_density` cost could not help.

### Structure

Since $d_{ji} = -d_{ij}$, the tables are antisymmetric with roles swapped:

$$\texttt{lo}_{ji} = -\texttt{hi}_{ij}, \qquad \texttt{hi}_{ji} = -\texttt{lo}_{ij}$$

Verified on the baked file: `lo.T == -hi` exactly, diagonal zero, width matrix symmetric. So there
are only $\binom{7}{2} = 21$ independent bands; the full 7×7 is stored for vectorisation, and the
penalty double-counts each pair — a constant factor of 2 that cannot change an argmin.

### Why quantiles and not mean ± kσ

These distributions are badly non-Gaussian:

| pair | skew | excess kurtosis | empirical 5–95 | Gaussian $\mu \pm 1.645\sigma$ | full min/max |
|---|---|---|---|---|---|
| q1−q3 | −0.76 | **28.9** | [−0.43, +0.56] w=1.00 | [−0.77, +0.84] w=**1.61** | [−5.07, +4.99] |
| q3−q5 | 0.09 | 6.5 | [−0.93, +1.07] w=2.00 | [−1.02, +1.07] w=2.09 | [−4.09, +5.20] |
| q1−q7 | 0.10 | 2.7 | [−1.39, +1.23] w=2.62 | [−1.36, +1.24] w=2.60 | [−4.20, +4.82] |

`q1−q3` has excess kurtosis **28.9** — sharply peaked with heavy tails. A Gaussian fit gives a band
**61% too wide** because $\sigma$ is inflated by rare excursions. Coverage check: the empirical band
holds 90.0% of frames by construction, the Gaussian band holds 95.0% — a whole extra tail admitted.
min/max would be useless: the full range of `q1−q3` is 10 radians, which permits everything.

---

### 5b. Table B — which pairs are the planner's to choose

Table A says *where* humans put each pair. Table B says *which pairs the planner may move without
fighting the task*. It comes from geometry only — no human behaviour enters it.

### The null space

At configuration $q$ the Jacobian $J(q) \in \mathbb{R}^{6\times 7}$ maps joint velocity to
end-effector velocity, $\dot x = J(q)\dot q$. Six outputs, seven inputs, so generically
$\operatorname{rank}(J) = 6$ and there is a 1-D **null space**: a joint direction producing zero
end-effector motion. That is the elbow shuffle. Its orthogonal projector is

$$N(q) = I - J^{+}J$$

Verified numerically at a real configuration: $N = N^\top$, $N^2 = N$, $\operatorname{tr}(N) = 1.000
= 7 - 6$ — exactly one free dimension.

### Scoring a direction

For a unit vector $v$, $\|Nv\|^2 \in [0,1]$ is the **fraction of $v$ that is free motion**: 1 if
moving along $v$ leaves the gripper perfectly still, 0 if it only moves the gripper.

The pair direction is "joint $i$ up, joint $j$ down, equally", normalised so all 21 pairs are scored
on the same footing:

$$v_{ij} = \frac{e_i - e_j}{\sqrt{2}}, \qquad
w_{ij} = \mathbb{E}_q\!\left[\|N(q)\,v_{ij}\|^2\right]$$

$N(q)$ depends on $q$ — the null space rotates as the arm moves — so the expectation is over 4,000
sampled human configurations.

### The shortcut the code uses

Because $N$ is a **symmetric idempotent** projector,

$$\|Nv\|^2 = v^\top N^\top N v = v^\top N^2 v = v^\top N v$$

(verified: both give `0.059292` at the test configuration). That linearises the expectation:

$$w_{ij} = \mathbb{E}_q\!\left[v_{ij}^\top N(q) v_{ij}\right]
        = v_{ij}^\top \underbrace{\mathbb{E}_q[N(q)]}_{A} v_{ij}$$

So accumulate **one** 7×7 matrix over 4,000 configurations, then read off all 49 entries as cheap
quadratic forms instead of 49 × 4,000 projections:

```python
A = mean over q of (I - pinv(J) @ J)      # one 7x7
w[i, j] = v @ A @ v                        # v = (e_i - e_j)/sqrt(2)
```

### Does $w$ mean what it claims?

Direct check — how much end-effector motion each pair direction actually causes:

| pair | $w_{ij}$ | EE linear | EE angular |
|---|---|---|---|
| q1−q3 | **0.728** | **3.3 cm/rad** | 17 °/rad |
| q3−q5 | 0.395 | 36.1 cm/rad | 69 °/rad |
| q1−q7 | 0.263 | 40.7 cm/rad | 80 °/rad |
| q5−q7 | 0.234 | 6.5 cm/rad | 31 °/rad |
| q1−q2 | 0.177 | 59.5 cm/rad | 57 °/rad |
| q4−q6 | 0.006 | 28.1 cm/rad | 7 °/rad |
| q2−q4 | **0.002** | **80.8 cm/rad** | 81 °/rad |

High $w$ ↔ the gripper barely moves. Low $w$ ↔ the gripper flies. **This is why the penalty cannot
fight the task**: a pair that would drag the gripper off target is automatically down-weighted to
nothing. No hand-tuning, no exclusion list.

### Structural properties

- $w_{ij} \in [0,1]$, guaranteed by $N$ being a projector. Observed range `[0.002, 0.728]`.
- Symmetric, $w_{ij} = w_{ji}$: $v_{ji} = -v_{ij}$ and the form is quadratic.
- **No sum rule.** The 21 pair directions are not orthogonal, so $\sum_{i<j} w_{ij} = 3.485$, not 1.
  (An orthonormal basis would sum to $\operatorname{tr}(A) = 1$.) These are relative importances,
  not a probability distribution.

---

## 6. The grasp roll

A parallel jaw is invariant to a pi roll about its approach axis, so **every grasp has two equally
valid end-effector poses**:

$$T^{\text{ee}}_{\text{alt}} = T^{\text{ee}} \cdot R_z(\pi)$$

cuTAMP only ever builds one of them — whichever representative M2T2 emitted — and roughly half the
time that is the wrist branch teleoperation never uses. Both IK branches of a single pose inherit
that pose's roll, so §4's selection cannot reach it. `solve_with_grasp_roll` solves both poses and
selects across the union.

**The stored grasp is rolled to match.** For this arm the roll commutes with the tool offset,

$$T^{\text{tool}}_{\text{ee}} \, R_z(\pi) \, \left(T^{\text{tool}}_{\text{ee}}\right)^{-1} = R_z(\pi)$$

verified exactly, so the *same* matrix applies on the grasp side: wherever the twin wins,
`obj_from_grasp` is post-multiplied by $R_z(\pi)$. Measured grasp/configuration consistency after
selection: **0.02 mm / 0.04°**. Only applies where the grasp is a 4x4 (M2T2 grasps) — the 4/6-DOF
sampled parameterisations cannot carry a roll, and are skipped.

Applied at **Pick** only. At Place the grasp is already bound by the preceding Pick, so rolling
there would contradict it; Place gets branch selection alone.

### Two cuRobo hazards this path had to work around

Both were found by measurement, and both silently produce plausible-but-wrong results:

**`solve_batch` normalises its goal quaternion in place, and `Pose` objects alias cuRobo's internal
goal buffer.** A goal held across a *second* solve is overwritten — measured drift **1.41** on a
unit quaternion, i.e. the first goal literally becomes the second. `Pose.clone()` does not protect
it. The fix is structural: **score each pool immediately after its own solve**, while its goal is
still intact. Deferring the scoring made every branch of the first pool fail its FK check, so the
twin won 100% of the time and the comparison was meaningless — it read as a 41% residual instead of
the true 7%.

**Changing the IK batch size trips the cuda graph** (`"changing goal type, cuda graph reset not
available"`). The two poses must therefore be two *same-size* `solve_batch` calls, not one
double-width batch.

## 7. What happens at plan time

`select_posture_branch` runs once per IK solve, for each of the `k` returned branches:

**Step 1 — verify it reaches the target.** cuRobo's `IKResult.success` comes back **all-False** in
this batched path (and `position_error` all-zero), so branches are validated by forward kinematics
instead: within `posture_pos_tol` (5 mm) and `posture_rot_tol` (0.05 rad ≈ 2.9°). This incidentally
filters non-reaching solutions the stock path accepts blindly — max FK error over 512 endpoints was
6.19 mm / 60.7° stock versus 0.01 mm selected.

**Step 2 — score the posture**, `penalty = -log p(q)` under the baked mixture (§4), or the pairwise
hinge if the reference was baked without one (§5). With `posture_grasp_roll` on, the pi-rolled twin
pose is solved and scored the same way and the two pools are concatenated (§6).

**Step 3 — keep the argmin**, with invalid branches masked to $+\infty$.

Note the density gives a *strict* ranking — unlike the pairwise hinge, which is exactly zero
anywhere inside the bands and therefore deferred to cuRobo's own ordering among in-band branches.

### Fallbacks

Selection can never lose a plan the stock configuration would have found:

| situation | behaviour |
|---|---|
| no FK-valid branch for a particle | keep cuRobo's top seed (`solution[:, 0]`) |
| prior file missing or malformed | log once, disable selection |
| prior covers more joints than the arm has DOF | log once, disable selection |
| `posture_selection_seeds: 0` | returns a tensor **byte-identical** to the previous code path |

---

## 8. Configuration

One knob:

```yaml
tamp_overrides:
  posture_selection_seeds: 12     # branches to score per endpoint; 0 or 1 = off
  posture_grasp_roll: true        # also try the pi-rolled twin grasp (default true)
```

Ready-made configs: `data-collection/cfg/tamp/*_learned_posture.yml` (one per task).

Everything else — which joints matter, how far they may vary — comes from the baked file. Nothing to
tune per scene, per task or per object.

`TAMPConfiguration` also exposes `posture_ref` (path to an alternative baked file) and the two FK
tolerances, all optional.

Note `ik_num_seeds` is raised to 3× the branch count automatically: cuRobo's `return_seeds` cannot
exceed the solver's `num_seeds`, and branch diversity thins as the two converge.

---

## 9. Baking and porting

```bash
# default: DROID proprio shard cache, FR3
python -m cutamp.scripts.bake_posture_ref

# a different corpus and robot
python -m cutamp.scripts.bake_posture_ref --source lerobot \
    --path ~/.cache/huggingface/lerobot/<user>/<repo> --robot fr3_robotiq
```

The script is self-contained — corpus loading, Jacobian and table build all live in it, so
regenerating the prior needs only cuTAMP, numpy and the corpus on disk.

The Jacobian is **central-differenced through cuRobo's own FK** rather than a hardcoded DH table
(cuRobo refuses to emit an analytic one: *"Outputting jacobian is not supported"*). That keeps the
baker robot-agnostic. Cross-checked against an analytic FR3 DH Jacobian: bands bit-identical, weights
max abs diff **2.6e-04**, pair ranking unchanged.

Re-bake against a different robot or corpus and the whole prior moves with it. Which joints matter is
an *output*, not an input.

---

## 10. Measured results

Scored by **5-NN distance to held-out DROID episodes** — non-parametric and from a different model
family than the selector, so no mixture appears in the metric. DROID is split **by episode** (80/20;
a frame split leaks, since frames within an episode are highly correlated), the prior is fit on
train, the yardstick built on held-out.

% of selected configurations beyond held-out DROID's own p99 in 7-D:

| task | stock | + branch selection | + grasp roll | twin won |
|---|---|---|---|---|
| plate | 41% | ~41% | **12%** | 48% |
| fruits | 47% | 41% | **7%** | 54% |
| pack | 42% | ~42% | **14%** | 41% |
| held-out DROID (ceiling) | — | — | 1% | — |

Earlier, on the same yardstick, the pairwise tables scored 43–49% and the density 41–46% — i.e.
**the model upgrade was worth a few points and the grasp roll was worth ~30**. That ordering is the
main practical finding: the residual was never a ranking problem.

Corroborating this, an **oracle** that picks the yardstick-optimal branch from a pool of 96
candidates for the *given* pose still leaves 37%. The given pose was the ceiling; the twin pose is
what breaks it.

End-effector poses are unchanged (grasp/configuration consistency 0.02 mm / 0.04°); cost is one
extra IK batch per endpoint, ~24 ms at 512 particles against multi-second plans.

## 11. Limitations

**A GMM is a global, unimodal-per-component model of a manifold-like distribution.** Human
configurations concentrate near a low-dimensional set; a mixture approximates that with ellipsoids.
K=16 was chosen by held-out likelihood, but likelihood rewards covering probability mass, not
capturing the manifold's shape — a normalising flow would fit better at the cost of a far larger
artifact and a training loop.

**It is a density, not a constraint.** Unlike the pairwise hinge, every branch gets a strictly
positive penalty, so the selector always expresses a preference even when all branches are
perfectly reasonable. That is what makes it stronger, and also what makes it less auditable: there
is no "this is fine" region you can read off a table.

**The residual is 7–14%, against 1% for DROID itself.** Selection can only choose among what IK
proposes; closing the last gap means changing what is generated (seeding, or the grasp set), not how
it is ranked.

**The pairwise fallback is a product of marginals over pairs.** It cannot express "`q1−q3` may be
wide *when* `q5−q7` is narrow", and it is provably blind to common mode (§9). Its 5/95 cut is also
a policy choice, not a fact.

**$\mathbb{E}[N]$ averages projectors**, which is meaningful only if the null space is reasonably
stable across the sampled region. It is here — the arm works in a tabletop volume — but on a robot
ranging over a much wider workspace the average smears and every $w$ drifts toward an uninformative
middle. The tell is the eigenvalue spread of $A$: on DROID configurations it is `[0.78, 0.20, 0.01,
…]`, one dominant direction; on uniformly sampled configurations it is `[0.43, 0.24, 0.13, …]`, and
a fixed table would be the wrong model.

**$w$ measures freeness, not brokenness.** It says the planner may move a pair without cost to the
task — not that the planner currently gets it wrong. Those correlate only **+0.40**. Table B alone
would spend effort on pairs that are already fine; pairing it with Table A's bands is what makes the
penalty zero when nothing is wrong.

**The roll is applied at Pick only** (§6), and only for M2T2 4x4 grasps. A scene whose grasps come
from the 4/6-DOF samplers gets branch selection alone.

**Everything above is offline.** Real recorded poses, real IK, real trajopt — but never a live scene
with obstacles. `k` branches are scored before collision pruning in these measurements; in clutter
some branches are pruned, so branch diversity should be re-checked on a saved run's world.
