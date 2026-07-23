# Claim 3 empirical attempt 1: periodic traffic flow

## Target and provenance

This attempt targets the third aggregate challenge claim using the periodic
traffic-flow experiment from arXiv `2602.01941v1`, not the later v2 paper.
The challenge anchors match v1 Table 4: at the 2x extrapolation horizon,
FluxNet-D reports MAE `3.48e-3` versus `15.9e-3` for ResNet-AR, with
conservation error `7.7e-8` versus `2.1e-2`.

The released author source is pinned at
`ec0cafe3bb48cb7f2497723c5e12c6ebc518442c` (2026-05-26) and postdates v1.
It retains the v1 architecture and 100-epoch setting, but its default dataset
generator only emits the training horizon (`T=4`) and the current paper/source
contains later configuration drift. The reproduction wrapper therefore calls
the unmodified released Rusanov solver for `T=8` test trajectories and imports
the unmodified official FluxNet-D and CNN/ResNet model classes.

## Pre-registered approach

- Generate all seven released periodic LWR cases at `N=256`, solver step
  `dt=0.016`, saved every 10 solver steps.
- Preserve the paper's stratified split: 100 train, 50 validation, 100 test.
- Train one FluxNet-D and one matched residual CNN/ResNet baseline using the
  official 32-channel, 6-block, kernel-5 implementations. FluxNet-D uses the
  official 11-point stencil.
- Use five-step pushforward training, AdamW (`lr=1e-3`, `weight_decay=1e-2`),
  v1 DCL weight `0.1`, 100 epochs, and seed 42.
- Use MPS, no loader workers, and batch 96 to stay within the local 30-minute
  ceiling while two unrelated CPU jobs are active. The released script uses
  batch 16; this deviation and the single seed prevent a paper-table claim.
- Evaluate every test trajectory for all 50 saved rollout steps (`T=8`),
  retaining full predicted and reference fields plus per-step metrics.

## Expected outputs

- `outputs/traffic_v1_attempt1/dataset_manifest.json`
- `outputs/traffic_v1_attempt1/training_history.json`
- `outputs/traffic_v1_attempt1/fluxnet_d_best.pt`
- `outputs/traffic_v1_attempt1/resnet_ar_best.pt`
- `outputs/traffic_v1_attempt1/traffic_rollouts_raw.npz`
- `outputs/traffic_v1_attempt1/summary.json`

## Decision criteria fixed before training

- **Direct improvement:** FluxNet-D has lower final and late-horizon rollout
  MAE than the matched baseline, fewer/nonexistent divergent trajectories,
  machine-scale conservation drift, and predictions approximately within
  `[0,1]`. For this float32 MPS run, the operational thresholds are maximum
  relative sum drift `<=1e-5` and conditional lower/upper violation magnitude
  `<=1e-2`. A trajectory is called divergent if it contains a non-finite value
  or any absolute prediction greater than 10.
- **Contradiction in this controlled setting:** the matched baseline has equal
  or lower final and late-horizon MAE/stability under identical data, epochs,
  batching, and seed.
- **Otherwise:** classify the result as toy/inconclusive. Even a positive
  single-seed result is `toy` relative to the five-seed paper table because of
  batch-size/source-version drift and the absence of a five-seed uncertainty
  estimate.

This attempt evaluates only periodic traffic flow. It does not reproduce the
shallow-water or spinodal-decomposition parts of the aggregate legacy claim.

## Commands executed

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python 'torch>=2.7,<2.8' \
  'numpy>=2,<3' 'h5py>=3.12,<4' 'matplotlib>=3.9,<4' \
  'pytest>=8,<9' 'tqdm>=4.67,<5'
PYTHONPATH=repro .venv/bin/python -m pytest repro/tests -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=repro \
  .venv/bin/python repro/src/traffic_v1_empirical.py \
  --epochs 100 --batch-size 96 --device mps \
  --output-dir outputs/traffic_v1_attempt1
```

The full run completed both 100-epoch trainings in 1,511.6 seconds
(FluxNet-D 927.4 seconds; ResNet-AR 584.1 seconds), excluding final rollout
and artifact compression. FluxNet-D's selected checkpoint was epoch 94
(`validation total=2.153e-6`); ResNet-AR's was epoch 87
(`validation total=3.945e-5`).

## Observed result

All metrics below were independently recomputed from
`traffic_rollouts_raw.npz` after the run. Values labeled `std` are standard
deviations over 100 test trajectories, not over training seeds.

| Metric | FluxNet-D | ResNet-AR | v1 Table 4 |
|---|---:|---:|---:|
| Final rollout MAE, mean | 0.003052 | 0.013734 | 0.00348 vs 0.0159 |
| Final rollout MAE, trajectory std | 0.001165 | 0.007281 | seed std 0.00012 vs 0.00056 |
| Late-horizon MAE | 0.002441 | 0.011400 | not reported |
| Late-horizon error slope/step | 5.30e-5 | 2.07e-4 | qualitative stability claim |
| Divergent trajectories | 0/100 | 0/100 | not reported |
| Max relative sum drift | 3.34e-7 | 2.29e-1 | 7.7e-8 vs 2.1e-2 |
| Lower violation rate | 0.366% | 2.046% | 0.52% vs 0.32% |
| Upper violation rate | 0.313% | 1.260% | 2.87% vs 3.20% |
| Conditional lower magnitude | 9.99e-4 | 5.71e-3 | 1.13e-3 vs 37.6e-3 |
| Conditional upper magnitude | 1.57e-3 | 16.17e-3 | 0.84e-3 vs 15.6e-3 |

The observed final-MAE ratio is `0.2222`; v1 reports `0.2189`. FluxNet-D
also has lower final MAE than the baseline in every released case category:
traffic jam, speed-limit zone, red light, three shock directions, and
rarefaction. The preregistered direct-improvement criteria are all met.

## Assessment

This is a **substantive scientific attempt**: it executes the released solver
and both official model classes at the paper's spatial resolution, full split,
architecture, epoch count, training horizon, extrapolation horizon, and
pushforward depth, and it retains per-cell predictions for every test step.
Within this controlled setting, the result **supports** the periodic
traffic-flow portion of the claim and closely matches v1's accuracy ratio.

For challenge grading it remains **toy**, not verified. It is one MPS seed;
batch 96 replaces the released script's batch 16; the only available author
commit postdates v1; and the aggregate challenge claim also names shallow
water and spinodal decomposition. No claim is made that the trajectory-level
standard deviation reproduces the paper's five-seed uncertainty.

## Artifact verification

- Raw NPZ shape: truth and each model prediction are `(100, 51, 256)`.
- All truth and prediction values are finite.
- Recomputed raw metrics match `summary.json` within absolute tolerance
  `1e-12`.
- Repository suite: `8 passed`.
- Pinned `official/` checkout remains clean at the declared commit.
- Raw artifact SHA-256:
  `2ea19da97ebced8173262b56a07a38d8f631639711505dcce0c4cefe9c297a9b`.

## Recommended logbook integration

Preserve the existing proposition audit and D-head counter-control. Rename its
current "C3 - Proposition 3" label to avoid collision with the challenge's
empirical C3, then add a separate empirical-claim section containing this
attempt, exact command, v1 provenance, raw artifact link/hash, matched metrics,
and the explicit `toy` limitation. Replace the old blanket statement that all
trained benchmarks are excluded with a narrower statement that shallow-water
and spinodal training remain excluded.
