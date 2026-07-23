# FluxNet: transport guarantees and traffic-rollout audit

This repository audits the exact transport-head claims in **FluxNet: Learning
Capacity-Constrained Local Transport Operators for Conservative and Bounded PDE
Surrogates** (ICML 2026; arXiv:2602.01941; OpenReview `1KRpajnd6u`).

It combines a small clean-room NumPy implementation of Propositions 1–3 with
direct execution of the unmodified author cores, pinned at commit
`ec0cafe3bb48cb7f2497723c5e12c6ebc518442c`. The structural audit covers the
official 1D `N`, `L`, and `U` heads. A separate bounded empirical attempt trains
the official FluxNet-D and residual-CNN classes on the complete released
periodic traffic split and evaluates 50-step extrapolative rollouts.

## What is verified

- **C1 — Proposition 1 (conservation).** Periodic directed transport preserves
  the global sum. The clean-room checks cover 96 signed 1D plans and 48 signed
  2D plans, with maximum absolute mass errors `8.44e-15` and `1.51e-14`.
- **C2 — Proposition 2 (L-head lower bound).** When the input starts at or
  above the lower bound, capacity-limited outgoing flow preserves that bound.
  All 96 deterministic 1D plans retain a positive minimum slack (`0.01197`).
- **Local P3 — Proposition 3 (U-head upper bound).** When the input starts at or
  below the upper bound, capacity-limited incoming flow preserves that bound.
  All 96 deterministic 1D plans retain a positive minimum slack (`0.01415`).
- **Author-core cross-check.** Across 12 seeded, small CPU configurations for
  each of the official `N`, `L`, and `U` heads, the largest float32 mass error
  is `1.91e-6`; the one-sided heads keep positive slacks (`0.08346` lower and
  `0.08297` upper).
- **Judged C3 — empirical PDE behavior (partial).** On all 100 released periodic
  traffic test trajectories, the trained FluxNet-D reaches final MAE `0.003052`
  versus `0.013734` for the matched ResNet-AR baseline. Its ratio `0.2222`
  closely matches arXiv v1 Table 4's `0.2189`, with zero divergent trajectories
  and maximum relative mass drift `3.34e-7` over all 50 rollout steps.
- **Local C3 shallow-water evidence (mixed backend).** On the identical 50-test,
  120-step protocol, local MPS FluxNet-LAP has final MAE
  `(0.004443, 0.006851, 0.005117)` versus returned CUDA FNO projection
  `(0.008300, 0.015773, 0.014137)`. The fail-closed audit proves identical
  config/source/dataset/test identities while retaining separate backend
  provenance. FluxNet wins all final and late accuracy fields, but the fixed
  controlled verdict is `inconclusive` because the preregistered relative
  momentum-drift statistic divides zero initial momentum by a `1e-12` floor.

## Run

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy pytest trackio
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/Lan-zs/FluxNet official
git -C official checkout ec0cafe3bb48cb7f2497723c5e12c6ebc518442c
PYTHONPATH=repro .venv/bin/python -m pytest repro/tests -q
PYTHONPATH=repro .venv/bin/python repro/run_audit.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=repro \
  .venv/bin/python repro/src/traffic_v1_empirical.py \
  --epochs 100 --batch-size 96 --device mps \
  --output-dir outputs/traffic_v1_attempt1
```

The deterministic structural report is at
[`outputs/summary.json`](outputs/summary.json); the trained traffic attempt is
documented in [`docs/claim3_traffic_attempt1.md`](docs/claim3_traffic_attempt1.md).
The completed shallow-water integration is documented in
[`docs/claim3_shallow_water_attempt2_mixed_backend.md`](docs/claim3_shallow_water_attempt2_mixed_backend.md).
The `official/` checkout is intentionally ignored: it is a separately pinned
dependency, not copied source.

## Scope and limitation

FluxNet's D-head averages its lower- and upper-bounded branches. Conservation
is retained, but the paper expressly treats simultaneous dual bounds as an
empirical DCL result rather than a universal hard guarantee. The committed
counter-control finds a valid D-head output below zero (`-0.05329`) while
preserving mass exactly. This is expected and is included to prevent a false
dual-bound claim.

The published/judged empirical attempt supports only periodic traffic flow and
is reported as `toy` challenge evidence. The newer shallow-water attempt is
local evidence, uses one seed split across MPS FluxNet and CUDA FNO backends,
and does not reproduce paper uncertainty or runtime estimates. Spinodal remains
outside the completed evidence, so the repository still does not claim a full
multi-dataset Table 3/4 verification.
