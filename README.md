# FluxNet: CPU transport-guarantee audit

This repository audits the exact transport-head claims in **FluxNet: Learning
Capacity-Constrained Local Transport Operators for Conservative and Bounded PDE
Surrogates** (ICML 2026; arXiv:2602.01941; OpenReview `1KRpajnd6u`).

It combines a small clean-room NumPy implementation of Propositions 1–3 with
direct CPU execution of the unmodified 1D `N`, `L`, and `U` heads from the
authors' public source, pinned at commit
`ec0cafe3bb48cb7f2497723c5e12c6ebc518442c`. It is not a trained-PDE benchmark
reproduction.

## What is verified

- **C1 — Proposition 1 (conservation).** Periodic directed transport preserves
  the global sum. The clean-room checks cover 96 signed 1D plans and 48 signed
  2D plans, with maximum absolute mass errors `8.44e-15` and `1.51e-14`.
- **C2 — Proposition 2 (L-head lower bound).** When the input starts at or
  above the lower bound, capacity-limited outgoing flow preserves that bound.
  All 96 deterministic 1D plans retain a positive minimum slack (`0.01197`).
- **C3 — Proposition 3 (U-head upper bound).** When the input starts at or
  below the upper bound, capacity-limited incoming flow preserves that bound.
  All 96 deterministic 1D plans retain a positive minimum slack (`0.01415`).
- **Author-core cross-check.** Across 12 seeded, small CPU configurations for
  each of the official `N`, `L`, and `U` heads, the largest float32 mass error
  is `1.91e-6`; the one-sided heads keep positive slacks (`0.08346` lower and
  `0.08297` upper).

## Run

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy pytest trackio
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/Lan-zs/FluxNet official
git -C official checkout ec0cafe3bb48cb7f2497723c5e12c6ebc518442c
PYTHONPATH=repro .venv/bin/python -m pytest repro/tests -q
PYTHONPATH=repro .venv/bin/python repro/run_audit.py
```

The deterministic report is committed at
[`outputs/summary.json`](outputs/summary.json). The `official/` checkout is
intentionally ignored: it is a separately pinned dependency, not copied source.

## Scope and limitation

FluxNet's D-head averages its lower- and upper-bounded branches. Conservation
is retained, but the paper expressly treats simultaneous dual bounds as an
empirical DCL result rather than a universal hard guarantee. The committed
counter-control finds a valid D-head output below zero (`-0.05329`) while
preserving mass exactly. This is expected and is included to prevent a false
dual-bound claim.

This audit excludes learned PDE rollout accuracy, long-horizon stability,
runtime comparisons, and all trained benchmark tables.
