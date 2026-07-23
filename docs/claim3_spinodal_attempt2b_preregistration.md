# Claim 3 Attempt 2b preregistration: spinodal 100dt mechanism study

## Status and claim boundary

This document fixes the protocol before data generation or training. No
spinodal trajectory, checkpoint, or empirical result has been produced. Heavy
execution is gated until PID `18410` exits and the shallow-water Attempt 2
resource gate has been reviewed.

This is a released-training-scale **128x128 mechanism study**, not a literal
reproduction of arXiv v1 Table 5. The literal evaluation requires 20 pointwise
and 100 statistical trajectories at 1024x1024 plus an NVIDIA-A800-matched
timing comparison. The pinned release supplies neither those data nor trained
checkpoints, its test generator is itself fixed at 128x128, and this Apple M2
cannot make an attributable claim about the paper's `17.3x` A800 speedup.

## Pinned provenance and known conflicts

- Author source: `Lan-zs/FluxNet` commit
  `ec0cafe3bb48cb7f2497723c5e12c6ebc518442c`.
- Paper target: arXiv `2602.01941v1`, Table 5 and Appendix E.
- Released generator: periodic 128x128 double-precision explicit
  Cahn--Hilliard finite differences, base `dt=0.01`, five-point Laplacians,
  chemical-potential log clamp `1e-10`, and final concentration clamp to
  `[0,1]`.
- Source train seed `12345`, validation seed `67890`, and 20 independent test
  seeds `22345 + 12345*j` for `j=0,...,19`.
- The released CUDA generator writes six decimal places, emits frames from
  `t=0`, and has no DAT-to-HDF5 converter even though the released loader needs
  HDF5 `phi_data`. The v1 training interval begins at `t=2000dt`.
- The paper says DCL weight `1.0`; all three released training scripts specify
  `dcl_weight=0.5` and an additional manual DCL loss weight of `0.5`. This study
  follows and records the executable released loss exactly; it does not silently
  relabel that value as the paper's gamma.
- The released runner defaults to evaluation without training, has no MPS path
  or checkpoint-resume path, and reports time-averaged rollout MAE although v1
  Table 5 is final-time MAE. The reproduction wrapper must repair these
  execution defects while retaining the released scientific configuration.

Every generated file records the official commit and SHA-256 hashes of the
CUDA generator, CPU port, released model, released trainer/dataloader, selected
100dt script, and this preregistration.

## Fixed experiment

The primary model is the author-implemented `FluxNet_D_pf` at the v1/released
100dt configuration: one concentration input, 32 base channels, four residual
blocks, kernel size 5, 5x5 transport neighborhood (24 directed neighbors),
AdamW with learning rate `1e-3` and weight decay `1e-2`, batch 16, seed 42,
pushforward unroll 2, 100 epochs, and `ndt=10` over frames saved every 10 base
solver updates. This model is chosen before results because v1 reports it as
the best pointwise-accuracy/speed tradeoff: final MAE
`2.16 +/- 0.08e-2` and `3.8x` on the paper's hardware.

The reference generator will be a direct C++/OpenMP port of the released CUDA
kernels. It must preserve double precision, periodic halo-update order,
five-point expressions, constants, clamps, and step/save boundaries; compile
without fast-math or fused contraction; and stream directly to chunked HDF5.
CURAND's bitstream is not portable, so the port uses independently seeded
uniform random values from a documented standard-library generator. This is a
distribution-faithful deviation, not bitwise CUDA identity. Concentrations are
rounded to six decimals before float32 HDF5 storage to match the released DAT
precision.

- Train and validation each run 52,000 base updates, save every 10, and retain
  exactly frames `t=2000,...,52000` (5,001 frames each).
- Each of 20 test simulations runs 102,000 base updates. It retains only the
  evaluation-required states `t=2000,...,102000` every 100 base updates (1,001
  frames), avoiding approximately 11.8 GiB of unused intermediate storage
  without changing any evaluated state.
- Train on the one source-seeded training trajectory, choose checkpoints only
  from the source-seeded validation trajectory, and evaluate every test seed.
- Atomically checkpoint after each completed epoch with model, optimizer,
  scheduler, RNG, completed epoch, source/data fingerprint, history, and best
  state. A one-epoch gate keeps the target fingerprint fixed at 100 epochs and
  must resume deterministically.
- Prefer MPS with CPU fallback forbidden during the measured gate. Record
  epoch/batch timing, peak RSS, and device synchronization. One model runs at a
  time.

## Fixed metrics and controls

For every test trajectory, roll from its true `t=2000` state through all 1,000
100dt model steps to `t=102000`. Preserve final and selected intermediate truth
and predictions, per-time MAE, relative mass drift, lower/upper violation rates
and conditional magnitudes, phase fractions at threshold `0.6`, and radial
autocorrelation over radii `0,...,63`. The radial statistic follows the released
analysis: FFT autocorrelation of the concentration field itself, `fftshift`,
division by cell count, unit-width radial averaging, and mean absolute radial
error.

The statistical control pairs the 20 independent phase-field references into
10 fixed adjacent-seed pairs and measures their intrinsic radial-correlation
disagreement at the same evaluation times. The model/reference comparison is
matched by initial condition; this is stronger and less confounded than the
released analyzer's cross-initial-condition comparison. Bootstrap confidence
intervals resample whole trajectories or fixed reference pairs, never grid
cells or time points.

An independent audit must recompute final MAE, mass/bound metrics, radial
curves, AUCs, and all verdict booleans from the raw artifacts without importing
the experiment's metric functions.

## Success, falsification, and resource gates

The scoped mechanism result **supports** the spinodal component only if all of
the following hold:

1. all 20 rollouts finish with finite values;
2. maximum relative mass drift is at most `1e-5`;
3. final pointwise MAE is at most `4.32e-2` (twice the v1 100dt point estimate);
4. over the extrapolation interval `T=1...2`, the upper 95% bootstrap confidence
   bound for matched model/reference radial-error AUC divided by intrinsic
   phase-field/phase-field AUC is at most `1.25`.

If the lower 95% bound of the radial-AUC ratio is greater than `1.25`, the
statistical-preservation component is **falsified**. A mixed result, failed
integrity audit, incomplete model, or incomplete 20-trajectory evaluation is
**inconclusive**. Bound violations are always reported; no threshold is invented
from Table 5, which does not specify one for this result.

Before a full run, generate a short double-precision oracle trajectory and
require the CPU port and an independent NumPy reference to agree within the
six-decimal serialization tolerance. Then measure several steady-state
training batches. Abort rather than silently shrink the protocol if peak usage
exceeds 12 GiB, any operation falls back from MPS, or the projected wall time is
not locally sustainable. The 1000dt/17.3x extension remains a separate future
gate and cannot be inferred from a successful 100dt study.

## Prepared execution gates

The solver port, atomic HDF5 generator, exact released-loss training wrapper,
rollout evaluator, and separately implemented raw-artifact auditor are prepared.
The plan stage records 22 unique trajectories, 30,022 retained frames, and
1,967,521,792 uncompressed field bytes. Seven targeted tests cover the NumPy
oracle, invalid schedules, fixed seeds/frame counts, deterministic compiler
reuse, 100dt training windows, effective loss arithmetic, radial statistics,
and whole-sample bootstrap parity. These are implementation checks, not
scientific results.

The gated command sequence is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=repro .venv/bin/python \
  repro/src/spinodal_attempt2b.py --stage plan

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=repro .venv/bin/python \
  repro/src/spinodal_attempt2b.py --stage generate --threads 4 \
  --block-pid <active-shallow-water-pid>

PYTORCH_ENABLE_MPS_FALLBACK=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=repro .venv/bin/python \
  repro/src/spinodal_attempt2b.py --stage train --device mps \
  --max-new-epochs 1
```

After the one-epoch checkpoint/resume gate is independently reviewed, omit
`--max-new-epochs`, then run `--stage evaluate` and finally
`repro/src/audit_spinodal_attempt2b.py`. No generation, training, evaluation,
or audit command in this sequence has yet been run on the scientific protocol.
