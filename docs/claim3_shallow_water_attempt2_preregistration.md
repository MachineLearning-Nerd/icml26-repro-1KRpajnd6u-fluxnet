# Claim 3 Attempt 2 preregistration: shallow water

## Status boundary

This section records the boundary when the protocol was frozen: the document
and accompanying driver were **preparation only**, no shallow-water trajectory
had been generated, no model had been trained, and heavy stages were gated
while PID `18410` was active. The accepted traffic Attempt 1 artifacts and the
C1/C2 evidence were out of scope and remain unchanged.

## Execution ledger (post-registration; no protocol amendment)

The resource gate subsequently cleared and the preregistered experiment began.
At the `2026-07-20T10:01+05:30` evidence snapshot, the fixed dataset is complete
at aggregate SHA-256
`ee1d684d479f989f5588afbd40f493a58079ac9ed10dd97d03b16ad6725473cf`,
FluxNet-LAP is complete at 100/100 epochs, and the same active process has
reached 23/100 FNO epochs before handing authority to the checkpointed CUDA
continuation. A pre-existing local handoff attempted evaluation too early: it
produced a valid FluxNet-only rollout and the driver's completion guard refused
local FNO evaluation at 23/100. The premature attempt is excluded from the
final comparison; the authoritative FNO continuation must complete first.

The partial FluxNet result is recorded only as an execution fact, not as a
protocol amendment: final MAE is `(0.00444280, 0.00685067, 0.00511738)` and late
MAE is `(0.00445029, 0.00673975, 0.00537206)` for `(h, mx, my)`, with zero
divergence and zero negative-depth cells. The fixed relative-drift diagnostic
revealed that zero initial momentum L1 norms make the preregistered `+1e-12`
floor produce very large `mx`/`my` ratios despite absolute integral drift below
`2.8e-5`. Because this was observed after registration, the criterion below is
not changed; it will fail as written and the undefined-scale edge case must be
reported transparently.

The independent protocol audit passed without rehashing the HDF5 payloads
during live MPS training. It binds the current ledger to the exact official
commit, six hard-coded source hashes, fixed run configuration, all 120 unique
trajectory identities and byte sizes, and the fixed aggregate dataset hash.
Its five synthetic research-integrity controls all trigger the intended
failure paths. This is evidence-validator calibration, not an automated model
test suite. The snapshot is
`outputs/shallow_water_attempt2/protocol_audit.json`; the final audit requires
both models to have 100 contiguous epochs and independently rechecks the raw
rollout arrays, sample order, initial state, time grid, metrics, checkpoint
hashes, per-case results, and controlled verdict. It also identifies every
evaluation plan that occurs before both model-completion events, so incomplete
handoffs cannot be mistaken for final evidence.

The authoritative CUDA FNO result is accepted only through the fail-closed
return procedure in `docs/claim3_colab_return_verification.md`. Its transport
hash, immutable CUDA event ledger, safely loaded checkpoints, fixed provenance,
and raw rollout metrics must all validate before optional hash-named import.

The returned archive subsequently passed that procedure at SHA-256
`9879b652cd74e77c0305a87ab2fcaaae24a750579024ec9e6d6c4e2e44b95daf`
and 922,070,927 bytes. The final mixed-backend evidence audit keeps the complete
local FluxNet MPS and returned FNO CUDA roots separate, excludes the incomplete
local FNO, rehashes both data chains, proves identical raw test truth, and
independently recomputes the comparison. Its result and limitations are in
`docs/claim3_shallow_water_attempt2_mixed_backend.md`.

## Target and provenance

The target is arXiv `2602.01941v1` Table 3 at the 2x extrapolation horizon. The
paper reports final per-field MAE for FluxNet-LAP of `3.12e-3`, `4.41e-3`, and
`4.64e-3` for `(h, mx, my)`, versus `6.74e-3`, `12.9e-3`, and `11.5e-3` for the
best post-hoc baseline, FNO with box-and-mass projection. FluxNet-LAP reports
conservation errors
`(3.3e-8, 6.0e-8, 3.2e-8)` and zero depth violations.

The author source is pinned at
`ec0cafe3bb48cb7f2497723c5e12c6ebc518442c`. There is real version drift: v1
describes 64 channels, 4 blocks, kernel 3, and 100 epochs; the post-v1 released
single-seed script uses 64 channels, 6 blocks, kernel 5, and 100 epochs, while
the released multi-seed script changes the budget to 300 epochs. This bounded
attempt preregisters the v1 architecture and 100-epoch configuration. The
post-v1 6-block/kernel-5 sensitivity is explicitly
out of scope. The attempt must not switch architecture or to 300 epochs after
seeing results.

Every run records the official commit plus SHA-256 hashes of the generator,
both model sources, the released configuration, every HDF5 trajectory, the
aggregate dataset, checkpoints, and raw rollout artifacts. The ignored paths
are `data/shallow_water_attempt2/` and `outputs/shallow_water_attempt2/`.

## Fixed experiment

- Exactly two author-implemented v1 headline models: `FluxNet_SW_LAP_pf` and
  `FNO_SW_Proj_box_mass_pf`. The weak ResNet-AR control and other ablations are
  not needed to test the best-baseline comparison and are not added.
- Complete official stratified split: 50 train, 20 validation, and 50 test
  trajectories across Case A1, A2, B1, and B2.
- Official NumPy finite-volume/Rusanov/SSP-RK3 solver on 128x128, conservatively
  downsampled to 64x64, fixed solver `dt=0.004`, saved every 10 solver steps.
- Training and validation end at `T=2.4`; every test trajectory is generated
  from the same released solver to `T=4.8` for 120 learned rollout steps.
- FluxNet-LAP uses the v1 64-channel, 4-block, kernel-3 architecture and released
  3x3 transport stencil. The projection baseline uses the v1/released FNO
  configuration (16 modes, width 64, 4 layers) followed by box-and-mass depth
  projection. Both use AdamW (`lr=1e-3`, `weight_decay=1e-2`), batch 16, seed
  42, and five-step pushforward training with equal one-step and terminal loss
  weights.
- Train for exactly 100 epochs. Checkpoints are atomically replaced after every
  completed epoch and include model, optimizer, scheduler, RNG, history,
  dataset/source fingerprint, and best state. Resume is enabled by default.
- Prefer MPS on the Apple M2; CPU is supported. Training uses one Torch CPU
  thread and in-memory batches for deterministic, epoch-boundary resume. The
  wrapper preserves the
  released loss, architecture, and split, but its deterministic in-memory batch
  permutation is not asserted to be bitwise identical to the released
  multi-worker CUDA run. JSONL events report epoch
  timing, elapsed time, learning rate, and peak RSS.

The full float32 HDF5 dataset has a raw upper bound of about 0.50 GiB before
gzip; the streaming generator should remain below 1 GiB RSS. Training is the
dominant cost. Until a one-epoch measured gate exists, reserve roughly 8-20
hours per 100-epoch model on M2 MPS (16-40 hours for the two-model seed), with
about 3-5 GiB free for checkpoints, raw predictions, and temporary compression.
Do not start unless at least 10 GiB disk is free and PID `18410` has exited.

## Fixed evaluation and artifacts

Roll out all 50 test trajectories from the true initial state through every one
of the 120 saved steps. Preserve per-cell truth and predictions, trajectory and
case identifiers, per-step per-field MAE, divergence flags, and per-field
integral drift. Report final MAE and late MAE over `T>2.4`, divergence (nonfinite
or any absolute predicted state above 100), `h<0` rate and conditional
magnitude, and maximum integral drift normalized by the initial field L1 norm.

Expected outputs:

- `data/shallow_water_attempt2/manifest.json`
- `outputs/shallow_water_attempt2/run_config.json`
- `outputs/shallow_water_attempt2/events.jsonl`
- `outputs/shallow_water_attempt2/spinodal_boundary.json`
- `outputs/shallow_water_attempt2/models/<model>/{latest,best}_checkpoint.pt`
- `outputs/shallow_water_attempt2/models/<model>/training_history.json`
- `outputs/shallow_water_attempt2/raw/<model>_rollouts.npz`
- `outputs/shallow_water_attempt2/metrics/<model>.json`
- `outputs/shallow_water_attempt2/comparison.json`
- `outputs/shallow_water_attempt2/protocol_audit.json`
- `outputs/shallow_water_attempt2/audit.json`

## Success and falsification fixed before execution

The controlled result **supports** the shallow-water component only if
FluxNet-LAP has lower final and late MAE than the best projection baseline for
all three
fields, has zero divergent trajectories, exactly zero negative-depth cells,
and maximum L1-normalized integral drift no greater than `1e-5` for each field.

Paper-anchor consistency is a separate, stricter flag: each FluxNet final MAE
must be no more than 2x the v1 value and each FluxNet/baseline final-MAE ratio
must be no more than 2x the corresponding v1 ratio. Failure of this flag does
not get hidden by the controlled comparison.

The result **contradicts** the controlled accuracy advantage if the projection
baseline is equal or better for all fields at both final and late horizons. Mixed outcomes,
any failed integrity audit, or inability to finish both models and all 50
rollouts is **inconclusive**. Numerical instability, MPS incompatibility, or a
runtime overrun is reported as a failed/incomplete attempt, never silently
reconfigured after results are visible.

## Commands, after the resource gate clears

```bash
if ps -p 18410 >/dev/null 2>&1; then exit 1; fi
df -h .
PYTHONPATH=repro .venv/bin/python repro/src/shallow_water_attempt2.py --stage plan
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=repro \
  .venv/bin/python repro/src/shallow_water_attempt2.py \
  --stage prepare-data --device mps --block-pid 18410
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=repro \
  .venv/bin/python repro/src/shallow_water_attempt2.py \
  --stage train --device mps --epochs 100 --batch-size 16 \
  --model FluxNet_SW_LAP_pf --max-new-epochs 1 --block-pid 18410
# Review the measured epoch_seconds and resource use before resuming. The target
# remains fingerprinted at 100 epochs; this gate does not create a 1-epoch run.
tail -n 2 outputs/shallow_water_attempt2/events.jsonl
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=repro \
  .venv/bin/python repro/src/shallow_water_attempt2.py \
  --stage train --device mps --epochs 100 --batch-size 16 --block-pid 18410
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=repro \
  .venv/bin/python repro/src/shallow_water_attempt2.py \
  --stage evaluate --device mps --evaluation-batch-size 2 --block-pid 18410
PYTHONPATH=repro .venv/bin/python repro/src/audit_shallow_water_attempt2.py \
  --output-dir outputs/shallow_water_attempt2 --rehash-data
```

While training is active, the non-evaluative protocol snapshot is refreshed
without competing for HDF5 read bandwidth:

```bash
PYTHONPATH=repro .venv/bin/python repro/src/audit_shallow_water_attempt2.py \
  --output-dir outputs/shallow_water_attempt2 --mode protocol
```

## Spinodal boundary

Spinodal decomposition is not part of this shallow-water attempt. The pinned
repository publishes only CUDA `.cu` generators and no dataset, checkpoint,
tag, release asset, or DAT-to-HDF5 converter. A subsequent source audit found
that the double-precision finite-difference generator can be ported faithfully
to CPU apart from CURAND bitstream identity, while the released FluxNet-D model
can execute on MPS. That makes a separately preregistered 128x128 mechanism
study feasible after this attempt, but it does not make the paper's literal
1024x1024, 20/100-trajectory evaluation or same-A800 17.3x timing comparison
locally reproducible. This driver therefore records spinodal as
`not_attempted_separate_protocol`; no spinodal empirical conclusion may be
drawn from the shallow-water artifacts.
