# Reproduction status

- Paper: FluxNet: Learning Capacity-Constrained Local Transport Operators for
  Conservative and Bounded PDE Surrogates
- OpenReview: `1KRpajnd6u`
- arXiv: `2602.01941`
- Author source: `Lan-zs/FluxNet` at
  `ec0cafe3bb48cb7f2497723c5e12c6ebc518442c`
- Last adjudicated official verdict: **5/6** at parent Space SHA
  `e367aa22ce98c5926ee41ac5a74a7a2fbf78f364` (`verified`, `verified`, `toy`),
  judged `2026-07-19T16:26:27+00:00` with medium quality. The existing canonical
  Space now contains the C3 repair at SHA
  `0383d1d9ef5f3ca5d032d52dfdf63f5204ec6052`; it has not yet been rejudged.
- Status: all terminal spinodal evidence has been imported and independently
  audited. The repair is publicly live, exactly read back, and awaiting the
  challenge judge; it does not assume a score change in advance.

## Evidence

- Research integrity is evaluated from fixed protocols, signed/raw arrays,
  independent metric recomputation, negative controls, and content hashes; an
  automated test-suite result is not used as judge evidence.
- 96 signed 1D and 48 signed 2D periodic transport plans conserve global mass
  to at most `8.44e-15` and `1.51e-14`, respectively.
- The 96 valid L-head and 96 valid U-head plans retain minimum one-sided slacks
  of `0.01197` and `0.01415`.
- The unmodified official `N`, `L`, and `U` cores pass 36 seeded CPU checks;
  their maximum float32 mass error is `1.91e-6`.
- A deterministic D-head counter-control detects a lower-bound violation while
  retaining exact mass, matching the paper's stated empirical-only dual-bound
  scope.
- The complete released periodic traffic benchmark (100/50/100 trajectories,
  seven categories, `N=256`, 100 epochs, 50-step rollout) gives final MAE
  `0.003052` for FluxNet-D versus `0.013734` for ResNet-AR; the ratio is `0.2222`
  versus arXiv v1's `0.2189`.
- Independent raw-NPZ recomputation has zero metric drift; FluxNet-D wins in all
  seven categories, has zero divergent trajectories, and preserves relative
  mass to `3.34e-7`.

## Empirical-attempt boundary

This substantive attempt received the official `toy` verdict: it covers the traffic portion
of judged C3 with the released solver and official model classes, but uses one
MPS seed, batch 96, and a post-v1 author checkout. At verdict time,
shallow-water and spinodal experiments were unexecuted. The active materially
different C3 attempt now addresses shallow water without weakening the accepted
traffic evidence.

## Shallow-water Attempt 2 (mixed-backend audit complete)

The arXiv-v1 shallow-water protocol is preregistered in
`docs/claim3_shallow_water_attempt2_preregistration.md`. Its harness fixes the
complete 50/20/50 released split, v1 64-channel/4-block/kernel-3/100-epoch
FluxNet-LAP configuration, and the strongest author-implemented comparison,
FNO with box-and-mass projection. The complete 50/20/50 dataset is generated
and semantically reverified at aggregate SHA-256
`ee1d684d479f989f5588afbd40f493a58079ac9ed10dd97d03b16ad6725473cf`.
Both fixed 100-epoch models passed their one-epoch MPS resource gates with
fallback disabled. FluxNet-LAP has now completed all 100 epochs; the best
checkpoint is epoch 99 with validation total loss `3.7188257717e-6` and SHA-256
`2e52ccd3cfe3900da0ede665c9cbb2ff136752d1ba3cd724d60b6191979c09f9`.
The local process resumed the preregistered FNO box-and-mass baseline and
stopped cleanly at durable epoch 23/100, with best validation total loss
`0.0021455667`. That incomplete local FNO remains untouched and excluded. The
authoritative CUDA continuation completed 100 epochs in Colab, was evaluated,
returned, transport-verified, and imported into a new archive-hash-named
directory. The minimal Colab handoff is `notebooks/fluxnet_fno_colab.ipynb` plus
`outputs/colab/fluxnet_colab_fno_cuda_bundle.tar` (SHA-256
`f717ebee4fa3ce5ce3783c5e6883f39a79199b6d60d0a3c9a5bf38960aa4e050`);
the bundle contains only the research wrapper and verified dataset, not tests.
The returned result archive is 922,070,927 bytes at SHA-256
`9879b652cd74e77c0305a87ab2fcaaae24a750579024ec9e6d6c4e2e44b95daf`,
exactly matching the final Colab stdout.

The complete FluxNet-LAP checkpoint's local 50-trajectory, 120-step rollout
has final MAE `(0.00444280, 0.00685067, 0.00511738)` and late MAE
`(0.00445029, 0.00673975, 0.00537206)` for `(h, mx, my)`, with zero divergence
and zero negative-depth cells. Its raw NPZ is 542,479,303 bytes at SHA-256
`b1305a513068e9e320c1d75d62e32acf36ba68b6182c1d575bad4c0633130d86`.

The preregistered relative-drift statistic exposes a zero-denominator edge
case: some trajectories have zero initial momentum L1 norm. With the fixed
`+1e-12` denominator floor, maximum relative drift is `2.126e7` for `mx` and
`2.678e7` for `my`, although maximum absolute drift is only `2.60e-5` and
`2.78e-5`. This observed outcome is not silently redefined; the fixed direct
support criterion will fail for momentum drift and the undefined-scale
diagnostic must be reported alongside the later comparison.

The independent protocol audit passes against the exact source commit, six
hard-coded source hashes, fixed run configuration, all 120 unique manifest
identities and file sizes, the fixed aggregate dataset hash, and the complete
training/event ledger. Its validator self-calibration detects all five
preregistered tampering controls: a changed headline metric, changed raw metric
array, altered initial condition, swapped trajectory order, and nonmonotonic
time grid. The snapshot report is
`outputs/shallow_water_attempt2/protocol_audit.json`. The separate fail-closed
mixed-backend audit rehashes all 120 HDF5 files locally and in the original
Colab input bundle, proves identical config/source/dataset/test identities,
safely checks both complete checkpoints, recomputes both raw result trees, and
binds the accepted transport report. It passes at
`outputs/shallow_water_attempt2/mixed_backend_audit.json` (SHA-256
`84d32c179a32ebf6f6090c6a4fe87a10894c39e53fa557b59b64ec44bd20e798`).

The Colab return is not trusted by filename. The fail-closed verifier in
`repro/src/verify_colab_fno_return.py` requires the archive SHA-256 printed by
Colab, safely extracts only the fixed result inventory, validates 100 contiguous
CUDA epochs/resume/completion from immutable events, safely inspects checkpoint
provenance, and independently recomputes every FNO raw metric. Optional import
uses a new archive-hash-named directory and never overwrites local evidence; see
`docs/claim3_colab_return_verification.md`. The final integration audit and
result are documented in
`docs/claim3_shallow_water_attempt2_mixed_backend.md`.

FluxNet's final MAE is `(0.00444280, 0.00685067, 0.00511738)` versus returned
FNO's `(0.00829965, 0.01577282, 0.01413663)` for `(h, mx, my)`; FluxNet also
wins every late-MAE field. The paper-anchor flag passes. The preregistered
controlled verdict remains **inconclusive** because the fixed relative
momentum-drift criterion fails on zero initial-momentum denominators, although
absolute FluxNet momentum drift remains below `2.8e-5`. The evidence therefore
supports the accuracy comparison but does not satisfy every fixed direct-support
criterion.

A separate 128x128 spinodal 100dt mechanism study is preregistered. Its
double-precision C++ solver agrees with an independent NumPy oracle, and the
atomic generation/training/evaluation harness fixes 22 trajectories and the
released effective loss. Campaign T4 Job `6a5dd080bee6ee1cf4ed215e` completed
naturally after 100 epochs. The terminal importer bound its 77-file,
2,245,101,804-byte immutable return to the exact image, labels, mounts, source
and dataset identities, then the independent raw-HDF5 audit found 20/20 finite
rollouts, mean final MAE `0.03254118 <= 0.0432`, maximum mass drift
`1.2764e-08 <= 1e-5`, and radial-AUC ratio upper-95 `0.38035 <= 1.25`.
All six preregistered tampering controls were detected. The terminal report is
`outputs/hf-jobs/spinodal-attempt2b/terminal-import-verification.json` at
SHA-256 `426ef46d61c29ab83a7e8f94fc91f9385e0de0db704ee2e144b009ce59a4ca7f`.
This is a bounded 128x128 100dt mechanism study, not literal 1024x1024 or A800
timing parity.

## Publication gates

- [x] Independent source, checkpoint, raw-array, category, and test review.
- [x] Build and remotely read back `reproduction-fluxnet-traffic/repro-bundle:v3`
  (28 files, manifest `c22c9c91d9fd75a434c5628676f2dde8486947e0f1a5a3961640453db6841699`).
- [x] The existing `DineshAI/1KRpajnd6u` Space was repaired with a CAS commit
  from `e367aa22...f364` to `0383d1d9...6052`; all 18 returned files are an
  exact byte-for-byte match to the approved candidate and the runtime is
  `RUNNING`. No new Space was created.
- [x] The current bundled validator is recorded as incompatible with the
  challenge-required canonical OpenReview-ID Space name because it incorrectly
  requires a `repro-` repo slug. The direct exact remote readback and required
  `icml2026-repro` / `paper-1KRpajnd6u` tags are the applicable publication
  checks for this canonical Space.
- [x] Refresh the official verdict dataset: C3 changed from `inconclusive` to
  `toy`, raising the official paper score from 4/6 to 5/6.
