# Claim 3 shallow-water Attempt 2: mixed-backend evidence

## Evidence boundary

This result combines two completed artifacts without merging their provenance:

- `FluxNet_SW_LAP_pf` is the completed local Apple MPS run and rollout under
  `outputs/shallow_water_attempt2/`.
- `FNO_SW_Proj_box_mass_pf` is the completed CUDA run returned from Colab and
  imported under the accepted archive SHA-256 in
  `outputs/colab/verified-returns/9879b652cd74e77c0305a87ab2fcaaae24a750579024ec9e6d6c4e2e44b95daf/`.

The local FNO checkpoint remains untouched and excluded at 23/100 epochs. It
has no raw rollout or metric artifact and is not an input to the comparison.
This is a controlled scientific comparison over identical protocol and test
data, but not a same-device runtime or bitwise-determinism comparison.

## Fail-closed audit

`repro/src/audit_shallow_water_mixed_backend.py` independently verifies:

- the Colab stdout transport identity: SHA-256
  `9879b652cd74e77c0305a87ab2fcaaae24a750579024ec9e6d6c4e2e44b95daf`
  and 922,070,927 bytes;
- byte identity between the external transport report and the report inside
  the hash-named verified import;
- separate MPS and CUDA run-plan/checkpoint RNG provenance, 100 contiguous
  epochs for each selected artifact, and evaluation after completion;
- the same fixed config, official commit, six source hashes, wrapper, manifest,
  and aggregate dataset identity;
- all 120 local HDF5 files and all 120 HDF5 members inside the original Colab
  input bundle against the manifest;
- the exact same 50 ordered test identities, category order, time grid, and raw
  truth tensor across the two backends;
- every recorded summary, per-case result, and auxiliary raw metric array by
  independent recomputation; and
- the preregistered controlled comparison and five validator calibration
  controls.

The Mac-created input tar contains 126 AppleDouble `._*` metadata companions
(20,538 bytes total). The audit accepts only companions whose logical target is
an expected file or directory, bounds their sizes, records their inventory
digest, and rejects every other extra member. These files are inert on the
Colab Linux runtime and remain bound by the complete input-bundle hash.

Run the audit without training or evaluating either model:

```bash
PYTHONPATH=repro .venv/bin/python repro/src/audit_shallow_water_mixed_backend.py \
  --local-output-dir outputs/shallow_water_attempt2 \
  --verified-fno-dir outputs/colab/verified-returns/9879b652cd74e77c0305a87ab2fcaaae24a750579024ec9e6d6c4e2e44b95daf \
  --transport-report outputs/colab/colab_fno_return_verification.json \
  --archive outputs/colab/fluxnet_fno_cuda_results.tar.gz \
  --colab-input-bundle outputs/colab/fluxnet_colab_fno_cuda_bundle.tar \
  --expected-archive-sha256 9879b652cd74e77c0305a87ab2fcaaae24a750579024ec9e6d6c4e2e44b95daf \
  --expected-archive-bytes 922070927 \
  --write outputs/shallow_water_attempt2/mixed_backend_audit.json
```

The passing report is 21,748 bytes at SHA-256
`84d32c179a32ebf6f6090c6a4fe87a10894c39e53fa557b59b64ec44bd20e798`.
Both raw artifacts contain truth-array SHA-256
`2b6cd2d19aec683d6cf9a74b8c84251f7d65a286c9aa0db8409f46854b53cc46`.

## Result and verdict

Final MAE for `(h, mx, my)` is:

- FluxNet-LAP MPS: `(0.00444280, 0.00685067, 0.00511738)`
- FNO projection CUDA: `(0.00829965, 0.01577282, 0.01413663)`

Late MAE is:

- FluxNet-LAP MPS: `(0.00445029, 0.00673975, 0.00537206)`
- FNO projection CUDA: `(0.00846335, 0.01418782, 0.01302485)`

FluxNet is more accurate in every field at both horizons, with final-MAE ratios
of `(0.5353, 0.4343, 0.3620)` relative to FNO. Both models have zero divergent
trajectories and zero negative-depth cells, and the paper-anchor consistency
flag passes.

The fixed controlled result is nevertheless **inconclusive**, not `supports`.
The preregistered support rule requires FluxNet relative integral drift below
`1e-5` in every field. Some trajectories start with exactly zero momentum L1
norm, so the fixed `+1e-12` denominator produces momentum ratios of `2.126e7`
and `2.678e7`, despite absolute momentum drift below `2.8e-5`. This
post-registration edge case is reported without changing the criterion.

The exact Colab GPU model was not recorded in the returned event ledger. CUDA
selection, Torch CUDA build, checkpoint RNG state, and numerical evidence are
verified, but exact accelerator-name and runtime claims are not made. The
attempt is one seed and does not cover spinodal decomposition or paper
uncertainty estimates.
