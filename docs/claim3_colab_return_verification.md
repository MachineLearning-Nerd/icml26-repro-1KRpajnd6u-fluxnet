# Claim 3 Colab FNO return verification

## Boundary

The CUDA FNO run in Colab is authoritative. The incomplete local epoch-23
checkpoint is retained only as a fallback/control and must not be resumed while
the Colab run is active. The local CARS evaluation retries are not FNO evidence:
the driver correctly rejects their incomplete local FNO checkpoint.

The final Colab cell creates
`fluxnet_fno_cuda_results.tar.gz` and prints its SHA-256 and byte size. Preserve
that printed digest next to the downloaded archive. A filename or Google Drive
timestamp is not provenance.

## Fail-closed verifier

`repro/src/verify_colab_fno_return.py` performs a read-only verification by
default. It requires the digest printed by Colab and checks:

- the downloaded archive matches that exact transport digest;
- every tar member is a regular file or directory under the one expected root,
  no duplicate/extra file exists, and extraction stays below 3 GiB;
- the local input bundle is still exactly
  `f717ebee4fa3ce5ce3783c5e6883f39a79199b6d60d0a3c9a5bf38960aa4e050`,
  including wrapper SHA-256
  `be256f981820568d6671e8e963c9c6eba36c785ba33895ff137d05cb6bbdbe43`
  and manifest SHA-256
  `ba9d0f6e4336b21ab00bcd29b3259ebdb860f1765b119286c5a4006c80ef9aab`;
- immutable JSONL `run_plan` events identify CUDA for training and evaluation,
  the exact fixed configuration/source hashes, 100 contiguous training epochs,
  checkpoint resume, training completion, and later evaluation completion;
- the completion and best checkpoints load in PyTorch `weights_only` mode with
  only the NumPy RNG-state types allowlisted, match the independently recomputed
  fingerprint, contain 100 epochs and CUDA RNG state, and hash-match metrics;
- the returned NPZ has exactly 50 ordered 120-step rollouts, exact initial
  states, fixed time grid and case counts, no pickle payload, matching raw hash,
  and independently recomputed summary, per-case, and auxiliary-array metrics.

Run after downloading the archive, substituting the digest printed by Colab:

```bash
PYTHONPATH=repro .venv/bin/python repro/src/verify_colab_fno_return.py \
  --archive /path/to/fluxnet_fno_cuda_results.tar.gz \
  --expected-archive-sha256 <sha256-printed-by-colab> \
  --report outputs/colab/colab_fno_return_verification.json
```

This leaves the verified payload in a temporary directory and removes it after
the report is written. To retain an immutable local copy, add:

```bash
  --import-root outputs/colab/verified-returns
```

The imported child is named by the verified archive SHA-256 and existing paths
are never overwritten. Its `verification_report.json` records the accepted
chain.

## Accepted return

The returned archive passed this verifier at SHA-256
`9879b652cd74e77c0305a87ab2fcaaae24a750579024ec9e6d6c4e2e44b95daf`
and 922,070,927 bytes, exactly matching the final Colab stdout. The external
report and the report inside the hash-named import are byte-identical at
SHA-256 `1a465741cdbea4986a79599b3d9d0ddb4c7d5f14a141679c303e5bbe71a7ca71`.
The accepted FNO final MAE is `(0.00829965, 0.01577282, 0.01413663)` for
`(h, mx, my)`, with zero divergence and zero negative-depth cells.

The completed local-FluxNet/returned-FNO integration is independently checked
by `repro/src/audit_shallow_water_mixed_backend.py` and documented in
`docs/claim3_shallow_water_attempt2_mixed_backend.md`. That audit never copies
the returned FNO over the incomplete local FNO tree.

## Known provenance limit

The research wrapper records `device=cuda`, platform, Torch version, source and
dataset hashes, but not `torch.cuda.get_device_name(0)`. The notebook prints the
GPU model before training; preserve the executed-notebook output if hardware
model attribution is needed. Absence of that output does not invalidate the
CUDA/device, checkpoint, protocol, or numerical artifact checks, but it forbids
a claim about the exact GPU model.
