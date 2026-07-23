# Claim 3 spinodal terminal import and publication gate

Date prepared: 2026-07-20

## Scope

This is the fail-closed return path for the single authorized spinodal campaign,
HF Job `DineshAI/6a5dd080bee6ee1cf4ed215e`. It does not poll the active Job,
start or stop any Job, run a project test suite, mutate a bucket, or publish a
Space. Partial epochs and mutable checkpoints are operational information only.

The frozen execution anchors are:

| Item | SHA-256 |
|---|---|
| Terminal importer/verifier | `154f322af249f2370646deb3d9981d5b7a79e5d1d4fa604ec567cb748461d972` |
| Full source manifest | `f40896a9d8871f649b37d1318aaf7de76a89f2e2c4e31128cbc798ffffc6eff5` |
| Full runner | `6e574948dcf9d50a8d21ecb62924d3d6ede82ca2d609490c2700abb44a1f9762` |
| Independent raw-HDF5 auditor | `ead541330615c12005252f551e95f840a4abf79f04e19c6311ee2e3b709ae44f` |
| Frozen launcher | `c288d78dbac6bc0c6e05701b0cb2cf6d6aac0214c1c532b75c794e265a4154d8` |
| Bootstrap | `cc72aad692a6a6555a05d36273827057fb5288f25711397bfcff88c159625049` |
| Published-parent 16-file manifest | `b2ca7b00c9783ec4aca5f31408773fb8e3be5196029b7b4b2199a76c2812bd4d` |
| Local-only Space readiness gate | `754119965c34d856d31156906a1b4d294855fbe44400995bdb34a629affd1f8c` |
| Official source commit | `ec0cafe3bb48cb7f2497723c5e12c6ebc518442c` |
| Completed generated dataset expected by the active contract | `d437b96f6c7f8b0b5ff4b530af0d1fd8b74f2e57a9422f59c5e008b0fb193b06` |

The remote source of truth is
`hf://buckets/DineshAI/1KRpajnd6u-artifacts/hf-jobs/spinodal-attempt2b/full-v1`.

## One terminal command

Run this only after natural Job termination. Do not manually inspect first: the
importer itself performs exactly one Job inspection and refuses every status
except `COMPLETED`.

```bash
cd /Users/dineshjinjala/Documents/AllCode/ICMLPapers/papers/icml26-repro-1KRpajnd6u-fluxnet
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  repro/src/import_verify_hf_spinodal_attempt2b.py
```

The importer then performs this fixed sequence:

1. verifies the exact Job ID, `COMPLETED` status, T4 flavor, pinned image,
   bootstrap command, labels, and mount evidence;
2. records one recursive bucket inventory, requires all terminal sentinels,
   and downloads the prefix with `hf buckets sync` into a temporary directory;
3. records one post-download inventory and rejects any change during transport;
4. rejects symlinks, unexpected paths/sizes, missing files, and every mismatch
   in the full local content inventory;
5. verifies the immutable execution-contract digest, official commit, frozen
   source-manifest digest, exact 22-trajectory plan, per-file SHA-256 values,
   aggregate dataset digest, derived frame counts against stored HDF5 shapes
   and time grids, 100 contiguous training epochs, and safely loaded
   best/latest checkpoint identities;
6. verifies the exact 20 ordered raw predictions and sidecars, evaluation,
   completion record, terminal state, and independent audit links;
7. reruns `audit_spinodal_attempt2b.py` against all returned source and
   prediction HDF5 files and compares raw recomputed metrics and verdict;
8. proves rejection of nonterminal completion, state-hash, manifest-digest,
   prediction-hash, verdict, and recorded-metric tampering controls;
9. moves the verified tree to a new content-inventory-hash-named child under
   `outputs/hf-jobs/spinodal-attempt2b/verified-returns/` and writes
   `outputs/hf-jobs/spinodal-attempt2b/terminal-import-verification.json`.

No existing evidence path is overwritten. A failed check retains no imported
tree and grants no publication authority.

After a successful import, the next local-only gate is:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  repro/src/check_spinodal_space_repair_readiness.py
```

It exits blocked unless the terminal import is verified, all 20 raw-HDF5
predictions pass independent recomputation, all six negative controls are
detected, the verdict is exactly `supports`, the immutable snapshot binding is
intact, and all 16 published parent files still match. Passing this command
means only “eligible for root candidate build”; it performs no HF request and
grants no publication authority.

## Readback and decision

After a successful import, read the result without modifying it:

```bash
sha256sum outputs/hf-jobs/spinodal-attempt2b/terminal-import-verification.json
jq '{status, job, local_snapshot_sha256, completion_sha256, training, result, negative_controls_detected, publication_gate}' \
  outputs/hf-jobs/spinodal-attempt2b/terminal-import-verification.json
```

Interpret the preregistered verdict literally:

- `supports`: eligible for root review of a C3 evidence update;
- `inconclusive`: disclose which fixed gate missed; do not call it support;
- `falsified`: preserve and disclose the falsification;
- any integrity/import failure: do not publish or infer a scientific result.

## Existing-Space-only publication gate

Even a verified `supports` result does not itself authorize publication. The
only allowed target is the existing Space `DineshAI/1KRpajnd6u`, whose expected
pre-update SHA is `e367aa22ce98c5926ee41ac5a74a7a2fbf78f364`. Creating a new Space is
forbidden for this repair.

After a supporting import, root must separately review the exact hash-named
snapshot and report, update only the Claim 3/logbook conclusions warranted by
the terminal metrics while preserving the shallow-water registered failure and
spinodal scale limits, run the official validator, and approve one publication.
The validator command for a locally reviewed candidate is:

```bash
python ../../tmp/validate_icml_logbook.py --space DineshAI/1KRpajnd6u
```

There is deliberately no upload command in this handoff. After approval, the
existing Space must be published once and its resulting commit SHA and complete
file hashes read back before any fresh judge request.
