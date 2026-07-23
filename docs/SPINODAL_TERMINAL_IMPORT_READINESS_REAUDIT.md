# Spinodal terminal import readiness re-audit

## Outcome

The terminal import chain is ready, hash-consistent, and correctly blocked
while protected HF Job `DineshAI/6a5dd080bee6ee1cf4ed215e` remains naturally
`RUNNING`. Do not run the importer until HF reports `COMPLETED`.

At `2026-07-20T12:35:28Z`, the job had reached epoch 80/100 after 17,801
running seconds. Its latest epoch took 213.30 seconds. The preflight projected
7.608 buffered hours for the whole campaign, so natural completion may be
roughly 1.3 hours away if the remaining training, evaluation, HDF5 writes, and
audit behave as projected. This is an estimate, not a timeout or intervention
trigger.

No partial epoch, checkpoint, or metric is claim evidence.

## Exact judged parent

The official state is 5/6 at existing Space `DineshAI/1KRpajnd6u`, SHA
`e367aa22ce98c5926ee41ac5a74a7a2fbf78f364`, judged on 19 July 2026:

| Claim | Verdict |
| --- | --- |
| C1 conservation | `verified` |
| C2 structural bounds | `verified` |
| C3 empirical rollouts | `toy` |

The live Space is still running at the same SHA. The older local readiness
manifest covers 16 scoped `.trackio/logbook` paths. The complete remote parent
is now independently frozen as an exact 18-file manifest, including
`.gitattributes` and `style.css`, at SHA-256
`a10b68fb0ea8e748f74c5cb3bd984da495cd3808c6c48844fc523968aff34696`.
A read-only re-download at the exact revision matched every path and hash; its
deterministic tree SHA-256 is
`3bd97d175c9a5d230a7589492dda46730445855fc23e9b580cab426db5f2b759`.

## Live nonterminal state

The protected Job has the expected T4 flavor, pinned image, bootstrap command,
labels, and mounts. It must not be stopped, cancelled, relabelled, restarted,
or otherwise modified.

The current persistent bucket prefix contains 32 files and 1,343,490,816
bytes. All 22 generated source HDF5 trajectories and aggregate dataset SHA-256
`d437b96f6c7f8b0b5ff4b530af0d1fd8b74f2e57a9422f59c5e008b0fb193b06`
are present. The completion record, evaluation, audit certificate, and 20
prediction HDF5 files are correctly absent. This proves only that the bucket is
nonterminal; it must not be imported or interpreted yet.

## Frozen import chain

All files in both frozen source manifests pass checksum verification:

| Component | SHA-256 |
| --- | --- |
| Terminal importer | `154f322af249f2370646deb3d9981d5b7a79e5d1d4fa604ec567cb748461d972` |
| Full source manifest | `f40896a9d8871f649b37d1318aaf7de76a89f2e2c4e31128cbc798ffffc6eff5` |
| Full runner | `6e574948dcf9d50a8d21ecb62924d3d6ede82ca2d609490c2700abb44a1f9762` |
| Independent raw-HDF5 auditor | `ead541330615c12005252f551e95f840a4abf79f04e19c6311ee2e3b709ae44f` |
| Full launcher | `c288d78dbac6bc0c6e05701b0cb2cf6d6aac0214c1c532b75c794e265a4154d8` |
| Bootstrap | `cc72aad692a6a6555a05d36273827057fb5288f25711397bfcff88c159625049` |
| Preregistration | `f2f12fb97dda6428814f99d8ec101c08939040641d9e0e4021c59f4c7f88e314` |
| Readiness checker | `754119965c34d856d31156906a1b4d294855fbe44400995bdb34a629affd1f8c` |
| Terminal-readiness manifest | `2825cc953f4583af9b02a9103cb65250278fe6c28a399f322b03de3a837758a3` |

The local import environment has `h5py 3.16.0`, NumPy `2.5.1`, PyTorch
`2.7.1`, and the HF CLI. About 14 GiB was free at audit time. Recheck disk
space immediately before importing because the terminal raw prediction files
will materially increase the current 1.34 GB bucket size.

## One action after natural completion

After, and only after, HF reports `COMPLETED`, run once:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  repro/src/import_verify_hf_spinodal_attempt2b.py
```

The importer performs one terminal Job inspection, before/after bucket
inventories, a safe download, complete path/size/hash verification, exact
22-trajectory dataset verification, 100-epoch checkpoint verification, and
20 ordered prediction-HDF5 checks. It then independently recomputes all raw
metrics and the registered verdict, detects six tampering controls, and keeps
only an immutable content-inventory-hash-named local snapshot plus a new
terminal verification report. It never publishes and never overwrites an
existing import.

The raw support rule is frozen:

- all 20 trajectories finite;
- maximum relative mass drift at most `1e-5`;
- mean final MAE at most `0.0432`;
- radial-AUC ratio upper 95% bound at most `1.25`.

A radial-AUC lower 95% bound above `1.25` is `falsified`; every other result is
`inconclusive`. The claim is limited to a 128x128 released-training-scale 100dt
mechanism study, not literal 1024x1024, A800 timing, 1000dt, or arXiv-v1 Table 5.

After a successful import, run the local-only readiness gate:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  repro/src/check_spinodal_space_repair_readiness.py
```

It passes only for a verified `supports` result with the immutable snapshot,
raw-HDF5 audit, all six controls, and scoped parent files intact. Passing means
eligible for root candidate review only; it grants no publication authority.

## C3 score and same-Space boundary

The only remaining point is C3 moving from `toy` to `verified`, so a fresh
judge could raise 5/6 to 6/6. Even a supporting spinodal result does not
guarantee that outcome. The public repair must retain:

- accepted full-scale periodic traffic evidence;
- shallow-water comparative accuracy and stability wins;
- the shallow-water registered composite `inconclusive` result caused by the
  relative momentum-drift denominator edge case;
- one-seed and mixed-backend shallow-water limitations;
- one-training-seed spinodal scope and its 128x128/100dt limit.

No deterministic Space candidate builder exists; scientific wording remains a
root-reviewed outcome-dependent step. A fail-closed publisher now exists so
the eventual approved candidate cannot omit the two previously unscoped parent
files or race a changed live parent. It:

- accepts only a full, regular, symlink-free 18-file candidate tree;
- permits changes only to C3, `logbook.json`, and dependent conclusion,
  executive-summary, and index pages;
- preserves C1, C2, `.gitattributes`, `style.css`, and every other path by
  exact hash;
- requires the terminal supporting-import readiness gate;
- rechecks the live 18-file parent before mutation;
- passes exact SHA `e367aa22...f364` as `HfApi.create_commit(parent_commit=...)`;
- requires root-bound parent, candidate-tree, terminal-report, and validator
  approvals plus `FLUXNET_SPACE_REPAIR_APPROVED=1`;
- downloads the resulting exact commit and compares the complete path/hash
  union before writing a receipt.

The complete repair-source manifest has SHA-256
`3bd6b340e895453f8c197e0adc81f6fdd6a5bdd2ffa0caff78c6d7c4ffd9bc44`.
Its publisher SHA-256 is
`c9f97d40e278636eba9d62969a9dede604bd39a1799a3524388785ff3930dd53`.
The current parent can be verified without publication:

```sh
uv run repro/src/publish_spinodal_space_repair.py --verify-parent-only
```

If the imported spinodal verdict is `supports`, root must separately:

1. recheck that the live parent is still `e367aa22...f364`;
2. download and preserve the complete 18-file parent;
3. update only the C3 narrative and dependent summary/conclusion/navigation;
4. preserve C1 and C2 byte-for-byte;
5. run the official logbook validator;
6. run the non-mutating candidate check, which prints the exact hash-bound
   approval command:

   ```sh
   uv run repro/src/publish_spinodal_space_repair.py --check-only \
     --candidate-dir /absolute/path/to/full-18-file-candidate
   ```

7. inspect and explicitly run that printed command once;
8. require its exact-union readback receipt before asking for a fresh judge
   verdict.

For `inconclusive` or `falsified`, preserve and report the registered outcome
without relaxing thresholds or selecting new seeds. For an integrity failure,
do not publish or infer a scientific conclusion.

The machine-readable audit and complete 18-file parent inventory are in
`outputs/hf-jobs/spinodal-attempt2b/terminal-readiness-reaudit-20260720.json`.
