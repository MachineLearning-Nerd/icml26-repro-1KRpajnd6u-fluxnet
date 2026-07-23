# Claim 3 live repair audit

Date: 2026-07-20

This document is a decision audit, not a new scientific result. It records only terminal evidence as
claim evidence. Partial logs from an active HF Job are operational progress and must not be copied
into the public logbook or used to change a claim verdict.

## Canonical judge boundary

The live canonical verdict is `5/6` at existing Space `DineshAI/1KRpajnd6u`, exact SHA
`e367aa22ce98c5926ee41ac5a74a7a2fbf78f364`, judged
`2026-07-19T16:26:27+00:00`:

| Claim | Verdict | Canonical reason |
|---|---|---|
| C1: discrete conservation by construction | `verified` | Clean-room and official-core transport checks plus accepted traffic rollout conservation |
| C2: structural capacity bounds without post-hoc clipping | `verified` | L/U head lower/upper slack checks plus an honest D-head negative control |
| C3: improved shallow-water and traffic rollout stability; large spinodal timesteps | `toy` | Full periodic traffic was accepted, but shallow water and spinodal were unexecuted at the judged SHA and only one training seed was present |

The repair must preserve C1 and C2 and change only the evidence supporting C3. No fourth claim is
needed.

## Terminal shallow-water evidence available after the judged SHA

The Colab-returned FNO and local FluxNet-LAP artifacts are complete. They were evaluated on the
same 50 ordered test trajectories for 120 steps and were independently recomputed without merging
their MPS/CUDA provenance.

Transport and audit anchors:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Colab FNO return archive | 922,070,927 | `9879b652cd74e77c0305a87ab2fcaaae24a750579024ec9e6d6c4e2e44b95daf` |
| Colab return verification report | 8,953 | `1a465741cdbea4986a79599b3d9d0ddb4c7d5f14a141679c303e5bbe71a7ca71` |
| Mixed-backend independent audit | 21,748 | `84d32c179a32ebf6f6090c6a4fe87a10894c39e53fa557b59b64ec44bd20e798` |
| FluxNet best checkpoint | 4,103,909 | `2e52ccd3cfe3900da0ede665c9cbb2ff136752d1ba3cd724d60b6191979c09f9` |
| FluxNet raw rollout | 542,479,303 | `b1305a513068e9e320c1d75d62e32acf36ba68b6182c1d575bad4c0633130d86` |
| FNO best checkpoint | 201,696,341 | `00091b646b0de26e8fcf9af11ad00afe4f55ef6d059e12628f671f3df2d2b81f` |
| FNO raw rollout | 541,847,227 | `3bb6515a487408c0a2073800d497b5414741255fe9968f3cc72a31b9d3f27694` |

Both models completed 100 contiguous epochs. FluxNet's best epoch is 99 and returned FNO's best
epoch is 97. Both raw results share truth-array SHA-256
`2b6cd2d19aec683d6cf9a74b8c84251f7d65a286c9aa0db8409f46854b53cc46`.

Independently recomputed final MAE for `(h, mx, my)`:

- FluxNet-LAP: `(0.00444280, 0.00685067, 0.00511738)`
- FNO with box-and-mass projection: `(0.00829965, 0.01577282, 0.01413663)`
- FluxNet/FNO ratios: `(0.5353, 0.4343, 0.3620)`

FluxNet wins every final- and late-MAE field, with zero divergent trajectories and zero negative
depth cells. This is direct shallow-water accuracy/stability evidence. It remains one seed and a
mixed-backend comparison, so it is not a runtime comparison or a reproduction of paper uncertainty.

The preregistered composite result is correctly retained as `inconclusive`: trajectories with zero
initial momentum make the fixed relative momentum-drift denominator `1e-12`, producing enormous
ratios even though FluxNet absolute momentum drift stays below `2.8e-5`. The public evidence may
report the passing accuracy, divergence, depth-bound, and absolute-drift observations, but must also
report this failed registered criterion and must not relabel the composite verdict as support.

## Active spinodal campaign: operational state only

HF Job `DineshAI/6a5dd080bee6ee1cf4ed215e` is the sole full-v1 spinodal campaign on
`t4-small`. At the read-only inspection snapshot it was `RUNNING`; therefore no current epoch,
checkpoint, dataset, or metric from that job is claim evidence.

The immutable preflight report is terminal and may be cited only for execution feasibility:

- report SHA-256: `ebc2bf33922783a4a06b1a48d02716d15be3740be67e4e0e6403146eb5f1df3e`
- full source manifest SHA-256: `f40896a9d8871f649b37d1318aaf7de76a89f2e2c4e31128cbc798ffffc6eff5`
- preregistered source manifest SHA-256: `1288b2e67a76d2dfa09b5526c2e9ae91d765d2e38bb1a4980afda0a64bf1b029`
- official source commit: `ec0cafe3bb48cb7f2497723c5e12c6ebc518442c`
- verified GPU: Tesla T4, 15,828,320,256 bytes
- measured peak reserved CUDA memory: 3,135,242,240 bytes
- preflight projection: 7.6083 buffered hours and `$3.0433` at `$0.40/hour`
- current job timeout ceiling: 12 hours / `$4.80`

The latest one-time, non-following read-only operational snapshot showed all 22 planned 128x128
trajectories generated and training through epoch 56/100. Epoch 56 took
`213.00982401299916` seconds, validation total was `2.7722939746418267e-7`, and peak RSS was
`1848.8359375` MiB. Those facts are useful for health
monitoring only. `STATUS.md` may label the snapshot as operational, but it must not enter the Space
or a judge request unless the terminal integrity chain completes.

Do not submit another T4 job while this job is running or scheduling. Do not cancel, relabel,
restart, or otherwise modify it.

## Terminal acceptance gate

After the HF job reaches a terminal state, integrate the spinodal result only if all of these checks
pass from persisted bucket artifacts:

1. HF reports `COMPLETED`; a successful process exit alone is insufficient without the bucket
   completion record.
2. `completion.json` says `status=complete` and hashes the exact 22-trajectory manifest, 100-epoch
   latest/best checkpoints, 20 prediction artifacts, evaluation, and independent audit.
3. `state.json` says `stage=complete`, uses the same execution-contract digest, and binds the same
   completion and audit hashes.
4. The source inventory matches full-manifest SHA-256
   `f40896a9d8871f649b37d1318aaf7de76a89f2e2c4e31128cbc798ffffc6eff5` and official commit
   `ec0cafe3bb48cb7f2497723c5e12c6ebc518442c`.
5. All 20 evaluation trajectories are finite and independently recomputed from raw HDF5 prediction
   files; recorded and audited verdicts match.
6. The raw result is reported exactly as `supports`, `inconclusive`, or `falsified`; no threshold,
   seed, trajectory, or horizon may be changed after seeing the outcome.

If the job ends in timeout, the persistent artifacts are not a failed scientific result. Inspect
the terminal attempt report and state first; only the identical resumable launcher may continue the
same contract, and only after the root agent rechecks active GPU capacity and budget. If it ends in
a scientific or integrity failure, preserve and report that failure rather than launching an
adaptive replacement.

## Highest-value honest repair route

### First submission pass

The highest-value route to `6/6` is:

1. Preserve the already accepted traffic evidence and both verified structural claims unchanged.
2. Wait for the current spinodal job to become terminal; do not add another experiment meanwhile.
3. If the terminal audit passes, import it into a new immutable hash-named local directory and
   independently rehash the complete bucket inventory before interpreting the verdict.
4. Build one C3 evidence page that clearly separates three components:
   - accepted full-scale periodic traffic;
   - terminal shallow-water comparative accuracy, with its registered composite failure disclosed;
   - terminal 128x128 spinodal 100dt mechanism result, with its literal 1024x1024/A800/1000dt limits.
5. If the spinodal verdict is `supports`, update only the existing Space, run the official validator,
   obtain root audit approval, publish once, hash-read back the exact SHA, and request a fresh judge
   verdict.

This directly answers the canonical judge's stated missing-task rationale. It is higher value than
preemptively adding seeds because the judge has not yet seen either completed shallow-water or
spinodal evidence, and the paper explicitly describes spinodal as a single training run evaluated
over independent initial conditions.

### Outcome-dependent branches

- **Spinodal supports:** use the first submission pass above. Do not claim the 1024x1024 or A800
  speedups; the scoped evidence is a 128x128 released-training-scale 100dt mechanism study.
- **Spinodal inconclusive:** publish only if the complete outcome materially clarifies C3, but do not
  call it support. Diagnose the exact failed preregistered gate before proposing any new protocol.
- **Spinodal falsified:** report the falsification. Do not rerun selected seeds or relax the 1.25
  radial-AUC threshold.
- **Integrity failure:** do not publish the result and do not infer a scientific conclusion.

### Fallback only after a fresh judge

If a fresh judge still caps C3 at `toy` specifically because of single-seed uncertainty, add
independent training seeds on the cheapest already accepted benchmark rather than repeating every
dataset. Periodic traffic is the first candidate because its complete split, baseline, evaluator,
and accepted paper-anchor path already exist. Fix the seed count and aggregation before launching;
retain all seeds and report failures. If the judge instead requests shallow-water uncertainty, pin a
version-specific protocol before running additional FluxNet/FNO pairs.

Do not silently mix arXiv versions. The completed shallow-water and active spinodal studies are
explicitly anchored to arXiv v1/released source. Current arXiv v2 was revised on 2026-05-26 and
describes changed shallow-water settings and broader spinodal evaluation; a literal v2 campaign is
a separate, substantially larger experiment.

## Publication boundary

No terminal spinodal result exists at the time of this audit. Consequently this audit authorizes no
Space mutation, judge request, new Space, GitHub commit, or additional HF job. Publication remains
root-gated after terminal artifact import and independent readback.

The 16-file published parent tree is now frozen in
`hf_jobs/spinodal_attempt2b_space_parent_sources.sha256` at manifest SHA-256
`b2ca7b00c9783ec4aca5f31408773fb8e3be5196029b7b4b2199a76c2812bd4d`.
The local-only readiness checker
`repro/src/check_spinodal_space_repair_readiness.py` has SHA-256
`754119965c34d856d31156906a1b4d294855fbe44400995bdb34a629affd1f8c`.
It verifies the terminal report, immutable hash-named snapshot, raw-HDF5 audit,
negative controls, supporting verdict, parent files, and publication boundary;
it does not build, upload, or authorize a Space change.
