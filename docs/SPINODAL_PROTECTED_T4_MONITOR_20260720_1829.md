# Protected spinodal T4 monitor - 2026-07-20 18:29 IST

This is a paper-local, read-only operational snapshot. It did not inspect or
copy the mutable bucket prefix, import an artifact, run a test or scientific
command, submit/cancel/alter an HF Job, signal a process, or publish anything.

## Job identity and state

- Job: `DineshAI/6a5dd080bee6ee1cf4ed215e`
- Flavor: `t4-small`
- State: `RUNNING`
- Labels:
  - `paper=1KRpajnd6u`
  - `purpose=spinodal-attempt2b-full-v1`
  - `attempt=3`
  - `name=fluxnet-spinodal-attempt2b-full-v1`
- Started: `2026-07-20T07:38:49.587000+00:00`
- Observed running duration: `19269` seconds

Because the Job has not ended naturally, the terminal-only bucket and import
gate remains closed. No bucket inventory or terminal artifact was read.

## Immutable bootstrap and dataset evidence in logs

The non-following Job log confirms:

- h5py `3.14.0`, HDF5 `1.14.6`;
- frozen full-source-manifest SHA-256
  `f40896a9d8871f649b37d1318aaf7de76a89f2e2c4e31128cbc798ffffc6eff5`;
- 17 source entries verified by the bootstrap;
- all 22 preregistered trajectories generated: one train, one validation,
  and 20 test trajectories;
- aggregate generated-dataset SHA-256
  `d437b96f6c7f8b0b5ff4b530af0d1fd8b74f2e57a9422f59c5e008b0fb193b06`.

These identities match the frozen terminal-import contract documented in
`docs/claim3_spinodal_terminal_import.md`.

## Latest completed operational epoch

The latest complete log record is epoch `87/100`:

- epoch seconds: `213.17408462100138`;
- cumulative training seconds: `18560.764049236`;
- learning rate: `0.0005`;
- peak RSS: `1849.03125` MiB;
- train total: `3.290761738227081e-08`;
- validation total: `2.793410769437902e-07`;
- `is_best: false`.

The latest best-marked record visible in the snapshot is epoch `84`, with
validation total `6.974906411175573e-08`. These are operational partial values,
not claim evidence and not an evaluation verdict.

At the observed roughly 213 seconds per epoch, the remaining 13 training
epochs alone are approximately 46 minutes. Terminal evaluation, audit, hashing,
and persistence follow training, so this is not a completion ETA guarantee.

## Gate decision

Leave the Job untouched. Do not import or inspect terminal bucket artifacts
until HF reports that this exact Job ended naturally. Only then may a read-only
terminal audit require the frozen source and dataset hashes, 100 contiguous
epochs, completion/state links, evaluation, independent audit, terminal
sentinels, and stable bucket inventory. This snapshot establishes no terminal
SUCCESS and authorizes no publication.
