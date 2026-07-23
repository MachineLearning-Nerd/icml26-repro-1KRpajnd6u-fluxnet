# Claim 3 — Empirical PDE rollouts and large steps

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_flux_traffic_01", "created_at": "2026-07-19T16:08:01+00:00", "title": "Claim target and preregistered protocol"}
-->
The aggregate judged claim names improved shallow-water and traffic rollout
stability plus large-step spinodal behavior. This attempt targets only the
periodic-traffic component, also listed as anchored Claim 4. The comparison is
arXiv `2602.01941v1`, Table 4: FluxNet-D final MAE `3.48e-3`, ResNet-AR
`15.9e-3`, ratio `0.2189`. The available author source is pinned at
`ec0cafe3bb48cb7f2497723c5e12c6ebc518442c` and postdates v1, so the result is
compared to—but not labeled a literal reproduction of—the v1 multi-seed table.

The executed protocol uses all seven released periodic cases, 100/50/100
trajectories, `N=256`, training horizon `T=4`, test horizon `T=8`, official
32-channel/6-block/kernel-5 classes, FluxNet-D's 11-point stencil, five-step
pushforward training, 100 epochs, DCL weight `0.1`, AdamW, seed 42, and identical
batch 96 for both models. Batch 96 rather than the released 16 is an explicit
local-runtime deviation. Full fields—not only aggregates—are retained.

---
<!-- trackio-cell
{"type": "code", "id": "cell_flux_traffic_02", "created_at": "2026-07-19T16:08:02+00:00", "title": "Executed full traffic attempt", "command": [".venv/bin/python", "repro/src/traffic_v1_empirical.py", "--epochs", "100", "--batch-size", "96", "--device", "mps", "--output-dir", "outputs/traffic_v1_attempt1"], "exit_code": 0, "duration_s": 1511.6}
-->
````bash
$ OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=repro \
  .venv/bin/python repro/src/traffic_v1_empirical.py \
  --epochs 100 --batch-size 96 --device mps \
  --output-dir outputs/traffic_v1_attempt1
````

````output
assessment=toy
controlled_result=supports
test_trajectories=100 rollout_steps=50 spatial_cells=256
fluxnet_d_final_mae=0.0030518334824591875
resnet_ar_final_mae=0.013734421692788601
observed_final_mae_ratio=0.22220327515221
paper_v1_final_mae_ratio=0.2188679245283019
fluxnet_d_late_horizon_mae=0.002441241405904293
resnet_ar_late_horizon_mae=0.011399900540709496
fluxnet_d_divergent_trajectories=0
resnet_ar_divergent_trajectories=0
fluxnet_d_max_relative_sum_drift=3.3362272233716794e-07
raw_npz_sha256=2ea19da97ebced8173262b56a07a38d8f631639711505dcce0c4cefe9c297a9b
````

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_flux_traffic_03", "created_at": "2026-07-19T16:08:03+00:00", "title": "Independent verification and limitations"}
-->
An independent read of `traffic_rollouts_raw.npz` recomputed every summary and
per-step array with zero numerical drift. All truth and prediction values are
finite; each field has shape `(100,51,256)`. FluxNet-D's final MAE is below the
baseline separately in every category: Case1 `0.003392<0.009897`, Case2A
`0.003347<0.023353`, Case2B `0.003088<0.015667`, Case3_0
`0.002794<0.010610`, Case3m `0.001794<0.013926`, Case3p
`0.003802<0.008833`, and Case4 `0.003060<0.012814`.

The selected checkpoint epochs (`94` FluxNet-D, `87` ResNet-AR) agree exactly
with the minima in the recorded histories. The suite passes `8/8`; the raw NPZ
SHA-256 is `2ea19da97ebced8173262b56a07a38d8f631639711505dcce0c4cefe9c297a9b`.

**Assessment: toy.** This is one MPS seed; trajectory standard deviations are
not seed uncertainties. Batch size and author-source version differ from v1,
and no shallow-water or spinodal model was trained. The result supports the
traffic portion but cannot verify the aggregate empirical claim.
