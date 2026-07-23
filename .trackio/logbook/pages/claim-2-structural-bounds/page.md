# Claim 2 — Structural one-sided bounds

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_flux_c2_01", "created_at": "2026-07-19T16:12:04+00:00", "title": "Claim, certificates, and counter-control"}
-->
**Judged claim:** FluxNet enforces bounds structurally for capacity-constrained
quantities without post-hoc clipping.

**Assessment: verified for the proven single-bound heads.** For valid initial
fields, all 96 deterministic L-head plans retain minimum lower slack `0.01197`,
and all 96 U-head plans retain minimum upper slack `0.01415`. Direct execution of
the unmodified official heads across 12 seeded configurations per head retains
minimum slacks `0.08346` (L) and `0.08297` (U). The implementation uses capacity-
limited transport rather than post-hoc clipping or projection.

The D-head is deliberately separated from Propositions 2 and 3. It averages
lower- and upper-oriented branches and uses a dual consistency loss (DCL), so it
does not inherit a universal hard simultaneous-bound guarantee. A deterministic
negative control finds an otherwise valid D-head output below zero (`-0.05329`)
while preserving mass exactly. This expected counterexample prevents the verified
L/U result from being overstated as a theorem about D-head dual bounds.

---
<!-- trackio-cell
{"type": "code", "id": "cell_flux_c2_02", "created_at": "2026-07-19T16:12:05+00:00", "title": "Executed bounds and D-head control", "command": [".venv/bin/python", "repro/run_audit.py"], "exit_code": 0, "duration_s": 1.586}
-->
````output
minimum_l_head_lower_slack=0.011970382002095162
minimum_u_head_upper_slack=0.014147849553212688
official_minimum_l_head_lower_slack=0.08345960080623627
official_minimum_u_head_upper_slack=0.08297455310821533
d_head_counterexample_found=true
d_head_output_min=-0.053291049378971944
d_head_counterexample_mass_error=0.0
````
