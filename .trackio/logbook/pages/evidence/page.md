# Evidence


---
<!-- trackio-cell
{"type": "code", "id": "cell_32a806e8d531", "created_at": "2026-07-17T07:34:51+00:00", "title": "Deterministic Proposition 1–3 audit", "command": ["bash", "-lc", "PYTHONPATH=repro .venv/bin/python repro/run_audit.py"], "exit_code": 0, "duration_s": 1.586}
-->
````bash
$ bash -lc 'PYTHONPATH=repro .venv/bin/python repro/run_audit.py'
````

exit 0 · 1.6s


````output
{
  "claim_1_to_3_independent_transport": {
    "max_1d_mass_error": 8.43769498715119e-15,
    "max_2d_mass_error": 1.509903313490213e-14,
    "minimum_l_head_lower_slack": 0.011970382002095162,
    "minimum_u_head_upper_slack": 0.014147849553212688,
    "one_dimensional_cases": 96,
    "two_dimensional_cases": 48
  },
  "d_head_empirical_only_control": {
    "counterexample_found": true,
    "mass_error": 0.0,
    "output_range": [
      -0.053291049378971944,
      0.7603605228592789
    ],
    "trial": 2
  },
  "official_core_head_check": {
    "configurations": 36,
    "max_mass_error_float32": 1.9073486328125e-06,
    "minimum_l_head_lower_slack": 0.08345960080623627,
    "minimum_u_head_upper_slack": 0.08297455310821533,
    "official_commit": "ec0cafe3bb48cb7f2497723c5e12c6ebc518442c"
  },
  "paper": {
    "arxiv_id": "2602.01941",
    "openreview_id": "1KRpajnd6u"
  },
  "scope": "Propositions 1\u20133 transport heads; no trained PDE benchmark claim."
}

````
