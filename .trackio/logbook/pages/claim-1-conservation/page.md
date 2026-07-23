# Claim 1 — Discrete conservation

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_flux_c1_01", "created_at": "2026-07-19T16:12:02+00:00", "title": "Claim, method, and assessment"}
-->
**Judged claim:** FluxNet guarantees discrete conservation by construction
through local transport operators on regular grids.

**Assessment: verified.** A clean-room NumPy audit exercises periodic directed
transport independently of the author implementation. It covers 96 signed 1D
plans and 48 signed 2D plans, with maximum absolute global-sum errors
`8.44e-15` and `1.51e-14`. A second check imports the unmodified author `N`, `L`,
and `U` 1D heads from commit
`ec0cafe3bb48cb7f2497723c5e12c6ebc518442c`; 36 seeded configurations have
maximum float32 mass error `1.91e-6`.

The traffic attempt independently extends this check through trained rollout:
over 100 trajectories and 50 learned steps, FluxNet-D's maximum relative sum
drift is `3.34e-7`, while the matched residual baseline reaches `0.2293`.

---
<!-- trackio-cell
{"type": "code", "id": "cell_flux_c1_02", "created_at": "2026-07-19T16:12:03+00:00", "title": "Executed structural certificate", "command": [".venv/bin/python", "repro/run_audit.py"], "exit_code": 0, "duration_s": 1.586}
-->
````bash
$ PYTHONPATH=repro .venv/bin/python repro/run_audit.py
````

````output
clean_room_1d_cases=96
clean_room_2d_cases=48
max_1d_mass_error=8.43769498715119e-15
max_2d_mass_error=1.509903313490213e-14
official_configurations=36
official_max_mass_error_float32=1.9073486328125e-06
official_commit=ec0cafe3bb48cb7f2497723c5e12c6ebc518442c
````

Periodic pairing and the shared sender/receiver flux are essential assumptions.
The separately pinned `official/` checkout is clean, and no files are imported
from an unpinned author revision.
