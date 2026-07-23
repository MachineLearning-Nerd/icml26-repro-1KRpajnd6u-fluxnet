# Executive summary

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_flux_exec_20260719", "created_at": "2026-07-19T16:08:00+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-19T16:08:00+00:00"}
-->
The last confirmed official verdict is **4/6** at Space SHA
`74ba2b60cc42e38011a6d1659ddc04869cc5c677`: structural C1 and C2 are
`verified`, while empirical C3 is `inconclusive` because the prior logbook ran no
trained PDE benchmark. This revision preserves those proofs and adds one bounded,
materially different attempt on the complete released periodic traffic benchmark.
No score increase is claimed before a fresh official verdict.

| Judged claim | Local assessment | Decisive evidence |
| --- | --- | --- |
| C1 — discrete conservation | **Verified** | 96 1D + 48 2D clean-room plans and 36 official-head configurations; machine-scale residuals |
| C2 — structural bounds | **Verified for L/U heads** | Positive lower/upper slack in clean-room and official-core checks; D-head counter-control prevents overclaiming |
| C3 — empirical PDE behavior | **Traffic portion supported; aggregate claim toy** | Full traffic split: FluxNet-D/ResNet final MAE `0.003052/0.013734`, ratio `0.2222`; shallow water/spinodal unexecuted |

## Scope & cost

| Item | This revision | Full aggregate C3 replication |
| --- | --- | --- |
| Data | Complete released periodic traffic split | Traffic + shallow water + spinodal |
| Hardware | Local Apple MPS | Multi-seed accelerator runs |
| Training | 2 models × 100 epochs, one seed | Paper seed matrix across all tasks |
| Wall time | 1,511.6 s training | Not attempted |
| Cost | USD 0 | Separate compute authority needed |
| Outcome | Traffic portion supported; `toy` challenge evidence | Not claimed |

All 100 test trajectories and 50 rollout steps are retained. Independent
recomputation matches every stored metric exactly, FluxNet-D beats ResNet-AR in
all seven released categories, and the complete suite passes **8/8** tests.

---
<!-- trackio-cell
{"type": "figure", "id": "cell_flux_poster_20260719", "created_at": "2026-07-19T16:12:00+00:00", "title": "FluxNet reproduction poster", "pinned": true, "pinned_at": "2026-07-19T16:12:00+00:00"}
-->
````html
<!-- poster_embed.html -->
<iframe src="poster_embed.html" title="FluxNet reproduction audit poster" width="100%" height="820" loading="lazy"></iframe>
````
