# Reproduction status

- Paper: FluxNet: Learning Capacity-Constrained Local Transport Operators for
  Conservative and Bounded PDE Surrogates
- OpenReview: `1KRpajnd6u`
- arXiv: `2602.01941`
- Author source: `Lan-zs/FluxNet` at
  `ec0cafe3bb48cb7f2497723c5e12c6ebc518442c`
- Status: CPU claim audit complete; GitHub and Trackio publication in progress;
  Hugging Face Space publication will remain queued until the account quota is
  available.

## Evidence

- Five unit tests pass.
- 96 signed 1D and 48 signed 2D periodic transport plans conserve global mass
  to at most `8.44e-15` and `1.51e-14`, respectively.
- The 96 valid L-head and 96 valid U-head plans retain minimum one-sided slacks
  of `0.01197` and `0.01415`.
- The unmodified official `N`, `L`, and `U` cores pass 36 seeded CPU checks;
  their maximum float32 mass error is `1.91e-6`.
- A deterministic D-head counter-control detects a lower-bound violation while
  retaining exact mass, matching the paper's stated empirical-only dual-bound
  scope.
