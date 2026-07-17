"""Run independent and official-core CPU checks for FluxNet Propositions 1–3."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from src.transport import d_head_1d, l_head_1d, periodic_transport_1d, periodic_transport_2d, u_head_1d


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "summary.json"


def simplex(rng: np.random.Generator, directions: int, cells: int) -> np.ndarray:
    return rng.dirichlet(np.ones(directions), size=cells).T


def independent_audit(rng: np.random.Generator) -> dict[str, object]:
    offsets = (-2, -1, 1, 2)
    mass_errors: list[float] = []
    lower_slacks: list[float] = []
    upper_slacks: list[float] = []
    cases = 0
    for length in (5, 17, 63):
        for _ in range(32):
            field = rng.normal(size=length)
            flows = rng.normal(size=(len(offsets), length))
            updated = periodic_transport_1d(field, flows, offsets)
            mass_errors.append(float(abs(updated.sum() - field.sum())))

            lower = float(rng.uniform(-0.6, 0.2))
            lower_field = lower + rng.gamma(shape=1.3, scale=0.8, size=length)
            lower_updated, _ = l_head_1d(
                lower_field,
                rng.uniform(0.0, 1.0 - 1e-10, size=length),
                simplex(rng, len(offsets), length),
                offsets,
                lower,
            )
            lower_slacks.append(float((lower_updated - lower).min()))

            upper = float(rng.uniform(0.7, 2.0))
            upper_field = upper - rng.gamma(shape=1.3, scale=0.35, size=length)
            upper_updated, _ = u_head_1d(
                upper_field,
                rng.uniform(0.0, 1.0 - 1e-10, size=length),
                simplex(rng, len(offsets), length),
                offsets,
                upper,
            )
            upper_slacks.append(float((upper - upper_updated).min()))
            cases += 1

    offsets_2d = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
    mass_errors_2d: list[float] = []
    for shape in ((3, 4), (7, 9), (11, 5)):
        for _ in range(16):
            field = rng.normal(size=shape)
            flows = rng.normal(size=(len(offsets_2d), *shape))
            mass_errors_2d.append(float(abs(periodic_transport_2d(field, flows, offsets_2d).sum() - field.sum())))
    return {
        "one_dimensional_cases": cases,
        "max_1d_mass_error": max(mass_errors),
        "minimum_l_head_lower_slack": min(lower_slacks),
        "minimum_u_head_upper_slack": min(upper_slacks),
        "two_dimensional_cases": len(mass_errors_2d),
        "max_2d_mass_error": max(mass_errors_2d),
    }


def d_head_control(rng: np.random.Generator) -> dict[str, object]:
    """Confirm the paper's limitation: D has no universal hard dual bound."""
    offsets = (-2, -1, 1, 2)
    for trial in range(256):
        field = rng.uniform(0.0, 1.0, size=5)
        output = d_head_1d(
            field,
            rng.uniform(size=5),
            simplex(rng, len(offsets), 5),
            rng.uniform(size=5),
            simplex(rng, len(offsets), 5),
            offsets,
            0.0,
            1.0,
        )
        if output.min() < 0.0 or output.max() > 1.0:
            return {
                "counterexample_found": True,
                "trial": trial,
                "output_range": [float(output.min()), float(output.max())],
                "mass_error": float(abs(output.sum() - field.sum())),
            }
    raise AssertionError("expected a D-head dual-bound counter-control")


def load_author_model(name: str, class_name: str):
    source = ROOT / "official" / "src" / "models" / f"{name}.py"
    if not source.exists():
        raise FileNotFoundError("official source missing; clone and pin it as documented")
    spec = importlib.util.spec_from_file_location(f"author_{name}", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def official_core_audit() -> dict[str, object]:
    """Exercise unmodified official 1D N/L/U heads without training code."""
    L = load_author_model("fluxnet_l_1d", "FluxNet_L_1D")
    U = load_author_model("fluxnet_u_1d", "FluxNet_U_1D")
    N = load_author_model("fluxnet_n_1d", "FluxNet_N_1D")
    errors: list[float] = []
    lower_slacks: list[float] = []
    upper_slacks: list[float] = []
    for seed in range(12):
        torch.manual_seed(seed)
        length = 25
        external = torch.randn(3, 1, length)
        lower_model = L(in_channels=2, base_channels=8, num_blocks=0, neighborhood_size=5, lower_bound=0.2).eval()
        lower_input = 0.2 + 0.8 * torch.rand(3, 1, length)
        with torch.no_grad():
            lower_output, _ = lower_model(torch.cat([lower_input, external], dim=1))
        errors.append(float((lower_output.sum(-1) - lower_input.sum(-1)).abs().max()))
        lower_slacks.append(float((lower_output - 0.2).min()))

        upper_model = U(in_channels=2, base_channels=8, num_blocks=0, neighborhood_size=5, upper_bound=0.8).eval()
        upper_input = 0.8 * torch.rand(3, 1, length)
        with torch.no_grad():
            upper_output, _ = upper_model(torch.cat([upper_input, external], dim=1))
        errors.append(float((upper_output.sum(-1) - upper_input.sum(-1)).abs().max()))
        upper_slacks.append(float((0.8 - upper_output).min()))

        unconstrained = N(in_channels=2, base_channels=8, num_blocks=0, neighborhood_size=5).eval()
        signed_input = torch.randn(3, 1, length)
        with torch.no_grad():
            signed_output, _ = unconstrained(torch.cat([signed_input, external], dim=1))
        errors.append(float((signed_output.sum(-1) - signed_input.sum(-1)).abs().max()))
    return {
        "official_commit": "ec0cafe3bb48cb7f2497723c5e12c6ebc518442c",
        "configurations": 36,
        "max_mass_error_float32": max(errors),
        "minimum_l_head_lower_slack": min(lower_slacks),
        "minimum_u_head_upper_slack": min(upper_slacks),
    }


def main() -> None:
    rng = np.random.default_rng(20260717)
    summary = {
        "paper": {"arxiv_id": "2602.01941", "openreview_id": "1KRpajnd6u"},
        "scope": "Propositions 1–3 transport heads; no trained PDE benchmark claim.",
        "claim_1_to_3_independent_transport": independent_audit(rng),
        "official_core_head_check": official_core_audit(),
        "d_head_empirical_only_control": d_head_control(rng),
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
