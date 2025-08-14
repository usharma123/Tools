from typing import Any, Dict, Optional
from pydantic import BaseModel
from fastapi import APIRouter
from textwrap import shorten

router = APIRouter()


class SummarizeInputs(BaseModel):
    markov: Optional[Dict[str, Any]] = None
    ab_test: Optional[Dict[str, Any]] = None
    power_curve: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    forecast: Optional[Dict[str, Any]] = None


@router.post("/tools/summarize_results")
def summarize_results(args: SummarizeInputs) -> Dict[str, Any]:
    paragraphs: list[str] = []
    if args.markov:
        mk = args.markov
        section = []
        if "stationary_estimate" in mk:
            pi = mk["stationary_estimate"]
            section.append("Estimated stationary distribution: " + ", ".join(f"{p:.3f}" for p in pi) + ".")
        if "tv_distance" in mk:
            section.append(f"Total-variation distance to true stationary: {mk['tv_distance']:.4f}.")
        if "max_ci_width" in mk:
            section.append(f"Max 95% CI width across states: {mk['max_ci_width']:.4f}.")
        if "spectral_analysis" in mk and isinstance(mk["spectral_analysis"], dict):
            sg = mk["spectral_analysis"].get("spectral_gap")
            if sg is not None:
                section.append(f"Spectral gap ≈ {sg:.4f}, indicating convergence rate 1-λ2.")
        if section:
            interpret = []
            if "tv_distance" in mk:
                tv = float(mk["tv_distance"])  # closeness to target
                interpret.append("TV distance near 0 indicates good convergence; consider more steps/trials if high.")
            if "max_ci_width" in mk:
                interpret.append("Narrow CI indicates precise long‑run state probabilities.")
            paragraphs.append("Markov chain results: " + " ".join(section) + " " + " ".join(interpret))
    if args.ab_test:
        ab = args.ab_test
        if ab.get("mode") == "binary":
            pa = ab.get("group_a", {}).get("rate")
            pb = ab.get("group_b", {}).get("rate")
            pval = ab.get("p_value")
            effect = ab.get("effect", {}).get("value")
            if pa is not None and pb is not None and pval is not None and effect is not None:
                significance = "statistically significant" if pval < 0.05 else "not statistically significant"
                direction = "increase" if effect > 0 else "decrease"
                paragraphs.append(f"A/B test (binary): variant A={pa:.3f}, B={pb:.3f} (a {abs(effect):.3f} {direction}). The difference is {significance} at α=0.05 (p={pval:.3g}).")
        elif ab.get("mode") == "continuous":
            diff = ab.get("effect", {}).get("value")
            pval = ab.get("p_value")
            if diff is not None and pval is not None:
                significance = "statistically significant" if pval < 0.05 else "not statistically significant"
                paragraphs.append(f"A/B test (continuous): mean difference {diff:+.3f}; this is {significance} at α=0.05 (p={pval:.3g}).")
    if args.power_curve:
        pc = args.power_curve
        if pc.get("mode") == "mde_vs_n":
            grid = pc.get("mde_rel_grid", [])
            nA = pc.get("n_per_arm_A", [])
            if grid and nA:
                alpha = pc.get("alpha", 0.05)
                power = pc.get("power", 0.8)
                paragraphs.append(f"Power planning: with α={alpha} and target power≈{power}, to detect {grid[0]*100:.1f}% lift need n≈{int(nA[0])}/arm; for {grid[-1]*100:.1f}% lift, about n≈{int(nA[-1])}.")
        if pc.get("mode") == "power_vs_n":
            ngrid = pc.get("n_grid", [])
            power = pc.get("power", [])
            if ngrid and power:
                paragraphs.append(f"Power growth: at n={int(ngrid[0])}/arm, power≈{power[0]:.2f}; at n={int(ngrid[-1])}, power≈{power[-1]:.2f}.")
    if args.forecast:
        fc = args.forecast
        aic = fc.get("aic"); order = fc.get("model_order"); seas = fc.get("seasonal_order")
        if aic is not None and order is not None:
            paragraphs.append(f"Forecast model: ARIMA{tuple(order)}{('x' + str(tuple(seas)) if seas is not None else '')} with AIC={float(aic):.2f}.")
    if args.notes:
        paragraphs.append(shorten(args.notes, width=280, placeholder="…"))
    if not paragraphs:
        paragraphs.append("No results provided to summarize.")
    return {"summary": " ".join(paragraphs)}


