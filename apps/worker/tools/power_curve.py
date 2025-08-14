from typing import Any, Dict, List, Optional
import json, time, os, uuid, math
import numpy as np
from pydantic import BaseModel, Field
from fastapi import APIRouter
from scipy import stats
from .utils import fig_to_data_url

router = APIRouter()


def _n_per_arm_for_mde(p1: float, d_abs: float, alpha: float, power: float, two_tailed: bool, ratio: float) -> tuple[float, float]:
    p2 = p1 + d_abs
    if not (0 <= p2 <= 1):
        return float("nan"), float("nan")
    tails = 2 if two_tailed else 1
    z_alpha = stats.norm.ppf(1 - alpha / tails)
    z_beta = stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (z_alpha * math.sqrt(2 * pbar * (1 - pbar)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    n_equal = num / (d_abs * d_abs)
    if abs(ratio - 1.0) < 1e-12:
        return n_equal, n_equal
    r = ratio
    nA = n_equal * (1 + r) / (4 * r)
    nB = r * nA
    return nA, nB


def _power_for_n(p1: float, d_abs: float, alpha: float, two_tailed: bool, nA: float, nB: float) -> float:
    p2 = p1 + d_abs
    if not (0 <= p2 <= 1):
        return float("nan")
    tails = 2 if two_tailed else 1
    z_alpha = stats.norm.ppf(1 - alpha / tails)
    var = p1 * (1 - p1) / nA + p2 * (1 - p2) / nB
    if var <= 0:
        return 0.0
    sd = math.sqrt(var)
    mu_over_sd = d_abs / sd
    right = 1 - stats.norm.cdf(z_alpha - mu_over_sd)
    left = stats.norm.cdf(-z_alpha - mu_over_sd)
    return max(0.0, min(1.0, right + left))


class PowerCurveArgs(BaseModel):
    mode: str = Field(..., description="'mde_vs_n' or 'power_vs_n'")
    baseline: float = Field(..., ge=0.0, le=1.0)
    alpha: float = Field(0.05, ge=0.0, le=1.0)
    two_tailed: bool = True
    ratio: float = Field(1.0, gt=0.0)  # nB/nA
    mde_rel_grid: Optional[List[float]] = None
    power: Optional[float] = Field(0.8, ge=0.0, le=1.0)
    mde_rel: Optional[float] = None
    n_grid: Optional[List[int]] = None


@router.post("/tools/power_curve")
def power_curve(args: PowerCurveArgs) -> Dict[str, Any]:
    t0 = time.perf_counter()
    p1 = args.baseline
    if args.mode == "mde_vs_n":
        assert args.mde_rel_grid is not None and len(args.mde_rel_grid) > 0, "Provide mde_rel_grid."
        xs = np.array(args.mde_rel_grid, dtype=float)
        nA_list, nB_list, nTotal = [], [], []
        for rel in xs:
            d_abs = p1 * rel
            nA, nB = _n_per_arm_for_mde(p1, d_abs, args.alpha, args.power or 0.8, args.two_tailed, args.ratio)
            nA_list.append(nA); nB_list.append(nB); nTotal.append(nA + nB)
        title = f"Sample size vs MDE (baseline={p1:.3f}, power={args.power:.2f}, α={args.alpha})"
        x_labels = [f"{x*100:.1f}%" for x in xs]
        datasets = [
            { 'label': 'n per arm (A)', 'data': nA_list, 'borderColor': '#FF6384', 'backgroundColor': '#FF638420', 'fill': False, 'tension': 0.1 },
            { 'label': 'total N', 'data': nTotal, 'borderColor': 'rgba(0,0,0,0.25)', 'backgroundColor': 'rgba(0,0,0,0.05)', 'fill': False, 'tension': 0.1 }
        ]
        if abs(args.ratio - 1.0) > 1e-12:
            datasets.insert(1, { 'label': 'n per arm (B)', 'data': nB_list, 'borderColor': '#36A2EB', 'backgroundColor': '#36A2EB20', 'fill': False, 'tension': 0.1 })
        datasets_js = json.dumps(datasets)
        x_labels_js = json.dumps(x_labels)
        html_content = f"""
<!DOCTYPE html>
<html><head><title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
</head>
<body>
<div style="max-width:800px;margin:0 auto;background:#fff;padding:20px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.1)">
<h1 style="text-align:center;color:#333">{title}</h1>
<canvas id="chart"></canvas></div>
<script>
const ctx=document.getElementById('chart').getContext('2d');
new Chart(ctx,{{type:'line',data:{{labels:{x_labels_js},datasets:{datasets_js}}},options:{{responsive:true,maintainAspectRatio:false,aspectRatio:2,plugins:{{legend:{{display:true}}}}}}}});
</script>
</body></html>
"""
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6.5, 3.5))
            ax.plot([x*100 for x in xs], nA_list, label='n per arm (A)')
            ax.plot([x*100 for x in xs], nTotal, label='total N', color='k', alpha=0.25)
            if abs(args.ratio - 1.0) > 1e-12:
                ax.plot([x*100 for x in xs], nB_list, label='n per arm (B)')
            ax.set_xlabel('MDE (% relative lift)'); ax.set_ylabel('Sample size'); ax.set_title(title); ax.legend()
            data_url = fig_to_data_url(fig)
        except Exception:
            data_url = None
        out = {"tool_version": "power_curve/0.1.0", "mode": "mde_vs_n", "baseline": p1, "alpha": args.alpha, "power": args.power, "ratio": args.ratio, "mde_rel_grid": xs.tolist(), "n_per_arm_A": nA_list, "n_per_arm_B": nB_list, "n_total": nTotal, "artifact_url": data_url}
    elif args.mode == "power_vs_n":
        assert args.mde_rel is not None and args.n_grid is not None and len(args.n_grid) > 0, "Provide mde_rel and n_grid."
        d_abs = p1 * float(args.mde_rel)
        xs = np.array(args.n_grid, dtype=float)
        powers = [_power_for_n(p1, d_abs, args.alpha, args.two_tailed, nA, args.ratio * nA) for nA in xs]
        title = f"Power vs n (baseline={p1:.3f}, α={args.alpha}, ratio={args.ratio})"
        x_labels = [str(int(x)) for x in xs]
        datasets = [
            { 'label': f'power at MDE={args.mde_rel*100:.1f}%', 'data': powers, 'borderColor': '#FF6384', 'backgroundColor': '#FF638420', 'fill': False, 'tension': 0.1 },
            { 'label': 'target power (0.8)', 'data': [0.8]*len(xs), 'borderColor': 'rgba(0,0,0,0.3)', 'backgroundColor': 'rgba(0,0,0,0.1)', 'fill': False, 'borderDash': [5,5], 'tension': 0 }
        ]
        datasets_js = json.dumps(datasets)
        x_labels_js = json.dumps(x_labels)
        html_content = f"""
<!DOCTYPE html>
<html><head><title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
</head>
<body>
<div style="max-width:800px;margin:0 auto;background:#fff;padding:20px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.1)">
<h1 style="text-align:center;color:#333">{title}</h1>
<canvas id="chart"></canvas></div>
<script>
const ctx=document.getElementById('chart').getContext('2d');
new Chart(ctx,{{type:'line',data:{{labels:{x_labels_js},datasets:{datasets_js}}},options:{{responsive:true,maintainAspectRatio:false,aspectRatio:2,scales:{{y:{{beginAtZero:true,max:1}}}}}}}});
</script>
</body></html>
"""
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6.5, 3.5))
            ax.plot(xs, powers, label=f'power at MDE={args.mde_rel*100:.1f}%')
            ax.axhline(0.8, color='k', linestyle='--', alpha=0.3, label='target power (0.8)')
            ax.set_xlabel('n per arm (A)'); ax.set_ylabel('Power'); ax.set_title(title); ax.set_ylim(0,1); ax.legend()
            data_url = fig_to_data_url(fig)
        except Exception:
            data_url = None
        out = {"tool_version": "power_curve/0.1.0", "mode": "power_vs_n", "baseline": p1, "alpha": args.alpha, "ratio": args.ratio, "mde_rel": args.mde_rel, "n_grid": xs.tolist(), "power": powers, "artifact_url": data_url}
    else:
        raise ValueError("mode must be 'mde_vs_n' or 'power_vs_n'")
    out["runtime_ms"] = (time.perf_counter() - t0) * 1000.0
    out["tool_version"] = out.get("tool_version", "power_curve/0.1.0")
    return out


