from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from fastapi import APIRouter
from io import StringIO
import time
import matplotlib.dates as mdates
from .utils import fig_to_data_url

router = APIRouter()


class DIDArgs(BaseModel):
    csv: str
    date_col: str = "date"
    metric_col: str = "metric"
    entity_col: str = "entity"
    treated_entity: str
    pre_period: List[str]
    post_period: List[str]


@router.post("/tools/causal_impact")
def causal_impact(args: DIDArgs) -> Dict[str, Any]:
    import pandas as pd
    import statsmodels.formula.api as smf
    t0 = time.perf_counter()
    if "\n" in args.csv:
        df = pd.read_csv(StringIO(args.csv), parse_dates=[args.date_col])
    else:
        df = pd.read_csv(args.csv, parse_dates=[args.date_col])
    df["post"] = ((df[args.date_col] >= pd.Timestamp(args.post_period[0])) & (df[args.date_col] <= pd.Timestamp(args.post_period[1]))).astype(int)
    df["treated"] = (df[args.entity_col] == args.treated_entity).astype(int)
    df["did"] = (df["post"] * df["treated"]).astype(int)
    df[args.metric_col] = pd.to_numeric(df[args.metric_col], errors="coerce")
    df = df.dropna(subset=[args.metric_col])
    clusters = int(df[args.entity_col].nunique())
    warnings: List[str] = []
    if clusters < 3:
        warnings.append(f"Warning: cluster-robust SE disabled (n_unique_entities={clusters} < 3). Estimates may be fragile.")
    formula = f"{args.metric_col} ~ post + treated + did"
    if clusters >= 3:
        mod = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df[args.entity_col]})
        smry_label = "statsmodels_cluster"
    else:
        mod = smf.ols(formula, data=df).fit()
        smry_label = "statsmodels_nonrobust"
    ate = float(mod.params["did"])  # type: ignore
    se = float(mod.bse["did"])  # type: ignore
    ci = [float(ate - 1.96 * se), float(ate + 1.96 * se)]
    p = float(mod.pvalues["did"]) if "did" in mod.pvalues else float("nan")  # type: ignore
    grp = df.groupby([args.date_col, "treated"])[args.metric_col].mean().unstack()
    try:
        grp = grp.rename(columns={False: "control", True: "treated"})
    except Exception:
        pass
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    grp.plot(ax=ax)
    ax.axvline(pd.Timestamp(args.post_period[0]), ls="--", lw=1)
    ax.set_title("Actual KPI (treated vs control)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Metric")
    ax.legend(title="group")
    try:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate(rotation=30, ha='right')
    except Exception:
        pass
    img_url = fig_to_data_url(fig)
    import os, uuid
    os.makedirs('artifacts', exist_ok=True)
    filename = f"did_{uuid.uuid4().hex[:8]}.html"
    filepath = os.path.join('artifacts', filename)
    warn_html = "" if not warnings else ("<ul>" + "".join(f"<li>{w}</li>" for w in warnings) + "</ul>")
    warn_block = f'<div class="warn">{warn_html}</div>' if warnings else ''
    html = f"""
<!DOCTYPE html><html><head><meta charset='utf-8'/><title>Causal Impact</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:20px;background:#f8f9fa;}}.card{{background:white;border-radius:8px;padding:16px;box-shadow:0 2px 10px rgba(0,0,0,0.08);max-width:900px;margin:0 auto;}}details{{margin-top:12px;}}pre{{white-space: pre-wrap;}}.warn{{color:#a15c00;background:#fff3cd;border:1px solid #ffeeba;padding:8px 12px;border-radius:6px;margin:12px 0;}}</style>
</head><body><div class="card"><h2>Causal Impact</h2><img alt="Causal Impact" src="{img_url}" style="max-width:100%; height:auto;"/>{warn_block}<details><summary>Details</summary><pre>{smry_label}\n{mod.summary().as_text()}</pre></details></div></body></html>
"""
    with open(filepath, 'w') as f:
        f.write(html)
    return {"tool_version": "causal_impact/0.1.0", "ate": float(ate), "ci": ci, "p_value": p, "model_summary": f"{smry_label}:\n" + mod.summary().as_text()[:980], "artifact_url": f"/artifacts/{filename}", "image_base64": img_url, "warnings": warnings, "runtime_ms": (time.perf_counter()-t0)*1000}


