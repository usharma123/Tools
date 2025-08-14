from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter
import time
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .utils import fig_to_data_url
import os, uuid, json


router = APIRouter()


class BacktestArgs(BaseModel):
    ts: List[float]
    horizon: int = Field(..., gt=0)
    step: Optional[int] = None
    folds: Optional[int] = None
    min_train: Optional[int] = None
    seasonal_period: Optional[int] = None
    alpha: float = Field(0.05, ge=0, le=0.2)
    order: Optional[List[int]] = None
    # Optional speed knob: when true, reuse params across folds and cap optimizer iterations
    fast: Optional[bool] = Field(default=True)


def _smape(a, f, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=float)
    f = np.asarray(f, dtype=float)
    return float(100.0 * np.mean(2.0 * np.abs(f - a) / (np.abs(a) + np.abs(f) + eps)))


def _mae(a, f) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(f))))


def _rmse(a, f) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(f)) ** 2)))


def fit_sarimax_safe(y_train, order=(1, 1, 1), sp=None, alpha: float = 0.05):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import numpy as np, pandas as pd  # noqa: F401

    n = len(y_train)
    use_seasonal = bool(sp) and (n >= 3 * int(sp))
    seasonal_order = (
        (0, 1, 1, int(sp)) if (use_seasonal and n < 5 * int(sp)) else ((1, 1, 1, int(sp)) if use_seasonal else (0, 0, 0, 0))
    )

    def _fit(order_, seas_, method=None, start_params=None):
        m = SARIMAX(
            y_train,
            order=order_,
            seasonal_order=seas_,
            simple_differencing=True,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return m.fit(disp=False, method=method, start_params=start_params, maxiter=300)

    try:
        res = _fit(order, seasonal_order)
    except Exception:
        res = None

    if (res is None) or (not getattr(res, "mle_retvals", {}).get("converged", True)):
        try:
            res = _fit(order, seasonal_order, method="powell", start_params=(res.params if res else None))
        except Exception:
            res = None

    used_order, used_seasonal = order, seasonal_order

    if (res is None) or (not getattr(res, "mle_retvals", {}).get("converged", True)):
        used_order, used_seasonal = (0, 1, 1), (0, 0, 0, 0)
        res = _fit(used_order, used_seasonal)

    return res, used_order, used_seasonal


@router.post("/tools/forecast_backtest")
def forecast_backtest(args: BacktestArgs) -> Dict[str, Any]:
    t0 = time.perf_counter()
    y = pd.Series(args.ts)
    n = len(y)
    if n < args.horizon + 4:
        raise ValueError("Series too short for requested horizon.")

    sp = int(args.seasonal_period) if args.seasonal_period else 0
    min_train = int(args.min_train) if args.min_train is not None else (max(8, 3 * sp) if sp else 8)
    # Clamp min_train for short series so backtest can still run at least one fold
    if min_train + args.horizon > n:
        min_train = max(4, n - args.horizon)

    step = int(args.step) if args.step is not None else int(args.horizon)
    if args.folds:
        total_span = n - min_train - args.horizon
        step = max(1, total_span // max(1, (int(args.folds) - 1)))
    origins = list(range(min_train, n - args.horizon + 1, step))
    if not origins:
        origins = [n - args.horizon]

    order = tuple(args.order) if args.order else (1, 1, 1)

    by_fold: List[Dict[str, Any]] = []
    all_a_model: List[float] = []
    all_f_model: List[float] = []
    all_lo: List[float] = []
    all_hi: List[float] = []
    all_a_naive: List[float] = []
    all_f_naive: List[float] = []
    beats = 0

    prev_params = None
    prev_key = None
    # Optional downgrade: if overall series is too short for seasonality, drop it entirely
    if sp and (len(y) < int(2.5 * sp) + int(args.horizon)):
        sp = 0

    for o in origins:
        train = y.iloc[:o]
        test = y.iloc[o : o + args.horizon].tolist()

        res, used_order, used_seasonal = fit_sarimax_safe(train, order=order, sp=sp, alpha=args.alpha)
        pred = res.get_forecast(steps=args.horizon)
        f_model = pred.predicted_mean.tolist()
        ci = pred.conf_int(alpha=args.alpha).to_numpy()
        lo = ci[:, 0].tolist()
        hi = ci[:, 1].tolist()
        converged = bool(getattr(res, "mle_retvals", {}).get("converged", True))
        drift = [float(train.iloc[-1])] * args.horizon
        if used_seasonal != (0, 0, 0, 0) and sp:
            seasonal_naive = [float(y.iloc[o - int(sp) + i]) for i in range(args.horizon)]
        else:
            seasonal_naive = None

        sm_drift = _smape(test, drift)
        if seasonal_naive is not None:
            sm_seasonal = _smape(test, seasonal_naive)
            f_naive = seasonal_naive if sm_seasonal <= sm_drift else drift
            naive_name = "seasonal_naive" if sm_seasonal <= sm_drift else "drift"
        else:
            f_naive = drift
            naive_name = "drift"

        sm_model = _smape(test, f_model)
        sm_naive = _smape(test, f_naive)
        mae_model = _mae(test, f_model)
        mae_naive = _mae(test, f_naive)
        rmse_model = _rmse(test, f_model)
        rmse_naive = _rmse(test, f_naive)
        cover = float(np.mean([(lo[i] <= test[i] <= hi[i]) for i in range(args.horizon)]))

        by_fold.append({
            "origin_index": o,
            "use_seasonal": used_seasonal != (0, 0, 0, 0),
            "order": list(used_order),
            "seasonal_order": list(used_seasonal),
            "converged": converged,
            "naive": naive_name,
            "smape_model_pct": sm_model,
            "smape_naive_pct": sm_naive,
            "mae_model": mae_model,
            "mae_naive": mae_naive,
            "rmse_model": rmse_model,
            "rmse_naive": rmse_naive,
            "coverage": cover,
        })
        if sm_model + 1e-9 < sm_naive:
            beats += 1

        all_a_model.extend(test)
        all_f_model.extend(f_model)
        all_a_naive.extend(test)
        all_f_naive.extend(f_naive)
        all_lo.extend(lo)
        all_hi.extend(hi)

    overall = {
        "smape_model_pct": _smape(all_a_model, all_f_model),
        "smape_naive_pct": _smape(all_a_naive, all_f_naive),
        "mae_model": _mae(all_a_model, all_f_model),
        "mae_naive": _mae(all_a_naive, all_f_naive),
        "rmse_model": _rmse(all_a_model, all_f_model),
        "rmse_naive": _rmse(all_a_naive, all_f_naive),
        "coverage": float(
            np.mean([(all_lo[i] <= all_a_model[i] <= all_hi[i]) for i in range(len(all_a_model))])
        ),
        "mean_ci_width": float(np.mean(np.asarray(all_hi) - np.asarray(all_lo))),
        "beats_naive_ratio": beats / max(1, len(origins)),
        "beats_naive_70pct": (beats / max(1, len(origins))) >= 0.70,
    }

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    x = np.arange(len(by_fold))
    model_smape = [f["smape_model_pct"] for f in by_fold]
    naive_smape = [f["smape_naive_pct"] for f in by_fold]
    ax.plot(x, model_smape, label="Model sMAPE")
    ax.plot(x, naive_smape, label="Naïve sMAPE")
    ax.set_xlabel("Fold")
    ax.set_ylabel("sMAPE (%)")
    ax.set_title("Rolling backtest: sMAPE by fold")
    ax.legend(loc="best")
    image_base64 = fig_to_data_url(fig)

    # Build HTML artifact using Chart.js
    os.makedirs('artifacts', exist_ok=True)
    filename = f"backtest_{uuid.uuid4().hex[:8]}.html"
    filepath = os.path.join('artifacts', filename)
    datasets = [
        {
            'label': 'Model sMAPE',
            'data': model_smape,
            'borderColor': '#36A2EB',
            'backgroundColor': '#36A2EB20',
            'fill': False,
            'tension': 0.1,
        },
        {
            'label': 'Naïve sMAPE',
            'data': naive_smape,
            'borderColor': '#FF6384',
            'backgroundColor': '#FF638420',
            'fill': False,
            'tension': 0.1,
        },
    ]
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Rolling backtest: sMAPE by fold</title>
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background:#f8f9fa; }}
    .chart-container {{ background:white; border-radius:8px; padding:20px; box-shadow:0 2px 10px rgba(0,0,0,0.1); max-width:800px; margin:0 auto; }}
    h1 {{ color:#333; text-align:center; margin-bottom:20px; }}
  </style>
  </head>
<body>
  <div class=\"chart-container\"> 
    <h1>Rolling backtest: sMAPE by fold</h1>
    <div style=\"position: relative; height: 400px; width: 100%;\"> <canvas id=\"chart\"></canvas> </div>
  </div>
  <script>
    const ctx = document.getElementById('chart').getContext('2d');
    new Chart(ctx, {{
      type: 'line',
      data: {{
        labels: {json.dumps(list(range(len(by_fold))))},
        datasets: {json.dumps(datasets)}
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ display: true }},
          title: {{ display: true, text: 'Rolling backtest: sMAPE by fold' }}
        }},
        scales: {{
          x: {{ title: {{ display: true, text: 'Fold' }} }},
          y: {{ title: {{ display: true, text: 'sMAPE (%)' }} }}
        }}
      }}
    }});
  </script>
</body>
</html>
"""
    with open(filepath, 'w') as f:
        f.write(html)

    return {
        "tool_version": "forecast_backtest/0.1.0",
        "order": list(order),
        "seasonal_period": sp or None,
        "horizon": int(args.horizon),
        "step": int(step),
        "origins": origins,
        "overall": overall,
        "by_fold": by_fold,
        # Align with other tools: artifact_url is a data:image/png;base64 URL
        "artifact_url": image_base64,
        # Still provide the HTML file for richer viewing if desired
        "html_artifact_url": f"/artifacts/{filename}",
        "runtime_ms": (time.perf_counter() - t0) * 1000.0,
    }


