from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter
import time
import numpy as np
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .utils import fig_to_data_url, fit_sarimax_safe

router = APIRouter()


class ARIMAArgs(BaseModel):
    ts: List[float]
    horizon: int = Field(..., gt=0)
    seasonal_period: Optional[int] = None
    alpha: float = Field(0.05, ge=0, le=0.2)
    backtest_k: Optional[int] = Field(default=None, gt=0)
    start_date: Optional[str] = None
    freq: Optional[str] = None


@router.post("/tools/forecast_arima")
def forecast_arima(args: ARIMAArgs) -> Dict[str, Any]:
    import pandas as pd
    t0 = time.perf_counter()
    y = pd.Series(args.ts)
    index_is_datetime = False
    try:
        if args.start_date and args.freq:
            dt_index = pd.date_range(start=args.start_date, periods=len(y), freq=args.freq)
            y.index = dt_index
            index_is_datetime = True
    except Exception:
        index_is_datetime = False
    # --- Seasonality guard + tiny auto-inference ---
    order = (1, 1, 1)
    seasonal_order = (0, 0, 0, 0)

    sp = args.seasonal_period

    # If seasonal_period not provided, try a tiny ACF-based heuristic
    if sp is None:
        try:
            candidates = [7, 12, 24, 52]
            best_lag = None
            best_score = 0.0
            yv = y.values
            yv = yv - float(np.mean(yv))
            denom = float(np.sqrt(np.sum(yv ** 2))) + 1e-12
            for lag in candidates:
                if len(yv) < 2 * lag:
                    continue
                # simple normalized auto-correlation at given lag
                num = float(np.dot(yv[lag:], yv[:-lag]))
                ac = num / (denom * denom)
                if abs(ac) > abs(best_score):
                    best_score = ac
                    best_lag = lag
            # require a modest threshold
            if best_lag is not None and abs(best_score) >= 0.2:
                sp = int(best_lag)
        except Exception:
            sp = None

    res, used_order, used_seasonal = fit_sarimax_safe(y, order=(1,1,1), sp=sp, alpha=args.alpha)
    pred = res.get_forecast(steps=args.horizon)
    y_hat = pred.predicted_mean.tolist()
    ci_df = pred.conf_int(alpha=args.alpha)
    ci_low = ci_df.iloc[:, 0].tolist()
    ci_high = ci_df.iloc[:, 1].tolist()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    if index_is_datetime:
        try:
            offset = pd.tseries.frequencies.to_offset(args.freq) if args.freq else None
        except Exception:
            offset = None
        if offset is not None:
            f_start = y.index[-1] + offset
            f_idx = pd.date_range(start=f_start, periods=args.horizon, freq=args.freq)
        else:
            f_idx = np.arange(len(y), len(y) + args.horizon)
        ax.plot(y.index, y, label="Historical")
        ax.plot(f_idx, y_hat, label="Forecast")
        ax.fill_between(f_idx, ci_low, ci_high, color="orange", alpha=0.2, label=f"{int((1-args.alpha)*100)}% CI")
        try:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate(rotation=30, ha='right')
        except Exception:
            pass
    else:
        ax.plot(y.index, y, label="Historical")
        f_idx = np.arange(len(y), len(y) + args.horizon)
        ax.plot(f_idx, y_hat, label="Forecast")
        ax.fill_between(f_idx, ci_low, ci_high, color="orange", alpha=0.2, label=f"{int((1-args.alpha)*100)}% CI")
    ax.set_title("ARIMA forecast"); ax.set_xlabel("Time" if index_is_datetime else "t"); ax.set_ylabel("Value"); ax.legend(loc="best")
    img = fig_to_data_url(fig)
    # CI sanity check warning
    try:
        band = float(np.max(ci_high) - np.min(ci_low))
        span = float(np.max(y) - np.min(y) + 1e-9)
        ci_warn = bool(band > 10.0 * span)
    except Exception:
        ci_warn = False

    warn = False
    try:
        if hasattr(res, "mle_retvals"):
            warn = not res.mle_retvals.get("converged", True)
    except Exception:
        warn = False

    out: Dict[str, Any] = {
        "tool_version": "forecast_arima/0.1.2",
        "model_order": list(used_order),
        "seasonal_order": list(used_seasonal),
        "aic": float(getattr(res, "aic", float("nan"))),
        "forecast": y_hat,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "horizon": int(args.horizon),
        "artifact_url": img,
        "runtime_ms": (time.perf_counter() - t0) * 1000,
        "warnings": {"ci_unreasonably_wide": ci_warn, "converged": (not warn), "note": ("Fallback used or seasonality dropped" if (warn or tuple(used_seasonal)==(0,0,0,0)) else None)},
    }
    try:
        if args.backtest_k is not None and args.backtest_k > 0 and args.backtest_k < len(y):
            k = int(args.backtest_k)
            y_train = y.iloc[:-k]
            y_test = y.iloc[-k:]
            bt_res, _, _ = fit_sarimax_safe(y_train, order=order, sp=sp, alpha=args.alpha)
            bt_pred = bt_res.get_forecast(steps=k)
            y_hat_k = bt_pred.predicted_mean.values
            naive_last = float(y_train.iloc[-1])
            naive_fc = np.full(k, naive_last)
            def smape(y_true, y_pred):
                y_true = np.asarray(y_true, dtype=float)
                y_pred = np.asarray(y_pred, dtype=float)
                denom = np.abs(y_true) + np.abs(y_pred)
                mask = denom > 0
                if not np.any(mask):
                    return 0.0
                return float(100.0 * np.mean(np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))
            smape_model = smape(y_test.values, y_hat_k)
            smape_naive = smape(y_test.values, naive_fc)
            out["backtest"] = {"k": k, "smape": smape_model, "naive_smape": smape_naive, "improvement": float(smape_naive - smape_model)}
    except Exception:
        pass
    try:
        if index_is_datetime:
            hist_idx = [str(ts.to_pydatetime().date()) if hasattr(ts, 'to_pydatetime') else str(ts) for ts in y.index]
            fc_idx = [str(ts.to_pydatetime().date()) if hasattr(ts, 'to_pydatetime') else str(ts) for ts in f_idx]
        else:
            hist_idx = list(range(len(y)))
            fc_idx = list(range(len(y), len(y) + args.horizon))
        out["history_index"] = hist_idx
        out["forecast_index"] = fc_idx
        out["index_is_datetime"] = index_is_datetime
    except Exception:
        pass
    return out


