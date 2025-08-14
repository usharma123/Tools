from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter
import time
import numpy as np
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .utils import fig_to_data_url

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
    order = (1, 1, 1)
    seasonal_order = (0, 0, 0, 0)
    if args.seasonal_period:
        seasonal_order = (1, 1, 1, args.seasonal_period)
    res = SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
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
    ax.set_title("ARIMA forecast"); ax.set_xlabel("Index"); ax.set_ylabel("Value"); ax.legend(loc="best")
    img = fig_to_data_url(fig)
    out: Dict[str, Any] = {
        "tool_version": "forecast_arima/0.1.0",
        "model_order": order,
        "seasonal_order": seasonal_order,
        "aic": float(res.aic),
        "forecast": y_hat,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "horizon": int(args.horizon),
        "artifact_url": img,
        "runtime_ms": (time.perf_counter() - t0) * 1000,
    }
    try:
        if args.backtest_k is not None and args.backtest_k > 0 and args.backtest_k < len(y):
            k = int(args.backtest_k)
            y_train = y.iloc[:-k]
            y_test = y.iloc[-k:]
            bt_res = SARIMAX(y_train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
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


