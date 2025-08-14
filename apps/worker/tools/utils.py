from io import BytesIO
import base64
from typing import Any, Dict
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fig_to_data_url(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def total_variation_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p1 - p2)))


def stationary_solve(T: np.ndarray) -> np.ndarray:
    S = T.shape[0]
    A = T.T - np.eye(S)
    A[-1, :] = 1.0
    b = np.zeros(S)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    pi = np.maximum(pi, 0)
    return pi / pi.sum()


def spectral_gap_analysis(T: np.ndarray) -> Dict[str, float]:
    eigenvalues, _ = np.linalg.eig(T)
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    if len(eigenvalues) > 1:
        sub = float(np.abs(eigenvalues[1]))
        gap = 1 - sub
    else:
        sub = 0.0
        gap = 1.0
    return {"subdominant_magnitude": sub, "spectral_gap": gap, "convergence_rate": sub}


def auto_tune_parameters(T: np.ndarray, current_steps: int, current_trials: int, tv_distance: float, ci_width: float,
                         target_tv: float = 0.02, target_ci: float = 0.02) -> Dict[str, Any]:
    spectral_info = spectral_gap_analysis(T)
    convergence_rate = spectral_info["convergence_rate"]
    suggestions: Dict[str, int] = {}
    if tv_distance > target_tv and convergence_rate > 0:
        if convergence_rate < 1:
            suggested_steps = int(np.ceil(np.log(target_tv / tv_distance) / np.log(convergence_rate)))
            suggestions["steps"] = max(suggested_steps, current_steps * 2)
        else:
            suggestions["steps"] = current_steps * 2
    if ci_width > target_ci:
        scaling = (ci_width / target_ci) ** 2
        suggested_trials = int(np.ceil(current_trials * scaling))
        suggestions["trials"] = max(suggested_trials, current_trials * 2)
    return {
        "suggestions": suggestions,
        "spectral_analysis": spectral_info,
        "current_tv": tv_distance,
        "current_ci": ci_width,
    }


# Robust SARIMAX fitting helper used by forecasting tools
def fit_sarimax_safe(y_train, order=(1, 1, 1), sp=None, alpha: float = 0.05):
    """Fit SARIMAX robustly:
       - require enough data for seasonality, else drop seasonal
       - prefer simpler seasonal order on short samples
       - optimizer fallback when convergence fails
       - fallback to nonseasonal (0,1,1) if still unhappy
    """
    import numpy as _np  # noqa: F401
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    n = len(y_train)
    use_seasonal = bool(sp) and (n >= 3 * int(sp))
    if use_seasonal:
        seasonal_order = (0, 1, 1, int(sp)) if n < 5 * int(sp) else (1, 1, 1, int(sp))
    else:
        seasonal_order = (0, 0, 0, 0)

    def _fit(order_, seasonal_order_, method=None, start_params=None):
        model = SARIMAX(
            y_train,
            order=order_,
            seasonal_order=seasonal_order_,
            simple_differencing=True,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, method=method, start_params=start_params, maxiter=300)

    res = None
    try:
        res = _fit(order, seasonal_order)
    except Exception:
        res = None

    if (res is None) or (not getattr(res, "mle_retvals", {}).get("converged", True)):
        try:
            res = _fit(order, seasonal_order, method="powell", start_params=(res.params if res is not None else None))
        except Exception:
            res = None

    used_order = order
    used_seasonal_order = seasonal_order

    if (res is None) or (not getattr(res, "mle_retvals", {}).get("converged", True)):
        try:
            used_order = (0, 1, 1)
            used_seasonal_order = (0, 0, 0, 0)
            res = _fit(used_order, used_seasonal_order)
        except Exception:
            class Stub:
                params = None
                mle_retvals = {"converged": False}
                aic = float("nan")

                def get_forecast(self, steps):
                    import pandas as pd
                    import numpy as np
                    last = float(y_train.iloc[-1])

                    class P:
                        predicted_mean = pd.Series([last] * steps)

                        def conf_int(self, alpha=0.05):
                            lo = np.array([last] * steps) - 1.0
                            hi = np.array([last] * steps) + 1.0
                            import pandas as pd
                            return pd.DataFrame({"lower": lo, "upper": hi})

                    return P()

            res = Stub()

    # Sanity check that forecast works
    _ = res.get_forecast(steps=1)
    return res, used_order, used_seasonal_order

