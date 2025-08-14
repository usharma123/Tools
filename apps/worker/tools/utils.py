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


