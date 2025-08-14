from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np
from pydantic import BaseModel, Field
from fastapi import APIRouter

from .utils import fig_to_data_url, total_variation_distance, stationary_solve, spectral_gap_analysis, auto_tune_parameters

router = APIRouter()


class MarkovMCSInputs(BaseModel):
    transition: List[List[float]]
    start: int = 0
    steps: int = 1000
    trials: int = 5000
    burnin: int = 0
    seed: int = Field(default=12345, description="Random seed for reproducibility")
    metric: str = Field(default="stationary", description="'stationary', 'avg_reward', or 'trajectory'")
    rewards: Optional[List[float]] = None
    ci: float = 0.95
    track_trajectory: bool = Field(default=False, description="Track per-step trajectories for line charts")
    stability_check: bool = Field(default=False, description="Run multiple seeds for stability check")
    auto_tune: bool = Field(default=False, description="Auto-tune parameters if convergence fails")


@router.post("/tools/markov_mcs")
def run_markov_mcs(args: MarkovMCSInputs) -> Dict[str, Any]:
    T = np.array(args.transition, dtype=float)
    assert T.ndim == 2 and T.shape[0] == T.shape[1], "Transition must be square"
    assert np.allclose(T.sum(axis=1), 1.0, atol=1e-8), "Rows must sum to 1"

    n = T.shape[0]
    rng = np.random.default_rng(args.seed)
    cumT = np.cumsum(T, axis=1)

    def simulate_once():
        s = args.start
        for _ in range(args.burnin):
            s = int(np.searchsorted(cumT[s], rng.random()))
        visits = np.zeros(n, dtype=int)
        rew = 0.0
        trajectory = [] if args.track_trajectory else None
        for _ in range(args.steps):
            visits[s] += 1
            if args.rewards is not None:
                rew += args.rewards[s]
            if args.track_trajectory:
                trajectory.append(visits.copy())
            s = int(np.searchsorted(cumT[s], rng.random()))
        return visits, rew, trajectory

    visits_all = np.zeros((args.trials, n), dtype=int)
    reward_all = np.zeros(args.trials, dtype=float)
    trajectories_all = [] if args.track_trajectory else None
    for _ in range(args.trials):
        v, r, trajectory = simulate_once()
        visits_all[_] = v
        reward_all[_] = r
        if args.track_trajectory:
            trajectories_all.append(trajectory)

    out: Dict[str, Any] = {"steps": args.steps, "trials": args.trials, "seed": args.seed, "reproducible": True}
    if args.metric == "stationary":
        freq = visits_all.sum(axis=0) / visits_all.sum()
        se = np.sqrt(freq * (1 - freq) / (args.steps * args.trials))
        z = 1.959963984540054
        out["stationary_estimate"] = freq.tolist()
        out["stationary_ci_low"] = (freq - z * se).clip(0, 1).tolist()
        out["stationary_ci_high"] = (freq + z * se).clip(0, 1).tolist()
        ci_high = np.array(out["stationary_ci_high"])
        ci_low = np.array(out["stationary_ci_low"])
        ci_widths = ci_high - ci_low
        out["ci_widths"] = ci_widths.tolist()
        out["max_ci_width"] = float(np.max(ci_widths))
        pi_target = stationary_solve(T)
        tv_distance = total_variation_distance(freq, pi_target)
        out["tv_distance"] = float(tv_distance)
        out["pi_target"] = pi_target.tolist()
        out["spectral_analysis"] = spectral_gap_analysis(T)
        if args.stability_check:
            stability_results = []
            for seed in [args.seed, args.seed + 1000, args.seed + 2000]:
                rng = np.random.default_rng(seed)
                seed_visits = np.zeros((args.trials, n), dtype=int)
                for k in range(args.trials):
                    s = args.start
                    for _ in range(args.burnin):
                        s = int(np.searchsorted(cumT[s], rng.random()))
                    visits = np.zeros(n, dtype=int)
                    for __ in range(args.steps):
                        visits[s] += 1
                        s = int(np.searchsorted(cumT[s], rng.random()))
                    seed_visits[k] = visits
                seed_freq = seed_visits.sum(axis=0) / seed_visits.sum()
                seed_tv = total_variation_distance(seed_freq, pi_target)
                stability_results.append(seed_tv)
            median_tv = float(np.median(stability_results))
            tv_variance = float(np.var(stability_results))
            out["stability_check"] = {"median_tv": median_tv, "tv_variance": tv_variance, "individual_tvs": [float(tv) for tv in stability_results]}
        if args.auto_tune:
            out["auto_tune"] = auto_tune_parameters(T, args.steps, args.trials, tv_distance, out["max_ci_width"])  # type: ignore
    if args.track_trajectory:
        trajectories_array = np.array(trajectories_all)
        cumulative_counts = np.mean(trajectories_array, axis=0)
        step_numbers = np.arange(1, args.steps + 1)
        cumulative_shares = cumulative_counts / step_numbers[:, np.newaxis]
        cumulative_means = cumulative_counts / step_numbers[:, np.newaxis]
        out["trajectory_data"] = {
            "steps": list(range(1, args.steps + 1)),
            "cumulative_means": cumulative_means.tolist(),
            "cumulative_counts": cumulative_counts.tolist(),
            "cum_share": cumulative_shares.tolist(),
            "states": [f"State {i}" for i in range(n)],
        }
        if len(cumulative_shares) > 0:
            final_shares = cumulative_shares[-1]
            out["final_cum_share"] = final_shares.tolist()
            final_counts = cumulative_counts[-1]
            total_final = np.sum(final_counts)
            if total_final > 0:
                final_freq = final_counts / total_final
                se = np.sqrt(final_freq * (1 - final_freq) / args.trials)
                z = 1.959963984540054
                out["final_ci_low"] = (final_freq - z * se).clip(0, 1).tolist()
                out["final_ci_high"] = (final_freq + z * se).clip(0, 1).tolist()
                final_ci_high = np.array(out["final_ci_high"])
                final_ci_low = np.array(out["final_ci_low"])
                ci_widths = final_ci_high - final_ci_low
                out["final_ci_widths"] = ci_widths.tolist()
                out["final_max_ci_width"] = float(np.max(ci_widths))
                pi_target = stationary_solve(T)
                out["final_tv_distance"] = float(total_variation_distance(final_freq, pi_target))
                out["final_pi_target"] = pi_target.tolist()
    else:
        if args.rewards is None:
            args.rewards = list(range(n))
        avg = reward_all.mean() / args.steps
        se = reward_all.std(ddof=1) / np.sqrt(args.trials) / args.steps
        z = 1.959963984540054
        out["avg_reward"] = float(avg)
        out["avg_reward_ci"] = [float(avg - z * se), float(avg + z * se)]
    return out


