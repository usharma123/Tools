from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json
import math
import time
from scipy import stats

app = FastAPI(title="Agent Tools")

# Create artifacts directory if it doesn't exist
import os
os.makedirs('artifacts', exist_ok=True)

# Mount static files for artifacts
app.mount("/artifacts", StaticFiles(directory="artifacts"), name="artifacts")

@app.get("/health")
def health():
    return {"ok": True}

def _fig_to_data_url(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

def total_variation_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate total variation distance between two probability distributions."""
    return 0.5 * np.sum(np.abs(p1 - p2))

def stationary_solve(T: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution using robust linear solve."""
    S = T.shape[0]
    A = T.T - np.eye(S)           # left eigenvector: pi^T T = pi^T  =>  (T^T - I) pi = 0
    A[-1, :] = 1.0                # replace one row with sum-to-one
    b = np.zeros(S); b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    pi = np.maximum(pi, 0)        # ensure non-negative
    return pi / pi.sum()           # normalize

def spectral_gap_analysis(T: np.ndarray) -> dict:
    """Analyze the spectral gap for convergence rate estimation."""
    eigenvalues, eigenvectors = np.linalg.eig(T)
    # Sort by magnitude (descending)
    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    
    # Find the second largest eigenvalue (subdominant)
    if len(eigenvalues) > 1:
        subdominant_magnitude = np.abs(eigenvalues[1])
        spectral_gap = 1 - subdominant_magnitude
    else:
        subdominant_magnitude = 0
        spectral_gap = 1
    
    return {
        "subdominant_magnitude": subdominant_magnitude,
        "spectral_gap": spectral_gap,
        "convergence_rate": subdominant_magnitude
    }

def auto_tune_parameters(T: np.ndarray, current_steps: int, current_trials: int, 
                        tv_distance: float, ci_width: float, target_tv: float = 0.02, 
                        target_ci: float = 0.02) -> dict:
    """Auto-tune parameters based on convergence analysis."""
    spectral_info = spectral_gap_analysis(T)
    convergence_rate = spectral_info["convergence_rate"]
    
    suggestions = {}
    
    # If TV distance is too high, suggest more steps based on spectral gap
    if tv_distance > target_tv and convergence_rate > 0:
        # TV shrinks like convergence_rate^steps
        # We want: convergence_rate^new_steps <= target_tv / tv_distance
        # So: new_steps >= log(target_tv / tv_distance) / log(convergence_rate)
        if convergence_rate < 1:
            suggested_steps = int(np.ceil(np.log(target_tv / tv_distance) / np.log(convergence_rate)))
            suggestions["steps"] = max(suggested_steps, current_steps * 2)
        else:
            suggestions["steps"] = current_steps * 2
    
    # If CI width is too high, suggest more trials
    if ci_width > target_ci:
        # CI width scales like 1/sqrt(trials * steps)
        # We want: new_trials * new_steps >= (trials * steps) * (ci_width / target_ci)^2
        scaling_factor = (ci_width / target_ci) ** 2
        suggested_trials = int(np.ceil(current_trials * scaling_factor))
        suggestions["trials"] = max(suggested_trials, current_trials * 2)
    
    return {
        "suggestions": suggestions,
        "spectral_analysis": spectral_info,
        "current_tv": tv_distance,
        "current_ci": ci_width
    }

# AB Test T-Test functionality
class ABTTArgs(BaseModel):
    # Binary mode
    successes_a: Optional[int] = None
    trials_a:   Optional[int] = None
    successes_b: Optional[int] = None
    trials_b:   Optional[int] = None
    # Continuous mode
    mean_a: Optional[float] = None
    sd_a:   Optional[float] = None
    n_a:    Optional[int]   = None
    mean_b: Optional[float] = None
    sd_b:   Optional[float] = None
    n_b:    Optional[int]   = None

    alpha: float = 0.05
    two_tailed: bool = True
    equal_var: bool = False        # only for continuous mode
    assume_independent: bool = True

    @model_validator(mode="after")
    def exactly_one_mode(self):
        binary = all(v is not None for v in
            [self.successes_a, self.trials_a, self.successes_b, self.trials_b])
        cont = all(v is not None for v in
            [self.mean_a, self.sd_a, self.n_a, self.mean_b, self.sd_b, self.n_b])
        assert binary ^ cont, "Provide either binary (successes/trials) OR continuous (mean/sd/n), not both."
        return self

@app.post("/tools/ab_test_ttest")
def ab_test_ttest(args: ABTTArgs) -> Dict[str, Any]:
    t0 = time.perf_counter()
    out: Dict[str, Any] = {"tool_version": "ab_test_ttest/0.2.0", "alpha": args.alpha, "two_tailed": args.two_tailed}

    if all(v is not None for v in [args.successes_a, args.trials_a, args.successes_b, args.trials_b]):
        # ----- Binary proportions -----
        sa, na = int(args.successes_a), int(args.trials_a)
        sb, nb = int(args.successes_b), int(args.trials_b)
        assert 0 <= sa <= na and 0 <= sb <= nb and na > 0 and nb > 0, "Invalid successes/trials."

        pa, pb = sa/na, sb/nb
        diff = pb - pa

        tails = 2 if args.two_tailed else 1
        # Test stat (pooled SE under H0)
        p_pool = (sa + sb) / (na + nb)
        se_H0 = math.sqrt(p_pool*(1 - p_pool)*(1/na + 1/nb))
        z = diff / se_H0 if se_H0 > 0 else float("inf")
        p_value = (1 - stats.norm.cdf(abs(z))) * tails

        # CI for true difference (unpooled SE)
        zcrit = stats.norm.ppf(1 - args.alpha/tails)
        se_diff = math.sqrt(pa*(1-pa)/na + pb*(1-pb)/nb)
        ci = [diff - zcrit*se_diff, diff + zcrit*se_diff]

        # Per-arm CIs (use Wald for simplicity; you can switch to Wilson)
        se_a = math.sqrt(pa*(1-pa)/na); se_b = math.sqrt(pb*(1-pb)/nb)
        ci_a = [max(0.0, pa - zcrit*se_a), min(1.0, pa + zcrit*se_a)]
        ci_b = [max(0.0, pb - zcrit*se_b), min(1.0, pb + zcrit*se_b)]

        out.update({
            "mode": "binary",
            "group_a": {"successes": sa, "trials": na, "rate": pa, "ci": ci_a},
            "group_b": {"successes": sb, "trials": nb, "rate": pb, "ci": ci_b},
            "effect": {"name": "absolute_diff", "value": diff, "ci": ci},
            "relative_lift": (diff/pa) if pa > 0 else None,
            "test_stat": {"z": z},
            "p_value": p_value,
            "assumptions": {
                "independent_samples": args.assume_independent,
                "large_sample_normal": True
            }
        })
    else:
        # ----- Continuous (Welch t by default) -----
        ma, sda, na = float(args.mean_a), float(args.sd_a), int(args.n_a)
        mb, sdb, nb = float(args.mean_b), float(args.sd_b), int(args.n_b)
        assert na > 1 and nb > 1 and sda >= 0 and sdb >= 0, "Invalid mean/sd/n."

        diff = mb - ma
        tails = 2 if args.two_tailed else 1

        if args.equal_var:
            sp2 = ((na-1)*sda*sda + (nb-1)*sdb*sdb) / (na+nb-2)
            se = math.sqrt(sp2*(1/na + 1/nb))
            df = na + nb - 2
        else:
            se2 = (sda*sda)/na + (sdb*sdb)/nb
            se = math.sqrt(se2)
            df = (se2*se2) / (((sda*sda)/(na*na*(na-1))) + ((sdb*sdb)/(nb*nb*(nb-1))))
        t = diff / se if se > 0 else float("inf")
        p_value = (1 - stats.t.cdf(abs(t), df)) * tails
        tcrit = stats.t.ppf(1 - args.alpha/tails, df)
        ci = [diff - tcrit*se, diff + tcrit*se]

        # Per-arm 95% CI on the mean
        ci_a = [ma - tcrit*(sda/math.sqrt(na)), ma + tcrit*(sda/math.sqrt(na))]
        ci_b = [mb - tcrit*(sdb/math.sqrt(nb)), mb + tcrit*(sdb/math.sqrt(nb))]

        out.update({
            "mode": "continuous",
            "group_a": {"mean": ma, "sd": sda, "n": na, "ci": ci_a},
            "group_b": {"mean": mb, "sd": sdb, "n": nb, "ci": ci_b},
            "effect": {"name": "mean_diff", "value": diff, "ci": ci},
            "test_stat": {"t": t, "df": df},
            "p_value": p_value,
            "assumptions": {
                "independent_samples": args.assume_independent,
                "welch": not args.equal_var,
                "normal_or_clt": True
            }
        })

    out["runtime_ms"] = (time.perf_counter() - t0) * 1000.0
    return out

# Markov Chain + Monte Carlo Simulation
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

@app.post("/tools/markov_mcs")
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
        
        for step in range(args.steps):
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
    
    for k in range(args.trials):
        v, r, trajectory = simulate_once()
        visits_all[k] = v
        reward_all[k] = r
        if args.track_trajectory:
            trajectories_all.append(trajectory)

    out: Dict[str, Any] = {"steps": args.steps, "trials": args.trials, "seed": args.seed, "reproducible": True}

    if args.metric == "stationary":
        freq = visits_all.sum(axis=0) / visits_all.sum()
        se = np.sqrt(freq * (1 - freq) / (args.steps * args.trials))
        z = 1.959963984540054  # ~95%
        out["stationary_estimate"] = freq.tolist()
        out["stationary_ci_low"] = (freq - z * se).clip(0, 1).tolist()
        out["stationary_ci_high"] = (freq + z * se).clip(0, 1).tolist()
        
        # Calculate CI width for each state
        ci_high = np.array(out["stationary_ci_high"])
        ci_low = np.array(out["stationary_ci_low"])
        ci_widths = ci_high - ci_low
        out["ci_widths"] = ci_widths.tolist()
        out["max_ci_width"] = float(np.max(ci_widths))
        
        # Calculate total variation distance from true stationary distribution as convergence metric
        try:
            pi_target = stationary_solve(T)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Failed to compute stationary distribution: matrix may be singular or reducible. Error: {e}")
        
        tv_distance = total_variation_distance(freq, pi_target)
        out["tv_distance"] = float(tv_distance)
        out["pi_target"] = pi_target.tolist()  # Store for reference
        
        # Spectral analysis for auto-tuning
        spectral_info = spectral_gap_analysis(T)
        out["spectral_analysis"] = spectral_info
        
        # Stability check: run multiple seeds if requested
        if args.stability_check:
            stability_results = []
            seeds = [args.seed, args.seed + 1000, args.seed + 2000]
            for seed in seeds:
                np.random.seed(seed)
                rng = np.random.default_rng(seed)
                seed_visits = np.zeros((args.trials, n), dtype=int)
                
                for k in range(args.trials):
                    s = args.start
                    for _ in range(args.burnin):
                        s = int(np.searchsorted(cumT[s], rng.random()))
                    visits = np.zeros(n, dtype=int)
                    
                    for step in range(args.steps):
                        visits[s] += 1
                        s = int(np.searchsorted(cumT[s], rng.random()))
                    seed_visits[k] = visits
                
                seed_freq = seed_visits.sum(axis=0) / seed_visits.sum()
                seed_tv = total_variation_distance(seed_freq, pi_target)
                stability_results.append(seed_tv)
            
            median_tv = np.median(stability_results)
            tv_variance = np.var(stability_results)
            out["stability_check"] = {
                "median_tv": float(median_tv),
                "tv_variance": float(tv_variance),
                "individual_tvs": [float(tv) for tv in stability_results]
            }
        
        # Auto-tuning if requested
        if args.auto_tune:
            auto_tune_info = auto_tune_parameters(T, args.steps, args.trials, tv_distance, out["max_ci_width"])
            out["auto_tune"] = auto_tune_info
    
    # Compute trajectory data if requested (regardless of metric)
    if args.track_trajectory:
        # Calculate cumulative counts across trials for each state
        trajectories_array = np.array(trajectories_all)  # Shape: (trials, steps, states)
        cumulative_counts = np.mean(trajectories_array, axis=0)  # Shape: (steps, states)
        
        # Calculate cumulative shares (fractions) - this is the key improvement
        step_numbers = np.arange(1, args.steps + 1)
        cumulative_shares = cumulative_counts / step_numbers[:, np.newaxis]  # Shape: (steps, states)
        
        # Also keep the original cumulative means for backward compatibility
        cumulative_means = cumulative_counts / step_numbers[:, np.newaxis]  # Shape: (steps, states)
        
        # Convert to lists for JSON serialization
        out["trajectory_data"] = {
            "steps": list(range(1, args.steps + 1)),
            "cumulative_means": cumulative_means.tolist(),
            "cumulative_counts": cumulative_counts.tolist(),
            "cum_share": cumulative_shares.tolist(),  # New: cumulative shares (fractions)
            "states": [f"State {i}" for i in range(n)]
        }
        
        # Calculate convergence metrics for trajectory
        if len(cumulative_shares) > 0:
            # Use final cumulative share as stationary estimate
            final_shares = cumulative_shares[-1]
            out["final_cum_share"] = final_shares.tolist()
            
            # Calculate CI for final shares
            final_counts = cumulative_counts[-1]
            total_final = np.sum(final_counts)
            if total_final > 0:
                final_freq = final_counts / total_final
                se = np.sqrt(final_freq * (1 - final_freq) / args.trials)
                z = 1.959963984540054
                out["final_ci_low"] = (final_freq - z * se).clip(0, 1).tolist()
                out["final_ci_high"] = (final_freq + z * se).clip(0, 1).tolist()
                
                # Calculate CI width for final shares
                final_ci_high = np.array(out["final_ci_high"])
                final_ci_low = np.array(out["final_ci_low"])
                ci_widths = final_ci_high - final_ci_low
                out["final_ci_widths"] = ci_widths.tolist()
                out["final_max_ci_width"] = float(np.max(ci_widths))
                
                # Calculate total variation distance from true stationary distribution
                try:
                    pi_target = stationary_solve(T)
                except np.linalg.LinAlgError as e:
                    raise ValueError(f"Failed to compute stationary distribution: matrix may be singular or reducible. Error: {e}")
                
                tv_distance = total_variation_distance(final_freq, pi_target)
                out["final_tv_distance"] = float(tv_distance)
                out["final_pi_target"] = pi_target.tolist()  # Store for reference
    else:
        if args.rewards is None:
            # Default rewards: reward = state index
            args.rewards = list(range(n))
        avg = reward_all.mean() / args.steps
        se = reward_all.std(ddof=1) / np.sqrt(args.trials) / args.steps
        z = 1.959963984540054
        out["avg_reward"] = float(avg)
        out["avg_reward_ci"] = [float(avg - z * se), float(avg + z * se)]
    return out

# Plot
class PlotArgs(BaseModel):
    # Series data (resolved by API from variable references)
    series: Optional[Dict[str, List[float]]] = None
    x: Optional[List[float]] = None  # Custom x-axis values
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    ref_lines_y: Optional[List[float]] = None  # Reference lines at stationary probabilities

# Bar chart for stationary estimates
class PlotBarArgs(BaseModel):
    series: Optional[Dict[str, List[float]]] = None
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    ref_lines_y: Optional[List[float]] = None  # Reference lines at stationary probabilities

@app.post("/tools/plot_line")
def plot_line(args: PlotArgs):
    # Generate Chart.js HTML
    title = args.title or "Line Chart"
    xlabel = args.xlabel or "Index"
    ylabel = args.ylabel or "Value"
    
    # Check if series data is available
    if not args.series:
        return {"error": "No series data provided. Variable references should be resolved by the API."}
    
    # Prepare data for Chart.js
    datasets = []
    colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
    
    for i, (label, ys) in enumerate(args.series.items()):
        color = colors[i % len(colors)]
        datasets.append({
            'label': label,
            'data': ys,
            'borderColor': color,
            'backgroundColor': color + '20',
            'fill': False,
            'tension': 0.1
        })
    
    # Convert datasets to proper JavaScript format
    import json
    datasets_js = json.dumps(datasets)
    
    # Prepare reference lines if provided
    ref_lines_js = ""
    if args.ref_lines_y:
        ref_lines_data = []
        for i, y_value in enumerate(args.ref_lines_y):
            ref_lines_data.append({
                'type': 'line',
                'mode': 'horizontal',
                'scaleID': 'y',
                'value': y_value,
                'borderColor': 'rgba(0, 0, 0, 0.3)',
                'borderWidth': 2,
                'borderDash': [5, 5],
                'label': {
                    'content': f'Stationary {i+1}: {y_value:.3f}',
                    'enabled': True,
                    'position': 'end'
                }
            })
        ref_lines_js = f", annotation: {{ annotations: {json.dumps(ref_lines_data)} }}"

    # Create HTML with Chart.js
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background-color: #f8f9fa;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 800px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="chart-container">
        <h1>{title}</h1>
        <div style="position: relative; height: 400px; width: 100%;">
            <canvas id="chart"></canvas>
        </div>
    </div>
    
    <script>
        const ctx = document.getElementById('chart').getContext('2d');
        
        // Determine if this is a line chart (multiple data points) or bar chart (single data points)
        const isLineChart = Object.values({list(args.series.values())}).some(arr => arr.length > 1);
        
        new Chart(ctx, {{
            type: isLineChart ? 'line' : 'bar',
            data: {{
                labels: {json.dumps(args.x) if args.x else f"Array.from({{length: Math.max(...Object.values({list(args.series.values())}).map(arr => arr.length))}}, (_, i) => i + 1)"},
                datasets: {datasets_js}
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                aspectRatio: 2,
                plugins: {{
                    title: {{
                        display: true,
                        text: '{title}',
                        font: {{
                            size: 16,
                            weight: 'bold'
                        }}
                    }},
                    legend: {{
                        display: isLineChart
                    }}{ref_lines_js}
                }},
                scales: {{
                    x: {{
                        display: true,
                        title: {{
                            display: true,
                            text: '{xlabel}'
                        }}
                    }},
                    y: {{
                        display: true,
                        title: {{
                            display: true,
                            text: '{ylabel}'
                        }},
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Save HTML to a file and return the URL
    import os
    import uuid
    
    # Create artifacts directory if it doesn't exist
    os.makedirs('artifacts', exist_ok=True)
    
    # Generate unique filename
    filename = f"plot_{uuid.uuid4().hex[:8]}.html"
    filepath = os.path.join('artifacts', filename)
    
    with open(filepath, 'w') as f:
        f.write(html_content)
    
    # Return the URL to the HTML file
    return {"artifact_url": f"/artifacts/{filename}"}

@app.post("/tools/plot_bar")
def plot_bar(args: PlotBarArgs):
    # Generate Chart.js HTML for bar chart
    if not args.series:
        return {"error": "No series data provided"}
    
    # Prepare a single dataset for Chart.js with multiple bars
    labels = list(args.series.keys())
    data_values = [data[0] if isinstance(data, list) and len(data) > 0 else 0 for data in args.series.values()]
    
    # Generate colors for each bar
    background_colors = [f"rgba({hash(label) % 256}, {(hash(label) >> 8) % 256}, {(hash(label) >> 16) % 256}, 0.6)" for label in labels]
    border_colors = [f"rgba({hash(label) % 256}, {(hash(label) >> 8) % 256}, {(hash(label) >> 16) % 256}, 1)" for label in labels]
    
    dataset = {
        "label": args.title or "Distribution",
        "data": data_values,
        "backgroundColor": background_colors,
        "borderColor": border_colors,
        "borderWidth": 1
    }
    
    # Convert the single dataset to a list of datasets for Chart.js
    datasets_js_str = json.dumps([dataset])
    
    # Prepare reference lines if provided
    ref_lines_js = ""
    if args.ref_lines_y:
        ref_lines_data = []
        for i, y_value in enumerate(args.ref_lines_y):
            ref_lines_data.append({
                'type': 'line',
                'mode': 'horizontal',
                'scaleID': 'y',
                'value': y_value,
                'borderColor': 'rgba(0, 0, 0, 0.3)',
                'borderWidth': 2,
                'borderDash': [5, 5],
                'label': {
                    'content': f'Stationary {i+1}: {y_value:.3f}',
                    'enabled': True,
                    'position': 'end'
                }
            })
        ref_lines_js = f", annotation: {{ annotations: {json.dumps(ref_lines_data)} }}"
    
    # Create HTML with Chart.js
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{args.title or 'Bar Chart'}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background-color: #f8f9fa;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 600px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="chart-container">
        <h1>{args.title or 'Bar Chart'}</h1>
        <div style="position: relative; height: 300px; width: 100%;">
            <canvas id="chart"></canvas>
        </div>
    </div>
    
    <script>
        const ctx = document.getElementById('chart').getContext('2d');
        
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(list(args.series.keys()))},
                datasets: {datasets_js_str}
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                aspectRatio: 2,
                plugins: {{
                    title: {{
                        display: true,
                        text: '{args.title or "Bar Chart"}',
                        font: {{
                            size: 16,
                            weight: 'bold'
                        }}
                    }},
                    legend: {{
                        display: false
                    }}{ref_lines_js}
                }},
                scales: {{
                    x: {{
                        display: true,
                        title: {{
                            display: true,
                            text: '{args.xlabel or "States"}'
                        }}
                    }},
                    y: {{
                        display: true,
                        title: {{
                            display: true,
                            text: '{args.ylabel or "Probability"}'
                        }},
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Save HTML to a file and return the URL
    import os
    import uuid
    
    # Create artifacts directory if it doesn't exist
    os.makedirs('artifacts', exist_ok=True)
    
    # Generate unique filename
    filename = f"bar_{uuid.uuid4().hex[:8]}.html"
    filepath = os.path.join('artifacts', filename)
    
    with open(filepath, 'w') as f:
        f.write(html_content)
    
    # Return the URL to the HTML file
    return {"artifact_url": f"/artifacts/{filename}"}

# Bar chart with CI whiskers
class BarWithCIArgs(BaseModel):
    labels: List[str]
    values: List[float]
    ci_low: List[float]
    ci_high: List[float]
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    ylim: Optional[List[float]] = None

@app.post("/tools/plot_bar_with_ci")
def plot_bar_with_ci(args: BarWithCIArgs):
    assert len(args.labels)==len(args.values)==len(args.ci_low)==len(args.ci_high) and len(args.values)>0, "Length mismatch"
    
    # Generate Chart.js HTML for bar chart with CI
    title = args.title or "Bar Chart with CI"
    xlabel = args.xlabel or "Groups"
    ylabel = args.ylabel or "Value"
    
    # Prepare data for Chart.js
    background_colors = [f"rgba({hash(label) % 256}, {(hash(label) >> 8) % 256}, {(hash(label) >> 16) % 256}, 0.6)" for label in args.labels]
    border_colors = [f"rgba({hash(label) % 256}, {(hash(label) >> 8) % 256}, {(hash(label) >> 16) % 256}, 1)" for label in args.labels]
    
    # Create dataset with error bars
    dataset = {
        "label": title,
        "data": args.values,
        "backgroundColor": background_colors,
        "borderColor": border_colors,
        "borderWidth": 1
    }
    
    # Convert the dataset to a list for Chart.js
    datasets_js_str = json.dumps([dataset])
    
    # Prepare error bar data
    error_bars = []
    for i, (value, ci_low, ci_high) in enumerate(zip(args.values, args.ci_low, args.ci_high)):
        error_bars.append({
            "x": i,
            "y": value,
            "yMin": ci_low,
            "yMax": ci_high
        })
    
    error_bars_js = json.dumps(error_bars)
    
    # Create HTML with Chart.js
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background-color: #f8f9fa;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 600px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="chart-container">
        <h1>{title}</h1>
        <div style="position: relative; height: 400px; width: 100%;">
            <canvas id="chart"></canvas>
        </div>
    </div>
    
    <script>
        const ctx = document.getElementById('chart').getContext('2d');
        const errorBars = {error_bars_js};
        
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(args.labels)},
                datasets: {datasets_js_str}
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                aspectRatio: 2,
                plugins: {{
                    title: {{
                        display: true,
                        text: '{title}',
                        font: {{
                            size: 16,
                            weight: 'bold'
                        }}
                    }},
                    legend: {{
                        display: false
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                const dataIndex = context.dataIndex;
                                const errorBar = errorBars[dataIndex];
                                return [
                                    `Value: ${{context.parsed.y.toFixed(3)}}`,
                                    `95% CI: [${{errorBar.yMin.toFixed(3)}}, ${{errorBar.yMax.toFixed(3)}}]`
                                ];
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        display: true,
                        title: {{
                            display: true,
                            text: '{xlabel}'
                        }}
                    }},
                    y: {{
                        display: true,
                        title: {{
                            display: true,
                            text: '{ylabel}'
                        }},
                        beginAtZero: true
                    }}
                }}
            }}
        }});
        
        // Add custom error bars
        const chart = Chart.getChart(ctx);
        chart.options.plugins.customCanvasBackgroundColor = {{
            id: 'customCanvasBackgroundColor',
            beforeDraw: (chart) => {{
                const {{ctx}} = chart;
                ctx.save();
                ctx.globalCompositeOperation = 'destination-over';
                ctx.fillStyle = 'white';
                ctx.fillRect(0, 0, chart.width, chart.height);
                ctx.restore();
            }}
        }};
        
        // Draw error bars
        chart.options.plugins.errorBars = {{
            id: 'errorBars',
            afterDraw: (chart) => {{
                const {{ctx, chartArea: {{left, top, right, bottom}}, scales}} = chart;
                const xScale = scales.x;
                const yScale = scales.y;
                
                ctx.save();
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.8)';
                ctx.lineWidth = 1;
                
                errorBars.forEach((errorBar, index) => {{
                    const x = xScale.getPixelForValue(index);
                    const y = yScale.getPixelForValue(errorBar.y);
                    const yMin = yScale.getPixelForValue(errorBar.yMin);
                    const yMax = yScale.getPixelForValue(errorBar.yMax);
                    
                    // Vertical line
                    ctx.beginPath();
                    ctx.moveTo(x, yMin);
                    ctx.lineTo(x, yMax);
                    ctx.stroke();
                    
                    // Horizontal caps
                    const capWidth = 4;
                    ctx.beginPath();
                    ctx.moveTo(x - capWidth, yMin);
                    ctx.lineTo(x + capWidth, yMin);
                    ctx.stroke();
                    
                    ctx.beginPath();
                    ctx.moveTo(x - capWidth, yMax);
                    ctx.lineTo(x + capWidth, yMax);
                    ctx.stroke();
                }});
                
                ctx.restore();
            }}
        }};
        
        chart.update();
    </script>
</body>
</html>
"""
    
    # Save HTML to a file and return the URL
    import os
    import uuid
    
    # Create artifacts directory if it doesn't exist
    os.makedirs('artifacts', exist_ok=True)
    
    # Generate unique filename
    filename = f"bar_ci_{uuid.uuid4().hex[:8]}.html"
    filepath = os.path.join('artifacts', filename)
    
    with open(filepath, 'w') as f:
        f.write(html_content)
    
    # Return the URL to the HTML file
    return {"artifact_url": f"/artifacts/{filename}"}

# Power Curve functionality for A/B testing
def _n_per_arm_for_mde(
    p1: float, d_abs: float, alpha: float, power: float, two_tailed: bool, ratio: float
) -> tuple[float, float]:
    """Return (nA, nB) to detect absolute lift d_abs with given alpha/power."""
    p2 = p1 + d_abs
    if not (0 <= p2 <= 1):
        return float("nan"), float("nan")
    tails = 2 if two_tailed else 1
    z_alpha = stats.norm.ppf(1 - alpha / tails)
    z_beta  = stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (z_alpha * math.sqrt(2 * pbar * (1 - pbar)) +
           z_beta  * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    n_equal = num / (d_abs * d_abs)
    if abs(ratio - 1.0) < 1e-12:
        return n_equal, n_equal
    # simple unequal allocation approximation
    r = ratio
    nA = n_equal * (1 + r) / (4 * r)
    nB = r * nA
    return nA, nB

def _power_for_n(
    p1: float, d_abs: float, alpha: float, two_tailed: bool, nA: float, nB: float
) -> float:
    """Approximate power at absolute lift d_abs for given per-arm sizes."""
    p2 = p1 + d_abs
    if not (0 <= p2 <= 1):
        return float("nan")
    tails = 2 if two_tailed else 1
    z_alpha = stats.norm.ppf(1 - alpha / tails)
    # Under H1, diff ~ Normal(d_abs, var_unpooled)
    var = p1*(1-p1)/nA + p2*(1-p2)/nB
    if var <= 0: 
        return 0.0
    sd = math.sqrt(var)
    # symmetrical two-sided rejection: |Z| > z_alpha
    # Power ≈ P(Z > z_alpha - mu/sd) + P(Z < -z_alpha - mu/sd)
    mu_over_sd = d_abs / sd
    right = 1 - stats.norm.cdf(z_alpha - mu_over_sd)
    left  = stats.norm.cdf(-z_alpha - mu_over_sd)
    return max(0.0, min(1.0, right + left))

class PowerCurveArgs(BaseModel):
    mode: str = Field(..., description="'mde_vs_n' or 'power_vs_n'")
    baseline: float = Field(..., ge=0.0, le=1.0)
    alpha: float = Field(0.05, ge=0.0, le=1.0)
    two_tailed: bool = True
    ratio: float = Field(1.0, gt=0.0)  # nB/nA

    # For mode='mde_vs_n'
    mde_rel_grid: Optional[List[float]] = None  # e.g., [0.02,0.04,...] => relative lift
    power: Optional[float] = Field(0.8, ge=0.0, le=1.0)

    # For mode='power_vs_n'
    mde_rel: Optional[float] = None
    n_grid: Optional[List[int]] = None  # per-arm n for A (B = ratio*n)

@app.post("/tools/power_curve")
def power_curve(args: PowerCurveArgs) -> Dict[str, Any]:
    t0 = time.perf_counter()
    p1 = args.baseline
    tails = 2 if args.two_tailed else 1

    if args.mode == "mde_vs_n":
        assert args.mde_rel_grid is not None and len(args.mde_rel_grid) > 0, "Provide mde_rel_grid."
        xs = np.array(args.mde_rel_grid, dtype=float)
        nA_list, nB_list, nTotal = [], [], []
        for rel in xs:
            d_abs = p1 * rel
            nA, nB = _n_per_arm_for_mde(p1, d_abs, args.alpha, args.power, args.two_tailed, args.ratio)
            nA_list.append(nA); nB_list.append(nB); nTotal.append(nA + nB)
        
        # Generate Chart.js HTML
        title = f"Sample size vs MDE (baseline={p1:.3f}, power={args.power:.2f}, α={args.alpha})"
        xlabel = "MDE (% relative lift)"
        ylabel = "Sample size"
        
        # Prepare data for Chart.js
        datasets = []
        colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
        
        # Convert x values to percentages for display
        x_labels = [f"{x*100:.1f}%" for x in xs]
        
        datasets.append({
            'label': 'n per arm (A)',
            'data': nA_list,
            'borderColor': colors[0],
            'backgroundColor': colors[0] + '20',
            'fill': False,
            'tension': 0.1
        })
        
        if abs(args.ratio - 1.0) > 1e-12:
            datasets.append({
                'label': 'n per arm (B)',
                'data': nB_list,
                'borderColor': colors[1],
                'backgroundColor': colors[1] + '20',
                'fill': False,
                'tension': 0.1
            })
        
        datasets.append({
            'label': 'total N',
            'data': nTotal,
            'borderColor': colors[2],
            'backgroundColor': colors[2] + '20',
            'fill': False,
            'tension': 0.1
        })
        
        # Convert datasets to proper JavaScript format
        import json
        datasets_js = json.dumps(datasets)
        x_labels_js = json.dumps(x_labels)
        
        # Create HTML with Chart.js
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background-color: #f8f9fa;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 800px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="chart-container">
        <h1>{title}</h1>
        <div style="position: relative; height: 400px; width: 100%;">
            <canvas id="chart"></canvas>
        </div>
    </div>
    
    <script>
        const ctx = document.getElementById('chart').getContext('2d');
        
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {x_labels_js},
                datasets: {datasets_js}
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                aspectRatio: 2,
                plugins: {{
                    title: {{
                        display: true,
                        text: '{title}',
                        font: {{
                            size: 16,
                            weight: 'bold'
                        }}
                    }},
                    legend: {{
                        display: true
                    }}
                }},
                scales: {{
                    x: {{
                        display: true,
                        title: {{
                            display: true,
                            text: '{xlabel}'
                        }}
                    }},
                    y: {{
                        display: true,
                        title: {{
                            display: true,
                            text: '{ylabel}'
                        }},
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        
        # Save HTML to a file and return the URL
        import os
        import uuid
        
        # Create artifacts directory if it doesn't exist
        os.makedirs('artifacts', exist_ok=True)
        
        # Generate unique filename
        filename = f"power_curve_mde_{uuid.uuid4().hex[:8]}.html"
        filepath = os.path.join('artifacts', filename)
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        url = f"/artifacts/{filename}"
        
        out = {
            "tool_version": "power_curve/0.1.0",
            "mode": "mde_vs_n",
            "baseline": p1,
            "alpha": args.alpha,
            "power": args.power,
            "ratio": args.ratio,
            "mde_rel_grid": xs.tolist(),
            "n_per_arm_A": nA_list,
            "n_per_arm_B": nB_list,
            "n_total": nTotal,
            "artifact_url": url
        }

    elif args.mode == "power_vs_n":
        assert args.mde_rel is not None and args.n_grid is not None and len(args.n_grid) > 0, "Provide mde_rel and n_grid."
        d_abs = p1 * float(args.mde_rel)
        xs = np.array(args.n_grid, dtype=float)
        powers = []
        for nA in xs:
            nB = args.ratio * nA
            powers.append(_power_for_n(p1, d_abs, args.alpha, args.two_tailed, nA, nB))
        
        # Generate Chart.js HTML
        title = f"Power vs n (baseline={p1:.3f}, α={args.alpha}, ratio={args.ratio})"
        xlabel = "n per arm (A)"
        ylabel = "Power"
        
        # Prepare data for Chart.js
        datasets = []
        colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
        
        # Convert x values to strings for display
        x_labels = [str(int(x)) for x in xs]
        
        datasets.append({
            'label': f'power at MDE={args.mde_rel*100:.1f}%',
            'data': powers,
            'borderColor': colors[0],
            'backgroundColor': colors[0] + '20',
            'fill': False,
            'tension': 0.1
        })
        
        # Add reference line at 0.8
        datasets.append({
            'label': 'target power (0.8)',
            'data': [0.8] * len(xs),
            'borderColor': 'rgba(0, 0, 0, 0.3)',
            'backgroundColor': 'rgba(0, 0, 0, 0.1)',
            'fill': False,
            'borderDash': [5, 5],
            'tension': 0
        })
        
        # Convert datasets to proper JavaScript format
        import json
        datasets_js = json.dumps(datasets)
        x_labels_js = json.dumps(x_labels)
        
        # Create HTML with Chart.js
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background-color: #f8f9fa;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 800px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="chart-container">
        <h1>{title}</h1>
        <div style="position: relative; height: 400px; width: 100%;">
            <canvas id="chart"></canvas>
        </div>
    </div>
    
    <script>
        const ctx = document.getElementById('chart').getContext('2d');
        
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {x_labels_js},
                datasets: {datasets_js}
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                aspectRatio: 2,
                plugins: {{
                    title: {{
                        display: true,
                        text: '{title}',
                        font: {{
                            size: 16,
                            weight: 'bold'
                        }}
                    }},
                    legend: {{
                        display: true
                    }}
                }},
                scales: {{
                    x: {{
                        display: true,
                        title: {{
                            display: true,
                            text: '{xlabel}'
                        }}
                    }},
                    y: {{
                        display: true,
                        title: {{
                            display: true,
                            text: '{ylabel}'
                        }},
                        beginAtZero: true,
                        max: 1
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        
        # Save HTML to a file and return the URL
        import os
        import uuid
        
        # Create artifacts directory if it doesn't exist
        os.makedirs('artifacts', exist_ok=True)
        
        # Generate unique filename
        filename = f"power_curve_power_{uuid.uuid4().hex[:8]}.html"
        filepath = os.path.join('artifacts', filename)
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        url = f"/artifacts/{filename}"
        
        out = {
            "tool_version": "power_curve/0.1.0",
            "mode": "power_vs_n",
            "baseline": p1,
            "alpha": args.alpha,
            "ratio": args.ratio,
            "mde_rel": args.mde_rel,
            "n_grid": xs.tolist(),
            "power": powers,
            "artifact_url": url
        }
    else:
        raise ValueError("mode must be 'mde_vs_n' or 'power_vs_n'")

    out["runtime_ms"] = (time.perf_counter() - t0) * 1000.0
    return out
