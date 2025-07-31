from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json

app = FastAPI(title="Agent Tools")

# Create artifacts directory if it doesn't exist
import os
os.makedirs('artifacts', exist_ok=True)

# Mount static files for artifacts
app.mount("/artifacts", StaticFiles(directory="artifacts"), name="artifacts")

@app.get("/health")
def health():
    return {"ok": True}

def fig_to_data_url(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

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
                labels: isLineChart ? Array.from({{length: Math.max(...Object.values({list(args.series.values())}).map(arr => arr.length))}}, (_, i) => i + 1) : {list(args.series.keys())},
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
    
    # Prepare datasets for Chart.js
    datasets_js = []
    for label, data in args.series.items():
        # For bar chart, we expect single values per state
        if isinstance(data, list) and len(data) > 0:
            # Take the last value if it's a trajectory, otherwise use the single value
            value = data[-1] if len(data) > 1 else data[0]
            datasets_js.append({
                "label": label,
                "data": [value],
                "backgroundColor": f"rgba({hash(label) % 256}, {(hash(label) >> 8) % 256}, {(hash(label) >> 16) % 256}, 0.6)",
                "borderColor": f"rgba({hash(label) % 256}, {(hash(label) >> 8) % 256}, {(hash(label) >> 16) % 256}, 1)",
                "borderWidth": 1
            })
    
    # Convert datasets to proper JavaScript format
    datasets_js_str = json.dumps(datasets_js)
    
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
