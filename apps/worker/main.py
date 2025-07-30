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

# Markov Chain + Monte Carlo Simulation
class MarkovMCSInputs(BaseModel):
    transition: List[List[float]]
    start: int = 0
    steps: int = 1000
    trials: int = 5000
    burnin: int = 0
    seed: Optional[int] = None
    metric: str = Field(default="stationary", description="'stationary' or 'avg_reward'")
    rewards: Optional[List[float]] = None
    ci: float = 0.95

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
        for _ in range(args.steps):
            visits[s] += 1
            if args.rewards is not None:
                rew += args.rewards[s]
            s = int(np.searchsorted(cumT[s], rng.random()))
        return visits, rew

    visits_all = np.zeros((args.trials, n), dtype=int)
    reward_all = np.zeros(args.trials, dtype=float)
    for k in range(args.trials):
        v, r = simulate_once()
        visits_all[k] = v
        reward_all[k] = r

    out: Dict[str, Any] = {"steps": args.steps, "trials": args.trials, "seed": args.seed}

    if args.metric == "stationary":
        freq = visits_all.sum(axis=0) / visits_all.sum()
        se = np.sqrt(freq * (1 - freq) / (args.steps * args.trials))
        z = 1.959963984540054  # ~95%
        out["stationary_estimate"] = freq.tolist()
        out["stationary_ci_low"] = (freq - z * se).clip(0, 1).tolist()
        out["stationary_ci_high"] = (freq + z * se).clip(0, 1).tolist()
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
    series: Dict[str, List[float]]
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None

@app.post("/tools/plot_line")
def plot_line(args: PlotArgs):
    # Generate Chart.js HTML
    title = args.title or "Line Chart"
    xlabel = args.xlabel or "Index"
    ylabel = args.ylabel or "Value"
    
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
    
    # Create HTML with Chart.js
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
            type: 'bar',
            data: {{
                labels: {list(args.series.keys())},
                datasets: [{{
                    label: 'Probability',
                    data: {list(args.series.values())},
                    backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'],
                    borderColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'],
                    borderWidth: 1
                }}]
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
                        max: 1.0
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
