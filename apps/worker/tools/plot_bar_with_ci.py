from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter
import json, os, uuid
import numpy as np
from .utils import fig_to_data_url

router = APIRouter()


class BarWithCIArgs(BaseModel):
    labels: List[str]
    values: List[float]
    ci_low: List[float]
    ci_high: List[float]
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    ylim: Optional[List[float]] = None


@router.post("/tools/plot_bar_with_ci")
def plot_bar_with_ci(args: BarWithCIArgs) -> Dict[str, Any]:
    assert len(args.labels) == len(args.values) == len(args.ci_low) == len(args.ci_high) and len(args.values) > 0, "Length mismatch"
    title = args.title or "Bar Chart with CI"
    xlabel = args.xlabel or "Groups"
    ylabel = args.ylabel or "Value"
    background_colors = [f"rgba({hash(label) % 256}, {(hash(label) >> 8) % 256}, {(hash(label) >> 16) % 256}, 0.6)" for label in args.labels]
    border_colors = [f"rgba({hash(label) % 256}, {(hash(label) >> 8) % 256}, {(hash(label) >> 16) % 256}, 1)" for label in args.labels]
    dataset = {"label": title, "data": args.values, "backgroundColor": background_colors, "borderColor": border_colors, "borderWidth": 1}
    datasets_js_str = json.dumps([dataset])

    error_bars = []
    for i, (value, lo, hi) in enumerate(zip(args.values, args.ci_low, args.ci_high)):
        error_bars.append({"x": i, "y": value, "yMin": lo, "yMax": hi})
    error_bars_js = json.dumps(error_bars)

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background:#f8f9fa; }}
    .chart-container {{ background:white; border-radius:8px; padding:20px; box-shadow:0 2px 10px rgba(0,0,0,0.1); max-width:600px; margin:0 auto; }}
    h1 {{ color:#333; text-align:center; margin-bottom:20px; }}
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
      data: {{ labels: {json.dumps(args.labels)}, datasets: {datasets_js_str} }},
      options: {{
        responsive: true, maintainAspectRatio: false, aspectRatio: 2,
        plugins: {{
          title: {{ display: true, text: '{title}', font: {{ size: 16, weight: 'bold' }} }},
          legend: {{ display: false }},
          tooltip: {{ callbacks: {{ label: function(context) {{ const eb = errorBars[context.dataIndex]; return [`Value: ${{context.parsed.y.toFixed(3)}}`, `95% CI: [${{eb.yMin.toFixed(3)}}, ${{eb.yMax.toFixed(3)}}]`]; }} }} }}
        }},
        scales: {{ x: {{ display: true, title: {{ display: true, text: '{xlabel}' }} }}, y: {{ display: true, title: {{ display: true, text: '{ylabel}' }} , beginAtZero: true }} }}
      }}
    }});
  </script>
</body>
</html>
"""

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        x = np.arange(len(args.labels))
        ax.bar(x, args.values, color="#9BD0F5", edgecolor="#4B9CD3")
        ax.errorbar(x, args.values, yerr=[np.array(args.values) - np.array(args.ci_low), np.array(args.ci_high) - np.array(args.values)], fmt='none', ecolor='black', elinewidth=1, capsize=3)
        ax.set_xticks(x, args.labels)
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if args.ylim and len(args.ylim) == 2: ax.set_ylim(args.ylim[0], args.ylim[1])
        img_b64_ci = fig_to_data_url(fig)
    except Exception:
        img_b64_ci = None

    os.makedirs('artifacts', exist_ok=True)
    filename = f"bar_ci_{uuid.uuid4().hex[:8]}.html"
    with open(os.path.join('artifacts', filename), 'w') as f:
        f.write(html_content)
    out = {"artifact_url": f"/artifacts/{filename}"}
    if img_b64_ci: out["image_base64"] = img_b64_ci
    return out


