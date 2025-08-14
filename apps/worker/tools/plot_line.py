from typing import Dict, List, Optional
from pydantic import BaseModel
from fastapi import APIRouter
import json, os, uuid
from .utils import fig_to_data_url
import matplotlib.pyplot as plt

router = APIRouter()


class PlotArgs(BaseModel):
    series: Optional[Dict[str, List[float]]] = None
    x: Optional[List[float]] = None
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    ref_lines_y: Optional[List[float]] = None


@router.post("/tools/plot_line")
def plot_line(args: PlotArgs):
    title = args.title or "Line Chart"
    xlabel = args.xlabel or "Index"
    ylabel = args.ylabel or "Value"
    if not args.series:
        return {"error": "No series data provided. Variable references should be resolved by the API."}

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
    datasets_js = json.dumps(datasets)
    ref_lines_js = ""
    if args.ref_lines_y:
        ref_lines_data = []
        for i, y in enumerate(args.ref_lines_y):
            ref_lines_data.append({
                'type': 'line', 'mode': 'horizontal', 'scaleID': 'y', 'value': y,
                'borderColor': 'rgba(0, 0, 0, 0.3)', 'borderWidth': 2, 'borderDash': [5, 5],
                'label': {'content': f'Stationary {i+1}: {y:.3f}', 'enabled': True, 'position': 'end'}
            })
        ref_lines_js = f", annotation: {{ annotations: {json.dumps(ref_lines_data)} }}"

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background:#f8f9fa; }}
    .chart-container {{ background:white; border-radius:8px; padding:20px; box-shadow:0 2px 10px rgba(0,0,0,0.1); max-width:800px; margin:0 auto; }}
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
          title: {{ display: true, text: '{title}', font: {{ size: 16, weight: 'bold' }} }},
          legend: {{ display: isLineChart }}{ref_lines_js}
        }},
        scales: {{
          x: {{ display: true, title: {{ display: true, text: '{xlabel}' }} }},
          y: {{ display: true, title: {{ display: true, text: '{ylabel}' }} , beginAtZero: true }}
        }}
      }}
    }});
  </script>
</body>
</html>
"""

    try:
        fig, ax = plt.subplots(figsize=(6, 3))
        first_series = next(iter(args.series.values()))
        x_vals = args.x if args.x is not None else list(range(1, len(first_series) + 1))
        for label, ys in args.series.items():
            ax.plot(x_vals, ys, label=str(label))
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if len(args.series) > 1: ax.legend(loc="best")
        if args.ref_lines_y:
            for y in args.ref_lines_y:
                ax.axhline(y, color="black", linestyle="--", alpha=0.4)
        img_b64 = fig_to_data_url(fig)
    except Exception:
        img_b64 = None

    os.makedirs('artifacts', exist_ok=True)
    filename = f"plot_{uuid.uuid4().hex[:8]}.html"
    filepath = os.path.join('artifacts', filename)
    with open(filepath, 'w') as f:
        f.write(html_content)
    out = {"artifact_url": f"/artifacts/{filename}"}
    if img_b64: out["image_base64"] = img_b64
    return out


