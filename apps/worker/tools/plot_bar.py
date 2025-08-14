from typing import Dict, List, Optional
from pydantic import BaseModel
from fastapi import APIRouter
import json, os, uuid
from .utils import fig_to_data_url
import matplotlib.pyplot as plt

router = APIRouter()


class PlotBarArgs(BaseModel):
    series: Optional[Dict[str, List[float]]] = None
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    ref_lines_y: Optional[List[float]] = None


@router.post("/tools/plot_bar")
def plot_bar(args: PlotBarArgs):
    if not args.series:
        return {"error": "No series data provided"}
    labels = list(args.series.keys())
    data_values = [data[0] if isinstance(data, list) and len(data) > 0 else 0 for data in args.series.values()]
    dataset = {
        "label": args.title or "Distribution",
        "data": data_values,
        "backgroundColor": [f"rgba({hash(label) % 256}, {(hash(label) >> 8) % 256}, {(hash(label) >> 16) % 256}, 0.6)" for label in labels],
        "borderColor": [f"rgba({hash(label) % 256}, {(hash(label) >> 8) % 256}, {(hash(label) >> 16) % 256}, 1)" for label in labels],
        "borderWidth": 1
    }
    datasets_js_str = json.dumps([dataset])
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
  <title>{args.title or 'Bar Chart'}</title>
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
    <h1>{args.title or 'Bar Chart'}</h1>
    <div style="position: relative; height: 300px; width: 100%;">
      <canvas id="chart"></canvas>
    </div>
  </div>
  <script>
    const ctx = document.getElementById('chart').getContext('2d');
    new Chart(ctx, {{
      type: 'bar',
      data: {{ labels: {json.dumps(labels)}, datasets: {datasets_js_str} }},
      options: {{
        responsive: true, maintainAspectRatio: false, aspectRatio: 2,
        plugins: {{ title: {{ display: true, text: '{args.title or "Bar Chart"}', font: {{ size: 16, weight: 'bold' }} }}, legend: {{ display: false }}{ref_lines_js} }},
        scales: {{ x: {{ display: true, title: {{ display: true, text: '{args.xlabel or "States"}' }} }}, y: {{ display: true, title: {{ display: true, text: '{args.ylabel or "Probability"}' }} , beginAtZero: true }} }}
      }}
    }});
  </script>
</body>
</html>
"""

    try:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(labels, data_values, color="#9BD0F5")
        ax.set_title(args.title or "Bar Chart")
        ax.set_xlabel(args.xlabel or "States")
        ax.set_ylabel(args.ylabel or "Probability")
        if args.ref_lines_y:
            for y in args.ref_lines_y:
                ax.axhline(y, color="black", linestyle="--", alpha=0.4)
        img_b64_bar = fig_to_data_url(fig)
    except Exception:
        img_b64_bar = None

    os.makedirs('artifacts', exist_ok=True)
    filename = f"bar_{uuid.uuid4().hex[:8]}.html"
    with open(os.path.join('artifacts', filename), 'w') as f:
        f.write(html_content)
    out = {"artifact_url": f"/artifacts/{filename}"}
    if img_b64_bar: out["image_base64"] = img_b64_bar
    return out


