# Problem Solver MVP

A sophisticated AI-powered analysis system that combines Markov chain Monte Carlo simulations with interactive HTML visualizations. The system uses a FastAPI backend for computational tools and a Next.js frontend for user interaction.

## Architecture

```
Tools/
├── apps/
│   ├── web/          # Next.js frontend
│   │   ├── src/
│   │   │   ├── app/
│   │   │   │   ├── api/solve/route.ts  # API endpoint for tool orchestration
│   │   │   │   └── page.tsx            # React frontend
│   │   │   ├── lib/
│   │   │   │   ├── tools.ts            # Tool definitions
│   │   │   │   └── plan.ts             # Analysis plan schemas
│   │   │   └── components/
│   │   └── package.json
│   └── worker/       # FastAPI backend
│       ├── main.py   # Tool implementations
│       └── artifacts/ # Generated HTML charts
```

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies for the worker
cd apps/worker
pip install fastapi uvicorn numpy matplotlib

# Install Node.js dependencies for the web app
cd ../web
npm install
```

### 2. Start the Services

```bash
# Terminal 1: Start the FastAPI worker
cd apps/worker
python main.py

# Terminal 2: Start the Next.js frontend
cd apps/web
npm run dev
```

### 3. Access the Application

- **Frontend**: http://localhost:3000
- **Worker API**: http://localhost:8000
- **Health Check**: http://localhost:8000/health

## Available Tools

### Markov Chain Monte Carlo (`markov_mcs`)

**Purpose**: Simulate Markov chains and estimate stationary distributions.

**Parameters**:
- `transition`: 2D array of transition probabilities
- `steps`: Number of simulation steps (default: 1000)
- `trials`: Number of Monte Carlo trials (default: 5000)
- `metric`: "stationary" or "avg_reward" (default: "stationary")

**Example**:
```json
{
  "transition": [[0.9, 0.1], [0.2, 0.8]],
  "steps": 1000,
  "trials": 10000,
  "metric": "stationary"
}
```

**Output**:
```json
{
  "stationary_estimate": [0.667, 0.333],
  "stationary_ci_low": [0.666, 0.331],
  "stationary_ci_high": [0.668, 0.335]
}
```

### Interactive Plot (`plot_line`)

**Purpose**: Generate interactive HTML charts using Chart.js.

**Parameters**:
- `series`: Object with data series
- `title`: Chart title
- `xlabel`: X-axis label
- `ylabel`: Y-axis label

**Example**:
```json
{
  "series": {
    "State 0": [0.67],
    "State 1": [0.33]
  },
  "title": "Stationary Distribution",
  "xlabel": "States",
  "ylabel": "Probability"
}
```

**Output**: HTML file with interactive bar chart

## 📊 Usage Examples

### Basic Markov Chain Analysis

**Query**: "Estimate the stationary distribution for T=[[0.9,0.1],[0.2,0.8]] with 1000 trials and 1000 steps"

**What happens**:
1. AI generates analysis plan
2. Executes `markov_mcs` with transition matrix
3. Returns stationary distribution estimates

### Markov Chain + Visualization

**Query**: "Estimate the stationary distribution for T=[[0.9,0.1],[0.2,0.8]] with 1000 trials and 1000 steps, then plot the stationary distribution"

**What happens**:
1. AI generates analysis plan
2. Executes `markov_mcs` to get stationary distribution
3. **Chains** the result to `plot_line` with real data
4. Generates interactive bar chart showing probabilities

### Complex Financial Analysis

**Query**: "Plot a comprehensive time series analysis with multiple datasets: Revenue Growth [120,135,148,162,178,195,214,235,258,284], Costs [80,85,92,98,105,112,120,128,137,146], and Profit Margin [33.3,37.0,37.8,39.5,41.0,42.6,43.9,45.5,46.9,48.6]"

**What happens**:
1. AI generates analysis plan
2. Executes `plot_line` with multiple data series
3. Creates interactive line chart with three metrics

## 🔧 Tool Chaining

The system supports **automatic tool chaining** where the output of one tool becomes the input of another.

### How Chaining Works

1. **Parse all tool calls** from AI response
2. **Execute tools in sequence**
3. **Replace placeholder data** with real results
4. **Generate final artifacts**

### Example Chain

```javascript
// AI generates both calls
TOOL_CALL:markov_mcs:{"transition": [[0.9,0.1],[0.2,0.8]], "steps": 1000, "trials": 10000}
TOOL_CALL:plot_line:{"series": {"State 0": [0], "State 1": [0]}, "title": "Stationary Distribution"}

// System executes and chains
1. Run markov_mcs → get stationary_estimate [0.667, 0.333]
2. Replace plot_line placeholder with real data
3. Run plot_line → generate HTML chart
```

## Chart Types

### Bar Charts (Default for Distributions)

- **Best for**: Stationary distributions, probability comparisons
- **Features**: Clear state labels, probability scale (0-1)
- **Example**: Stationary distribution of Markov chain states

### Line Charts

- **Best for**: Time series, multiple data series
- **Features**: Interactive hover effects, multiple colors
- **Example**: Financial performance over time

## Development Workflow

### Making Changes

1. **Edit worker code** (`apps/worker/main.py`)
2. **Restart worker**: `pkill -f "python main.py" && cd apps/worker && python main.py`
3. **Test changes**: Use curl or the web interface

### Adding New Tools

1. **Add tool function** in `main.py`
2. **Add tool definition** in `apps/web/src/lib/tools.ts`
3. **Update parsing logic** in `apps/web/src/app/api/solve/route.ts`

### Debugging

- **Check worker logs**: Look for tool execution messages
- **Check API logs**: Look for parsing and chaining messages
- **Test individual tools**: Use curl to test worker endpoints directly

## Troubleshooting

### Common Issues

**Worker not starting**:
```bash
# Install missing dependencies
pip install fastapi uvicorn numpy matplotlib
```

**Charts not showing**:
- Check if worker is running: `curl http://localhost:8000/health`
- Check if artifacts directory exists: `ls apps/worker/artifacts/`

**Tool chaining not working**:
- Check API logs for parsing errors
- Verify both tools are being executed: `jq '.toolResults | length'`

**Blank charts**:
- Restart worker after code changes
- Check if real data is being passed to plot tool

### Health Checks

```bash
# Check worker health
curl http://localhost:8000/health

# Test individual tools
curl -X POST http://localhost:8000/tools/markov_mcs \
  -H "Content-Type: application/json" \
  -d '{"transition": [[0.9,0.1],[0.2,0.8]], "steps": 100, "trials": 100}'

# Test plotting
curl -X POST http://localhost:8000/tools/plot_line \
  -H "Content-Type: application/json" \
  -d '{"series": {"Test": [1,2,3]}, "title": "Test Chart"}'
```

## Advanced Usage

### Custom Analysis Plans

The AI can generate custom analysis plans for complex problems:

```json
{
  "objective": "Analyze market state transitions",
  "steps": [
    "Define transition matrix",
    "Run Monte Carlo simulation",
    "Calculate stationary distribution",
    "Visualize results"
  ],
  "report_outline": [
    "Introduction",
    "Methodology",
    "Results",
    "Conclusion"
  ]
}
```

### Complex Queries

**Epidemiological Model**:
"Simulate disease spread with states: Susceptible (0.8,0.2,0), Infected (0.1,0.6,0.3), Recovered (0.05,0.1,0.85). Run 4000 trials with 2500 steps, calculate the endemic equilibrium, and plot the disease prevalence over time."

**Queue Theory**:
"Model a service queue with transition matrix [[0.3,0.7,0,0],[0.2,0.4,0.4,0],[0,0.3,0.5,0.2],[0.1,0,0.3,0.6]] representing queue lengths 0-3. Run 2500 trials with 1800 steps, calculate the probability of each queue length in steady state, and plot the queue length distribution."

## Key Features

- ✅ **Interactive HTML Charts**: Professional Chart.js visualizations
- ✅ **Tool Chaining**: Automatic data flow between tools
- ✅ **Markov Chain Analysis**: Monte Carlo simulation with confidence intervals
- ✅ **Real-time Processing**: Immediate results with progress feedback
- ✅ **Extensible Architecture**: Easy to add new tools and capabilities
- ✅ **Professional UI**: Clean, responsive web interface

## Future Enhancements

- **More Chart Types**: Pie charts, histograms, 3D plots
- **Additional Tools**: Statistical tests, optimization algorithms
- **Batch Processing**: Handle multiple analyses simultaneously
- **Export Options**: PDF reports, data downloads
- **Advanced Chaining**: Conditional tool execution based on results

---

**Built with**: Next.js, FastAPI, Chart.js, TypeScript, Python 