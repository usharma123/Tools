# Problem Solver MVP

A sophisticated AI-powered analysis system that combines statistical analysis tools (Markov chain Monte Carlo simulations and A/B testing power analysis) with interactive HTML visualizations. The system uses a FastAPI backend (powered by Uvicorn) for computational tools and a Next.js frontend for user interaction.

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
│   │   │   │   ├── tools_ab_power.ts   # A/B testing tool schemas
│   │   │   │   └── plan.ts             # Analysis plan schemas & reference resolution
│   │   │   └── components/
│   │   └── package.json
│   └── worker/       # FastAPI backend
│       ├── main.py   # Tool implementations (Uvicorn-based)
│       └── artifacts/ # Generated HTML charts
```

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies for the worker
cd apps/worker
pip install fastapi uvicorn numpy matplotlib scipy

# Install Node.js dependencies for the web app
cd ../web
npm install
```

### 2. Start the Services

```bash
# Terminal 1: Start the FastAPI worker with Uvicorn
cd apps/worker
uvicorn main:app --reload

# Terminal 2: Start the Next.js frontend
cd ../web
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
- `seed`: Random seed for reproducibility

**Example**:
```json
{
  "transition": [[0.9, 0.1], [0.2, 0.8]],
  "steps": 1000,
  "trials": 10000,
  "metric": "stationary",
  "seed": 12345
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

### Power Curve Analysis (`power_curve`)

**Purpose**: Analyze statistical power for A/B testing with two-proportion tests. Supports two modes: calculating required sample sizes for different minimum detectable effects (MDE), or calculating power at different sample sizes.

**Parameters**:
- `mode`: "mde_vs_n" or "power_vs_n"
- `baseline`: Baseline conversion rate (0-1)
- `alpha`: Significance level (default: 0.05)
- `two_tailed`: Whether test is two-tailed (default: true)
- `ratio`: Allocation ratio between groups (default: 1.0)
- `power`: Target power for MDE mode (default: 0.8)
- `mde_rel_grid`: Array of relative MDE values for MDE mode
- `mde_rel`: Fixed relative MDE for power mode
- `n_grid`: Array of sample sizes for power mode

**Example (MDE mode)**:
```json
{
  "mode": "mde_vs_n",
  "baseline": 0.05,
  "mde_rel_grid": [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1],
  "power": 0.8,
  "alpha": 0.05
}
```

**Example (Power mode)**:
```json
{
  "mode": "power_vs_n",
  "baseline": 0.05,
  "mde_rel": 0.2,
  "n_grid": [2000, 4000, 6000, 8000, 10000, 15000],
  "alpha": 0.05
}
```

**Output**:
```json
{
  "mode": "mde_vs_n",
  "n_per_arm_A": [752702, 336101, 189937, 122123, 85198, 48363, 31233],
  "mde_rel_grid": [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]
}
```

### Interactive Plot (`plot_line` and `plot_bar`)

**Purpose**: Generate interactive HTML charts using Chart.js. `plot_bar` is optimized for stationary distributions, showing each state as a separate bar.

**Parameters**:
- `series`: Object with data series
- `x`: Optional custom x-axis values
- `title`: Chart title
- `xlabel`: X-axis label
- `ylabel`: Y-axis label

**Example (`plot_line`)**:
```json
{
  "series": {
    "Power": [0.45, 0.67, 0.82, 0.91, 0.96, 0.99]
  },
  "x": [2000, 4000, 6000, 8000, 10000, 15000],
  "title": "Power vs Sample Size",
  "xlabel": "Sample Size per Arm",
  "ylabel": "Power"
}
```

**Output**: HTML file with an interactive chart.

## 📊 Usage Examples

### Basic Markov Chain Analysis

**Query**: "Estimate the stationary distribution for T=[[0.9,0.1],[0.2,0.8]] with 1000 trials and 1000 steps"

**What happens**:
1. AI generates analysis plan.
2. Executes `markov_mcs` with the transition matrix.
3. Returns stationary distribution estimates.

### Markov Chain + Visualization

**Query**: "Estimate the stationary distribution for T=[[0.9,0.1],[0.2,0.8]] with 1000 trials and 1000 steps, then plot the stationary distribution"

**What happens**:
1. AI generates analysis plan.
2. Executes `markov_mcs` to get the stationary distribution.
3. **Chains** the result to `plot_bar` with the real data.
4. Generates an interactive bar chart showing the probabilities for each state.

### A/B Testing Power Analysis

**Query**: "Run a power analysis for an A/B test comparing conversion rates. Baseline conversion rate is 5%. We want to detect a 20% relative improvement (to 6%). Calculate required sample sizes for different minimum detectable effects: 2%, 3%, 4%, 5%, 6%, 8%, 10% relative lift. Then calculate power at different sample sizes: 2000, 4000, 6000, 8000, 10000, 15000 users per arm for detecting a 20% relative improvement. Create visualizations showing the relationships."

**What happens**:
1. AI generates a 4-step analysis plan.
2. Executes `power_curve` in MDE mode to calculate sample sizes vs MDE.
3. Executes `plot_line` to visualize the MDE→n relationship.
4. Executes `power_curve` in power mode to calculate power vs sample size.
5. Executes `plot_line` to visualize the power→n relationship.
6. Evaluates success criteria including monotonicity checks.

### Complex Financial Analysis

**Query**: "Plot a comprehensive time series analysis with multiple datasets: Revenue Growth [120,135,148,162,178,195,214,235,258,284], Costs [80,85,92,98,105,112,120,128,137,146], and Profit Margin [33.3,37.0,37.8,39.5,41.0,42.6,43.9,45.5,46.9,48.6]"

**What happens**:
1. AI generates analysis plan.
2. Executes `plot_line` with multiple data series.
3. Creates an interactive line chart with the three metrics.

## 🔧 Advanced Features

### Reference Resolution System

The system supports dynamic data references using `_from` fields in analysis plans. References are resolved to actual numbers before tool execution.

**Example**:
```json
{
  "tool": "plot_line",
  "args": {
    "y_from": "$curve_power.power",
    "x_from": "$curve_power.n_grid",
    "label": "Power vs Sample Size",
    "title": "Power vs Sample Size"
  }
}
```

**How it works**:
1. `$curve_power.power` references the `power` field from a previous `power_curve` tool result.
2. The executor resolves this to the actual array of power values.
3. The tool receives real numbers, not references.

### Machine-Checkable Success Criteria

The system supports automated validation of analysis results with domain-specific checks.

**Example**:
```json
{
  "success_criteria": {
    "mde_monotonicity": "n_per_arm_A strictly decreases as MDE increases",
    "power_monotonicity": "power strictly increases with n"
  }
}
```

**Validation**:
- **MDE Monotonicity**: Verifies that required sample size decreases as minimum detectable effect increases.
- **Power Monotonicity**: Verifies that statistical power increases as sample size increases.
- **Numerical Tolerance**: Uses tolerance-based comparison to handle floating-point precision.

### Tool Chaining with Explicit Step IDs

Complex analyses can reference specific tool outputs using explicit step IDs.

**Example**:
```json
{
  "steps": [
    {
      "id": "curve_mde",
      "tool": "power_curve",
      "args": { "mode": "mde_vs_n", ... }
    },
    {
      "id": "curve_power", 
      "tool": "power_curve",
      "args": { "mode": "power_vs_n", ... }
    },
    {
      "tool": "plot_line",
      "args": {
        "y_from": "$curve_mde.n_per_arm_A",
        "x_from": "$curve_mde.mde_rel_grid",
        "label": "Sample Size vs MDE"
      }
    }
  ]
}
```

## Chart Types

### Bar Charts (Default for Distributions)

- **Best for**: Stationary distributions, probability comparisons.
- **Features**: Clear state labels, probability scale (0-1), one bar per state.
- **Example**: Stationary distribution of Markov chain states.

### Line Charts

- **Best for**: Time series, multiple data series, power curves.
- **Features**: Interactive hover effects, multiple colors, custom x-axis values.
- **Example**: Power vs sample size relationships, financial performance over time.

## Development Workflow

### Making Changes

1. **Edit worker code** (`apps/worker/main.py`).
2. **Restart worker**: The `uvicorn --reload` command will automatically restart the server on changes.
3. **Test changes**: Use curl or the web interface.

### Adding New Tools

1. **Add tool function** in `main.py`.
2. **Add tool definition** in `apps/web/src/lib/tools.ts` or `tools_ab_power.ts`.
3. **Update parsing logic** in `apps/web/src/app/api/solve/route.ts`.

### Debugging

- **Check worker logs**: Look for tool execution messages in the Uvicorn terminal.
- **Check API logs**: Look for parsing and chaining messages in the Next.js terminal.
- **Test individual tools**: Use curl to test worker endpoints directly.

## Troubleshooting

### Common Issues

**Worker not starting**:
```bash
# Install missing dependencies
pip install fastapi uvicorn numpy matplotlib scipy
```

**Charts not showing**:
- Check if worker is running: `curl http://localhost:8000/health`
- Check if artifacts directory exists: `ls apps/worker/artifacts/`

**Tool chaining not working**:
- Check API logs for parsing errors.
- Verify both tools are being executed: `jq '.toolResults | length'`

**Blank charts**:
- Restart worker after code changes.
- Check if real data is being passed to the plot tool.

**Reference resolution errors**:
- Verify step IDs are unique when calling the same tool multiple times.
- Check that referenced fields exist in the source tool's output.

### Health Checks

```bash
# Check worker health
curl http://localhost:8000/health

# Test individual tools
curl -X POST http://localhost:8000/tools/markov_mcs 
  -H "Content-Type: application/json" 
  -d '{"transition": [[0.9,0.1],[0.2,0.8]], "steps": 100, "trials": 100, "seed": 12345}'

# Test power curve tool
curl -X POST http://localhost:8000/tools/power_curve 
  -H "Content-Type: application/json" 
  -d '{"mode": "power_vs_n", "baseline": 0.05, "mde_rel": 0.2, "n_grid": [2000,4000,6000], "alpha": 0.05}'

# Test plotting
curl -X POST http://localhost:8000/tools/plot_line 
  -H "Content-Type: application/json" 
  -d '{"series": {"Power": [0.45,0.67,0.82]}, "x": [2000,4000,6000], "title": "Test Chart"}'
```

## Advanced Usage

### Custom Analysis Plans

The AI can generate custom analysis plans for complex problems:

```json
{
  "objective": "Analyze A/B testing power relationships",
  "steps": [
    "Calculate sample sizes for different MDEs",
    "Visualize MDE→n relationship", 
    "Calculate power for different sample sizes",
    "Visualize power→n relationship"
  ],
  "success_criteria": {
    "mde_monotonicity": "n_per_arm_A strictly decreases as MDE increases",
    "power_monotonicity": "power strictly increases with n"
  }
}
```

### Complex Queries

**Epidemiological Model**:
"Simulate disease spread with states: Susceptible (0.8,0.2,0), Infected (0.1,0.6,0.3), Recovered (0.05,0.1,0.85). Run 4000 trials with 2500 steps, calculate the endemic equilibrium, and plot the disease prevalence over time."

**Queue Theory**:
"Model a service queue with transition matrix [[0.3,0.7,0,0],[0.2,0.4,0.4,0],[0,0.3,0.5,0.2],[0.1,0,0.3,0.6]] representing queue lengths 0-3. Run 2500 trials with 1800 steps, calculate the probability of each queue length in steady state, and plot the queue length distribution."

**Comprehensive A/B Testing**:
"Run a power analysis for an e-commerce A/B test. Baseline conversion rate is 3.2%. We want to detect improvements from 2% to 15% relative lift. Calculate required sample sizes for each MDE, then calculate power at sample sizes from 5000 to 50000 users per arm for detecting a 10% relative improvement. Create visualizations for both relationships and validate the monotonicity of the results."

## Key Features

- ✅ **Interactive HTML Charts**: Professional Chart.js visualizations with custom x-axis support.
- ✅ **Advanced Tool Chaining**: Automatic data flow between tools with reference resolution.
- ✅ **Statistical Analysis**: Markov chain Monte Carlo simulation and A/B testing power analysis.
- ✅ **Machine-Checkable Validation**: Automated success criteria evaluation with domain-specific checks.
- ✅ **Reference Resolution**: Dynamic data references using `_from` fields in analysis plans.
- ✅ **Real-time Processing**: Immediate results with progress feedback.
- ✅ **Extensible Architecture**: Easy to add new tools and capabilities.
- ✅ **Professional UI**: Clean, responsive web interface.

## Future Enhancements

- **More Chart Types**: Pie charts, histograms, 3D plots.
- **Additional Statistical Tools**: T-tests, chi-square tests, regression analysis.
- **Batch Processing**: Handle multiple analyses simultaneously.
- **Export Options**: PDF reports, data downloads.
- **Advanced Chaining**: Conditional tool execution based on results.
- **More Success Criteria**: Additional domain-specific validation rules.

---

**Built with**: Next.js, FastAPI, Uvicorn, Chart.js, TypeScript, Python, SciPy
 