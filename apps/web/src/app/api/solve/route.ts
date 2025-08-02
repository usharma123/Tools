import { NextRequest } from "next/server";
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import { run_markov_mcs, plot_line, plot_bar } from "@/lib/tools";
import { ab_test_ttest, plot_bar_with_ci, power_curve, abTestParams, barWithCIParams, powerCurveParams } from "@/lib/tools_ab_power";
import { AnalysisPlan, SuccessCriteria, materializeArgs } from "@/lib/plan";

export const runtime = "nodejs";

const SYSTEM = `
You are Chief Analyst.
1) First output a VALID AnalysisPlan JSON with proper schema validation.
2) Execute steps strictly via available tools.
3) If inputs are missing, ask targeted questions.
4) Prefer simplest valid methods; include assumptions & limitations.
5) Reference any returned artifact_url in your write-up.
Return a concise Decision and Evidence.

IMPORTANT: Choose tools based on the analysis context:
- Use markov_mcs for Markov chain analysis, convergence studies, state transitions
- Use ab_test_ttest for A/B testing, conversion rate comparisons, treatment effects
- Use power_curve for sample size planning, power analysis, MDE calculations (NO seed parameter)
- Use plot_* tools for visualization of results

Available tools:
- markov_mcs: Run Monte Carlo on a Markov chain. Parameters: transition (array of arrays), steps (number), trials (number), metric (stationary/avg_reward/trajectory), track_trajectory (boolean)
- plot_line: Create a simple line chart. Parameters: series (object with arrays), title (string), xlabel (string), ylabel (string)
- plot_bar: Create a bar chart. Parameters: series (object with arrays), title (string), xlabel (string), ylabel (string)
- ab_test_ttest: Two-sample test (binary or continuous). Parameters: binary mode (successes_a, trials_a, successes_b, trials_b) OR continuous mode (mean_a, sd_a, n_a, mean_b, sd_b, n_b), alpha, two_tailed, equal_var (continuous only)
- plot_bar_with_ci: Bar chart with 95% CI whiskers. Parameters: labels, values, ci_low, ci_high, title, xlabel, ylabel, ylim
- power_curve: Plot power relationships for two-proportion A/B: either n vs MDE or power vs n. Parameters: mode (mde_vs_n/power_vs_n), baseline, alpha, two_tailed, ratio, power (for mde_vs_n), mde_rel_grid (for mde_vs_n), mde_rel (for power_vs_n), n_grid (for power_vs_n). IMPORTANT: power_curve is deterministic and does NOT accept seed parameter.

AnalysisPlan Schema Examples:

For A/B Testing:
{
  "objective": "Compare conversion rates between variants A and B",
  "assumptions": ["Binary outcomes", "Independent samples", "Large enough sample size"],
  "steps": [
    {
      "tool": "ab_test_ttest",
      "args": {
        "successes_a": 100,
        "trials_a": 1000,
        "successes_b": 120,
        "trials_b": 1000,
        "alpha": 0.05,
        "two_tailed": true
      }
    },
    {
      "tool": "plot_bar_with_ci",
      "args": {
        "labels": ["A", "B"],
        "values_from": ["$ab_test_ttest.group_a.rate", "$ab_test_ttest.group_b.rate"],
        "ci_low_from": ["$ab_test_ttest.group_a.ci.0", "$ab_test_ttest.group_b.ci.0"],
        "ci_high_from": ["$ab_test_ttest.group_a.ci.1", "$ab_test_ttest.group_b.ci.1"],
        "title": "Conversion by Variant (95% CI)",
        "ylabel": "Rate"
      }
    }
  ],
  "success_criteria": {"description": "AB test completed successfully"}
}

For Power Analysis:
{
  "objective": "Determine sample size needed for different MDE levels",
  "assumptions": ["Two-proportion test", "Equal allocation"],
  "steps": [
    {
      "tool": "power_curve",
      "args": {
        "mode": "mde_vs_n",
        "baseline": 0.05,
        "alpha": 0.05,
        "power": 0.8,
        "two_tailed": true,
        "ratio": 1.0,
        "mde_rel_grid": [0.02,0.03,0.04,0.05,0.06,0.08,0.10]
      }
    }
  ],
  "success_criteria": {"description": "Power curve generated successfully"}
}

For Comprehensive Power Analysis with Visualizations:
{
  "objective": "Analyze power relationships for A/B testing with multiple visualizations",
  "assumptions": ["Two-proportion test", "Equal allocation", "Fixed baseline rate"],
  "steps": [
    {
      "tool": "power_curve",
      "args": {
        "mode": "mde_vs_n",
        "baseline": 0.05,
        "alpha": 0.05,
        "power": 0.8,
        "two_tailed": true,
        "ratio": 1.0,
        "mde_rel_grid": [0.02,0.03,0.04,0.05,0.06,0.08,0.10]
      }
    },
    {
      "tool": "power_curve",
      "args": {
        "mode": "power_vs_n",
        "baseline": 0.05,
        "alpha": 0.05,
        "two_tailed": true,
        "ratio": 1.0,
        "mde_rel": 0.20,
        "n_grid": [2000,4000,6000,8000,10000,15000]
      }
    },
    {
      "tool": "plot_line",
      "args": {
        "series_from": "$power_curve.power",
        "labels": ["Power vs Sample Size"],
        "title": "Power vs Sample Size",
        "xlabel": "Sample Size per Arm",
        "ylabel": "Power"
      }
    }
  ],
  "success_criteria": {"description": "Power analysis completed with visualizations"}
}

For Markov Chain Analysis:
{
  "objective": "Analyze convergence of Markov chain",
  "assumptions": ["Ergodic chain", "Valid transition matrix"],
  "steps": [
    {
      "tool": "markov_mcs",
      "args": {
        "transition": [[0.9,0.1],[0.2,0.8]],
        "steps": 100,
        "trials": 1000,
        "metric": "stationary",
        "track_trajectory": true
      }
    }
  ],
  "success_criteria": {"description": "Convergence analysis completed"}
}
        "two_tailed": true,
        "ratio": 1.0,
        "n_grid": [2000, 4000, 6000, 8000, 10000, 15000]
      }
    }
  ],
  "success_criteria": {
    "description": "string",
    "metrics": ["string"],
    "thresholds": {"metric": number}
  },
  "report_outline": ["Decision", "Evidence", "Assumptions", "Limitations", "Next steps"]
}

IMPORTANT: 
- Output ONLY the AnalysisPlan JSON, no other text.
- Use proper schema validation for all tool arguments.
- ALWAYS include a numeric seed for reproducibility (e.g., "seed": 12345).
- ALWAYS include machine-checkable success criteria with numeric thresholds:
  * tv_distance_max: 0.02 — total-variation distance between final share and stationary estimate
  * ci_width_max: 0.02 — 95% CI half-width per state
  * min_trials: 10000 — align with your args
- For convergence plots, use cumulative shares (fractions) instead of counts:
  * series_from: "$markov_mcs.trajectory_data.cum_share" 
  * ylabel: "Cumulative share"
  * ref_lines_y: "$markov_mcs.pi_target" (adds dashed reference lines at stationary probabilities)
- For trajectory analysis, use metric="stationary" and track_trajectory=true to compute both stationary stats and trajectory in one run
- Create a bar chart snapshot of stationary estimates alongside the line chart using plot_bar tool
- For robust analysis, ALWAYS set stability_check: true and auto_tune: true in markov_mcs args
- The plot_line and plot_bar tools will automatically use real data from markov_mcs results and set appropriate y-axis labels.
- For AB tests, use variable references to chain results: values: ["$ab_test_ttest.group_a.rate", "$ab_test_ttest.group_b.rate"]
`;

async function executeTool(toolName: string, params: unknown) {
  try {
    if (toolName === "markov_mcs") {
      return await run_markov_mcs(params);
    } else if (toolName === "plot_line") {
      return await plot_line(params);
    } else if (toolName === "plot_bar") {
      return await plot_bar(params);
    } else if (toolName === "ab_test_ttest") {
      return await ab_test_ttest(params);
    } else if (toolName === "plot_bar_with_ci") {
      return await plot_bar_with_ci(params);
    } else if (toolName === "power_curve") {
      return await power_curve(params);
    } else {
      throw new Error(`Unknown tool: ${toolName}`);
    }
  } catch (error) {
    return { error: `Tool execution failed: ${error}` };
  }
}

function parseAnalysisPlan(text: string): AnalysisPlan | null {
  try {
    // Try to extract JSON from the text
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      console.error("No JSON found in text");
      return null;
    }
    
    const jsonStr = jsonMatch[0];
    const parsed = JSON.parse(jsonStr);
    
    // Validate against the schema
    const validated = AnalysisPlan.parse(parsed);
    console.log("Validated AnalysisPlan:", validated);
    return validated;
  } catch (error) {
    console.error("Failed to parse or validate AnalysisPlan:", error);
    return null;
  }
}

function tvDistance(p: number[], q: number[]): number {
  return 0.5 * p.reduce((s, pi, i) => s + Math.abs(pi - q[i]), 0);
}

function strictlyDecreasing(arr: number[], eps = 1e-9) {
  for (let i = 1; i < arr.length; i++) {
    if (!(arr[i] < arr[i-1] - eps)) return false;
  }
  return true;
}

function strictlyIncreasing(arr: number[], eps = 1e-9) {
  for (let i = 1; i < arr.length; i++) {
    if (!(arr[i] > arr[i-1] + eps)) return false;
  }
  return true;
}

function evaluateSuccessCriteria(
  criteria: SuccessCriteria, 
  markovResult: Record<string, unknown>
): { passed: boolean; details: Record<string, unknown>; decision: string } {
  const details: Record<string, unknown> = {};
  const pass: Record<string, boolean> = {};
  let passed = true;

  // Check minimum trials if specified
  if (criteria.min_trials !== undefined && 'trials' in markovResult) {
    const trials = markovResult.trials as number;
    details.trials = trials;
    pass.min_trials = trials >= criteria.min_trials;
    passed = passed && pass.min_trials;
  }

  // Check CI width if specified - handle both stationary and trajectory results
  if (criteria.ci_width_max !== undefined) {
    let maxCiWidth = 0;
    if ('max_ci_width' in markovResult) {
      // Stationary result
      maxCiWidth = markovResult.max_ci_width as number;
    } else if ('final_max_ci_width' in markovResult) {
      // Trajectory result
      maxCiWidth = markovResult.final_max_ci_width as number;
    } else if ('stationary_ci_high' in markovResult && 'stationary_ci_low' in markovResult) {
      // Fallback: calculate from CI arrays
      const ciHigh = markovResult.stationary_ci_high as number[];
      const ciLow = markovResult.stationary_ci_low as number[];
      maxCiWidth = Math.max(...ciHigh.map((high, i) => high - ciLow[i]));
    }
    details.ci_width = maxCiWidth;
    pass.ci_width = maxCiWidth <= criteria.ci_width_max;
    passed = passed && pass.ci_width;
  }

  // Check total variation distance if specified - compute deterministically
  if (criteria.tv_distance_max !== undefined) {
    let computedTvDistance = 0;
    let pFinal: number[] = [];
    let piTarget: number[] = [];
    
    // Get final distribution and target
    if ('stationary_estimate' in markovResult && 'pi_target' in markovResult) {
      // Stationary result
      pFinal = markovResult.stationary_estimate as number[];
      piTarget = markovResult.pi_target as number[];
    } else if ('final_cum_share' in markovResult && 'final_pi_target' in markovResult) {
      // Trajectory result
      pFinal = markovResult.final_cum_share as number[];
      piTarget = markovResult.final_pi_target as number[];
    } else if ('final_cum_share' in markovResult && 'pi_target' in markovResult) {
      // Trajectory result with pi_target from stationary computation
      pFinal = markovResult.final_cum_share as number[];
      piTarget = markovResult.pi_target as number[];
    }
    
    if (pFinal.length > 0 && piTarget.length > 0) {
      computedTvDistance = tvDistance(pFinal, piTarget);
    } else {
      // Fallback to worker's computation if data not available
      if ('tv_distance' in markovResult) {
        computedTvDistance = markovResult.tv_distance as number;
      } else if ('final_tv_distance' in markovResult) {
        computedTvDistance = markovResult.final_tv_distance as number;
      }
    }
    
    details.tv_distance = computedTvDistance;
    pass.tv = computedTvDistance <= criteria.tv_distance_max;
    passed = passed && pass.tv;
    
    // Check stability if available
    if ('stability_check' in markovResult) {
      const stability = markovResult.stability_check as Record<string, unknown>;
      if ('median_tv' in stability) {
        const medianTv = stability.median_tv as number;
        details.median_tv = medianTv;
        pass.stability = medianTv <= criteria.tv_distance_max;
        passed = passed && pass.stability;
      }
    }
  }

  // Check convergence threshold if specified
  if (criteria.convergence_threshold !== undefined && 'stationary_estimate' in markovResult) {
    const stationary = markovResult.stationary_estimate as number[];
    const maxProb = Math.max(...stationary);
    details.max_probability = maxProb;
    pass.convergence = maxProb > criteria.convergence_threshold;
    passed = passed && pass.convergence;
  }

  // Check standard deviation if specified
  if (criteria.max_std_dev !== undefined && 'stationary_estimate' in markovResult) {
    const stationary = markovResult.stationary_estimate as number[];
    const mean = stationary.reduce((a, b) => a + b, 0) / stationary.length;
    const variance = stationary.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / stationary.length;
    const stdDev = Math.sqrt(variance);
    details.std_dev = stdDev;
    pass.std_dev = stdDev < criteria.max_std_dev;
    passed = passed && pass.std_dev;
  }

  // Check MDE monotonicity for power curve results
  if (criteria.mde_monotonicity && 'n_per_arm_A' in markovResult && 'mde_rel_grid' in markovResult) {
    const nPerArm = markovResult.n_per_arm_A as number[];
    const mdeGrid = markovResult.mde_rel_grid as number[];
    
    if (nPerArm.length > 1 && mdeGrid.length > 1) {
      const isMonotonic = strictlyDecreasing(nPerArm);
      details.mde_monotonicity = isMonotonic;
      pass.mde_monotonicity = isMonotonic;
      passed = passed && pass.mde_monotonicity;
    }
  }

  // Check power monotonicity for power curve results
  if (criteria.power_monotonicity && 'power' in markovResult && 'n_grid' in markovResult) {
    const power = markovResult.power as number[];
    const nGrid = markovResult.n_grid as number[];
    
    if (power.length > 1 && nGrid.length > 1) {
      const isMonotonic = strictlyIncreasing(power);
      details.power_monotonicity = isMonotonic;
      pass.power_monotonicity = isMonotonic;
      passed = passed && pass.power_monotonicity;
    }
  }

  // Generate Decision string
  const decisionParts: string[] = [];
  if (criteria.min_trials !== undefined) {
    decisionParts.push(`Trials: ${pass.min_trials ? 'PASS' : 'FAIL'} (${details.trials}/${criteria.min_trials})`);
  }
  if (criteria.ci_width_max !== undefined) {
    const ciWidth = details.ci_width as number;
    decisionParts.push(`CI Width: ${pass.ci_width ? 'PASS' : 'FAIL'} (${ciWidth.toFixed(6)} ≤ ${criteria.ci_width_max})`);
  }
  if (criteria.tv_distance_max !== undefined) {
    const tvDistance = details.tv_distance as number;
    decisionParts.push(`TV Distance: ${pass.tv ? 'PASS' : 'FAIL'} (${tvDistance.toFixed(6)} ≤ ${criteria.tv_distance_max})`);
  }
  if (criteria.convergence_threshold !== undefined) {
    const maxProb = details.max_probability as number;
    decisionParts.push(`Convergence: ${pass.convergence ? 'PASS' : 'FAIL'} (${maxProb.toFixed(3)} > ${criteria.convergence_threshold})`);
  }
  if (criteria.max_std_dev !== undefined) {
    const stdDev = details.std_dev as number;
    decisionParts.push(`Std Dev: ${pass.std_dev ? 'PASS' : 'FAIL'} (${stdDev.toFixed(6)} < ${criteria.max_std_dev})`);
  }
  if (criteria.mde_monotonicity) {
    const mdeMonotonic = details.mde_monotonicity as boolean;
    decisionParts.push(`MDE Monotonicity: ${pass.mde_monotonicity ? 'PASS' : 'FAIL'} (${mdeMonotonic ? 'strictly decreasing' : 'not monotonic'})`);
  }
  if (criteria.power_monotonicity) {
    const powerMonotonic = details.power_monotonicity as boolean;
    decisionParts.push(`Power Monotonicity: ${pass.power_monotonicity ? 'PASS' : 'FAIL'} (${powerMonotonic ? 'strictly increasing' : 'not monotonic'})`);
  }

  const decision = `OVERALL: ${passed ? 'PASS' : 'FAIL'} | ${decisionParts.join(' | ')}`;

  return { passed, details, decision };
}

function resolveVariableReferences(
  args: Record<string, unknown>, 
  allToolResults: Array<{tool: string; params: unknown; result: unknown}>
): Record<string, unknown> {
  const resolvedArgs: Record<string, unknown> = {};
  
        // Special handling for plot_line with variable references
      if (args.tool === 'plot_line' && 'series_from' in args) {
        // Handle plot_line variable references
        const seriesFrom = args.series_from as string;
        const labels = args.labels as string;
        const x = args.x as string;
        const refLinesY = args.ref_lines_y as string;
        
        // Resolve the data
        const seriesData = resolveSingleReference(seriesFrom, allToolResults);
        const labelsData = resolveSingleReference(labels, allToolResults);
        const xData = resolveSingleReference(x, allToolResults);
        const refLinesData = refLinesY ? resolveSingleReference(refLinesY, allToolResults) : null;
        
        if (seriesData && labelsData && xData) {
          // Convert to the format expected by plot_line
          const series: Record<string, number[]> = {};
          const labelsArray = labelsData as string[];
          const seriesArray = seriesData as number[][];
          
          for (let i = 0; i < labelsArray.length; i++) {
            series[labelsArray[i]] = seriesArray.map(row => row[i]);
          }
          
          resolvedArgs.series = series;
          resolvedArgs.title = args.title;
          resolvedArgs.xlabel = args.xlabel;
          resolvedArgs.ylabel = args.ylabel;
          if (refLinesData) {
            resolvedArgs.ref_lines_y = refLinesData as number[];
          }
          
          console.log(`Converted variable references to series:`, series);
        } else {
          console.warn('Could not resolve all variable references for plot_line');
          return args; // Return original if resolution fails
        }
      } else if (args.tool === 'plot_bar' && 'series_from' in args) {
        // Handle plot_bar variable references
        const seriesFrom = args.series_from as string;
        const labels = args.labels as string;
        const refLinesY = args.ref_lines_y as string;
        
        // Resolve the data
        const seriesData = resolveSingleReference(seriesFrom, allToolResults);
        const labelsData = resolveSingleReference(labels, allToolResults);
        const refLinesData = refLinesY ? resolveSingleReference(refLinesY, allToolResults) : null;
        
        if (seriesData && labelsData) {
          // Convert to the format expected by plot_bar
          const series: Record<string, number[]> = {};
          const labelsArray = labelsData as string[];
          const seriesArray = seriesData as number[];
          
          for (let i = 0; i < labelsArray.length; i++) {
            series[labelsArray[i]] = [seriesArray[i]]; // Bar chart expects single values
          }
          
          resolvedArgs.series = series;
          resolvedArgs.title = args.title;
          resolvedArgs.xlabel = args.xlabel;
          resolvedArgs.ylabel = args.ylabel;
          if (refLinesData) {
            resolvedArgs.ref_lines_y = refLinesData as number[];
          }
          
          console.log(`Converted variable references to bar series:`, series);
        } else {
          console.warn('Could not resolve all variable references for plot_bar');
          return args; // Return original if resolution fails
        }
      } else {
    // Handle other tools normally
    for (const [key, value] of Object.entries(args)) {
      if (typeof value === 'string' && value.startsWith('$')) {
        const resolvedValue = resolveSingleReference(value, allToolResults);
        if (resolvedValue !== undefined) {
          resolvedArgs[key] = resolvedValue;
          console.log(`Resolved ${value} to:`, resolvedValue);
        } else {
          resolvedArgs[key] = value; // Keep original if resolution fails
        }
      } else {
        resolvedArgs[key] = value;
      }
    }
  }
  
  return resolvedArgs;
}

function resolveSingleReference(
  reference: string, 
  allToolResults: Array<{tool: string; params: unknown; result: unknown}>
): unknown {
  if (!reference.startsWith('$')) return undefined;
  
  const path = reference.substring(1).split('.');
  const toolName = path[0];
  const toolResult = allToolResults.find(r => r.tool === toolName);
  
  if (toolResult && toolResult.result && typeof toolResult.result === 'object') {
    let currentValue: unknown = toolResult.result;
    
    // Navigate through the path
    for (let i = 1; i < path.length; i++) {
      const pathPart = path[i];
      if (currentValue && typeof currentValue === 'object' && pathPart in currentValue) {
        currentValue = (currentValue as Record<string, unknown>)[pathPart];
      } else {
        console.warn(`Could not resolve path ${path[i]} in ${reference}`);
        return undefined;
      }
    }
    
    return currentValue;
  } else {
    console.warn(`Tool result not found for ${toolName} in ${reference}`);
    return undefined;
  }
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  const { query, plan } = body;

  let analysisPlan: AnalysisPlan | null = null;
  let text = "";

  // If a plan is provided directly, use it
  if (plan) {
    try {
      analysisPlan = AnalysisPlan.parse(plan);
      text = JSON.stringify(plan, null, 2);
    } catch (error) {
      return new Response(JSON.stringify({ 
        error: "Failed to parse provided plan",
        details: String(error)
      }), {
        headers: { "content-type": "application/json" },
      });
    }
    } else if (query) {
    // Generate plan using AI
    const { text: generatedText } = await generateText({
      model: openai("gpt-4o-mini"),
      system: SYSTEM,
      prompt: `Problem: ${query}

IMPORTANT: Output ONLY a valid AnalysisPlan JSON. No other text.

The AnalysisPlan should include:
- objective: Clear description of what to accomplish
- assumptions: List of assumptions made
- steps: Array of tool calls with proper arguments
- success_criteria: How to measure success
- report_outline: Sections for the final report

Example for power analysis with 4 steps (2 power curves + 2 visualizations):
{
  "objective": "Analyze power relationships for A/B testing with multiple steps and visualization",
  "assumptions": ["Two-proportion test", "Equal allocation", "Fixed baseline rate"],
  "steps": [
    {
      "id": "curve_mde",
      "tool": "power_curve",
      "args": {
        "mode": "mde_vs_n",
        "baseline": 0.05,
        "alpha": 0.05,
        "power": 0.8,
        "two_tailed": true,
        "ratio": 1.0,
        "mde_rel_grid": [0.02,0.03,0.04,0.05,0.06,0.08,0.10]
      }
    },
    {
      "id": "curve_power",
      "tool": "power_curve",
      "args": {
        "mode": "power_vs_n",
        "baseline": 0.05,
        "alpha": 0.05,
        "two_tailed": true,
        "ratio": 1.0,
        "mde_rel": 0.20,
        "n_grid": [2000,4000,6000,8000,10000,15000]
      }
    },
    {
      "tool": "plot_line",
      "args": {
        "y_from": "$curve_mde.n_per_arm_A",
        "x_from": "$curve_mde.mde_rel_grid",
        "label": "Sample Size vs MDE",
        "title": "Sample Size vs MDE",
        "xlabel": "MDE (% relative lift)",
        "ylabel": "Sample Size per Arm"
      }
    },
    {
      "tool": "plot_line",
      "args": {
        "y_from": "$curve_power.power",
        "x_from": "$curve_power.n_grid",
        "label": "Power vs Sample Size",
        "title": "Power vs Sample Size",
        "xlabel": "Sample Size per Arm",
        "ylabel": "Power"
      }
    }
  ],
  "success_criteria": {
    "description": "Power analysis completed with visualizations and monotonicity checks",
    "metrics": ["power_curve_generation", "visualization_quality", "monotonicity"],
    "mde_monotonicity": "n_per_arm_A strictly decreases as MDE increases",
    "power_monotonicity": "power strictly increases with n"
  },
  "report_outline": ["Decision", "Evidence", "Assumptions", "Limitations", "Next steps"]
}

Example for convergence analysis with cumulative shares:
{
  "objective": "Analyze convergence using cumulative shares and create visualization",
  "assumptions": ["Markov chain is ergodic", "Transition matrix is valid"],
  "steps": [
    {
      "tool": "markov_mcs",
      "args": {
        "transition": [[0.9, 0.1], [0.2, 0.8]],
        "steps": 1000,
        "trials": 10000,
        "metric": "stationary",
        "track_trajectory": true,
        "stability_check": true,
        "auto_tune": true,
        "seed": 12345
      }
    },
    {
      "tool": "plot_line",
      "args": {
        "series_from": "$markov_mcs.trajectory_data.cum_share",
        "labels": "$markov_mcs.trajectory_data.states",
        "x": "$markov_mcs.trajectory_data.steps",
        "title": "Cumulative Share Convergence Over Time",
        "xlabel": "Steps",
        "ylabel": "Cumulative share",
        "ref_lines_y": "$markov_mcs.pi_target"
      }
    },
    {
      "tool": "plot_bar",
      "args": {
        "series_from": "$markov_mcs.stationary_estimate",
        "labels": "$markov_mcs.trajectory_data.states",
        "title": "Final Stationary Distribution",
        "xlabel": "States",
        "ylabel": "Probability",
        "ref_lines_y": "$markov_mcs.pi_target"
      }
    }
  ],
  "success_criteria": {
    "description": "Convergence analysis with machine-checkable metrics",
    "metrics": ["convergence", "visualization_quality"],
    "tv_distance_max": 0.02,
    "ci_width_max": 0.02,
    "min_trials": 10000
  },
  "report_outline": ["Decision", "Evidence", "Assumptions", "Limitations", "Next steps"]
}

Output ONLY the AnalysisPlan JSON.`,
    });

    text = generatedText;
    
    // Parse and validate the AnalysisPlan
    analysisPlan = parseAnalysisPlan(text);
    if (!analysisPlan) {
      return new Response(JSON.stringify({ 
        error: "Failed to parse or validate AnalysisPlan",
        text: text 
      }), {
        headers: { "content-type": "application/json" },
      });
    }
  } else {
    return new Response(JSON.stringify({ 
      error: "Either 'query' or 'plan' must be provided"
    }), {
      headers: { "content-type": "application/json" },
    });
  }

  const allToolResults: Array<{tool: string; params: unknown; result: unknown}> = [];
  let successEvaluation: { passed: boolean; details: Record<string, unknown>; decision: string } | null = null;

  // Execute all steps in sequence with chaining
  console.log(`Executing ${analysisPlan.steps.length} steps from AnalysisPlan`);
  
  // Build results map for materialization
  const results: Record<string, any> = {};
  
  for (let i = 0; i < analysisPlan.steps.length; i++) {
    const step = analysisPlan.steps[i];
    console.log(`Executing step ${i + 1}: ${step.tool} with params:`, step.args);
    
    try {
      // Materialize arguments (resolve references to numbers)
      const materializedArgs = materializeArgs(step.tool, step.args, results);
      console.log(`Materialized args for ${step.tool}:`, materializedArgs);
      
      const result = await executeTool(step.tool, materializedArgs);
      allToolResults.push({
        tool: step.tool,
        params: materializedArgs,
        result
      });
      // Store results with step ID or index to handle multiple calls to same tool
      const stepKey = step.id || `${step.tool}_${i + 1}`;
      results[stepKey] = result;
      results[step.tool] = result; // Keep original for backward compatibility
      console.log(`Tool result:`, result);
      
      // Evaluate success criteria after markov_mcs or power_curve completes
      if ((step.tool === 'markov_mcs' || step.tool === 'power_curve') && analysisPlan.success_criteria) {
        // For power curve tools, we need to check both MDE and power monotonicity
        if (step.tool === 'power_curve') {
          // Check if this is the MDE tool (first power curve)
          if (result.mode === 'mde_vs_n') {
            const mdeEvaluation = evaluateSuccessCriteria(analysisPlan.success_criteria, result as Record<string, unknown>);
            console.log(`MDE success criteria evaluation:`, mdeEvaluation);
            // Store MDE evaluation for later use
            results['mde_evaluation'] = mdeEvaluation;
          }
          // Check if this is the power tool (second power curve)
          if (result.mode === 'power_vs_n') {
            const powerEvaluation = evaluateSuccessCriteria(analysisPlan.success_criteria, result as Record<string, unknown>);
            console.log(`Power success criteria evaluation:`, powerEvaluation);
            // Combine both evaluations
            const mdeEval = results['mde_evaluation'] || { passed: true, details: {}, decision: "PASS" };
            successEvaluation = {
              passed: mdeEval.passed && powerEvaluation.passed,
              details: { ...mdeEval.details, ...powerEvaluation.details },
              decision: `MDE: ${mdeEval.decision} | Power: ${powerEvaluation.decision}`
            };
          }
        } else {
          // For markov_mcs, use the original logic
          successEvaluation = evaluateSuccessCriteria(analysisPlan.success_criteria, result as Record<string, unknown>);
          console.log(`Success criteria evaluation:`, successEvaluation);
        }
      }
    } catch (error) {
      console.error(`Error executing step ${step.tool}:`, error);
      allToolResults.push({
        tool: step.tool,
        params: step.args,
        result: { error: String(error) }
      });
    }
  }

  return new Response(JSON.stringify({ 
    text: text, 
    toolResults: allToolResults,
    originalText: text,
    analysisPlan: analysisPlan,
    successEvaluation: successEvaluation
  }), {
    headers: { "content-type": "application/json" },
  });
}
