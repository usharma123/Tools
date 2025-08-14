import { z } from "zod";
import { abTestParams as abParamsA, barWithCIParams as barParamsA, powerCurveParams } from "./tools_ab_power";
import { abTestParams as abParamsB, barCIParams as barParamsB } from "./tools_stats";
import { didParams } from "./tools_causal";
import { arimaParams } from "./tools_forecast";
import { backtestParams } from "./tools_backtest";

// Helper function to get value by path
function getByPath(obj: any, path: string) {
  return path.split(".").reduce((o, k) => (o == null ? o : o[k]), obj);
}

// Helper function to resolve references
function resolveRef(v: any, results: Record<string, any>) {
  if (typeof v !== "string" || !v.startsWith("$")) return v;
  const [toolNameWithIndex, ...rest] = v.slice(1).split(".");
  
  // Handle step indices like "power_curve_2"
  let toolName = toolNameWithIndex;
  let source = results[toolName];
  
  // If not found, try with step index
  if (!source && toolNameWithIndex.includes("_")) {
    source = results[toolNameWithIndex];
  }
  const path = rest.join(".");
  if (!path) return source; // allow "$tool" to resolve to the entire result object
  return getByPath(source, path);
}

// Helper function to materialize arguments
export function materializeArgs(toolName: string, args: any, results: Record<string, any>) {
  const out: any = {};
  for (const [k, v] of Object.entries(args)) out[k] = resolveRef(v, results);

  // Lightweight ref-ops for numbers/arrays
  function applyOp(value: any, op: string, extra: any) {
    switch (op) {
      case "last_k_mean": {
        const k = extra?.k ?? 10;
        if (!Array.isArray(value) || value.length === 0) return value;
        const slice = value.slice(-k);
        return slice.reduce((a: number, b: number) => a + (Number(b) || 0), 0) / slice.length;
      }
      case "divide": {
        const byRef = extra?.by;
        const denom = typeof byRef === "string" ? resolveRef(byRef, results) : byRef;
        return typeof value === "number" && typeof denom === "number" && denom !== 0 ? value / denom : null;
      }
      default:
        return value;
    }
  }

  function resolveRefWithOps(v: any): any {
    if (v && typeof v === "object" && "ref" in v) {
      const raw = resolveRef((v as any).ref, results);
      return (v as any).op ? applyOp(raw, (v as any).op, v) : raw;
    }
    return v;
  }

  for (const [k, v] of Object.entries(args)) {
    if (v && typeof v === "object" && (v as any).ref) {
      out[k] = resolveRefWithOps(v);
    }
  }

  // Strip non-applicable parameters for deterministic tools
  if (toolName === "power_curve") {
    // Ensure no seed or randomness-related fields sneak in
    if ("seed" in out) delete out.seed;
  }

  // Special handling for plot_bar_with_ci ref-forms
  if (toolName === "plot_bar_with_ci") {
    if (Array.isArray(args.values_from)) {
      out.values = args.values_from.map((r: string) => resolveRef(r, results));
    }
    if (Array.isArray(args.ci_low_from)) {
      out.ci_low = args.ci_low_from.map((r: string) => resolveRef(r, results));
    }
    if (Array.isArray(args.ci_high_from)) {
      out.ci_high = args.ci_high_from.map((r: string) => resolveRef(r, results));
    }
    delete out.values_from;
    delete out.ci_low_from;
    delete out.ci_high_from;
  }

  // For plot_line that uses series_from + labels + x
  if (toolName === "plot_line" && args.series_from && args.labels) {
    const seriesData = resolveRef(args.series_from, results);
    const labelsResolved = resolveRef(args.labels, results);

    // Normalize labels into a flat string array
    const labelArray: string[] = Array.isArray(labelsResolved)
      ? labelsResolved.map((r: any) => (typeof r === "string" ? r : String(r)))
      : [String(labelsResolved)];

    // Handle both 1D and 2D series data
    if (Array.isArray(seriesData) && seriesData.length > 0) {
      if (Array.isArray(seriesData[0])) {
        // 2D case. seriesData could be shaped as [steps][states] or [series][steps].
        const Y: number[][] = seriesData;
        const looksLikeStepsByStates = Array.isArray(Y[0]) && Y[0].length === labelArray.length;
        // If rows are steps and columns are states, transpose to [series][steps]
        const seriesByLabel: number[][] = looksLikeStepsByStates
          ? labelArray.map((_, colIndex) => Y.map(row => row[colIndex]))
          : Y;

        out.series = {};
        labelArray.forEach((lab: string, i: number) => {
          if (seriesByLabel[i]) out.series[lab] = seriesByLabel[i];
        });
      } else {
        // 1D array case (single series)
        const Y: number[] = seriesData as number[];
        const labelName = labelArray[0] ?? "Series";
        out.series = { [labelName]: Y };
      }
    }
    delete out.series_from;
    delete out.labels;
  }

  // For plot_line that only provides series_from (single series)
  if (toolName === "plot_line" && args.series_from && !args.labels) {
    const seriesData = resolveRef(args.series_from, results);
    if (Array.isArray(seriesData)) {
      const labelName = typeof args.label === "string" ? args.label : "Series";
      out.series = { [labelName]: seriesData };
      delete out.series_from;
    }
  }

  // For plot_line that uses y_from + x_from + label (single series)
  if (toolName === "plot_line" && args.y_from && args.label) {
    const yData = resolveRef(args.y_from, results);
    const xData = args.x_from ? resolveRef(args.x_from, results) : null;
    const label = args.label;
    
    if (Array.isArray(yData)) {
      out.series = { [label]: yData };
      if (xData && Array.isArray(xData)) {
        out.x = xData;
      }
    }
    delete out.y_from;
    delete out.x_from;
    delete out.label;
  }

  // For plot_bar that uses series_from + labels
  if (toolName === "plot_bar" && args.series_from && args.labels) {
    const seriesData: number[] = resolveRef(args.series_from, results); // shape [n]
    const labelsResolved = resolveRef(args.labels, results);
    const labels: string[] = Array.isArray(labelsResolved)
      ? labelsResolved.map((r: any) => (typeof r === "string" ? r : String(r)))
      : [String(labelsResolved)];

    out.series = {};
    labels.forEach((lab: string, i: number) => {
      out.series[lab] = [seriesData?.[i] ?? 0];
    });
    delete out.series_from;
    delete out.labels;
  }
  return out;
}

// Define specific argument schemas for each tool
export const MarkovMcsArgs = z.object({
  transition: z.array(z.array(z.number())),
  start: z.number().optional(),
  steps: z.number().optional(),
  trials: z.number().optional(),
  burnin: z.number().optional(),
  seed: z.number().optional(),
  metric: z.enum(["stationary", "avg_reward", "trajectory"]).optional(),
  rewards: z.array(z.number()).optional(),
  ci: z.number().optional(),
  track_trajectory: z.boolean().optional(),
});

export const PlotLineArgs = z.object({
  // Support both direct series and variable references
  series: z.record(z.string(), z.array(z.number())).optional(),
  series_from: z.string().optional(), // Variable reference like "$markov_mcs.trajectory_data.cumulative_means"
  labels: z.union([z.string(), z.array(z.string())]).optional(), // Variable reference for labels (string or array)
  // Single series helpers
  y_from: z.string().optional(), // Variable reference for y values
  x_from: z.string().optional(), // Variable reference for x values
  label: z.string().optional(), // Single label for single series
  title: z.string().optional(),
  xlabel: z.string().optional(),
  ylabel: z.string().optional(),
  ref_lines_y: z.union([z.string(), z.array(z.number())]).optional(), // Reference lines at stationary probabilities (string for variable reference, array for resolved values)
});

export const PlotBarArgs = z.object({
  // Support both direct series and variable references
  series: z.record(z.string(), z.array(z.number())).optional(),
  series_from: z.string().optional(), // Variable reference like "$markov_mcs.stationary_estimate"
  labels: z.string().optional(), // Variable reference for labels
  title: z.string().optional(),
  xlabel: z.string().optional(),
  ylabel: z.string().optional(),
  ref_lines_y: z.union([z.string(), z.array(z.number())]).optional(), // Reference lines at stationary probabilities
});

// New summarization tool args
export const SummarizeResultsArgs = z.object({
  markov: z.any().optional(),
  ab_test: z.any().optional(),
  power_curve: z.any().optional(),
  notes: z.string().optional(),
});

// Define the Step schema with discriminated unions
export const Step = z.discriminatedUnion("tool", [
  z.object({
    id: z.string().optional(),
    tool: z.literal("markov_mcs"),
    args: MarkovMcsArgs,
  }),
  z.object({
    id: z.string().optional(),
    tool: z.literal("plot_line"),
    args: PlotLineArgs,
  }),
  z.object({
    id: z.string().optional(),
    tool: z.literal("plot_bar"),
    args: PlotBarArgs,
  }),
  z.object({
    id: z.string().optional(),
    tool: z.literal("ab_test_ttest"),
    args: z.union([abParamsA, abParamsB]),
  }),
  z.object({
    id: z.string().optional(),
    tool: z.literal("plot_bar_with_ci"),
    args: z.union([
      // Original strict schema (older module) and new stats module
      barParamsA,
      barParamsB,
      // Reference-based schema
      z.object({
        labels: z.array(z.string()).min(1),
        values_from: z.array(z.string()).min(1),
        ci_low_from: z.array(z.string()).min(1),
        ci_high_from: z.array(z.string()).min(1),
        title: z.string().optional(),
        xlabel: z.string().optional(),
        ylabel: z.string().optional(),
        ylim: z.tuple([z.number(), z.number()]).optional(),
      })
    ]),
  }),
  z.object({
    id: z.string().optional(),
    tool: z.literal("power_curve"),
    args: powerCurveParams,
  }),
  z.object({
    id: z.string().optional(),
    tool: z.literal("summarize_results"),
    args: SummarizeResultsArgs,
  }),
  z.object({
    id: z.string().optional(),
    tool: z.literal("causal_impact"),
    args: didParams,
  }),
  z.object({
    id: z.string().optional(),
    tool: z.literal("forecast_arima"),
    args: arimaParams,
  }),
  z.object({
    id: z.string().optional(),
    tool: z.literal("forecast_backtest"),
    args: backtestParams,
  }),
  z.object({
    id: z.string().optional(),
    tool: z.literal("choice_logit"),
    args: z.object({
      csv: z.string(),
      choice_col: z.string().optional(),
      alt_col: z.string().optional(),
      chosen_col: z.string().optional(),
      feature_cols: z.array(z.string()).min(1),
      standardize: z.boolean().optional(),
      add_alt_dummies: z.boolean().optional(),
      base_alt: z.string().optional(),
      l2_lambda: z.number().optional(),
      scenarios: z.array(z.object({
        name: z.string(),
        scope_alts: z.array(z.string()).optional(),
        adjustments: z.record(z.string(), z.union([
          z.number(),
          z.string(),
          z.object({ mode: z.enum(["add","mul"]).default("add"), value: z.number() })
        ]))
      })).optional()
    })
  }),
]);

export const SuccessCriteria = z.object({
  description: z.string(),
  metrics: z.array(z.string()).optional(),
  thresholds: z.record(z.string(), z.number()).optional(),
  // Machine-checkable criteria
  ci_width_max: z.number().optional(), // Maximum allowed confidence interval width
  convergence_threshold: z.number().optional(), // Minimum required convergence
  min_trials: z.number().optional(), // Minimum number of trials required
  max_std_dev: z.number().optional(), // Maximum allowed standard deviation
  tv_distance_max: z.number().optional(), // Maximum allowed total variation distance
  // Power curve monotonicity checks
  mde_monotonicity: z.string().optional(), // Description of MDE monotonicity check
  power_monotonicity: z.string().optional(), // Description of power monotonicity check
  // Forecast-specific checks (TS)
  forecast_require_lengths_match_horizon: z.boolean().optional(),
  forecast_require_forecast_within_ci: z.boolean().optional(),
  // If provided, expect worker to return backtest block when backtest_k is set in tool args
  forecast_backtest_require_better_than_naive: z.boolean().optional(),
  forecast_backtest_smape_max: z.number().optional(),
  forecast_backtest_improvement_min: z.number().optional(),
  // Choice model-specific checks
  choice_require_converged: z.boolean().optional(),
  choice_require_no_separation: z.boolean().optional(),
  choice_max_group_prob_max: z.number().optional(),
  choice_pseudo_r2_max: z.number().optional(),
  choice_coef_sign: z.record(z.string(), z.enum(["positive", "negative", "nonzero"]).optional()).optional(),
  choice_scenario_expectations: z.array(z.object({
    scenario_name: z.string(),
    alt: z.string(),
    direction: z.enum(["increase", "decrease"]).optional(),
    min_delta: z.number().optional()
  })).optional()
});

export const AnalysisPlan = z.object({
  objective: z.string(),
  assumptions: z.array(z.string()).optional(),
  steps: z.array(Step).min(1),
  success_criteria: SuccessCriteria.optional(),
  report_outline: z.array(z.string()).default([
    "Decision", "Evidence", "Assumptions", "Limitations", "Next steps"
  ]),
});

export type AnalysisPlan = z.infer<typeof AnalysisPlan>;
export type Step = z.infer<typeof Step>;
export type MarkovMcsArgs = z.infer<typeof MarkovMcsArgs>;
export type PlotLineArgs = z.infer<typeof PlotLineArgs>;
export type SuccessCriteria = z.infer<typeof SuccessCriteria>;
export type SummarizeResultsArgs = z.infer<typeof SummarizeResultsArgs>;
