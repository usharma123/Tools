import { z } from "zod";

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
  labels: z.string().optional(), // Variable reference for labels
  x: z.string().optional(), // Variable reference for x-axis data
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

// Define the Step schema with discriminated unions
export const Step = z.discriminatedUnion("tool", [
  z.object({
    tool: z.literal("markov_mcs"),
    args: MarkovMcsArgs,
  }),
  z.object({
    tool: z.literal("plot_line"),
    args: PlotLineArgs,
  }),
  z.object({
    tool: z.literal("plot_bar"),
    args: PlotBarArgs,
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
