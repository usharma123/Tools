import { z } from "zod";

// Common schemas used across multiple tools
export const abTestParams = z.union([
  // Binary test mode
  z.object({
    successes_a: z.number().int().nonnegative(),
    trials_a: z.number().int().positive(),
    successes_b: z.number().int().nonnegative(),
    trials_b: z.number().int().positive(),
    alpha: z.number().min(0).max(1).default(0.05),
    two_tailed: z.boolean().default(true),
    assume_independent: z.boolean().default(true),
  }),
  // Continuous test mode
  z.object({
    mean_a: z.number(), 
    sd_a: z.number().nonnegative(), 
    n_a: z.number().int().positive(),
    mean_b: z.number(), 
    sd_b: z.number().nonnegative(), 
    n_b: z.number().int().positive(),
    alpha: z.number().min(0).max(1).default(0.05),
    two_tailed: z.boolean().default(true),
    equal_var: z.boolean().default(false),
    assume_independent: z.boolean().default(true),
  })
]);

export const barWithCIParams = z.object({
  labels: z.array(z.string()).min(1),
  values: z.array(z.number()).min(1),
  ci_low: z.array(z.number()).min(1),
  ci_high: z.array(z.number()).min(1),
  title: z.string().optional(),
  xlabel: z.string().optional(),
  ylabel: z.string().optional(),
  ylim: z.tuple([z.number(), z.number()]).optional(),
});

export const powerCurveParams = z.union([
  // MDE vs N mode
  z.object({
    mode: z.literal("mde_vs_n"),
    baseline: z.number().min(0).max(1),
    alpha: z.number().min(0).max(1).default(0.05),
    power: z.number().min(0).max(1).default(0.8),
    two_tailed: z.boolean().default(true),
    ratio: z.number().positive().default(1.0),
    mde_rel_grid: z.array(z.number()).min(1),  // Allow any number array for flexibility
  }),
  // Power vs N mode
  z.object({
    mode: z.literal("power_vs_n"),
    baseline: z.number().min(0).max(1),
    alpha: z.number().min(0).max(1).default(0.05),
    two_tailed: z.boolean().default(true),
    ratio: z.number().positive().default(1.0),
    mde_rel: z.number(),  // Allow any number for flexibility
    n_grid: z.array(z.number().int().positive()).min(1),
  })
]);

export const markovMcsParams = z.object({
  transition: z.array(z.array(z.number())),
  start: z.number().optional(),
  steps: z.number().optional(),
  trials: z.number().optional(),
  burnin: z.number().optional(),
  seed: z.number(),
  metric: z.enum(["stationary","avg_reward","trajectory"]).optional(),
  rewards: z.array(z.number()).optional(),
  ci: z.number().optional(),
  track_trajectory: z.boolean().optional(),
});

export const plotLineParams = z.object({
  // Support both direct series and variable references
  series: z.record(z.string(), z.array(z.number())).optional(),
  series_from: z.string().optional(),
  labels: z.union([z.string(), z.array(z.string())]).optional(),
  // Single series helpers
  y_from: z.string().optional(),
  x_from: z.string().optional(),
  label: z.string().optional(),
  title: z.string().optional(),
  xlabel: z.string().optional(),
  ylabel: z.string().optional(),
  ref_lines_y: z.union([z.string(), z.array(z.number())]).optional(),
});

export const plotBarParams = z.object({
  // Support both direct series and variable references
  series: z.record(z.string(), z.array(z.number())).optional(),
  series_from: z.string().optional(),
  labels: z.string().optional(),
  title: z.string().optional(),
  xlabel: z.string().optional(),
  ylabel: z.string().optional(),
  ref_lines_y: z.union([z.string(), z.array(z.number())]).optional(),
});

export const summarizeParams = z.object({
  markov: z.any().optional(),
  ab_test: z.any().optional(),
  power_curve: z.any().optional(),
  forecast: z.any().optional(),
  backtest: z.any().optional(),
  notes: z.string().optional(),
});