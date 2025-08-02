import { z } from "zod";
const WORKER = process.env.WORKER_URL || "http://localhost:8000";

// --- ab_test_ttest (both modes) ---
export const abTestParams = z.union([
  z.object({
    successes_a: z.number().int().nonnegative(),
    trials_a: z.number().int().positive(),
    successes_b: z.number().int().nonnegative(),
    trials_b: z.number().int().positive(),
    alpha: z.number().min(0).max(1).default(0.05),
    two_tailed: z.boolean().default(true),
    assume_independent: z.boolean().default(true),
  }),
  z.object({
    mean_a: z.number(), sd_a: z.number().nonnegative(), n_a: z.number().int().positive(),
    mean_b: z.number(), sd_b: z.number().nonnegative(), n_b: z.number().int().positive(),
    alpha: z.number().min(0).max(1).default(0.05),
    two_tailed: z.boolean().default(true),
    equal_var: z.boolean().default(false),
    assume_independent: z.boolean().default(true),
  })
]);

export async function ab_test_ttest(args: unknown) {
  const parsed = abTestParams.parse(args);
  const res = await fetch(`${WORKER}/tools/ab_test_ttest`, {
    method: "POST", headers: {"content-type":"application/json"}, body: JSON.stringify(parsed)
  });
  if (!res.ok) throw new Error(`ab_test_ttest failed: ${res.status}`);
  return res.json();
}

// --- plot_bar_with_ci ---
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

export async function plot_bar_with_ci(args: unknown) {
  const parsed = barWithCIParams.parse(args);
  const res = await fetch(`${WORKER}/tools/plot_bar_with_ci`, {
    method: "POST", headers: {"content-type":"application/json"}, body: JSON.stringify(parsed)
  });
  if (!res.ok) throw new Error(`plot_bar_with_ci failed: ${res.status}`);
  return res.json();
}

// --- power_curve ---
export const powerCurveParams = z.union([
  z.object({
    mode: z.literal("mde_vs_n"),
    baseline: z.number().min(0).max(1),
    alpha: z.number().min(0).max(1).default(0.05),
    two_tailed: z.boolean().default(true),
    ratio: z.number().positive().default(1),
    power: z.number().min(0).max(1).default(0.8),
    mde_rel_grid: z.array(z.number()).min(1)  // e.g., [0.02,0.03,...]
  }),
  z.object({
    mode: z.literal("power_vs_n"),
    baseline: z.number().min(0).max(1),
    alpha: z.number().min(0).max(1).default(0.05),
    two_tailed: z.boolean().default(true),
    ratio: z.number().positive().default(1),
    mde_rel: z.number(),                       // e.g., 0.1 for +10%
    n_grid: z.array(z.number().int().positive()).min(1)
  })
]);

export async function power_curve(args: unknown) {
  const parsed = powerCurveParams.parse(args);
  const res = await fetch(`${WORKER}/tools/power_curve`, {
    method: "POST", headers: {"content-type":"application/json"}, body: JSON.stringify(parsed)
  });
  if (!res.ok) throw new Error(`power_curve failed: ${res.status}`);
  return res.json();
} 