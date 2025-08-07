import { z } from "zod";
const WORKER = process.env.WORKER_URL || "http://localhost:8000";

// ab_test_ttest params (binary or continuous)
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
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(parsed),
  });
  if (!res.ok) throw new Error("ab_test_ttest failed");
  return res.json();
}

// plot_bar_with_ci params
export const barCIParams = z.object({
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
  const parsed = barCIParams.parse(args);
  const res = await fetch(`${WORKER}/tools/plot_bar_with_ci`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(parsed),
  });
  if (!res.ok) throw new Error("plot_bar_with_ci failed");
  return res.json();
}


