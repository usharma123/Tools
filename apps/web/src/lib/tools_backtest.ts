import { z } from "zod";
const WORKER = process.env.WORKER_URL || "http://localhost:8000";

export const backtestParams = z.object({
  ts: z.array(z.number()).min(8),
  horizon: z.number().int().positive(),
  step: z.number().int().positive().optional(),
  folds: z.number().int().min(2).optional(),
  min_train: z.number().int().min(4).optional(),
  seasonal_period: z.number().int().positive().optional(),
  alpha: z.number().min(0).max(0.2).optional(),
  order: z
    .tuple([
      z.number().int().nonnegative(),
      z.number().int().nonnegative(),
      z.number().int().nonnegative(),
    ])
    .optional(),
});

export async function forecast_backtest(args: unknown) {
  const body = JSON.stringify(backtestParams.parse(args));
  const res = await fetch(`${WORKER}/tools/forecast_backtest`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body,
  });
  if (!res.ok) throw new Error("forecast_backtest failed");
  return res.json();
}


