import { z } from "zod";
const WORKER = process.env.WORKER_URL || "http://localhost:8000";

export const arimaParams = z.object({
  ts: z.array(z.number()).min(3),
  horizon: z.number().int().positive(),
  seasonal_period: z.number().int().positive().optional(),
  alpha: z.number().min(0).max(0.2).optional(),
  backtest_k: z.number().int().positive().optional()
  , start_date: z.string().optional()
  , freq: z.string().optional()
});

export async function forecast_arima(args: unknown){
  const body = JSON.stringify(arimaParams.parse(args));
  const res  = await fetch(`${WORKER}/tools/forecast_arima`,{
    method:"POST", headers:{"content-type":"application/json"}, body});
  if(!res.ok) throw new Error("forecast_arima failed");
  return res.json();
}


