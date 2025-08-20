import { abTestParams, barWithCIParams, powerCurveParams } from "./schemas";

const WORKER = process.env.WORKER_URL || "http://localhost:8000";

// Export schemas for backward compatibility
export { abTestParams, barWithCIParams, powerCurveParams };

export async function ab_test_ttest(args: unknown) {
  const parsed = abTestParams.parse(args);
  const res = await fetch(`${WORKER}/tools/ab_test_ttest`, {
    method: "POST", headers: {"content-type":"application/json"}, body: JSON.stringify(parsed)
  });
  if (!res.ok) throw new Error(`ab_test_ttest failed: ${res.status}`);
  return res.json();
}


export async function plot_bar_with_ci(args: unknown) {
  const parsed = barWithCIParams.parse(args);
  const res = await fetch(`${WORKER}/tools/plot_bar_with_ci`, {
    method: "POST", headers: {"content-type":"application/json"}, body: JSON.stringify(parsed)
  });
  if (!res.ok) throw new Error(`plot_bar_with_ci failed: ${res.status}`);
  return res.json();
}


export async function power_curve(args: unknown) {
  const parsed = powerCurveParams.parse(args);
  const res = await fetch(`${WORKER}/tools/power_curve`, {
    method: "POST", headers: {"content-type":"application/json"}, body: JSON.stringify(parsed)
  });
  if (!res.ok) throw new Error(`power_curve failed: ${res.status}`);
  return res.json();
} 