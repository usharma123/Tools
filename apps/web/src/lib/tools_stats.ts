import { abTestParams, barWithCIParams } from "./schemas";

const WORKER = process.env.WORKER_URL || "http://localhost:8000";

// Export schemas for backward compatibility
export { abTestParams };

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

// Alias for backward compatibility
export const barCIParams = barWithCIParams;

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


