import { z } from "zod";

const WORKER = process.env.WORKER_URL || "http://localhost:8000";

const markovMcsParams = z.object({
  transition: z.array(z.array(z.number())),
  start: z.number().optional(),
  steps: z.number().optional(),
  trials: z.number().optional(),
  burnin: z.number().optional(),
  seed: z.number().optional(),
  metric: z.enum(["stationary","avg_reward"]).optional(),
  rewards: z.array(z.number()).optional(),
  ci: z.number().optional(),
});

export async function run_markov_mcs(args: unknown) {
  const parsed = markovMcsParams.parse(args);
  try {
    const res = await fetch(`${WORKER}/tools/markov_mcs`, {
      method: "POST",
      headers: {"content-type":"application/json"},
      body: JSON.stringify(parsed)
    });
    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`markov_mcs failed: ${res.status} - ${errorText}`);
    }
    const responseText = await res.text();
    if (!responseText) {
      throw new Error("Empty response from worker");
    }
    return JSON.parse(responseText);
  } catch (error) {
    console.error("Error calling markov_mcs:", String(error));
    throw new Error(`Failed to call markov_mcs: ${error instanceof Error ? error.message : String(error)}`);
  }
}

const plotLineParams = z.object({
  series: z.record(z.string(), z.array(z.number())),
  title: z.string().optional(),
  xlabel: z.string().optional(),
  ylabel: z.string().optional(),
});

export async function plot_line(args: unknown) {
  const parsed = plotLineParams.parse(args);
  try {
    const res = await fetch(`${WORKER}/tools/plot_line`, {
      method: "POST",
      headers: {"content-type":"application/json"},
      body: JSON.stringify(parsed)
    });
    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`plot_line failed: ${res.status} - ${errorText}`);
    }
    const responseText = await res.text();
    if (!responseText) {
      throw new Error("Empty response from worker");
    }
    return JSON.parse(responseText);
  } catch (error) {
    console.error("Error calling plot_line:", String(error));
    throw new Error(`Failed to call plot_line: ${error instanceof Error ? error.message : String(error)}`);
  }
}
