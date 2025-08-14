import type { ChatPlan } from "./types";
import { tools } from "./tools/index";
import { z } from "zod";
import { missingFields } from "./slotFiller";

export type PlannerInput = {
  messages: { role: "user"|"assistant"|"system"; content: string }[];
  assets?: Array<{ assetId: string; kind: "table"|"timeseries"|"file"; meta: any }>;
};

function extractFirstBracketMatrix(text: string): number[][] | null {
  const start = text.indexOf("[[");
  if (start < 0) return null;
  let depth = 0;
  let end = -1;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (ch === "[") depth++;
    else if (ch === "]") {
      depth--;
      if (depth === 0) { end = i; break; }
    }
  }
  if (end < 0) return null;
  const candidate = text.slice(start, end + 1);
  try {
    const parsed = JSON.parse(candidate);
    if (Array.isArray(parsed) && parsed.every((row: unknown) => Array.isArray(row) && (row as unknown[]).every((x) => typeof x === "number"))) {
      return parsed as number[][];
    }
  } catch {}
  return null;
}

function extractNumberAfter(text: string, key: string): number | undefined {
  const r = new RegExp(`(\\d+(?:e\\d+)?)\\s*${key}`, "i");
  const m = text.match(r);
  if (!m) return undefined;
  const n = Number(m[1]);
  return Number.isFinite(n) ? n : undefined;
}

// VERY simple planner: choose tool by keywords; in prod, swap with an LLM.
export async function buildPlan(input: PlannerInput): Promise<ChatPlan> {
  const last = input.messages[input.messages.length - 1]?.content?.toLowerCase() || "";
  // crude routing
  if (last.includes("markov") || last.includes("stationary") || last.includes("transition")) {
    const text = input.messages[input.messages.length - 1]?.content || "";
    const maybeT = extractFirstBracketMatrix(text);
    const maybeSteps = extractNumberAfter(text, "steps");
    const maybeTrials = extractNumberAfter(text, "trials");
    const args: any = {
      transition: maybeT ?? "$asset:ts.transition",
      steps: maybeSteps ?? 1000,
      trials: maybeTrials ?? 10000,
      metric: "stationary",
      track_trajectory: true,
      seed: 12345,
    };
    const step = { id: "mk", tool: "markov_mcs", args } as const;
    const asks = missingFields(tools[step.tool].params, step.args);
    return asks.length ? { steps: [step as any], ask: asks } : { steps: [step as any] };
  }
  if (last.includes("forecast")) {
    const step = { id: "fc", tool: "forecast_arima", args: { ts: "$asset:ts.series", horizon: 6, seasonal_period: 12 } } as const;
    const asks = missingFields(tools[step.tool].params, step.args);
    return asks.length ? { steps: [step as any], ask: asks } : { steps: [step as any] };
  }
  if (last.includes("ab test") || last.includes("a/b")) {
    const step = { id: "ab", tool: "ab_test_ttest", args: {} } as const;
    const asks = missingFields(tools[step.tool].params, step.args);
    return asks.length ? { steps: [step as any], ask: asks } : { steps: [step as any] };
  }
  if (last.includes("causal") || last.includes("policy")) {
    const step = { id: "did", tool: "causal_impact", args: { csv: "$asset:table.csv" } } as const;
    const asks = missingFields(tools[step.tool].params, step.args);
    return asks.length ? { steps: [step as any], ask: asks } : { steps: [step as any] };
  }
  // fallback: ask user what they want
  return { steps: [], ask: [{ path: "tool", question: "What would you like to do? (forecast, A/B, causal)" }] };
}


