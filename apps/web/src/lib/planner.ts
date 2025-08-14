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

function extractNumberByKey(text: string, key: string): number | undefined {
  const r = new RegExp(`${key}\\s*[:=]\\s*([0-9]*\\.?[0-9]+)`, "i");
  const m = text.match(r);
  if (!m) return undefined;
  const n = Number(m[1]);
  return Number.isFinite(n) ? n : undefined;
}

function extractFirstBracketArrayNumbers(text: string): number[] | null {
  const start = text.indexOf("[");
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
    if (Array.isArray(parsed) && parsed.every((x: unknown) => typeof x === "number")) return parsed as number[];
  } catch {}
  return null;
}

// VERY simple planner: choose tool by keywords; in prod, swap with an LLM.
export async function buildPlanVerbose(input: PlannerInput): Promise<{ plan: ChatPlan; logs: Array<{ message: string; meta?: Record<string, any> }> }> {
  const msgs = input.messages;
  const lastMsg = msgs[msgs.length - 1]?.content || "";
  const last = lastMsg.toLowerCase();
  const logs: Array<{ message: string; meta?: Record<string, any> }> = [];

  // Detect most recent intent by scanning backwards and picking the first match
  type Intent = 'power'|'backtest'|'forecast'|'markov'|'ab'|'causal'|null;
  const detectIntent = (): { intent: Intent; text: string } => {
    for (let i = msgs.length - 1; i >= 0; i--) {
      const t = msgs[i]?.content || "";
      const s = t.toLowerCase();
      if (s.includes("power curve") || (s.includes("power") && (s.includes("mde") || s.includes("sample size") || s.includes("n in")))) return { intent: 'power', text: t };
      if (s.includes("backtest")) return { intent: 'backtest', text: t };
      if (s.includes("forecast")) return { intent: 'forecast', text: t };
      if (s.includes("markov") || s.includes("stationary") || s.includes("transition")) return { intent: 'markov', text: t };
      if (s.includes("ab test") || s.includes("a/b")) return { intent: 'ab', text: t };
      if (s.includes("causal") || s.includes("policy")) return { intent: 'causal', text: t };
    }
    return { intent: null, text: lastMsg };
  };

  const { intent, text: intentText } = detectIntent();
  logs.push({ message: "intent_detected", meta: { intent } });

  // crude routing with priority on latest intent only
  if (intent === 'power') {
    const text = intentText || lastMsg;
    const baseline = extractNumberByKey(text, "baseline") ?? 0.05;
    const mdeRelExplicit = extractNumberByKey(text, "mde") ?? extractNumberByKey(text, "relative mde");
    const effectAbs = extractNumberByKey(text, "effect");
    const mde_rel = (typeof mdeRelExplicit === 'number') ? mdeRelExplicit : ((typeof effectAbs === 'number' && typeof baseline === 'number' && baseline > 0) ? (effectAbs / baseline) : 0.1);
    const nGrid = (extractFirstBracketArrayNumbers(text) || [2000, 4000, 6000]).map(x => Math.max(1, Math.round(x)));
    const step = { id: "curve_power", tool: "power_curve", args: { mode: "power_vs_n", baseline, alpha: 0.05, two_tailed: true, ratio: 1, mde_rel, n_grid: nGrid } } as const;
    const asks = missingFields((tools as any)[step.tool].params, step.args);
    logs.push({ message: "route_power_curve", meta: { baseline, mde_rel: mde_rel, n_grid: nGrid } });
    const plan = asks.length ? { steps: [step as any], ask: asks } : { steps: [step as any] };
    return { plan, logs };
  }
  if (intent === 'markov') {
    const text = intentText || lastMsg;
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
    logs.push({ message: "route_markov", meta: { steps: args.steps, trials: args.trials } });
    const plan = asks.length ? { steps: [step as any], ask: asks } : { steps: [step as any] };
    return { plan, logs };
  }
  if (intent === 'forecast') {
    // Try to parse series from the latest message (answer to ask)
    const extractCommaSeparatedNumbers = (s: string): number[] | null => {
      const m = s.match(/(?:^|\s)(?:\d+\.?\d*)(?:\s*,\s*\d+\.?\d*){2,}(?:\s|$)/);
      if (!m) return null;
      const raw = m[0].trim().replace(/^\s|\s$/g, '');
      const arr = raw.split(/\s*,\s*/).map(Number).filter(n => Number.isFinite(n));
      return arr.length >= 3 ? arr : null;
    };
    const source = intentText || lastMsg;
    const fromLast = extractCommaSeparatedNumbers(source) || extractFirstBracketArrayNumbers(source) || [];
    const ts = (fromLast.length >= 3 ? fromLast : undefined) ?? "$asset:ts.series";
    const step = { id: "fc", tool: "forecast_arima", args: { ts, horizon: 6, seasonal_period: 12 } } as const;
    const asks = missingFields(tools[step.tool].params, step.args);
    logs.push({ message: "route_forecast_arima", meta: { ts_len: Array.isArray(fromLast) ? fromLast.length : undefined } });
    const plan = asks.length ? { steps: [step as any], ask: asks } : { steps: [step as any] };
    return { plan, logs };
  }
  if (intent === 'backtest') {
    const text = intentText || lastMsg;
    // Parse series
    const extractCommaSeparatedNumbers = (s: string): number[] | null => {
      const m = s.match(/(?:^|\s)(?:\d+\.?\d*)(?:\s*,\s*\d+\.?\d*){2,}(?:\s|$)/);
      if (!m) return null;
      const raw = m[0].trim().replace(/^\s|\s$/g, '');
      const arr = raw.split(/\s*,\s*/).map(Number).filter(n => Number.isFinite(n));
      return arr.length >= 3 ? arr : null;
    };
    // Parse ARIMA(p,d,q)
    const extractArimaOrder = (s: string): [number, number, number] | undefined => {
      const m = s.match(/arima\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)/i);
      if (!m) return undefined;
      return [Number(m[1]), Number(m[2]), Number(m[3])];
    };
    const fromText = extractCommaSeparatedNumbers(text) || extractFirstBracketArrayNumbers(text) || [];
    const ts = (fromText.length >= 8 ? fromText : undefined) ?? "$asset:ts.series";
    const horizon = extractNumberByKey(text, "horizon") ?? 6;
    const folds = extractNumberAfter(text, "folds") ?? 5;
    const seasonal_period = extractNumberByKey(text, "seasonal_period") ?? 12;
    const alpha = extractNumberByKey(text, "alpha") ?? 0.05;
    const order = extractArimaOrder(text) ?? [1, 1, 1];
    const args: any = { ts, horizon, folds, seasonal_period, alpha, order };
    const step = { id: "bt", tool: "forecast_backtest", args } as const;
    const asks = missingFields(tools[step.tool].params, step.args);
    logs.push({ message: "route_forecast_backtest", meta: { ts_len: Array.isArray(fromText) ? fromText.length : undefined, horizon, folds, seasonal_period, alpha, order } });
    const plan = asks.length ? { steps: [step as any], ask: asks } : { steps: [step as any] };
    return { plan, logs };
  }
  if (intent === 'ab') {
    const step = { id: "ab", tool: "ab_test_ttest", args: {} } as const;
    const asks = missingFields(tools[step.tool].params, step.args);
    logs.push({ message: "route_ab_test" });
    const plan = asks.length ? { steps: [step as any], ask: asks } : { steps: [step as any] };
    return { plan, logs };
  }
  if (intent === 'causal') {
    const step = { id: "did", tool: "causal_impact", args: { csv: "$asset:table.csv" } } as const;
    const asks = missingFields(tools[step.tool].params, step.args);
    logs.push({ message: "route_causal_impact" });
    const plan = asks.length ? { steps: [step as any], ask: asks } : { steps: [step as any] };
    return { plan, logs };
  }
  // fallback: ask user what they want
  logs.push({ message: "route_unknown" });
  return { plan: { steps: [], ask: [{ path: "tool", question: "What would you like to do? (forecast, A/B, causal)" }] }, logs };
}

export async function buildPlan(input: PlannerInput): Promise<ChatPlan> {
  const { plan } = await buildPlanVerbose(input);
  return plan;
}


