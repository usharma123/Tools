import { stableStringify } from "./hash";

function getByPath(obj: any, dotted: string) {
  return dotted.split(".").reduce((o, k) => (o == null ? o : (o as any)[k]), obj);
}

function isDollarRef(v: any): v is string {
  return typeof v === "string" && v.startsWith("$");
}

function applyOp(value: any, op?: string, extra?: any, results?: any) {
  switch (op) {
    case "last_k_mean": {
      const k = extra?.k ?? 10;
      if (!Array.isArray(value)) return value;
      const slice = value.slice(-k);
      return slice.reduce((a: number, b: number) => a + b, 0) / (slice.length || 1);
    }
    case "divide": {
      const denomRef = extra?.by;
      const denom = typeof denomRef === "string" && denomRef.startsWith("$")
        ? getByPath(results, denomRef.slice(1))
        : denomRef;
      return typeof value === "number" && typeof denom === "number" && denom !== 0 ? value / denom : null;
    }
    default:
      return value;
  }
}

export function resolveValue(v: any, results: Record<string, any>) {
  // Object ref with op: { ref:"$step.field", op:"..." }
  if (v && typeof v === "object" && "ref" in v) {
    const raw = getByPath(results, (v as any).ref.slice(1));
    return applyOp(raw, (v as any).op as string | undefined, v, results);
  }
  // Dollar string ref: "$step.field"
  if (isDollarRef(v)) return getByPath(results, v.slice(1));
  // Plain value or arrays/objects: recurse
  if (Array.isArray(v)) return v.map((x) => resolveValue(x, results));
  if (v && typeof v === "object") {
    const out: any = {};
    for (const [k, val] of Object.entries(v)) out[k] = resolveValue(val, results);
    return out;
  }
  return v;
}

/** Build final numeric args expected by tools (also supports plot helpers). */
export function materializeArgs(toolName: string, args: any, results: Record<string, any>) {
  const out: any = resolveValue(args, results);

  // Single-series helpers for plot_line (if present)
  if (toolName === "plot_line" && out.y_from) {
    const y = out.y_from; const x = out.x_from ?? Array.from({ length: y.length }, (_, i) => i);
    const label = out.label ?? "Series";
    out.series = { [label]: y }; out.x = x;
    delete out.y_from; delete out.x_from; delete out.label;
  }

  // plot_bar_with_ci ref-form → concrete arrays
  if (toolName === "plot_bar_with_ci" && out.values_from) {
    out.values  = out.values_from; delete out.values_from;
    out.ci_low  = out.ci_low_from; delete out.ci_low_from;
    out.ci_high = out.ci_high_from; delete out.ci_high_from;
  }

  // Strip unknowns (e.g., 'seed' for deterministic tools)
  if (toolName === "power_curve") delete out.seed;

  return out;
}


