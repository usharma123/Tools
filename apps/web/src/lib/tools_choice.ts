import { z } from "zod";

const WORKER = process.env.WORKER_URL || "http://localhost:8000";

const adjustment = z.union([
  z.number(),
  z.object({ mode: z.enum(["add","mul"]).default("add"), value: z.number() })
]);

const scenario = z.object({
  name: z.string(),
  scope_alts: z.array(z.string()).optional(),
  adjustments: z.record(z.string(), adjustment)   // feature -> additive or multiplicative
});

export const choiceLogitParams = z.object({
  csv: z.string(),
  choice_col: z.string().default("choice_id"),
  alt_col: z.string().default("alt_id"),
  chosen_col: z.string().default("chosen"),
  feature_cols: z.array(z.string()).min(1),
  standardize: z.boolean().default(false),
  add_alt_dummies: z.boolean().default(false),
  base_alt: z.string().optional(),
  l2_lambda: z.number().default(1e-2),
  scenarios: z.array(scenario).optional()
});

// Relaxed plan schema: allow numbers OR strings for adjustments; coercion happens in adapter
export const choiceLogitPlanParams = z.object({
  csv: z.string(),
  choice_col: z.string().default("choice_id"),
  alt_col: z.string().default("alt_id"),
  chosen_col: z.string().default("chosen"),
  feature_cols: z.array(z.string()).min(1),
  standardize: z.boolean().default(false),
  add_alt_dummies: z.boolean().default(false),
  base_alt: z.string().optional(),
  l2_lambda: z.number().default(1e-2),
  scenarios: z.array(z.object({
    name: z.string(),
    scope_alts: z.array(z.string()).optional(),
    adjustments: z.record(z.string(), z.union([
      z.number(),
      z.string(),
      z.object({ mode: z.enum(["add","mul"]).default("add"), value: z.number() })
    ]))
  })).optional()
});

export async function choice_logit(args: unknown){
  const body = JSON.stringify(choiceLogitParams.parse(args));
  const res  = await fetch(`${WORKER}/tools/choice_logit`, {
    method:"POST", headers:{"content-type":"application/json"}, body
  });
  if(!res.ok) throw new Error(`choice_logit failed: ${res.status}`);
  return res.json();
}


