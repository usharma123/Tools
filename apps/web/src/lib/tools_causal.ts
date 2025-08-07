import { z } from "zod";

const WORKER = process.env.WORKER_URL || "http://localhost:8000";

export const didParams = z.object({
  csv: z.string(),             // raw CSV or resolved path
  date_col: z.string().optional().default("date"),
  metric_col: z.string().optional().default("metric"),
  entity_col: z.string().optional().default("entity"),
  treated_entity: z.string(),
  pre_period: z.tuple([z.string(), z.string()]),
  post_period: z.tuple([z.string(), z.string()])
});

export async function causal_impact(args: unknown){
  const body = JSON.stringify(didParams.parse(args));
  const res  = await fetch(`${WORKER}/tools/causal_impact`,{
    method:"POST", headers:{"content-type":"application/json"}, body});
  if(!res.ok) throw new Error("causal_impact failed");
  return res.json();
}


