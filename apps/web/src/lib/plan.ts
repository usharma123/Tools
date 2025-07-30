import { z } from "zod";

export const Step = z.object({
  tool: z.enum(["markov_mcs", "plot_line"]),
  args: z.record(z.string(), z.any())
});

export const AnalysisPlan = z.object({
  objective: z.string(),
  assumptions: z.array(z.string()).optional(),
  steps: z.array(Step).min(1),
  report_outline: z.array(z.string()).default([
    "Decision","Evidence","Assumptions","Limitations","Next steps"
  ])
});

export type AnalysisPlan = z.infer<typeof AnalysisPlan>;
