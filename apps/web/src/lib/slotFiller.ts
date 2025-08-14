import { z } from "zod";
import type { ChatAsk } from "./types";

export function missingFields(schema: z.ZodTypeAny, partial: any): ChatAsk[] {
  try { schema.parse(partial); return []; }
  catch (e: any) {
    // Flatten issues, including nested union errors, to collect concrete field paths
    const paths: Set<string> = new Set();
    const queue: any[] = Array.isArray(e?.issues) ? [...e.issues] : [];
    while (queue.length) {
      const issue = queue.shift();
      if (!issue) continue;
      // Zod's invalid_union includes unionErrors with nested issues
      if (issue.code === 'invalid_union' && Array.isArray(issue.unionErrors)) {
        for (const ue of issue.unionErrors) {
          if (Array.isArray(ue.issues)) queue.push(...ue.issues);
        }
        continue;
      }
      const pathArr: any[] = Array.isArray(issue.path) ? issue.path : [];
      const path = pathArr.join(".");
      if (path) paths.add(path);
    }

    // If no concrete paths were found (rare), return a generic ask
    if (paths.size === 0) return [{ path: "args", question: "Please provide required arguments for this tool." }];

    const asks: ChatAsk[] = [];
    const makeQuestion = (path: string): string => {
      const p = path.toLowerCase();
      // Power curve
      if (p.endsWith('baseline')) return "What’s the baseline rate? (e.g., 0.05)";
      if (p.endsWith('mde_rel')) return "Smallest relative lift to detect (relative)? (e.g., 0.10)";
      if (p.endsWith('mde_rel_grid')) return "Provide a grid of relative MDE values (e.g., [0.02,0.05,0.10])";
      if (p.endsWith('n_grid')) return "Provide the per-arm sample sizes (e.g., [1000,2000,5000])";
      if (p.endsWith('power')) return "Target power? (e.g., 0.8)";

      // Markov
      if (p.endsWith('transition')) return "Provide the transition matrix (e.g., [[0.9,0.1],[0.2,0.8]])";
      if (p.endsWith('steps')) return "How many steps? (e.g., 1000)";
      if (p.endsWith('trials')) return "How many trials? (e.g., 10000)";

      // Forecast
      if (p.endsWith('ts')) return "Provide the time series (array) or attach as an asset.";
      if (p.endsWith('horizon')) return "How many periods should we forecast?";

      // Causal impact
      if (p.endsWith('csv')) return "Provide a CSV file/asset with the panel data (attach and reference as $asset:table.csv).";
      if (p.endsWith('date_col')) return "Which column contains dates?";
      if (p.endsWith('metric_col')) return "Which column contains the metric?";
      if (p.endsWith('entity_col')) return "Which column identifies entities?";
      if (p.includes('treated_entity')) return "Which entity is treated?";
      if (p.includes('pre_period')) return "What is the pre period? (e.g., [YYYY-MM-DD,YYYY-MM-DD])";
      if (p.includes('post_period')) return "What is the post period? (e.g., [YYYY-MM-DD,YYYY-MM-DD])";

      // A/B test (binary)
      if (p.endsWith('successes_a')) return "How many successes for group A?";
      if (p.endsWith('trials_a')) return "How many trials for group A?";
      if (p.endsWith('successes_b')) return "How many successes for group B?";
      if (p.endsWith('trials_b')) return "How many trials for group B?";
      // A/B test (continuous)
      if (p.endsWith('mean_a')) return "What is mean for group A?";
      if (p.endsWith('sd_a')) return "What is standard deviation for group A?";
      if (p.endsWith('n_a')) return "What is sample size for group A?";
      if (p.endsWith('mean_b')) return "What is mean for group B?";
      if (p.endsWith('sd_b')) return "What is standard deviation for group B?";
      if (p.endsWith('n_b')) return "What is sample size for group B?";

      // Default
      return `Please provide ${path}`;
    };

    for (const path of Array.from(paths)) asks.push({ path, question: makeQuestion(path) });
    return asks;
  }
}


