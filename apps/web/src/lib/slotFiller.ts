import { z } from "zod";
import type { ChatAsk } from "./types";

export function missingFields(schema: z.ZodTypeAny, partial: any): ChatAsk[] {
  try { schema.parse(partial); return []; }
  catch (e: any) {
    const asks: ChatAsk[] = [];
    for (const issue of e.issues ?? []) {
      const path = issue.path.join(".");
      if (!path) continue;
      const p = path.toLowerCase();
      let question = `Please provide ${path}`;
      if (p.endsWith("baseline")) question = "What’s the baseline rate? (e.g., 0.05)";
      if (p.endsWith("mde_rel"))  question = "Smallest relative lift to detect? (e.g., 0.10)";
      if (p.endsWith("horizon"))  question = "How many periods should we forecast?";
      if (p.includes("treated_entity")) question = "Which entity is treated?";
      if (p.includes("pre_period")) question = "What is the pre period? (e.g., [YYYY-MM-DD,YYYY-MM-DD])";
      if (p.includes("post_period")) question = "What is the post period? (e.g., [YYYY-MM-DD,YYYY-MM-DD])";
      asks.push({ path, question });
    }
    return asks;
  }
}


