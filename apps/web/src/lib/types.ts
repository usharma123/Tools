export type ChatAsk = { path: string; question: string };
export type ChatPlanStep = { id: string; tool: string; args: any };
export type ChatPlan = { steps: ChatPlanStep[]; ask?: ChatAsk[] };


