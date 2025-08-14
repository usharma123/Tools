import { z } from "zod";
import { abTestParams, ab_test_ttest } from "@/lib/tools_stats";
import { barCIParams, plot_bar_with_ci } from "@/lib/tools_stats";
import { arimaParams, forecast_arima } from "@/lib/tools_forecast";
import { backtestParams, forecast_backtest } from "@/lib/tools_backtest";
import { powerCurveParams, power_curve } from "@/lib/tools_ab_power";
import { didParams, causal_impact } from "@/lib/tools_causal";
import { run_markov_mcs, plot_line, plot_bar, summarize_results } from "@/lib/tools";
import { markovMcsParams, plotLineParams, plotBarParams, summarizeParams } from "@/lib/tools";
import { choiceLogitParams, choice_logit } from "@/lib/tools_choice";

export type ToolSpec = {
  name: string;
  description: string;
  version: string;
  params: z.ZodTypeAny;
  execute: (args: any) => Promise<any>;
};

// keep versions here so cache invalidates on updates
export const tools: Record<string, ToolSpec> = {
  markov_mcs: {
    name: "markov_mcs",
    description: "Monte Carlo simulation for Markov chains",
    version: "0.1.0",
    params: markovMcsParams,
    execute: run_markov_mcs,
  },
  ab_test_ttest: {
    name: "ab_test_ttest",
    description: "A/B test for binary or continuous outcomes",
    version: "0.1.0",
    params: abTestParams,
    execute: ab_test_ttest,
  },
  plot_line: {
    name: "plot_line",
    description: "Simple line plot renderer",
    version: "0.1.0",
    params: plotLineParams,
    execute: plot_line,
  },
  plot_bar: {
    name: "plot_bar",
    description: "Simple bar plot renderer",
    version: "0.1.0",
    params: plotBarParams,
    execute: plot_bar,
  },
  plot_bar_with_ci: {
    name: "plot_bar_with_ci",
    description: "Bar plot with 95% CI whiskers",
    version: "0.1.0",
    params: barCIParams,
    execute: plot_bar_with_ci,
  },
  power_curve: {
    name: "power_curve",
    description: "Power curves for two-proportion A/B: MDE vs n or power vs n",
    version: "0.1.0",
    params: powerCurveParams,
    execute: power_curve,
  },
  forecast_arima: {
    name: "forecast_arima",
    description: "ARIMA forecast with confidence bands",
    version: "0.1.0",
    params: arimaParams,
    execute: forecast_arima,
  },
  forecast_backtest: {
    name: "forecast_backtest",
    description: "Rolling-origin ARIMA backtest vs naïve",
    version: "0.1.0",
    params: backtestParams,
    execute: forecast_backtest,
  },
  causal_impact: {
    name: "causal_impact",
    description: "Diff-in-diff causal impact with clustered SEs",
    version: "0.1.0",
    params: didParams,
    execute: causal_impact,
  },
  summarize_results: {
    name: "summarize_results",
    description: "Summarize analytical results into prose",
    version: "0.1.0",
    params: summarizeParams,
    execute: summarize_results,
  },
  choice_logit: {
    name: "choice_logit",
    description: "Conditional logit with scenario simulations",
    version: "0.2.0",
    params: choiceLogitParams,
    execute: choice_logit,
  },
};


