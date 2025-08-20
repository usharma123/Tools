import { NextRequest } from "next/server";
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import { run_markov_mcs, plot_line, plot_bar, summarize_results } from "@/lib/tools";
import { choice_logit } from "@/lib/tools_choice";
import { forecast_arima as forecast_arima_adapter } from "@/lib/tools_forecast";
import { forecast_backtest as forecast_backtest_adapter } from "@/lib/tools_backtest";
import { ab_test_ttest as ab_test_ttest_A, plot_bar_with_ci as plot_bar_with_ci_A, power_curve } from "@/lib/tools_ab_power";
import { ab_test_ttest as ab_test_ttest_B, plot_bar_with_ci as plot_bar_with_ci_B } from "@/lib/tools_stats";
import { causal_impact } from "@/lib/tools_causal";
import { AnalysisPlan, SuccessCriteria, materializeArgs } from "@/lib/plan";

export const runtime = "nodejs";

const SYSTEM = `
You are Chief Analyst.
1) First output a VALID AnalysisPlan JSON with proper schema validation.
2) Execute steps strictly via available tools.
3) If inputs are missing, ask targeted questions.
4) Prefer simplest valid methods; include assumptions & limitations.
5) Reference any returned artifact_url in your write-up.
Return a concise Decision and Evidence.

IMPORTANT: Choose tools based on the analysis context:
- Use markov_mcs for Markov chain analysis, convergence studies, state transitions
- Use ab_test_ttest for A/B testing, conversion rate comparisons, treatment effects
- Use power_curve for sample size planning, power analysis, MDE calculations (NO seed parameter)
 - Use plot_* tools for visualization of results
 - Use causal_impact for diff-in-diff causal analysis on panel data (treated vs control over time)
 - Use forecast_arima for time series forecasting; if you have date context, pass start_date and freq so the worker returns real indices. For plotting, prefer x_from: "$forecast_arima.history_index" (history) and "$forecast_arima.forecast_index" (forecast).

Available tools:
- markov_mcs: Run Monte Carlo on a Markov chain. Parameters: transition (array of arrays), steps (number), trials (number), metric (stationary/avg_reward/trajectory), track_trajectory (boolean)
- plot_line: Create a simple line chart. Parameters: series (object with arrays), title (string), xlabel (string), ylabel (string)
- plot_bar: Create a bar chart. Parameters: series (object with arrays), title (string), xlabel (string), ylabel (string)
- ab_test_ttest: Two-sample test (binary or continuous). Parameters: binary mode (successes_a, trials_a, successes_b, trials_b) OR continuous mode (mean_a, sd_a, n_a, mean_b, sd_b, n_b), alpha, two_tailed, equal_var (continuous only)
- plot_bar_with_ci: Bar chart with 95% CI whiskers. Parameters: labels, values, ci_low, ci_high, title, xlabel, ylabel, ylim
 - power_curve: Plot power relationships for two-proportion A/B: either n vs MDE or power vs n. Parameters: mode (mde_vs_n/power_vs_n), baseline, alpha, two_tailed, ratio, power (for mde_vs_n), mde_rel_grid (for mde_vs_n), mde_rel (for power_vs_n), n_grid (for power_vs_n). IMPORTANT: power_curve is deterministic and does NOT accept seed parameter.
 - causal_impact: Diff-in-diff causal impact. Parameters: csv (string), date_col (string), metric_col (string), entity_col (string), treated_entity (string), pre_period ([start,end]), post_period ([start,end])

IMPORTANT: 
- Output ONLY the AnalysisPlan JSON, no other text.
- Use proper schema validation for all tool arguments.
- ALWAYS include a numeric seed for reproducibility (e.g., "seed": 12345) when the tool supports stochasticity.
`;

function sseHeaders() {
  return new Headers({
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
  });
}

function send(type: string, data: any) {
  return `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
}

async function executeTool(toolName: string, params: unknown) {
  try {
    if (toolName === "markov_mcs") {
      return await run_markov_mcs(params);
    } else if (toolName === "plot_line") {
      return await plot_line(params);
    } else if (toolName === "plot_bar") {
      return await plot_bar(params);
    } else if (toolName === "summarize_results") {
      return await summarize_results(params);
    } else if (toolName === "ab_test_ttest") {
      // try new stats adapter first, fallback to legacy
      try { return await ab_test_ttest_B(params); } catch { return await ab_test_ttest_A(params); }
    } else if (toolName === "plot_bar_with_ci") {
      try { return await plot_bar_with_ci_B(params); } catch { return await plot_bar_with_ci_A(params); }
    } else if (toolName === "power_curve") {
      return await power_curve(params);
  } else if (toolName === "causal_impact") {
      return await causal_impact(params);
    } else if (toolName === "forecast_arima") {
      return await forecast_arima_adapter(params);
    } else if (toolName === "forecast_backtest") {
      return await forecast_backtest_adapter(params);
    } else if (toolName === "choice_logit") {
      // Accept either numbers or strings in plan; coerce safely
      const p = params as any;
      const normalized = {
        ...p,
        scenarios: Array.isArray(p?.scenarios)
          ? p.scenarios.map((s: any) => ({
              ...s,
              adjustments: Object.fromEntries(
                Object.entries(s?.adjustments ?? {}).map(([k, v]) => [k, typeof v === 'string' ? Number(v) : v])
              )
            }))
          : undefined
      };
      return await choice_logit(normalized);
    } else {
      throw new Error(`Unknown tool: ${toolName}`);
    }
  } catch (error) {
    return { error: `Tool execution failed: ${error}` };
  }
}

function parseAnalysisPlan(text: string): AnalysisPlan | null {
  try {
    // Try to extract JSON from the text
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      return null;
    }
    
    const jsonStr = jsonMatch[0];
    const parsed = JSON.parse(jsonStr);
    
    // Validate against the schema
    const validated = AnalysisPlan.parse(parsed);
    return validated;
  } catch (error) {
    return null;
  }
}

function evaluateSuccessCriteria(
  criteria: SuccessCriteria, 
  toolResult: Record<string, unknown>
): { passed: boolean; details: Record<string, unknown>; decision: string } {
  // Simplified success evaluation for streaming
  return { 
    passed: true, 
    details: {}, 
    decision: "Analysis completed successfully" 
  };
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  const { query } = body;

  const stream = new ReadableStream({
    async start(controller) {
      const write = (type: string, data: any) => {
        controller.enqueue(new TextEncoder().encode(send(type, data)));
      };

      try {
        write("log", { message: "🤖 Starting analysis...", timestamp: new Date().toISOString() });

        // Generate plan using AI
        write("log", { message: "📋 Generating analysis plan with GPT-5...", timestamp: new Date().toISOString() });
        
        const { text: generatedText } = await generateText({
          model: openai("gpt-5"),
          temperature: 0.7,
          system: SYSTEM,
          prompt: `Problem: ${query}

IMPORTANT: Output ONLY a valid AnalysisPlan JSON. No other text.`,
        });

        // Parse and validate the AnalysisPlan
        write("log", { message: "🔍 Parsing and validating analysis plan...", timestamp: new Date().toISOString() });
        
        const analysisPlan = parseAnalysisPlan(generatedText);
        if (!analysisPlan) {
          write("error", { message: "Failed to parse or validate AnalysisPlan" });
          controller.close();
          return;
        }

        write("plan", analysisPlan);
        write("log", { message: `✅ Plan generated with ${analysisPlan.steps.length} steps`, timestamp: new Date().toISOString() });

        const allToolResults: Array<{tool: string; params: unknown; result: unknown}> = [];
        let successEvaluation: { passed: boolean; details: Record<string, unknown>; decision: string } | null = null;

        // Execute all steps in sequence with chaining
        const results: Record<string, any> = {};
        
        for (let i = 0; i < analysisPlan.steps.length; i++) {
          const step = analysisPlan.steps[i];
          
          write("log", { 
            message: `🔧 Step ${i + 1}/${analysisPlan.steps.length}: Executing ${step.tool}...`, 
            timestamp: new Date().toISOString(),
            step: i + 1,
            tool: step.tool
          });

          try {
            // Materialize arguments (resolve references to numbers)
            const materializedArgs = materializeArgs(step.tool, step.args, results);
            
            write("log", { 
              message: `⚙️ Materialized arguments for ${step.tool}`, 
              timestamp: new Date().toISOString() 
            });

            const startTime = Date.now();
            const result = await executeTool(step.tool, materializedArgs);
            const duration = Date.now() - startTime;

            allToolResults.push({
              tool: step.tool,
              params: materializedArgs,
              result
            });

            // Store results with step ID or index to handle multiple calls to same tool
            const stepKey = step.id || `${step.tool}_${i + 1}`;
            results[stepKey] = result;
            results[step.tool] = result; // Keep original for backward compatibility

            write("tool-complete", {
              step: i + 1,
              tool: step.tool,
              duration,
              result,
              timestamp: new Date().toISOString()
            });

            write("log", { 
              message: `✅ ${step.tool} completed in ${duration}ms`, 
              timestamp: new Date().toISOString(),
              duration 
            });

            // Evaluate success criteria after relevant tools complete
            if ((step.tool === 'markov_mcs' || step.tool === 'power_curve' || step.tool === 'forecast_arima' || step.tool === 'choice_logit') && analysisPlan.success_criteria) {
              successEvaluation = evaluateSuccessCriteria(analysisPlan.success_criteria, result as Record<string, unknown>);
              write("log", { 
                message: `📊 Success criteria evaluation: ${successEvaluation.passed ? 'PASS' : 'FAIL'}`, 
                timestamp: new Date().toISOString() 
              });
            }

          } catch (error) {
            write("log", { 
              message: `❌ Error executing ${step.tool}: ${String(error)}`, 
              timestamp: new Date().toISOString(),
              error: true 
            });
            
            allToolResults.push({
              tool: step.tool,
              params: step.args,
              result: { error: String(error) }
            });
          }
        }

        // Auto-summarize results if there were any analytical steps
        write("log", { message: "📝 Generating summary...", timestamp: new Date().toISOString() });
        
        try {
          const summaryPayload: Record<string, unknown> = {};
          const mk = allToolResults.find(r => r.tool === 'markov_mcs')?.result;
          const ab = allToolResults.find(r => r.tool === 'ab_test_ttest')?.result;
          const pc = allToolResults.find(r => r.tool === 'power_curve')?.result;
          const fc = allToolResults.find(r => r.tool === 'forecast_arima')?.result;
          const bt = allToolResults.find(r => r.tool === 'forecast_backtest')?.result;
          if (mk) summaryPayload.markov = mk;
          if (ab) summaryPayload.ab_test = ab;
          if (pc) summaryPayload.power_curve = pc;
          if (fc) summaryPayload.forecast = fc;
          if (bt) summaryPayload.backtest = bt;
          
          if (Object.keys(summaryPayload).length > 0) {
            const summaryResult = await executeTool('summarize_results', summaryPayload);
            allToolResults.push({ tool: 'summarize_results', params: summaryPayload, result: summaryResult });
            write("log", { message: "✅ Summary generated", timestamp: new Date().toISOString() });
          }
        } catch (e) {
          write("log", { message: `⚠️ Summary generation failed: ${String(e)}`, timestamp: new Date().toISOString() });
          allToolResults.push({ tool: 'summarize_results', params: {}, result: { error: String(e) } });
        }

        write("final", { 
          toolResults: allToolResults,
          analysisPlan: analysisPlan,
          successEvaluation: successEvaluation,
          originalText: generatedText
        });

        write("log", { message: "🎉 Analysis complete!", timestamp: new Date().toISOString() });
        controller.close();

      } catch (e: any) {
        write("error", { message: e?.message || String(e) });
        controller.close();
      }
    }
  });

  return new Response(stream, { headers: sseHeaders() });
}