"use client";
import { useState } from "react";

export default function Home() {
  const [q, setQ] = useState(
    "Estimate the stationary distribution for T=[[0.9,0.1],[0.2,0.8]] with 1e4 trials and 1000 steps, then plot the cumulative visits per state."
  );
  const [out, setOut] = useState<{ 
    text?: string; 
    toolResults?: Array<{tool: string; params: unknown; result: Record<string, unknown>}>; 
    originalText?: string;
    successEvaluation?: { passed: boolean; details: Record<string, unknown>; decision: string };
    analysisPlan?: Record<string, any>;
  } | null>(null);
  const [loading, setLoading] = useState(false);

  // Render markdown-like paragraphs while preserving possible data URLs
  const renderText = (text: string) => {
    return (
      <div className="space-y-3 text-gray-800 leading-relaxed">
        {text.split(/\n{2,}/).map((block, idx) => (
          <p key={idx} dangerouslySetInnerHTML={{ __html: block.replace(/\n/g, '<br/>') }} />
        ))}
      </div>
    );
  };

  // Pretty rendering for the generated AnalysisPlan (instead of raw JSON)
  const renderPlan = () => {
    if (!out?.analysisPlan) return null;
    const plan = out.analysisPlan as Record<string, any>;
    const steps: Array<any> = plan.steps ?? [];
    const sc: Record<string, any> | undefined = plan.success_criteria;

    const thresholdItems: Array<{k: string; v: number}> = [];
    if (sc) {
      const keys = [
        "tv_distance_max",
        "ci_width_max",
        "min_trials",
        "max_std_dev",
        "convergence_threshold",
      ];
      keys.forEach(k => {
        if (typeof sc[k] === 'number') thresholdItems.push({ k, v: sc[k] });
      });
    }

    const niceKey = (k: string) =>
      k.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());

    const summarizeArgs = (tool: string, args: any) => {
      try {
        if (tool === 'markov_mcs') {
          return `steps=${args.steps ?? '-'}, trials=${args.trials ?? '-'}, metric=${args.metric ?? 'stationary'}, seed=${args.seed ?? '-'}`;
        }
        if (tool === 'plot_line') {
          return args.title ? `${args.title}` : 'Line chart';
        }
        if (tool === 'plot_bar') {
          return args.title ? `${args.title}` : 'Bar chart';
        }
        if (tool === 'power_curve') {
          return `mode=${args.mode}, baseline=${args.baseline}`;
        }
      } catch {}
      return '';
    };

    return (
      <section className="rounded-xl border bg-white shadow-sm overflow-hidden">
        <div className="px-5 py-4 border-b bg-gray-50">
          <h3 className="text-xl font-semibold">Plan</h3>
          {plan.objective && (
            <p className="text-gray-700 mt-1">{plan.objective}</p>
          )}
        </div>
        <div className="p-5 space-y-6">
          {Array.isArray(plan.assumptions) && plan.assumptions.length > 0 && (
            <div>
              <div className="text-sm font-medium text-gray-600 mb-2">Assumptions</div>
              <ul className="list-disc pl-5 space-y-1 text-gray-800">
                {plan.assumptions.map((a: string, i: number) => (
                  <li key={i}>{a}</li>
                ))}
              </ul>
            </div>
          )}

          {steps.length > 0 && (
            <div>
              <div className="text-sm font-medium text-gray-600 mb-3">Steps</div>
              <ol className="space-y-3">
                {steps.map((s: any, i: number) => (
                  <li key={i} className="flex items-start gap-3">
                    <div className="mt-1 h-6 w-6 shrink-0 rounded-full bg-black text-white text-xs grid place-items-center">{i+1}</div>
                    <div className="flex-1 rounded-lg border p-3">
                      <div className="flex items-center justify-between">
                        <div className="font-medium">{s.tool}</div>
                        {s.args?.title && (
                          <div className="text-sm text-gray-600">{s.args.title}</div>
                        )}
                      </div>
                      <div className="text-sm text-gray-700 mt-1">{summarizeArgs(s.tool, s.args)}</div>
                    </div>
                  </li>
                ))}
              </ol>
            </div>
          )}

          {sc && (
            <div className="space-y-2">
              <div className="text-sm font-medium text-gray-600">Success Criteria</div>
              {sc.description && (
                <p className="text-gray-800">{sc.description}</p>
              )}
              {thresholdItems.length > 0 && (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {thresholdItems.map(({k,v}) => (
                    <div key={k} className="rounded-md border p-3">
                      <div className="text-xs uppercase tracking-wide text-gray-500">{niceKey(k)}</div>
                      <div className="text-lg font-semibold">{v}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          <details>
            <summary className="text-sm text-gray-600 cursor-pointer">View plan JSON</summary>
            <pre className="text-xs bg-gray-50 p-3 rounded-md overflow-auto border mt-2">{JSON.stringify(plan, null, 2)}</pre>
          </details>
        </div>
      </section>
    );
  };

  // Function to render success evaluation decision block
  const renderSuccessEvaluation = () => {
    if (!out?.successEvaluation) return null;
    
    const { passed, details, decision } = out.successEvaluation;
    
    return (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">Success Evaluation:</h3>
        <div className={`border rounded p-4 ${passed ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
          <div className="font-mono text-sm">
            <div className={`font-bold ${passed ? 'text-green-800' : 'text-red-800'}`}>
              {decision}
            </div>
            <div className="mt-2 text-gray-700">
              <h4 className="font-medium mb-1">Details:</h4>
              <pre className="text-xs bg-white p-2 rounded overflow-auto">
                {JSON.stringify(details, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Function to render tool results with artifacts (HTML iframe or base64 img)
  const renderToolResults = () => {
    if (!out?.toolResults) return null;
    
    return (
      <div className="space-y-6">
        <h3 className="text-xl font-semibold">Results</h3>
        {out.toolResults.map((result, index: number) => (
          <div key={index} className="rounded-xl border bg-white shadow-sm">
            <div className="px-4 py-3 border-b bg-gray-50 rounded-t-xl">
              <h4 className="font-medium capitalize">{result.tool.replaceAll('_',' ')}</h4>
            </div>
            <div className="p-4 space-y-4">
              {/* If this is the summarizer, show a prose paragraph */}
              {result.tool === 'summarize_results' ? (
                <p className="text-gray-800 leading-relaxed">{(result.result as any)?.summary}</p>
              ) : (
                <>
              {(() => {
                const resultObj = result.result as Record<string, unknown>;
                const b64 = resultObj?.image_base64 as string | undefined;
                const url = resultObj?.artifact_url as string | undefined;
                if (b64 && typeof b64 === 'string') {
                  return (
                    <div className="w-full overflow-hidden rounded-md border">
                      <img src={b64} alt="chart" className="w-full h-auto" />
                    </div>
                  );
                }
                if (url && typeof url === 'string') {
                  // If server returned a data URL, render as <img>; otherwise use iframe
                  if (url.startsWith('data:image')) {
                    return (
                      <div className="w-full overflow-hidden rounded-md border">
                        <img src={url} alt="chart" className="w-full h-auto" />
                      </div>
                    );
                  }
                  return (
                    <iframe 
                      src={`http://localhost:8000${url}`}
                      width="100%"
                      height="420"
                      frameBorder="0"
                      className="rounded-md border"
                      title="Interactive Chart"
                    />
                  );
                }
                return null;
              })()}
              </>
              )}
              <details className="mt-2">
                <summary className="text-sm text-gray-600 cursor-pointer">Raw JSON</summary>
                <pre className="text-xs bg-gray-50 p-3 rounded-md overflow-auto border mt-2">
                  {JSON.stringify(result.result, null, 2)}
                </pre>
              </details>
            </div>
            {(() => {
              const resultObj = result.result as Record<string, unknown>;
              // Deprecated block retained for backward compatibility
              return null;
            })()}
          </div>
        ))}
      </div>
    );
  };

  async function run() {
    setLoading(true);
    const res = await fetch("/api/solve", {
      method: "POST",
      headers: {"content-type":"application/json"},
      body: JSON.stringify({ query: q })
    });
    const data = await res.json();
    setOut(data);
    setLoading(false);
  }

  return (
    <main className="relative px-4 sm:px-6 lg:px-8 py-8 md:py-12 max-w-7xl mx-auto space-y-8">
      <header className="sticky top-0 z-10 -mx-4 md:-mx-6 px-4 md:px-6 py-3 backdrop-blur supports-[backdrop-filter]:bg-white/40 dark:supports-[backdrop-filter]:bg-black/20 rounded-b-xl border-b border-black/5">
        <div className="mx-auto max-w-5xl flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div>
              <h1 className="text-lg md:text-xl font-semibold tracking-tight">Benched</h1>
              <p className="hidden sm:block text-xs text-zinc-600 dark:text-zinc-400">Clean, modern agentic solver</p>
            </div>
          </div>
          <div className="hidden sm:flex items-center gap-3 text-xs text-zinc-600 dark:text-zinc-400" />
        </div>
      </header>

      <section className="gradient-border rounded-2xl">
        <div className="glass rounded-[calc(1rem-2px)] p-4 md:p-6">
          <label htmlFor="prompt" className="sr-only">Prompt</label>
          <textarea
            id="prompt"
            className="w-full border rounded-xl p-3 md:p-4 bg-white text-zinc-900 dark:bg-zinc-900/70 dark:text-zinc-100 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-emerald-400/40 focus:border-emerald-300/40 text-[16px]"
            rows={4}
            value={q}
            onChange={e=>setQ(e.target.value)}
            placeholder="Describe a task, e.g., 'Forecast next month's demand with a 95% CI'"
            onKeyDown={(e)=>{ if((e.metaKey||e.ctrlKey) && e.key==='Enter'){ run(); } }}
          />

          <div className="mt-3 flex flex-wrap items-center gap-2">
            <button type="button" onClick={()=>setQ('Compute power curve for a two-arm AB test with baseline=0.1, effect=0.02, n in [1000,5000].')}
              className="text-xs rounded-full border px-2.5 py-1 bg-white/70 hover:bg-white/90 transition">Example: Power curve</button>
            <button type="button" onClick={()=>setQ('Forecast next 12 months with seasonality, return 95% CI and a line plot.')}
              className="text-xs rounded-full border px-2.5 py-1 bg-white/70 hover:bg-white/90 transition">Example: Forecast</button>
            <button type="button" onClick={()=>setQ('Run a Markov chain simulation with 1e4 trials, 1000 steps for T=[[0.9,0.1],[0.2,0.8]].')}
              className="text-xs rounded-full border px-2.5 py-1 bg-white/70 hover:bg-white/90 transition">Example: Markov</button>
          </div>

          <div className="mt-4 flex items-center justify-between">
            <div className="hidden sm:flex gap-2 text-xs text-zinc-600 dark:text-zinc-400">
              <span className="rounded-md border px-2 py-1 bg-white/80 text-emerald-700 border-emerald-200">Charts</span>
              <span className="rounded-md border px-2 py-1 bg-white/80 text-cyan-700 border-cyan-200">Artifacts</span>
              <span className="rounded-md border px-2 py-1 bg-white/80 text-fuchsia-700 border-fuchsia-200">Success Checks</span>
            </div>
            <button
              onClick={run}
              disabled={loading}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-500 via-teal-500 to-cyan-500 text-white shadow-sm shadow-emerald-500/20 hover:from-emerald-600 hover:via-teal-600 hover:to-cyan-600 disabled:opacity-50"
            >
              {loading ? "Solving..." : "Run"}
            </button>
          </div>
        </div>
      </section>

      {renderPlan()}

      {renderSuccessEvaluation()}

      <div className="relative">
        <div className="absolute -inset-x-8 -top-10 -z-10 h-24 bg-gradient-to-r from-emerald-400/40 via-cyan-400/30 to-fuchsia-400/40 blur-2xl" />
        {renderToolResults()}
      </div>
    </main>
  );
}

