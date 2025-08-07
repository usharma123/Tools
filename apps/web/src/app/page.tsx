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
    <main className="p-6 max-w-4xl mx-auto space-y-6">
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">Agentic Solver</h1>
        <p className="text-sm text-gray-600">Describe a task. Results will render as charts with clean visuals.</p>
      </div>
      <div className="rounded-xl border bg-white shadow-sm overflow-hidden">
        <div className="p-4">
          <textarea className="w-full border rounded-md p-3 focus:outline-none focus:ring-2 focus:ring-black/20" rows={4} value={q} onChange={e=>setQ(e.target.value)} />
          <div className="mt-3 flex justify-end">
            <button onClick={run} disabled={loading} className="px-4 py-2 rounded-md bg-black text-white disabled:opacity-50">
              {loading ? "Solving..." : "Run"}
            </button>
          </div>
        </div>
      </div>

      {renderPlan()}

      {renderSuccessEvaluation()}

      {renderToolResults()}
    </main>
  );
}

