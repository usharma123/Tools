"use client";
import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useChatRun } from "@/components/useChatRun";

type Msg = { role: "user" | "assistant"; content: string };

export default function ChatPage() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Msg[]>([
    { role: "assistant" as const, content: "Hi! Tell me what you want to analyze (e.g., power curve, forecast, Markov)." },
  ]);
  const { run, plan, asks, events, logs, final } = useChatRun();
  const [sending, setSending] = useState(false);
  const postedPlanRef = useRef(false);
  const postedAsksCountRef = useRef(0);
  const postedEventsCountRef = useRef(0);
  const postedFinalRef = useRef(false);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const search = useSearchParams();

  useEffect(() => {
    const q = search?.get("q");
    if (q) setInput(q);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-scroll logs to bottom when new logs arrive
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs]);

  const onSend = async () => {
    // basic telemetry for debugging
    try { console.log("[chat] send clicked", { input }); } catch {}
    const text = input.trim();
    if (!text) return;
    const next: Msg[] = [...messages, { role: "user" as const, content: text }];
    setMessages(next);
    setInput("");
    setSending(true);
    await run({ messages: next });
    setSending(false);
  };

  // Stream -> chat bubbles
  useEffect(() => {
    if (plan && !postedPlanRef.current) {
      setMessages((msgs) => [
        ...msgs,
        { role: "assistant", content: "Drafting plan…" } as const,
      ]);
      postedPlanRef.current = true;
    }
  }, [plan]);

  useEffect(() => {
    if (asks && asks.length > postedAsksCountRef.current) {
      const newly = asks.slice(postedAsksCountRef.current);
      postedAsksCountRef.current = asks.length;
      setMessages((msgs) => [
        ...msgs,
        { role: "assistant", content: `I need more info: ${newly.map((a:any)=>a.question).join("; ")}` } as const,
      ]);
    }
  }, [asks]);

  useEffect(() => {
    const n = events?.length ?? 0;
    if (n > postedEventsCountRef.current) {
      const newly = events.slice(postedEventsCountRef.current);
      postedEventsCountRef.current = n;
      const lines = newly.map((e: any) => {
        if (e?.tool) return `${e?.tool} ${e?.cached ? "(cached)" : e?.type || e?.id ? "done" : "running"}`;
        if (e?.type === "error") return `Error: ${e?.payload?.message ?? "see details"}`;
        return `event`;
      }).join("\n");
      setMessages((msgs) => [
        ...msgs,
        { role: "assistant", content: lines } as const,
      ]);
    }
  }, [events]);

  useEffect(() => {
    if (final && !postedFinalRef.current) {
      postedFinalRef.current = true;
      setMessages((msgs) => [
        ...msgs,
        { role: "assistant", content: "Done. See results below." } as const,
      ]);
    }
  }, [final]);

  return (
    <main className="min-h-[80vh] grid place-items-center px-4 sm:px-6 lg:px-8 py-10">
      <div className="w-full max-w-3xl">
        <div className="text-center mb-6">
          <h1 className="text-3xl font-semibold tracking-tight">Chat</h1>
          <p className="text-sm text-gray-600">Conversational runner for tools via /api/chat</p>
        </div>

      <section className="relative rounded-3xl p-[1.5px] bg-gradient-to-br from-violet-300/60 via-fuchsia-300/60 to-rose-300/60 shadow-xl overflow-hidden">
        <div className="rounded-3xl bg-white/70 backdrop-blur">
        {/* Message list */}
        <div className="p-4 md:p-6 space-y-4 min-h-[55vh] bg-gradient-to-b from-white/80 to-white/60">
          <div className="max-h-[55vh] overflow-y-auto pr-1 space-y-3 pb-8">
            {messages.map((m, i) => {
              const isUser = m.role === "user";
              return (
                <div key={i} className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
                  <div className={`max-w-[85%] px-4 py-2 rounded-2xl text-[15px] leading-relaxed shadow-sm ${isUser ? "bg-gradient-to-r from-violet-500 via-fuchsia-500 to-rose-500 text-white" : "bg-white border border-gray-200"}`}>
                    {m.content}
                  </div>
                </div>
              );
            })}

            {sending && !plan && (!events || events.length === 0) && (
              <div className="flex justify-start">
                <div className="px-4 py-2 rounded-2xl bg-white border border-gray-200 text-sm text-gray-600">
                  <span className="inline-flex items-center gap-1">
                    <span className="size-2 rounded-full bg-gray-400 animate-bounce [animation-delay:0ms]"></span>
                    <span className="size-2 rounded-full bg-gray-400 animate-bounce [animation-delay:120ms]"></span>
                    <span className="size-2 rounded-full bg-gray-400 animate-bounce [animation-delay:240ms]"></span>
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Real-time logs */}
          {logs && logs.length > 0 && (
            <div className="rounded-xl border bg-slate-50/80 shadow-sm overflow-hidden">
              <div className="px-4 py-2 bg-slate-100 border-b">
                <div className="text-sm font-medium text-slate-700 flex items-center gap-2">
                  <div className="size-2 rounded-full bg-green-500 animate-pulse"></div>
                  Processing
                </div>
              </div>
              <div className="p-3 max-h-40 overflow-y-auto space-y-1">
                {logs.map((log: any, idx: number) => (
                  <div key={idx} className="flex items-start gap-2 text-xs">
                    <div className="text-slate-400 font-mono whitespace-nowrap">
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </div>
                    <div className={`flex-1 font-mono ${
                      log.error ? 'text-red-600' : 
                      log.message?.startsWith('✅') ? 'text-green-600' :
                      log.message?.startsWith('🔧') ? 'text-blue-600' :
                      log.message?.startsWith('📋') ? 'text-purple-600' :
                      log.message?.startsWith('🎉') ? 'text-emerald-600' :
                      'text-slate-600'
                    }`}>
                      {log.message}
                      {log.duration && (
                        <span className="text-slate-400 ml-2">({log.duration}ms)</span>
                      )}
                    </div>
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            </div>
          )}

          {/* Streamed plan */}
          {plan && (
            <details className="group rounded-xl border bg-white/90 p-3 shadow-sm">
              <summary className="cursor-pointer select-none text-sm text-gray-800 font-medium">
                Analysis Plan
              </summary>
              <div className="mt-2 space-y-2">
                {plan.objective && (
                  <div>
                    <div className="text-xs font-medium text-gray-600">Objective</div>
                    <div className="text-sm text-gray-800">{plan.objective}</div>
                  </div>
                )}
                {plan.steps && plan.steps.length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-gray-600 mb-2">Steps ({plan.steps.length})</div>
                    <div className="space-y-1">
                      {plan.steps.map((step: any, idx: number) => (
                        <div key={idx} className="flex items-center gap-2 text-sm">
                          <div className="size-5 rounded-full bg-slate-100 text-slate-600 text-xs flex items-center justify-center font-medium">
                            {idx + 1}
                          </div>
                          <span className="font-medium">{step.tool}</span>
                          {step.args?.title && (
                            <span className="text-gray-500">- {step.args.title}</span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                <details>
                  <summary className="text-xs text-gray-500 cursor-pointer">View raw JSON</summary>
                  <pre className="mt-2 text-xs bg-gray-50 p-2 rounded border overflow-auto">{JSON.stringify(plan, null, 2)}</pre>
                </details>
              </div>
            </details>
          )}

          {/* If the planner needs more info */}
          {asks && asks.length > 0 && (
            <div className="rounded-xl border bg-amber-50/80 p-3 shadow-sm">
              <div className="text-sm font-medium mb-1">Missing info</div>
              <ul className="list-disc pl-5 space-y-1 text-sm">
                {asks.map((a: any, idx: number) => (
                  <li key={idx}><span className="font-mono text-xs">{a.path}</span>: {a.question}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Tool events */}
          {events && events.length > 0 && (
            <details className="group rounded-xl border bg-white/90 p-3 shadow-sm">
              <summary className="cursor-pointer select-none text-sm text-gray-800 font-medium">Execution</summary>
              <ol className="mt-2 space-y-2">
                {events.map((e: any, idx: number) => (
                  <li key={idx} className="rounded border p-2 text-xs bg-gray-50">
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="font-medium">{e.tool || e.id || 'step'}</span>
                        {typeof e.size === 'number' && (
                          <span className="ml-2 text-gray-500">({Math.round(e.size/1024)} KB)</span>
                        )}
                        {e.cached && <span className="ml-2 text-green-700">cached</span>}
                      </div>
                    </div>
                    {e.artifact && typeof e.artifact === 'string' && (
                      <div className="mt-2">
                        {e.artifact.startsWith('data:image') ? (
                          <img src={e.artifact} alt="artifact" className="w-full h-auto rounded border" />
                        ) : (
                          <iframe src={`http://localhost:8000${e.artifact}`} width="100%" height="360" className="rounded border" />
                        )}
                      </div>
                    )}
                  </li>
                ))}
              </ol>
            </details>
          )}

          {/* Success evaluation */}
          {final?.successEvaluation && (
            <div className={`rounded-xl border p-3 shadow-sm ${
              final.successEvaluation.passed 
                ? 'bg-green-50 border-green-200' 
                : 'bg-red-50 border-red-200'
            }`}>
              <div className="text-sm font-medium mb-2">Success Evaluation</div>
              <div className={`font-mono text-xs mb-2 ${
                final.successEvaluation.passed 
                  ? 'text-green-800' 
                  : 'text-red-800'
              }`}>
                {final.successEvaluation.decision}
              </div>
              <details>
                <summary className="text-xs text-gray-600 cursor-pointer">View details</summary>
                <pre className="mt-2 text-xs bg-white p-2 rounded border overflow-auto">
                  {JSON.stringify(final.successEvaluation.details, null, 2)}
                </pre>
              </details>
            </div>
          )}

          {/* Final results */}
          {final && (
            <details open className="group rounded-xl border bg-white/90 p-3 shadow-sm">
              <summary className="cursor-pointer select-none text-sm text-gray-800 font-medium">Results Summary</summary>
              {final?.summary?.summary ? (
                <p className="mt-1 text-sm text-gray-800 leading-relaxed">{final.summary.summary}</p>
              ) : null}
              
              {/* Show results in a cleaner format */}
              {final.results && final.results.length > 0 && (
                <div className="mt-3 space-y-3">
                  {final.results.map((result: any, idx: number) => (
                    <div key={idx} className="rounded border p-3 bg-gray-50">
                      <div className="text-sm font-medium mb-1 capitalize">
                        {result.tool?.replace(/_/g, ' ')}
                      </div>
                      
                      {/* Show artifacts if available - prioritize base64 image over iframe */}
                      {result.result?.image_base64 ? (
                        <div className="mt-2">
                          <img 
                            src={result.result.image_base64} 
                            alt={`${result.tool} result`} 
                            className="w-full h-auto rounded border" 
                          />
                        </div>
                      ) : result.result?.artifact_url ? (
                        <div className="mt-2">
                          <iframe 
                            src={`http://localhost:8000${result.result.artifact_url}`}
                            width="100%" 
                            height="300" 
                            className="rounded border" 
                            title={`${result.tool} result`}
                          />
                        </div>
                      ) : null}
                      
                      <details className="mt-2">
                        <summary className="text-xs text-gray-600 cursor-pointer">Raw data</summary>
                        <pre className="mt-1 text-xs bg-white p-2 rounded border overflow-auto">
                          {JSON.stringify(result.result, null, 2)}
                        </pre>
                      </details>
                    </div>
                  ))}
                </div>
              )}
            </details>
          )}
        </div>

        {/* Composer */}
        <div className="sticky bottom-0 border-t bg-white/70 backdrop-blur px-3 py-3">
          <div className="flex flex-wrap gap-2 pb-2">
            {["Power curve plan","Forecast next 12 months","Run Markov with T=[[0.9,0.1],[0.2,0.8]]"].map((s) => (
              <button key={s} onClick={()=>setInput(s)} className="text-xs rounded-full border px-2.5 py-1 bg-white/80 hover:bg-white transition">{s}</button>
            ))}
          </div>
          <div className="relative flex items-center gap-2 rounded-full bg-white shadow ring-1 ring-black/5 px-3 h-12">
            <textarea
              rows={1}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSend(); }
                if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); onSend(); }
              }}
              placeholder="Ask e.g. 'Run power curve…' or 'Forecast next 6 months…'"
              className="min-h-[1.75rem] w-full resize-none bg-transparent outline-none text-[15px] placeholder-zinc-500"
            />
            <button
              onClick={onSend}
              type="button"
              disabled={sending}
              className="shrink-0 inline-flex items-center justify-center h-9 px-4 rounded-full bg-gradient-to-r from-violet-500 via-fuchsia-500 to-rose-500 text-white shadow-sm disabled:opacity-50"
            >
              {sending ? "Sending…" : "Send"}
            </button>
          </div>
        </div>
        </div>
      </section>
      </div>
    </main>
  );
}


