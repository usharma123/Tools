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
  } | null>(null);
  const [loading, setLoading] = useState(false);

  // Function to process text and replace artifact_url with iframe
  const processText = (text: string) => {
    // Replace artifact_url with iframe for HTML files
    const processedText = text.replace(
      /artifact_url/g, 
      '<iframe src="http://localhost:8000/artifacts/plot_placeholder.html" width="100%" height="400" frameborder="0" style="border: 1px solid #ddd; border-radius: 4px;"></iframe>'
    );
    return processedText.replace(/\n/g, "<br/>");
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

  // Function to render tool results with artifacts
  const renderToolResults = () => {
    if (!out?.toolResults) return null;
    
    return (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">Tool Results:</h3>
        {out.toolResults.map((result, index: number) => (
          <div key={index} className="border rounded p-4 bg-gray-50">
            <h4 className="font-medium">{result.tool}</h4>
            <pre className="text-sm bg-white p-2 rounded mt-2 overflow-auto">
              {JSON.stringify(result.result, null, 2)}
            </pre>
            {(() => {
              const resultObj = result.result as Record<string, unknown>;
              const artifactUrl = resultObj?.artifact_url;
              if (artifactUrl && typeof artifactUrl === 'string') {
                return (
                  <div className="mt-4">
                    <h5 className="font-medium mb-2">Chart:</h5>
                    <iframe 
                      src={`http://localhost:8000${artifactUrl}`}
                      width="100%" 
                      height="400" 
                      frameBorder="0" 
                      style={{ border: '1px solid #ddd', borderRadius: '4px' }}
                      title="Interactive Chart"
                    />
                  </div>
                );
              }
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
    <main className="p-6 max-w-3xl mx-auto space-y-4">
      <h1 className="text-xl font-semibold">Problem Solver MVP</h1>
      <textarea className="w-full border p-3 rounded" rows={4} value={q} onChange={e=>setQ(e.target.value)} />
      <button onClick={run} disabled={loading} className="px-4 py-2 rounded bg-black text-white">
        {loading ? "Solving..." : "Solve"}
      </button>

      {out?.text && (
        <article className="prose max-w-none">
          {/* Render text; show any data-URL images embedded by the model */}
          <div dangerouslySetInnerHTML={{__html: processText(out.text)}} />
        </article>
      )}

      {renderSuccessEvaluation()}

      {renderToolResults()}
    </main>
  );
}

