import { useState } from "react";

export function useChatRun() {
  const [plan, setPlan] = useState<any>(null);
  const [asks, setAsks] = useState<any[]>([]);
  const [events, setEvents] = useState<any[]>([]);
  const [logs, setLogs] = useState<any[]>([]);
  const [final, setFinal] = useState<any>(null);

  const run = async (body: any) => {
    try {
      setPlan(null); setAsks([]); setEvents([]); setLogs([]); setFinal(null);
      
      // Extract the user's query from the chat messages
      const messages = body.messages || [];
      const lastUserMessage = messages.filter((m: any) => m.role === "user").pop();
      const query = lastUserMessage?.content;
      
      if (!query) {
        setEvents([{ type: "error", payload: { message: "No user query found in messages" } }]);
        return;
      }
      
      // Use /api/solve-stream for real-time logging
      const res = await fetch("/api/solve-stream", {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "accept": "text/event-stream",
        },
        cache: "no-store",
        body: JSON.stringify({ query }),
      });
      
      if (!res.ok) {
        const txt = await res.text();
        setEvents([{ type: "error", payload: { status: res.status, body: txt } }]);
        return;
      }
      
      if (!res.body) {
        setEvents([{ type: "error", payload: { message: "No response body (stream missing)" } }]);
        return;
      }
      
      // Handle streaming response
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        let idx;
        
        while ((idx = buffer.indexOf("\n\n")) !== -1) {
          const chunk = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          const lines = chunk.split("\n");
          let eventType = "message";
          let dataLine = "";
          
          for (const line of lines) {
            if (line.startsWith("event:")) eventType = line.slice(6).trim();
            if (line.startsWith("data:")) dataLine += line.slice(5).trim();
          }
          
          try {
            const payload = dataLine ? JSON.parse(dataLine) : null;
            
            if (eventType === "plan") {
              setPlan(payload);
            } else if (eventType === "ask") {
              setAsks(payload);
            } else if (eventType === "log") {
              setLogs(prevLogs => [...prevLogs, payload]);
            } else if (eventType === "tool-complete") {
              // Add to events when a tool completes
              setEvents(prevEvents => [...prevEvents, {
                id: `tool_${payload.step}`,
                tool: payload.tool,
                type: "done",
                result: payload.result,
                artifact: payload.result?.image_base64 || payload.result?.artifact_url,
                size: payload.result ? JSON.stringify(payload.result).length : 0,
                duration: payload.duration
              }]);
            } else if (eventType === "final") {
              const finalResult: any = {
                results: payload.toolResults || [],
                summary: payload.toolResults?.find((r: any) => r.tool === 'summarize_results')?.result
              };
              
              if (payload.successEvaluation) {
                finalResult.successEvaluation = payload.successEvaluation;
              }
              
              setFinal(finalResult);
            } else if (eventType === "error") {
              setEvents(prevEvents => [...prevEvents, { type: "error", payload }]);
            }
          } catch (e) {
            // ignore JSON parse errors on partial chunks
          }
        }
      }
      
    } catch (e: any) {
      setEvents([{ type: "error", payload: { message: String(e) } }]);
    }
  };

  return { run, plan, asks, events, logs, final };
}


