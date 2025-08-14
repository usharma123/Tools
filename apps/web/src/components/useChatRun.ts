import { useRef, useState } from "react";

export function useChatRun() {
  const [plan, setPlan] = useState<any>(null);
  const [asks, setAsks] = useState<any[]>([]);
  const [events, setEvents] = useState<any[]>([]);
  const [final, setFinal] = useState<any>(null);
  const esRef = useRef<EventSource | null>(null);

  const run = async (body: any) => {
    try {
      setPlan(null); setAsks([]); setEvents([]); setFinal(null);
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "accept": "text/event-stream",
        },
        cache: "no-store",
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const txt = await res.text();
        setEvents((x) => [...x, { type: "error", payload: { status: res.status, body: txt } }]);
        return;
      }
      if (!res.body) {
        setEvents((x) => [...x, { type: "error", payload: { message: "No response body (stream missing)" } }]);
        return;
      }
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
            if (eventType === "plan") setPlan(payload);
            else if (eventType === "ask") { setAsks(payload); }
            else if (eventType === "tool:start") setEvents((x) => [...x, payload]);
            else if (eventType === "tool:done") setEvents((x) => [...x, payload]);
            else if (eventType === "final") setFinal(payload);
            else if (eventType === "error") setEvents((x) => [...x, { type: "error", payload }]);
          } catch (e) {
            // ignore JSON parse errors on partial chunks
          }
        }
      }
    } catch (e: any) {
      setEvents((x) => [...x, { type: "error", payload: { message: String(e) } }]);
    }
  };

  return { run, plan, asks, events, final };
}


