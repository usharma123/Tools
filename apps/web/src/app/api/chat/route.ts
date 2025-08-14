import { NextRequest } from "next/server";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
let Redis: any = null;
try { ({ Redis } = require("@upstash/redis")); } catch {}
let LRUCache: any = null;
try { ({ LRUCache } = require("lru-cache")); } catch {}
import { tools } from "@/lib/tools/index";
import { buildPlanVerbose } from "@/lib/planner";
import { materializeArgs } from "@/lib/resolve";
import { cacheKey } from "@/lib/hash";

const redis: any = (Redis && process.env.KV_URL && process.env.KV_TOKEN)
  ? new (Redis as any)({ url: process.env.KV_URL, token: process.env.KV_TOKEN })
  : ({ async get() { return null; }, async set() {} });
const lru: any = LRUCache ? new (LRUCache as any)({ max: 500, ttl: 10 * 60 * 1000 }) : { get(){return null;}, set(){}, has(){return false;} };

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

// Resolve "$asset:<id>.<path>" placeholders recursively using provided assetIndex
function resolveAssetsPlaceholders(args: any, assetIndex: Record<string, any>): any {
  const resolveOne = (v: any): any => {
    if (typeof v === "string" && v.startsWith("$asset:")) {
      const raw = v.slice("$asset:".length);
      const [id, ...rest] = raw.split(".");
      const base = assetIndex[id];
      if (!base) return v;
      if (rest.length === 0) return base;
      return rest.reduce((o, k) => (o == null ? o : (o as any)[k]), base);
    }
    if (Array.isArray(v)) return v.map(resolveOne);
    if (v && typeof v === "object") {
      const out: any = {};
      for (const [k, vv] of Object.entries(v)) out[k] = resolveOne(vv);
      return out;
    }
    return v;
  };
  return resolveOne(args);
}

export async function POST(req: NextRequest) {
  const payload = await req.json(); // { messages, assets? }
  const stream = new ReadableStream({
    async start(controller) {
      const write = (type: string, data: any) => controller.enqueue(new TextEncoder().encode(send(type, data)));
      try {
        // 1) Build plan
        const { plan, logs } = await buildPlanVerbose(payload);
        write("plan", plan);
        if (Array.isArray(logs)) {
          for (const l of logs) write("log", l);
        }

        // Ask for missing info
        if (plan.ask?.length) { write("ask", plan.ask); controller.close(); return; }

        // 2) Execute steps
        const results: Record<string, any> = {};
        const assetIndex: Record<string, any> = {};
        try {
          for (const a of payload.assets ?? []) assetIndex[a.assetId] = a.meta ?? {};
        } catch {}

        const allToolResults: Array<{ tool: string; result: any }> = [];
        for (const step of plan.steps) {
          const spec = (tools as any)[step.tool];
          if (!spec) throw new Error(`Unknown tool: ${step.tool}`);
          write("tool:start", { id: step.id, tool: step.tool });

          // args: resolve $refs + $assets
          const tmp = resolveAssetsPlaceholders(step.args, assetIndex);
          const concreteArgs = materializeArgs(step.tool, tmp, results);

          // cache lookup
          const key = cacheKey({ tool: step.tool, ver: spec.version, args: concreteArgs });
          let out = lru.get(key) || await redis.get(key);

          let wasCached = Boolean(out);
          if (!out) {
            out = await spec.execute(concreteArgs);                    // call worker
            const sz = Buffer.byteLength(JSON.stringify(out), "utf8");
            if (sz < 200_000) { lru.set(key, out); await redis.set(key, out, { ex: 60 * 60 * 24 }); }
          }

          // Emit a single completion event per tool, include full result
          write("tool:done", {
            id: step.id,
            tool: step.tool,
            cached: wasCached,
            result: out,
          });
          allToolResults.push({ tool: step.tool, result: out });
          (results as any)[step.id || step.tool] = out;
        }

        // 3) Summarize results when applicable
        let summary: any = null;
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
            const spec = (tools as any)['summarize_results'];
            if (spec) summary = await spec.execute(summaryPayload);
          }
        } catch {}

        write("final", { results, summary });
        controller.close();
      } catch (e: any) {
        write("error", { message: e?.message || String(e) });
        controller.close();
      }
    }
  });
  return new Response(stream, { headers: sseHeaders() });
}


