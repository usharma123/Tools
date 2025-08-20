import { markovMcsParams, plotLineParams, plotBarParams, summarizeParams } from "./schemas";

const WORKER = process.env.WORKER_URL || "http://localhost:8000";

// Export schemas for backward compatibility
export { markovMcsParams, plotLineParams, plotBarParams, summarizeParams };

export async function run_markov_mcs(args: unknown) {
  const parsed = markovMcsParams.parse(args);
  try {
    const res = await fetch(`${WORKER}/tools/markov_mcs`, {
      method: "POST",
      headers: {"content-type":"application/json"},
      body: JSON.stringify(parsed)
    });
    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`markov_mcs failed: ${res.status} - ${errorText}`);
    }
    const responseText = await res.text();
    if (!responseText) {
      throw new Error("Empty response from worker");
    }
    return JSON.parse(responseText);
  } catch (error) {
    console.error("Error calling markov_mcs:", String(error));
    throw new Error(`Failed to call markov_mcs: ${error instanceof Error ? error.message : String(error)}`);
  }
}

export async function plot_line(args: unknown) {
  const parsed = plotLineParams.parse(args);
  try {
    const res = await fetch(`${WORKER}/tools/plot_line`, {
      method: "POST",
      headers: {"content-type":"application/json"},
      body: JSON.stringify(parsed)
    });
    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`plot_line failed: ${res.status} - ${errorText}`);
    }
    const responseText = await res.text();
    if (!responseText) {
      throw new Error("Empty response from worker");
    }
    return JSON.parse(responseText);
  } catch (error) {
    console.error("Error calling plot_line:", String(error));
    throw new Error(`Failed to call plot_line: ${error instanceof Error ? error.message : String(error)}`);
  }
}

export async function plot_bar(args: unknown) {
  const parsed = plotBarParams.parse(args);
  try {
    const res = await fetch(`${WORKER}/tools/plot_bar`, {
      method: "POST",
      headers: {"content-type":"application/json"},
      body: JSON.stringify(parsed)
    });
    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`plot_bar failed: ${res.status} - ${errorText}`);
    }
    const responseText = await res.text();
    if (!responseText) {
      throw new Error("Empty response from worker");
    }
    return JSON.parse(responseText);
  } catch (error) {
    console.error("Error calling plot_bar:", String(error));
    throw new Error(`Failed to call plot_bar: ${error instanceof Error ? error.message : String(error)}`);
  }
}

export async function summarize_results(args: unknown) {
  const parsed = summarizeParams.parse(args);
  try {
    const res = await fetch(`${WORKER}/tools/summarize_results`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(parsed),
    });
    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`summarize_results failed: ${res.status} - ${errorText}`);
    }
    const responseText = await res.text();
    if (!responseText) {
      throw new Error("Empty response from worker");
    }
    return JSON.parse(responseText);
  } catch (error) {
    console.error("Error calling summarize_results:", String(error));
    throw new Error(`Failed to call summarize_results: ${error instanceof Error ? error.message : String(error)}`);
  }
}
