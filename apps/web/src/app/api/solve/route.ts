import { NextRequest } from "next/server";
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import { run_markov_mcs, plot_line } from "@/lib/tools";

export const runtime = "nodejs";

const SYSTEM = `
You are Chief Analyst.
1) First output an AnalysisPlan JSON (objective, steps, report_outline).
2) Execute steps strictly via available tools.
3) If inputs are missing, ask targeted questions.
4) Prefer simplest valid methods; include assumptions & limitations.
5) Reference any returned artifact_url in your write-up.
Return a concise Decision and Evidence.

Available tools:
- markov_mcs: Run Monte Carlo on a Markov chain. Parameters: transition (array of arrays), steps (number), trials (number), metric (stationary/avg_reward)
- plot_line: Create a simple line chart. Parameters: series (object with arrays), title (string), xlabel (string), ylabel (string)

When you need to use a tool, output: TOOL_CALL:tool_name:parameters_json
For example:
TOOL_CALL:markov_mcs:{"transition": [[0.9, 0.1], [0.2, 0.8]], "steps": 1000, "trials": 1000}
TOOL_CALL:plot_line:{"series": {"State 0": [1,2,3], "State 1": [4,5,6]}, "title": "Cumulative Visits", "xlabel": "Trials", "ylabel": "Visits"}

IMPORTANT: 
- If the query asks for plotting or visualization, you MUST use the plot_line tool to create charts after the markov_mcs tool is executed.
- For plotting stationary distributions, use the stationary_estimate values as the data points.
- ALWAYS include all required parameters in the JSON (title, xlabel, ylabel for plot_line).
- Make sure JSON is complete and properly closed with all braces.
`;

async function executeTool(toolName: string, params: unknown) {
  try {
    if (toolName === "markov_mcs") {
      return await run_markov_mcs(params);
    } else if (toolName === "plot_line") {
      return await plot_line(params);
    } else {
      throw new Error(`Unknown tool: ${toolName}`);
    }
  } catch (error) {
    return { error: `Tool execution failed: ${error}` };
  }
}

function parseToolCalls(text: string) {
  const toolCalls = [];
  
  // Find all TOOL_CALL patterns - use a simpler approach
  const toolCallRegex = /TOOL_CALL:(\w+):/g;
  let match;
  let lastIndex = 0;
  
  while ((match = toolCallRegex.exec(text)) !== null) {
    try {
      const toolName = match[1];
      const startPos = match.index + match[0].length;
      
      // Find the complete JSON object by counting braces
      let braceCount = 0;
      let endPos = startPos;
      let foundStart = false;
      
      for (let i = startPos; i < text.length; i++) {
        if (text[i] === '{') {
          if (!foundStart) foundStart = true;
          braceCount++;
        }
        if (text[i] === '}') {
          braceCount--;
          if (braceCount === 0 && foundStart) {
            endPos = i + 1;
            break;
          }
        }
      }
      
      const jsonStr = text.substring(startPos, endPos);
      const params = JSON.parse(jsonStr);
      toolCalls.push({ toolName, params });
      console.log(`Found tool call: ${toolName} with params:`, params);
    } catch (error) {
      console.error(`Failed to parse tool call at position ${match.index}:`, error);
    }
  }
  
  return toolCalls;
}

export async function POST(req: NextRequest) {
  const { query } = await req.json();

  const { text } = await generateText({
    model: openai("gpt-4o-mini"),
    system: SYSTEM,
    prompt: `Problem: ${query}

IMPORTANT: You MUST output ALL required tool calls in your first response. Do not wait for results.

For example, if the problem asks for a Markov chain simulation AND plotting, output BOTH tool calls:

TOOL_CALL:markov_mcs:{"transition": [[0.9, 0.1], [0.2, 0.8]], "steps": 1000, "trials": 1000}
TOOL_CALL:plot_line:{"series": {"State 0": [0.667], "State 1": [0.333]}, "title": "Stationary Distribution", "xlabel": "States", "ylabel": "Probability"}

CRITICAL: If the query mentions "Markov chain", "stationary distribution", or "simulation", you MUST include the markov_mcs tool call.
CRITICAL: If the query mentions "plot", "visualize", or "chart", you MUST include the plot_line tool call.

For plotting, use placeholder data that will be replaced with actual results.

First, output an AnalysisPlan JSON. Then output ALL required tool calls. Finally, provide your analysis.

When you need to use tools, output them in the format:
TOOL_CALL:tool_name:parameters_json`,
  });

  // Parse ALL tool calls from the first response
  const allToolCalls = parseToolCalls(text);
  const allToolResults: Array<{tool: string; params: unknown; result: unknown}> = [];

  // Execute all tool calls in sequence with chaining
  console.log(`Found ${allToolCalls.length} tool calls to execute`);
  
  for (let i = 0; i < allToolCalls.length; i++) {
    const toolCall = allToolCalls[i];
    console.log(`Executing tool: ${toolCall.toolName} with params:`, toolCall.params);
    
    try {
      // If this is a plot_line call and we have previous results, use real data
      if (toolCall.toolName === 'plot_line' && allToolResults.length > 0) {
        const markovResult = allToolResults.find(r => r.tool === 'markov_mcs');
        if (markovResult && 
            typeof markovResult.result === 'object' && 
            markovResult.result !== null &&
            'stationary_estimate' in markovResult.result &&
            Array.isArray((markovResult.result as any).stationary_estimate)) {
          // Replace placeholder data with real stationary distribution
          toolCall.params.series = {
            "State 0": [(markovResult.result as any).stationary_estimate[0]],
            "State 1": [(markovResult.result as any).stationary_estimate[1]]
          };
          console.log(`Using real data for plot:`, toolCall.params.series);
        }
      }
      
      const result = await executeTool(toolCall.toolName, toolCall.params);
      allToolResults.push({
        tool: toolCall.toolName,
        params: toolCall.params,
        result
      });
      console.log(`Tool result:`, result);
    } catch (error) {
      console.error(`Error executing tool ${toolCall.toolName}:`, error);
      allToolResults.push({
        tool: toolCall.toolName,
        params: toolCall.params,
        result: { error: String(error) }
      });
    }
  }

  return new Response(JSON.stringify({ 
    text: text, 
    toolResults: allToolResults,
    originalText: text 
  }), {
    headers: { "content-type": "application/json" },
  });
}
