import argparse
import json
import sys
from typing import Any, Dict, Tuple, Callable
import re


def _load_tool(tool: str) -> Tuple[Callable[[Any], Any], Any]:

	if tool == "plot_line":
		from tools.plot_line import plot_line, PlotArgs
		return plot_line, PlotArgs
	if tool == "plot_bar":
		from tools.plot_bar import plot_bar, PlotBarArgs
		return plot_bar, PlotBarArgs
	if tool == "plot_bar_with_ci":
		from tools.plot_bar_with_ci import plot_bar_with_ci, BarWithCIArgs
		return plot_bar_with_ci, BarWithCIArgs
	if tool == "forecast_arima":
		from tools.forecast_arima import forecast_arima, ARIMAArgs
		return forecast_arima, ARIMAArgs
	if tool == "forecast_backtest":
		from tools.forecast_backtest import forecast_backtest, BacktestArgs
		return forecast_backtest, BacktestArgs
	if tool == "ab_test_ttest":
		from tools.ab_test_ttest import ab_test_ttest, ABTTArgs
		return ab_test_ttest, ABTTArgs
	if tool == "markov_mcs":
		from tools.markov_mcs import run_markov_mcs, MarkovMCSInputs
		return run_markov_mcs, MarkovMCSInputs
	if tool == "power_curve":
		from tools.power_curve import power_curve, PowerCurveArgs
		return power_curve, PowerCurveArgs

	raise ValueError(f"Unknown tool: {tool}")


def _read_json_payload(args: argparse.Namespace) -> Dict[str, Any]:
	if args.json is not None:
		# Treat empty string as empty dict for convenience
		if isinstance(args.json, str) and args.json.strip() == "":
			return {}
		# Otherwise parse provided JSON
		return json.loads(args.json)
	if args.file:
		with open(args.file, "r", encoding="utf-8") as f:
			return json.load(f)
	# Fallback to stdin
	data = sys.stdin.read().strip()
	if not data:
		raise ValueError("No JSON provided. Use --json, --file, or pipe to stdin.")
	return json.loads(data)


def main() -> None:
	parser = argparse.ArgumentParser(description="Run worker tool functions from CLI and return JSON.")
	parser.add_argument("--tool", required=False, help="Tool name (e.g., forecast_arima, plot_line)")
	parser.add_argument("--json", help="Inline JSON string of arguments")
	parser.add_argument("--file", help="Path to JSON file of arguments")
	parser.add_argument("--list", action="store_true", help="List available tools and their arg models")
	parser.add_argument("--text", help="Plain-English instruction to be parsed into tool args")
	parsed = parser.parse_args()

	if parsed.list:
		tools = [
			{"name": "plot_line", "args_model": "PlotArgs"},
			{"name": "plot_bar", "args_model": "PlotBarArgs"},
			{"name": "plot_bar_with_ci", "args_model": "BarWithCIArgs"},
			{"name": "forecast_arima", "args_model": "ARIMAArgs"},
			{"name": "forecast_backtest", "args_model": "BacktestArgs"},
			{"name": "ab_test_ttest", "args_model": "ABTTArgs"},
			{"name": "markov_mcs", "args_model": "MarkovMCSInputs"},
			{"name": "power_curve", "args_model": "PowerCurveArgs"},
		]
		print(json.dumps({"tools": tools}))
		return

	if not parsed.tool:
		print(json.dumps({"ok": False, "error": "--tool is required unless --list is provided"}))
		return

	func, model = _load_tool(parsed.tool)

	def _parse_text_to_args(tool_name: str, text: str) -> Dict[str, Any]:
		# Very lightweight heuristics to map plain English to args
		t = text.strip()
		if not t:
			return {}
		def _num_list(s: str):
			vals = re.split(r"[\s,]+", s.strip())
			out = []
			for v in vals:
				if not v:
					continue
				try:
					out.append(float(v))
				except Exception:
					pass
			return out

		if tool_name == "plot_line":
			series: Dict[str, Any] = {}
			for m in re.finditer(r"([A-Za-z0-9_]+)\s*:\s*([0-9.,\s-]+)", t):
				name = m.group(1)
				nums = _num_list(m.group(2))
				if nums:
					series[name] = nums
			title = None
			m = re.search(r"title\s*:\s*(.+?)(?=\s+[a-z]+:|$)", t, re.IGNORECASE)
			if m: title = m.group(1).strip()
			xlabel = None
			m = re.search(r"xlabel\s*:\s*(.+?)(?=\s+[a-z]+:|$)", t, re.IGNORECASE)
			if m: xlabel = m.group(1).strip()
			ylabel = None
			m = re.search(r"ylabel\s*:\s*(.+?)(?=\s+[a-z]+:|$)", t, re.IGNORECASE)
			if m: ylabel = m.group(1).strip()
			out: Dict[str, Any] = {"series": series}
			if title: out["title"] = title
			if xlabel: out["xlabel"] = xlabel
			if ylabel: out["ylabel"] = ylabel
			return out

		if tool_name == "forecast_arima":
			series = []
			m = re.search(r"series\s*:\s*([0-9.,\s-]+)", t, re.IGNORECASE)
			if m:
				series = _num_list(m.group(1))
			horizon = None
			m = re.search(r"horizon\s+([0-9]+)", t, re.IGNORECASE)
			if m: horizon = int(m.group(1))
			seasonal = None
			m = re.search(r"season(al)?\s+([0-9]+)", t, re.IGNORECASE)
			if m: seasonal = int(m.group(2))
			alpha = None
			m = re.search(r"alpha\s+([0-9.]+)", t, re.IGNORECASE)
			if m:
				try: alpha = float(m.group(1))
				except Exception: alpha = None
			out2: Dict[str, Any] = {"ts": series, "horizon": horizon or 3}
			if seasonal is not None: out2["seasonal_period"] = seasonal
			if alpha is not None: out2["alpha"] = alpha
			return out2

		return {}

	if parsed.text and not (parsed.json or parsed.file):
		payload_dict = _parse_text_to_args(parsed.tool, parsed.text)
	else:
		payload_dict = _read_json_payload(parsed)
	try:
		# Build Pydantic model if available
		args_obj = model(**payload_dict) if model is not None else payload_dict
		res = func(args_obj)
	except Exception as e:
		print(json.dumps({"ok": False, "error": str(e)}))
		return

	# Ensure JSON serializable
	try:
		out = json.dumps(res, ensure_ascii=False)
	except TypeError:
		# Best-effort conversion
		def default(o):
			try:
				return o.__dict__
			except Exception:
				return str(o)
		out = json.dumps(res, default=default, ensure_ascii=False)

	print(out)


if __name__ == "__main__":
	main()


