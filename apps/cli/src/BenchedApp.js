import React, {useEffect, useMemo, useState} from 'react';
import {Box, Text, useInput} from 'ink';
import {execa} from 'execa';
import path from 'path';
import {fileURLToPath} from 'url';
import {renderBannerToSpans} from './art.js';

const toolsPretty = {
	plot_line: 'Plot Line',
	plot_bar: 'Plot Bar',
	plot_bar_with_ci: 'Bar with CI',
	forecast_arima: 'Forecast ARIMA',
	forecast_backtest: 'Forecast Backtest',
	ab_test_ttest: 'A/B Test T-test',
	markov_mcs: 'Markov MCS',
	power_curve: 'Power Curve'
};

function useWorkerDir() {
	return useMemo(() => {
		const __dirname = path.dirname(fileURLToPath(import.meta.url));
		return path.resolve(__dirname, '../../worker');
	}, []);
}

export const BenchedApp = () => {
	const workerDir = useWorkerDir();
	const [step, setStep] = useState('load'); // load | pick | edit | run | done
	const [list, setList] = useState([]);
	const [idx, setIdx] = useState(0);
	const [tool, setTool] = useState(null);
	const [json, setJson] = useState('{}');
	const [output, setOutput] = useState(null);
	const [error, setError] = useState(null);
	const [editBuffer, setEditBuffer] = useState('');

	useEffect(() => {
		(async () => {
			try {
				const runner = path.join(workerDir, 'runner.py');
				const proc = await execa('python3', [runner, '--list'], {cwd: workerDir, reject: false});
				const parsed = JSON.parse(proc.stdout || '{}');
				const tools = (parsed.tools || []).map(t => t.name);
				setList(tools);
				setTool(tools[0] || null);
				setStep('pick');
			} catch (e) {
				setError(String(e));
				setStep('done');
			}
		})();
	}, [workerDir]);

	useInput((input, key) => {
		if (step === 'pick') {
			if (key.upArrow) {
				const next = Math.max(0, idx - 1);
				setIdx(next);
				setTool(list[next] || tool);
			}
			if (key.downArrow) {
				const next = Math.min(list.length - 1, idx + 1);
				setIdx(next);
				setTool(list[next] || tool);
			}
			if (key.return) setStep('edit');
			return;
		}
		if (step === 'edit') {
			if (key.return) {
				const buf = editBuffer.trim();
				setJson(buf.length ? buf : '{}');
				setStep('run');
				return;
			}
			if (key.backspace || key.delete) {
				setEditBuffer(b => (b.length ? b.slice(0, -1) : b));
				return;
			}
			if (input) {
				setEditBuffer(b => b + input);
			}
		}
	});

	useEffect(() => {
		(async () => {
			if (step !== 'run' || !tool) return;
			try {
				const runner = path.join(workerDir, 'runner.py');
				const args = ['--tool', tool, '--json', json];
				const proc = await execa('python3', [runner, ...args], {cwd: workerDir, reject: false});
				if (proc.exitCode !== 0) throw new Error(proc.stderr || `exit ${proc.exitCode}`);
				setOutput(proc.stdout);
				setStep('done');
			} catch (e) {
				setError(String(e));
				setStep('done');
			}
		})();
	}, [step, tool, json, workerDir]);

	const banner = renderBannerToSpans();
	return React.createElement(
		Box,
		{flexDirection: 'column'},
		[
			React.createElement(
				Box,
				{key: 'banner', flexDirection: 'column', marginBottom: 1},
				banner.map((spans, r) => React.createElement(
					Text,
					{key: `b${r}`},
					spans.map((s, i) => React.createElement(Text, {key: `s${r}-${i}`, color: s.color}, s.text))
				))
			),
			(step === 'load') ? React.createElement(Text, {key: 'load', color: 'yellow'}, 'Loading tools…') : null,
			(step === 'pick') ? React.createElement(
				Box,
				{key: 'pick', flexDirection: 'column', marginTop: 1},
				[
					React.createElement(Text, {key: 'picklabel'}, 'Select a tool (↑/↓, Enter):'),
					...list.map((t, i) => (
						React.createElement(Text, {key: t, color: (i === idx ? 'cyan' : undefined)}, `${i === idx ? '› ' : '  '}${toolsPretty[t] || t}`)
					))
				]
			) : null,
			(step === 'edit') ? React.createElement(
				Box,
				{key: 'edit', flexDirection: 'column', marginTop: 1},
				[
					React.createElement(Text, {key: 'editlabel'}, ['Enter JSON args for ', React.createElement(Text, {color: 'cyan', key: 'tool'}, tool), ' then press Enter to run:']),
					React.createElement(Text, {key: 'editbuf', dimColor: true}, editBuffer || json || '{}'),
					React.createElement(Text, {key: 'tip', dimColor: true}, '(Tip: Type or paste JSON; press Enter to run.)')
				]
			) : null,
			(step === 'run') ? React.createElement(Text, {key: 'run', color: 'yellow'}, 'Running…') : null,
			(step === 'done') ? React.createElement(
				Box,
				{key: 'done', flexDirection: 'column', marginTop: 1},
				[
					error ? React.createElement(Text, {key: 'err', color: 'red'}, error) : React.createElement(Text, {key: 'out'}, output)
				]
			) : null
		]
	);
};


