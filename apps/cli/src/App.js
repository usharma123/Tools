import React, {useEffect, useState, useMemo} from 'react';
import {Box, Text} from 'ink';
import terminalImage from 'terminal-image';
import {execa} from 'execa';
import path from 'path';
import {fileURLToPath} from 'url';
import {promises as fs} from 'fs';
import os from 'os';

function decodeDataUrlToBuffer(dataUrl) {
	if (!dataUrl || typeof dataUrl !== 'string') return null;
	const prefix = 'data:image/png;base64,';
	if (dataUrl.startsWith(prefix)) {
		const b64 = dataUrl.slice(prefix.length);
		return Buffer.from(b64, 'base64');
	}
	return null;
}

export const App = ({tool, json, file, worker, renderMode = 'auto', open = false}) => {
	const [status, setStatus] = useState('running');
	const [result, setResult] = useState(null);
	const [error, setError] = useState(null);
	const workerDir = useMemo(() => {
		if (worker) return worker;
		const __dirname = path.dirname(fileURLToPath(import.meta.url));
		return path.resolve(__dirname, '../../worker');
	}, [worker]);

	useEffect(() => {
		(async () => {
			try {
				const runner = path.join(workerDir, 'runner.py');
				const args = ['--tool', tool];
				if (json) args.push('--json', json);
				if (file) args.push('--file', file);
				const proc = await execa('python3', [runner, ...args], {
					cwd: workerDir,
					reject: false
				});
				if (proc.exitCode !== 0) {
					setError(proc.stderr || `Runner exit code ${proc.exitCode}`);
					setStatus('done');
					return;
				}
				const parsed = JSON.parse(proc.stdout || '{}');
				setResult(parsed);
				setStatus('done');
			} catch (e) {
				setError(String(e));
				setStatus('done');
			}
		})();
	}, [tool, json, file, workerDir]);

	const [imageBlock, setImageBlock] = useState(null);
	useEffect(() => {
		(async () => {
			if (!result) return;
			const maybeImages = [];
			if (typeof result.artifact_url === 'string') maybeImages.push(result.artifact_url);
			if (typeof result.image_base64 === 'string') maybeImages.push(result.image_base64);
			if (Array.isArray(result.images)) maybeImages.push(...result.images);
			for (const img of maybeImages) {
				const buf = decodeDataUrlToBuffer(img);
				if (!buf) continue;
				try {
					if (renderMode === 'browser') break;
					const rendered = await terminalImage.buffer(buf, {width: '50%', height: '50%'});
					setImageBlock(rendered);
					break;
				} catch {}
			}
		})();
	}, [result, renderMode]);

	function sparkline(values) {
		if (!Array.isArray(values) || values.length < 2) return null;
		const blocks = ['▁','▂','▃','▄','▅','▆','▇','█'];
		let min = Infinity, max = -Infinity;
		for (const v of values) { if (typeof v === 'number' && isFinite(v)) { if (v < min) min = v; if (v > max) max = v; } }
		if (!isFinite(min) || !isFinite(max) || min === max) return null;
		const out = values.map(v => {
			if (typeof v !== 'number' || !isFinite(v)) return blocks[0];
			const idx = Math.max(0, Math.min(7, Math.floor(((v - min) / (max - min)) * 7)));
			return blocks[idx];
		}).join('');
		return out;
	}

	const asciiFallback = useMemo(() => {
		if (!result || imageBlock || renderMode === 'browser') return null;
		// Try to find a numeric array to visualize
		const candidateKeys = ['forecast', 'values', 'data', 'smape_model_pct', 'smape_naive_pct'];
		for (const key of candidateKeys) {
			const val = result[key];
			if (Array.isArray(val) && val.length >= 2 && val.every(x => typeof x === 'number')) {
				const s = sparkline(val);
				if (s) return {label: key, s};
			}
		}
		// Backtest arrays
		if (Array.isArray(result.by_fold)) {
			const model = result.by_fold.map(f => f?.smape_model_pct).filter(v => typeof v === 'number');
			const s = sparkline(model);
			if (s) return {label: 'by_fold.smape_model_pct', s};
		}
		return null;
	}, [result, imageBlock, renderMode]);

	// Optional GUI opening (macOS): browser for HTML, or write PNG to temp and open
	useEffect(() => {
		(async () => {
			if (!result || !open) return;
			const htmlUrl = result.html_artifact_url || (typeof result.artifact_url === 'string' && result.artifact_url.endsWith('.html') ? result.artifact_url : null);
			if (renderMode === 'browser' || htmlUrl) {
				const __dirname = path.dirname(fileURLToPath(import.meta.url));
				const workerDirAbs = worker ? worker : path.resolve(__dirname, '../../worker');
				const rel = (htmlUrl || '').replace(/^\/?artifacts\//, 'artifacts/');
				const absPath = path.resolve(workerDirAbs, rel);
				try { await execa('open', [absPath]); } catch {}
				return;
			}
			const b64 = typeof result.image_base64 === 'string' ? result.image_base64 : (typeof result.artifact_url === 'string' && result.artifact_url.startsWith('data:image/') ? result.artifact_url : null);
			const buf = b64 ? decodeDataUrlToBuffer(b64) : null;
			if (!buf) return;
			const tmp = path.join(os.tmpdir(), `tools-cli-${Date.now()}.png`);
			await fs.writeFile(tmp, buf);
			try { await execa('open', [tmp]); } catch {}
		})();
	}, [result, open, renderMode, worker]);

	return React.createElement(
		Box,
		{flexDirection: 'column'},
		[
			React.createElement(
				Text,
				{key: 'hdr'},
				[
					'Tool: ',
					React.createElement(Text, {key: 'tool', color: 'cyan'}, tool)
				]
			),
			status === 'running' ? React.createElement(Text, {key: 'run', color: 'yellow'}, 'Running…') : null,
			error ? React.createElement(Text, {key: 'err', color: 'red'}, String(error)) : null,
			result ? React.createElement(
				Box,
				{key: 'resbox', flexDirection: 'column', marginTop: 1},
				[
					React.createElement(Text, {key: 'ok', color: 'green'}, 'OK'),
					(renderMode !== 'browser' && imageBlock) ? React.createElement(Text, {key: 'img'}, imageBlock) : null,
					(!imageBlock && asciiFallback) ? React.createElement(
						Box,
						{key: 'ascii', flexDirection: 'column'},
						[
							React.createElement(Text, {key: 'asciilabel', dimColor: true}, `ASCII preview (${asciiFallback.label}):`),
							React.createElement(Text, {key: 'asciival'}, asciiFallback.s)
						]
					) : null,
					(!imageBlock && !asciiFallback) ? React.createElement(Text, {key: 'noimg', dimColor: true}, 'No inline image found; see artifacts HTML if provided.') : null,
					React.createElement(
						Box,
						{key: 'json', marginTop: 1},
						React.createElement(Text, null, JSON.stringify(result, null, 2))
					)
				]
			) : null
		]
	);
};


