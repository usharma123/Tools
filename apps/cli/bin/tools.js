#!/usr/bin/env node
import meow from 'meow';
import React from 'react';
import {render} from 'ink';
import {App} from '../src/App.js';

const cli = meow(
  `
Usage
  $ tools --tool <name> --json '<args>'

Options
  --tool     Tool name, e.g. forecast_arima, plot_line
  --json     Inline JSON payload for tool args
  --file     Path to JSON file for tool args
  --worker   Path to Python worker dir (default: ../../worker)
  --render   Render mode: auto | terminal | browser (default: auto)
  --open     Also open in GUI (browser/image)

Examples
  $ tools --tool plot_line --json '{"series":{"A":[1,2,3]}}'
`,
  {
    importMeta: import.meta,
    flags: {
      tool: {type: 'string', isRequired: true},
      json: {type: 'string'},
      file: {type: 'string'},
      worker: {type: 'string'},
      render: {type: 'string', default: 'auto'},
      open: {type: 'boolean', default: false}
    }
  }
);

render(React.createElement(App, { tool: cli.flags.tool, json: cli.flags.json, file: cli.flags.file, worker: cli.flags.worker, renderMode: cli.flags.render, open: cli.flags.open }));


