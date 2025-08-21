export const BENCHED_BANNER = [
  ' ██████  ███████ ███    ██  ██████  ███████  ██████  ███████ ██████  ',
  ' ██   ██ ██      ████   ██ ██       ██      ██    ██ ██      ██   ██ ',
  ' ██████  █████   ██ ██  ██ ██   ███ █████   ██    ██ █████   ██████  ',
  ' ██   ██ ██      ██  ██ ██ ██    ██ ██      ██    ██ ██      ██   ██ ',
  ' ██   ██ ███████ ██   ████  ██████  ███████  ██████  ███████ ██   ██ ',
];

export function renderBannerToSpans() {
  // Return array of {text, color} for simple left-to-right gradient effect
  const colors = ['cyan', 'cyanBright', 'magenta', 'magentaBright'];
  return BENCHED_BANNER.map((line, row) => {
    const spans = [];
    const seg = Math.max(1, Math.floor(line.length / colors.length));
    for (let i = 0; i < colors.length; i++) {
      const start = i * seg;
      const end = (i === colors.length - 1) ? line.length : (i + 1) * seg;
      spans.push({ text: line.slice(start, end), color: colors[i] });
    }
    return spans;
  });
}


