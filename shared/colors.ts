import type { Label } from './schema.ts';

const LABEL_PALETTE = ['#4f8cff', '#ff6b6b', '#3ddc97', '#ffb547', '#c77dff', '#29c7d9', '#ff8fc7', '#a3d936', '#ff8c42', '#8d99ff'];

function hslToHex(hue: number, saturation: number, lightness: number): string {
  const amplitude = saturation * Math.min(lightness, 1 - lightness);
  const channel = (n: number) => {
    const k = (n + hue / 30) % 12;
    const value = lightness - amplitude * Math.max(-1, Math.min(k - 3, 9 - k, 1));
    return Math.round(value * 255).toString(16).padStart(2, '0');
  };
  return `#${channel(0)}${channel(8)}${channel(4)}`;
}

/** A color no existing label has: the first free palette color, then hues spread by the golden angle. */
export function nextLabelColor(labels: Pick<Label, 'color'>[]): string {
  const used = new Set(labels.map((l) => l.color.toLowerCase()));
  const free = LABEL_PALETTE.find((color) => !used.has(color));
  if (free) return free;
  for (let i = 0; i < 2000; i++) {
    const color = hslToHex((i * 137.508) % 360, 0.75, 0.62);
    if (!used.has(color)) return color;
  }
  return LABEL_PALETTE[0];
}
