import type { BBox } from '../../../shared/schema.ts';

export type Point = { x: number; y: number };
export type Size = { width: number; height: number };
export type View = { scale: number; tx: number; ty: number };

export const HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'] as const;
export type Handle = (typeof HANDLES)[number];

const MIN_SCALE = 0.02;
const MAX_SCALE = 40;
const MIN_BOX_SIZE = 1;

export const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

export const clampPoint = (p: Point, size: Size): Point => ({
  x: clamp(p.x, 0, size.width),
  y: clamp(p.y, 0, size.height),
});

export const rectFromPoints = (a: Point, b: Point): BBox => ({
  x: Math.min(a.x, b.x),
  y: Math.min(a.y, b.y),
  w: Math.abs(a.x - b.x),
  h: Math.abs(a.y - b.y),
});

export function handlePosition(box: BBox, handle: Handle): Point {
  const x = handle.includes('w') ? box.x : handle.includes('e') ? box.x + box.w : box.x + box.w / 2;
  const y = handle.includes('n') ? box.y : handle.includes('s') ? box.y + box.h : box.y + box.h / 2;
  return { x, y };
}

/** Moves one edge or corner of the box to `p`. Dragging past the opposite side flips the box. */
export function resizeBox(box: BBox, handle: Handle, p: Point, size: Size): BBox {
  const q = clampPoint(p, size);
  let left = box.x;
  let right = box.x + box.w;
  let top = box.y;
  let bottom = box.y + box.h;
  if (handle.includes('w')) left = q.x;
  if (handle.includes('e')) right = q.x;
  if (handle.includes('n')) top = q.y;
  if (handle.includes('s')) bottom = q.y;

  const rect = rectFromPoints({ x: left, y: top }, { x: right, y: bottom });
  const w = Math.max(rect.w, MIN_BOX_SIZE);
  const h = Math.max(rect.h, MIN_BOX_SIZE);
  return { x: Math.min(rect.x, size.width - w), y: Math.min(rect.y, size.height - h), w, h };
}

export const screenToImage = (view: View, p: Point): Point => ({
  x: (p.x - view.tx) / view.scale,
  y: (p.y - view.ty) / view.scale,
});

export function fitView(image: Size, viewport: Size, padding = 24): View {
  const scale = clamp(
    Math.min((viewport.width - padding * 2) / image.width, (viewport.height - padding * 2) / image.height),
    MIN_SCALE,
    MAX_SCALE,
  );
  return {
    scale,
    tx: (viewport.width - image.width * scale) / 2,
    ty: (viewport.height - image.height * scale) / 2,
  };
}

/** Zooms by `factor` keeping the image point under `anchor` fixed on screen. */
export function zoomAt(view: View, factor: number, anchor: Point): View {
  const scale = clamp(view.scale * factor, MIN_SCALE, MAX_SCALE);
  const ratio = scale / view.scale;
  return { scale, tx: anchor.x - (anchor.x - view.tx) * ratio, ty: anchor.y - (anchor.y - view.ty) * ratio };
}
