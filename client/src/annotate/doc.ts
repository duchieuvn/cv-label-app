import type { Annotation, BBox, Label } from '../../../shared/schema.ts';
import { clamp, type Point, type Size } from './geometry.ts';

export { nextLabelColor } from '../../../shared/colors.ts';

export type Doc = Annotation[];
export type Selection = { annotationId: string; keypointId: string | null } | null;

/** randomUUID only exists in secure contexts, which excludes plain-HTTP LAN access. */
export function newId(): string {
  if (crypto.randomUUID) return crypto.randomUUID();
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, '0')).join('');
}

const replace = (doc: Doc, updated: Annotation): Doc => doc.map((a) => (a.id === updated.id ? updated : a));

export const setAnnotation = replace;

export const removeAnnotation = (doc: Doc, id: string): Doc => doc.filter((a) => a.id !== id);

/** Removes a placed keypoint, and the whole annotation if that leaves it with no geometry. */
export function removeKeypoint(doc: Doc, id: string, keypointId: string): Doc {
  const a = doc.find((x) => x.id === id);
  if (!a) return doc;
  const { [keypointId]: _removed, ...keypoints } = a.keypoints;
  if (!a.bbox && Object.keys(keypoints).length === 0) return removeAnnotation(doc, id);
  return replace(doc, { ...a, keypoints });
}

export function placeKeypoint(a: Annotation, keypointId: string, p: Point, v: 1 | 2): Annotation {
  return { ...a, keypoints: { ...a.keypoints, [keypointId]: { x: p.x, y: p.y, v } } };
}

export function toggleVisibility(a: Annotation, keypointId: string): Annotation {
  const point = a.keypoints[keypointId];
  if (!point) return a;
  return { ...a, keypoints: { ...a.keypoints, [keypointId]: { ...point, v: point.v === 2 ? 1 : 2 } } };
}

/**
 * The next unplaced keypoint after `afterId` in schema order, wrapping around.
 * `afterId` itself is only returned if it is unplaced and nothing else is.
 */
export function nextUnplaced(label: Label, a: Annotation, afterId: string | null): string | null {
  const ids = label.keypoints.map((k) => k.id);
  const start = afterId ? ids.indexOf(afterId) + 1 : 0;
  for (let i = 0; i < ids.length; i++) {
    const id = ids[(start + i) % ids.length];
    if (!a.keypoints[id]) return id;
  }
  return null;
}

/** Keypoint ids are per label, so changing label drops them. Null if that would leave no geometry. */
export function changeLabel(a: Annotation, labelId: string): Annotation | null {
  if (a.labelId === labelId) return a;
  if (!a.bbox) return null;
  return { ...a, labelId, keypoints: {} };
}

/** Translates a whole annotation rigidly, stopping when any of its parts would leave the image. */
export function moveAnnotation(a: Annotation, dx: number, dy: number, size: Size): Annotation {
  const points = Object.values(a.keypoints);
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  if (a.bbox) {
    xs.push(a.bbox.x, a.bbox.x + a.bbox.w);
    ys.push(a.bbox.y, a.bbox.y + a.bbox.h);
  }
  const moveX = clamp(dx, -Math.min(...xs), size.width - Math.max(...xs));
  const moveY = clamp(dy, -Math.min(...ys), size.height - Math.max(...ys));
  return {
    ...a,
    bbox: a.bbox && { ...a.bbox, x: a.bbox.x + moveX, y: a.bbox.y + moveY },
    keypoints: Object.fromEntries(
      Object.entries(a.keypoints).map(([id, p]) => [id, { ...p, x: p.x + moveX, y: p.y + moveY }]),
    ),
  };
}

export const boxAnnotation = (labelId: string, bbox: BBox): Annotation => ({ id: newId(), labelId, bbox, keypoints: {} });

export type History = { imageId: string | null; past: Doc[]; present: Doc; future: Doc[] };

const HISTORY_LIMIT = 100;

export const emptyHistory: History = { imageId: null, past: [], present: [], future: [] };

export const resetHistory = (imageId: string, doc: Doc): History => ({ imageId, past: [], present: doc, future: [] });

export const pushHistory = (h: History, doc: Doc): History => ({
  ...h,
  past: [...h.past, h.present].slice(-HISTORY_LIMIT),
  present: doc,
  future: [],
});

export function undo(h: History): History {
  if (h.past.length === 0) return h;
  return { ...h, past: h.past.slice(0, -1), present: h.past[h.past.length - 1], future: [h.present, ...h.future] };
}

export function redo(h: History): History {
  if (h.future.length === 0) return h;
  return { ...h, past: [...h.past, h.present], present: h.future[0], future: h.future.slice(1) };
}
