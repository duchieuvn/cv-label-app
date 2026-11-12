import { describe, expect, it, vi } from 'vitest';
import { MODES } from '../../client/src/modes.ts';
import type { Annotation, Label } from '../../shared/schema.ts';
import {
  changeLabel,
  emptyHistory,
  moveAnnotation,
  nextLabelColor,
  nextUnplaced,
  pushHistory,
  redo,
  removeKeypoint,
  undo,
} from '../../client/src/annotate/doc.ts';
import { fitView, resizeBox, screenToImage, zoomAt } from '../../client/src/annotate/geometry.ts';
import { SaveQueue } from '../../client/src/annotate/saveQueue.ts';

describe('nextLabelColor', () => {
  it('never repeats a color, including past the end of the palette', () => {
    const labels: { color: string }[] = [];
    for (let i = 0; i < 100; i++) labels.push({ color: nextLabelColor(labels) });
    expect(new Set(labels.map((l) => l.color)).size).toBe(100);
    for (const { color } of labels) expect(color).toMatch(/^#[0-9a-f]{6}$/);
  });

  it('skips colors in use, ignoring case, and reuses one freed by a deleted label', () => {
    const first = nextLabelColor([]);
    const second = nextLabelColor([{ color: first.toUpperCase() }]);
    expect(second).not.toBe(first);
    expect(nextLabelColor([{ color: second }])).toBe(first);
  });
});

const size = { width: 100, height: 100 };
const label: Label = {
  id: 'L',
  name: 'l',
  color: '#000000',
  keypoints: [
    { id: 'a', name: 'a' },
    { id: 'b', name: 'b' },
    { id: 'c', name: 'c' },
  ],
  skeleton: [],
};

describe('resizeBox', () => {
  const box = { x: 10, y: 10, w: 20, h: 20 };

  it('moves only the dragged edge', () => {
    expect(resizeBox(box, 'e', { x: 50, y: 99 }, size)).toEqual({ x: 10, y: 10, w: 40, h: 20 });
  });

  it('flips when dragged past the opposite side', () => {
    expect(resizeBox(box, 'se', { x: 5, y: 5 }, size)).toEqual({ x: 5, y: 5, w: 5, h: 5 });
  });

  it('clamps to the image', () => {
    expect(resizeBox(box, 'se', { x: 500, y: 500 }, size)).toEqual({ x: 10, y: 10, w: 90, h: 90 });
  });

  it('never collapses to zero size, even at the image edge', () => {
    const result = resizeBox({ x: 99, y: 0, w: 1, h: 10 }, 'w', { x: 100, y: 0 }, size);
    expect(result.w).toBeGreaterThan(0);
    expect(result.x + result.w).toBeLessThanOrEqual(100);
  });
});

describe('view transforms', () => {
  it('fit centers the image inside the padding', () => {
    const view = fitView({ width: 100, height: 50 }, { width: 248, height: 248 });
    expect(view.scale).toBe(2);
    expect(view.tx).toBe(24);
    expect(view.ty).toBe(74);
  });

  it('zoom keeps the point under the cursor fixed', () => {
    const view = { scale: 2, tx: 10, ty: 20 };
    const anchor = { x: 110, y: 220 };
    const before = screenToImage(view, anchor);
    const after = screenToImage(zoomAt(view, 1.7, anchor), anchor);
    expect(after.x).toBeCloseTo(before.x);
    expect(after.y).toBeCloseTo(before.y);
  });
});

describe('moveAnnotation', () => {
  const a: Annotation = {
    id: '1',
    labelId: 'L',
    bbox: { x: 10, y: 10, w: 10, h: 10 },
    keypoints: { a: { x: 5, y: 15, v: 2 } },
  };

  it('moves box and keypoints together', () => {
    const moved = moveAnnotation(a, 3, 4, size);
    expect(moved.bbox).toEqual({ x: 13, y: 14, w: 10, h: 10 });
    expect(moved.keypoints.a).toEqual({ x: 8, y: 19, v: 2 });
  });

  it('stops rigidly when a keypoint outside the box reaches the edge', () => {
    const moved = moveAnnotation(a, -50, 0, size);
    expect(moved.keypoints.a.x).toBe(0);
    expect(moved.bbox).toMatchObject({ x: 5, w: 10 });
  });
});

describe('keypoint helpers', () => {
  const a = (keypoints: Annotation['keypoints']): Annotation => ({ id: '1', labelId: 'L', bbox: null, keypoints });
  const p = { x: 1, y: 1, v: 2 as const };

  it('nextUnplaced walks forward and wraps past skipped points', () => {
    expect(nextUnplaced(label, a({}), null)).toBe('a');
    expect(nextUnplaced(label, a({ a: p }), 'a')).toBe('b');
    expect(nextUnplaced(label, a({ b: p, c: p }), 'c')).toBe('a');
    expect(nextUnplaced(label, a({ a: p, b: p, c: p }), 'c')).toBeNull();
  });

  it('removing the last keypoint of a box-less annotation removes the annotation', () => {
    expect(removeKeypoint([a({ a: p })], '1', 'a')).toEqual([]);
    expect(removeKeypoint([a({ a: p, b: p })], '1', 'a')[0].keypoints).toEqual({ b: p });
  });

  it('changing label drops keypoints and refuses to leave no geometry', () => {
    expect(changeLabel(a({ a: p }), 'other')).toBeNull();
    const boxed = { ...a({ a: p }), bbox: { x: 0, y: 0, w: 1, h: 1 } };
    expect(changeLabel(boxed, 'other')).toMatchObject({ labelId: 'other', keypoints: {} });
  });
});

describe('history', () => {
  const doc = (id: string): Annotation[] => [{ id, labelId: 'L', bbox: { x: 0, y: 0, w: 1, h: 1 }, keypoints: {} }];

  it('undoes and redoes, and a new edit clears redo', () => {
    let h = pushHistory(pushHistory(emptyHistory, doc('1')), doc('2'));
    h = undo(h);
    expect(h.present).toEqual(doc('1'));
    h = redo(h);
    expect(h.present).toEqual(doc('2'));
    h = pushHistory(undo(h), doc('3'));
    expect(h.future).toEqual([]);
    expect(undo(emptyHistory)).toBe(emptyHistory);
  });

  it('caps history length', () => {
    let h = emptyHistory;
    for (let i = 0; i < 150; i++) h = pushHistory(h, doc(String(i)));
    expect(h.past.length).toBe(100);
  });
});

describe('SaveQueue', () => {
  const annotations = (id: string): Annotation[] => [{ id, labelId: 'L', bbox: { x: 0, y: 0, w: 1, h: 1 }, keypoints: {} }];

  it('merges waiting payloads into one request', async () => {
    const save = vi.fn(async () => {});
    const queue = new SaveQueue(save);
    queue.schedule('i', { annotations: annotations('old') });
    queue.schedule('i', { annotations: annotations('new') });
    queue.schedule('i', { status: 'done' });
    await queue.flush();
    expect(save).toHaveBeenCalledTimes(1);
    expect(save).toHaveBeenCalledWith('i', { annotations: annotations('new'), status: 'done' }, false);
    expect(queue.state.kind).toBe('saved');
  });

  it('never runs two saves at once, so older writes cannot land last', async () => {
    const order: string[] = [];
    let release = () => {};
    const save = vi.fn(async (_id: string, payload: { annotations?: Annotation[] }) => {
      const name = payload.annotations![0].id;
      order.push(`start ${name}`);
      if (name === 'first') await new Promise<void>((r) => (release = r));
      order.push(`end ${name}`);
    });
    const queue = new SaveQueue(save);
    queue.schedule('i', { annotations: annotations('first') });
    const firstFlush = queue.flush();
    queue.schedule('i', { annotations: annotations('second') });
    const secondFlush = queue.flush();
    await new Promise((r) => setTimeout(r));
    release();
    await Promise.all([firstFlush, secondFlush]);
    expect(order).toEqual(['start first', 'end first', 'start second', 'end second']);
  });

  it('keeps a failed payload, with newer edits winning, until a retry succeeds', async () => {
    vi.useFakeTimers();
    try {
      const save = vi.fn<(id: string, payload: object) => Promise<void>>().mockRejectedValueOnce(new Error('offline')).mockResolvedValue();
      const queue = new SaveQueue(save);
      queue.schedule('i', { annotations: annotations('edit'), status: 'done' });
      await queue.flush();
      expect(queue.state).toEqual({ kind: 'error', message: 'offline' });
      expect(queue.pendingFor('i')).toBeDefined();

      queue.schedule('i', { annotations: annotations('newer') });
      await vi.advanceTimersByTimeAsync(3000);
      expect(save).toHaveBeenLastCalledWith('i', { annotations: annotations('newer'), status: 'done' }, false);
      expect(queue.state.kind).toBe('saved');
      expect(queue.pendingFor('i')).toBeUndefined();
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('MODES', () => {
  it('starts every mode on one of its own tools', () => {
    for (const mode of Object.values(MODES)) expect(mode.tools).toContain(mode.defaultTool);
  });

  it('keeps each mode to its own tools and features', () => {
    expect(MODES.bbox).toMatchObject({ tools: ['select', 'bbox'], showBoxes: true, keypoints: false, autoBox: true });
    expect(MODES.keypoint).toMatchObject({ tools: ['select', 'keypoint'], showBoxes: false, keypoints: true, autoBox: false });
    expect(JSON.stringify(MODES.bbox.shortcuts).toLowerCase()).not.toContain('keypoint');
    expect(JSON.stringify(MODES.keypoint.shortcuts).toLowerCase()).not.toContain('box');
  });
});
