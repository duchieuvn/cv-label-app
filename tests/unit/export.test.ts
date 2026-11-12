import { describe, expect, it } from 'vitest';
import type { Annotation, ImageDetail, Label } from '../../shared/schema.ts';
import { effectiveBBox, selectImages, toCoco, toYolo, uniqueStems } from '../../server/export.ts';

const car: Label = { id: 'car', name: 'car', color: '#000000', keypoints: [], skeleton: [] };
const person: Label = {
  id: 'person',
  name: 'person',
  color: '#ffffff',
  keypoints: [
    { id: 'head', name: 'head' },
    { id: 'foot', name: 'foot' },
    { id: 'hand', name: 'hand' },
  ],
  skeleton: [['head', 'foot']],
};

const image = (annotations: Annotation[], filename = 'a.jpg', status: 'todo' | 'done' = 'done'): ImageDetail => ({
  id: filename,
  projectId: 'p',
  filename,
  width: 200,
  height: 100,
  status,
  annotationCount: annotations.length,
  annotations,
  folder: 'uploads',
  imageLabels: [],
  fileState: 'ok',
  sourceId: null,
  autoBoxed: false,
});

const boxed: Annotation = { id: '1', labelId: 'car', bbox: { x: 20, y: 10, w: 40, h: 20 }, keypoints: {} };
const pose: Annotation = {
  id: '2',
  labelId: 'person',
  bbox: { x: 0, y: 0, w: 100, h: 100 },
  keypoints: { head: { x: 50, y: 10, v: 2 }, hand: { x: 100, y: 50, v: 1 } },
};
const keypointOnly: Annotation = { id: '3', labelId: 'person', bbox: null, keypoints: { head: { x: 200, y: 100, v: 2 } } };

describe('selectImages', () => {
  const todoEmpty = image([], 'empty.jpg', 'todo');
  const todoLabeled = image([boxed], 'labeled.jpg', 'todo');
  const doneEmpty = image([], 'negative.jpg', 'done');

  it('done scope keeps only done images, including true negatives', () => {
    expect(selectImages([todoEmpty, todoLabeled, doneEmpty], 'done')).toEqual([doneEmpty]);
  });

  it('annotated scope never includes untouched todo images', () => {
    expect(selectImages([todoEmpty, todoLabeled, doneEmpty], 'annotated')).toEqual([todoLabeled, doneEmpty]);
  });
});

describe('effectiveBBox', () => {
  it('stays inside the image for a single keypoint at the far corner', () => {
    expect(effectiveBBox(keypointOnly, { width: 200, height: 100 })).toEqual({ x: 199, y: 99, w: 1, h: 1 });
  });
});

describe('toCoco', () => {
  const coco = toCoco({ name: 'p', labels: [car, person] }, [image([boxed, pose])]);

  it('writes boxes as xywh with area', () => {
    expect(coco.annotations[0]).toMatchObject({ bbox: [20, 10, 40, 20], area: 800, category_id: 1, image_id: 1, iscrowd: 0 });
    expect(coco.annotations[0]).not.toHaveProperty('keypoints');
  });

  it('writes keypoints in schema order with zeros for unplaced ones', () => {
    expect(coco.annotations[1].keypoints).toEqual([50, 10, 2, 0, 0, 0, 100, 50, 1]);
    expect(coco.annotations[1].num_keypoints).toBe(2);
  });

  it('writes a 1-indexed skeleton on keypoint categories only', () => {
    expect(coco.categories[1]).toMatchObject({ keypoints: ['head', 'foot', 'hand'], skeleton: [[1, 2]] });
    expect(coco.categories[0]).not.toHaveProperty('keypoints');
  });
});

describe('toYolo', () => {
  it('uses plain detection format when no label has keypoints', () => {
    const yolo = toYolo({ name: 'p', labels: [car] }, [image([boxed])]);
    expect(yolo.files[0].content).toBe('0 0.2 0.2 0.2 0.2\n');
    expect(yolo.yaml).not.toContain('kpt_shape');
    expect(yolo.yaml).toContain('0: "car"');
  });

  it('uses pose format and pads labels with fewer keypoints', () => {
    const yolo = toYolo({ name: 'p', labels: [car, person] }, [image([boxed, pose])]);
    const [boxLine, poseLine] = yolo.files[0].content.trim().split('\n');
    expect(yolo.yaml).toContain('kpt_shape: [3, 3]');
    expect(boxLine).toBe('0 0.2 0.2 0.2 0.2 0 0 0 0 0 0 0 0 0');
    expect(poseLine).toBe('1 0.25 0.5 0.5 1 0.25 0.1 2 0 0 0 0.5 0.5 1');
  });

  it('writes an empty label file for a true negative', () => {
    expect(toYolo({ name: 'p', labels: [car] }, [image([])]).files[0].content).toBe('');
  });

  it('gives colliding stems distinct names', () => {
    expect(uniqueStems(['a.jpg', 'a.png', 'A.bmp', 'b.jpg'])).toEqual(['a', 'a_1', 'A_2', 'b']);
  });
});
