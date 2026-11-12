import { describe, expect, it } from 'vitest';
import type { Label } from '../../shared/schema.ts';
import { ImportFileError, matchImages, parseDataset, planImport, type ImportTarget } from '../../server/import.ts';

const image = (overrides: Partial<ImportTarget> & { id: string; filename: string }): ImportTarget => ({
  folder: 'uploads',
  sourceId: null,
  width: 100,
  height: 80,
  status: 'todo',
  annotationCount: 0,
  autoBoxed: false,
  fileState: 'ok',
  imageLabels: [],
  ...overrides,
});

const scanned = (id: string, folder: string, filename: string, overrides: Partial<ImportTarget> = {}) =>
  image({ id, folder, filename, sourceId: 'src', ...overrides });

const coco = {
  images: [
    { id: 1, file_name: 'dataset/val/a/1.jpg', width: 100, height: 80 },
    { id: 2, file_name: 'dataset/val/b/2.jpg', width: 100, height: 80 },
  ],
  annotations: [
    { id: 1, image_id: 1, category_id: 1, bbox: [10, 10, 20, 20] },
    { id: 2, image_id: 1, category_id: 2, bbox: [-5, 5, 10, 10] },
    { id: 3, image_id: 2, category_id: 1, bbox: [200, 200, 10, 10] },
  ],
  categories: [
    { id: 1, name: 'bottle' },
    { id: 2, name: 'Cap' },
  ],
};

const options = { createLabels: true, overwrite: false, markDone: false };

describe('parseDataset', () => {
  it('reads COCO, keeping unplaced keypoints out and 1-indexed skeletons mapped', () => {
    const dataset = parseDataset({
      images: [{ id: 'img', file_name: 'p.jpg' }],
      annotations: [{ image_id: 'img', category_id: 7, bbox: [0, 0, 5, 5], keypoints: [1, 2, 2, 0, 0, 0, 3, 4, 1] }],
      categories: [{ id: 7, name: 'person', keypoints: ['head', 'hip', 'foot'], skeleton: [[1, 2], [2, 3], [3, 9]] }],
    });
    expect(dataset.format).toBe('coco');
    expect(dataset.categories[0]).toEqual({ name: 'person', keypoints: ['head', 'hip', 'foot'], skeleton: [[0, 1], [1, 2]] });
    expect(dataset.images[0].annotations[0].keypoints).toEqual([{ x: 1, y: 2, v: 2 }, null, { x: 3, y: 4, v: 1 }]);
    expect(dataset.warnings[0]).toContain('skeleton edge [3, 9]');
  });

  it('reports annotations pointing at unknown images or categories instead of failing', () => {
    const dataset = parseDataset({ ...coco, annotations: [{ image_id: 9, category_id: 1, bbox: [0, 0, 1, 1] }, { image_id: 1, category_id: 9, bbox: [0, 0, 1, 1] }] });
    expect(dataset.images.flatMap((i) => i.annotations)).toHaveLength(0);
    expect(dataset.warnings).toHaveLength(2);
  });

  it('reads the native export, with image labels and status', () => {
    const dataset = parseDataset({
      project: { name: 'x', mode: 'keypoint' },
      labels: [{ id: 'L', name: 'person', color: '#ff0000', keypoints: [{ id: 'k1', name: 'head' }], skeleton: [] }],
      images: [
        { filename: 'a.png', folder: 'uploads', imageLabels: ['L'], width: 10, height: 10, status: 'done', annotations: [{ id: 'a', labelId: 'L', bbox: null, keypoints: { k1: { x: 1, y: 1, v: 2 } } }] },
        { filename: 'b.png', folder: 'cam/1', imageLabels: [], width: 10, height: 10, status: 'todo', annotations: [] },
      ],
    });
    expect(dataset.format).toBe('labelapp');
    expect(dataset.images[0]).toMatchObject({ path: 'a.png', imageLabels: ['person'], status: 'done' });
    expect(dataset.images[0].annotations[0].keypoints).toEqual([{ x: 1, y: 1, v: 2 }]);
    expect(dataset.images[1].path).toBe('cam/1/b.png');
  });

  it('rejects files that are neither format, with a reason', () => {
    expect(() => parseDataset([])).toThrow(ImportFileError);
    expect(() => parseDataset({ foo: 1 })).toThrow(/neither a COCO file/);
    expect(() => parseDataset({ images: 'x', categories: [] })).toThrow(/Not a valid COCO file/);
  });
});

describe('matchImages', () => {
  const images = [scanned('a', 'dataset/val/a', '1.jpg'), scanned('b', 'dataset/val/b', '1.jpg'), scanned('c', 'dataset/train/c', '3.jpg'), image({ id: 'u', filename: 'up.png' })];

  it('prefers the longest shared trailing path and reports ties as ambiguous', () => {
    const [exact, suffix, other, byName, tie, none] = matchImages(
      ['dataset/val/a/1.jpg', 'elsewhere/val/b/1.jpg', 'x/train/c/3.jpg', 'up.png', '1.jpg', 'missing.jpg'],
      images,
    );
    expect(exact).toEqual({ image: images[0] });
    expect(suffix).toEqual({ image: images[1] });
    expect(other).toEqual({ image: images[2] });
    expect(byName).toEqual({ image: images[3] });
    expect(tie).toEqual({ ambiguous: true });
    expect(none).toBeNull();
  });

  it('accepts Windows separators', () => {
    expect(matchImages(['dataset\\val\\a\\1.jpg'], images)).toEqual([{ image: images[0] }]);
  });
});

describe('planImport', () => {
  const bottle: Label = { id: 'bottle', name: 'bottle', color: '#4f8cff', keypoints: [], skeleton: [] };
  const targets = [scanned('a', 'dataset/val/a', '1.jpg'), scanned('b', 'dataset/val/b', '2.jpg')];

  it('matches labels by name ignoring case, creates missing ones, and clamps boxes to the image', () => {
    const plan = planImport({ mode: 'bbox', labels: [bottle] }, targets, parseDataset(coco), options);
    expect(plan.labelsChanged).toBe(true);
    expect(plan.labels.map((l) => l.name)).toEqual(['bottle', 'Cap']);
    expect(plan.labels[1].color).not.toBe(bottle.color);
    expect(plan.summary).toMatchObject({ matched: 2, imagesWritten: 1, annotations: 2, droppedAnnotations: 1, newLabels: ['Cap'] });
    const [write] = plan.writes;
    expect(write.imageId).toBe('a');
    expect(write.annotations.map((a) => a.labelId)).toEqual(['bottle', plan.labels[1].id]);
    expect(write.annotations[1].bbox).toEqual({ x: 0, y: 5, w: 5, h: 10 });
    expect(write.status).toBeUndefined();
  });

  it('skips categories without a label when creation is off', () => {
    const plan = planImport({ mode: 'bbox', labels: [bottle] }, targets, parseDataset(coco), { ...options, createLabels: false });
    expect(plan.labelsChanged).toBe(false);
    expect(plan.summary).toMatchObject({ annotations: 1, droppedAnnotations: 2, newLabels: [] });
  });

  it('leaves a person\'s annotations alone unless overwrite is on, but replaces unreviewed predictions', () => {
    const edited = [scanned('a', 'dataset/val/a', '1.jpg', { annotationCount: 1 }), scanned('b', 'dataset/val/b', '2.jpg', { annotationCount: 3, autoBoxed: true })];
    const kept = planImport({ mode: 'bbox', labels: [bottle] }, edited, parseDataset(coco), options);
    expect(kept.summary).toMatchObject({ alreadyAnnotated: 1, imagesWritten: 1 });
    expect(kept.writes[0]).toMatchObject({ imageId: 'b', annotations: [] });
    const replaced = planImport({ mode: 'bbox', labels: [bottle] }, edited, parseDataset(coco), { ...options, overwrite: true });
    expect(replaced.summary).toMatchObject({ alreadyAnnotated: 0, imagesWritten: 2 });
  });

  it('skips unavailable images and images whose size differs from the file', () => {
    const broken = [scanned('a', 'dataset/val/a', '1.jpg', { fileState: 'missing' }), scanned('b', 'dataset/val/b', '2.jpg', { width: 50 })];
    const plan = planImport({ mode: 'bbox', labels: [bottle] }, broken, parseDataset(coco), options);
    expect(plan.summary).toMatchObject({ matched: 2, unavailable: 1, sizeMismatch: 1, imagesWritten: 0 });
    expect(plan.summary.warnings.some((w) => w.includes('100×80') && w.includes('50×80'))).toBe(true);
  });

  it('marks every matched image done when asked, including negatives', () => {
    const plan = planImport({ mode: 'bbox', labels: [bottle] }, targets, parseDataset(coco), { ...options, markDone: true });
    expect(plan.writes.map((w) => [w.imageId, w.status, w.annotations.length])).toEqual([
      ['a', 'done', 2],
      ['b', 'done', 0],
    ]);
  });

  it('ignores keypoints in a bbox project and counts them', () => {
    const dataset = parseDataset({
      images: [{ id: 1, file_name: '1.jpg' }],
      annotations: [{ image_id: 1, category_id: 1, bbox: [0, 0, 5, 5], keypoints: [1, 1, 2] }],
      categories: [{ id: 1, name: 'person', keypoints: ['head'] }],
    });
    const plan = planImport({ mode: 'bbox', labels: [] }, targets, dataset, options);
    expect(plan.labels[0].keypoints).toEqual([]);
    expect(plan.summary.keypointsIgnored).toBe(1);
    expect(plan.writes[0].annotations[0].keypoints).toEqual({});
  });

  it('extends an existing label with the file\'s keypoints and skeleton in a keypoint project', () => {
    const person: Label = { id: 'p', name: 'person', color: '#ff0000', keypoints: [{ id: 'h', name: 'head' }], skeleton: [] };
    const dataset = parseDataset({
      images: [{ id: 1, file_name: '1.jpg' }],
      annotations: [{ image_id: 1, category_id: 1, bbox: [0, 0, 5, 5], keypoints: [1, 1, 2, 3, 3, 1] }],
      categories: [{ id: 1, name: 'person', keypoints: ['head', 'foot'], skeleton: [[1, 2]] }],
    });
    const plan = planImport({ mode: 'keypoint', labels: [person] }, targets, dataset, options);
    const label = plan.labels[0];
    expect(label.keypoints.map((k) => k.name)).toEqual(['head', 'foot']);
    expect(label.skeleton).toEqual([['h', label.keypoints[1].id]]);
    expect(plan.summary.extendedLabels).toEqual(['person']);
    const [a] = plan.writes[0].annotations;
    expect(a.keypoints).toEqual({ h: { x: 1, y: 1, v: 2 }, [label.keypoints[1].id]: { x: 3, y: 3, v: 1 } });
    expect(a.bbox).toEqual({ x: 0, y: 0, w: 5, h: 5 });
  });

  it('applies image labels and status from a native export', () => {
    const dataset = parseDataset({
      labels: [{ id: 'L', name: 'bottle', color: '#000000', keypoints: [], skeleton: [] }, { id: 'T', name: 'blurry', color: '#000000', keypoints: [], skeleton: [] }],
      images: [{ filename: '1.jpg', folder: 'val/a', imageLabels: ['T'], status: 'done', annotations: [] }],
    });
    const plan = planImport({ mode: 'bbox', labels: [bottle] }, targets, dataset, options);
    expect(plan.summary.newLabels).toEqual(['blurry']);
    expect(plan.writes[0]).toEqual({ imageId: 'a', annotations: [], status: 'done', imageLabels: [plan.labels[1].id] });
  });
});
