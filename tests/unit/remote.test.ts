import { describe, expect, it } from 'vitest';
import type { AutoBoxCheckpoint, Label, RemoteBundle, RemoteResults } from '../../shared/schema.ts';
import { remoteBundleSchema } from '../../shared/schema.ts';
import { toYoloPose } from '../../server/autobox.ts';
import { convertResults, planBundle, START_WEIGHTS } from '../../server/remote.ts';
import type { AutoBoxImage } from '../../server/store.ts';

const car: Label = { id: 'car', name: 'car', color: '#00ff00', keypoints: [], skeleton: [] };
const person: Label = {
  id: 'person',
  name: 'person',
  color: '#ff0000',
  keypoints: [
    { id: 'head', name: 'head' },
    { id: 'foot', name: 'foot' },
  ],
  skeleton: [['head', 'foot']],
};

const image = (id: string, annotations: AutoBoxImage['annotations'] = []): AutoBoxImage => ({
  id,
  projectId: 'p',
  width: 100,
  height: 50,
  annotations,
  storedName: `${id}.png`,
  relPath: null,
  sourcePath: null,
});
const options = { model: 'm' as const, epochs: 20, patience: 5, imgsz: 640, batch: 16, confidence: 0.4 };

describe('toYoloPose', () => {
  it('writes x y v per slot in label order, padded to the dataset count', () => {
    const lines = toYoloPose(
      [
        { id: 'a', labelId: 'person', bbox: null, keypoints: { foot: { x: 50, y: 25, v: 1 } } },
        { id: 'b', labelId: 'car', bbox: { x: 0, y: 0, w: 10, h: 10 }, keypoints: {} },
      ],
      [car, person],
      { width: 100, height: 50 },
      3,
    );
    expect(lines[0].cls).toBe(1);
    expect(lines[0].keypoints).toEqual([0, 0, 0, 0.5, 0.5, 1, 0, 0, 0]);
    expect(lines[1]).toMatchObject({ cls: 0, keypoints: [0, 0, 0, 0, 0, 0, 0, 0, 0] });
  });
});

describe('planBundle', () => {
  const training = Array.from({ length: 12 }, (_, i) => image(`t${i}`, [{ id: 'a', labelId: 'car', bbox: { x: 10, y: 10, w: 20, h: 20 }, keypoints: {} }]));

  it('bundles only the labelled images, with relative paths and the base weights of the chosen size', () => {
    const plan = planBundle({ id: 'p', name: 'cars', mode: 'bbox', labels: [car, person] }, training, options, null);
    expect(plan.files).toHaveLength(12);
    expect(plan.files[0].entry).toBe('images/t0000000.png');
    expect(plan.job).toMatchObject({ task: 'detect', kptShape: null, weights: 'yolov8m.pt', imgsz: 640, maxEpochs: 20, patience: 5, confidence: 0.4, classes: ['car', 'person'], train: true });
    const job = plan.job as { training: { path: string; split: string; boxes: unknown[] }[]; targets: unknown[] };
    expect(job.training[0]).toMatchObject({ path: 'images/t0000000.png', boxes: [{ cls: 0 }] });
    expect(new Set(job.training.map((t) => t.split))).toEqual(new Set(['train', 'val']));
    // The server only trains, so the rest of the dataset stays home.
    expect(job.targets).toEqual([]);
    expect(plan.record).toMatchObject({ projectId: 'p', task: 'detect', modelSize: 'm', classIds: ['car', 'person'], keypointIds: [[], ['head', 'foot']], checkpoint: null });
  });

  it('starts a keypoint project from the pose weights, medium by default', () => {
    const { model: defaultModel } = remoteBundleSchema.parse({});
    expect(defaultModel).toBe('m');
    const plan = planBundle({ id: 'p', name: 'people', mode: 'keypoint', labels: [person] }, training.map((t) => ({ ...t, annotations: [{ id: 'a', labelId: 'person', bbox: null, keypoints: { head: { x: 5, y: 5, v: 2 } } }] })), { ...options, model: defaultModel }, null);
    expect(plan.job).toMatchObject({ task: 'pose', weights: 'yolov8m-pose.pt' });
  });

  it('continues from a checkpoint with its own size and class order, and pose bundles carry kpt_shape', () => {
    const checkpoint: AutoBoxCheckpoint = {
      id: 'c1',
      projectId: 'p',
      projectName: 'cars',
      trainedAt: 't',
      trainImages: 5,
      epochs: 3,
      bestMap50: 0.5,
      classIds: ['person', 'car'],
      classNames: ['person', 'car'],
      modelSize: 's',
      canContinue: true,
      missingLabels: [],
    };
    const plan = planBundle({ id: 'p', name: 'poses', mode: 'keypoint', labels: [car, person] }, training, options, checkpoint);
    expect(plan.job).toMatchObject({ task: 'pose', kptShape: [2, 3], weights: START_WEIGHTS, classes: ['person', 'car'] });
    expect(plan.record).toMatchObject({ modelSize: 's', checkpoint: 'c1', classIds: ['person', 'car'], keypointIds: [['head', 'foot'], []] });
  });
});

describe('convertResults', () => {
  const bundle: RemoteBundle = {
    id: 'b',
    projectId: 'p',
    createdAt: 't',
    task: 'pose',
    modelSize: 'm',
    imgsz: 640,
    confidence: 0.5,
    classIds: ['person', 'car'],
    classNames: ['person', 'car'],
    keypointIds: [['head', 'foot'], []],
    checkpoint: null,
  };
  const results = (predictions: RemoteResults['predictions']): RemoteResults => ({
    bundleId: 'b',
    task: 'pose',
    trainedAt: 't',
    trainImages: 10,
    epochs: 4,
    bestMap50: 0.9,
    stoppedEarly: true,
    predictions,
    skipped: [],
  });
  const sizes = new Map([
    ['x', { width: 100, height: 50 }],
    ['y', { width: 100, height: 50 }],
  ]);

  it('maps slots to keypoint ids, keeps confident keypoints only, clamps, and drops what no longer fits', () => {
    const { writes, dropped } = convertResults(
      bundle,
      results([
        {
          id: 'x',
          boxes: [
            { cls: 0, x: 1, y: 1, w: 10, h: 10, keypoints: [[5, 5, 0.9], [120, 3, 0.8]] },
            { cls: 0, x: 1, y: 1, w: 10, h: 10, keypoints: [[5, 5, 0.1], [6, 6, 0.2]] },
            { cls: 1, x: 1, y: 1, w: 10, h: 10 },
          ],
        },
        { id: 'y', boxes: [] },
        { id: 'not-in-project', boxes: [] },
        { id: 'x', boxes: [] },
      ]),
      [car, person],
      sizes,
    );
    expect(dropped).toBe(2);
    expect(writes).toHaveLength(2);
    expect(writes[0].imageId).toBe('x');
    // Keypoints only in a keypoint project; the car box has none, so it is dropped; the unsure person is dropped too.
    expect(writes[0].annotations).toHaveLength(1);
    expect(writes[0].annotations[0]).toMatchObject({ labelId: 'person', bbox: null, keypoints: { head: { x: 5, y: 5, v: 2 }, foot: { x: 100, y: 3, v: 2 } } });
    expect(writes[1]).toEqual({ imageId: 'y', annotations: [] });
  });

  it('drops predictions for a label or keypoint deleted since the bundle was made', () => {
    const headless: Label = { ...person, keypoints: [{ id: 'foot', name: 'foot' }], skeleton: [] };
    const { writes } = convertResults(
      bundle,
      results([{ id: 'x', boxes: [{ cls: 0, x: 1, y: 1, w: 10, h: 10, keypoints: [[5, 5, 0.9], [6, 6, 0.9]] }, { cls: 1, x: 1, y: 1, w: 10, h: 10 }] }]),
      [headless],
      sizes,
    );
    expect(writes[0].annotations).toHaveLength(1);
    expect(writes[0].annotations[0].keypoints).toEqual({ foot: { x: 6, y: 6, v: 2 } });
  });

  it('writes boxes for a detect bundle', () => {
    const { writes } = convertResults(
      { ...bundle, task: 'detect' },
      { ...results([{ id: 'x', boxes: [{ cls: 1, x: 95, y: 45, w: 10, h: 10 }] }]), task: 'detect' },
      [car, person],
      sizes,
    );
    expect(writes[0].annotations[0]).toMatchObject({ labelId: 'car', bbox: { x: 95, y: 45, w: 5, h: 5 }, keypoints: {} });
  });
});
