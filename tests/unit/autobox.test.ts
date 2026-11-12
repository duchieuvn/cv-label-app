import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import type { Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Annotation, AutoBoxInfo, AutoBoxStatus, ImageDetail, ImageMeta, Label, Project, UploadResult } from '../../shared/schema.ts';
import { createApp } from '../../server/app.ts';
import { AutoBoxManager, chooseTrainingStart, chooseValidation, toPredictedAnnotations, toYoloBoxes, TRAINING, type YoloBox } from '../../server/autobox.ts';
import { makePng } from '../helpers.ts';

const car: Label = { id: 'car', name: 'car', color: '#00ff00', keypoints: [], skeleton: [] };
const person: Label = { id: 'person', name: 'person', color: '#ff0000', keypoints: [], skeleton: [] };
const posed: Label = {
  ...person,
  keypoints: [
    { id: 'head', name: 'head' },
    { id: 'foot', name: 'foot' },
  ],
};
const box = (id = 'a'): Annotation => ({ id, labelId: 'car', bbox: { x: 10, y: 10, w: 20, h: 20 }, keypoints: {} });
const size = { width: 100, height: 50 };
/** The worker writes six decimals, so differences beyond that are floating point noise. */
const rounded = (boxes: YoloBox[]) => boxes.map(({ cls, cx, cy, w, h }) => ({ cls, cx: +cx.toFixed(6), cy: +cy.toFixed(6), w: +w.toFixed(6), h: +h.toFixed(6) }));

describe('chooseValidation', () => {
  const ids = Array.from({ length: 50 }, (_, i) => `image-${i}`);

  it('holds out a fifth of the images, and at least one', () => {
    const large = Array.from({ length: 2000 }, (_, i) => `image-${i}`);
    expect(chooseValidation(large).size).toBeGreaterThan(300);
    expect(chooseValidation(large).size).toBeLessThan(500);
    expect(chooseValidation(['only', 'two']).size).toBe(1);
  });

  it('keeps at least a tenth for validation and always leaves images to train on', () => {
    for (let n = 2; n < 40; n++) {
      const some = Array.from({ length: n }, (_, i) => `small-${i}`);
      const size = chooseValidation(some).size;
      expect(size).toBeGreaterThanOrEqual(Math.max(1, Math.round(n / 10)));
      expect(size).toBeLessThan(n);
    }
  });

  it('never moves a training image into validation as a project grows', () => {
    const before = ids.slice(0, 30);
    const validationBefore = chooseValidation(before);
    const validationAfter = chooseValidation(ids);
    for (const id of before) {
      if (validationAfter.has(id)) expect(validationBefore.has(id)).toBe(true);
    }
  });

  it('chooses the same images whatever order they are listed in', () => {
    expect([...chooseValidation([...ids].reverse())].sort()).toEqual([...chooseValidation(ids)].sort());
  });
});

describe('chooseTrainingStart', () => {
  const other: Label = { ...car, id: 'other', name: 'other' };

  it('starts over without a saved model', () => {
    expect(chooseTrainingStart(null, [car, person])).toEqual({ continued: false, classes: [car, person] });
  });

  it('continues with the saved class order, even if labels were reordered', () => {
    expect(chooseTrainingStart({ classIds: ['car', 'person'] }, [person, car])).toEqual({ continued: true, classes: [car, person] });
  });

  it('starts over once labels were added, removed, or swapped for the same number of others', () => {
    expect(chooseTrainingStart({ classIds: ['car'] }, [car, person]).continued).toBe(false);
    expect(chooseTrainingStart({ classIds: ['car', 'person'] }, [car]).continued).toBe(false);
    expect(chooseTrainingStart({ classIds: ['car', 'person'] }, [car, other])).toEqual({ continued: false, classes: [car, other] });
  });
});

describe('toYoloBoxes', () => {
  it('normalizes to the center and size, with classes in label order', () => {
    const annotations: Annotation[] = [{ id: 'a', labelId: 'car', bbox: { x: 10, y: 10, w: 20, h: 10 }, keypoints: {} }];
    expect(rounded(toYoloBoxes(annotations, [person, car], size))).toEqual([{ cls: 1, cx: 0.2, cy: 0.3, w: 0.2, h: 0.2 }]);
  });

  it('keeps a keypoint-only object, so it is not learned as background', () => {
    const pose: Annotation = { id: 'a', labelId: 'person', bbox: null, keypoints: { head: { x: 10, y: 10, v: 2 }, foot: { x: 20, y: 30, v: 2 } } };
    expect(rounded(toYoloBoxes([pose], [posed], size))).toEqual([{ cls: 0, cx: 0.15, cy: 0.4, w: 0.1, h: 0.4 }]);
  });

  it('clamps a box that is outside the image by the stored tolerance', () => {
    const edge: Annotation = { id: 'a', labelId: 'car', bbox: { x: -0.5, y: 0, w: 100.5, h: 50.5 }, keypoints: {} };
    expect(rounded(toYoloBoxes([edge], [car], size))).toEqual([{ cls: 0, cx: 0.5, cy: 0.5, w: 1, h: 1 }]);
  });
});

describe('toPredictedAnnotations', () => {
  it('clamps boxes to the image and drops those with no area left', () => {
    const result = toPredictedAnnotations(
      [
        { cls: 0, x: -5, y: 40, w: 20, h: 20 },
        { cls: 0, x: 200, y: 0, w: 10, h: 10 },
      ],
      [car],
      [car],
      size,
    );
    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({ labelId: 'car', bbox: { x: 0, y: 40, w: 15, h: 10 }, keypoints: {} });
  });

  it('drops a class whose label was deleted while the job ran', () => {
    const boxes = [
      { cls: 0, x: 1, y: 1, w: 5, h: 5 },
      { cls: 1, x: 1, y: 1, w: 5, h: 5 },
    ];
    expect(toPredictedAnnotations(boxes, [car, person], [person], size).map((a) => a.labelId)).toEqual(['person']);
  });
});

describe('auto box API', () => {
  let dataDir: string;
  let server: Server;
  let base: string;
  let store: ReturnType<typeof createApp>['store'];
  let project: Project;
  let images: ImageMeta[];

  const send = (path: string, method: string, body?: unknown) =>
    fetch(`${base}${path}`, { method, headers: { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) });
  const info = async (): Promise<AutoBoxInfo> => (await send(`/projects/${project.id}/autobox`, 'GET')).json();
  const start = (body?: object) => send(`/projects/${project.id}/autobox`, 'POST', body);
  const predictMode = (confidence?: number) => start({ mode: 'predict', confidence });
  const finished = async (): Promise<AutoBoxStatus> => {
    let job: AutoBoxStatus | null = null;
    await vi.waitFor(async () => {
      job = (await info()).job;
      expect(job?.state).not.toBe('running');
    });
    return job!;
  };
  const annotate = (id: string, annotations: Annotation[]) => send(`/images/${id}/annotations`, 'PUT', { annotations });
  const get = async (id: string): Promise<ImageDetail> => (await send(`/images/${id}`, 'GET')).json();
  const jobFile = () => JSON.parse(readFileSync(process.env.FAKE_AUTOBOX_COPY!, 'utf8'));

  beforeAll(async () => {
    dataDir = mkdtempSync(join(tmpdir(), 'labelapp-autobox-'));
    process.env.FAKE_AUTOBOX_COPY = join(dataDir, 'job-copy.json');
    const created = createApp(dataDir, { autoBoxCommand: [process.execPath, resolve(import.meta.dirname, '../fake_autobox_worker.ts')] });
    store = created.store;
    await new Promise<void>((done) => {
      server = created.app.listen(0, '127.0.0.1', () => done());
    });
    base = `http://127.0.0.1:${(server.address() as AddressInfo).port}/api`;
  });

  afterAll(() => {
    server.close();
    store.close();
    rmSync(dataDir, { recursive: true, force: true });
  });

  /** Twelve images: the first ten have boxes, and the last two are still empty. */
  beforeEach(async () => {
    project = await (await send('/projects', 'POST', { name: 'auto', labels: [car, person] })).json();
    const form = new FormData();
    for (let i = 0; i < 12; i++) form.append('files', new Blob([new Uint8Array(makePng(100, 50))]), `${i}.png`);
    const response = await fetch(`${base}/projects/${project.id}/images`, { method: 'POST', body: form });
    images = ((await response.json()) as UploadResult).uploaded;
    for (const image of images.slice(0, 10)) await annotate(image.id, [box()]);
  });

  afterEach(() => {
    delete process.env.FAKE_AUTOBOX;
  });

  it('requires enough images with boxes, and something to draw on', async () => {
    await annotate(images[0].id, []);
    const tooFew = await start();
    expect(tooFew.status).toBe(422);
    expect((await tooFew.json()).error).toContain('at least 10');

    await annotate(images[0].id, [box()]);
    for (const image of images.slice(10)) await annotate(image.id, [box()]);
    expect((await start()).status).toBe(422);
  });

  it('trains on saved boxes, and draws on the rest without marking them done', async () => {
    expect(await info()).toMatchObject({ trainable: 10, targets: 2, autoBoxed: 0, job: null });
    expect((await start()).status).toBe(202);
    expect(await finished()).toMatchObject({ state: 'done', predicted: 2, total: 2, boxedImages: 2, boxes: 2, stoppedEarly: true, bestMap50: 0.75 });

    const predicted = await get(images[11].id);
    expect(predicted).toMatchObject({ status: 'todo', autoBoxed: true, annotationCount: 1 });
    expect(predicted.annotations[0]).toMatchObject({ labelId: 'car', bbox: { x: 1, y: 1, w: 50, h: 25 } });
    expect(await get(images[0].id)).toMatchObject({ autoBoxed: false, annotationCount: 1 });
    expect(await info()).toMatchObject({ trainable: 10, targets: 2, autoBoxed: 2 });
  });

  it('sends only file locations and normalized boxes, using an absolute path for every image', async () => {
    await start();
    await finished();
    const job = jobFile();
    expect(job).toMatchObject({ weights: 'yolov8n.pt', patience: TRAINING.patience, confidence: TRAINING.confidence, classes: ['car', 'person'] });
    expect(job.training).toHaveLength(10);
    expect(rounded(job.training[0].boxes)).toEqual([{ cls: 0, cx: 0.2, cy: 0.4, w: 0.2, h: 0.4 }]);
    for (const item of [...job.training, ...job.targets]) expect(item.path.startsWith('/')).toBe(true);
  });

  it('never trains on unreviewed predictions, and replaces them on the next run', async () => {
    await start();
    await finished();
    const first = (await get(images[11].id)).annotations[0].id;

    await start();
    await finished();
    expect(jobFile().training).toHaveLength(10);
    expect(jobFile().targets.map((t: { id: string }) => t.id).sort()).toEqual([images[10].id, images[11].id].sort());
    expect((await get(images[11].id)).annotations[0].id).not.toBe(first);
  });

  it('treats a prediction as reviewed once it is edited, or while its image is done', async () => {
    await start();
    await finished();

    await annotate(images[10].id, [box('edited')]);
    expect(await get(images[10].id)).toMatchObject({ autoBoxed: false });
    expect(await info()).toMatchObject({ trainable: 11, targets: 1, autoBoxed: 1 });

    const bulk = (status: string) => send(`/projects/${project.id}/images/bulk`, 'POST', { selector: { imageIds: [images[11].id] }, status });
    await bulk('done');
    expect(await get(images[11].id)).toMatchObject({ autoBoxed: false, status: 'done' });
    expect(await info()).toMatchObject({ trainable: 12, targets: 0, autoBoxed: 0 });

    // Undoing the group action takes the acceptance back.
    await bulk('todo');
    expect(await get(images[11].id)).toMatchObject({ autoBoxed: true, status: 'todo' });
    expect(await info()).toMatchObject({ trainable: 11, targets: 1, autoBoxed: 1 });
  });

  it('leaves done images and images with a human annotation alone', async () => {
    await send(`/images/${images[10].id}/annotations`, 'PUT', { status: 'done' });
    await annotate(images[11].id, [box('human')]);
    expect(await info()).toMatchObject({ trainable: 11, targets: 0 });
    expect((await start()).status).toBe(422);
    expect(await get(images[10].id)).toMatchObject({ annotationCount: 0, status: 'done' });
    expect((await get(images[11].id)).annotations).toEqual([box('human')]);
  });

  it('is only available in bounding box projects', async () => {
    const other: Project = await (await send('/projects', 'POST', { name: 'points', mode: 'keypoint', labels: [posed] })).json();
    const response = await send(`/projects/${other.id}/autobox`, 'POST', {});
    expect(response.status).toBe(422);
    expect((await response.json()).error).toContain('bounding box projects');
  });

  it('does not overwrite an image edited after the job started', async () => {
    await annotate(images[10].id, [box('human')]);
    expect(store.saveAutoBoxes(images[10].id, [box('predicted')])).toBe(false);
    expect((await get(images[10].id)).annotations[0].id).toBe('human');
  });

  it('never writes to an unavailable image', async () => {
    store.setFileState(images[10].id, 'missing');
    expect(store.saveAutoBoxes(images[10].id, [box('predicted')])).toBe(false);
  });

  it('keeps unreviewed predictions out of the annotated export', async () => {
    await start();
    await finished();
    await annotate(images[0].id, [box('kept')]);
    const coco = await (await send(`/projects/${project.id}/export?format=coco&scope=annotated`, 'GET')).json();
    expect(coco.images.map((i: { file_name: string }) => i.file_name)).not.toContain('11.png');
    expect(coco.images.map((i: { file_name: string }) => i.file_name)).toContain('0.png');
  });

  it('removes only unreviewed predictions', async () => {
    await start();
    await finished();
    await annotate(images[10].id, [box('edited')]);
    expect(await (await send(`/projects/${project.id}/autobox/predictions`, 'DELETE')).json()).toEqual({ cleared: 1 });
    expect(await get(images[11].id)).toMatchObject({ annotationCount: 0, autoBoxed: false });
    expect(await get(images[10].id)).toMatchObject({ annotationCount: 1 });
    expect(await get(images[0].id)).toMatchObject({ annotationCount: 1 });
  });

  it('runs one job at a time, and cancelling keeps what was drawn and stops writing', async () => {
    process.env.FAKE_AUTOBOX = 'hang';
    await start();
    await vi.waitFor(async () => expect((await info()).job?.predicted).toBe(1));
    expect((await start()).status).toBe(409);
    expect((await send(`/projects/${project.id}/autobox/predictions`, 'DELETE')).status).toBe(409);

    expect((await send(`/projects/${project.id}/autobox`, 'DELETE')).status).toBe(204);
    expect(await finished()).toMatchObject({ state: 'cancelled', boxedImages: 1 });
    expect((await send(`/projects/${project.id}/autobox`, 'DELETE')).status).toBe(409);
  });

  describe('checkpoints', () => {
    const modelDir = () => join(dataDir, 'models', 'projects', project.id);
    const weightsOf = (id: string) => join(modelDir(), `${id}.pt`);
    const checkpoints = async () => (await info()).checkpoints;
    const deleteCheckpoint = (id: string) => send(`/projects/${project.id}/autobox/checkpoints/${encodeURIComponent(id)}`, 'DELETE');
    const train = async (body?: object) => {
      expect((await start(body)).status).toBe(202);
      return finished();
    };

    it('starts from the base weights, and offers them by name', async () => {
      expect(await info()).toMatchObject({ checkpoints: [], baseCheckpoint: 'yolov8n.pt', defaultCheckpoint: 'yolov8n.pt', defaultPatience: TRAINING.patience, maxEpochs: TRAINING.maxEpochs });
      expect(await train()).toMatchObject({ continued: false, checkpointTrainedAt: null, patience: TRAINING.patience });
      expect(jobFile().weights).toBe('yolov8n.pt');
    });

    it('adds a checkpoint for each training, newest first, without replacing any', async () => {
      await train();
      const [first] = await checkpoints();
      expect(first).toMatchObject({ epochs: 2, bestMap50: 0.75, classIds: ['car', 'person'], canContinue: true });
      expect(readFileSync(weightsOf(first.id), 'utf8')).toBe('fake model');

      await train();
      const list = await checkpoints();
      expect(list).toHaveLength(2);
      expect(list[1].id).toBe(first.id);
      expect(list[0].trainedAt > list[1].trainedAt).toBe(true);
    });

    it('continues from the newest usable checkpoint unless told otherwise', async () => {
      await train();
      const [first] = await checkpoints();
      expect((await info()).defaultCheckpoint).toBe(first.id);
      expect(await train()).toMatchObject({ state: 'done', continued: true, checkpointTrainedAt: first.trainedAt });
      expect(jobFile().weights).toBe(weightsOf(first.id));

      const [second] = await checkpoints();
      expect((await info()).defaultCheckpoint).toBe(second.id);
      await train();
      expect(jobFile().weights).toBe(weightsOf(second.id));
    });

    it('trains from the chosen checkpoint, including an older one or the base weights', async () => {
      await train();
      const [first] = await checkpoints();
      await train();

      expect(await train({ checkpoint: first.id })).toMatchObject({ continued: true, checkpointTrainedAt: first.trainedAt });
      expect(jobFile().weights).toBe(weightsOf(first.id));

      expect(await train({ checkpoint: 'yolov8n.pt' })).toMatchObject({ continued: false, checkpointTrainedAt: null });
      expect(jobFile().weights).toBe('yolov8n.pt');
    });

    it('uses the chosen patience from a checkpoint, and 20 if none is chosen', async () => {
      await train();
      const [first] = await checkpoints();

      expect(await train({ checkpoint: first.id, patience: 5 })).toMatchObject({ patience: 5 });
      expect(jobFile().patience).toBe(5);

      expect(await train({ checkpoint: first.id })).toMatchObject({ patience: TRAINING.patience });
      expect(jobFile().patience).toBe(TRAINING.patience);
    });

    it('never lets the base weights use another patience', async () => {
      const response = await start({ checkpoint: 'yolov8n.pt', patience: 5 });
      expect(response.status).toBe(422);
      expect((await response.json()).error).toContain('yolov8n.pt always uses');
      // With nothing to continue, the default is also the base weights.
      expect((await start({ patience: 5 })).status).toBe(422);
      expect((await info()).job).toBeNull();

      await train({ checkpoint: 'yolov8n.pt' });
      expect(jobFile()).toMatchObject({ weights: 'yolov8n.pt', patience: TRAINING.patience });
    });

    it('rejects a patience outside its range, and patience when only drawing', async () => {
      await train();
      const [first] = await checkpoints();
      for (const patience of [0, -1, 1.5, 'long', 1001]) expect((await start({ checkpoint: first.id, patience })).status).toBe(400);
      expect((await start({ mode: 'predict', patience: 5 })).status).toBe(422);
    });

    it('rejects a checkpoint that does not exist, and treats its name as an id only', async () => {
      await train();
      for (const checkpoint of ['missing', '../../../yolov8n', '/etc/passwd']) {
        expect((await start({ checkpoint })).status).toBe(422);
        expect((await start({ mode: 'predict', checkpoint })).status).toBe(422);
      }
      expect((await deleteCheckpoint('missing')).status).toBe(404);
      expect((await deleteCheckpoint('../../../job-copy')).status).toBe(404);
      expect(existsSync(process.env.FAKE_AUTOBOX_COPY!)).toBe(true);
    });

    it('keeps the class numbers when continuing after labels were reordered', async () => {
      await train();
      await send(`/projects/${project.id}`, 'PATCH', { labels: [person, car] });

      expect(await train()).toMatchObject({ continued: true });
      expect(jobFile().classes).toEqual(['car', 'person']);
      expect(jobFile().training[0].boxes[0].cls).toBe(0);
      expect((await checkpoints())[0].classIds).toEqual(['car', 'person']);
      // The fake worker predicts class 0, which is still car.
      expect((await get(images[11].id)).annotations[0].labelId).toBe('car');
    });

    it('cannot continue from a checkpoint once labels were added, though it can still draw', async () => {
      await train();
      const [old] = await checkpoints();
      const bird: Label = { ...car, id: 'bird', name: 'bird', color: '#0000ff' };
      await send(`/projects/${project.id}`, 'PATCH', { labels: [bird, car, person] });
      expect(await info()).toMatchObject({ defaultCheckpoint: 'yolov8n.pt', checkpoints: [{ id: old.id, canContinue: false }] });

      const refused = await start({ checkpoint: old.id });
      expect(refused.status).toBe(422);
      expect((await refused.json()).error).toContain('cannot continue');

      expect((await predictMode()).status).toBe(202);
      expect(await finished()).toMatchObject({ state: 'done', checkpointTrainedAt: old.trainedAt });
      // Class 0 of the old checkpoint is car, whatever the labels' order is now.
      expect((await get(images[11].id)).annotations[0].labelId).toBe('car');

      expect(await train()).toMatchObject({ continued: false });
      expect(jobFile()).toMatchObject({ weights: 'yolov8n.pt', classes: ['bird', 'car', 'person'] });
      expect(jobFile().training[0].boxes[0].cls).toBe(1);
      expect((await checkpoints())[0]).toMatchObject({ classIds: ['bird', 'car', 'person'], canContinue: true });
    });

    it('draws with the chosen checkpoint, or the newest', async () => {
      await train();
      const [first] = await checkpoints();
      await train();
      const [second] = await checkpoints();

      await predictMode();
      expect(await finished()).toMatchObject({ mode: 'predict', continued: false, checkpointTrainedAt: second.trainedAt });
      expect(jobFile().modelPath).toBe(weightsOf(second.id));

      await start({ mode: 'predict', checkpoint: first.id });
      expect(await finished()).toMatchObject({ mode: 'predict', continued: false, checkpointTrainedAt: first.trainedAt });
      expect(jobFile().modelPath).toBe(weightsOf(first.id));
      expect(await checkpoints()).toHaveLength(2);
    });

    it('applies each checkpoint with its own class numbers', async () => {
      await train();
      const [carFirst] = await checkpoints();
      const bird: Label = { ...car, id: 'bird', name: 'bird', color: '#0000ff' };
      await send(`/projects/${project.id}`, 'PATCH', { labels: [bird, car, person] });
      await train();
      const [birdFirst] = await checkpoints();

      await start({ mode: 'predict', checkpoint: birdFirst.id });
      await finished();
      expect((await get(images[11].id)).annotations[0].labelId).toBe('bird');

      await start({ mode: 'predict', checkpoint: carFirst.id });
      await finished();
      expect((await get(images[11].id)).annotations[0].labelId).toBe('car');
    });

    it('offers other projects\' checkpoints, matched by label name, and adds the labels drawing needs', async () => {
      await train();
      const [own] = await checkpoints();
      const other: Project = await (await send('/projects', 'POST', { name: 'fresh', labels: [] })).json();
      const form = new FormData();
      form.append('files', new Blob([new Uint8Array(makePng(100, 50))]), 'x.png');
      const { uploaded } = (await (await fetch(`${base}/projects/${other.id}/images`, { method: 'POST', body: form })).json()) as UploadResult;
      const otherInfo = async (): Promise<AutoBoxInfo> => (await send(`/projects/${other.id}/autobox`, 'GET')).json();

      const before = await otherInfo();
      expect(before.checkpoints).toEqual([]);
      const shared = before.otherCheckpoints.find((c) => c.id === `${project.id}/${own.id}`)!;
      expect(shared).toMatchObject({ projectId: project.id, projectName: 'auto', classNames: ['car', 'person'], canContinue: false, missingLabels: ['car', 'person'] });
      // Without an explicit pick, drawing never reaches into another project.
      expect((await send(`/projects/${other.id}/autobox`, 'POST', { mode: 'predict' })).status).toBe(422);
      // Nor can another project's checkpoint be deleted from here.
      expect((await send(`/projects/${other.id}/autobox/checkpoints/${encodeURIComponent(shared.id)}`, 'DELETE')).status).toBe(404);

      expect((await send(`/projects/${other.id}/autobox`, 'POST', { mode: 'predict', checkpoint: shared.id })).status).toBe(202);
      await vi.waitFor(async () => expect((await otherInfo()).job?.state).toBe('done'));
      const labels = ((await (await send(`/projects/${other.id}`, 'GET')).json()) as Project).labels;
      expect(labels.map((l) => l.name)).toEqual(['car', 'person']);
      expect((await get(uploaded[0].id)).annotations[0]).toMatchObject({ labelId: labels[0].id, bbox: { x: 1, y: 1, w: 50, h: 25 } });
      // With the labels in place, training can continue from it, and a checkpoint trained here records its own ids.
      const after = (await otherInfo()).otherCheckpoints.find((c) => c.id === shared.id)!;
      expect(after).toMatchObject({ canContinue: true, missingLabels: [] });
      expect(existsSync(join(dataDir, 'models', 'projects', other.id))).toBe(false);
    });

    it('cannot draw with the base weights', async () => {
      await train();
      const response = await start({ mode: 'predict', checkpoint: 'yolov8n.pt' });
      expect(response.status).toBe(422);
      expect((await response.json()).error).toContain('cannot draw');
    });

    it('deletes one checkpoint, keeping the others and drawn boxes', async () => {
      await train();
      const [first] = await checkpoints();
      await train();
      const [second] = await checkpoints();

      expect((await deleteCheckpoint(second.id)).status).toBe(204);
      expect(await checkpoints()).toEqual([first]);
      expect(await info()).toMatchObject({ defaultCheckpoint: first.id, autoBoxed: 2 });
      expect(existsSync(weightsOf(second.id))).toBe(false);
      expect(existsSync(weightsOf(first.id))).toBe(true);
      expect((await start({ checkpoint: second.id })).status).toBe(422);

      expect((await deleteCheckpoint(first.id)).status).toBe(204);
      expect(await info()).toMatchObject({ checkpoints: [], defaultCheckpoint: 'yolov8n.pt', autoBoxed: 2 });
      expect((await predictMode()).status).toBe(422);
    });

    it('will not delete a checkpoint during a job', async () => {
      await train();
      const [first] = await checkpoints();
      process.env.FAKE_AUTOBOX = 'hang';
      await start();
      await vi.waitFor(async () => expect((await info()).job?.predicted).toBe(1));
      expect((await deleteCheckpoint(first.id)).status).toBe(409);
      await send(`/projects/${project.id}/autobox`, 'DELETE');
      await finished();
      expect(existsSync(weightsOf(first.id))).toBe(true);
    });

    it('refuses to draw before a checkpoint exists', async () => {
      const response = await predictMode();
      expect(response.status).toBe(422);
      expect((await response.json()).error).toContain('no trained model');
    });

    it('keeps checkpoints across a restart', async () => {
      await train();
      const list = await checkpoints();
      const restarted = new AutoBoxManager(store, { dataDir, command: ['unused'], fileOf: () => '' });
      expect(restarted.checkpoints({ id: project.id, name: project.name, labels: [car, person] })).toEqual(list);
    });

    it('turns a model saved in the old layout into a checkpoint, once', async () => {
      await train();
      const [{ id: _id, canContinue: _canContinue, projectId: _projectId, projectName: _projectName, missingLabels: _missing, ...model }] = await checkpoints();
      const old = join(dataDir, 'models', 'projects', 'old-project');
      mkdirSync(old, { recursive: true });
      writeFileSync(join(old, 'best.pt'), 'old model');
      writeFileSync(join(old, 'model.json'), JSON.stringify(model));

      for (let start = 0; start < 2; start++) {
        const restarted = new AutoBoxManager(store, { dataDir, command: ['unused'], fileOf: () => '' });
        expect(restarted.checkpoints({ id: 'old-project', name: 'old', labels: [car, person] })).toEqual([{ ...model, id: 'best', projectId: 'old-project', projectName: 'old', canContinue: true, missingLabels: [] }]);
      }
      expect(readFileSync(join(old, 'best.pt'), 'utf8')).toBe('old model');
      expect(readdirSync(old).sort()).toEqual(['best.json', 'best.pt']);
      rmSync(old, { recursive: true });
    });

    it('ignores a checkpoint with a part missing or unreadable', async () => {
      await train();
      const [good] = await checkpoints();
      writeFileSync(join(modelDir(), 'no-weights.json'), '{}');
      writeFileSync(join(modelDir(), 'broken.json'), 'not json');
      copyFileSync(weightsOf(good.id), join(modelDir(), 'broken.pt'));
      expect(await checkpoints()).toEqual([good]);
    });

    it('draws without training, replacing only unreviewed predictions', async () => {
      await train();
      const first = (await get(images[11].id)).annotations[0].id;
      await annotate(images[10].id, [box('edited')]);

      expect((await predictMode()).status).toBe(202);
      expect(await finished()).toMatchObject({ state: 'done', mode: 'predict', trainImages: 0, epoch: 0, predicted: 1, total: 1, boxes: 1 });
      expect(jobFile()).toMatchObject({ train: false, training: [] });
      expect((await get(images[11].id)).annotations[0].id).not.toBe(first);
      expect((await get(images[10].id)).annotations[0].id).toBe('edited');
    });

    it('does not need enough training images once a checkpoint exists', async () => {
      await train();
      for (const image of images.slice(0, 5)) await annotate(image.id, []);
      expect((await info()).trainable).toBe(5);
      expect((await start()).status).toBe(422);
      expect((await predictMode()).status).toBe(202);
      await finished();
    });

    it('applies classes as trained, after labels are reordered and deleted', async () => {
      await train();
      await send(`/projects/${project.id}`, 'PATCH', { labels: [person, car] });
      await predictMode();
      await finished();
      expect((await get(images[11].id)).annotations[0].labelId).toBe('car');

      // Class 0 was car, so once car is gone its predictions have nowhere to go.
      await send(`/projects/${project.id}/autobox/predictions`, 'DELETE');
      for (const image of images.slice(0, 10)) await annotate(image.id, []);
      expect((await send(`/projects/${project.id}`, 'PATCH', { labels: [person] })).status).toBe(200);
      await predictMode();
      expect(await finished()).toMatchObject({ state: 'done', boxes: 0 });
    });

    it('adds nothing when training fails', async () => {
      await train();
      const before = await checkpoints();

      process.env.FAKE_AUTOBOX = 'crash';
      await start();
      await finished();
      expect(await checkpoints()).toEqual(before);
    });

    it('deletes every checkpoint with its project', async () => {
      await train();
      await train();
      expect(readdirSync(modelDir()).filter((name) => name.endsWith('.pt'))).toHaveLength(2);
      await send(`/projects/${project.id}`, 'DELETE');
      expect(existsSync(modelDir())).toBe(false);
    });
  });

  describe('confidence', () => {
    it('uses the default, or the value given for either mode', async () => {
      expect((await info()).defaultConfidence).toBe(TRAINING.confidence);
      await start();
      expect(await finished()).toMatchObject({ confidence: TRAINING.confidence });

      await start({ confidence: 0.3 });
      expect(await finished()).toMatchObject({ confidence: 0.3 });
      expect(jobFile().confidence).toBe(0.3);

      await predictMode(0.9);
      expect(await finished()).toMatchObject({ mode: 'predict', confidence: 0.9 });
      expect(jobFile().confidence).toBe(0.9);
    });

    it('rejects a value outside 0 to 1, and an unknown mode', async () => {
      expect((await start({ confidence: 0 })).status).toBe(400);
      expect((await start({ confidence: 1.5 })).status).toBe(400);
      expect((await start({ confidence: 'high' })).status).toBe(400);
      expect((await start({ mode: 'other' })).status).toBe(400);
      expect((await info()).job).toBeNull();
    });
  });

  it('reports a worker error, and a crash with its output', async () => {
    process.env.FAKE_AUTOBOX = 'error';
    await start();
    expect(await finished()).toMatchObject({ state: 'failed', error: 'no ultralytics here' });

    process.env.FAKE_AUTOBOX = 'crash';
    await start();
    expect(await finished()).toMatchObject({ state: 'failed', error: 'worker exploded' });
    expect(await get(images[11].id)).toMatchObject({ annotationCount: 0 });
  });

  it('explains a missing Python instead of failing silently', async () => {
    const other = createApp(mkdtempSync(join(dataDir, 'nopython-')), { python: '/nonexistent/python3' });
    const listener = await new Promise<Server>((done) => {
      const s = other.app.listen(0, '127.0.0.1', () => done(s));
    });
    const url = `http://127.0.0.1:${(listener.address() as AddressInfo).port}/api`;
    const created: Project = other.store.createProject('p', [car]);
    const form = new FormData();
    for (let i = 0; i < 11; i++) form.append('files', new Blob([new Uint8Array(makePng(100, 50))]), `${i}.png`);
    const { uploaded } = (await (await fetch(`${url}/projects/${created.id}/images`, { method: 'POST', body: form })).json()) as UploadResult;
    for (const image of uploaded.slice(0, 10)) other.store.saveImage(image.id, { annotations: [box()] });

    expect((await fetch(`${url}/projects/${created.id}/autobox`, { method: 'POST' })).status).toBe(202);
    await vi.waitFor(async () => {
      const { job } = (await (await fetch(`${url}/projects/${created.id}/autobox`)).json()) as AutoBoxInfo;
      expect(job?.state).toBe('failed');
      expect(job?.error).toContain('LABELAPP_PYTHON');
    });
    listener.close();
    other.store.close();
  });
});
