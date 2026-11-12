import { randomUUID } from 'node:crypto';
import { constants, existsSync, mkdirSync } from 'node:fs';
import { access, readFile, realpath, rename, rm, stat } from 'node:fs/promises';
import { basename, isAbsolute, join, resolve, sep } from 'node:path';
import express, { type NextFunction, type Request, type Response } from 'express';
import multer from 'multer';
import yazl from 'yazl';
import { z } from 'zod';
import {
  bulkUpdateSchema,
  createProjectSchema,
  createSourceSchema,
  findAnnotationError,
  findLabelModeError,
  imageStatusSchema,
  importOptionsSchema,
  saveAnnotationsSchema,
  startAutoBoxSchema,
  startScanSchema,
  updateProjectSchema,
  type AutoBoxCheckpoint,
  type ImageMeta,
  type Label,
  type UploadResult,
} from '../shared/schema.ts';
import { AutoBoxManager, labelsForClasses, MIN_TRAINING_IMAGES, TRAINING } from './autobox.ts';
import { selectImages, selectLabeledImages, toCoco, toImageLabels, toNativeJson, toYolo } from './export.ts';
import { inspectImage } from './imageInfo.ts';
import { ImportFileError, parseDataset, planImport } from './import.ts';
import { ScanManager } from './scan.ts';
import { createStore } from './store.ts';

const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;

class HttpError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function parse<T extends z.ZodType>(schema: T, value: unknown): z.output<T> {
  const result = schema.safeParse(value);
  if (!result.success) {
    const issue = result.error.issues[0];
    const path = issue.path.join('.');
    throw new HttpError(400, path ? `${path}: ${issue.message}` : issue.message);
  }
  return result.data;
}

/** Browsers encode multipart filenames as UTF-8, but multer decodes them as latin1. */
const decodeFilename = (raw: string) => {
  const name = basename(Buffer.from(raw, 'latin1').toString('utf8').replaceAll('\\', '/')).slice(0, 255);
  return name === '.' || name === '..' || name === '' ? 'image' : name;
};

const isLoopback = (host: string) => host === 'localhost' || host === '::1' || host.startsWith('127.');

const isInside = (path: string, root: string) => path === root || path.startsWith(root.endsWith(sep) ? root : `${root}${sep}`);

export type AppOptions = {
  host?: string;
  importRoots?: string[];
  /** The Python that runs auto boxing, which needs the ultralytics package. */
  python?: string;
  /** Replaces the auto box worker command entirely. The job file's path is added as its last argument. */
  autoBoxCommand?: string[];
};

type Checkpoints = { own: AutoBoxCheckpoint[]; other: AutoBoxCheckpoint[] };

/**
 * The checkpoint to draw with, which defaults to the project's newest. Another project's must be named, so a
 * default never reaches across projects. The base weights know COCO's classes, not a project's labels.
 */
function pickCheckpointToDraw({ own, other }: Checkpoints, picked: string | undefined): AutoBoxCheckpoint {
  if (picked === TRAINING.weights) throw new HttpError(422, `${TRAINING.weights} does not know this project's labels, so it cannot draw. Choose a trained checkpoint.`);
  if (picked === undefined && own.length === 0) throw new HttpError(422, 'This project has no trained model yet. Train one first, or choose another project\'s checkpoint.');
  const checkpoint = picked === undefined ? own[0] : [...own, ...other].find((c) => c.id === picked);
  if (!checkpoint) throw new HttpError(422, 'That checkpoint no longer exists.');
  return checkpoint;
}

/** The checkpoint to start training from, which defaults to the project's newest usable one. Null means the base weights. */
function pickCheckpointToTrain({ own, other }: Checkpoints, picked: string | undefined): AutoBoxCheckpoint | null {
  if (picked === TRAINING.weights) return null;
  if (picked === undefined) return own.find((c) => c.canContinue) ?? null;
  const checkpoint = [...own, ...other].find((c) => c.id === picked);
  if (!checkpoint) throw new HttpError(422, 'That checkpoint no longer exists.');
  if (!checkpoint.canContinue) throw new HttpError(422, `Labels were added or removed since that checkpoint was trained, so training cannot continue from it. Start from ${TRAINING.weights}, or draw with it instead.`);
  return checkpoint;
}

export function createApp(dataDir: string, options: AppOptions = {}) {
  const store = createStore(dataDir);
  const scans = new ScanManager(store);
  const host = options.host ?? '127.0.0.1';
  const importRoots = options.importRoots ?? [];
  const imagesRoot = join(dataDir, 'images');
  const tmpDir = join(dataDir, 'tmp');
  mkdirSync(imagesRoot, { recursive: true });
  mkdirSync(tmpDir, { recursive: true });

  const upload = multer({ dest: tmpDir, limits: { fileSize: MAX_UPLOAD_BYTES } });
  const app = express();
  app.disable('x-powered-by');

  // No auth by design (specs.md section 1), so stop other websites driving this localhost API.
  app.use('/api', (req, _res, next) => {
    const origin = req.headers.origin;
    if (origin) {
      let originHost: string | null = null;
      try {
        originHost = new URL(origin).host;
      } catch {}
      if (originHost !== req.headers.host) throw new HttpError(403, 'Cross-origin requests are not allowed');
    }
    next();
  });
  app.use('/api', express.json({ limit: '50mb' }));

  const imageQuery = z.object({
    status: imageStatusSchema.optional(),
    folder: z.string().min(1).optional(),
    recursive: z.enum(['0', '1']).optional(),
    label: z.string().min(1).optional(),
  });

  const requireProject = (id: string) => {
    const project = store.getProject(id);
    if (!project) throw new HttpError(404, 'Project not found');
    return project;
  };
  const requireImage = (id: string) => {
    const image = store.getImage(id);
    if (!image) throw new HttpError(404, 'Image not found');
    return image;
  };

  /** Reading arbitrary folders is only safe for the local user, unless limited to chosen parents. */
  const assertImportAllowed = async (path: string) => {
    if (importRoots.length === 0) {
      if (!isLoopback(host)) throw new HttpError(403, 'Folder import is disabled on this address. Set IMPORT_ROOTS to allow it.');
      return;
    }
    const roots = await Promise.all(importRoots.map((r) => realpath(r).catch(() => null)));
    if (!roots.some((root) => root && isInside(path, root))) {
      throw new HttpError(403, 'This folder is outside the allowed import folders.');
    }
  };

  const resolveFolder = async (input: string) => {
    if (!isAbsolute(input)) throw new HttpError(400, 'Enter an absolute path, starting from the root of the disk.');
    let path: string;
    try {
      path = await realpath(input);
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (code === 'ENOENT') throw new HttpError(400, `Folder does not exist: ${input}`);
      throw new HttpError(400, `Folder cannot be read: ${input}`);
    }
    await assertImportAllowed(path);
    if (!(await stat(path)).isDirectory()) throw new HttpError(400, `Not a folder: ${input}`);
    try {
      await access(path, constants.R_OK | constants.X_OK);
    } catch {
      throw new HttpError(400, `Folder cannot be read: ${input}`);
    }
    return path;
  };

  const requireSource = (id: string) => {
    const source = store.getSource(id);
    if (!source) throw new HttpError(404, 'Folder not found');
    return source;
  };

  /** The file behind an image, wherever it lives. */
  const imagePath = (image: { projectId: string; storedName: string; sourcePath: string | null; relPath: string | null }) =>
    image.sourcePath && image.relPath ? join(image.sourcePath, image.relPath) : join(imagesRoot, image.projectId, image.storedName);

  const autoBox = new AutoBoxManager(store, {
    dataDir,
    command: options.autoBoxCommand ?? [options.python ?? 'python3', resolve(import.meta.dirname, 'autobox_worker.py')],
    fileOf: imagePath,
  });

  app.get('/api/projects', (_req, res) => {
    res.json(store.listProjects());
  });

  app.post('/api/projects', (req, res) => {
    const body = parse(createProjectSchema, req.body);
    const modeError = findLabelModeError(body.labels, body.mode);
    if (modeError) throw new HttpError(422, modeError);
    res.status(201).json(store.createProject(body.name, body.labels, body.mode));
  });

  app.get('/api/projects/:id', (req, res) => {
    res.json(requireProject(req.params.id));
  });

  app.patch('/api/projects/:id', (req, res) => {
    const project = requireProject(req.params.id);
    const body = parse(updateProjectSchema, req.body);
    let prunedKeypoints = 0;
    if (body.labels) {
      const modeError = findLabelModeError(body.labels, project.mode);
      if (modeError) throw new HttpError(422, modeError);
      const result = store.updateLabels(req.params.id, body.labels);
      if (result.blockedBy.length > 0) {
        throw new HttpError(409, `These labels are in use and cannot be deleted: ${result.blockedBy.join(', ')}`);
      }
      prunedKeypoints = result.prunedKeypoints;
    }
    if (body.name) store.renameProject(req.params.id, body.name);
    res.json({ project: store.getProject(req.params.id), prunedKeypoints });
  });

  app.delete('/api/projects/:id', async (req, res) => {
    requireProject(req.params.id);
    for (const source of store.listSources(req.params.id)) scans.cancel(source.id);
    autoBox.cancel(req.params.id);
    autoBox.removeAllCheckpoints(req.params.id);
    store.deleteProject(req.params.id);
    await rm(join(imagesRoot, req.params.id), { recursive: true, force: true });
    res.status(204).end();
  });

  app.get('/api/projects/:id/images', (req, res) => {
    requireProject(req.params.id);
    const query = parse(imageQuery, req.query);
    res.json(store.listImages(req.params.id, { ...query, recursive: query.recursive !== '0' }));
  });

  app.get('/api/projects/:id/folders', (req, res) => {
    requireProject(req.params.id);
    res.json(store.listFolders(req.params.id));
  });

  app.post('/api/projects/:id/images/bulk', (req, res) => {
    const project = requireProject(req.params.id);
    const body = parse(bulkUpdateSchema, req.body);
    const unknown = [...body.addLabels, ...body.removeLabels].find((id) => !project.labels.some((l) => l.id === id));
    if (unknown) throw new HttpError(422, `Unknown label ${unknown}`);
    res.json(store.bulkUpdate(project.id, body));
  });

  /** Checkpoints a project can use: its own, and those of every other bounding box project. */
  const checkpointsFor = (project: { id: string; name: string; labels: Label[] }): Checkpoints => ({
    own: autoBox.checkpoints(project),
    other: autoBox.otherCheckpoints(project, store.listProjects().filter((p) => p.mode === 'bbox')),
  });

  app.get('/api/projects/:id/autobox', (req, res) => {
    const project = requireProject(req.params.id);
    const { own: checkpoints, other } = checkpointsFor(project);
    res.json({
      job: autoBox.get(req.params.id),
      checkpoints,
      otherCheckpoints: other,
      baseCheckpoint: TRAINING.weights,
      defaultCheckpoint: checkpoints.find((c) => c.canContinue)?.id ?? TRAINING.weights,
      defaultPatience: TRAINING.patience,
      maxEpochs: TRAINING.maxEpochs,
      defaultConfidence: TRAINING.confidence,
      ...store.autoBoxCounts(req.params.id),
      minTraining: MIN_TRAINING_IMAGES,
    });
  });

  app.post('/api/projects/:id/autobox', (req, res) => {
    let project = requireProject(req.params.id);
    const { mode, confidence, checkpoint: picked, patience } = parse(startAutoBoxSchema, req.body ?? {});
    if (project.mode !== 'bbox') throw new HttpError(422, 'Auto boxing is only available in bounding box projects.');
    if (autoBox.isAnyRunning()) throw new HttpError(409, 'An auto box job is already running. Only one runs at a time.');
    const { trainable, targets } = store.autoBoxCounts(project.id);
    const saved = checkpointsFor(project);
    const checkpoint = mode === 'predict' ? pickCheckpointToDraw(saved, picked) : pickCheckpointToTrain(saved, picked);
    if (patience !== undefined && (mode === 'predict' || !checkpoint)) {
      throw new HttpError(422, `Patience can only be chosen when training from a saved checkpoint. ${TRAINING.weights} always uses ${TRAINING.patience}.`);
    }
    if (mode === 'train' && trainable < MIN_TRAINING_IMAGES) {
      throw new HttpError(422, `Auto boxing needs at least ${MIN_TRAINING_IMAGES} images with saved boxes. This project has ${trainable}.`);
    }
    if (targets === 0) throw new HttpError(422, 'Every image already has annotations or is done, so there is nothing to draw on.');
    if (mode === 'predict' && checkpoint && checkpoint.projectId !== project.id) {
      // Another project's model predicts classes this project may have no label for yet, so they are added first.
      // The project's own never add any: a label deleted since training stays deleted, and its predictions are dropped.
      const added = labelsForClasses(checkpoint, project.labels);
      if (added.length > 0) {
        store.updateLabels(project.id, [...project.labels, ...added]);
        project = requireProject(project.id);
      }
    }
    const options = { mode, confidence: confidence ?? TRAINING.confidence, patience: patience ?? TRAINING.patience, checkpoint };
    res.status(202).json(autoBox.start(project.id, project.labels, store.autoBoxImages(project.id), options));
  });

  app.delete('/api/projects/:id/autobox', (req, res) => {
    requireProject(req.params.id);
    if (!autoBox.cancel(req.params.id)) throw new HttpError(409, 'No auto box job is running for this project.');
    res.status(204).end();
  });

  app.delete('/api/projects/:id/autobox/checkpoints/:checkpoint', (req, res) => {
    const project = requireProject(req.params.id);
    if (autoBox.isRunning(project.id)) throw new HttpError(409, 'Wait for the auto box job to finish, or cancel it.');
    // Matching against the list means a request never chooses a path.
    // Only the project's own can be deleted here; another project's is deleted from that project.
    const checkpoint = autoBox.checkpoints(project).find((c) => c.id === req.params.checkpoint);
    if (!checkpoint) throw new HttpError(404, 'That checkpoint no longer exists.');
    autoBox.removeCheckpoint(project.id, checkpoint.id);
    res.status(204).end();
  });

  app.delete('/api/projects/:id/autobox/predictions', (req, res) => {
    requireProject(req.params.id);
    if (autoBox.isRunning(req.params.id)) throw new HttpError(409, 'Wait for the auto box job to finish, or cancel it.');
    res.json({ cleared: store.clearAutoBoxes(req.params.id) });
  });

  app.get('/api/projects/:id/sources', (req, res) => {
    requireProject(req.params.id);
    res.json(store.listSources(req.params.id).map((source) => ({ ...source, scan: scans.get(source.id) })));
  });

  app.post('/api/projects/:id/sources', async (req, res) => {
    requireProject(req.params.id);
    const path = await resolveFolder(parse(createSourceSchema, req.body).path);
    // Overlapping sources would list the same file twice.
    const overlap = store.listSources(req.params.id).find((s) => isInside(path, s.path) || isInside(s.path, path));
    if (overlap) throw new HttpError(409, `This folder overlaps ${overlap.path}, which is already in the project.`);
    const source = store.createSource(req.params.id, basename(path) || 'root', path);
    res.status(202).json({ ...source, scan: scans.start(source, { includeExcluded: false }) });
  });

  app.get('/api/sources/:id/scan', (req, res) => {
    requireSource(req.params.id);
    res.json(scans.get(req.params.id));
  });

  app.post('/api/sources/:id/scan', async (req, res) => {
    const source = requireSource(req.params.id);
    const { includeExcluded } = parse(startScanSchema, req.body ?? {});
    await assertImportAllowed(source.path);
    if (scans.isRunning(source.id)) throw new HttpError(409, 'This folder is already being scanned.');
    res.status(202).json(scans.start(source, { includeExcluded }));
  });

  app.delete('/api/sources/:id/scan', (req, res) => {
    requireSource(req.params.id);
    if (!scans.cancel(req.params.id)) throw new HttpError(409, 'No scan is running for this folder.');
    res.status(204).end();
  });

  app.delete('/api/sources/:id', (req, res) => {
    requireSource(req.params.id);
    scans.cancel(req.params.id);
    store.deleteSource(req.params.id);
    res.status(204).end();
  });

  app.post('/api/projects/:id/images', upload.array('files'), async (req, res) => {
    const files = (req.files ?? []) as Express.Multer.File[];
    const result: UploadResult = { uploaded: [], rejected: [] };
    try {
      requireProject(req.params.id as string);
      const projectDir = join(imagesRoot, req.params.id as string);
      mkdirSync(projectDir, { recursive: true });

      for (const file of files) {
        const filename = decodeFilename(file.originalname);
        const info = await inspectImage(file.path).catch(() => null);
        if (!info) {
          result.rejected.push({ filename, reason: 'Not a supported image (JPEG, PNG, WebP, GIF, BMP)' });
          continue;
        }
        const id = randomUUID();
        const storedName = `${id}${info.extension}`;
        await rename(file.path, join(projectDir, storedName));
        const image: ImageMeta = store.addImage({
          id,
          projectId: req.params.id as string,
          filename,
          storedName,
          width: info.width,
          height: info.height,
        });
        result.uploaded.push(image);
      }
    } finally {
      // Files moved into the project no longer exist here, so this only removes rejects.
      await Promise.all(files.map((f) => rm(f.path, { force: true })));
    }
    res.status(201).json(result);
  });

  const flag = z.preprocess((v) => (v === '1' || v === 'true' ? true : v === '0' || v === 'false' ? false : v), z.boolean());
  const importFields = z.object({ createLabels: flag.optional(), overwrite: flag.optional(), markDone: flag.optional(), dryRun: flag.optional() });

  app.post('/api/projects/:id/import', upload.single('file'), async (req, res) => {
    const file = req.file;
    try {
      const project = requireProject(req.params.id as string);
      if (!file) throw new HttpError(400, 'Attach the annotation file as the "file" field.');
      // Form fields arrive as strings, so flags are accepted as 1/0 or true/false.
      const options = parse(importOptionsSchema, parse(importFields, req.body ?? {}));
      let data: unknown;
      try {
        data = JSON.parse(await readFile(file.path, 'utf8'));
      } catch {
        throw new HttpError(400, 'The file is not valid JSON.');
      }
      let dataset;
      try {
        dataset = parseDataset(data);
      } catch (error) {
        if (error instanceof ImportFileError) throw new HttpError(400, error.message);
        throw error;
      }
      const plan = planImport(project, store.listImages(project.id), dataset, options);
      const modeError = findLabelModeError(plan.labels, project.mode);
      if (modeError) throw new HttpError(422, modeError);
      if (!options.dryRun) store.applyImport(project.id, plan.labelsChanged ? plan.labels : null, plan.writes);
      res.json({ summary: plan.summary, dryRun: options.dryRun, project: store.getProject(project.id) });
    } finally {
      if (file) await rm(file.path, { force: true });
    }
  });

  app.get('/api/images/:id', (req, res) => {
    res.json(requireImage(req.params.id));
  });

  app.put('/api/images/:id/annotations', (req, res) => {
    const image = requireImage(req.params.id);
    const body = parse(saveAnnotationsSchema, req.body);
    const { labels } = requireProject(image.projectId);
    if (body.annotations) {
      // The stored dimensions may no longer describe the file, so coordinates cannot be trusted.
      if (image.fileState !== 'ok') throw new HttpError(409, 'This image is unavailable, so its annotations are read-only.');
      const error = findAnnotationError(body.annotations, labels, image);
      if (error) throw new HttpError(422, error);
    }
    if (body.imageLabels) {
      if (new Set(body.imageLabels).size !== body.imageLabels.length) throw new HttpError(422, 'Duplicate image labels');
      const unknown = body.imageLabels.find((id) => !labels.some((l) => l.id === id));
      if (unknown) throw new HttpError(422, `Unknown label ${unknown}`);
    }
    store.saveImage(image.id, body);
    res.json(store.getImage(image.id));
  });

  app.delete('/api/images/:id', async (req, res) => {
    const file = store.getFile(req.params.id);
    if (!file) throw new HttpError(404, 'Image not found');
    if (file.sourceId && file.relPath) {
      // Source folders belong to the user: forget the image, never touch the file.
      store.excludeImage(req.params.id, file.sourceId, file.relPath);
    } else {
      store.deleteImage(req.params.id);
      await rm(imagePath(file), { force: true });
    }
    res.status(204).end();
  });

  app.get('/api/images/:id/file', (req, res, next) => {
    const file = store.getFile(req.params.id);
    if (!file) throw new HttpError(404, 'Image not found');
    // Set from the file's contents: a scanned image can have any name, including "x.html".
    if (file.mime) res.type(file.mime);
    const [root, path] = file.sourcePath && file.relPath ? [file.sourcePath, file.relPath] : [resolve(imagesRoot, file.projectId), file.storedName];
    res.sendFile(
      path,
      { root, maxAge: file.sourceId ? 0 : '1y', immutable: !file.sourceId, headers: { 'X-Content-Type-Options': 'nosniff' } },
      (error) => {
        if (error && !res.headersSent) next(new HttpError(404, 'Image file is not available'));
      },
    );
  });

  const exportQuery = z.object({
    format: z.enum(['json', 'coco', 'yolo', 'image_labels']),
    scope: z.enum(['done', 'annotated', 'labeled']).optional(),
    includeImages: z.enum(['0', '1']).default('0'),
  });

  app.get('/api/projects/:id/export', (req, res) => {
    const project = requireProject(req.params.id);
    const query = parse(exportQuery, req.query);
    const all = store.listImagesWithAnnotations(project.id);
    const sourcePaths = new Map(store.listSources(project.id).map((s) => [s.id, s.path]));
    const baseName = project.name.replace(/[^\p{L}\p{N}_-]+/gu, '_') || 'export';

    const fileOf = (image: (typeof all)[number]) =>
      imagePath({ ...image, sourcePath: image.sourceId ? (sourcePaths.get(image.sourceId) ?? null) : null });
    const available = (image: (typeof all)[number]) => image.fileState === 'ok' && existsSync(fileOf(image));
    const includeImages = query.includeImages === '1';

    const sendZip = (zip: yazl.ZipFile, name: string, warnings: string[]) => {
      if (warnings.length > 0) {
        const text = `These images were left out because their files are unavailable:\n${warnings.join('\n')}\n`;
        zip.addBuffer(Buffer.from(text), 'export_warnings.txt');
      }
      zip.end();
      res.attachment(name);
      zip.outputStream.on('error', () => res.destroy());
      zip.outputStream.pipe(res);
    };

    if (query.format === 'image_labels') {
      if (query.scope === 'annotated') throw new HttpError(400, 'scope for image_labels must be labeled or done');
      const images = selectLabeledImages(all, query.scope === 'done' ? 'done' : 'labeled');
      const output = toImageLabels(project, images);
      const zip = new yazl.ZipFile();
      const warnings: string[] = [];
      let csv = 'image,label\n';
      for (const file of output.files) {
        if (includeImages && !available(file.image)) {
          warnings.push(file.path);
          continue;
        }
        csv += file.rows.join('');
        if (includeImages) zip.addFile(fileOf(file.image), `images/${file.path}`, { compress: false });
      }
      zip.addBuffer(Buffer.from(csv), 'labels.csv');
      zip.addBuffer(Buffer.from(output.classes), 'classes.txt');
      sendZip(zip, `${baseName}_image_labels.zip`, warnings);
      return;
    }

    if (query.scope === 'labeled') throw new HttpError(400, 'scope "labeled" only applies to the image_labels format');
    const images = selectImages(all, query.scope === 'annotated' ? 'annotated' : 'done');

    if (query.format === 'json' || query.format === 'coco') {
      const payload = query.format === 'json' ? toNativeJson(project, images) : toCoco(project, images);
      res.attachment(`${baseName}_${query.format}.json`).json(payload);
      return;
    }

    const yolo = toYolo(project, images);
    const zip = new yazl.ZipFile();
    const warnings: string[] = [];
    zip.addBuffer(Buffer.from(yolo.yaml), 'data.yaml');
    for (const file of yolo.files) {
      if (includeImages && !available(file.image)) {
        warnings.push(file.imageName);
        continue;
      }
      zip.addBuffer(Buffer.from(file.content), file.labelName);
      // Images are already compressed, so storing them saves CPU for no size penalty.
      if (includeImages) zip.addFile(fileOf(file.image), file.imageName, { compress: false });
    }
    sendZip(zip, `${baseName}_yolo.zip`, warnings);
  });

  app.use('/api', (_req, _res, next) => next(new HttpError(404, 'Not found')));

  const clientDir = resolve(import.meta.dirname, '../dist');
  if (existsSync(clientDir)) app.use(express.static(clientDir));

  app.use((error: unknown, _req: Request, res: Response, _next: NextFunction) => {
    if (res.headersSent) {
      res.destroy();
      return;
    }
    if (error instanceof HttpError) {
      res.status(error.status).json({ error: error.message });
    } else if (error instanceof multer.MulterError) {
      const tooLarge = error.code === 'LIMIT_FILE_SIZE';
      res.status(tooLarge ? 413 : 400).json({ error: tooLarge ? 'File is larger than the 100 MB limit' : error.message });
    } else if (error instanceof SyntaxError && 'body' in error) {
      res.status(400).json({ error: 'Invalid JSON' });
    } else {
      console.error(error);
      res.status(500).json({ error: 'Internal server error' });
    }
  });

  return { app, store };
}
