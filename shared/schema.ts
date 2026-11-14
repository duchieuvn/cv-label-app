import { z } from 'zod';

const id = z.string().min(1).max(100);
const name = z.string().trim().min(1).max(200);

export const labelSchema = z
  .object({
    id,
    name,
    color: z.string().regex(/^#[0-9a-fA-F]{6}$/),
    keypoints: z.array(z.object({ id, name })).default([]),
    skeleton: z.array(z.tuple([id, id])).default([]),
  })
  .superRefine((label, ctx) => {
    const ids = new Set(label.keypoints.map((k) => k.id));
    const names = new Set(label.keypoints.map((k) => k.name));
    if (ids.size !== label.keypoints.length) {
      ctx.addIssue({ code: 'custom', message: `Label "${label.name}" has duplicate keypoint ids` });
    }
    if (names.size !== label.keypoints.length) {
      ctx.addIssue({ code: 'custom', message: `Label "${label.name}" has duplicate keypoint names` });
    }
    for (const [a, b] of label.skeleton) {
      if (a === b || !ids.has(a) || !ids.has(b)) {
        ctx.addIssue({ code: 'custom', message: `Label "${label.name}" has an invalid skeleton edge` });
      }
    }
  });

export const labelsSchema = z.array(labelSchema).superRefine((labels, ctx) => {
  if (new Set(labels.map((l) => l.id)).size !== labels.length) {
    ctx.addIssue({ code: 'custom', message: 'Duplicate label ids' });
  }
  if (new Set(labels.map((l) => l.name)).size !== labels.length) {
    ctx.addIssue({ code: 'custom', message: 'Label names must be unique' });
  }
});

export const bboxSchema = z.object({
  x: z.number(),
  y: z.number(),
  w: z.number().positive(),
  h: z.number().positive(),
});

export const placedKeypointSchema = z.object({
  x: z.number(),
  y: z.number(),
  v: z.union([z.literal(1), z.literal(2)]),
});

export const annotationSchema = z.object({
  id,
  labelId: id,
  bbox: bboxSchema.nullable(),
  keypoints: z.record(z.string(), placedKeypointSchema),
});

export const imageStatusSchema = z.enum(['todo', 'done']);

/** What a project annotates, fixed when it is created. Each mode's UI is described in client/src/modes.ts. */
export const projectModeSchema = z.enum(['bbox', 'keypoint']);
export const createProjectSchema = z.object({ name, mode: projectModeSchema.default('bbox'), labels: labelsSchema.default([]) });
export const updateProjectSchema = z.object({ name: name.optional(), labels: labelsSchema.optional() });
export const saveAnnotationsSchema = z.object({
  annotations: z.array(annotationSchema).optional(),
  status: imageStatusSchema.optional(),
  imageLabels: z.array(id).optional(),
});
export const createSourceSchema = z.object({ path: z.string().min(1).max(4096) });
export const startScanSchema = z.object({ includeExcluded: z.boolean().default(false) });
/** What a remote training bundle asks the server to do. See specs.md section 10, "Remote training". */
export const remoteBundleSchema = z.object({
  /** YOLOv8 size letter. Ignored when continuing from a checkpoint, which fixes the architecture. */
  model: z.enum(['n', 's', 'm', 'l', 'x']).default('m'),
  epochs: z.coerce.number().int().min(1).max(1000).default(300),
  patience: z.coerce.number().int().min(1).max(1000).default(50),
  imgsz: z.coerce.number().int().min(320).max(1280).multipleOf(32).default(640),
  batch: z.coerce.number().int().min(1).max(256).default(16),
  confidence: z.coerce.number().min(0.01).max(0.99).default(0.65),
  /** A checkpoint id from the project's list to continue from. Only ever matched against that list. */
  checkpoint: z.string().min(1).max(200).optional(),
});

/** Starts a one-click remote run. The password is used for one SSH login and is never stored or returned. */
export const startRemoteSchema = remoteBundleSchema.extend({
  target: z.string().trim().min(3).max(300).regex(/^[^\s/]+@[^\s@/]+$/, 'Enter the server as user@host'),
  password: z.string().max(1000).default(''),
  /** Upload the images as they are instead of resized to the training size. */
  keepOriginals: z.boolean().default(false),
});

const keypointResult = z.tuple([z.number().finite(), z.number().finite(), z.number().finite()]);
/** What the worker writes as results.json in standalone mode. Coordinates are image pixels of the original image. */
export const remoteResultsSchema = z.object({
  bundleId: z.string().min(1).max(100),
  task: z.enum(['detect', 'pose']),
  trainedAt: z.string().min(1),
  trainImages: z.number().int().min(0),
  epochs: z.number().int().min(0),
  bestMap50: z.number().finite().nullable(),
  stoppedEarly: z.boolean().default(false),
  predictions: z.array(
    z.object({
      id: z.string().min(1).max(100),
      boxes: z.array(
        z.object({
          cls: z.number().int().min(0),
          x: z.number().finite(),
          y: z.number().finite(),
          w: z.number().finite(),
          h: z.number().finite(),
          keypoints: z.array(keypointResult).optional(),
        }),
      ),
    }),
  ),
  skipped: z.array(z.object({ path: z.string(), reason: z.string() })).default([]),
});

export const startAutoBoxSchema = z.object({
  mode: z.enum(['train', 'predict']).default('train'),
  confidence: z.number().min(0.01).max(0.99).optional(),
  /** A checkpoint id from the project's list, or the base weights' name. Only ever matched against that list. */
  checkpoint: z.string().min(1).max(100).optional(),
  patience: z.number().int().min(1).max(1000).optional(),
});

const folderSelector = z.object({
  folder: z.string().min(1),
  recursive: z.boolean(),
  except: z.array(id).default([]),
});
export const importOptionsSchema = z.object({
  createLabels: z.boolean().default(true),
  overwrite: z.boolean().default(false),
  markDone: z.boolean().default(false),
  dryRun: z.boolean().default(false),
});

export const bulkUpdateSchema = z
  .object({
    selector: z.union([folderSelector, z.object({ imageIds: z.array(id) })]),
    addLabels: z.array(id).default([]),
    removeLabels: z.array(id).default([]),
    status: imageStatusSchema.optional(),
  })
  .superRefine((body, ctx) => {
    if (body.addLabels.some((l) => body.removeLabels.includes(l))) {
      ctx.addIssue({ code: 'custom', message: 'A label cannot be both added and removed' });
    }
  });

/**
 * A path inside a folder, with `/` separators. It can never leave the folder it is joined to: no empty, `.` or `..`
 * segments, no leading `/`, and no `\`, which some systems treat as a separator.
 */
export const isSafeRelPath = (path: string) =>
  !path.includes('\\') && !path.includes('\0') && path.split('/').every((s) => s !== '' && s !== '.' && s !== '..');
const relPath = z.string().min(1).max(4096).refine(isSafeRelPath, 'Invalid relative path');

/** The pseudo-source key of uploaded images in a project bundle. Their `relPath` is the original filename. */
export const BUNDLE_UPLOADS = 'uploads';
export const BUNDLE_KIND = 'labelapp-project';
export const BUNDLE_VERSION = 1;

/** A project copied to another computer, without image files. See specs.md section 13. */
export const projectBundleSchema = z.object({
  kind: z.literal(BUNDLE_KIND),
  version: z.literal(BUNDLE_VERSION),
  exportedAt: z.string().max(100),
  project: z.object({ name, mode: projectModeSchema, labels: labelsSchema }),
  sources: z.array(
    z.object({
      key: id,
      name: name.refine((n) => !n.includes('/'), 'Folder names cannot contain "/"'),
      originalPath: z.string().max(4096),
      exclusions: z.array(relPath).default([]),
    }),
  ),
  images: z.array(
    z.object({
      source: id,
      relPath,
      filename: z.string().min(1).max(255),
      width: z.number().int().positive(),
      height: z.number().int().positive(),
      status: imageStatusSchema,
      annotations: z.array(annotationSchema),
      imageLabels: z.array(id).default([]),
      /** The stored flag. It counts as an unreviewed prediction only while the image is `todo`. */
      autoBoxed: z.boolean().default(false),
    }),
  ),
});

/** Where each bundle folder is on this computer, by source key. Null or missing leaves it out. */
export const openBundleSchema = z.object({
  name: name.optional(),
  paths: z.record(z.string(), z.string().max(4096).nullable()).default({}),
});

export type ProjectMode = z.infer<typeof projectModeSchema>;
export type Label = z.infer<typeof labelSchema>;
export type BBox = z.infer<typeof bboxSchema>;
export type PlacedKeypoint = z.infer<typeof placedKeypointSchema>;
export type Annotation = z.infer<typeof annotationSchema>;
export type ImageStatus = z.infer<typeof imageStatusSchema>;
export type BulkUpdate = z.infer<typeof bulkUpdateSchema>;
export type BulkResult = { matched: number; changed: string[] };
export type FileState = 'ok' | 'missing' | 'changed';

/** What an annotation import did, or would do for a dry run. See specs.md section 12. */
export type ImportSummary = {
  format: 'coco' | 'labelapp';
  imagesInFile: number;
  matched: number;
  unmatched: number;
  ambiguous: number;
  /** Matched images skipped because a person already annotated them and overwrite was off. */
  alreadyAnnotated: number;
  /** Matched images skipped because the file's size differs from the project's, so coordinates would not fit. */
  sizeMismatch: number;
  /** Matched images skipped because their file is missing or changed. */
  unavailable: number;
  imagesWritten: number;
  annotations: number;
  /** Annotations skipped for an unknown label or because nothing of them lay inside the image. */
  droppedAnnotations: number;
  /** Annotations whose keypoints were left out because the project is a bbox project. */
  keypointsIgnored: number;
  newLabels: string[];
  /** Existing labels that gained keypoint definitions from the file. */
  extendedLabels: string[];
  warnings: string[];
};

export type ImportResult = { summary: ImportSummary; dryRun: boolean; project: Project };

export type ProjectBundle = z.infer<typeof projectBundleSchema>;
export type OpenBundleOptions = z.infer<typeof openBundleSchema>;

/** What opening a bundle would do, returned by a dry run. */
export type BundlePreview = {
  name: string;
  mode: ProjectMode;
  labelCount: number;
  imageCount: number;
  folders: {
    key: string;
    name: string;
    /** Where the folder was on the computer that exported it, as a hint. Empty for uploads. */
    originalPath: string;
    imageCount: number;
    samples: string[];
    /** The path this request gave, resolved, or null when none was given. */
    path: string | null;
    /** Why that path cannot be used, or null. */
    error: string | null;
    /** How many of up to 20 of the folder's files exist under that path. */
    found: number;
    checked: number;
  }[];
};

export type OpenBundleResult = { project: Project; warnings: string[] };

export type RemoteBundleOptions = z.infer<typeof remoteBundleSchema>;
export type RemoteResults = z.infer<typeof remoteResultsSchema>;

/** What the app remembers about a bundle it issued, which is what maps its results back to labels. */
export type RemoteBundle = {
  id: string;
  projectId: string;
  createdAt: string;
  task: AutoBoxTask;
  modelSize: string;
  imgsz: number;
  confidence: number;
  classIds: string[];
  classNames: string[];
  keypointIds: string[][];
  /** The checkpoint it continues from, if any. */
  checkpoint: string | null;
};

export type RemoteJobPhase = 'bundling' | 'connecting' | 'preparing' | 'installing' | 'uploading' | 'training' | 'downloading' | 'importing' | 'drawing';

/** A one-click remote run, from bundling to the local drawing with the new checkpoint. Status lives in memory, like the local job's. */
export type RemoteJobStatus = {
  state: 'running' | 'done' | 'cancelled' | 'failed';
  phase: RemoteJobPhase;
  bundleId: string | null;
  target: string;
  modelSize: string;
  /** Bytes uploaded so far and the bundle's size. */
  uploaded: number;
  uploadTotal: number;
  gpu: string | null;
  trainImages: number;
  epoch: number;
  bestMap50: number | null;
  summary: RemoteResultsSummary | null;
  /** The checkpoint the run added, once imported. */
  checkpointId: string | null;
  /** What happened to the local drawing afterwards: started, or why not. */
  drawing: string | null;
  error: string | null;
};

/** What importing a bundle's results did. */
export type RemoteResultsSummary = {
  bundleId: string;
  task: AutoBoxTask;
  trainedAt: string;
  trainImages: number;
  epochs: number;
  bestMap50: number | null;
  /** Images the file holds predictions for. */
  predicted: number;
  /** Images that received at least one annotation. */
  boxedImages: number;
  annotations: number;
  /** Predictions dropped: edited since the bundle was made, no longer in the project, or for a deleted label. */
  dropped: number;
  skipped: { path: string; reason: string }[];
  checkpointAdded: boolean;
  checkpointId: string | null;
};

export type Project = {
  id: string;
  name: string;
  mode: ProjectMode;
  labels: Label[];
  createdAt: string;
  imageCount: number;
  doneCount: number;
};

export type ImageMeta = {
  id: string;
  projectId: string;
  filename: string;
  width: number;
  height: number;
  status: ImageStatus;
  annotationCount: number;
  folder: string;
  imageLabels: string[];
  fileState: FileState;
  sourceId: string | null;
  /** Its boxes are predictions nobody has reviewed: a `todo` image not edited since auto boxing. */
  autoBoxed: boolean;
};

export type ImageDetail = ImageMeta & { annotations: Annotation[] };

export type AutoBoxStatus = {
  state: 'running' | 'done' | 'cancelled' | 'failed';
  /** `train` trains a new checkpoint first, and `predict` draws with a saved checkpoint. */
  mode: 'train' | 'predict';
  phase: 'preparing' | 'training' | 'predicting';
  /** Training started from a saved checkpoint rather than the base weights. */
  continued: boolean;
  /** When the checkpoint this job started from or draws with was trained. Null for the base weights. */
  checkpointTrainedAt: string | null;
  confidence: number;
  patience: number;
  trainImages: number;
  valImages: number;
  epoch: number;
  bestMap50: number | null;
  stoppedEarly: boolean;
  predicted: number;
  total: number;
  boxedImages: number;
  boxes: number;
  skipped: number;
  errors: { path: string; reason: string }[];
  error: string | null;
};

/** What is stored about a trained model. `classIds` holds the label id of each model class, fixed when it was trained. */
/** What a model detects: boxes, or boxes with keypoints. A project's mode decides which it can use. */
export type AutoBoxTask = 'detect' | 'pose';

export type AutoBoxModel = {
  trainedAt: string;
  trainImages: number;
  epochs: number;
  bestMap50: number | null;
  classIds: string[];
  /** The label names behind `classIds` when trained, which let another project use the model. Older files lack it. */
  classNames?: string[];
  /** Older files lack it and are `detect`. */
  task?: AutoBoxTask;
  /** Per class, the keypoint ids behind the model's keypoint slots when trained. Pose models only. */
  keypointIds?: string[][];
  /** YOLOv8 size letter. Older files lack it and are `n`. */
  modelSize?: string;
  /** Where it was trained. Older files lack it and are `local`. */
  source?: 'local' | 'remote';
};

export type AutoBoxCheckpoint = AutoBoxModel & {
  /** The handle a request names it by. For another project's checkpoint it is `<projectId>/<checkpoint>`. */
  id: string;
  classNames: string[];
  /** The project that trained it. */
  projectId: string;
  projectName: string;
  /** Training can start from it. It cannot once labels were added or removed, though it can still draw. */
  canContinue: boolean;
  /** Class names this project has no label for. Drawing with it adds them. */
  missingLabels: string[];
};

export type AutoBoxInfo = {
  job: AutoBoxStatus | null;
  /** The project's saved checkpoints, newest first. */
  checkpoints: AutoBoxCheckpoint[];
  /** Checkpoints of other bounding box projects, newest first. Their classes are matched to this project's labels by name. */
  otherCheckpoints: AutoBoxCheckpoint[];
  /** The name of the base weights, which training can start from but which cannot draw. */
  baseCheckpoint: string;
  /** What training starts from unless told otherwise: the newest usable checkpoint, or the base weights. */
  defaultCheckpoint: string;
  /** Used for the base weights, always, and the starting value for a saved checkpoint. */
  defaultPatience: number;
  maxEpochs: number;
  defaultConfidence: number;
  /** Images a run would train on: those with a saved box that is not an unreviewed prediction. */
  trainable: number;
  /** Images a run would draw on: `todo` images with no annotations or only unreviewed predictions. */
  targets: number;
  /** Images currently holding unreviewed predictions. */
  autoBoxed: number;
  minTraining: number;
};

/**
 * Labels in a bbox project cannot define keypoints. The annotation invariants then leave a box as the
 * only thing an annotation there can hold, so nothing else needs checking per mode.
 */
export function findLabelModeError(labels: Label[], mode: ProjectMode): string | null {
  const label = mode === 'bbox' ? labels.find((l) => l.keypoints.length > 0) : undefined;
  return label ? `Label "${label.name}" defines keypoints, which a bounding box project cannot use.` : null;
}

export type ScanStatus = {
  state: 'running' | 'done' | 'cancelled' | 'failed';
  foldersVisited: number;
  added: number;
  skipped: number;
  unchanged: number;
  missing: number;
  changed: number;
  recovered: number;
  excluded: number;
  errorCount: number;
  errors: { path: string; reason: string }[];
};

export type Source = {
  id: string;
  projectId: string;
  name: string;
  path: string;
  imageCount: number;
  scan: ScanStatus | null;
};

export type FolderNode = {
  path: string;
  name: string;
  parent: string | null;
  direct: number;
  total: number;
};

export type UploadResult = {
  uploaded: ImageMeta[];
  rejected: { filename: string; reason: string }[];
};

/** Returns a message for the first violated invariant (specs.md section 3), or null. */
export function findAnnotationError(
  annotations: Annotation[],
  labels: Label[],
  size: { width: number; height: number },
): string | null {
  const TOLERANCE = 0.5;
  const inX = (v: number) => v >= -TOLERANCE && v <= size.width + TOLERANCE;
  const inY = (v: number) => v >= -TOLERANCE && v <= size.height + TOLERANCE;
  const seen = new Set<string>();

  for (const a of annotations) {
    if (seen.has(a.id)) return `Duplicate annotation id ${a.id}`;
    seen.add(a.id);

    const label = labels.find((l) => l.id === a.labelId);
    if (!label) return `Annotation ${a.id} references unknown label ${a.labelId}`;

    const placed = Object.entries(a.keypoints);
    if (!a.bbox && placed.length === 0) return `Annotation ${a.id} has neither a bbox nor keypoints`;

    if (a.bbox) {
      const { x, y, w, h } = a.bbox;
      if (!inX(x) || !inY(y) || !inX(x + w) || !inY(y + h)) {
        return `Annotation ${a.id} has a bbox outside the image`;
      }
    }
    for (const [keypointId, point] of placed) {
      if (!label.keypoints.some((k) => k.id === keypointId)) {
        return `Annotation ${a.id} references unknown keypoint ${keypointId}`;
      }
      if (!inX(point.x) || !inY(point.y)) return `Annotation ${a.id} has a keypoint outside the image`;
    }
  }
  return null;
}

/**
 * Applies a new label schema to stored annotations: placed keypoints whose definition was
 * removed are dropped, as are annotations left with no geometry. Returns the result and the
 * number of keypoints removed.
 */
export function pruneAnnotations(annotations: Annotation[], labels: Label[]): { annotations: Annotation[]; pruned: number } {
  let pruned = 0;
  const result: Annotation[] = [];
  for (const a of annotations) {
    const label = labels.find((l) => l.id === a.labelId);
    const known = new Set(label?.keypoints.map((k) => k.id));
    const entries = Object.entries(a.keypoints);
    const kept = entries.filter(([keypointId]) => known.has(keypointId));
    pruned += entries.length - kept.length;
    if (a.bbox || kept.length > 0) result.push({ ...a, keypoints: Object.fromEntries(kept) });
  }
  return { annotations: result, pruned };
}
