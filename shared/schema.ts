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
export type AutoBoxModel = {
  trainedAt: string;
  trainImages: number;
  epochs: number;
  bestMap50: number | null;
  classIds: string[];
  /** The label names behind `classIds` when trained, which let another project use the model. Older files lack it. */
  classNames?: string[];
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
