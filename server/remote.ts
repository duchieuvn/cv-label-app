import { randomUUID } from 'node:crypto';
import type { Annotation, AutoBoxCheckpoint, Label, ProjectMode, RemoteBundle, RemoteBundleOptions, RemoteResults } from '../shared/schema.ts';
import {
  baseWeights,
  chooseTrainingStart,
  chooseValidation,
  keypointCount,
  taskOf,
  toPredictedAnnotations,
  toYoloBoxes,
  toYoloPose,
  type ModelClass,
  type PredictedBox,
} from './autobox.ts';
import type { AutoBoxImage } from './store.ts';

/** The name used inside a bundle and on the server for a checkpoint it continues from. */
export const START_WEIGHTS = 'start.pt';

export type BundleFile = { entry: string; image: AutoBoxImage };
export type BundlePlan = {
  record: RemoteBundle;
  /** The worker's job file, with paths relative to the bundle root. */
  job: Record<string, unknown>;
  /** Image files to add, one per training image and target. */
  files: BundleFile[];
};

const extensionOf = (image: AutoBoxImage) => {
  const name = image.relPath ?? image.storedName;
  const dot = name.lastIndexOf('.');
  return dot > 0 && dot >= name.length - 6 ? name.slice(dot).toLowerCase() : '';
};

/**
 * Everything a remote training run needs, and nothing else: the labelled images with their labels, and a job
 * file the worker runs in standalone mode. The rest of the dataset stays here, where the new checkpoint draws
 * afterwards. The record is what the app keeps, so that the results can be mapped back to label and keypoint
 * ids however the project changes meanwhile.
 */
export function planBundle(
  project: { id: string; name: string; mode: ProjectMode; labels: Label[] },
  training: AutoBoxImage[],
  options: RemoteBundleOptions,
  checkpoint: AutoBoxCheckpoint | null,
): BundlePlan {
  const task = taskOf(project.mode);
  const { continued, classes } = chooseTrainingStart(checkpoint, project.labels);
  const slots = task === 'pose' ? keypointCount(classes) : 0;
  const validation = chooseValidation(training.map((i) => i.id));
  // A checkpoint fixes the architecture, so its size wins over the option.
  const modelSize = continued && checkpoint ? (checkpoint.modelSize ?? 'n') : options.model;
  const files: BundleFile[] = [];
  const add = (prefix: string, image: AutoBoxImage, index: number) => {
    const entry = `images/${prefix}${String(index).padStart(7, '0')}${extensionOf(image)}`;
    files.push({ entry, image });
    return entry;
  };

  const record: RemoteBundle = {
    id: randomUUID(),
    projectId: project.id,
    createdAt: new Date().toISOString(),
    task,
    modelSize,
    imgsz: options.imgsz,
    confidence: options.confidence,
    classIds: classes.map((l) => l.id),
    classNames: classes.map((l) => l.name),
    keypointIds: classes.map((l) => l.keypoints.map((k) => k.id)),
    checkpoint: continued && checkpoint ? checkpoint.id : null,
  };

  const job = {
    bundleId: record.id,
    project: project.name,
    task,
    kptShape: task === 'pose' ? [slots, 3] : null,
    weights: continued ? START_WEIGHTS : baseWeights(task, modelSize),
    imgsz: options.imgsz,
    maxEpochs: options.epochs,
    patience: options.patience,
    batch: options.batch,
    confidence: options.confidence,
    train: true,
    classes: classes.map((l) => l.name),
    training: training.map((image, index) => ({
      path: add('t', image, index),
      width: image.width,
      height: image.height,
      split: validation.has(image.id) ? 'val' : 'train',
      boxes: task === 'pose' ? toYoloPose(image.annotations, classes, image, slots) : toYoloBoxes(image.annotations, classes, image),
    })),
    // The server only trains; drawing happens here with the checkpoint it returns.
    targets: [],
  };
  return { record, job, files };
}

export type ResultsWrite = { imageId: string; annotations: Annotation[] };

/**
 * Turns a results file into annotations for the project as it is now. A training run brings none back, but
 * the format allows them. Classes and keypoint slots are the bundle's, so a label or keypoint deleted since
 * has nowhere to go and its predictions are dropped. An image not in the project, or with another size, is
 * dropped too; the store refuses edited ones.
 */
export function convertResults(
  bundle: RemoteBundle,
  results: RemoteResults,
  labels: Label[],
  sizes: Map<string, { width: number; height: number }>,
): { writes: ResultsWrite[]; dropped: number } {
  const classes: ModelClass[] = bundle.classIds.map((id, i) => ({ id, name: bundle.classNames[i], keypointIds: bundle.keypointIds[i] ?? [] }));
  const seen = new Set<string>();
  let dropped = 0;
  const writes: ResultsWrite[] = [];
  for (const prediction of results.predictions) {
    const size = sizes.get(prediction.id);
    if (!size || seen.has(prediction.id)) {
      dropped++;
      continue;
    }
    seen.add(prediction.id);
    const boxes: PredictedBox[] = prediction.boxes.map((b) => ({ ...b, keypoints: b.keypoints?.map(([x, y, c]) => [x, y, c]) }));
    let annotations: Annotation[];
    try {
      annotations = toPredictedAnnotations(boxes, classes, labels, size, bundle.task);
    } catch {
      dropped++;
      continue;
    }
    writes.push({ imageId: prediction.id, annotations });
  }
  return { writes, dropped };
}
