import { spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { createInterface } from "node:readline";
import type { Readable } from "node:stream";
import {
  findAnnotationError,
  type Annotation,
  type AutoBoxCheckpoint,
  type AutoBoxModel,
  type AutoBoxStatus,
  type AutoBoxTask,
  type Label,
  type PlacedKeypoint,
  type ProjectMode,
} from "../shared/schema.ts";
import { nextLabelColor } from "../shared/colors.ts";
import { effectiveBBox } from "./export.ts";
import type { AutoBoxImage, Store } from "./store.ts";

export const MIN_TRAINING_IMAGES = 10;
const VALIDATION_FRACTION = 0.2;
const MIN_VALIDATION_FRACTION = 0.1;
const MAX_LISTED_ERRORS = 100;
const STDERR_TAIL = 2000;
export const TRAINING = {
  imgsz: 640,
  maxEpochs: 300,
  patience: 20,
  confidence: 0.65,
  /** A predicted keypoint scoring below this is left unplaced. */
  keypointConfidence: 0.5,
};

/** What a project's mode trains: boxes, or boxes with keypoints. */
export const taskOf = (mode: ProjectMode): AutoBoxTask => (mode === "keypoint" ? "pose" : "detect");

/** The base weights for a task and YOLOv8 size letter, such as yolov8m-pose.pt. */
export const baseWeights = (task: AutoBoxTask, size = "n") => `yolov8${size}${task === "pose" ? "-pose" : ""}.pt`;

export type YoloBox = {
  cls: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
  /** Pose only: `x y v` per keypoint slot, normalized, padded with zeros to the task's keypoint count. */
  keypoints?: number[];
};
export type PredictedBox = {
  cls: number;
  x: number;
  y: number;
  w: number;
  h: number;
  /** Pose only: `[x, y, confidence]` per keypoint slot, in image pixels. */
  keypoints?: [number, number, number][];
};
/** A model class: the label it stands for and, for pose models, the label's keypoint ids per slot. */
export type ModelClass = { id: string; name?: string; keypointIds?: string[] };

/** The keypoint slots per class: YOLO needs one count for the whole dataset, so shorter labels are padded. */
export const keypointCount = (labels: Label[]) => Math.max(0, ...labels.map((l) => l.keypoints.length));
type Size = { width: number; height: number };
type JobImages = { training: AutoBoxImage[]; targets: AutoBoxImage[] };
type JobOptions = {
  mode: "train" | "predict";
  task: AutoBoxTask;
  confidence: number;
  patience: number;
  /** The checkpoint to start training from or draw with. Null means the base weights. */
  checkpoint: AutoBoxCheckpoint | null;
};

/**
 * Chooses the images held out for early stopping. An image's hash alone decides, so adding images
 * never moves one from training into validation, where a continued model would be scored on images
 * it has already learned. Small projects top up from the next hashes to keep a usable set.
 */
export function chooseValidation(ids: string[]): Set<string> {
  const score = (id: string) =>
    parseInt(createHash("sha1").update(id).digest("hex").slice(0, 8), 16) /
    2 ** 32;
  const scored = ids
    .map((id) => ({ id, score: score(id) }))
    .sort((a, b) => a.score - b.score);
  const below = scored.filter((s) => s.score < VALIDATION_FRACTION).length;
  const minimum = Math.max(1, Math.round(ids.length * MIN_VALIDATION_FRACTION));
  const size = Math.min(Math.max(below, minimum), ids.length - 1);
  return new Set(scored.slice(0, size).map((s) => s.id));
}

const sameName = (a: string, b: string) =>
  a.trim().toLowerCase() === b.trim().toLowerCase();

/**
 * The label behind each of a model's classes: the label with the class's id, or failing that the one
 * with its name, which is how another project's checkpoint finds its classes here. Null where neither exists.
 */
export function resolveClasses(
  model: Pick<AutoBoxModel, "classIds" | "classNames">,
  labels: Label[],
): (Label | null)[] {
  return model.classIds.map((id, i) => {
    const name = model.classNames?.[i];
    return (
      labels.find((l) => l.id === id) ??
      (name === undefined ? null : labels.find((l) => sameName(l.name, name))) ??
      null
    );
  });
}

/**
 * Whether training can start from a saved model, and the classes to train with. Class numbers keep
 * the saved order, because the model's weights mean those classes whatever order the labels are in
 * now. Once labels were added or removed the meanings no longer line up, so training starts over.
 */
export function chooseTrainingStart(
  model: Pick<AutoBoxModel, "classIds" | "classNames"> | null,
  labels: Label[],
): { continued: boolean; classes: Label[] } {
  if (model === null || model.classIds.length !== labels.length)
    return { continued: false, classes: labels };
  const resolved = resolveClasses(model, labels);
  const classes = resolved.filter((l): l is Label => l !== null);
  const matches =
    classes.length === labels.length &&
    new Set(classes.map((l) => l.id)).size === labels.length;
  return matches ? { continued: true, classes } : { continued: false, classes: labels };
}

/** The labels a project needs before another project's checkpoint can draw in it. */
export function labelsForClasses(
  model: Pick<AutoBoxModel, "classIds" | "classNames">,
  labels: Label[],
): Label[] {
  const added: Label[] = [];
  resolveClasses(model, labels).forEach((label, i) => {
    const name = model.classNames?.[i]?.trim();
    if (label || !name || added.some((l) => sameName(l.name, name))) return;
    added.push({ id: randomUUID(), name, color: nextLabelColor([...labels, ...added]), keypoints: [], skeleton: [] });
  });
  return added;
}

/** An instance's normalized box in YOLO form, or null if its label is not a class or nothing of it is inside. */
function normalizedBox(a: Annotation, labels: Label[], size: Size): YoloBox | null {
  const clamp = (value: number) => Math.min(1, Math.max(0, value));
  const cls = labels.findIndex((l) => l.id === a.labelId);
  const box = effectiveBBox(a, size);
  const left = clamp(box.x / size.width);
  const top = clamp(box.y / size.height);
  const w = clamp((box.x + box.w) / size.width) - left;
  const h = clamp((box.y + box.h) / size.height) - top;
  return cls >= 0 && w > 0 && h > 0 ? { cls, cx: left + w / 2, cy: top + h / 2, w, h } : null;
}

/**
 * One training line per object. A keypoint-only instance gets the same box as in exports, because
 * leaving it out would teach the model that a real object is background.
 */
export function toYoloBoxes(
  annotations: Annotation[],
  labels: Label[],
  size: Size,
): YoloBox[] {
  return annotations.flatMap((a) => {
    const box = normalizedBox(a, labels, size);
    return box ? [box] : [];
  });
}

/**
 * Pose training lines: each box followed by `x y v` per keypoint slot in the label's order, padded with
 * zeros to `slots`, as the YOLO exporter does. An unplaced keypoint is `0 0 0`.
 */
export function toYoloPose(
  annotations: Annotation[],
  labels: Label[],
  size: Size,
  slots: number,
): YoloBox[] {
  const clamp = (value: number) => Math.min(1, Math.max(0, value));
  return annotations.flatMap((a) => {
    const box = normalizedBox(a, labels, size);
    if (!box) return [];
    const label = labels[box.cls];
    const keypoints: number[] = [];
    for (let k = 0; k < slots; k++) {
      const definition = label.keypoints[k];
      const p = definition && a.keypoints[definition.id];
      if (p) keypoints.push(clamp(p.x / size.width), clamp(p.y / size.height), p.v);
      else keypoints.push(0, 0, 0);
    }
    return [{ ...box, keypoints }];
  });
}

/**
 * Turns a worker's boxes into annotations that satisfy the stored invariants. The model's classes
 * are the labels when it was trained, and a label deleted since then has no annotations to keep.
 */
export function toPredictedAnnotations(
  boxes: PredictedBox[],
  classes: ModelClass[],
  currentLabels: Label[],
  size: Size,
  task: AutoBoxTask = "detect",
): Annotation[] {
  const round = (value: number) => Math.round(value * 100) / 100;
  const annotations = boxes.flatMap((box): Annotation[] => {
    const cls = classes[box.cls];
    const label = cls && currentLabels.find((l) => l.id === cls.id);
    if (!label) return [];
    const x = Math.max(0, round(box.x));
    const y = Math.max(0, round(box.y));
    const w = round(Math.min(box.x + box.w, size.width) - x);
    const h = round(Math.min(box.y + box.h, size.height) - y);
    if (task === "detect") {
      return w > 0 && h > 0
        ? [{ id: randomUUID(), labelId: label.id, bbox: { x, y, w, h }, keypoints: {} }]
        : [];
    }
    // A keypoint project stores keypoints only (specs.md section 11). Slots map to the ids saved at
    // training time, and a keypoint deleted since then has nowhere to go.
    const keypoints: Record<string, PlacedKeypoint> = {};
    (box.keypoints ?? []).forEach(([kx, ky, confidence], slot) => {
      const keypointId = cls.keypointIds?.[slot];
      if (!keypointId || confidence < TRAINING.keypointConfidence) return;
      if (!label.keypoints.some((k) => k.id === keypointId)) return;
      const px = round(Math.min(size.width, Math.max(0, kx)));
      const py = round(Math.min(size.height, Math.max(0, ky)));
      keypoints[keypointId] = { x: px, y: py, v: 2 };
    });
    return Object.keys(keypoints).length > 0
      ? [{ id: randomUUID(), labelId: label.id, bbox: null, keypoints }]
      : [];
  });
  const error = findAnnotationError(annotations, currentLabels, size);
  if (error) throw new Error(error);
  return annotations;
}

type Job = { status: AutoBoxStatus; cancelled: boolean; stop: () => void };
export type CheckpointOwner = { id: string; name: string; labels: Label[]; mode: ProjectMode };
type Options = {
  dataDir: string;
  command: string[];
  fileOf: (image: AutoBoxImage) => string;
};

/**
 * Runs one training and prediction job at a time in a child process, since a job uses every CPU
 * core it can get. Status lives in memory, so a restart abandons a running job and the worker exits.
 */
export class AutoBoxManager {
  private jobs = new Map<string, Job>();
  private store: Store;
  private options: Options;
  private workRoot: string;
  private modelsDir: string;

  constructor(store: Store, options: Options) {
    this.store = store;
    this.options = options;
    this.workRoot = resolve(options.dataDir, "tmp", "autobox");
    this.modelsDir = resolve(options.dataDir, "models");
    // Anything here was left by a run that a restart abandoned.
    rmSync(this.workRoot, { recursive: true, force: true });
    this.adoptOldModels();
  }

  /** A project used to keep one model, as best.pt and model.json. Those are now its checkpoint "best". */
  private adoptOldModels() {
    const projects = join(this.modelsDir, "projects");
    if (!existsSync(projects)) return;
    for (const projectId of readdirSync(projects)) {
      const old = join(projects, projectId, "model.json");
      if (existsSync(old)) renameSync(old, join(projects, projectId, "best.json"));
    }
  }

  private checkpointFile(projectId: string, id: string, extension: string) {
    return join(this.modelsDir, "projects", projectId, `${id}${extension}`);
  }

  /** The models saved under a project, newest first, with class names filled in from its labels where the file predates them. */
  private readModels(owner: CheckpointOwner): (AutoBoxModel & { file: string; classNames: string[] })[] {
    const folder = join(this.modelsDir, "projects", owner.id);
    if (!existsSync(folder)) return [];
    return readdirSync(folder)
      .filter((name) => name.endsWith(".json"))
      .flatMap((name) => {
        const file = name.slice(0, -".json".length);
        if (!existsSync(this.checkpointFile(owner.id, file, ".pt"))) return [];
        try {
          const model: AutoBoxModel = JSON.parse(readFileSync(join(folder, name), "utf8"));
          const classNames = model.classNames ?? model.classIds.map((id) => owner.labels.find((l) => l.id === id)?.name ?? "");
          // A detector is useless in a keypoint project and the other way round, so each project sees its task only.
          if ((model.task ?? "detect") !== taskOf(owner.mode)) return [];
          return [{ ...model, file, classNames }];
        } catch {
          return [];
        }
      })
      .sort((a, b) => b.trainedAt.localeCompare(a.trainedAt));
  }

  /** The project's saved checkpoints of its task, newest first. They outlive the job and a restart. */
  checkpoints(project: CheckpointOwner): AutoBoxCheckpoint[] {
    return this.readModels(project).map(({ file, ...model }) => ({
      ...model,
      id: file,
      projectId: project.id,
      projectName: project.name,
      canContinue: chooseTrainingStart(model, project.labels).continued,
      missingLabels: [],
    }));
  }

  /**
   * Other projects' checkpoints, which this project can draw with or train from because their classes are
   * matched by label name. Handles carry the owner's id, so a request can never reach a file by path.
   */
  otherCheckpoints(project: CheckpointOwner, others: CheckpointOwner[]): AutoBoxCheckpoint[] {
    return others
      .filter((owner) => owner.id !== project.id && owner.mode === project.mode)
      .flatMap((owner) =>
        this.readModels(owner).map(({ file, ...model }) => ({
          ...model,
          id: `${owner.id}/${file}`,
          projectId: owner.id,
          projectName: owner.name,
          canContinue: chooseTrainingStart(model, project.labels).continued,
          missingLabels: labelsForClasses(model, project.labels).map((l) => l.name),
        })),
      )
      .sort((a, b) => b.trainedAt.localeCompare(a.trainedAt));
  }

  /** The weights file of a checkpoint from the project's lists, wherever its owner keeps it. */
  weightsPath(checkpoint: AutoBoxCheckpoint): string {
    return this.checkpointFile(checkpoint.projectId, checkpoint.id.slice(checkpoint.id.lastIndexOf("/") + 1), ".pt");
  }

  removeCheckpoint(projectId: string, id: string) {
    rmSync(this.checkpointFile(projectId, id, ".json"), { force: true });
    rmSync(this.checkpointFile(projectId, id, ".pt"), { force: true });
  }

  removeAllCheckpoints(projectId: string) {
    rmSync(join(this.modelsDir, "projects", projectId), {
      recursive: true,
      force: true,
    });
  }

  /**
   * Adds a checkpoint, renaming into place so a failure never leaves half of one. Besides the local job,
   * this takes the weights a remote run trained, which the results import has already tied to a bundle.
   */
  importCheckpoint(
    projectId: string,
    weights: string,
    model: AutoBoxModel,
  ): string {
    mkdirSync(join(this.modelsDir, "projects", projectId), { recursive: true });
    const id = randomUUID();
    const saved = this.checkpointFile(projectId, id, ".pt");
    const details = this.checkpointFile(projectId, id, ".json");
    copyFileSync(weights, `${saved}.tmp`);
    renameSync(`${saved}.tmp`, saved);
    writeFileSync(`${details}.tmp`, JSON.stringify(model));
    renameSync(`${details}.tmp`, details);
    return id;
  }

  get(projectId: string): AutoBoxStatus | null {
    return this.jobs.get(projectId)?.status ?? null;
  }

  isRunning(projectId: string): boolean {
    return this.get(projectId)?.state === "running";
  }

  isAnyRunning(): boolean {
    return [...this.jobs.values()].some(
      (job) => job.status.state === "running",
    );
  }

  /** Stops the job, which never writes again once this returns. */
  cancel(projectId: string): boolean {
    const job = this.jobs.get(projectId);
    if (job?.status.state !== "running") return false;
    job.cancelled = true;
    job.stop();
    return true;
  }

  start(
    projectId: string,
    labels: Label[],
    images: JobImages,
    options: JobOptions,
  ): AutoBoxStatus {
    const job: Job = {
      cancelled: false,
      stop: () => {},
      status: {
        state: "running",
        mode: options.mode,
        phase: options.mode === "train" ? "preparing" : "predicting",
        continued: options.mode === "train" && options.checkpoint !== null,
        checkpointTrainedAt: options.checkpoint?.trainedAt ?? null,
        confidence: options.confidence,
        patience: options.patience,
        trainImages: 0,
        valImages: 0,
        epoch: 0,
        bestMap50: null,
        stoppedEarly: false,
        predicted: 0,
        total: images.targets.length,
        boxedImages: 0,
        boxes: 0,
        skipped: 0,
        errors: [],
        error: null,
      },
    };
    this.jobs.set(projectId, job);
    const workDir = join(this.workRoot, randomUUID());
    console.log(`[autobox ${projectId.slice(0, 8)}] ${options.mode} (${options.task}): ${images.training.length} training, ${images.targets.length} targets`);
    void this.run(job, projectId, labels, images, options, workDir)
      .catch((error) => {
        job.status.state = "failed";
        job.status.error =
          error instanceof Error ? error.message : String(error);
      })
      .finally(() => {
        const { status } = job;
        console.log(`[autobox ${projectId.slice(0, 8)}] ${status.state}: ${status.boxes} annotations on ${status.boxedImages} images${status.error ? `, error: ${status.error}` : ""}`);
        return rm(workDir, { recursive: true, force: true });
      });
    return job.status;
  }

  private async run(
    job: Job,
    projectId: string,
    labels: Label[],
    images: JobImages,
    options: JobOptions,
    workDir: string,
  ) {
    const fileOf = (image: AutoBoxImage) => resolve(this.options.fileOf(image));
    const training = options.mode === "train" ? images.training : [];
    const validation = chooseValidation(training.map((i) => i.id));
    const { checkpoint } = options;
    const modelPath = checkpoint && this.weightsPath(checkpoint);
    const trainingClasses = chooseTrainingStart(checkpoint, labels).classes;
    const slots = options.task === "pose" ? keypointCount(trainingClasses) : 0;
    // A checkpoint's class numbers mean the labels it was trained on, whatever the project has now.
    const classes: ModelClass[] =
      options.mode === "train"
        ? trainingClasses.map((l) => ({ id: l.id, name: l.name, keypointIds: l.keypoints.map((k) => k.id) }))
        : checkpoint
          ? resolveClasses(checkpoint, labels).map((l, i) => ({ id: l?.id ?? "", keypointIds: checkpoint.keypointIds?.[i] ?? [] }))
          : [];
    const manifest = {
      ...TRAINING,
      task: options.task,
      kptShape: options.task === "pose" ? [slots, 3] : null,
      weights: job.status.continued ? modelPath : baseWeights(options.task),
      confidence: options.confidence,
      patience: options.patience,
      train: options.mode === "train",
      modelPath,
      workDir,
      classes: trainingClasses.map((l) => l.name),
      training: training.map((image) => ({
        path: fileOf(image),
        width: image.width,
        height: image.height,
        split: validation.has(image.id) ? "val" : "train",
        boxes:
          options.task === "pose"
            ? toYoloPose(image.annotations, trainingClasses, image, slots)
            : toYoloBoxes(image.annotations, trainingClasses, image),
      })),
      targets: images.targets.map((image) => ({
        id: image.id,
        path: fileOf(image),
        width: image.width,
        height: image.height,
      })),
    };
    await mkdir(workDir, { recursive: true });
    mkdirSync(join(this.modelsDir, "config"), { recursive: true });
    const manifestPath = join(workDir, "job.json");
    await writeFile(manifestPath, JSON.stringify(manifest));
    if (job.cancelled) {
      job.status.state = "cancelled";
      return;
    }

    const sizes = new Map(
      images.targets.map((i) => [i.id, { width: i.width, height: i.height }]),
    );
    const [command, ...args] = this.options.command;
    // Ultralytics gets its own settings file, and may neither report usage nor install packages.
    const child = spawn(command, [...args, manifestPath], {
      cwd: this.modelsDir,
      stdio: ["pipe", "ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        YOLO_CONFIG_DIR: join(this.modelsDir, "config"),
        YOLO_OFFLINE: "true",
        YOLO_AUTOINSTALL: "false",
        YOLO_VERBOSE: "false",
      },
    });
    job.stop = () => child.kill("SIGTERM");

    let stderr = "";
    child.stderr!.setEncoding("utf8").on("data", (text: string) => {
      stderr = (stderr + text).slice(-STDERR_TAIL);
    });
    createInterface({ input: child.stdio[3] as Readable }).on(
      "line",
      (line) => {
        if (job.cancelled) return;
        try {
          this.handle(job, projectId, options.task, classes, sizes, JSON.parse(line));
        } catch (error) {
          job.status.error =
            error instanceof Error ? error.message : String(error);
          job.stop();
        }
      },
    );

    const { status } = job;
    await new Promise<void>((done) => {
      child.on("error", (error) => {
        status.error = `Could not start ${command}: ${error.message}. Install Python with 'pip install ultralytics', or set LABELAPP_PYTHON.`;
        done();
      });
      child.on("close", (code) => {
        if (code !== 0 && !status.error && !job.cancelled)
          status.error =
            stderr.trim() || `The auto box worker stopped with code ${code}.`;
        done();
      });
    });
    status.state = job.cancelled
      ? "cancelled"
      : status.error
        ? "failed"
        : "done";
  }

  private handle(
    job: Job,
    projectId: string,
    task: AutoBoxTask,
    classes: ModelClass[],
    sizes: Map<string, Size>,
    message: Record<string, any>,
  ) {
    const { status } = job;
    switch (message.type) {
      case "prepared":
        status.phase = "training";
        status.trainImages = message.trainImages;
        status.valImages = message.valImages;
        break;
      case "epoch":
        status.epoch = message.epoch;
        status.bestMap50 = message.bestMap50;
        break;
      case "trained":
        status.phase = "predicting";
        status.epoch = message.epoch;
        status.stoppedEarly = message.stoppedEarly;
        status.bestMap50 = message.bestMap50;
        this.importCheckpoint(projectId, message.weights, {
          trainedAt: new Date().toISOString(),
          trainImages: status.trainImages,
          epochs: message.epoch,
          bestMap50: message.bestMap50,
          classIds: classes.map((c) => c.id),
          classNames: classes.map((c) => c.name ?? ""),
          task,
          keypointIds: classes.map((c) => c.keypointIds ?? []),
          modelSize: "n",
          source: "local",
        });
        break;
      case "pred": {
        const size = sizes.get(message.id);
        const labels = this.store.labelsOf(projectId);
        status.predicted++;
        if (!size || !labels) break;
        const annotations = toPredictedAnnotations(
          message.boxes,
          classes,
          labels,
          size,
          task,
        );
        if (
          this.store.saveAutoBoxes(message.id, annotations) &&
          annotations.length > 0
        ) {
          status.boxedImages++;
          status.boxes += annotations.length;
        }
        break;
      }
      case "skip":
        if (status.phase === "predicting") status.predicted++;
        status.skipped++;
        if (status.errors.length < MAX_LISTED_ERRORS)
          status.errors.push({ path: message.path, reason: message.reason });
        break;
      case "error":
        status.error = message.message;
        break;
    }
  }
}
