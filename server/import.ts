import { randomUUID } from 'node:crypto';
import { z } from 'zod';
import { nextLabelColor } from '../shared/colors.ts';
import type { Annotation, BBox, ImageStatus, ImportSummary, Label, PlacedKeypoint, ProjectMode } from '../shared/schema.ts';

/** A problem with the file as a whole. Anything per image or per annotation is reported in the summary instead. */
export class ImportFileError extends Error {}

export type ImportCategory = { name: string; keypoints: string[]; skeleton: [number, number][] };
export type ImportAnnotation = {
  /** Index into the dataset's categories. */
  category: number;
  bbox: BBox | null;
  /** One entry per category keypoint, in order; null where unplaced. */
  keypoints: (PlacedKeypoint | null)[];
};
export type ImportImage = {
  path: string;
  width: number | null;
  height: number | null;
  annotations: ImportAnnotation[];
  /** Image label names, which only the native format carries. */
  imageLabels: string[];
  /** The status recorded in the file, which only the native format carries. */
  status: ImageStatus | null;
};
export type ImportDataset = { format: 'coco' | 'labelapp'; categories: ImportCategory[]; images: ImportImage[]; warnings: string[] };

const MAX_WARNINGS = 50;

const finite = z.number().finite();
const cocoSchema = z.object({
  images: z.array(z.object({ id: z.union([z.number(), z.string()]), file_name: z.string().min(1), width: finite.optional(), height: finite.optional() })),
  annotations: z
    .array(
      z.object({
        image_id: z.union([z.number(), z.string()]),
        category_id: z.union([z.number(), z.string()]),
        bbox: z.array(finite).optional(),
        keypoints: z.array(finite).optional(),
      }),
    )
    .default([]),
  categories: z.array(
    z.object({
      id: z.union([z.number(), z.string()]),
      name: z.string().trim().min(1).max(200),
      keypoints: z.array(z.string().trim().min(1)).optional(),
      skeleton: z.array(z.tuple([z.number().int(), z.number().int()])).optional(),
    }),
  ),
});

const nativeSchema = z.object({
  labels: z.array(
    z.object({
      id: z.string(),
      name: z.string().trim().min(1).max(200),
      keypoints: z.array(z.object({ id: z.string(), name: z.string().trim().min(1) })).default([]),
      skeleton: z.array(z.tuple([z.string(), z.string()])).default([]),
    }),
  ),
  images: z.array(
    z.object({
      filename: z.string().min(1),
      folder: z.string().default(''),
      imageLabels: z.array(z.string()).default([]),
      width: finite.optional(),
      height: finite.optional(),
      status: z.enum(['todo', 'done']).optional(),
      annotations: z
        .array(
          z.object({
            labelId: z.string(),
            bbox: z.object({ x: finite, y: finite, w: finite, h: finite }).nullable().default(null),
            keypoints: z.record(z.string(), z.object({ x: finite, y: finite, v: finite })).default({}),
          }),
        )
        .default([]),
    }),
  ),
});

function describeIssue(error: z.ZodError): string {
  const issue = error.issues[0];
  const path = issue.path.join('.');
  return path ? `${path}: ${issue.message}` : issue.message;
}

/** Reads a COCO or native LabelApp export into one shape. Throws ImportFileError when the file is neither. */
export function parseDataset(data: unknown): ImportDataset {
  if (typeof data !== 'object' || data === null || Array.isArray(data)) throw new ImportFileError('The file is not a JSON object.');
  const warnings: string[] = [];
  const warn = (message: string) => {
    if (warnings.length < MAX_WARNINGS) warnings.push(message);
  };

  if ('categories' in data && 'images' in data) {
    const result = cocoSchema.safeParse(data);
    if (!result.success) throw new ImportFileError(`Not a valid COCO file. ${describeIssue(result.error)}`);
    const coco = result.data;
    const categoryIndex = new Map(coco.categories.map((c, i) => [String(c.id), i]));
    const categories: ImportCategory[] = coco.categories.map((c) => {
      const keypoints = c.keypoints ?? [];
      const skeleton = (c.skeleton ?? []).flatMap(([a, b]): [number, number][] => {
        // COCO skeletons are 1-indexed.
        const edge: [number, number] = [a - 1, b - 1];
        const valid = edge.every((k) => k >= 0 && k < keypoints.length) && edge[0] !== edge[1];
        if (!valid) warn(`Category "${c.name}": skeleton edge [${a}, ${b}] does not join two keypoints and was ignored.`);
        return valid ? [edge] : [];
      });
      return { name: c.name, keypoints, skeleton };
    });
    const images = new Map<string, ImportImage>();
    for (const image of coco.images) {
      const key = String(image.id);
      if (images.has(key)) {
        warn(`Image id ${key} appears twice; the first was kept.`);
        continue;
      }
      images.set(key, { path: image.file_name, width: image.width ?? null, height: image.height ?? null, annotations: [], imageLabels: [], status: null });
    }
    coco.annotations.forEach((a, n) => {
      const image = images.get(String(a.image_id));
      const category = categoryIndex.get(String(a.category_id));
      if (!image) return warn(`Annotation ${n + 1} references unknown image_id ${a.image_id} and was skipped.`);
      if (category === undefined) return warn(`Annotation ${n + 1} references unknown category_id ${a.category_id} and was skipped.`);
      const bbox = a.bbox && a.bbox.length === 4 ? { x: a.bbox[0], y: a.bbox[1], w: a.bbox[2], h: a.bbox[3] } : null;
      const names = categories[category].keypoints;
      const keypoints: (PlacedKeypoint | null)[] = names.map((_, k) => {
        const flat = a.keypoints;
        if (!flat || flat.length < (k + 1) * 3) return null;
        const [x, y, v] = [flat[k * 3], flat[k * 3 + 1], flat[k * 3 + 2]];
        return v > 0 ? { x, y, v: v === 1 ? 1 : 2 } : null;
      });
      image.annotations.push({ category, bbox, keypoints });
    });
    return { format: 'coco', categories, images: [...images.values()], warnings };
  }

  if ('labels' in data && 'images' in data) {
    const result = nativeSchema.safeParse(data);
    if (!result.success) throw new ImportFileError(`Not a valid LabelApp export. ${describeIssue(result.error)}`);
    const native = result.data;
    const labelIndex = new Map(native.labels.map((l, i) => [l.id, i]));
    const categories: ImportCategory[] = native.labels.map((l) => ({
      name: l.name,
      keypoints: l.keypoints.map((k) => k.name),
      skeleton: l.skeleton.flatMap(([a, b]): [number, number][] => {
        const edge: [number, number] = [l.keypoints.findIndex((k) => k.id === a), l.keypoints.findIndex((k) => k.id === b)];
        return edge[0] >= 0 && edge[1] >= 0 && edge[0] !== edge[1] ? [edge] : [];
      }),
    }));
    const images = native.images.map((image): ImportImage => {
      const annotations = image.annotations.flatMap((a, n): ImportAnnotation[] => {
        const category = labelIndex.get(a.labelId);
        if (category === undefined) {
          warn(`${image.filename}: annotation ${n + 1} references unknown label ${a.labelId} and was skipped.`);
          return [];
        }
        const keypoints = native.labels[category].keypoints.map((k): PlacedKeypoint | null => {
          const p = a.keypoints[k.id];
          return p && p.v > 0 ? { x: p.x, y: p.y, v: p.v === 1 ? 1 : 2 } : null;
        });
        return [{ category, bbox: a.bbox, keypoints }];
      });
      const imageLabels = image.imageLabels.flatMap((id) => {
        const index = labelIndex.get(id);
        return index === undefined ? [] : [native.labels[index].name];
      });
      const path = image.folder && image.folder !== 'uploads' ? `${image.folder}/${image.filename}` : image.filename;
      return { path, width: image.width ?? null, height: image.height ?? null, annotations, imageLabels, status: image.status ?? null };
    });
    return { format: 'labelapp', categories, images, warnings };
  }

  throw new ImportFileError('The file is neither a COCO file (images, annotations, categories) nor a LabelApp export (labels, images).');
}

export type MatchableImage = { id: string; folder: string; filename: string; sourceId: string | null };
export type Match<T> = { image: T } | { ambiguous: true } | null;

const segments = (path: string) => path.replaceAll('\\', '/').split('/').filter((s) => s !== '' && s !== '.');
const imageSegments = (image: MatchableImage) => (image.sourceId ? [...segments(image.folder), image.filename] : [image.filename]);

/**
 * Finds the project image each path refers to. Paths rarely match exactly, because the file was made from
 * a different folder, so the image sharing the longest run of trailing path segments wins. A tie is ambiguous.
 */
export function matchImages<T extends MatchableImage>(paths: string[], images: T[]): Match<T>[] {
  const byName = new Map<string, { image: T; segments: string[] }[]>();
  for (const image of images) {
    const list = byName.get(image.filename) ?? [];
    list.push({ image, segments: imageSegments(image) });
    byName.set(image.filename, list);
  }
  return paths.map((path) => {
    const parts = segments(path);
    const candidates = byName.get(parts[parts.length - 1] ?? '') ?? [];
    let best = 0;
    let winners: T[] = [];
    for (const candidate of candidates) {
      let shared = 0;
      while (
        shared < parts.length &&
        shared < candidate.segments.length &&
        parts[parts.length - 1 - shared] === candidate.segments[candidate.segments.length - 1 - shared]
      ) {
        shared++;
      }
      if (shared > best) {
        best = shared;
        winners = [candidate.image];
      } else if (shared === best && shared > 0) {
        winners.push(candidate.image);
      }
    }
    if (winners.length === 1) return { image: winners[0] };
    return winners.length > 1 ? { ambiguous: true } : null;
  });
}

export type ImportOptions = {
  /** Create labels for categories the project lacks. Otherwise their annotations are skipped. */
  createLabels: boolean;
  /** Replace annotations a person saved. Otherwise images with any are skipped. Unreviewed predictions are always replaced. */
  overwrite: boolean;
  /** Mark every image the file describes as done, including those with no annotations. */
  markDone: boolean;
};

export type ImportTarget = MatchableImage & {
  width: number;
  height: number;
  status: ImageStatus;
  annotationCount: number;
  autoBoxed: boolean;
  fileState: string;
  imageLabels: string[];
};

export type ImportWrite = { imageId: string; annotations: Annotation[]; status?: ImageStatus; imageLabels?: string[] };


export type ImportPlan = { labels: Label[]; labelsChanged: boolean; writes: ImportWrite[]; summary: ImportSummary };

const sameName = (a: string, b: string) => a.trim().toLowerCase() === b.trim().toLowerCase();

/** Clamps a box to the image, returning null if nothing of it lies inside. */
function clampBox(box: BBox, size: { width: number; height: number }): BBox | null {
  const x1 = Math.max(0, Math.min(size.width, box.x));
  const y1 = Math.max(0, Math.min(size.height, box.y));
  const x2 = Math.max(0, Math.min(size.width, box.x + box.w));
  const y2 = Math.max(0, Math.min(size.height, box.y + box.h));
  return x2 - x1 > 0 && y2 - y1 > 0 ? { x: x1, y: y1, w: x2 - x1, h: y2 - y1 } : null;
}

/**
 * Decides what an import would change, without changing anything. Labels are matched by name; the result's
 * `labels` is the full list to store, which differs from the project's only when labels were created or extended.
 */
export function planImport(
  project: { mode: ProjectMode; labels: Label[] },
  images: ImportTarget[],
  dataset: ImportDataset,
  options: ImportOptions,
): ImportPlan {
  const labels: Label[] = project.labels.map((l) => ({ ...l, keypoints: [...l.keypoints], skeleton: [...l.skeleton] }));
  let labelsChanged = false;
  const newLabels: string[] = [];
  const extendedLabels = new Set<string>();
  const warnings = [...dataset.warnings];
  const warn = (message: string) => {
    if (warnings.length < MAX_WARNINGS) warnings.push(message);
  };

  // Per category: the label it maps to and, per category keypoint, the label's keypoint id.
  const mapping = dataset.categories.map((category): { labelId: string; keypointIds: string[] } | null => {
    let label = labels.find((l) => sameName(l.name, category.name));
    const wantsKeypoints = project.mode === 'keypoint';
    if (!label) {
      if (!options.createLabels) return null;
      label = {
        id: randomUUID(),
        name: category.name.trim(),
        color: nextLabelColor(labels),
        keypoints: wantsKeypoints ? category.keypoints.map((name) => ({ id: randomUUID(), name })) : [],
        skeleton: [],
      };
      labels.push(label);
      labelsChanged = true;
      newLabels.push(label.name);
    } else if (wantsKeypoints) {
      for (const name of category.keypoints) {
        if (label.keypoints.some((k) => sameName(k.name, name))) continue;
        label.keypoints.push({ id: randomUUID(), name });
        labelsChanged = true;
        extendedLabels.add(label.name);
      }
    }
    const keypointIds = category.keypoints.map((name) => label.keypoints.find((k) => sameName(k.name, name))?.id ?? '');
    if (wantsKeypoints) {
      for (const [a, b] of category.skeleton) {
        const edge: [string, string] = [keypointIds[a], keypointIds[b]];
        if (label.skeleton.some(([x, y]) => (x === edge[0] && y === edge[1]) || (x === edge[1] && y === edge[0]))) continue;
        label.skeleton.push(edge);
        labelsChanged = true;
        if (!newLabels.includes(label.name)) extendedLabels.add(label.name);
      }
    }
    return { labelId: label.id, keypointIds };
  });
  const labelIdOf = (name: string) => labels.find((l) => sameName(l.name, name))?.id ?? null;

  const summary: ImportSummary = {
    format: dataset.format,
    imagesInFile: dataset.images.length,
    matched: 0,
    unmatched: 0,
    ambiguous: 0,
    alreadyAnnotated: 0,
    sizeMismatch: 0,
    unavailable: 0,
    imagesWritten: 0,
    annotations: 0,
    droppedAnnotations: 0,
    keypointsIgnored: 0,
    newLabels,
    extendedLabels: [],
    warnings,
  };

  const matches = matchImages(dataset.images.map((i) => i.path), images);
  const writes: ImportWrite[] = [];
  const claimed = new Set<string>();
  dataset.images.forEach((entry, index) => {
    const match = matches[index];
    if (!match) {
      summary.unmatched++;
      return warn(`${entry.path}: no image in the project has this path.`);
    }
    if ('ambiguous' in match) {
      summary.ambiguous++;
      return warn(`${entry.path}: several images in the project could be this file.`);
    }
    const image = match.image;
    if (claimed.has(image.id)) {
      summary.ambiguous++;
      return warn(`${entry.path}: another entry in the file already matched this image.`);
    }
    claimed.add(image.id);
    summary.matched++;
    if (image.fileState !== 'ok') {
      summary.unavailable++;
      return warn(`${entry.path}: the image file is unavailable, so its annotations are read-only.`);
    }
    if ((entry.width !== null && entry.width !== image.width) || (entry.height !== null && entry.height !== image.height)) {
      summary.sizeMismatch++;
      return warn(`${entry.path}: the file says ${entry.width}×${entry.height} but the project has ${image.width}×${image.height}.`);
    }
    if (image.annotationCount > 0 && !image.autoBoxed && !options.overwrite) {
      summary.alreadyAnnotated++;
      return;
    }

    const annotations: Annotation[] = [];
    for (const a of entry.annotations) {
      const target = mapping[a.category];
      if (!target) {
        summary.droppedAnnotations++;
        continue;
      }
      const bbox = a.bbox ? clampBox(a.bbox, image) : null;
      const keypoints: Record<string, PlacedKeypoint> = {};
      if (project.mode === 'keypoint') {
        a.keypoints.forEach((p, k) => {
          if (!p || !target.keypointIds[k]) return;
          keypoints[target.keypointIds[k]] = { x: Math.max(0, Math.min(image.width, p.x)), y: Math.max(0, Math.min(image.height, p.y)), v: p.v };
        });
      } else if (a.keypoints.some((p) => p !== null)) {
        summary.keypointsIgnored++;
      }
      if (!bbox && Object.keys(keypoints).length === 0) {
        summary.droppedAnnotations++;
        continue;
      }
      annotations.push({ id: randomUUID(), labelId: target.labelId, bbox, keypoints });
    }

    const write: ImportWrite = { imageId: image.id, annotations };
    const status = options.markDone ? 'done' : entry.status === 'done' ? 'done' : null;
    if (status && status !== image.status) write.status = status;
    const imageLabels = [...image.imageLabels];
    for (const name of entry.imageLabels) {
      const id = labelIdOf(name);
      if (id && !imageLabels.includes(id)) imageLabels.push(id);
    }
    if (imageLabels.length !== image.imageLabels.length) write.imageLabels = imageLabels;

    // An image with nothing to add is left alone, so a listed negative stays todo unless asked otherwise.
    if (annotations.length === 0 && image.annotationCount === 0 && !write.status && !write.imageLabels) return;
    writes.push(write);
    summary.imagesWritten++;
    summary.annotations += annotations.length;
  });

  summary.extendedLabels = [...extendedLabels];
  // Labels only matter if something uses them; a dry run with nothing to write still reports them.
  return { labels, labelsChanged, writes, summary };
}
