import { basename } from 'node:path';
import { promisify } from 'node:util';
import { gunzip, gzip } from 'node:zlib';
import {
  BUNDLE_KIND,
  BUNDLE_UPLOADS,
  BUNDLE_VERSION,
  findAnnotationError,
  findLabelModeError,
  isSafeRelPath,
  projectBundleSchema,
  type Annotation,
  type ImageStatus,
  type Label,
  type ProjectBundle,
  type ProjectMode,
} from '../shared/schema.ts';

/** The most JSON a bundle may unpack to. V8 cannot hold a much longer string, and this fits millions of images. */
export const MAX_BUNDLE_JSON_BYTES = 256 * 1024 * 1024;
/** How many of a folder's files a dry run looks for, to tell whether a chosen path is the right folder. */
export const SAMPLE_CHECKS = 20;

export class BundleError extends Error {}

export type BundleSourceInput = { id: string; name: string; path: string };
export type BundleImageInput = {
  sourceId: string | null;
  relPath: string | null;
  filename: string;
  width: number;
  height: number;
  status: ImageStatus;
  annotations: Annotation[];
  imageLabels: string[];
  autoBoxed: boolean;
};

/** Describes a whole project, every image included, without its files. Images keep the order given. */
export function buildBundle(
  project: { name: string; mode: ProjectMode; labels: Label[] },
  sources: BundleSourceInput[],
  exclusions: Map<string, string[]>,
  images: BundleImageInput[],
  now = new Date(),
): ProjectBundle {
  const keys = new Map(sources.map((s, index) => [s.id, `s${index + 1}`]));
  return {
    kind: BUNDLE_KIND,
    version: BUNDLE_VERSION,
    exportedAt: now.toISOString(),
    project: { name: project.name, mode: project.mode, labels: project.labels },
    sources: sources.map((s) => ({ key: keys.get(s.id)!, name: s.name, originalPath: s.path, exclusions: exclusions.get(s.id) ?? [] })),
    images: images.map((image) => {
      const key = image.sourceId === null ? undefined : keys.get(image.sourceId);
      return {
        source: key && image.relPath ? key : BUNDLE_UPLOADS,
        relPath: key && image.relPath ? image.relPath : isSafeRelPath(image.filename) ? image.filename : 'image',
        filename: image.filename,
        width: image.width,
        height: image.height,
        status: image.status,
        annotations: image.annotations,
        imageLabels: image.imageLabels,
        autoBoxed: image.autoBoxed,
      };
    }),
  };
}

export async function serializeBundle(bundle: ProjectBundle): Promise<Buffer> {
  return promisify(gzip)(Buffer.from(JSON.stringify(bundle)));
}

/** Reads a bundle file, gzipped or plain JSON, and checks it holds a valid project. Throws BundleError. */
export async function parseBundle(data: Buffer): Promise<ProjectBundle> {
  let json = data;
  if (data[0] === 0x1f && data[1] === 0x8b) {
    try {
      json = await promisify(gunzip)(data, { maxOutputLength: MAX_BUNDLE_JSON_BYTES });
    } catch (error) {
      if (error instanceof RangeError || (error as NodeJS.ErrnoException).code === 'ERR_BUFFER_TOO_LARGE') {
        throw new BundleError('The project file is too large to open.');
      }
      throw new BundleError('The project file is damaged.');
    }
  }
  if (json.length > MAX_BUNDLE_JSON_BYTES) throw new BundleError('The project file is too large to open.');

  let raw: unknown;
  try {
    raw = JSON.parse(json.toString('utf8'));
  } catch {
    throw new BundleError('This is not a LabelApp project file.');
  }
  const header = raw as { kind?: unknown; version?: unknown } | null;
  if (typeof header !== 'object' || header === null || header.kind !== BUNDLE_KIND) {
    throw new BundleError('This is not a LabelApp project file. Use "Import annotations" in a project for COCO or LabelApp JSON.');
  }
  if (header.version !== BUNDLE_VERSION) {
    throw new BundleError(`This project file has version ${String(header.version)}, which this LabelApp cannot open. Update LabelApp.`);
  }

  const result = projectBundleSchema.safeParse(raw);
  if (!result.success) {
    const issue = result.error.issues[0];
    const path = issue.path.join('.');
    throw new BundleError(`The project file is invalid: ${path ? `${path}: ` : ''}${issue.message}`);
  }
  const error = findBundleError(result.data);
  if (error) throw new BundleError(`The project file is invalid: ${error}`);
  return result.data;
}

/** The invariants of specs.md section 3, plus what makes the bundle's references consistent. */
export function findBundleError(bundle: ProjectBundle): string | null {
  const { labels, mode } = bundle.project;
  const modeError = findLabelModeError(labels, mode);
  if (modeError) return modeError;

  const keys = new Set<string>();
  const names = new Set<string>();
  for (const source of bundle.sources) {
    if (source.key === BUNDLE_UPLOADS) return `"${BUNDLE_UPLOADS}" cannot be a folder key`;
    if (keys.has(source.key)) return `Duplicate folder key ${source.key}`;
    if (names.has(source.name)) return `Duplicate folder name ${source.name}`;
    keys.add(source.key);
    names.add(source.name);
  }

  const labelIds = new Set(labels.map((l) => l.id));
  const seen = new Set<string>();
  for (const image of bundle.images) {
    const where = `${image.source}/${image.relPath}`;
    if (image.source !== BUNDLE_UPLOADS) {
      if (!keys.has(image.source)) return `Image ${where} is in an unknown folder`;
      if (seen.has(where)) return `Image ${where} appears twice`;
      seen.add(where);
    }
    const annotationError = findAnnotationError(image.annotations, labels, image);
    if (annotationError) return `Image ${where}: ${annotationError}`;
    if (new Set(image.imageLabels).size !== image.imageLabels.length) return `Image ${where} has duplicate image labels`;
    const unknown = image.imageLabels.find((l) => !labelIds.has(l));
    if (unknown) return `Image ${where} has unknown image label ${unknown}`;
  }
  return null;
}

/** The folders a teammate has to place: each source, then uploads if there are any. */
export function bundleFolders(bundle: ProjectBundle) {
  const counts = new Map<string, number>();
  for (const image of bundle.images) counts.set(image.source, (counts.get(image.source) ?? 0) + 1);
  const folders = bundle.sources.map((s) => ({ key: s.key, name: s.name, originalPath: s.originalPath, imageCount: counts.get(s.key) ?? 0 }));
  const uploads = counts.get(BUNDLE_UPLOADS) ?? 0;
  if (uploads > 0) folders.push({ key: BUNDLE_UPLOADS, name: 'Uploaded images', originalPath: '', imageCount: uploads });
  return folders;
}

/** Up to `count` of a folder's paths, spread over the whole folder rather than the first few. */
export function samplePaths(bundle: ProjectBundle, key: string, count = SAMPLE_CHECKS): string[] {
  const paths = bundle.images.filter((i) => i.source === key).map((i) => i.relPath);
  if (paths.length <= count) return paths;
  return Array.from({ length: count }, (_, i) => paths[Math.floor((i * paths.length) / count)]);
}

export type OpenPlan = {
  name: string;
  mode: ProjectMode;
  labels: Label[];
  sources: { key: string; baseName: string; path: string; exclusions: string[] }[];
  images: {
    sourceKey: string;
    relPath: string;
    filename: string;
    width: number;
    height: number;
    status: ImageStatus;
    annotations: Annotation[];
    imageLabels: string[];
    autoBoxed: boolean;
  }[];
  warnings: string[];
};

/**
 * What to create from a bundle, given where its folders are on this computer (resolved paths by key). Folders with
 * no path are left out with their images. Uploads become a folder of their own, matched by original filename.
 */
export function planOpen(bundle: ProjectBundle, paths: Map<string, string>, name = bundle.project.name): OpenPlan {
  const warnings: string[] = [];
  const sources: OpenPlan['sources'] = [];
  for (const source of bundle.sources) {
    const path = paths.get(source.key);
    if (path) sources.push({ key: source.key, baseName: source.name, path, exclusions: source.exclusions });
  }
  const uploadsPath = paths.get(BUNDLE_UPLOADS);
  if (uploadsPath) sources.push({ key: BUNDLE_UPLOADS, baseName: basename(uploadsPath) || 'uploaded', path: uploadsPath, exclusions: [] });

  const placed = new Set(sources.map((s) => s.key));
  const left = new Map<string, number>();
  const uploadNames = new Set<string>();
  let duplicates = 0;
  const images: OpenPlan['images'] = [];
  for (const image of bundle.images) {
    if (!placed.has(image.source)) {
      left.set(image.source, (left.get(image.source) ?? 0) + 1);
      continue;
    }
    if (image.source === BUNDLE_UPLOADS) {
      // Only one file of a name can be in the chosen folder, so later images with that name have nothing to match.
      if (uploadNames.has(image.relPath)) {
        duplicates++;
        continue;
      }
      uploadNames.add(image.relPath);
    }
    images.push({ sourceKey: image.source, ...image });
  }

  for (const folder of bundleFolders(bundle)) {
    const count = left.get(folder.key);
    if (count) warnings.push(`${count} ${count === 1 ? 'image' : 'images'} in "${folder.name}" left out, because no folder was chosen for it.`);
  }
  if (duplicates > 0) {
    warnings.push(`${duplicates} uploaded ${duplicates === 1 ? 'image was' : 'images were'} left out, because an earlier uploaded image has the same name.`);
  }
  return { name, mode: bundle.project.mode, labels: bundle.project.labels, sources, images, warnings };
}
