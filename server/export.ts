import type { Annotation, BBox, ImageDetail, ImageStatus, Label, ProjectMode } from '../shared/schema.ts';

export type ExportScope = 'done' | 'annotated';
export type ExportProject = { name: string; labels: Label[] };

/**
 * Untouched todo images are always excluded; an empty label file would claim a true negative.
 * Unreviewed predictions are untouched too, however many boxes they hold.
 */
export function selectImages<T extends { status: ImageStatus; annotations: Annotation[]; autoBoxed: boolean }>(images: T[], scope: ExportScope): T[] {
  return images.filter((i) => i.status === 'done' || (scope === 'annotated' && i.annotations.length > 0 && !i.autoBoxed));
}

/** The annotation's box, or the tight box around its keypoints (at least 1px per side, inside the image). */
export function effectiveBBox(a: Annotation, size: { width: number; height: number }): BBox {
  if (a.bbox) return a.bbox;
  const points = Object.values(a.keypoints);
  const minX = Math.min(...points.map((p) => p.x));
  const minY = Math.min(...points.map((p) => p.y));
  const w = Math.max(Math.max(...points.map((p) => p.x)) - minX, 1);
  const h = Math.max(Math.max(...points.map((p) => p.y)) - minY, 1);
  return { x: Math.max(0, Math.min(minX, size.width - w)), y: Math.max(0, Math.min(minY, size.height - h)), w, h };
}

/** Image labels are a separate question from objects, so this selection ignores annotations entirely. */
export function selectLabeledImages<T extends { status: ImageStatus; imageLabels: string[] }>(images: T[], scope: 'labeled' | 'done'): T[] {
  return images.filter((i) => i.imageLabels.length > 0 && (scope === 'labeled' || i.status === 'done'));
}

/** The image's path within a dataset. Uploads have no folder structure, so they keep their plain name. */
export const relativePath = (image: { folder: string; filename: string; sourceId: string | null }) =>
  image.sourceId ? `${image.folder}/${image.filename}` : image.filename;

export function toNativeJson(project: ExportProject & { mode: ProjectMode }, images: ImageDetail[]) {
  return {
    project: { name: project.name, mode: project.mode },
    labels: project.labels,
    images: images.map((i) => ({
      filename: i.filename,
      folder: i.folder,
      imageLabels: i.imageLabels,
      width: i.width,
      height: i.height,
      status: i.status,
      annotations: i.annotations,
    })),
  };
}

export function toCoco(project: ExportProject, images: ImageDetail[]) {
  const categoryId = new Map(project.labels.map((l, index) => [l.id, index + 1]));
  const imageRecords = images.map((image, index) => ({
    id: index + 1,
    file_name: relativePath(image),
    width: image.width,
    height: image.height,
  }));

  let nextId = 1;
  const annotations = images.flatMap((image, index) =>
    image.annotations.map((a) => {
      const label = project.labels.find((l) => l.id === a.labelId)!;
      const box = effectiveBBox(a, image);
      const record: Record<string, unknown> = {
        id: nextId++,
        image_id: index + 1,
        category_id: categoryId.get(a.labelId),
        bbox: [box.x, box.y, box.w, box.h],
        area: box.w * box.h,
        iscrowd: 0,
      };
      if (label.keypoints.length > 0) {
        record.keypoints = label.keypoints.flatMap((k) => {
          const p = a.keypoints[k.id];
          return p ? [p.x, p.y, p.v] : [0, 0, 0];
        });
        record.num_keypoints = label.keypoints.filter((k) => a.keypoints[k.id]).length;
      }
      return record;
    }),
  );

  const categories = project.labels.map((l) => ({
    id: categoryId.get(l.id),
    name: l.name,
    supercategory: 'none',
    ...(l.keypoints.length > 0 && {
      keypoints: l.keypoints.map((k) => k.name),
      skeleton: l.skeleton.map((edge) => edge.map((kpId) => l.keypoints.findIndex((k) => k.id === kpId) + 1)),
    }),
  }));

  return {
    info: { description: project.name, date_created: new Date().toISOString() },
    images: imageRecords,
    annotations,
    categories,
  };
}

const splitName = (filename: string) => {
  const dot = filename.lastIndexOf('.');
  return dot > 0 ? [filename.slice(0, dot), filename.slice(dot)] : [filename, ''];
};

/** Assigns each image a unique stem, since YOLO matches label files to images by stem alone. */
export function uniqueStems(filenames: string[]): string[] {
  const taken = new Set<string>();
  return filenames.map((filename) => {
    const [stem] = splitName(filename);
    let candidate = stem;
    for (let n = 1; taken.has(candidate.toLowerCase()); n++) candidate = `${stem}_${n}`;
    taken.add(candidate.toLowerCase());
    return candidate;
  });
}

const num = (value: number) => {
  const clamped = Math.min(1, Math.max(0, value));
  return Number(clamped.toFixed(6)).toString();
};

/** Makes paths unique, because uploads can repeat a name and zip entries cannot. */
export function uniquePaths(paths: string[]): string[] {
  const taken = new Set<string>();
  return paths.map((path) => {
    const [stem, extension] = splitName(path);
    let candidate = path;
    for (let n = 1; taken.has(candidate.toLowerCase()); n++) candidate = `${stem}_${n}${extension}`;
    taken.add(candidate.toLowerCase());
    return candidate;
  });
}

const csvField = (value: string) => (/[",\r\n]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value);

/** The classification export: one row per image label, plus a stable class list. */
export function toImageLabels<T extends { folder: string; filename: string; sourceId: string | null; imageLabels: string[] }>(
  project: ExportProject,
  images: T[],
) {
  const paths = uniquePaths(images.map(relativePath));
  const files = images.map((image, index) => {
    const path = paths[index];
    const names = project.labels.filter((l) => image.imageLabels.includes(l.id)).map((l) => l.name);
    return { image, path, rows: names.map((name) => `${csvField(path)},${csvField(name)}\n`) };
  });
  return { classes: project.labels.map((l) => `${l.name}\n`).join(''), files };
}

export function toYolo<T extends ImageDetail>(project: ExportProject, images: T[]) {
  const classIndex = new Map(project.labels.map((l, index) => [l.id, index]));
  const maxKeypoints = Math.max(0, ...project.labels.map((l) => l.keypoints.length));
  const stems = uniqueStems(images.map((i) => i.filename));

  const files = images.map((image, index) => {
    const lines = image.annotations.map((a) => {
      const label = project.labels.find((l) => l.id === a.labelId)!;
      const box = effectiveBBox(a, image);
      const parts = [
        String(classIndex.get(a.labelId)),
        num((box.x + box.w / 2) / image.width),
        num((box.y + box.h / 2) / image.height),
        num(box.w / image.width),
        num(box.h / image.height),
      ];
      for (let k = 0; k < maxKeypoints; k++) {
        const definition = label.keypoints[k];
        const p = definition && a.keypoints[definition.id];
        if (p) parts.push(num(p.x / image.width), num(p.y / image.height), String(p.v));
        else parts.push('0', '0', '0');
      }
      return parts.join(' ');
    });
    const [, extension] = splitName(image.filename);
    return {
      image,
      imageName: `images/${stems[index]}${extension}`,
      labelName: `labels/${stems[index]}.txt`,
      content: lines.length > 0 ? `${lines.join('\n')}\n` : '',
    };
  });

  const yaml = [
    `# ${JSON.stringify(project.name)} exported from LabelApp.`,
    '# "path" is omitted so tools resolve train/val relative to this file.',
    'train: images',
    'val: images',
    ...(maxKeypoints > 0
      ? [
          `kpt_shape: [${maxKeypoints}, 3]`,
          '# Keypoint order per class:',
          ...project.labels.map((l, i) => `#   ${i}: ${JSON.stringify(l.keypoints.map((k) => k.name))}`),
        ]
      : []),
    'names:',
    ...project.labels.map((l, i) => `  ${i}: ${JSON.stringify(l.name)}`),
    '',
  ].join('\n');

  return { yaml, files };
}
