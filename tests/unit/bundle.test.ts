import { gzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import type { Annotation, Label, ProjectBundle } from '../../shared/schema.ts';
import { buildBundle, BundleError, findBundleError, MAX_BUNDLE_JSON_BYTES, parseBundle, planOpen, samplePaths, serializeBundle, type BundleImageInput } from '../../server/bundle.ts';

const car: Label = { id: 'car', name: 'car', color: '#00ff00', keypoints: [], skeleton: [] };
const person: Label = { id: 'person', name: 'person', color: '#ff0000', keypoints: [{ id: 'head', name: 'head' }], skeleton: [] };
const box: Annotation = { id: 'a1', labelId: 'car', bbox: { x: 1, y: 1, w: 5, h: 5 }, keypoints: {} };
const head: Annotation = { id: 'a2', labelId: 'person', bbox: null, keypoints: { head: { x: 2, y: 2, v: 2 } } };

const image = (overrides: Partial<BundleImageInput>): BundleImageInput => ({
  sourceId: 'src',
  relPath: 'a.png',
  filename: 'a.png',
  width: 10,
  height: 10,
  status: 'todo',
  annotations: [],
  imageLabels: [],
  autoBoxed: false,
  ...overrides,
});

const sample = (mode: 'bbox' | 'keypoint' = 'keypoint') =>
  buildBundle(
    { name: 'p', mode, labels: mode === 'bbox' ? [car] : [car, person] },
    [{ id: 'src', name: 'photos', path: '/home/a/photos' }],
    new Map([['src', ['gone.png']]]),
    [
      image({ relPath: 'cats/1.png', filename: '1.png', status: 'done', annotations: [box], imageLabels: ['car'] }),
      image({ relPath: 'cats/2.png', filename: '2.png', annotations: mode === 'bbox' ? [] : [head], autoBoxed: true }),
      image({ sourceId: null, relPath: null, filename: 'up.png' }),
      image({ sourceId: null, relPath: null, filename: 'up.png' }),
    ],
    new Date('2026-01-01T00:00:00Z'),
  );

describe('buildBundle', () => {
  it('keys sources, puts uploads under their filename, and keeps every image in order', () => {
    const bundle = sample();
    expect(bundle).toMatchObject({ kind: 'labelapp-project', version: 1, exportedAt: '2026-01-01T00:00:00.000Z' });
    expect(bundle.sources).toEqual([{ key: 's1', name: 'photos', originalPath: '/home/a/photos', exclusions: ['gone.png'] }]);
    expect(bundle.images.map((i) => [i.source, i.relPath])).toEqual([
      ['s1', 'cats/1.png'],
      ['s1', 'cats/2.png'],
      ['uploads', 'up.png'],
      ['uploads', 'up.png'],
    ]);
    expect(bundle.images[1].autoBoxed).toBe(true);
  });
});

describe('parseBundle', () => {
  const encode = (value: unknown) => gzipSync(JSON.stringify(value));

  it('round trips, gzipped or plain', async () => {
    const bundle = sample();
    expect(await parseBundle(await serializeBundle(bundle))).toEqual(bundle);
    expect(await parseBundle(Buffer.from(JSON.stringify(bundle)))).toEqual(bundle);
  });

  it('rejects other files and versions with a reason', async () => {
    await expect(parseBundle(Buffer.from('garbage'))).rejects.toThrow('not a LabelApp project file');
    await expect(parseBundle(encode({ images: [] }))).rejects.toThrow('not a LabelApp project file');
    await expect(parseBundle(encode({ ...sample(), version: 9 }))).rejects.toThrow('version 9');
    await expect(parseBundle(Buffer.from([0x1f, 0x8b, 1, 2, 3]))).rejects.toThrow(BundleError);
  });

  it('refuses paths that could leave their folder', async () => {
    for (const relPath of ['../x.png', '/etc/x.png', 'a//b.png', 'a/./b.png', 'a\\b.png']) {
      const bundle = sample();
      bundle.images[0].relPath = relPath;
      await expect(parseBundle(encode(bundle)), relPath).rejects.toThrow('Invalid relative path');
    }
  });

  it('refuses a file that unpacks too large', async () => {
    const bomb = gzipSync(Buffer.alloc(MAX_BUNDLE_JSON_BYTES + 1, 0x20));
    await expect(parseBundle(bomb)).rejects.toThrow('too large');
  });
});

describe('findBundleError', () => {
  const broken = (change: (b: ProjectBundle) => void) => {
    const bundle = structuredClone(sample());
    change(bundle);
    return findBundleError(bundle);
  };

  it('accepts a valid bundle, including boxes in a keypoint project', () => {
    expect(findBundleError(sample())).toBeNull();
    expect(findBundleError(sample('bbox'))).toBeNull();
  });

  it('enforces the annotation invariants and the mode', () => {
    expect(broken((b) => (b.images[0].annotations[0].labelId = 'nope'))).toContain('unknown label');
    expect(broken((b) => (b.images[0].annotations[0].bbox = { x: 1, y: 1, w: 50, h: 5 }))).toContain('outside');
    expect(broken((b) => (b.images[0].imageLabels = ['nope']))).toContain('unknown image label');
    expect(broken((b) => (b.images[0].imageLabels = ['car', 'car']))).toContain('duplicate image labels');
    expect(broken((b) => (b.project.mode = 'bbox'))).toContain('defines keypoints');
  });

  it('checks that references are consistent', () => {
    expect(broken((b) => (b.images[0].source = 's9'))).toContain('unknown folder');
    expect(broken((b) => (b.images[1].relPath = 'cats/1.png'))).toContain('appears twice');
    expect(broken((b) => b.sources.push({ ...b.sources[0], key: 's2' }))).toContain('Duplicate folder name');
    expect(broken((b) => (b.sources[0].key = 'uploads'))).toContain('cannot be a folder key');
  });
});

describe('planOpen', () => {
  it('places chosen folders, names uploads after their folder, and warns about what is left out', () => {
    const bundle = sample();
    const plan = planOpen(bundle, new Map([['uploads', '/data/my uploads']]), 'renamed');
    expect(plan.name).toBe('renamed');
    expect(plan.sources).toEqual([{ key: 'uploads', baseName: 'my uploads', path: '/data/my uploads', exclusions: [] }]);
    expect(plan.images.map((i) => i.relPath)).toEqual(['up.png']);
    expect(plan.warnings).toEqual([
      '2 images in "photos" left out, because no folder was chosen for it.',
      '1 uploaded image was left out, because an earlier uploaded image has the same name.',
    ]);
  });

  it('keeps source names and every field of each image', () => {
    const plan = planOpen(sample(), new Map([['s1', '/mnt/photos']]));
    expect(plan.sources).toEqual([{ key: 's1', baseName: 'photos', path: '/mnt/photos', exclusions: ['gone.png'] }]);
    expect(plan.images[0]).toMatchObject({ sourceKey: 's1', relPath: 'cats/1.png', status: 'done', annotations: [box], imageLabels: ['car'] });
    expect(plan.images[1].autoBoxed).toBe(true);
  });
});

describe('samplePaths', () => {
  it('spreads samples over the whole folder', () => {
    const images = Array.from({ length: 100 }, (_, i) => image({ relPath: `${i}.png`, filename: `${i}.png` }));
    const bundle = buildBundle({ name: 'p', mode: 'bbox', labels: [] }, [{ id: 'src', name: 'f', path: '/f' }], new Map(), images);
    const paths = samplePaths(bundle, 's1', 4);
    expect(paths).toEqual(['0.png', '25.png', '50.png', '75.png']);
    expect(samplePaths(bundle, 'uploads')).toEqual([]);
  });
});
