import { cpSync, existsSync, mkdirSync, mkdtempSync, readdirSync, renameSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import type { Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { gunzipSync, gzipSync } from 'node:zlib';
import { tmpdir } from 'node:os';
import { basename, dirname, join } from 'node:path';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import type { Annotation, FolderNode, ImageDetail, ImageMeta, Label, Project, ScanStatus, Source, UploadResult } from '../../shared/schema.ts';
import { createApp } from '../../server/app.ts';
import { makePng, unzip } from '../helpers.ts';

let dataDir: string;
let server: Server;
let base: string;
let close: () => void;

beforeAll(async () => {
  dataDir = mkdtempSync(join(tmpdir(), 'labelapp-test-'));
  const { app, store } = createApp(dataDir);
  close = store.close;
  await new Promise<void>((resolve) => {
    server = app.listen(0, '127.0.0.1', () => resolve());
  });
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}/api`;
});

afterAll(() => {
  server.close();
  close();
  rmSync(dataDir, { recursive: true, force: true });
});

const send = (path: string, method: string, body: unknown, headers: Record<string, string> = {}) =>
  fetch(`${base}${path}`, { method, headers: { 'Content-Type': 'application/json', ...headers }, body: JSON.stringify(body) });

const person: Label = {
  id: 'person',
  name: 'person',
  color: '#ff0000',
  keypoints: [
    { id: 'head', name: 'head' },
    { id: 'foot', name: 'foot' },
  ],
  skeleton: [['head', 'foot']],
};
const car: Label = { id: 'car', name: 'car', color: '#00ff00', keypoints: [], skeleton: [] };

async function upload(projectId: string, files: [string, Buffer][]): Promise<UploadResult> {
  const form = new FormData();
  for (const [name, data] of files) form.append('files', new Blob([new Uint8Array(data)]), name);
  const response = await fetch(`${base}/projects/${projectId}/images`, { method: 'POST', body: form });
  expect(response.status).toBe(201);
  return response.json();
}

let project: Project;
let imageId: string;

beforeEach(async () => {
  project = await (await send('/projects', 'POST', { name: 'test', mode: 'keypoint', labels: [person, car] })).json();
  const result = await upload(project.id, [['a.png', makePng(100, 50)]]);
  imageId = result.uploaded[0].id;
});

const save = (annotations: unknown) => send(`/images/${imageId}/annotations`, 'PUT', { annotations });
const box: Annotation = { id: 'a1', labelId: 'car', bbox: { x: 1, y: 1, w: 10, h: 10 }, keypoints: {} };
const pose: Annotation = {
  id: 'a2',
  labelId: 'person',
  bbox: null,
  keypoints: { head: { x: 5, y: 5, v: 2 }, foot: { x: 9, y: 9, v: 1 } },
};

describe('uploads', () => {
  it('reads dimensions from the file and keeps the original name as metadata', async () => {
    const { uploaded } = await upload(project.id, [['tệp ảnh.png', makePng(30, 20)]]);
    expect(uploaded[0]).toMatchObject({ filename: 'tệp ảnh.png', width: 30, height: 20, status: 'todo', annotationCount: 0 });
    const file = await fetch(`${base}/images/${uploaded[0].id}/file`);
    expect(file.headers.get('content-type')).toBe('image/png');
    expect(file.headers.get('x-content-type-options')).toBe('nosniff');
  });

  it('rejects active content by contents, whatever the name, without failing the batch', async () => {
    const svg = Buffer.from('<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>');
    const { uploaded, rejected } = await upload(project.id, [
      ['evil.png', svg],
      ['evil.svg', svg],
      ['good.png', makePng(4, 4)],
    ]);
    expect(uploaded.map((i) => i.filename)).toEqual(['good.png']);
    expect(rejected.map((r) => r.filename)).toEqual(['evil.png', 'evil.svg']);
    expect(readdirSync(join(dataDir, 'tmp'))).toEqual([]);
  });

  it('strips directories from uploaded names', async () => {
    const { uploaded } = await upload(project.id, [['..\\..\\evil.png', makePng(4, 4)]]);
    expect(uploaded[0].filename).toBe('evil.png');
  });
});

describe('saving annotations', () => {
  it('round-trips annotations and status', async () => {
    await save([box, pose]);
    await send(`/images/${imageId}/annotations`, 'PUT', { status: 'done' });
    const image: ImageDetail = await (await fetch(`${base}/images/${imageId}`)).json();
    expect(image.annotations).toEqual([box, pose]);
    expect(image).toMatchObject({ status: 'done', annotationCount: 2 });
  });

  it.each([
    ['unknown label', { ...box, labelId: 'nope' }],
    ['bbox outside the image', { ...box, bbox: { x: 95, y: 0, w: 10, h: 10 } }],
    ['keypoint outside the image', { ...pose, keypoints: { head: { x: 500, y: 5, v: 2 } } }],
    ['unknown keypoint', { ...pose, keypoints: { wing: { x: 5, y: 5, v: 2 } } }],
    ['no geometry', { ...box, bbox: null }],
  ])('rejects %s with 422', async (_name, annotation) => {
    const response = await save([annotation]);
    expect(response.status).toBe(422);
    expect((await response.json()).error).toEqual(expect.any(String));
  });

  it.each([
    ['zero-size bbox', { ...box, bbox: { x: 1, y: 1, w: 0, h: 10 } }],
    ['invalid visibility', { ...pose, keypoints: { head: { x: 5, y: 5, v: 0 } } }],
  ])('rejects %s with 400', async (_name, annotation) => {
    expect((await save([annotation])).status).toBe(400);
  });
});

describe('schema edits', () => {
  it('refuses to delete a label in use and changes nothing', async () => {
    await save([box]);
    const response = await send(`/projects/${project.id}`, 'PATCH', { labels: [person] });
    expect(response.status).toBe(409);
    expect((await response.json()).error).toContain('car');
    expect((await (await fetch(`${base}/projects/${project.id}`)).json()).labels).toHaveLength(2);
  });

  it('allows deleting an unused label', async () => {
    await save([pose]);
    expect((await send(`/projects/${project.id}`, 'PATCH', { labels: [person] })).status).toBe(200);
  });

  it('prunes removed keypoints, and annotations left with no geometry', async () => {
    const onlyHead: Annotation = { ...pose, id: 'a3', keypoints: { head: { x: 1, y: 1, v: 2 } } };
    await save([pose, onlyHead]);
    const withoutHead: Label = { ...person, keypoints: [{ id: 'foot', name: 'foot' }], skeleton: [] };
    const response = await send(`/projects/${project.id}`, 'PATCH', { labels: [withoutHead, car] });
    expect((await response.json()).prunedKeypoints).toBe(2);
    const image: ImageDetail = await (await fetch(`${base}/images/${imageId}`)).json();
    expect(image.annotations).toEqual([{ ...pose, keypoints: { foot: { x: 9, y: 9, v: 1 } } }]);
  });

  it('rejects bad schemas', async () => {
    const badEdge = { ...person, skeleton: [['head', 'nope']] };
    expect((await send(`/projects/${project.id}`, 'PATCH', { labels: [badEdge] })).status).toBe(400);
    expect((await send(`/projects/${project.id}`, 'PATCH', { labels: [car, { ...car, id: 'x' }] })).status).toBe(400);
  });
});

describe('export', () => {
  it('exports COCO for the chosen scope', async () => {
    await save([box]);
    const done = await (await fetch(`${base}/projects/${project.id}/export?format=coco`)).json();
    expect(done.images).toHaveLength(0);
    const annotated = await (await fetch(`${base}/projects/${project.id}/export?format=coco&scope=annotated`)).json();
    expect(annotated.annotations).toHaveLength(1);
  });

  it('streams a YOLO zip, with images when asked', async () => {
    await save([box]);
    const fetchZip = async (includeImages: string) => {
      const response = await fetch(`${base}/projects/${project.id}/export?format=yolo&scope=annotated&includeImages=${includeImages}`);
      expect(response.status).toBe(200);
      expect(response.headers.get('content-disposition')).toContain('.zip');
      return Buffer.from(await response.arrayBuffer()).toString('latin1');
    };
    const without = await fetchZip('0');
    expect(without.slice(0, 2)).toBe('PK');
    expect(without).toContain('labels/a.txt');
    expect(without).not.toContain('images/a.png');
    expect(await fetchZip('1')).toContain('images/a.png');
  });

  it('rejects an unknown format', async () => {
    expect((await fetch(`${base}/projects/${project.id}/export?format=voc`)).status).toBe(400);
  });
});

describe('annotation import', () => {
  const importFile = async (projectId: string, data: unknown, fields: Record<string, string> = {}) => {
    const form = new FormData();
    form.append('file', new Blob([JSON.stringify(data)], { type: 'application/json' }), 'labels.json');
    for (const [key, value] of Object.entries(fields)) form.append(key, value);
    return fetch(`${base}/projects/${projectId}/import`, { method: 'POST', body: form });
  };
  const coco = (fileName: string) => ({
    images: [{ id: 1, file_name: fileName, width: 100, height: 50 }],
    annotations: [{ id: 1, image_id: 1, category_id: 1, bbox: [10, 10, 20, 20] }],
    categories: [{ id: 1, name: 'bottle' }],
  });

  it('previews with dryRun, then writes boxes and creates the labels they need', async () => {
    const boxes: Project = await (await send('/projects', 'POST', { name: 'import', mode: 'bbox' })).json();
    const { uploaded } = await upload(boxes.id, [['photo.jpg', makePng(100, 50)]]);

    const preview = await importFile(boxes.id, coco('some/folder/photo.jpg'), { dryRun: '1' });
    expect(preview.status).toBe(200);
    const previewed = await preview.json();
    expect(previewed.dryRun).toBe(true);
    expect(previewed.summary).toMatchObject({ format: 'coco', matched: 1, imagesWritten: 1, annotations: 1, newLabels: ['bottle'] });
    expect(previewed.project.labels).toHaveLength(0);
    expect((await (await fetch(`${base}/images/${uploaded[0].id}`)).json()).annotations).toHaveLength(0);

    const result = await (await importFile(boxes.id, coco('some/folder/photo.jpg'))).json();
    expect(result.dryRun).toBe(false);
    expect(result.project.labels.map((l: Label) => l.name)).toEqual(['bottle']);
    const image: ImageDetail = await (await fetch(`${base}/images/${uploaded[0].id}`)).json();
    expect(image.annotations).toHaveLength(1);
    expect(image.annotations[0]).toMatchObject({ labelId: result.project.labels[0].id, bbox: { x: 10, y: 10, w: 20, h: 20 } });
    expect(image.status).toBe('todo');

    // A second import without overwrite leaves the saved work alone.
    const again = await (await importFile(boxes.id, coco('photo.jpg'))).json();
    expect(again.summary).toMatchObject({ alreadyAnnotated: 1, imagesWritten: 0 });
  });

  it('imports keypoints into a keypoint project and marks images done when asked', async () => {
    const data = {
      images: [{ id: 1, file_name: 'a.png', width: 100, height: 50 }],
      annotations: [{ id: 1, image_id: 1, category_id: 1, bbox: [1, 1, 10, 10], keypoints: [5, 5, 2, 9, 9, 1] }],
      categories: [{ id: 1, name: 'person', keypoints: ['head', 'foot'], skeleton: [[1, 2]] }],
    };
    const result = await (await importFile(project.id, data, { markDone: 'true' })).json();
    expect(result.summary).toMatchObject({ imagesWritten: 1, annotations: 1, newLabels: [], extendedLabels: [] });
    const image: ImageDetail = await (await fetch(`${base}/images/${imageId}`)).json();
    expect(image.status).toBe('done');
    expect(image.annotations[0].keypoints).toEqual({ head: { x: 5, y: 5, v: 2 }, foot: { x: 9, y: 9, v: 1 } });
  });

  it('rejects files that are not JSON or not a known format, without changing anything', async () => {
    const form = new FormData();
    form.append('file', new Blob(['not json']), 'x.json');
    const broken = await fetch(`${base}/projects/${project.id}/import`, { method: 'POST', body: form });
    expect(broken.status).toBe(400);
    expect((await importFile(project.id, { nothing: true })).status).toBe(400);
    expect((await importFile(project.id, { images: [], categories: 'x' })).status).toBe(400);
    const noFile = await fetch(`${base}/projects/${project.id}/import`, { method: 'POST', body: new FormData() });
    expect(noFile.status).toBe(400);
    expect((await importFile('missing', coco('a.png'))).status).toBe(404);
    expect(readdirSync(join(dataDir, 'tmp')).filter((f) => !f.startsWith('autobox'))).toHaveLength(0);
  });
});

describe('deletion', () => {
  it('removes image files with their project', async () => {
    const dir = join(dataDir, 'images', project.id);
    expect(existsSync(dir)).toBe(true);
    expect((await fetch(`${base}/projects/${project.id}`, { method: 'DELETE' })).status).toBe(204);
    expect(existsSync(dir)).toBe(false);
    expect((await fetch(`${base}/images/${imageId}`)).status).toBe(404);
  });

  it('removes one image file', async () => {
    expect((await fetch(`${base}/images/${imageId}`, { method: 'DELETE' })).status).toBe(204);
    expect(readdirSync(join(dataDir, 'images', project.id))).toEqual([]);
  });
});

describe('request safety', () => {
  it('rejects cross-site requests, including simple multipart uploads', async () => {
    const form = new FormData();
    form.append('files', new Blob([new Uint8Array(makePng(4, 4))]), 'x.png');
    const response = await fetch(`${base}/projects/${project.id}/images`, {
      method: 'POST',
      body: form,
      headers: { Origin: 'https://evil.example' },
    });
    expect(response.status).toBe(403);
  });

  it('allows same-origin requests', async () => {
    const host = new URL(base).host;
    expect((await send('/projects', 'POST', { name: 'ok' }, { Origin: `http://${host}` })).status).toBe(201);
  });

  it('returns JSON for malformed bodies and unknown routes', async () => {
    const malformed = await fetch(`${base}/projects`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{' });
    expect(malformed.status).toBe(400);
    expect(await malformed.json()).toEqual({ error: 'Invalid JSON' });
    expect((await fetch(`${base}/nope`)).status).toBe(404);
  });
});

describe('folder import', () => {
  let root: string;

  const write = (rel: string, data: Buffer | string) => {
    const path = join(root, rel);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, data);
  };
  const scanDone = async (sourceId: string): Promise<ScanStatus> => {
    for (;;) {
      const scan: ScanStatus = await (await fetch(`${base}/sources/${sourceId}/scan`)).json();
      if (scan.state !== 'running') return scan;
      await new Promise((r) => setTimeout(r, 10));
    }
  };
  const addSource = async (path = root): Promise<Source> => {
    const response = await send(`/projects/${project.id}/sources`, 'POST', { path });
    expect(response.status).toBe(202);
    const source: Source = await response.json();
    await scanDone(source.id);
    return source;
  };
  const rescan = async (sourceId: string, body: object = {}) => {
    expect((await send(`/sources/${sourceId}/scan`, 'POST', body)).status).toBe(202);
    return scanDone(sourceId);
  };
  const scanned = async (query = ''): Promise<ImageMeta[]> => {
    const all: ImageMeta[] = await (await fetch(`${base}/projects/${project.id}/images?${query}`)).json();
    return all.filter((i) => i.sourceId);
  };

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'labelapp-source-'));
  });
  afterEach(() => rmSync(root, { recursive: true, force: true }));

  it('finds images at every depth, by contents, and skips everything else', async () => {
    write('a.png', makePng(10, 10));
    write('sub/deep/er/b.png', makePng(20, 30));
    write('renamed.txt', makePng(5, 5));
    write('notes.png', 'not an image');
    write('.hidden/c.png', makePng(4, 4));
    write('.d.png', makePng(4, 4));
    const source = await addSource();
    const scan = await scanDone(source.id);
    expect(scan).toMatchObject({ state: 'done', added: 3, skipped: 1, errorCount: 0 });
    const name = basename(root);
    const images = await scanned();
    expect(images.map((i) => `${i.folder}/${i.filename}`).sort()).toEqual([`${name}/a.png`, `${name}/renamed.txt`, `${name}/sub/deep/er/b.png`].sort());
    expect(images.find((i) => i.filename === 'b.png')).toMatchObject({ width: 20, height: 30, status: 'todo', fileState: 'ok' });
  });

  it('does not follow symbolic links', async () => {
    const outside = mkdtempSync(join(tmpdir(), 'labelapp-outside-'));
    try {
      writeFileSync(join(outside, 'secret.png'), makePng(4, 4));
      write('a.png', makePng(4, 4));
      symlinkSync(outside, join(root, 'linked-dir'));
      symlinkSync(join(outside, 'secret.png'), join(root, 'linked.png'));
      symlinkSync(root, join(root, 'loop'));
      await addSource();
      expect((await scanned()).map((i) => i.filename)).toEqual(['a.png']);
    } finally {
      rmSync(outside, { recursive: true, force: true });
    }
  });

  it('sorts in natural order', async () => {
    for (const n of [10, 2, 1]) write(`img${n}.png`, makePng(4, 4));
    await addSource();
    expect((await scanned()).map((i) => i.filename)).toEqual(['img1.png', 'img2.png', 'img10.png']);
  });

  it('serves files by detected type, whatever the name, and handles awkward names', async () => {
    write('page.html', makePng(4, 4));
    write('sub dir/ảnh #1 100%?.png', makePng(4, 4));
    await addSource();
    for (const image of await scanned()) {
      const file = await fetch(`${base}/images/${image.id}/file`);
      expect(file.status).toBe(200);
      expect(file.headers.get('content-type')).toBe('image/png');
    }
    expect(await scanned()).toHaveLength(2);
  });

  it('gives clear errors for bad folders', async () => {
    write('file.png', makePng(4, 4));
    const post = (path: string) => send(`/projects/${project.id}/sources`, 'POST', { path });
    expect((await post('relative/path')).status).toBe(400);
    expect((await (await post(join(root, 'nope'))).json()).error).toContain('does not exist');
    expect((await (await post(join(root, 'file.png'))).json()).error).toContain('Not a folder');
  });

  it('rejects overlapping folders', async () => {
    write('sub/a.png', makePng(4, 4));
    await addSource(join(root, 'sub'));
    const post = (path: string) => send(`/projects/${project.id}/sources`, 'POST', { path });
    expect((await post(root)).status).toBe(409);
    expect((await post(join(root, 'sub'))).status).toBe(409);
  });

  it('keeps source names unique so folder trees cannot merge', async () => {
    const other = mkdtempSync(join(tmpdir(), 'labelapp-other-'));
    try {
      mkdirSync(join(root, 'uploads'));
      mkdirSync(join(root, 'x'));
      mkdirSync(join(other, 'x'));
      const names = [(await addSource(join(root, 'uploads'))).name, (await addSource(join(root, 'x'))).name, (await addSource(join(other, 'x'))).name];
      expect(names).toEqual(['uploads_2', 'x', 'x_2']);
    } finally {
      rmSync(other, { recursive: true, force: true });
    }
  });

  it('rescans safely: adds new, flags missing and changed, recovers, keeps annotations', async () => {
    write('keep.png', makePng(50, 50));
    write('gone.png', makePng(50, 50));
    write('resized.png', makePng(50, 50));
    const source = await addSource();
    const before = await scanned();
    const id = (name: string) => before.find((i) => i.filename === name)!.id;
    await send(`/images/${id('gone.png')}/annotations`, 'PUT', { annotations: [box] });

    expect(await rescan(source.id)).toMatchObject({ added: 0, unchanged: 3, missing: 0 });

    write('new.png', makePng(50, 50));
    rmSync(join(root, 'gone.png'));
    write('resized.png', makePng(70, 50));
    const second = await rescan(source.id);
    expect(second).toMatchObject({ added: 1, missing: 1, changed: 1, unchanged: 1 });

    const after = await scanned();
    expect(after.find((i) => i.filename === 'gone.png')).toMatchObject({ fileState: 'missing', annotationCount: 1 });
    expect(after.find((i) => i.filename === 'resized.png')).toMatchObject({ fileState: 'changed', width: 50 });

    // Unavailable images: annotations are read-only, but labels and status still work.
    const blocked = await send(`/images/${id('gone.png')}/annotations`, 'PUT', { annotations: [] });
    expect(blocked.status).toBe(409);
    const allowed = await send(`/images/${id('gone.png')}/annotations`, 'PUT', { imageLabels: ['car'], status: 'done' });
    expect(allowed.status).toBe(200);

    write('gone.png', makePng(50, 50));
    write('resized.png', makePng(50, 50));
    const third = await rescan(source.id);
    expect(third.recovered).toBe(2);
    const final = await scanned();
    expect(final.every((i) => i.fileState === 'ok')).toBe(true);
    expect(final.find((i) => i.filename === 'gone.png')!.annotationCount).toBe(1);
    expect(final).toHaveLength(4);
  });

  it('marks everything missing when the folder has moved, and recovers when it returns', async () => {
    write('sub/a.png', makePng(4, 4));
    const source = await addSource();
    const moved = `${root}-moved`;
    renameSync(root, moved);
    try {
      expect(await rescan(source.id)).toMatchObject({ missing: 1, errorCount: 1 });
    } finally {
      renameSync(moved, root);
    }
    expect(await rescan(source.id)).toMatchObject({ recovered: 1 });
  });

  it('never touches files on disk when deleting, and remembers deleted images', async () => {
    write('a.png', makePng(4, 4));
    write('b.png', makePng(4, 4));
    const source = await addSource();
    const a = (await scanned()).find((i) => i.filename === 'a.png')!;

    expect((await fetch(`${base}/images/${a.id}`, { method: 'DELETE' })).status).toBe(204);
    expect(existsSync(join(root, 'a.png'))).toBe(true);
    expect(await rescan(source.id)).toMatchObject({ added: 0, excluded: 1 });
    expect((await scanned()).map((i) => i.filename)).toEqual(['b.png']);

    expect(await rescan(source.id, { includeExcluded: true })).toMatchObject({ added: 1 });
    expect(await scanned()).toHaveLength(2);

    expect((await fetch(`${base}/sources/${source.id}`, { method: 'DELETE' })).status).toBe(204);
    expect(await scanned()).toHaveLength(0);
    expect(readdirSync(root).sort()).toEqual(['a.png', 'b.png']);

    await addSource();
    expect((await fetch(`${base}/projects/${project.id}`, { method: 'DELETE' })).status).toBe(204);
    expect(readdirSync(root).sort()).toEqual(['a.png', 'b.png']);
  });

  it('can be cancelled, keeping what it added', async () => {
    for (let i = 0; i < 300; i++) write(`f${i}.png`, makePng(4, 4));
    const response = await send(`/projects/${project.id}/sources`, 'POST', { path: root });
    const source: Source = await response.json();
    expect((await send(`/sources/${source.id}/scan`, 'POST', {})).status).toBe(409);
    expect((await fetch(`${base}/sources/${source.id}/scan`, { method: 'DELETE' })).status).toBe(204);
    const scan = await scanDone(source.id);
    expect(scan.state).toBe('cancelled');
    expect((await scanned()).length).toBe(scan.added);
    expect(await rescan(source.id)).toMatchObject({ state: 'done', added: 300 - scan.added });
    expect(await scanned()).toHaveLength(300);
  });

  it('restricts import when not on loopback', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'labelapp-remote-'));
    const { app, store } = createApp(dir, { host: '0.0.0.0' });
    const remote = await new Promise<Server>((resolve) => {
      const s = app.listen(0, '127.0.0.1', () => resolve(s));
    });
    const limited = createApp(join(dir, 'limited'), { host: '0.0.0.0', importRoots: [root] });
    const limitedServer = await new Promise<Server>((resolve) => {
      const s = limited.app.listen(0, '127.0.0.1', () => resolve(s));
    });
    try {
      const post = async (s: Server, path: string) => {
        const url = `http://127.0.0.1:${(s.address() as AddressInfo).port}/api`;
        const p = await (await fetch(`${url}/projects`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: 'r' }) })).json();
        return fetch(`${url}/projects/${p.id}/sources`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }) });
      };
      mkdirSync(join(root, 'inside'));
      expect((await post(remote, root)).status).toBe(403);
      expect((await post(limitedServer, join(root, 'inside'))).status).toBe(202);
      expect((await post(limitedServer, tmpdir())).status).toBe(403);
      symlinkSync(tmpdir(), join(root, 'escape'));
      expect((await post(limitedServer, join(root, 'escape'))).status).toBe(403);
    } finally {
      remote.close();
      limitedServer.close();
      store.close();
      limited.store.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  describe('sharing a project', () => {
    let copy: string;
    let uploads: string;
    beforeEach(() => {
      copy = mkdtempSync(join(tmpdir(), 'labelapp-copy-'));
      uploads = mkdtempSync(join(tmpdir(), 'labelapp-uploads-'));
    });
    afterEach(() => {
      rmSync(copy, { recursive: true, force: true });
      rmSync(uploads, { recursive: true, force: true });
    });

    const exportBundle = async (projectId = project.id) => {
      const response = await fetch(`${base}/projects/${projectId}/export?format=project`);
      expect(response.status).toBe(200);
      expect(response.headers.get('content-disposition')).toContain('.labelapp');
      return Buffer.from(await response.arrayBuffer());
    };
    const open = (bundle: Buffer, options: object, dryRun: boolean, url = base) => {
      const form = new FormData();
      form.append('file', new Blob([new Uint8Array(bundle)]), 'test.labelapp');
      form.append('options', JSON.stringify(options));
      form.append('dryRun', dryRun ? '1' : '0');
      return fetch(`${url}/projects/open`, { method: 'POST', body: form });
    };
    const imagesOf = async (projectId: string): Promise<ImageMeta[]> => (await fetch(`${base}/projects/${projectId}/images`)).json();
    const sourcesOf = async (projectId: string): Promise<Source[]> => (await fetch(`${base}/projects/${projectId}/sources`)).json();

    /** A project with an upload, a scanned folder with annotations, image labels, statuses and a deleted image. */
    const prepare = async () => {
      write('keep.png', makePng(50, 50));
      write('sub/resized.png', makePng(50, 50));
      write('gone.png', makePng(50, 50));
      write('deleted.png', makePng(50, 50));
      write('todo.png', makePng(50, 50));
      await addSource();
      const id = async (name: string) => (await scanned()).find((i) => i.filename === name)!.id;
      expect((await send(`/images/${await id('keep.png')}/annotations`, 'PUT', { annotations: [box], status: 'done', imageLabels: ['car'] })).status).toBe(200);
      expect((await send(`/images/${await id('gone.png')}/annotations`, 'PUT', { annotations: [box] })).status).toBe(200);
      expect((await send(`/images/${await id('resized.png')}/annotations`, 'PUT', { annotations: [pose] })).status).toBe(200);
      expect((await send(`/images/${imageId}/annotations`, 'PUT', { annotations: [box], status: 'done' })).status).toBe(200);
      expect((await fetch(`${base}/images/${await id('deleted.png')}`, { method: 'DELETE' })).status).toBe(204);
    };

    it('exports every image, todo included, with folders and deleted paths but no files', async () => {
      await prepare();
      const bundle = JSON.parse(gunzipSync(await exportBundle()).toString());
      expect(bundle).toMatchObject({ kind: 'labelapp-project', version: 1, project: { name: 'test', mode: 'keypoint', labels: [person, car] } });
      expect(bundle.sources).toEqual([{ key: 's1', name: basename(root), originalPath: root, exclusions: ['deleted.png'] }]);
      const byPath = Object.fromEntries(bundle.images.map((i: { source: string; relPath: string }) => [`${i.source}:${i.relPath}`, i]));
      expect(Object.keys(byPath).sort()).toEqual(['s1:gone.png', 's1:keep.png', 's1:sub/resized.png', 's1:todo.png', 'uploads:a.png']);
      expect(byPath['s1:keep.png']).toMatchObject({ status: 'done', imageLabels: ['car'], annotations: [box], autoBoxed: false, width: 50 });
      expect(byPath['s1:todo.png']).toMatchObject({ status: 'todo', annotations: [] });
    });

    it('opens on another path, matching files by path and keeping annotations read-only where files differ', async () => {
      await prepare();
      const bundle = await exportBundle();
      cpSync(root, copy, { recursive: true });
      rmSync(join(copy, 'gone.png'));
      writeFileSync(join(copy, 'sub/resized.png'), makePng(70, 50));
      writeFileSync(join(copy, 'new.png'), makePng(50, 50));
      writeFileSync(join(uploads, 'a.png'), makePng(100, 50));

      const preview = await (await open(bundle, { paths: { s1: copy, uploads } }, true)).json();
      expect(preview).toMatchObject({ name: 'test', mode: 'keypoint', labelCount: 2, imageCount: 5 });
      expect(preview.folders).toMatchObject([
        { key: 's1', name: basename(root), originalPath: root, imageCount: 4, path: copy, error: null, found: 3, checked: 4 },
        { key: 'uploads', imageCount: 1, path: uploads, error: null, found: 1, checked: 1 },
      ]);
      expect(await imagesOf(project.id)).toHaveLength(5);

      const response = await open(bundle, { name: 'shared', paths: { s1: copy, uploads } }, false);
      expect(response.status).toBe(201);
      const result = await response.json();
      expect(result).toMatchObject({ project: { name: 'shared', mode: 'keypoint', labels: [person, car] }, warnings: [] });
      const sources = await sourcesOf(result.project.id);
      expect(sources.map((s) => [s.name, s.path])).toEqual([
        [basename(root), copy],
        [basename(uploads), uploads],
      ].sort());
      for (const source of sources) expect((await scanDone(source.id)).state).toBe('done');

      const images = await imagesOf(result.project.id);
      const find = (name: string) => images.find((i) => i.filename === name)!;
      expect(images.map((i) => i.filename).sort()).toEqual(['a.png', 'gone.png', 'keep.png', 'new.png', 'resized.png', 'todo.png']);
      expect(find('keep.png')).toMatchObject({ fileState: 'ok', status: 'done', imageLabels: ['car'], annotationCount: 1, folder: basename(root) });
      expect(find('resized.png')).toMatchObject({ fileState: 'changed', width: 50, annotationCount: 1, folder: `${basename(root)}/sub` });
      expect(find('gone.png')).toMatchObject({ fileState: 'missing', annotationCount: 1 });
      expect(find('new.png')).toMatchObject({ fileState: 'ok', status: 'todo' });
      expect(find('a.png')).toMatchObject({ fileState: 'ok', status: 'done', annotationCount: 1, folder: basename(uploads) });
      expect((await send(`/images/${find('gone.png').id}/annotations`, 'PUT', { annotations: [] })).status).toBe(409);
      expect((await send(`/images/${find('keep.png').id}/annotations`, 'PUT', { annotations: [] })).status).toBe(200);
      expect((await fetch(`${base}/images/${find('keep.png').id}/file`)).status).toBe(200);

      // The original is untouched.
      expect((await imagesOf(project.id)).find((i) => i.filename === 'keep.png')).toMatchObject({ annotationCount: 1, fileState: 'ok' });
    });

    it('leaves out folders with no path, with a warning', async () => {
      await prepare();
      const result = await (await open(await exportBundle(), { paths: { s1: root } }, false)).json();
      expect(result.warnings).toEqual(['1 image in "Uploaded images" left out, because no folder was chosen for it.']);
      expect(await imagesOf(result.project.id)).toHaveLength(4);
    });

    it('reports unusable paths in a dry run and refuses them otherwise', async () => {
      await prepare();
      const bundle = await exportBundle();
      mkdirSync(join(copy, 'inner'));
      const preview = await (await open(bundle, { paths: { s1: copy, uploads: join(copy, 'inner') } }, true)).json();
      expect(preview.folders[0]).toMatchObject({ error: null, found: 0 });
      expect(preview.folders[1].error).toContain('overlaps');
      expect((await open(bundle, { paths: { s1: copy, uploads: join(copy, 'inner') } }, false)).status).toBe(409);
      expect((await open(bundle, { paths: { s1: join(copy, 'nope') } }, false)).status).toBe(400);
      expect((await open(bundle, { paths: { s1: 'relative' } }, false)).status).toBe(400);

      const before = (await (await fetch(`${base}/projects`)).json()).length;
      expect((await open(Buffer.from('{"images":[]}'), {}, false)).status).toBe(400);
      expect((await open(gzipSync('not json'), {}, true)).status).toBe(400);
      const wrongVersion = JSON.parse(gunzipSync(bundle).toString());
      wrongVersion.version = 2;
      expect((await (await open(Buffer.from(JSON.stringify(wrongVersion)), {}, true)).json()).error).toContain('version 2');
      const escaping = JSON.parse(gunzipSync(bundle).toString());
      escaping.images[0].relPath = '../outside.png';
      expect((await open(Buffer.from(JSON.stringify(escaping)), {}, true)).status).toBe(400);
      const broken = JSON.parse(gunzipSync(bundle).toString());
      broken.images.find((i: { annotations: unknown[] }) => i.annotations.length > 0).annotations[0].labelId = 'nobody';
      expect((await (await open(Buffer.from(JSON.stringify(broken)), {}, true)).json()).error).toContain('unknown label');
      expect((await (await fetch(`${base}/projects`)).json()).length).toBe(before);
    });

    it('applies IMPORT_ROOTS to chosen folders', async () => {
      await prepare();
      const bundle = await exportBundle();
      const dir = mkdtempSync(join(tmpdir(), 'labelapp-remote-'));
      const remote = createApp(dir, { host: '0.0.0.0' });
      const remoteServer = await new Promise<Server>((resolve) => {
        const s = remote.app.listen(0, '127.0.0.1', () => resolve(s));
      });
      try {
        const url = `http://127.0.0.1:${(remoteServer.address() as AddressInfo).port}/api`;
        const preview = await (await open(bundle, { paths: { s1: root } }, true, url)).json();
        expect(preview.folders[0].error).toContain('IMPORT_ROOTS');
        expect((await open(bundle, { paths: { s1: root } }, false, url)).status).toBe(403);
      } finally {
        remoteServer.close();
        remote.store.close();
        rmSync(dir, { recursive: true, force: true });
      }
    });

    it('points a source at another folder, recovering its images', async () => {
      await prepare();
      const opened = await (await open(await exportBundle(), { paths: { s1: copy } }, false)).json();
      const [source] = await sourcesOf(opened.project.id);
      await scanDone(source.id);
      expect((await imagesOf(opened.project.id)).every((i) => i.fileState === 'missing')).toBe(true);

      expect((await send(`/sources/${source.id}`, 'PATCH', { path: join(root, 'nope') })).status).toBe(400);
      const moved = await send(`/sources/${source.id}`, 'PATCH', { path: root });
      expect(moved.status).toBe(202);
      expect(await moved.json()).toMatchObject({ path: root, name: basename(root) });
      expect(await scanDone(source.id)).toMatchObject({ recovered: 4, added: 0, excluded: 1 });
      expect((await imagesOf(opened.project.id)).every((i) => i.fileState === 'ok')).toBe(true);
    });
  });

  describe('group actions', () => {
    const bulk = (body: object) => send(`/projects/${project.id}/images/bulk`, 'POST', body);
    let top: string;

    beforeEach(async () => {
      write('cats/1.png', makePng(4, 4));
      write('cats/big/2.png', makePng(4, 4));
      write('cats2/3.png', makePng(4, 4));
      await addSource();
      top = basename(root);
    });

    const labelsByName = async () => Object.fromEntries((await scanned()).map((i) => [i.filename, i.imageLabels]));

    it('builds a folder tree with recursive counts', async () => {
      const tree: FolderNode[] = await (await fetch(`${base}/projects/${project.id}/folders`)).json();
      expect(tree.find((n) => n.path === top)).toMatchObject({ direct: 0, total: 3, parent: null });
      expect(tree.find((n) => n.path === `${top}/cats`)).toMatchObject({ direct: 1, total: 2, parent: top });
      expect(tree.find((n) => n.path === 'uploads')).toMatchObject({ direct: 1, total: 1 });
    });

    it('labels a folder and its subfolders, matching whole path segments', async () => {
      const response = await bulk({ selector: { folder: `${top}/cats`, recursive: true }, addLabels: ['car'] });
      expect(await response.json()).toMatchObject({ matched: 2 });
      expect(await labelsByName()).toEqual({ '1.png': ['car'], '2.png': ['car'], '3.png': [] });
    });

    it('can exclude subfolders and individual images', async () => {
      await bulk({ selector: { folder: `${top}/cats`, recursive: false }, addLabels: ['car'] });
      expect(await labelsByName()).toEqual({ '1.png': ['car'], '2.png': [], '3.png': [] });

      const images = await scanned();
      const except = [images.find((i) => i.filename === '3.png')!.id];
      await bulk({ selector: { folder: top, recursive: true, except }, addLabels: ['person'] });
      expect(await labelsByName()).toEqual({ '1.png': ['car', 'person'], '2.png': ['person'], '3.png': [] });
    });

    it('reports only images actually changed, which is what makes undo exact', async () => {
      await bulk({ selector: { folder: `${top}/cats`, recursive: false }, addLabels: ['car'] });
      const result = await (await bulk({ selector: { folder: `${top}/cats`, recursive: true }, addLabels: ['car'] })).json();
      expect(result.matched).toBe(2);
      expect(result.changed).toHaveLength(1);
      await bulk({ selector: { imageIds: result.changed }, removeLabels: ['car'] });
      expect(await labelsByName()).toEqual({ '1.png': ['car'], '2.png': [], '3.png': [] });
    });

    it('filters images by folder and label', async () => {
      await bulk({ selector: { folder: `${top}/cats`, recursive: false }, addLabels: ['car'] });
      const q = new URLSearchParams({ folder: `${top}/cats` });
      expect(await scanned(q.toString())).toHaveLength(2);
      q.set('recursive', '0');
      expect(await scanned(q.toString())).toHaveLength(1);
      expect(await scanned('label=car')).toHaveLength(1);
    });

    it('sets status as a group and ignores images from other projects', async () => {
      const other: Project = await (await send('/projects', 'POST', { name: 'other', labels: [car] })).json();
      const { uploaded } = await upload(other.id, [['x.png', makePng(4, 4)]]);
      const result = await (await bulk({ selector: { imageIds: [uploaded[0].id, imageId] }, status: 'done' })).json();
      expect(result.changed).toEqual([imageId]);
      expect((await (await fetch(`${base}/images/${uploaded[0].id}`)).json()).status).toBe('todo');
    });

    it('validates requests', async () => {
      const selector = { folder: top, recursive: true };
      expect((await bulk({ selector, addLabels: ['car'], removeLabels: ['car'] })).status).toBe(400);
      expect((await bulk({ selector, addLabels: ['nope'] })).status).toBe(422);
    });

    it('blocks deleting a label used only as an image label', async () => {
      await bulk({ selector: { folder: top, recursive: true }, addLabels: ['car'] });
      const response = await send(`/projects/${project.id}`, 'PATCH', { labels: [person] });
      expect(response.status).toBe(409);
    });

    it('does not let image labels pull todo images into COCO or YOLO exports', async () => {
      await bulk({ selector: { folder: top, recursive: true }, addLabels: ['car'] });
      const coco = await (await fetch(`${base}/projects/${project.id}/export?format=coco&scope=annotated`)).json();
      expect(coco.images).toHaveLength(0);
    });

    it('exports image labels separately, keeping folders and listing unavailable files', async () => {
      await bulk({ selector: { folder: top, recursive: true }, addLabels: ['car'] });
      await bulk({ selector: { folder: `${top}/cats2`, recursive: true }, addLabels: ['person'] });
      const zipText = async (query: string) => {
        const response = await fetch(`${base}/projects/${project.id}/export?format=image_labels&${query}`);
        expect(response.status).toBe(200);
        return Buffer.from(await response.arrayBuffer());
      };
      const entries = async (query: string) => unzip(await zipText(query));

      const without = await entries('');
      expect(without['classes.txt']).toBe('person\ncar\n');
      expect(without['labels.csv'].split('\n').filter(Boolean).sort()).toEqual(
        ['image,label', `${top}/cats/1.png,car`, `${top}/cats/big/2.png,car`, `${top}/cats2/3.png,person`, `${top}/cats2/3.png,car`].sort(),
      );
      expect(Object.keys(without).some((k) => k.startsWith('images/'))).toBe(false);

      expect(await entries('scope=done')).toMatchObject({ 'labels.csv': 'image,label\n' });

      const withImages = await entries('includeImages=1');
      expect(Object.keys(withImages)).toContain(`images/${top}/cats/big/2.png`);

      rmSync(join(root, 'cats/1.png'));
      const degraded = await entries('includeImages=1');
      expect(degraded['export_warnings.txt']).toContain(`${top}/cats/1.png`);
      expect(degraded['labels.csv']).not.toContain('cats/1.png');
      expect(Object.keys(degraded)).not.toContain(`images/${top}/cats/1.png`);
      expect((await entries(''))['labels.csv']).toContain('cats/1.png');

      expect((await fetch(`${base}/projects/${project.id}/export?format=image_labels&scope=annotated`)).status).toBe(400);
      expect((await fetch(`${base}/projects/${project.id}/export?format=coco&scope=labeled`)).status).toBe(400);
    });

    it('quotes CSV values', async () => {
      const comma: Label = { ...car, id: 'comma', name: 'a, "b"', color: '#123456' };
      await send(`/projects/${project.id}`, 'PATCH', { labels: [person, car, comma] });
      await bulk({ selector: { folder: `${top}/cats`, recursive: false }, addLabels: ['comma'] });
      const response = await fetch(`${base}/projects/${project.id}/export?format=image_labels`);
      const entries = unzip(Buffer.from(await response.arrayBuffer()));
      expect(entries['labels.csv']).toContain(`${top}/cats/1.png,"a, ""b"""`);
    });

    it('leaves unavailable files out of a YOLO zip without failing it', async () => {
      await bulk({ selector: { folder: top, recursive: true }, status: 'done' });
      rmSync(join(root, 'cats/1.png'));
      const response = await fetch(`${base}/projects/${project.id}/export?format=yolo&includeImages=1`);
      const entries = unzip(Buffer.from(await response.arrayBuffer()));
      expect(entries['export_warnings.txt']).toContain('1.png');
      expect(Object.keys(entries)).not.toContain('labels/1.txt');
      expect(Object.keys(entries)).toContain('labels/2.txt');
    });
  });
});
