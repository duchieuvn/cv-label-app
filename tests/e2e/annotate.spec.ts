import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';
import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import type { Annotation, ImageDetail, ImageMeta, Label, Project, ProjectMode } from '../../shared/schema.ts';
import { makePng, unzip, withExifOrientation } from '../helpers.ts';

const CAR: Label = { id: 'car', name: 'car', color: '#3ddc97', keypoints: [], skeleton: [] };
const TRUCK: Label = { id: 'truck', name: 'truck', color: '#4d96ff', keypoints: [], skeleton: [] };
const BIRD: Label = {
  id: 'bird',
  name: 'bird',
  color: '#ff6b6b',
  keypoints: [
    { id: 'beak', name: 'beak' },
    { id: 'tail', name: 'tail' },
  ],
  skeleton: [['beak', 'tail']],
};

async function setup(request: APIRequestContext, imageCount = 1, width = 800, height = 600, mode: ProjectMode = 'bbox') {
  const labels = mode === 'bbox' ? [CAR, TRUCK] : [BIRD, CAR];
  const project: Project = await (await request.post('/api/projects', { data: { name: 'e2e', mode, labels } })).json();
  const multipart: Record<string, { name: string; mimeType: string; buffer: Buffer }> = {};
  const ids: string[] = [];
  for (let i = 0; i < imageCount; i++) {
    multipart.files = { name: `img${i}.png`, mimeType: 'image/png', buffer: makePng(width, height) };
    const result = await (await request.post(`/api/projects/${project.id}/images`, { multipart })).json();
    ids.push(result.uploaded[0].id);
  }
  return { project, ids };
}

async function open(page: Page, projectId: string, imageId: string) {
  await page.goto(`/#/projects/${projectId}/annotate/${imageId}`);
  await expect(page.getByTestId('canvas').locator('image')).toBeAttached();
  await expect(page.locator('.label-list button').first()).toBeVisible();
}

/** Converts image coordinates to page coordinates using the canvas's live transform. */
async function toPage(page: Page, x: number, y: number) {
  return page.getByTestId('canvas').evaluate(
    (svg, [px, py]) => {
      const rect = svg.getBoundingClientRect();
      const d = (svg as SVGSVGElement).dataset;
      return { x: rect.left + Number(d.tx) + px * Number(d.scale), y: rect.top + Number(d.ty) + py * Number(d.scale) };
    },
    [x, y],
  );
}

async function dragImage(page: Page, from: [number, number], to: [number, number]) {
  const a = await toPage(page, ...from);
  const b = await toPage(page, ...to);
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  await page.mouse.move((a.x + b.x) / 2, (a.y + b.y) / 2);
  await page.mouse.move(b.x, b.y);
  await page.mouse.up();
}

async function clickImage(page: Page, x: number, y: number, modifiers: { shift?: boolean } = {}) {
  const p = await toPage(page, x, y);
  if (modifiers.shift) await page.keyboard.down('Shift');
  await page.mouse.click(p.x, p.y);
  if (modifiers.shift) await page.keyboard.up('Shift');
}

async function saved(page: Page, request: APIRequestContext, imageId: string): Promise<Annotation[]> {
  // Mouse and keyboard calls can resolve before the page handles them, and until it does the
  // indicator still reads "Saved" from the previous edit. Two frames guarantee input has been processed.
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  await expect(page.getByTestId('save-state')).toHaveText('Saved');
  const image: ImageDetail = await (await request.get(`/api/images/${imageId}`)).json();
  return image.annotations;
}

const near = (actual: number, expected: number) => expect(Math.abs(actual - expected)).toBeLessThan(2);

test('draws, resizes and moves a box, with undo and persistence', async ({ page, request }) => {
  const { project, ids } = await setup(request);
  await open(page, project.id, ids[0]);

  await dragImage(page, [100, 100], [300, 250]);
  await expect(page.getByTestId('box')).toHaveCount(1);
  let [a] = await saved(page, request, ids[0]);
  expect(a.labelId).toBe('car');
  near(a.bbox!.x, 100);
  near(a.bbox!.y, 100);
  near(a.bbox!.w, 200);
  near(a.bbox!.h, 150);

  // Resizing works from the Box tool without drawing a new box.
  await dragImage(page, [300, 250], [400, 300]);
  await expect(page.getByTestId('box')).toHaveCount(1);
  [a] = await saved(page, request, ids[0]);
  near(a.bbox!.w, 300);
  near(a.bbox!.h, 200);

  await page.keyboard.press('v');
  await dragImage(page, [200, 200], [250, 220]);
  [a] = await saved(page, request, ids[0]);
  near(a.bbox!.x, 150);
  near(a.bbox!.y, 120);

  // One drag is one undo step.
  await page.keyboard.press('Control+z');
  [a] = await saved(page, request, ids[0]);
  near(a.bbox!.x, 100);
  await page.keyboard.press('Control+Shift+z');
  [a] = await saved(page, request, ids[0]);
  near(a.bbox!.x, 150);

  await page.reload();
  await expect(page.getByTestId('box')).toHaveCount(1);
  await page.keyboard.press('Delete'); // nothing selected after reload
  await expect(page.getByTestId('box')).toHaveCount(1);
});

test('ignores tiny accidental drags and clamps to the image', async ({ page, request }) => {
  const { project, ids } = await setup(request);
  await open(page, project.id, ids[0]);

  await clickImage(page, 400, 300);
  await expect(page.getByTestId('box')).toHaveCount(0);

  await dragImage(page, [700, 500], [5000, 5000]);
  const [a] = await saved(page, request, ids[0]);
  near(a.bbox!.x + a.bbox!.w, 800);
  near(a.bbox!.y + a.bbox!.h, 600);
  expect(a.bbox!.x + a.bbox!.w).toBeLessThanOrEqual(800);
});

test('Esc cancels a drag in progress without committing it', async ({ page, request }) => {
  const { project, ids } = await setup(request);
  await open(page, project.id, ids[0]);
  const a = await toPage(page, 100, 100);
  const b = await toPage(page, 300, 300);
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  await page.mouse.move(b.x, b.y);
  await page.keyboard.press('Escape');
  await page.mouse.up();
  await expect(page.getByTestId('box')).toHaveCount(0);
});

test('a bounding box project shows only what is about boxes', async ({ page, request }) => {
  const { project, ids } = await setup(request);
  const tools = page.getByRole('group', { name: 'Tools' }).getByRole('button');
  await open(page, project.id, ids[0]);

  await expect(tools).toHaveText([/^Select/, /^Box/]);
  await expect(page.getByRole('button', { name: /^Box/ })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByText('Nothing annotated yet. Draw a box.')).toBeVisible();

  // The keypoint shortcut does nothing here, so the next drag still draws a box.
  await page.keyboard.press('k');
  await expect(page.getByRole('button', { name: /^Box/ })).toHaveAttribute('aria-pressed', 'true');
  await dragImage(page, [100, 100], [300, 300]);
  await expect(page.getByTestId('box')).toHaveCount(1);
  await expect(page.getByRole('button', { name: /^Box/ })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('.details .mono')).toContainText('x 100');
  await expect(page.locator('.keypoint-list')).toHaveCount(0);

  await page.keyboard.press('?');
  const help = page.getByRole('dialog', { name: 'Keyboard shortcuts' });
  await expect(help).toContainText('Select, Box tool');
  await expect(help).not.toContainText(/keypoint/i);
  await page.keyboard.press('Escape');

  await page.goto(`/#/projects/${project.id}/settings`);
  await expect(page.getByTestId('project-mode')).toContainText('Bounding box');
  await expect(page.getByLabel('Label name')).toHaveCount(2);
  await expect(page.getByRole('heading', { name: /Keypoints/ })).toHaveCount(0);
  await expect(page.getByRole('heading', { name: /Skeleton/ })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Add COCO person' })).toHaveCount(0);

  await page.goto(`/#/projects/${project.id}`);
  await expect(page.getByTestId('project-mode')).toHaveText('Bounding box');
  await expect(page.getByRole('region', { name: 'Auto bounding box' })).toBeVisible();
  await expect(page.getByRole('tab', { name: /^Auto/ })).toBeVisible();
  await expect(page.getByLabel('Format').locator('option')).toContainText(['COCO detection JSON', 'YOLO detection (zip)']);
});

test('a keypoint project shows only what is about keypoints', async ({ page, request }) => {
  const { project, ids } = await setup(request, 1, 800, 600, 'keypoint');
  const tools = page.getByRole('group', { name: 'Tools' }).getByRole('button');
  await open(page, project.id, ids[0]);

  await expect(tools).toHaveText([/^Select/, /^Keypoint/]);
  await expect(page.getByRole('button', { name: /^Keypoint/ })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByText('Nothing annotated yet. Place a keypoint.')).toBeVisible();

  // The box shortcut does nothing here, and a click places the first point without any box.
  await page.keyboard.press('b');
  await expect(page.getByRole('button', { name: /^Keypoint/ })).toHaveAttribute('aria-pressed', 'true');
  await clickImage(page, 150, 150);
  await expect(page.getByText('Click to place "tail"')).toBeVisible();
  await clickImage(page, 250, 250, { shift: true });
  await expect(page.getByTestId('keypoint')).toHaveCount(2);
  await expect(page.getByTestId('box')).toHaveCount(0);
  const [a] = await saved(page, request, ids[0]);
  expect(a).toMatchObject({ labelId: 'bird', bbox: null });
  expect(a.keypoints.beak.v).toBe(2);
  expect(a.keypoints.tail.v).toBe(1);
  near(a.keypoints.tail.x, 250);

  // With every point placed the instance is finished, so the next click starts another.
  await clickImage(page, 500, 150);
  await expect(page.getByTestId('keypoint')).toHaveCount(3);
  expect(await saved(page, request, ids[0])).toHaveLength(2);

  await page.keyboard.press('?');
  const help = page.getByRole('dialog', { name: 'Keyboard shortcuts' });
  await expect(help).toContainText('Select, Keypoint tool');
  await expect(help).not.toContainText(/\bbox\b/i);
  await page.keyboard.press('Escape');

  await page.goto(`/#/projects/${project.id}/settings`);
  await expect(page.getByTestId('project-mode')).toContainText('Keypoints');
  await expect(page.getByRole('heading', { name: /Keypoints/ }).first()).toBeVisible();
  await expect(page.getByRole('button', { name: 'Add COCO person' })).toBeVisible();

  await page.goto(`/#/projects/${project.id}`);
  await expect(page.getByTestId('project-mode')).toHaveText('Keypoints');
  await expect(page.getByRole('tab', { name: /^All/ })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Auto bounding box' })).toHaveCount(0);
  const auto = page.getByRole('region', { name: 'Auto keypoints' });
  await expect(auto).toBeVisible();
  await expect(auto).toContainText('draws keypoints');
  await expect(auto).toContainText('Training needs at least 10 images with saved keypoints.');
  await expect(page.getByRole('tab', { name: /^Auto/ })).toBeVisible();
  await expect(page.getByLabel('Format').locator('option')).toContainText(['COCO keypoints JSON', 'YOLO pose (zip)']);
});

test('a box from before modes existed is kept but not shown in a keypoint project', async ({ page, request }) => {
  const { project, ids } = await setup(request, 1, 800, 600, 'keypoint');
  const bbox = { x: 100, y: 100, w: 200, h: 200 };
  const old: Annotation = { id: 'old', labelId: 'bird', bbox, keypoints: { beak: { x: 150, y: 150, v: 2 } } };
  await request.put(`/api/images/${ids[0]}/annotations`, { data: { annotations: [old] } });
  await open(page, project.id, ids[0]);

  await expect(page.getByTestId('keypoint')).toHaveCount(1);
  await expect(page.getByTestId('box')).toHaveCount(0);

  await dragImage(page, [150, 150], [170, 170]);
  const [a] = await saved(page, request, ids[0]);
  near(a.keypoints.beak.x, 170);
  expect(a.bbox).toEqual(bbox);
});

test('keypoint-only instances, skipping, occlusion toggle and dragging', async ({ page, request }) => {
  const { project, ids } = await setup(request, 1, 800, 600, 'keypoint');
  await open(page, project.id, ids[0]);

  await clickImage(page, 400, 300);
  let [a] = await saved(page, request, ids[0]);
  expect(a.bbox).toBeNull();
  expect(Object.keys(a.keypoints)).toEqual(['beak']);

  // Selecting by dragging an existing point moves only that point.
  await dragImage(page, [400, 300], [450, 320]);
  [a] = await saved(page, request, ids[0]);
  near(a.keypoints.beak.x, 450);
  near(a.keypoints.beak.y, 320);

  await page.keyboard.press('o');
  [a] = await saved(page, request, ids[0]);
  expect(a.keypoints.beak.v).toBe(1);

  await page.keyboard.press('ArrowRight');
  [a] = await saved(page, request, ids[0]);
  near(a.keypoints.beak.x, 451);

  // Deleting the only point of an instance removes the instance.
  await page.keyboard.press('Delete');
  expect(await saved(page, request, ids[0])).toEqual([]);
});

test('zoom and pan keep annotations aligned with image coordinates', async ({ page, request }) => {
  const { project, ids } = await setup(request);
  await open(page, project.id, ids[0]);

  const p = await toPage(page, 400, 300);
  await page.mouse.move(p.x, p.y);
  await page.mouse.wheel(0, -800);
  await expect.poll(() => page.getByTestId('canvas').getAttribute('data-scale').then(Number)).toBeGreaterThan(1.5);

  await page.keyboard.down('Space');
  await page.mouse.down();
  await page.mouse.move(p.x + 60, p.y + 40);
  await page.mouse.up();
  await page.keyboard.up('Space');
  await expect(page.getByTestId('box')).toHaveCount(0);

  await dragImage(page, [350, 280], [420, 330]);
  const [a] = await saved(page, request, ids[0]);
  near(a.bbox!.x, 350);
  near(a.bbox!.y, 280);
  near(a.bbox!.w, 70);
});

test('Done moves to the next todo image and an edit made just before leaving is saved', async ({ page, request }) => {
  const { project, ids } = await setup(request, 3);
  await open(page, project.id, ids[0]);

  await dragImage(page, [100, 100], [300, 300]);
  await page.keyboard.press('Control+Enter'); // no wait: tests the flush on navigation
  await expect(page).toHaveURL(new RegExp(ids[1]));

  await expect.poll(async () => (await (await request.get(`/api/images/${ids[0]}`)).json()).status).toBe('done');
  expect((await (await request.get(`/api/images/${ids[0]}`)).json()).annotations).toHaveLength(1);

  await page.keyboard.press('a');
  await expect(page).toHaveURL(new RegExp(ids[0]));
  await expect(page.getByTestId('box')).toHaveCount(1);
  await page.keyboard.press('d');
  await expect(page).toHaveURL(new RegExp(ids[1]));
  await expect(page.getByTestId('box')).toHaveCount(0);

  // Undo history does not cross images.
  await page.keyboard.press('Control+z');
  expect((await (await request.get(`/api/images/${ids[0]}`)).json()).annotations).toHaveLength(1);
});

test('a failed save is reported, kept, and retried', async ({ page, request }) => {
  const { project, ids } = await setup(request);
  await open(page, project.id, ids[0]);

  await page.route('**/annotations', (route) => route.abort());
  await dragImage(page, [100, 100], [300, 300]);
  await expect(page.getByTestId('save-state')).toContainText('Not saved');

  await page.unroute('**/annotations');
  await expect(page.getByTestId('save-state')).toHaveText('Saved', { timeout: 10_000 });
  expect((await (await request.get(`/api/images/${ids[0]}`)).json()).annotations).toHaveLength(1);
});

test('EXIF-rotated photos display upright and use the rotated frame', async ({ page, request, browser }) => {
  // Encode a JPEG in the browser: 80x40, left half red, right half blue.
  const scratch = await browser.newPage();
  const base64 = await scratch.evaluate(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 80;
    canvas.height = 40;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#ff0000';
    ctx.fillRect(0, 0, 40, 40);
    ctx.fillStyle = '#0000ff';
    ctx.fillRect(40, 0, 40, 40);
    return canvas.toDataURL('image/jpeg', 1).split(',')[1];
  });
  await scratch.close();

  // Orientation 6 means "rotate 90 degrees clockwise to display", so left (red) ends up on top.
  const buffer = withExifOrientation(Buffer.from(base64, 'base64'), 6);
  const project: Project = await (await request.post('/api/projects', { data: { name: 'exif', labels: [CAR] } })).json();
  const result = await (
    await request.post(`/api/projects/${project.id}/images`, {
      multipart: { files: { name: 'rotated.jpg', mimeType: 'image/jpeg', buffer } },
    })
  ).json();
  expect(result.uploaded[0]).toMatchObject({ width: 40, height: 80 });

  await open(page, project.id, result.uploaded[0].id);
  await page.getByTestId('canvas').locator('image').evaluate((el) => el.getBoundingClientRect());
  await page.waitForLoadState('networkidle');

  const colorAt = async (x: number, y: number) => {
    const p = await toPage(page, x, y);
    const shot = await page.screenshot({ clip: { x: p.x, y: p.y, width: 1, height: 1 } });
    return page.evaluate(async (data) => {
      const bitmap = await createImageBitmap(await (await fetch(`data:image/png;base64,${data}`)).blob());
      const canvas = new OffscreenCanvas(1, 1);
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(bitmap, 0, 0);
      return [...ctx.getImageData(0, 0, 1, 1).data].slice(0, 3);
    }, shot.toString('base64'));
  };

  const top = await colorAt(20, 20);
  const bottom = await colorAt(20, 60);
  expect(top[0]).toBeGreaterThan(200);
  expect(top[2]).toBeLessThan(80);
  expect(bottom[2]).toBeGreaterThan(200);
  expect(bottom[0]).toBeLessThan(80);
});

test('full journey through the UI: create, label, upload, annotate, export', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('New project name').fill('Journey');
  await page.getByLabel('Project mode').selectOption({ label: 'Keypoints' });
  await page.getByRole('button', { name: 'Create project' }).click();

  await expect(page.getByRole('heading', { name: 'Project settings' })).toBeVisible();
  await page.getByRole('button', { name: 'Add COCO person' }).click();
  await expect(page.getByLabel('Keypoint name')).toHaveCount(17);
  await page.getByRole('button', { name: 'Save changes' }).click();
  await expect(page.getByText('Saved.')).toBeVisible();
  await page.getByText('Back to project').click();

  await page.getByTestId('file-input').setInputFiles([
    { name: 'one.png', mimeType: 'image/png', buffer: makePng(640, 480) },
    { name: 'notes.png', mimeType: 'image/png', buffer: Buffer.from('not an image') },
  ]);
  await expect(page.getByText('1 file was not uploaded')).toBeVisible();
  await expect(page.getByRole('tab', { name: 'All (1)' })).toBeVisible();

  await page.getByRole('button', { name: 'Start annotating' }).click();
  await expect(page.getByTestId('canvas').locator('image')).toBeAttached();
  await expect(page.getByRole('button', { name: /^person/ }).first()).toBeVisible();

  await clickImage(page, 200, 120);
  await expect(page.getByText('Click to place "left_eye"')).toBeVisible();
  await page.getByRole('button', { name: /^Done/ }).click();
  await expect(page.getByText('Every image in this project is done.')).toBeVisible();
  await expect(page.getByTestId('save-state')).toHaveText('Saved');

  await page.getByRole('link', { name: /Journey/ }).click();
  await expect(page.getByRole('tab', { name: 'Done (1)' })).toBeVisible();
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('link', { name: 'Download' }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe('Journey_coco.json');

  const stream = await download.createReadStream();
  const chunks: Buffer[] = [];
  for await (const c of stream) chunks.push(c as Buffer);
  const coco = JSON.parse(Buffer.concat(chunks).toString());
  expect(coco.annotations).toHaveLength(1);
  expect(coco.annotations[0].num_keypoints).toBe(1);
  expect(coco.annotations[0].bbox[2]).toBeGreaterThanOrEqual(1);
  expect(coco.annotations[0].keypoints).toHaveLength(51);
  expect(coco.categories[0].keypoints).toHaveLength(17);
});

test('imports a folder, selects it, labels it as a group, undoes, and exports', async ({ page, request }) => {
  const root = mkdtempSync(join(tmpdir(), 'labelapp-e2e-source-'));
  try {
    mkdirSync(join(root, 'night/cam1'), { recursive: true });
    mkdirSync(join(root, 'day'));
    writeFileSync(join(root, 'night/a.png'), makePng(200, 100));
    writeFileSync(join(root, 'night/cam1/b.png'), makePng(200, 100));
    writeFileSync(join(root, 'day/c.png'), makePng(200, 100));
    writeFileSync(join(root, 'day/readme.txt'), 'not an image');

    const project: Project = await (await request.post('/api/projects', { data: { name: 'folders', labels: [CAR] } })).json();
    await page.goto(`/#/projects/${project.id}`);

    await page.getByLabel('Folder path').fill(join(root, 'nope'));
    await page.getByRole('button', { name: 'Add folder' }).click();
    await expect(page.getByRole('alert')).toContainText('does not exist');

    await page.getByLabel('Folder path').fill(root);
    await page.getByRole('button', { name: 'Add folder' }).click();
    await expect(page.getByTestId('scan-line')).toContainText('Scan finished');
    await expect(page.getByTestId('scan-line')).toContainText('3 added');
    await expect(page.getByTestId('scan-line')).toContainText('1 other files skipped');
    await expect(page.getByRole('tab', { name: 'All (3)' })).toBeVisible();

    // Folder filter, including subfolders by default.
    const top = basename(root);
    await page.locator(`[data-folder="${top}/night"]`).click();
    await expect(page.getByRole('tab', { name: 'All (2)' })).toBeVisible();
    await page.getByLabel('Include subfolders').uncheck();
    await expect(page.getByRole('tab', { name: 'All (1)' })).toBeVisible();
    await page.getByLabel('Include subfolders').check();
    await expect(page.getByRole('tab', { name: 'All (2)' })).toBeVisible();

    await page.getByRole('button', { name: /Select all in folder/ }).click();
    await expect(page.getByTestId('bulk-bar')).toContainText('2 images selected');

    // Removing one image from a folder selection keeps the rest.
    await page.getByLabel('Select b.png').uncheck();
    await expect(page.getByTestId('bulk-bar')).toContainText('1 image selected');
    await page.getByLabel('Select b.png').check();

    // Create a label in place and apply it.
    await page.getByLabel('Label for selected images').selectOption({ label: 'New label…' });
    await page.getByLabel('New label name').fill('night');
    await page.getByRole('button', { name: 'Add to 2 images' }).click();
    await expect(page.getByRole('status')).toContainText('Added night to 2 images');

    const list = async (): Promise<ImageMeta[]> => (await request.get(`/api/projects/${project.id}/images`)).json();
    const tagged = async () => (await list()).filter((i) => i.imageLabels.length > 0).map((i) => i.filename).sort();
    expect(await tagged()).toEqual(['a.png', 'b.png']);

    await page.getByRole('button', { name: 'Undo' }).click();
    await expect(page.getByRole('status')).toHaveCount(0);
    await expect.poll(tagged).toEqual([]);

    await page.getByRole('button', { name: 'Add to 2 images' }).click();
    await expect(page.getByRole('status')).toContainText('Added night to 2 images');

    // A label on the selection can be removed directly, and the removal can be undone.
    await page.getByRole('button', { name: 'Remove night from 2 images' }).click();
    await expect(page.getByRole('status')).toContainText('Removed night from 2 images');
    await expect.poll(tagged).toEqual([]);
    await expect(page.getByRole('button', { name: 'Remove night from 2 images' })).toHaveCount(0);
    await page.getByRole('button', { name: 'Undo' }).click();
    await expect(page.getByRole('button', { name: 'Remove night from 2 images' })).toBeVisible();

    // Each label created in place starts with a color no other label has.
    await page.getByLabel('Label for selected images').selectOption({ label: 'New label…' });
    await page.getByLabel('New label name').fill('dusk');
    await page.getByRole('button', { name: 'Add to 2 images' }).click();
    await expect(page.getByRole('status')).toContainText('Added dusk to 2 images');
    const labelColors = (await (await request.get(`/api/projects/${project.id}`)).json()).labels.map((l: { color: string }) => l.color);
    expect(new Set(labelColors).size).toBe(labelColors.length);
    await page.getByRole('button', { name: 'Remove dusk from 2 images' }).click();
    await expect(page.getByRole('status')).toContainText('Removed dusk from 2 images');

    // Changing the filter clears the selection.
    await page.getByRole('button', { name: 'All images' }).click();
    await expect(page.getByTestId('bulk-bar')).toHaveCount(0);

    // The annotator stays inside the folder and can edit image labels.
    await page.locator(`[data-folder="${top}/night"]`).click();
    await page.getByRole('button', { name: /Select all in folder/ }).click();
    await page.getByRole('button', { name: 'Annotate selection' }).click();
    await expect(page.getByTestId('canvas').locator('image')).toBeAttached();
    await expect(page.getByRole('navigation', { name: 'Images' }).getByRole('link')).toHaveCount(2);
    const chip = page.getByRole('button', { name: 'car', exact: true }).last();
    await chip.click();
    await expect(chip).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByTestId('save-state')).toHaveText('Saved');
    await expect.poll(async () => (await list()).find((i) => i.filename === 'a.png')!.imageLabels).toHaveLength(2);

    // Separate export: nothing is done, and the labels still come out.
    await page.getByRole('link', { name: /folders/ }).click();
    await page.getByLabel('Format').selectOption({ label: 'Image labels (zip)' });
    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('link', { name: 'Download' }).click();
    expect((await downloadPromise).suggestedFilename()).toBe('folders_image_labels.zip');

    // A file that disappears is flagged after a rescan, and its annotations become read-only.
    rmSync(join(root, 'day/c.png'));
    await page.getByRole('button', { name: 'Rescan', exact: true }).click();
    await expect(page.getByTestId('scan-line')).toContainText('1 missing');
    const missing = (await list()).find((i) => i.filename === 'c.png')!;
    expect(missing.fileState).toBe('missing');
    await page.goto(`/#/projects/${project.id}/annotate/${missing.id}`);
    await expect(page.getByTestId('unavailable-banner')).toBeVisible();
    await dragImage(page, [20, 20], [100, 80]);
    await expect(page.getByTestId('box')).toHaveCount(0);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('auto boxing marks predictions for review, and they can be accepted or removed', async ({ page, request }) => {
  const { project, ids } = await setup(request, 12, 100, 50);
  const box = (id: string): Annotation => ({ id, labelId: 'car', bbox: { x: 10, y: 10, w: 20, h: 20 }, keypoints: {} });
  const panel = page.getByRole('region', { name: 'Auto bounding box' });
  const start = panel.getByRole('button', { name: 'Train and draw boxes' });

  // Too few images with boxes, so it says what is missing.
  await request.put(`/api/images/${ids[0]}/annotations`, { data: { annotations: [box('a0')] } });
  await page.goto(`/#/projects/${project.id}`);
  await expect(start).toBeDisabled();
  await expect(panel).toContainText('Training needs at least 10 images');
  await expect(panel.getByRole('button', { name: 'Draw with trained model' })).toBeDisabled();
  await expect(page.getByTestId('autobox-model')).toHaveText('No trained model yet.');
  // The base weights are all there is to train from, and their patience cannot be changed.
  await expect(panel.getByLabel('Train from')).toHaveValue('yolov8n.pt');
  await expect(panel.getByLabel('Patience', { exact: true })).toBeDisabled();
  await expect(panel.getByLabel('Patience', { exact: true })).toHaveValue('20');
  await expect(panel).toContainText('Fixed at 20 for yolov8n.pt.');

  for (const id of ids.slice(1, 10)) await request.put(`/api/images/${id}/annotations`, { data: { annotations: [box(`a${id}`)] } });
  await page.reload();
  await expect(start).toBeEnabled();
  await start.click();
  await expect(page.getByTestId('autobox-status')).toContainText('Finished');
  await expect(page.getByTestId('autobox-status')).toContainText('Drew 2 boxes on 2 images');
  await expect(page.getByTestId('autobox-model')).toHaveCount(0);
  const job = async () => (await (await request.get(`/api/projects/${project.id}/autobox`)).json()).job;

  // The new checkpoint is selected for both, and its patience can be chosen.
  await expect(panel.getByLabel('Train from')).not.toHaveValue('yolov8n.pt');
  await expect(panel.getByLabel('Draw with').locator('option')).toHaveCount(1);
  await expect(panel.getByLabel('Patience', { exact: true })).toBeEnabled();

  // A checkpoint can draw again, without training, at the confidence the slider shows.
  await panel.getByLabel('Confidence').fill('0.4');
  await expect(page.getByTestId('autobox-confidence')).toHaveText('0.40');
  await panel.getByRole('button', { name: 'Draw with trained model' }).click();
  await expect(page.getByTestId('autobox-status')).toContainText('Finished with the checkpoint from');
  await expect(page.getByTestId('autobox-status')).toContainText('at confidence 0.40');

  // An impossible patience cannot be started.
  await panel.getByLabel('Patience', { exact: true }).fill('');
  await expect(start).toBeDisabled();
  await panel.getByLabel('Patience', { exact: true }).fill('301');
  await expect(start).toBeDisabled();

  // Training from a checkpoint uses the chosen patience, and adds another checkpoint.
  await panel.getByLabel('Patience', { exact: true }).fill('7');
  await start.click();
  await expect(page.getByTestId('autobox-status')).toContainText('Continued the checkpoint from');
  await expect(panel.getByLabel('Draw with').locator('option')).toHaveCount(2);
  expect(await job()).toMatchObject({ continued: true, patience: 7 });

  // Choosing the base weights puts the patience back to the fixed value, whatever was typed.
  await panel.getByLabel('Train from').selectOption('yolov8n.pt');
  await expect(panel.getByLabel('Patience', { exact: true })).toBeDisabled();
  await expect(panel.getByLabel('Patience', { exact: true })).toHaveValue('20');
  await start.click();
  await expect(page.getByTestId('autobox-status')).toContainText('Trained on');
  await expect(panel.getByLabel('Draw with').locator('option')).toHaveCount(3);
  expect(await job()).toMatchObject({ continued: false, patience: 20 });

  // The oldest checkpoint can be picked to draw with.
  const options = panel.getByLabel('Draw with').locator('option');
  const oldest = await options.last().getAttribute('value');
  await panel.getByLabel('Draw with').selectOption(oldest!);
  await panel.getByRole('button', { name: 'Draw with trained model' }).click();
  const { checkpoints } = await (await request.get(`/api/projects/${project.id}/autobox`)).json();
  await expect.poll(async () => (await job()).checkpointTrainedAt).toBe(checkpoints.at(-1).trainedAt);
  await expect(page.getByTestId('autobox-status')).toContainText('Finished with the checkpoint from');

  // Predictions stay todo and carry a badge, and the grid can show just them.
  await expect(page.getByRole('tab', { name: 'Auto (2)' })).toBeVisible();
  await expect(page.getByRole('tab', { name: 'Done (0)' })).toBeVisible();
  await page.getByRole('tab', { name: 'Auto (2)' }).click();
  await expect(page.locator('.image-grid li')).toHaveCount(2);
  await expect(page.locator('.badge.auto')).toHaveCount(2);

  // Accepting them through the group action removes the mark, and undo restores it.
  page.on('dialog', (dialog) => dialog.accept());
  await page.getByRole('button', { name: /^Select all/ }).click();
  await page.getByRole('button', { name: 'Mark done' }).click();
  await expect(page.getByRole('tab', { name: 'Done (2)' })).toBeVisible();
  await expect(page.getByRole('tab', { name: 'Auto (0)' })).toBeVisible();
  await page.getByRole('button', { name: 'Undo' }).click();
  await expect(page.getByRole('tab', { name: 'Auto (2)' })).toBeVisible();

  // Predicted boxes can be thrown away, which leaves the human boxes.
  await panel.getByRole('button', { name: 'Remove predicted boxes (2)' }).click();
  await expect(page.getByRole('tab', { name: 'Auto (0)' })).toBeVisible();
  const remaining: ImageMeta[] = await (await request.get(`/api/projects/${project.id}/images`)).json();
  expect(remaining.filter((i) => i.annotationCount > 0)).toHaveLength(10);

  // After adding a label, checkpoints can still draw but training cannot continue from them.
  const plane: Label = { id: 'plane', name: 'plane', color: '#0000ff', keypoints: [], skeleton: [] };
  expect((await request.patch(`/api/projects/${project.id}`, { data: { labels: [...project.labels, plane] } })).ok()).toBe(true);
  await page.reload();
  await expect(panel.getByLabel('Train from')).toHaveValue('yolov8n.pt');
  await expect(panel.getByLabel('Train from').locator('option[disabled]')).toHaveCount(3);
  await expect(page.getByTestId('autobox-model')).toContainText('Labels were added or removed since 3 checkpoints were trained');
  await expect(panel.getByRole('button', { name: 'Draw with trained model' })).toBeEnabled();

  // Deleting removes only the checkpoint selected, until none are left.
  for (const left of [2, 1, 0]) {
    await panel.getByRole('button', { name: 'Delete checkpoint' }).click();
    await expect(panel.getByLabel('Train from').locator('option')).toHaveCount(left + 1);
  }
  await expect(page.getByTestId('autobox-model')).toHaveText('No trained model yet.');
  await expect(panel.getByRole('button', { name: 'Draw with trained model' })).toBeDisabled();
  await expect(panel.getByRole('button', { name: 'Delete checkpoint' })).toHaveCount(0);
});

test('a remote run\'s results come back as predictions and a checkpoint', async ({ page, request }) => {
  const { project, ids } = await setup(request, 12, 100, 50);
  const box = (id: string): Annotation => ({ id, labelId: 'car', bbox: { x: 10, y: 10, w: 20, h: 20 }, keypoints: {} });
  for (const id of ids.slice(0, 10)) await request.put(`/api/images/${id}/annotations`, { data: { annotations: [box(`a${id}`)] } });
  await page.goto(`/#/projects/${project.id}`);
  const panel = page.getByRole('region', { name: 'Auto bounding box' });
  await panel.getByText('Train on a remote server').click();
  await panel.getByLabel('Remote model').selectOption('l');

  // One click does the whole round trip, and the password is gone from the page afterwards.
  const trainRemote = panel.getByRole('button', { name: 'Train on server and draw' });
  await expect(trainRemote).toBeDisabled();
  await panel.getByLabel('Server').fill('me@gpu.example');
  await panel.getByLabel('Password').fill('secret');
  await trainRemote.click();
  await expect(page.getByTestId('remote-status')).toContainText(/Finished training on me@gpu.example. Trained on \d+ images, 3 epochs, best mAP50 0.80. Drawing on 2 images here with the new checkpoint./);
  await expect(panel.getByLabel('Password')).toHaveValue('');
  // The server only trained; the drawing is the ordinary local job with the new checkpoint.
  await expect(page.getByTestId('autobox-status')).toContainText('Drew 2 boxes on 2 images');
  await expect(page.getByRole('tab', { name: 'Auto (2)' })).toBeVisible();
  await expect(panel.getByLabel('Draw with')).toContainText('yolov8l, remote');
  await expect(panel.getByLabel('Draw with').locator('option')).toHaveCount(1);
  await request.delete(`/api/projects/${project.id}/autobox/predictions`);

  // The manual steps are the same run by hand. The bundle link carries the chosen options, and the app remembers the bundle it made.
  await panel.getByText('Manual steps').click();
  const link = panel.getByRole('link', { name: 'Download bundle' });
  await expect(link).toHaveAttribute('href', /model=l/);
  const bundle = await request.get((await link.getAttribute('href'))!);
  expect(bundle.ok()).toBe(true);
  expect(bundle.headers()['content-disposition']).toContain('.zip');
  const entries = unzip(await bundle.body());
  expect(Object.keys(entries).filter((e) => e.startsWith('images/'))).toHaveLength(10);
  const bundleId = JSON.parse(entries['job.json']).bundleId as string;

  const results = {
    bundleId,
    task: 'detect',
    trainedAt: '2026-11-12T09:00:00.000Z',
    trainImages: 8,
    epochs: 25,
    bestMap50: 0.88,
    predictions: [],
    skipped: [],
  };
  await panel.getByTestId('remote-results').setInputFiles({ name: 'results.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(results)) });
  await panel.getByTestId('remote-weights').setInputFiles({ name: 'best.pt', mimeType: 'application/octet-stream', buffer: Buffer.from('weights') });
  await panel.getByRole('button', { name: 'Import results' }).click();
  await expect(page.getByTestId('remote-summary')).toContainText('Trained on 8 images, 25 epochs, best mAP50 0.88. The trained model was added as a checkpoint: use Draw with trained model.');
  await expect(panel.getByLabel('Draw with').locator('option')).toHaveCount(2);
});

test('the image list renders only visible rows and follows navigation',async ({ page, request }) => {
  const { project, ids } = await setup(request, 150, 40, 30);
  await open(page, project.id, ids[149]);

  const list = page.getByRole('navigation', { name: 'Images' });
  const current = list.locator('a.current');
  await expect(current).toContainText('150. img149.png');
  await expect(current).toBeInViewport();
  expect(await list.getByRole('link').count()).toBeLessThan(100);

  await list.evaluate((nav) => (nav.scrollTop = 0));
  await list.getByRole('link', { name: /^todo 1\. img0\.png/ }).click();
  await expect(page).toHaveURL(new RegExp(ids[0]));
  await expect(list.locator('a.current')).toContainText('1. img0.png');
  await page.keyboard.press('d');
  await expect(list.locator('a.current')).toContainText('2. img1.png');
});

test('rapid navigation keys each move exactly one image', async ({ page, request }) => {
  const { project, ids } = await setup(request, 5, 40, 30);
  await open(page, project.id, ids[0]);
  const current = page.getByRole('navigation', { name: 'Images' }).locator('a.current');

  await page.keyboard.press('d');
  await page.keyboard.press('d');
  await page.keyboard.press('a');
  await page.keyboard.press('d');
  await page.keyboard.press('d');
  await expect(page).toHaveURL(new RegExp(ids[3]));
  await expect(current).toContainText('4. img3.png');
});
