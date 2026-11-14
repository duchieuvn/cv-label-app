import { cpSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';
import { expect, test } from '@playwright/test';
import type { ImageMeta, Label, Project, Source } from '../../shared/schema.ts';
import { makePng } from '../helpers.ts';

const CAR: Label = { id: 'car', name: 'car', color: '#3ddc97', keypoints: [], skeleton: [] };

test('exports a project file and opens it as a new project on another path', async ({ page, request }) => {
  const root = mkdtempSync(join(tmpdir(), 'labelapp-e2e-share-'));
  const copy = mkdtempSync(join(tmpdir(), 'labelapp-e2e-copy-'));
  try {
    mkdirSync(join(root, 'day'));
    writeFileSync(join(root, 'a.png'), makePng(200, 100));
    writeFileSync(join(root, 'day/b.png'), makePng(200, 100));

    const project: Project = await (await request.post('/api/projects', { data: { name: 'Shared', labels: [CAR] } })).json();
    const source: Source = await (await request.post(`/api/projects/${project.id}/sources`, { data: { path: root } })).json();
    await expect.poll(async () => (await (await request.get(`/api/sources/${source.id}/scan`)).json()).state).toBe('done');
    const [a] = (await (await request.get(`/api/projects/${project.id}/images`)).json()) as ImageMeta[];
    const annotation = { id: 'x', labelId: 'car', bbox: { x: 10, y: 10, w: 50, h: 40 }, keypoints: {} };
    await request.put(`/api/images/${a.id}/annotations`, { data: { annotations: [annotation], status: 'done' } });

    await page.goto(`/#/projects/${project.id}`);
    await page.getByRole('region', { name: 'Export' }).getByRole('combobox').first().selectOption('project');
    await expect(page.getByTestId('project-export-note')).toBeVisible();
    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('link', { name: 'Download' }).click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toBe('Shared.labelapp');
    const file = join(copy, '..', `${basename(copy)}.labelapp`);
    await download.saveAs(file);

    // The teammate's copy of the photos is somewhere else.
    cpSync(root, copy, { recursive: true });

    await page.goto('/#/');
    const panel = page.getByTestId('open-shared');
    await panel.locator('summary').click();
    await panel.getByLabel('Project file').setInputFiles(file);
    const folder = panel.getByTestId('shared-folder');
    await expect(folder).toContainText(basename(root));
    await expect(folder).toContainText(root);
    await expect(panel.getByTestId('shared-folder-status')).toContainText('Left out unless a path is given.');

    await folder.getByRole('textbox').fill(join(copy, 'nope'));
    await expect(panel.getByTestId('shared-folder-status')).toContainText('does not exist');
    await expect(panel.getByRole('button', { name: 'Create project' })).toBeDisabled();

    await folder.getByRole('textbox').fill(copy);
    await expect(panel.getByTestId('shared-folder-status')).toContainText('2 of 2 sample files found.');
    await panel.getByLabel('Project name').fill('Shared copy');
    await panel.getByRole('button', { name: 'Create project' }).click();

    await expect(page.getByRole('heading', { name: 'Shared copy' })).toBeVisible();
    await expect(page.getByTestId('scan-line')).toContainText('Scan finished');
    await expect(page.getByTestId('scan-line')).toContainText('2 recovered');
    await expect(page.getByTestId('source')).toContainText(copy);

    const projects: Project[] = await (await request.get('/api/projects')).json();
    const opened = projects.find((p) => p.name === 'Shared copy')!;
    const images: ImageMeta[] = await (await request.get(`/api/projects/${opened.id}/images`)).json();
    expect(images.map((i) => [i.filename, i.fileState, i.status, i.annotationCount])).toEqual([
      ['a.png', 'ok', 'done', 1],
      ['b.png', 'ok', 'todo', 0],
    ]);
    rmSync(file, { force: true });
  } finally {
    rmSync(root, { recursive: true, force: true });
    rmSync(copy, { recursive: true, force: true });
  }
});
