import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { expect, it } from 'vitest';
import { createStore, MIGRATIONS } from '../../server/store.ts';

it('upgrades a database from the first version without losing data', () => {
  const dir = mkdtempSync(join(tmpdir(), 'labelapp-migrate-'));
  try {
    const old = new DatabaseSync(join(dir, 'labelapp.db'));
    old.exec(`BEGIN; ${MIGRATIONS[0]} PRAGMA user_version = 1; COMMIT;`);
    const labels = JSON.stringify([{ id: 'car', name: 'car', color: '#000000', keypoints: [], skeleton: [] }]);
    const annotations = JSON.stringify([{ id: 'a', labelId: 'car', bbox: { x: 1, y: 1, w: 2, h: 2 }, keypoints: {} }]);
    old.prepare(`INSERT INTO projects VALUES ('p', 'old', ?, '2026-01-01')`).run(labels);
    old.prepare(`INSERT INTO images (id, project_id, filename, stored_name, width, height, status, annotations, seq)
                 VALUES ('i', 'p', 'a.jpg', 'x.jpg', 10, 10, 'done', ?, 1)`).run(annotations);
    old.close();

    const store = createStore(dir);
    try {
      expect(store.getImage('i')).toMatchObject({
        filename: 'a.jpg',
        status: 'done',
        annotationCount: 1,
        folder: 'uploads',
        imageLabels: [],
        fileState: 'ok',
        sourceId: null,
      });
      expect(store.getFile('i')).toMatchObject({ storedName: 'x.jpg', sourceId: null });
      expect(store.listFolders('p')).toEqual([{ path: 'uploads', name: 'uploads', parent: null, direct: 1, total: 1 }]);
    } finally {
      store.close();
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

it('gives existing projects the mode their labels imply, keeping their annotations', () => {
  const dir = mkdtempSync(join(tmpdir(), 'labelapp-migrate-'));
  try {
    const old = new DatabaseSync(join(dir, 'labelapp.db'));
    const before = MIGRATIONS.findIndex((m) => m.includes('ADD COLUMN mode'));
    old.exec(`BEGIN; ${MIGRATIONS.slice(0, before).join('\n')} PRAGMA user_version = ${before}; COMMIT;`);
    const label = (id: string, keypoints: { id: string; name: string }[]) => ({ id, name: id, color: '#000000', keypoints, skeleton: [] });
    const insert = old.prepare(`INSERT INTO projects (id, name, labels, created_at) VALUES (?, ?, ?, '2026-01-01')`);
    insert.run('boxes', 'boxes', JSON.stringify([label('car', [])]));
    insert.run('none', 'none', '[]');
    insert.run('mixed', 'mixed', JSON.stringify([label('car', []), label('bird', [{ id: 'beak', name: 'beak' }])]));
    const mixed = [
      { id: 'a', labelId: 'car', bbox: { x: 1, y: 1, w: 2, h: 2 }, keypoints: {} },
      { id: 'b', labelId: 'bird', bbox: { x: 1, y: 1, w: 2, h: 2 }, keypoints: { beak: { x: 2, y: 2, v: 2 } } },
    ];
    old.prepare(`INSERT INTO images (id, project_id, filename, stored_name, width, height, status, annotations, seq)
                 VALUES ('i', 'mixed', 'a.jpg', 'x.jpg', 10, 10, 'done', ?, 1)`).run(JSON.stringify(mixed));
    old.close();

    const store = createStore(dir);
    try {
      expect(store.getProject('boxes')?.mode).toBe('bbox');
      expect(store.getProject('none')?.mode).toBe('bbox');
      expect(store.getProject('mixed')?.mode).toBe('keypoint');
      expect(store.getImage('i')?.annotations).toEqual(mixed);
    } finally {
      store.close();
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
