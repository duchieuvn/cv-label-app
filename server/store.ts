import { randomUUID } from 'node:crypto';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import {
  pruneAnnotations,
  type Annotation,
  type BulkResult,
  type BulkUpdate,
  type FileState,
  type FolderNode,
  type ImageDetail,
  type ImageMeta,
  type ImageStatus,
  type Label,
  type Project,
  type ProjectMode,
  type Source,
} from '../shared/schema.ts';

export const MIGRATIONS = [
  `CREATE TABLE projects (
     id TEXT PRIMARY KEY,
     name TEXT NOT NULL,
     labels TEXT NOT NULL,
     created_at TEXT NOT NULL
   );
   CREATE TABLE images (
     id TEXT PRIMARY KEY,
     project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
     filename TEXT NOT NULL,
     stored_name TEXT NOT NULL,
     width INTEGER NOT NULL,
     height INTEGER NOT NULL,
     status TEXT NOT NULL DEFAULT 'todo',
     annotations TEXT NOT NULL DEFAULT '[]',
     seq INTEGER NOT NULL
   );
   CREATE INDEX images_project ON images(project_id, seq);`,
  `CREATE TABLE sources (
     id TEXT PRIMARY KEY,
     project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
     name TEXT NOT NULL,
     path TEXT NOT NULL,
     UNIQUE (project_id, name)
   );
   CREATE TABLE source_exclusions (
     source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
     rel_path TEXT NOT NULL,
     PRIMARY KEY (source_id, rel_path)
   );
   ALTER TABLE images ADD COLUMN source_id TEXT REFERENCES sources(id) ON DELETE CASCADE;
   ALTER TABLE images ADD COLUMN rel_path TEXT;
   ALTER TABLE images ADD COLUMN folder TEXT NOT NULL DEFAULT 'uploads';
   ALTER TABLE images ADD COLUMN image_labels TEXT NOT NULL DEFAULT '[]';
   ALTER TABLE images ADD COLUMN file_state TEXT NOT NULL DEFAULT 'ok';
   ALTER TABLE images ADD COLUMN file_size INTEGER;
   ALTER TABLE images ADD COLUMN file_mtime REAL;
   ALTER TABLE images ADD COLUMN mime TEXT;
   CREATE UNIQUE INDEX images_source_path ON images(source_id, rel_path) WHERE source_id IS NOT NULL;
   CREATE INDEX images_folder ON images(project_id, folder);`,
  `ALTER TABLE images ADD COLUMN auto_boxed INTEGER NOT NULL DEFAULT 0;`,
  `ALTER TABLE projects ADD COLUMN mode TEXT NOT NULL DEFAULT 'bbox';
   UPDATE projects SET mode = 'keypoint'
     WHERE EXISTS (SELECT 1 FROM json_each(labels) WHERE json_array_length(value, '$.keypoints') > 0);`,
];

const IMAGE_META_COLUMNS = `id, project_id, filename, width, height, status, folder, image_labels,
  file_state, source_id, rel_path, seq, auto_boxed, json_array_length(annotations) AS annotation_count`;

// Marking an image done accepts its predictions, and marking it todo again takes that back, so an
// undone group action leaves them out of training. Only saving annotations clears auto_boxed for good.
const UNREVIEWED = `(auto_boxed = 1 AND status = 'todo')`;
const AUTOBOX_TRAINABLE = `(file_state = 'ok' AND NOT ${UNREVIEWED} AND annotations != '[]'
  AND EXISTS (SELECT 1 FROM json_each(annotations) WHERE json_extract(value, '$.bbox') IS NOT NULL))`;
const AUTOBOX_TARGET = `(file_state = 'ok' AND status = 'todo' AND (annotations = '[]' OR auto_boxed = 1))`;

export type ImageFilter = { status?: ImageStatus; folder?: string; recursive?: boolean; label?: string };
export type KnownFile = { id: string; width: number; height: number; size: number | null; mtime: number | null; state: FileState };
export type AutoBoxImage = {
  id: string;
  projectId: string;
  width: number;
  height: number;
  annotations: Annotation[];
  storedName: string;
  relPath: string | null;
  sourcePath: string | null;
};

const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });

/** Uploads come first in upload order, then scanned images in natural path order. */
function sortRows(rows: Row[]): Row[] {
  const path = (r: Row) => `${r.folder}/${r.filename}`;
  return rows.sort((a, b) => {
    if (!a.source_id && !b.source_id) return a.seq - b.seq;
    if (!a.source_id) return -1;
    if (!b.source_id) return 1;
    return collator.compare(path(a), path(b)) || a.seq - b.seq;
  });
}

/** SQL matching a folder on whole path segments, so "cats" does not match "cats2". */
function folderClause(folder: string, recursive: boolean): { sql: string; params: string[] } {
  if (!recursive) return { sql: 'folder = ?', params: [folder] };
  return { sql: '(folder = ? OR substr(folder, 1, ?) = ?)', params: [folder, String(folder.length + 1), `${folder}/`] };
}

type Row = Record<string, any>;

const toMeta = (r: Row): ImageMeta => ({
  id: r.id,
  projectId: r.project_id,
  filename: r.filename,
  width: r.width,
  height: r.height,
  status: r.status,
  annotationCount: r.annotation_count,
  folder: r.folder,
  imageLabels: JSON.parse(r.image_labels),
  fileState: r.file_state,
  sourceId: r.source_id,
  autoBoxed: r.auto_boxed === 1 && r.status === 'todo',
});

const toProject =(r: Row): Project => ({
  id: r.id,
  name: r.name,
  mode: r.mode,
  labels: JSON.parse(r.labels),
  createdAt: r.created_at,
  imageCount: r.image_count,
  doneCount: r.done_count,
});

export type Store = ReturnType<typeof createStore>;

export function createStore(dataDir: string) {
  mkdirSync(dataDir, { recursive: true });
  const db = new DatabaseSync(join(dataDir, 'labelapp.db'));
  // NORMAL is safe with WAL, and without it every image inserted by a scan would wait on an fsync.
  db.exec('PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL; PRAGMA foreign_keys = ON;');

  const version = (db.prepare('PRAGMA user_version').get() as Row).user_version as number;
  for (let v = version; v < MIGRATIONS.length; v++) {
    db.exec(`BEGIN; ${MIGRATIONS[v]} PRAGMA user_version = ${v + 1}; COMMIT;`);
  }

  function transaction<T>(fn: () => T): T {
    db.exec('BEGIN');
    try {
      const result = fn();
      db.exec('COMMIT');
      return result;
    } catch (error) {
      db.exec('ROLLBACK');
      throw error;
    }
  }

  const PROJECT_SELECT = `SELECT p.*,
      (SELECT COUNT(*) FROM images i WHERE i.project_id = p.id) AS image_count,
      (SELECT COUNT(*) FROM images i WHERE i.project_id = p.id AND i.status = 'done') AS done_count
    FROM projects p`;

  const store = {
    close: () => db.close(),

    listProjects(): Project[] {
      return (db.prepare(`${PROJECT_SELECT} ORDER BY p.created_at DESC`).all() as Row[]).map(toProject);
    },

    getProject(id: string): Project | null {
      const row = db.prepare(`${PROJECT_SELECT} WHERE p.id = ?`).get(id) as Row | undefined;
      return row ? toProject(row) : null;
    },

    createProject(name: string, labels: Label[], mode: ProjectMode = 'bbox'): Project {
      const id = randomUUID();
      db.prepare('INSERT INTO projects (id, name, mode, labels, created_at) VALUES (?, ?, ?, ?, ?)').run(
        id,
        name,
        mode,
        JSON.stringify(labels),
        new Date().toISOString(),
      );
      return store.getProject(id)!;
    },

    renameProject(id: string, name: string) {
      db.prepare('UPDATE projects SET name = ? WHERE id = ?').run(name, id);
    },

    /**
     * Replaces a project's labels, pruning placed keypoints whose definitions were removed.
     * Returns the names of removed labels that are still in use if any exist; nothing is changed then.
     */
    updateLabels(id: string, labels: Label[]): { blockedBy: string[]; prunedKeypoints: number } {
      return transaction(() => {
        const current = store.getProject(id)!;
        const newIds = new Set(labels.map((l) => l.id));
        const rows = db
          .prepare(`SELECT id, annotations FROM images WHERE project_id = ? AND annotations != '[]'`)
          .all(id) as Row[];
        const docs = rows.map((r) => ({ id: r.id as string, annotations: JSON.parse(r.annotations) as Annotation[] }));

        const tagged = db
          .prepare(`SELECT DISTINCT value AS label FROM images, json_each(images.image_labels) WHERE project_id = ?`)
          .all(id) as Row[];
        const used = new Set([...docs.flatMap((d) => d.annotations.map((a) => a.labelId)), ...tagged.map((r) => r.label as string)]);
        const blockedBy = current.labels.filter((l) => !newIds.has(l.id) && used.has(l.id)).map((l) => l.name);
        if (blockedBy.length > 0) return { blockedBy, prunedKeypoints: 0 };

        let prunedKeypoints = 0;
        const update = db.prepare('UPDATE images SET annotations = ? WHERE id = ?');
        for (const doc of docs) {
          const { annotations, pruned } = pruneAnnotations(doc.annotations, labels);
          if (pruned > 0) {
            update.run(JSON.stringify(annotations), doc.id);
            prunedKeypoints += pruned;
          }
        }
        db.prepare('UPDATE projects SET labels = ? WHERE id = ?').run(JSON.stringify(labels), id);
        return { blockedBy: [], prunedKeypoints };
      });
    },

    deleteProject(id: string) {
      db.prepare('DELETE FROM projects WHERE id = ?').run(id);
    },

    listImages(projectId: string, filter: ImageFilter = {}): ImageMeta[] {
      const where = ['project_id = ?'];
      const params: string[] = [projectId];
      if (filter.status) {
        where.push('status = ?');
        params.push(filter.status);
      }
      if (filter.folder) {
        const clause = folderClause(filter.folder, filter.recursive ?? true);
        where.push(clause.sql);
        params.push(...clause.params);
      }
      if (filter.label) {
        where.push('EXISTS (SELECT 1 FROM json_each(images.image_labels) WHERE value = ?)');
        params.push(filter.label);
      }
      const rows = db.prepare(`SELECT ${IMAGE_META_COLUMNS} FROM images WHERE ${where.join(' AND ')}`).all(...params) as Row[];
      return sortRows(rows).map(toMeta);
    },

    /** Every folder holding images, plus their ancestors, sorted so parents precede children. */
    listFolders(projectId: string): FolderNode[] {
      const rows = db
        .prepare('SELECT folder, COUNT(*) AS n FROM images WHERE project_id = ? GROUP BY folder')
        .all(projectId) as Row[];
      const nodes = new Map<string, FolderNode>();
      for (const { folder, n } of rows) {
        const parts = (folder as string).split('/');
        parts.forEach((name, i) => {
          const path = parts.slice(0, i + 1).join('/');
          const node = nodes.get(path) ?? { path, name, parent: i > 0 ? parts.slice(0, i).join('/') : null, direct: 0, total: 0 };
          node.total += n as number;
          if (i === parts.length - 1) node.direct += n as number;
          nodes.set(path, node);
        });
      }
      return [...nodes.values()].sort((a, b) => collator.compare(a.path, b.path));
    },

    /** Applies a group action in one transaction. Throws nothing partial: all images change or none. */
    bulkUpdate(projectId: string, body: BulkUpdate): BulkResult {
      return transaction(() => {
        let rows: Row[];
        if ('folder' in body.selector) {
          const clause = folderClause(body.selector.folder, body.selector.recursive);
          const except = new Set(body.selector.except);
          rows = (
            db.prepare(`SELECT id, image_labels, status FROM images WHERE project_id = ? AND ${clause.sql}`).all(projectId, ...clause.params) as Row[]
          ).filter((r) => !except.has(r.id));
        } else {
          const select = db.prepare('SELECT id, image_labels, status FROM images WHERE project_id = ? AND id = ?');
          rows = [...new Set(body.selector.imageIds)].flatMap((id) => {
            const row = select.get(projectId, id) as Row | undefined;
            return row ? [row] : [];
          });
        }

        const update = db.prepare('UPDATE images SET image_labels = ?, status = ? WHERE id = ?');
        const changed: string[] = [];
        for (const row of rows) {
          const before: string[] = JSON.parse(row.image_labels);
          const labels = [...before.filter((l) => !body.removeLabels.includes(l))];
          for (const l of body.addLabels) if (!labels.includes(l)) labels.push(l);
          const status = body.status ?? row.status;
          if (labels.length !== before.length || labels.some((l) => !before.includes(l)) || status !== row.status) {
            update.run(JSON.stringify(labels), status, row.id);
            changed.push(row.id);
          }
        }
        return { matched: rows.length, changed };
      });
    },

    getImage(id: string): ImageDetail | null {
      const row = db.prepare(`SELECT ${IMAGE_META_COLUMNS}, annotations FROM images WHERE id = ?`).get(id) as Row | undefined;
      return row ? { ...toMeta(row), annotations: JSON.parse(row.annotations) } : null;
    },

    getFile(id: string): { projectId: string; storedName: string; sourceId: string | null; sourcePath: string | null; relPath: string | null; mime: string | null } | null {
      const row = db
        .prepare(
          `SELECT i.project_id, i.stored_name, i.source_id, i.rel_path, i.mime, s.path AS source_path
           FROM images i LEFT JOIN sources s ON s.id = i.source_id WHERE i.id = ?`,
        )
        .get(id) as Row | undefined;
      if (!row) return null;
      return {
        projectId: row.project_id,
        storedName: row.stored_name,
        sourceId: row.source_id,
        sourcePath: row.source_path,
        relPath: row.rel_path,
        mime: row.mime,
      };
    },

    listImagesWithAnnotations(projectId: string): (ImageDetail & { storedName: string; relPath: string | null })[] {
      const rows = db
        .prepare(`SELECT ${IMAGE_META_COLUMNS}, annotations, stored_name FROM images WHERE project_id = ?`)
        .all(projectId) as Row[];
      return sortRows(rows).map((r) => ({
        ...toMeta(r),
        annotations: JSON.parse(r.annotations),
        storedName: r.stored_name,
        relPath: r.rel_path,
      }));
    },

    listSources(projectId: string): Omit<Source, 'scan'>[] {
      return (
        db
          .prepare(
            `SELECT s.id, s.project_id, s.name, s.path, (SELECT COUNT(*) FROM images i WHERE i.source_id = s.id) AS image_count
             FROM sources s WHERE s.project_id = ? ORDER BY s.name`,
          )
          .all(projectId) as Row[]
      ).map((r) => ({ id: r.id, projectId: r.project_id, name: r.name, path: r.path, imageCount: r.image_count }));
    },

    getSource(id: string): Omit<Source, 'scan'> | null {
      const row = db
        .prepare(
          `SELECT s.id, s.project_id, s.name, s.path, (SELECT COUNT(*) FROM images i WHERE i.source_id = s.id) AS image_count
           FROM sources s WHERE s.id = ?`,
        )
        .get(id) as Row | undefined;
      return row ? { id: row.id, projectId: row.project_id, name: row.name, path: row.path, imageCount: row.image_count } : null;
    },

    /** Names become the top level of the folder tree, so they must be unique; "uploads" is reserved. */
    createSource(projectId: string, baseName: string, path: string): Omit<Source, 'scan'> {
      const taken = new Set(store.listSources(projectId).map((s) => s.name));
      taken.add('uploads');
      let name = baseName;
      for (let n = 2; taken.has(name); n++) name = `${baseName}_${n}`;
      const id = randomUUID();
      db.prepare('INSERT INTO sources (id, project_id, name, path) VALUES (?, ?, ?, ?)').run(id, projectId, name, path);
      return store.getSource(id)!;
    },

    deleteSource(id: string) {
      db.prepare('DELETE FROM sources WHERE id = ?').run(id);
    },

    knownFiles(sourceId: string): Map<string, KnownFile> {
      const rows = db
        .prepare('SELECT id, rel_path, width, height, file_size, file_mtime, file_state FROM images WHERE source_id = ?')
        .all(sourceId) as Row[];
      return new Map(
        rows.map((r) => [
          r.rel_path as string,
          { id: r.id, width: r.width, height: r.height, size: r.file_size, mtime: r.file_mtime, state: r.file_state },
        ]),
      );
    },

    addScannedImage(image: {
      projectId: string;
      sourceId: string;
      relPath: string;
      filename: string;
      folder: string;
      mime: string;
      width: number;
      height: number;
      size: number;
      mtime: number;
    }) {
      db.prepare(
        `INSERT INTO images (id, project_id, filename, stored_name, width, height, seq, source_id, rel_path, folder, mime, file_size, file_mtime)
         VALUES (?, ?, ?, '', ?, ?, (SELECT COALESCE(MAX(seq), 0) + 1 FROM images WHERE project_id = ?), ?, ?, ?, ?, ?, ?)`,
      ).run(
        randomUUID(),
        image.projectId,
        image.filename,
        image.width,
        image.height,
        image.projectId,
        image.sourceId,
        image.relPath,
        image.folder,
        image.mime,
        image.size,
        image.mtime,
      );
    },

    setFileState(id: string, state: FileState, stat?: { size: number; mtime: number; mime: string }) {
      if (stat) {
        db.prepare('UPDATE images SET file_state = ?, file_size = ?, file_mtime = ?, mime = ? WHERE id = ?').run(state, stat.size, stat.mtime, stat.mime, id);
      } else {
        db.prepare('UPDATE images SET file_state = ? WHERE id = ?').run(state, id);
      }
    },

    listExclusions(sourceId: string): Set<string> {
      const rows = db.prepare('SELECT rel_path FROM source_exclusions WHERE source_id = ?').all(sourceId) as Row[];
      return new Set(rows.map((r) => r.rel_path as string));
    },

    clearExclusions(sourceId: string) {
      db.prepare('DELETE FROM source_exclusions WHERE source_id = ?').run(sourceId);
    },

    /** Removes a scanned image from the project and remembers it, so a rescan does not bring it back. */
    excludeImage(id: string, sourceId: string, relPath: string) {
      transaction(() => {
        db.prepare('INSERT OR IGNORE INTO source_exclusions (source_id, rel_path) VALUES (?, ?)').run(sourceId, relPath);
        db.prepare('DELETE FROM images WHERE id = ?').run(id);
      });
    },

    addImage(image: {
      id: string;
      projectId: string;
      filename: string;
      storedName: string;
      width: number;
      height: number;
    }): ImageMeta {
      db.prepare(
        `INSERT INTO images (id, project_id, filename, stored_name, width, height, seq)
         VALUES (?, ?, ?, ?, ?, ?, (SELECT COALESCE(MAX(seq), 0) + 1 FROM images WHERE project_id = ?))`,
      ).run(image.id, image.projectId, image.filename, image.storedName, image.width, image.height, image.projectId);
      return store.getImage(image.id)!;
    },

    saveImage(id: string, changes: { annotations?: Annotation[]; status?: ImageStatus; imageLabels?: string[] }) {
      if (changes.imageLabels) {
        db.prepare('UPDATE images SET image_labels = ? WHERE id = ?').run(JSON.stringify(changes.imageLabels), id);
      }
      if (changes.annotations) {
        db.prepare('UPDATE images SET annotations = ?, auto_boxed = 0 WHERE id = ?').run(JSON.stringify(changes.annotations), id);
      }
      if (changes.status) {
        db.prepare('UPDATE images SET status = ? WHERE id = ?').run(changes.status, id);
      }
    },

    /** Writes an import in one transaction: new labels first, then each image's annotations, status and image labels. */
    applyImport(projectId: string, labels: Label[] | null, writes: { imageId: string; annotations: Annotation[]; status?: ImageStatus; imageLabels?: string[] }[]) {
      transaction(() => {
        if (labels) db.prepare('UPDATE projects SET labels = ? WHERE id = ?').run(JSON.stringify(labels), projectId);
        // Imported boxes were drawn by a person, so they are reviewed work rather than predictions.
        const annotate = db.prepare(`UPDATE images SET annotations = ?, auto_boxed = 0 WHERE id = ? AND project_id = ? AND file_state = 'ok'`);
        const status = db.prepare('UPDATE images SET status = ? WHERE id = ? AND project_id = ?');
        const imageLabels = db.prepare('UPDATE images SET image_labels = ? WHERE id = ? AND project_id = ?');
        for (const write of writes) {
          annotate.run(JSON.stringify(write.annotations), write.imageId, projectId);
          if (write.status) status.run(write.status, write.imageId, projectId);
          if (write.imageLabels) imageLabels.run(JSON.stringify(write.imageLabels), write.imageId, projectId);
        }
      });
    },

    labelsOf(projectId: string): Label[] | null {
      const row = db.prepare('SELECT labels FROM projects WHERE id = ?').get(projectId) as Row | undefined;
      return row ? JSON.parse(row.labels) : null;
    },

    autoBoxCounts(projectId: string): { trainable: number; targets: number; autoBoxed: number } {
      const row = db
        .prepare(
          `SELECT COALESCE(SUM(${AUTOBOX_TRAINABLE}), 0) AS trainable, COALESCE(SUM(${AUTOBOX_TARGET}), 0) AS targets,
             COALESCE(SUM(${UNREVIEWED}), 0) AS auto_boxed
           FROM images WHERE project_id = ?`,
        )
        .get(projectId) as Row;
      return { trainable: row.trainable, targets: row.targets, autoBoxed: row.auto_boxed };
    },

    autoBoxImages(projectId: string): { training: AutoBoxImage[]; targets: AutoBoxImage[] } {
      const select = (condition: string) =>
        (
          db
            .prepare(
              `SELECT i.id, i.width, i.height, i.annotations, i.stored_name, i.rel_path, s.path AS source_path
               FROM images i LEFT JOIN sources s ON s.id = i.source_id WHERE i.project_id = ? AND ${condition}`,
            )
            .all(projectId) as Row[]
        ).map(
          (r): AutoBoxImage => ({
            id: r.id,
            projectId,
            width: r.width,
            height: r.height,
            annotations: JSON.parse(r.annotations),
            storedName: r.stored_name,
            relPath: r.rel_path,
            sourcePath: r.source_path,
          }),
        );
      return { training: select(AUTOBOX_TRAINABLE), targets: select(AUTOBOX_TARGET) };
    },

    /**
     * Writes predictions unless the image was reviewed since the job started, so a run can never
     * replace a person's work. Returns whether it wrote. With no detections the image is simply
     * empty again, which does not make it a true negative because it stays todo.
     */
    saveAutoBoxes(id: string, annotations: Annotation[]): boolean {
      const result = db
        .prepare(`UPDATE images SET annotations = ?, auto_boxed = ? WHERE id = ? AND ${AUTOBOX_TARGET}`)
        .run(JSON.stringify(annotations), annotations.length > 0 ? 1 : 0, id);
      return result.changes > 0;
    },

    /** Removes every unreviewed prediction in a project. */
    clearAutoBoxes(projectId: string): number {
      return Number(
        db
          .prepare(`UPDATE images SET annotations = '[]', auto_boxed = 0 WHERE project_id = ? AND file_state = 'ok' AND ${UNREVIEWED}`)
          .run(projectId).changes,
      );
    },

    deleteImage(id: string) {
      db.prepare('DELETE FROM images WHERE id = ?').run(id);
    },
  };
  return store;
}
