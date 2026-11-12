import { readdir, stat } from 'node:fs/promises';
import { join } from 'node:path';
import type { ScanStatus, Source } from '../shared/schema.ts';
import { inspectImage } from './imageInfo.ts';
import type { Store } from './store.ts';

const MAX_LISTED_ERRORS = 100;
const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });

type Job = { status: ScanStatus; cancelled: boolean };

/** Runs folder scans in the background. Status lives in memory, so a restart abandons running scans. */
export class ScanManager {
  private jobs = new Map<string, Job>();
  private store: Store;

  constructor(store: Store) {
    this.store = store;
  }

  get(sourceId: string): ScanStatus | null {
    return this.jobs.get(sourceId)?.status ?? null;
  }

  isRunning(sourceId: string): boolean {
    return this.get(sourceId)?.state === 'running';
  }

  /** Stops the scan, which never writes again once this returns. */
  cancel(sourceId: string): boolean {
    const job = this.jobs.get(sourceId);
    if (job?.status.state !== 'running') return false;
    job.cancelled = true;
    return true;
  }

  start(source: Omit<Source, 'scan'>, options: { includeExcluded: boolean }): ScanStatus {
    const job: Job = {
      cancelled: false,
      status: {
        state: 'running',
        foldersVisited: 0,
        added: 0,
        skipped: 0,
        unchanged: 0,
        missing: 0,
        changed: 0,
        recovered: 0,
        excluded: 0,
        errorCount: 0,
        errors: [],
      },
    };
    this.jobs.set(source.id, job);
    void this.run(job, source, options).catch((error) => {
      job.status.state = 'failed';
      this.addError(job, '', error);
    });
    return job.status;
  }

  private addError(job: Job, path: string, error: unknown) {
    job.status.errorCount++;
    if (job.status.errors.length < MAX_LISTED_ERRORS) {
      job.status.errors.push({ path, reason: error instanceof Error ? error.message : String(error) });
    }
  }

  private async run(job: Job, source: Omit<Source, 'scan'>, options: { includeExcluded: boolean }) {
    const { status } = job;
    const known = this.store.knownFiles(source.id);
    if (options.includeExcluded) this.store.clearExclusions(source.id);
    const exclusions = this.store.listExclusions(source.id);
    const seen = new Set<string>();
    const unreadable: string[] = [];

    const stack = [''];
    while (stack.length > 0 && !job.cancelled) {
      const dir = stack.pop()!;
      let entries;
      try {
        entries = await readdir(join(source.path, dir), { withFileTypes: true });
      } catch (error) {
        this.addError(job, dir || source.path, error);
        // A folder that has moved away really is missing, unlike one that merely refused access.
        if (dir !== '' || (error as NodeJS.ErrnoException).code !== 'ENOENT') unreadable.push(dir);
        continue;
      }
      if (job.cancelled) break;
      status.foldersVisited++;

      const visible = entries.filter((e) => !e.name.startsWith('.')).sort((a, b) => collator.compare(a.name, b.name));
      // Symbolic links are neither files nor directories to these checks, so they are never followed.
      const subfolders = visible.filter((e) => e.isDirectory());
      for (let i = subfolders.length - 1; i >= 0; i--) stack.push(dir ? `${dir}/${subfolders[i].name}` : subfolders[i].name);

      for (const entry of visible) {
        if (job.cancelled) break;
        if (!entry.isFile()) continue;
        const relPath = dir ? `${dir}/${entry.name}` : entry.name;
        seen.add(relPath);
        try {
          await this.visitFile(job, source, dir, entry.name, relPath, known, exclusions);
        } catch (error) {
          this.addError(job, relPath, error);
        }
      }
    }

    if (job.cancelled) {
      status.state = 'cancelled';
      return;
    }

    // A folder that could not be read says nothing about whether its files still exist.
    const inUnreadable = (relPath: string) => unreadable.some((dir) => dir === '' || relPath.startsWith(`${dir}/`));
    for (const [relPath, file] of known) {
      if (seen.has(relPath) || inUnreadable(relPath) || file.state === 'missing') continue;
      this.store.setFileState(file.id, 'missing');
      status.missing++;
    }
    status.state = 'done';
  }

  private async visitFile(
    job: Job,
    source: Omit<Source, 'scan'>,
    dir: string,
    name: string,
    relPath: string,
    known: ReturnType<Store['knownFiles']>,
    exclusions: Set<string>,
  ) {
    const { status } = job;
    const file = known.get(relPath);
    if (!file && exclusions.has(relPath)) {
      status.excluded++;
      return;
    }

    const info = await stat(join(source.path, relPath));
    const stamp = { size: info.size, mtime: info.mtimeMs };

    if (file && file.size === stamp.size && file.mtime === stamp.mtime) {
      if (file.state === 'ok') {
        status.unchanged++;
      } else if (!job.cancelled) {
        this.store.setFileState(file.id, 'ok');
        status.recovered++;
      }
      return;
    }

    let image;
    try {
      image = await inspectImage(join(source.path, relPath));
    } catch (error) {
      if (!file) throw error;
      image = null;
    }
    if (job.cancelled) return;

    if (!file) {
      if (!image) {
        status.skipped++;
        return;
      }
      this.store.addScannedImage({
        projectId: source.projectId,
        sourceId: source.id,
        relPath,
        filename: name,
        folder: dir ? `${source.name}/${dir}` : source.name,
        mime: image.mime,
        width: image.width,
        height: image.height,
        ...stamp,
      });
      status.added++;
      return;
    }

    if (image && image.width === file.width && image.height === file.height) {
      this.store.setFileState(file.id, 'ok', { ...stamp, mime: image.mime });
      if (file.state === 'ok') status.unchanged++;
      else status.recovered++;
    } else {
      if (file.state !== 'changed') status.changed++;
      this.store.setFileState(file.id, 'changed');
    }
  }
}
