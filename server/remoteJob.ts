import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { mkdirSync, rmSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import type { Readable } from 'node:stream';
import type { RemoteJobStatus, RemoteResultsSummary } from '../shared/schema.ts';

const STDERR_TAIL = 2000;

type Job = { status: RemoteJobStatus; cancelled: boolean; stop: () => void };

export type RemoteStart = {
  target: string;
  /** Used for one login and dropped. Empty means the SSH agent or key files. */
  password: string;
  modelSize: string;
  keepOriginals: boolean;
  /** Writes the bundle zip to the given path and returns its bundle id. */
  bundle: (path: string) => Promise<string>;
  /** Imports results.json and best.pt from the folder the script filled. */
  importResults: (dir: string) => Promise<RemoteResultsSummary>;
  /** Starts the local drawing with the new checkpoint, and says what happened. */
  draw: (checkpointId: string) => string;
};

export type RemoteJobOptions = {
  dataDir: string;
  /** The command that runs scripts/remote_train.py; the subcommand and arguments are appended. */
  command: string[];
};

/**
 * Runs the one-click remote flow (specs.md section 10) as a child process of the helper script: bundle
 * locally, upload, train or predict on the server, download, import. The password goes to the script on
 * stdin and is referenced nowhere else, so neither the status nor a log can hold it.
 */
export class RemoteJobManager {
  private jobs = new Map<string, Job>();
  private options: RemoteJobOptions;
  private workRoot: string;

  constructor(options: RemoteJobOptions) {
    this.options = options;
    this.workRoot = resolve(options.dataDir, 'tmp', 'remote');
    rmSync(this.workRoot, { recursive: true, force: true });
  }

  get(projectId: string): RemoteJobStatus | null {
    return this.jobs.get(projectId)?.status ?? null;
  }

  isRunning(projectId: string): boolean {
    return this.get(projectId)?.state === 'running';
  }

  isAnyRunning(): boolean {
    return [...this.jobs.values()].some((job) => job.status.state === 'running');
  }

  /** Stops the local side. Training already started keeps running on the server, and `fetch` can collect it. */
  cancel(projectId: string): boolean {
    const job = this.jobs.get(projectId);
    if (job?.status.state !== 'running') return false;
    job.cancelled = true;
    job.stop();
    return true;
  }

  start(projectId: string, start: RemoteStart): RemoteJobStatus {
    const job: Job = {
      cancelled: false,
      stop: () => {},
      status: {
        state: 'running',
        phase: 'bundling',
        bundleId: null,
        target: start.target,
        modelSize: start.modelSize,
        uploaded: 0,
        uploadTotal: 0,
        gpu: null,
        trainImages: 0,
        epoch: 0,
        bestMap50: null,
        summary: null,
        checkpointId: null,
        drawing: null,
        error: null,
      },
    };
    this.jobs.set(projectId, job);
    const workDir = join(this.workRoot, randomUUID());
    void this.run(job, start, workDir)
      .catch((error) => {
        job.status.state = 'failed';
        job.status.error = error instanceof Error ? error.message : String(error);
      })
      .finally(() => rmSync(workDir, { recursive: true, force: true }));
    return job.status;
  }

  private async run(job: Job, start: RemoteStart, workDir: string) {
    const { status } = job;
    mkdirSync(workDir, { recursive: true });
    const bundlePath = join(workDir, 'bundle.zip');
    status.bundleId = await start.bundle(bundlePath);
    if (job.cancelled) {
      status.state = 'cancelled';
      return;
    }

    const outDir = join(workDir, 'results');
    const [command, ...args] = this.options.command;
    const extra = ['--json', '--password-stdin', '--out', outDir, ...(start.keepOriginals ? ['--keep-originals'] : [])];
    const log = (line: string) => console.log(`[remote ${status.bundleId?.slice(0, 8)}] ${line}`);
    log(`Starting training on ${start.target} (yolov8${start.modelSize})`);
    const child = spawn(command, [...args, 'train', bundlePath, start.target, ...extra], {
      stdio: ['pipe', 'pipe', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });
    job.stop = () => child.kill('SIGTERM');
    child.stdin!.on('error', () => {});
    child.stdin!.end(`${start.password}\n`);
    start.password = '';

    // The script's own output, which never includes the password, goes to the console as it comes.
    createInterface({ input: child.stdout! }).on('line', (line) => {
      const text = line.replace(/\r.*/, '').trim();
      if (text) log(text);
    });
    let stderr = '';
    child.stderr!.setEncoding('utf8').on('data', (text: string) => {
      stderr = (stderr + text).slice(-STDERR_TAIL);
      for (const line of text.split('\n')) if (line.trim()) log(`stderr: ${line.trim()}`);
    });
    let fetched: string | null = null;
    createInterface({ input: child.stdio[3] as Readable }).on('line', (line) => {
      if (job.cancelled) return;
      let message: Record<string, any>;
      try {
        message = JSON.parse(line);
      } catch {
        return;
      }
      switch (message.type) {
        case 'phase':
          status.phase = message.phase;
          log(`phase: ${message.phase}`);
          break;
        case 'upload':
          status.uploaded = message.done;
          status.uploadTotal = message.total;
          break;
        case 'server':
          status.gpu = message.gpu ?? null;
          break;
        case 'prepared':
          status.phase = 'training';
          status.trainImages = message.trainImages;
          break;
        case 'epoch':
          status.epoch = message.epoch;
          status.bestMap50 = message.bestMap50;
          break;
        case 'trained':
          status.epoch = message.epoch;
          status.bestMap50 = message.bestMap50;
          break;
        case 'fetched':
          fetched = message.dir;
          break;
        case 'error':
          status.error = message.message;
          break;
      }
    });

    await new Promise<void>((done) => {
      child.on('error', (error) => {
        status.error = `Could not start ${command}: ${error.message}. The remote run needs Python with paramiko; set LABELAPP_PYTHON to the ml environment.`;
        done();
      });
      child.on('close', (code) => {
        if (code !== 0 && !status.error && !job.cancelled) status.error = stderr.trim() || `The remote script stopped with code ${code}.`;
        done();
      });
    });

    if (job.cancelled) {
      status.state = 'cancelled';
      log('cancelled');
      return;
    }
    if (status.error || !fetched) {
      status.state = 'failed';
      status.error ??= 'The remote script ended without downloading results.';
      log(`failed: ${status.error}`);
      return;
    }
    status.phase = 'importing';
    status.summary = await start.importResults(fetched);
    status.checkpointId = status.summary.checkpointId;
    // Nothing of the run is left on disk once it reports done, so a caller can rely on that.
    rmSync(workDir, { recursive: true, force: true });
    if (status.checkpointId) {
      status.phase = 'drawing';
      status.drawing = start.draw(status.checkpointId);
    } else {
      status.drawing = 'The server returned no trained model, so nothing was drawn.';
    }
    status.state = 'done';
    log(`done: checkpoint ${status.checkpointId ?? 'not added'}. ${status.drawing}`);
  }
}
