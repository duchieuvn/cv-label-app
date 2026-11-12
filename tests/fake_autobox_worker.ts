#!/usr/bin/env node
// Stands in for server/autobox_worker.py, speaking the same protocol without Python or training.
// FAKE_AUTOBOX selects a behavior: ok (default), hang, crash or error.
// FAKE_AUTOBOX_COPY is a path to keep a copy of the job file at, since the job's folder is removed.
import { copyFileSync, readFileSync, writeFileSync, writeSync } from 'node:fs';
import { join } from 'node:path';

const jobPath = process.argv.at(-1)!;
const job = JSON.parse(readFileSync(jobPath, 'utf8'));
const send = (message: object) => writeSync(3, `${JSON.stringify(message)}\n`);
const mode = process.env.FAKE_AUTOBOX ?? 'ok';

if (process.env.FAKE_AUTOBOX_COPY) copyFileSync(jobPath, process.env.FAKE_AUTOBOX_COPY);

if (mode === 'crash') {
  console.error('worker exploded');
  process.exit(3);
}
if (mode === 'error') {
  send({ type: 'error', message: 'no ultralytics here' });
  process.exit(1);
}

if (job.train) {
  const split = (name: string) => job.training.filter((item: { split: string }) => item.split === name).length;
  send({ type: 'prepared', trainImages: split('train'), valImages: split('val') });
  send({ type: 'epoch', epoch: 1, bestMap50: 0.5 });
  const weights = join(job.workDir, 'best.pt');
  writeFileSync(weights, 'fake model');
  send({ type: 'trained', epoch: 2, stoppedEarly: true, bestMap50: 0.75, weights });
}

if (mode === 'hang') {
  send({ type: 'pred', id: job.targets[0].id, boxes: [{ cls: 0, x: 1, y: 1, w: 5, h: 5 }] });
  setInterval(() => {}, 1000);
} else {
  for (const target of job.targets) {
    send({ type: 'pred', id: target.id, boxes: [{ cls: 0, x: 1, y: 1, w: target.width / 2, h: target.height / 2 }] });
  }
}
