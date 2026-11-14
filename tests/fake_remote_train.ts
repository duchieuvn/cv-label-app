#!/usr/bin/env node
// Stands in for scripts/remote_train.py in tests: speaks its --json protocol without SSH or training.
// FAKE_REMOTE selects a behavior: ok (default), fail (bad login), or hang. The password it reads from stdin is
// written to FAKE_REMOTE_PASSWORD_FILE, so a test can check what the app sent without it ever being logged.
import { mkdirSync, readFileSync, writeFileSync, writeSync } from 'node:fs';
import { join } from 'node:path';
import { inflateRawSync } from 'node:zlib';

const args = process.argv.slice(2);
const bundlePath = args[args.indexOf('train') + 1];
const outDir = args[args.indexOf('--out') + 1];
const send = (message: object) => writeSync(3, `${JSON.stringify(message)}\n`);
const mode = process.env.FAKE_REMOTE ?? 'ok';

const password = readFileSync(0, 'utf8').split('\n')[0];
if (process.env.FAKE_REMOTE_PASSWORD_FILE) writeFileSync(process.env.FAKE_REMOTE_PASSWORD_FILE, password);

/** The job file from the bundle's central directory, as tests/helpers.ts reads zips. */
function jobOf(zip: Buffer) {
  let end = zip.length - 22;
  while (zip.readUInt32LE(end) !== 0x06054b50) end--;
  let offset = zip.readUInt32LE(end + 16);
  for (let i = 0; i < zip.readUInt16LE(end + 10); i++) {
    const method = zip.readUInt16LE(offset + 10);
    const size = zip.readUInt32LE(offset + 20);
    const nameLength = zip.readUInt16LE(offset + 28);
    const name = zip.subarray(offset + 46, offset + 46 + nameLength).toString();
    const local = zip.readUInt32LE(offset + 42);
    const start = local + 30 + zip.readUInt16LE(local + 26) + zip.readUInt16LE(local + 28);
    if (name === 'job.json') {
      const data = zip.subarray(start, start + size);
      return JSON.parse((method === 8 ? inflateRawSync(data) : data).toString());
    }
    offset += 46 + nameLength + zip.readUInt16LE(offset + 30) + zip.readUInt16LE(offset + 32);
  }
  throw new Error('no job.json');
}

send({ type: 'phase', phase: 'connecting' });
if (mode === 'fail') {
  send({ type: 'error', message: 'Login failed: wrong password, or no SSH key the server accepts.' });
  process.exit(1);
}
const job = jobOf(readFileSync(bundlePath));
send({ type: 'phase', phase: 'preparing' });
send({ type: 'server', gpu: 'Fake GPU, 24576 MiB', torch: '2.0 True' });
send({ type: 'phase', phase: 'uploading' });
send({ type: 'upload', done: 1000, total: 1000 });
if (mode === 'hang') {
  setInterval(() => {}, 1000);
} else {
  if (job.train) {
    send({ type: 'phase', phase: 'training' });
    send({ type: 'prepared', trainImages: job.training.filter((t: { split: string }) => t.split === 'train').length, valImages: 1 });
    send({ type: 'epoch', epoch: 1, bestMap50: 0.4 });
    send({ type: 'trained', epoch: 3, stoppedEarly: true, bestMap50: 0.8 });
  } else {
    send({ type: 'phase', phase: 'predicting' });
  }
  const keypoints = (x: number, y: number) => (job.task === 'pose' ? { keypoints: [[x + 2, y + 2, 0.9], [x + 3, y + 3, 0.1]] } : {});
  const predictions = job.targets.map((target: { id: string; width: number; height: number }) => {
    send({ type: 'pred', id: target.id });
    return { id: target.id, boxes: [{ cls: 0, x: 1, y: 1, w: target.width / 2, h: target.height / 2, ...keypoints(1, 1) }] };
  });
  send({ type: 'done' });
  send({ type: 'phase', phase: 'downloading' });
  mkdirSync(outDir, { recursive: true });
  writeFileSync(
    join(outDir, 'results.json'),
    JSON.stringify({
      bundleId: job.bundleId,
      task: job.task,
      trainedAt: '2026-11-12T12:00:00.000Z',
      trainImages: job.train ? job.training.length - 1 : 0,
      epochs: job.train ? 3 : 0,
      bestMap50: job.train ? 0.8 : null,
      stoppedEarly: true,
      predictions,
      skipped: [],
    }),
  );
  writeFileSync(join(outDir, 'best.pt'), 'remote weights');
  send({ type: 'fetched', dir: outDir });
}
