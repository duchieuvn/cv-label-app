import { delimiter, resolve } from 'node:path';
import { createApp } from './app.ts';

const port = Number(process.env.PORT ?? 3001);
const host = process.env.HOST ?? '127.0.0.1';
const dataDir = resolve(process.env.DATA_DIR ?? 'data');

const importRoots = (process.env.IMPORT_ROOTS ?? '').split(delimiter).filter(Boolean);

const { app } = createApp(dataDir, { host, importRoots, python: process.env.LABELAPP_PYTHON });
app.listen(port, host, () => {
  console.log(`LabelApp listening on http://${host}:${port} (data: ${dataDir})`);
});
