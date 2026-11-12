import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { defineConfig } from '@playwright/test';

const dataDir = mkdtempSync(join(tmpdir(), 'labelapp-e2e-'));

// Uses the system Chrome so no browser download is needed.
export default defineConfig({
  testDir: 'tests/e2e',
  use: { baseURL: 'http://127.0.0.1:3101', channel: 'chrome', viewport: { width: 1400, height: 900 } },
  webServer: [
    {
      command: 'npm run build && npm start',
      url: 'http://127.0.0.1:3101/api/projects',
      // The browser tests cover the auto box UI, not training, which the worker's own run covers.
      env: { PORT: '3101', DATA_DIR: dataDir, LABELAPP_PYTHON: resolve('tests/fake_autobox_worker.ts') },
      reuseExistingServer: false,
    },
    {
      // The dev proxy, used by tests/e2e/dev-proxy.spec.ts.
      command: 'npx vite --port 5174 --strictPort',
      url: 'http://localhost:5174',
      env: { API_TARGET: 'http://127.0.0.1:3101' },
      reuseExistingServer: false,
    },
  ],
});
