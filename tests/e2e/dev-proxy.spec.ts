import { expect, test } from '@playwright/test';

const DEV = 'http://localhost:5174';

test('writes through the Vite dev proxy pass the Origin check', async ({ request }) => {
  const response = await request.post(`${DEV}/api/projects`, {
    headers: { Origin: DEV },
    data: { name: 'through the proxy' },
  });
  expect(response.status()).toBe(201);
});

test('the dev proxy still rejects cross-site requests', async ({ request }) => {
  const response = await request.post(`${DEV}/api/projects`, {
    headers: { Origin: 'https://evil.example' },
    data: { name: 'nope' },
  });
  expect(response.status()).toBe(403);
});
