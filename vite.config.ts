import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  root: 'client',
  plugins: [react()],
  build: { outDir: '../dist', emptyOutDir: true },
  server: {
    proxy: {
      '/api': {
        target: process.env.API_TARGET ?? 'http://127.0.0.1:3001',
        // The server compares Origin with Host to reject cross-site requests, so Host must stay
        // as the browser sent it. Vite rewrites it to the target unless told otherwise.
        changeOrigin: false,
      },
    },
  },
});
