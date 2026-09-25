import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The dev server proxies /api to FastAPI so the browser sees a single origin.
export default defineConfig({
  plugins: [react()],
  resolve: {
    // react-pdf pulls its own React resolution path. Without deduping, Vite's
    // dependency optimizer can hand the app and the viewer two different React
    // instances, which fails at runtime with "Invalid hook call".
    dedupe: ['react', 'react-dom'],
  },
  optimizeDeps: {
    include: ['react', 'react-dom', 'react-pdf', 'pdfjs-dist'],
  },
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET || 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 1200,
  },
});
