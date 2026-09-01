import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * The API is read-only, unauthenticated and bound to loopback, so the
 * dev server proxies to it rather than the browser calling it directly.
 * That keeps CORS out of the API - which would otherwise have to permit
 * an origin it has no way to authenticate - and keeps the port out of
 * component code, where it would be duplicated across every fetch.
 */
const API = 'http://127.0.0.1:8000'

const ENDPOINTS = [
  '/search',
  '/images',
  '/people',
  '/events',
  '/palette',
  '/duplicates',
  '/status',
  '/thumbnails',
]

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(
      ENDPOINTS.map((path) => [path, { target: API, changeOrigin: false }]),
    ),
  },
})
