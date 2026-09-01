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

/*
 * Prefixes, not exact paths. '/image' covers both /image/<sha> and
 * /images?offset=... - listing only '/images' let /image/<sha> fall
 * through to the SPA fallback, and the page parsed index.html as JSON.
 */
const ENDPOINTS = [
  '/search',
  '/image',
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
