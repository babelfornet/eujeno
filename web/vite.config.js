import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base: './' -> relative asset URLs, so the same build works whether it is
// served from a custom domain (eujeno.com) or a GitHub Pages project path
// (https://<user>.github.io/eujeno/). No per-host rebuild needed.
export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})
