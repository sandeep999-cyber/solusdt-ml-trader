import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/series': 'http://localhost:8000',
      '/inference': 'http://localhost:8000',
      '/reload': 'http://localhost:8000',
      '/checkpoint': 'http://localhost:8000',
    },
  },
})
