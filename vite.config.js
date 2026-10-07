import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true,  
    port: 3000,
    headers: {
      'Cross-Origin-Embedder-Policy': 'credentialless',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
    fs: {
      allow: ['..']
    }
  },
  assetsInclude: ['**/*.onnx', '**/*.wasm'],
  // Ensure ONNX files are treated as static assets
  publicDir: 'public',
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  },
  build: {
    rollupOptions: {
      external: []
    }
  },
  // Add custom handling for WebAssembly and ONNX files
  define: {
    global: 'globalThis',
  },
  worker: {
    format: 'es'
  }
})
