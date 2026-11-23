import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 构建产物写入项目根 static/，由 cineeye/serve.py 同端口托管。
// 运行时不依赖 CDN：React 已打进 bundle，离线机器打开即用。
// dev 模式下把 /api 外的所有后端请求代理到本地 8080 服务，方便前端热更新联调。
export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: '../static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8080',
      '/health': 'http://127.0.0.1:8080',
      '/labels': 'http://127.0.0.1:8080',
      '/predict': 'http://127.0.0.1:8080',
      '/identify': 'http://127.0.0.1:8080',
      '/search': 'http://127.0.0.1:8080',
      '/movies': 'http://127.0.0.1:8080',
      '/actors': 'http://127.0.0.1:8080',
      '/gallery': 'http://127.0.0.1:8080',
      '/tag-schema': 'http://127.0.0.1:8080',
      '/admin': 'http://127.0.0.1:8080',
    },
  },
})
