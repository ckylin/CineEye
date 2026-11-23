import React from 'react'
import { createRoot } from 'react-dom/client'
import UserApp from './UserApp.jsx'
import AdminApp from './AdminApp.jsx'
import './styles.css'

// 单一 SPA，按 URL 路径选择前台或后台（serve.py 对 / 与 /admin 都回退 index.html）
const isAdmin = window.location.pathname.startsWith('/admin')
const Root = isAdmin ? AdminApp : UserApp

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
)
