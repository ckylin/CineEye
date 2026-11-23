import React, { useState } from 'react'
import Assets from './pages/Assets.jsx'
import Upload from './pages/Upload.jsx'
import Train from './pages/Train.jsx'
import Status from './pages/Status.jsx'

// 管理后台外壳：左侧导航 + 内容区。仅本机/局域网使用，无登录。
const NAV = [
  { key: 'status', label: '状态中心', comp: Status },
  { key: 'upload', label: '上传素材', comp: Upload },
  { key: 'assets', label: '素材库 / 补标', comp: Assets },
  { key: 'train', label: '训练中心', comp: Train },
]

export default function AdminApp() {
  const [page, setPage] = useState('assets')
  const Current = NAV.find((n) => n.key === page)?.comp || Status

  return (
    <div className="admin-layout">
      <aside className="sidebar">
        <div className="brand">影瞳<small>CineEye 管理后台</small></div>
        <nav>
          {NAV.map((n) => (
            <a key={n.key} className={page === n.key ? 'active' : ''} onClick={() => setPage(n.key)}>{n.label}</a>
          ))}
          <a href="/" style={{ marginTop: 12 }}>← 返回用户前台</a>
        </nav>
        <div className="footer">内网自用 · v3.0</div>
      </aside>
      <main className="admin-main">
        <Current />
      </main>
    </div>
  )
}
