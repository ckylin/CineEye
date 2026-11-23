import React, { useState, useEffect } from 'react'
import { api } from '../api.js'

// 状态中心：模型 / 图库 / 元数据 / 演员 / 素材 / 任务 一屏概览。
export default function Status() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api.get('/admin/status').then(setData).catch((e) => setError(e.message))
    const timer = setInterval(() => api.get('/admin/status').then(setData).catch(() => {}), 5000)
    return () => clearInterval(timer)
  }, [])

  if (error) return <div className="error">{error}</div>
  if (!data) return <div className="muted"><span className="spinner" /> 加载状态…</div>

  const model = data.model
  const gallery = data.gallery || {}
  const meta = data.metadata || {}
  const actors = data.actors || {}

  return (
    <div>
      <div className="admin-header"><h1>状态中心</h1><span className="small muted">每 5 秒自动刷新</span></div>
      <div className="stat-grid">
        <div className="stat"><div className="num">{model ? model.num_classes : '—'}</div><div className="label">模型类别数</div></div>
        <div className="stat"><div className="num">{gallery.images ?? '—'}</div><div className="label">图库图片</div></div>
        <div className="stat"><div className="num">{meta.movies ?? '—'}</div><div className="label">电影档案</div></div>
        <div className="stat"><div className="num">{actors.total ?? actors.actors ?? '—'}</div><div className="label">演员</div></div>
        <div className="stat"><div className="num">{data.assets?.total ?? '—'}</div><div className="label">素材总数</div></div>
        <div className="stat"><div className="num">{model?.device || '—'}</div><div className="label">推理设备</div></div>
      </div>

      <div className="grid-2" style={{ marginTop: 20 }}>
        <div className="card">
          <h2>模型</h2>
          {model ? (
            <table><tbody>
              <tr><td>类别数</td><td>{model.num_classes}</td></tr>
              <tr><td>主干 / 模式</td><td>{model.backbone}</td></tr>
              <tr><td>设备</td><td>{model.device}</td></tr>
            </tbody></table>
          ) : <div className="hint">电影模型未加载（best.pt 可能不存在，演员/图库功能仍可用）</div>}
        </div>
        <div className="card">
          <h2>素材分布</h2>
          <div className="small muted">按内容类型</div>
          {Object.entries(data.assets?.by_kind || {}).length === 0 ? <div className="hint">暂无素材</div> : (
            <table><tbody>
              {Object.entries(data.assets?.by_kind || {}).map(([k, v]) => <tr key={k}><td>{k}</td><td>{v}</td></tr>)}
            </tbody></table>
          )}
        </div>
      </div>
    </div>
  )
}
