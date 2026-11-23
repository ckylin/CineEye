import React, { useState, useEffect, useCallback } from 'react'
import { api } from '../api.js'
import TagForm from '../components/TagForm.jsx'

// 素材库：表格列出 assets，按类型/用途/关键词过滤，点行打开补标抽屉。
export default function Assets() {
  const [items, setItems] = useState([])
  const [kind, setKind] = useState('')
  const [purpose, setPurpose] = useState('')
  const [q, setQ] = useState('')
  const [editing, setEditing] = useState(null)
  const [draft, setDraft] = useState({})
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)

  const load = useCallback(() => {
    const params = new URLSearchParams()
    if (kind) params.set('kind', kind)
    if (purpose) params.set('purpose', purpose)
    if (q) params.set('q', q)
    api.get('/admin/assets?' + params.toString()).then(setItems).catch((e) => setError(e.message))
  }, [kind, purpose, q])

  useEffect(() => { load() }, [load])

  async function save() {
    setSaving(true); setError('')
    try {
      // 只提交可变字段，避免把 readonly 的基础属性回写触发校验问题
      const payload = { ...draft }
      for (const k of ['asset_id', 'sha256', 'width', 'height', 'size', 'duration', 'ext', 'uploaded_at', 'file_name', 'asset_type']) delete payload[k]
      await api.patch('/admin/assets/' + editing.asset_id, payload)
      setEditing(null); load()
    } catch (e) {
      setError(e.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div>
      <div className="admin-header">
        <h1>素材库</h1>
        <button className="btn" onClick={load}>刷新</button>
      </div>
      <div className="card">
        <div className="toolbar">
          <input className="input" placeholder="搜索文件名 / 标签 / 备注" value={q}
            onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && load()} />
          <select className="select" style={{ width: 160 }} value={kind} onChange={(e) => setKind(e.target.value)}>
            <option value="">全部类型</option>
            {['frame','poster','still','behind_scenes','roadshow','actor_portrait','actor_life','screenshot','trailer','clip','other'].map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
          <select className="select" style={{ width: 140 }} value={purpose} onChange={(e) => setPurpose(e.target.value)}>
            <option value="">全部用途</option>
            {['train','query','gallery','reference'].map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
          <button className="btn" onClick={load}>筛选</button>
          <span className="muted small">共 {items.length} 条</span>
        </div>
        {error && <div className="error">{error}</div>}
        {items.length === 0 ? (
          <div className="hint">还没有素材。先到「上传素材」添加图片或视频，上传后在这里补标签。</div>
        ) : (
          <table>
            <thead><tr><th>文件</th><th>类型</th><th>内容类型</th><th>用途</th><th>划分</th><th>演员</th><th>操作</th></tr></thead>
            <tbody>
              {items.map((a) => (
                <tr key={a.asset_id}>
                  <td title={a.file_path}>{a.file_name || a.file_path?.split('/').pop()}</td>
                  <td><span className="badge">{a.asset_type}</span></td>
                  <td>{a.media_kind}</td>
                  <td>{a.purpose}</td>
                  <td>{a.split || <span className="muted">自动</span>}</td>
                  <td>{(a.person_ids || []).length || '—'}</td>
                  <td><button className="btn" onClick={() => { setEditing(a); setDraft({ ...a }) }}>补标</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {editing && (
        <div className="drawer-mask" onClick={() => setEditing(null)}>
          <div className="drawer" onClick={(e) => e.stopPropagation()}>
            <div className="drawer-header">
              <h2>编辑标签</h2>
              <button className="btn" onClick={() => setEditing(null)}>关闭</button>
            </div>
            {error && <div className="error">{error}</div>}
            <TagForm asset={editing} onChange={setDraft} />
            <div style={{ position: 'sticky', bottom: 0, background: 'var(--surface)', padding: '12px 0', borderTop: '1px solid var(--border)' }}>
              <button className="btn btn-primary" onClick={save} disabled={saving}>
                {saving ? <><span className="spinner" /> 保存中…</> : '保存标签'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
