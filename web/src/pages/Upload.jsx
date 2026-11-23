import React, { useState, useRef } from 'react'
import { api, fileToDataUrl } from '../api.js'

// 上传素材：拖拽/选择多文件 → 逐个 base64 上传入库 → 提示去素材库补标。
export default function Upload() {
  const [files, setFiles] = useState([])
  const [uploading, setUploading] = useState(false)
  const [results, setResults] = useState([])
  const [error, setError] = useState('')
  const [drag, setDrag] = useState(false)
  const inputRef = useRef(null)

  function addFiles(list) {
    const arr = Array.from(list || [])
    if (arr.length) { setFiles((prev) => [...prev, ...arr]); setError(''); setResults([]) }
  }

  function onDrop(e) { e.preventDefault(); setDrag(false); addFiles(e.dataTransfer.files) }

  async function upload() {
    if (!files.length) { setError('请先选择文件'); return }
    setUploading(true); setError(''); setResults([])
    const done = []
    for (const f of files) {
      try {
        const dataUrl = await fileToDataUrl(f)
        const asset = await api.post('/admin/upload', {
          file_base64: dataUrl,
          file_name: f.name,
          tags: { media_kind: f.type.startsWith('image/') ? 'screenshot' : 'clip' },
        })
        done.push({ name: f.name, id: asset.asset_id, duplicate: asset._reused || false })
      } catch (e) {
        done.push({ name: f.name, error: e.message })
      }
    }
    setResults(done); setFiles([]); setUploading(false)
  }

  return (
    <div>
      <div className="admin-header"><h1>上传素材</h1></div>
      <div className="card">
        <div className={`dropzone ${drag ? 'drag' : ''}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)} onDrop={onDrop}>
          <div className="icon">＋</div>
          <div>拖拽图片或视频到这里，或点击选择（可多选）</div>
          <div className="small muted">上传后到「素材库」补充标签与关联</div>
          <input ref={inputRef} type="file" multiple style={{ display: 'none' }}
            accept="image/*,video/*" onChange={(e) => addFiles(e.target.files)} />
        </div>

        {files.length > 0 && (
          <div style={{ marginTop: 16 }}>
            <div className="small muted" style={{ marginBottom: 6 }}>待上传 {files.length} 个文件</div>
            {files.map((f, i) => <div key={i} className="actor-row small">{f.name} <span className="muted">{(f.size / 1024).toFixed(0)} KB</span></div>)}
          </div>
        )}

        <div style={{ marginTop: 18 }}>
          <button className="btn btn-primary" onClick={upload} disabled={uploading || !files.length}>
            {uploading ? <><span className="spinner" /> 上传中…</> : '开始上传'}
          </button>
        </div>

        {error && <div className="error">{error}</div>}
        {results.length > 0 && (
          <div style={{ marginTop: 16 }}>
            <h3>上传结果</h3>
            <div className="hint">素材已入库。请到「素材库」为它们补充内容类型、关联电影/演员、训练用途等标签。</div>
            {results.map((r, i) => (
              <div className="actor-row small" key={i}>
                <span className={r.error ? 'badge badge-danger' : 'badge badge-success'}>{r.error ? '失败' : '成功'}</span>
                <span>{r.name}</span>
                {r.error ? <span className="error" style={{ margin: 0, padding: 0, border: 0, background: 'none' }}>{r.error}</span> : <span className="muted">{r.id}</span>}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
