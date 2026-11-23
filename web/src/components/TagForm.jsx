import React, { useState, useEffect } from 'react'
import { api } from '../api.js'

// 按 /tag-schema 契约动态生成五层标签表单。字段、枚举、中文名都来自后端契约，
// 新增 media_kind 只需改 configs/tag_schema.json，这里自动生效。
export default function TagForm({ asset, onChange }) {
  const [schema, setSchema] = useState(null)
  const [values, setValues] = useState({})
  const [movies, setMovies] = useState([])
  const [actors, setActors] = useState([])
  const [error, setError] = useState('')

  useEffect(() => {
    api.get('/tag-schema').then(setSchema).catch((e) => setError(e.message))
    // 拉电影/演员候选，供 multi_ref 多选
    api.get('/movies?limit=500').then((d) => setMovies(d.movies || [])).catch(() => {})
    api.get('/actors?limit=500').then((d) => setActors(d.actors || [])).catch(() => {})
  }, [])

  useEffect(() => {
    if (asset) setValues({ ...asset })
  }, [asset])

  function set(key, value) {
    const next = { ...values, [key]: value }
    setValues(next)
    onChange?.(next)
  }

  function renderField(f) {
    const key = f.key
    const val = values[key]
    const label = <span>{f.label}{f.optional_compliance && <span className="muted">（可选）</span>}</span>
    if (f.readonly) {
      return <label className="field" key={key}>{label}
        <div className="input" style={{ background: 'var(--surface-2)', color: 'var(--text-2)' }}>
          {typeof val === 'object' ? JSON.stringify(val) : (val ?? '—')}
        </div>
      </label>
    }
    if (f.type === 'enum') {
      return <label className="field" key={key}>{label}
        <select className="select" value={val ?? f.default ?? ''} onChange={(e) => set(key, e.target.value)}>
          {(f.options || []).map((o) => (
            <option value={o} key={o}>{f.labels?.[o] || o || '自动'}</option>
          ))}
        </select>
      </label>
    }
    if (f.type === 'bool') {
      return <label className="field" key={key}>
        <label style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <input type="checkbox" checked={!!val} onChange={(e) => set(key, e.target.checked)} />
          <span>{f.label}</span>
        </label>
      </label>
    }
    if (f.type === 'multi_ref') {
      const list = f.ref === 'movies' ? movies : actors
      const selected = Array.isArray(val) ? val : []
      return <label className="field" key={key}>{label}
        <select className="select" multiple value={selected} size={Math.min(5, Math.max(3, list.length))}
          onChange={(e) => set(key, Array.from(e.target.selectedOptions).map((o) => o.value))}>
          {list.map((item) => (
            <option value={item.id} key={item.id}>
              {item.primary_title || item.name} {item.year ? `(${item.year})` : ''}
            </option>
          ))}
        </select>
        <span className="small muted">按住 Ctrl / Cmd 可多选</span>
      </label>
    }
    if (f.type === 'tags') {
      const text = Array.isArray(val) ? val.join(', ') : (val || '')
      return <label className="field" key={key}>{label}
        <input className="input" placeholder={f.placeholder} value={text}
          onChange={(e) => set(key, e.target.value.split(',').map((s) => s.trim()).filter(Boolean))} />
      </label>
    }
    return <label className="field" key={key}>{label}
      <input className="input" type={f.type === 'number' ? 'number' : 'text'}
        placeholder={f.placeholder} value={val ?? ''} onChange={(e) => set(key, e.target.value)} />
    </label>
  }

  if (error) return <div className="error">{error}</div>
  if (!schema) return <div className="muted"><span className="spinner" /> 加载标签契约…</div>

  return (
    <div>
      {schema.layers.map((layer) => (
        <div className="layer" key={layer.key}>
          <h4>{layer.label}</h4>
          {layer.hint && <div className="hint">{layer.hint}</div>}
          <div className="grid-2">
            {layer.fields.filter((f) => !f.readonly || ['basic'].includes(layer.key)).map(renderField)}
          </div>
        </div>
      ))}
    </div>
  )
}
