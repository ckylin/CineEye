import React, { useState, useRef } from 'react'
import { api, fileToDataUrl } from './api.js'
import ResultCard from './components/ResultCard.jsx'

// 用户前台：上传图片/视频 → 识别 → 结果卡片。用户无需接触命令行。
export default function UserApp() {
  const [mode, setMode] = useState('movie') // movie=综合识别 / actor=只认演员
  const [mediaType, setMediaType] = useState('image') // image / video
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState([])
  const [error, setError] = useState('')
  const [drag, setDrag] = useState(false)
  const inputRef = useRef(null)

  function pickFile(f) {
    if (!f) return
    setFile(f)
    setError('')
    setResults([])
    if (f.type.startsWith('image/')) {
      fileToDataUrl(f).then(setPreview).catch(() => setPreview(''))
      setMediaType('image')
    } else {
      setPreview('')
      setMediaType('video')
    }
  }

  function onDrop(e) {
    e.preventDefault()
    setDrag(false)
    pickFile(e.dataTransfer.files[0])
  }

  async function identify() {
    if (!file) { setError('请先选择一张图片或一段视频'); return }
    setLoading(true); setError(''); setResults([])
    try {
      const dataUrl = await fileToDataUrl(file)
      let data
      if (mode === 'actor') {
        // 只认演员：多脸识别，不依赖电影模型
        data = await api.post('/actors/recognize', { image_base64: dataUrl })
        setResults((data.results || []).map((r) => ({
          label: '演员识别', faces: r.faces || [], _preview: dataUrl,
        })))
      } else if (mediaType === 'video') {
        // 视频识别：前端拿到 data URL 没有服务端路径，先上传成素材再识别其路径
        const asset = await api.post('/admin/upload', {
          file_base64: dataUrl, file_name: file.name, tags: { media_kind: 'clip' },
        })
        data = await api.post('/predict/video', { video: asset.file_path, num_frames: 16 })
        setResults([{ ...data, _preview: '' }])
      } else {
        // 图片综合识别：/identify 返回 { matches: [电影档案...], actor, ... }。
        // 注意电影档案在 matches 数组里，必须逐个取出来传给卡片；
        // 早期版本误把整个响应对象当成一部电影塞给 ResultCard，
        // 卡片找不到 title 字段就显示兜底文案"未知电影"（明明识别成功却提示未知）。
        data = await api.post('/identify', { image_base64: dataUrl })
        const matches = data.matches || []
        if (!matches.length) {
          setError('没有匹配到已知电影，可以试试官方海报或换一张剧照')
        }
        // 每张候选单独成卡，带上原图预览；演员信息由卡片从档案的 cast 字段读取
        setResults(matches.map((m) => ({ ...m, _preview: dataUrl })))
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <div className="hero">
        <h1>影瞳</h1>
        <p>上传一张剧照、海报或视频，认出它是哪部电影、有哪些演员</p>
      </div>

      <div className="card" style={{ maxWidth: 720, margin: '0 auto' }}>
        <div className="tabs">
          <button className={`tab ${mode === 'movie' ? 'active' : ''}`} onClick={() => setMode('movie')}>识别电影</button>
          <button className={`tab ${mode === 'actor' ? 'active' : ''}`} onClick={() => setMode('actor')}>只认演员</button>
        </div>

        <div
          className={`dropzone ${drag ? 'drag' : ''}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
        >
          <div className="icon">＋</div>
          {file ? (
            <div>
              <div>{file.name}</div>
              <div className="small muted">点击更换文件</div>
            </div>
          ) : (
            <div>
              <div>拖拽图片或视频到这里，或点击选择</div>
              <div className="small muted">支持 JPG / PNG / WEBP / MP4 等</div>
            </div>
          )}
          <input
            ref={inputRef} type="file" style={{ display: 'none' }}
            accept="image/*,video/*"
            onChange={(e) => pickFile(e.target.files[0])}
          />
        </div>

        <div style={{ textAlign: 'center', marginTop: 18 }}>
          <button className="btn btn-primary" onClick={identify} disabled={loading || !file}>
            {loading ? <><span className="spinner" /> 识别中…</> : '开始识别'}
          </button>
        </div>

        {error && <div className="error">{error}</div>}
      </div>

      {results.length > 0 && (
        <div style={{ maxWidth: 1000, margin: '0 auto' }}>
          <h2 style={{ fontSize: 18, margin: '8px 0 16px' }}>识别结果</h2>
          <div className="result-grid">
            {results.map((r, i) => (
              <ResultCard key={i} result={r} imageUrl={r._preview} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
