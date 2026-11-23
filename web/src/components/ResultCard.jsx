import React from 'react'

// 识别结果卡片：展示电影名、置信度、多语言别名、演员；兼容 /identify 与 /actors/recognize 的结构。
export default function ResultCard({ result, imageUrl }) {
  const movie = result.movie || result
  const title = movie.primary_title || movie.title || movie.label || '未知电影'
  const conf = movie.confidence ?? movie.best_score ?? movie.score
  const aliases = movie.aliases || movie.alias_candidates || []
  const cast = movie.cast || movie.actors || []
  const faces = result.faces || []

  return (
    <div className="result-card">
      {imageUrl && <img className="thumb" src={imageUrl} alt="输入预览" />}
      <div className="body">
        <div className="title">{title}</div>
        {conf != null && (
          <div className="conf small">置信度 {(conf * 100).toFixed(1)}%</div>
        )}
        {movie.year && <div className="muted small">{movie.year}</div>}

        {aliases.length > 0 && (
          <div style={{ marginTop: 10 }}>
            <div className="small muted" style={{ marginBottom: 4 }}>别名</div>
            {aliases.slice(0, 12).map((a, i) => {
              const text = typeof a === 'string' ? a : (a.alias || a.name || '')
              return <span className="alias-chip" key={i}>{text}</span>
            })}
          </div>
        )}

        {cast.length > 0 && (
          <div style={{ marginTop: 10 }}>
            <div className="small muted" style={{ marginBottom: 4 }}>演员</div>
            {cast.slice(0, 8).map((a, i) => {
              const name = typeof a === 'string' ? a : (a.name || a.character || '')
              return <div className="actor-row small" key={i}>{name}</div>
            })}
          </div>
        )}

        {faces.length > 0 && (
          <div style={{ marginTop: 10 }}>
            <div className="small muted" style={{ marginBottom: 4 }}>识别到的人脸</div>
            {faces.map((f, i) => (
              <div className="actor-row small" key={i}>
                <span className="badge badge-primary">#{i + 1}</span>
                <span>{f.actor || '未知人物'}</span>
                {f.confidence != null && <span className="muted">{(f.confidence * 100).toFixed(0)}%</span>}
              </div>
            ))}
          </div>
        )}

        {!title && <div className="muted small">没有识别到明确结果</div>}
      </div>
    </div>
  )
}
