import React, { useState, useEffect, useCallback } from 'react'
import { api } from '../api.js'

// 训练中心：选任务类型触发训练（异步），任务列表 + 进度轮询 + 实时日志。
export default function Train() {
  const [kind, setKind] = useState('train_movie')
  const [epochs, setEpochs] = useState('')
  const [tasks, setTasks] = useState([])
  const [selected, setSelected] = useState(null)
  const [detail, setDetail] = useState(null)
  const [error, setError] = useState('')
  const [starting, setStarting] = useState(false)

  const loadTasks = useCallback(() => {
    api.get('/admin/tasks').then((d) => setTasks(d.tasks || [])).catch((e) => setError(e.message))
  }, [])

  useEffect(() => { loadTasks() }, [loadTasks])

  // 选中任务或存在 running/queued 任务时，每 2 秒轮询详情与列表
  useEffect(() => {
    const hasActive = tasks.some((t) => t.status === 'running' || t.status === 'queued')
    if (!selected && !hasActive) return
    const timer = setInterval(async () => {
      if (selected) {
        try { setDetail(await api.get('/admin/tasks/' + selected)) } catch (e) { setError(e.message) }
      }
      loadTasks()
    }, 2000)
    return () => clearInterval(timer)
  }, [selected, tasks, loadTasks])

  async function start() {
    setStarting(true); setError('')
    try {
      const payload = { kind }
      if (epochs) payload.epochs = Number(epochs)
      const d = await api.post('/admin/train', payload)
      setSelected(d.task_id)
      setDetail(await api.get('/admin/tasks/' + d.task_id))
      loadTasks()
    } catch (e) {
      setError(e.message)
    } finally {
      setStarting(false)
    }
  }

  async function cancel(id) {
    try { await api.post('/admin/tasks/' + id + '/cancel', {}); loadTasks(); if (detail) setDetail(await api.get('/admin/tasks/' + id)) }
    catch (e) { setError(e.message) }
  }

  const statusBadge = (s) => ({
    queued: 'badge-warning', running: 'badge-primary', done: 'badge-success',
    failed: 'badge-danger', canceled: 'badge',
  }[s] || 'badge')

  return (
    <div>
      <div className="admin-header"><h1>训练中心</h1></div>
      <div className="grid-2">
        <div>
          <div className="card">
            <h2>触发训练</h2>
            <label className="field"><span>任务类型</span>
              <select className="select" value={kind} onChange={(e) => setKind(e.target.value)}>
                <option value="train_movie">电影识别模型（抽帧→清单→训练）</option>
                <option value="train_actor">演员识别模型（导入→训练）</option>
                <option value="build_gallery">重建图库索引</option>
                <option value="refresh_derived">同步标签到现有流程</option>
              </select>
            </label>
            <label className="field"><span>训练轮数（留空用默认）</span>
              <input className="input" type="number" placeholder="如 40" value={epochs} onChange={(e) => setEpochs(e.target.value)} />
            </label>
            <div className="hint">任务在后台异步执行，点击后立即返回任务号，不会卡住页面。可在下方查看进度与日志。</div>
            {error && <div className="error">{error}</div>}
            <button className="btn btn-primary" onClick={start} disabled={starting}>
              {starting ? <><span className="spinner" /> 提交中…</> : '开始训练'}
            </button>
          </div>
        </div>

        <div>
          <div className="card">
            <h2>任务列表</h2>
            {tasks.length === 0 ? <div className="hint">还没有训练任务</div> : (
              <table>
                <thead><tr><th>任务</th><th>类型</th><th>状态</th><th>进度</th><th></th></tr></thead>
                <tbody>
                  {tasks.map((t) => (
                    <tr key={t.id} onClick={() => setSelected(t.id)} style={{ cursor: 'pointer', background: selected === t.id ? '#f5f9ff' : undefined }}>
                      <td className="small">{t.id.slice(0, 22)}</td>
                      <td className="small">{t.kind}</td>
                      <td><span className={`badge ${statusBadge(t.status)}`}>{t.status}</span></td>
                      <td className="small">{t.progress}%</td>
                      <td>
                        {(t.status === 'running' || t.status === 'queued') &&
                          <button className="btn btn-danger small" onClick={(e) => { e.stopPropagation(); cancel(t.id) }}>取消</button>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>

      {detail && (
        <div className="card">
          <div className="admin-header">
            <h2>任务详情 · {detail.id}</h2>
            <span className={`badge ${statusBadge(detail.status)}`}>{detail.status}</span>
          </div>
          <div className="small muted">类型：{detail.kind} ｜ 创建：{detail.created_at} ｜ 进度：{detail.progress}%</div>
          <div className="progress"><div style={{ width: detail.progress + '%' }} /></div>
          <h3 style={{ marginTop: 16 }}>实时日志</h3>
          <div className="log-view">{detail.log_tail || '（暂无日志输出）'}</div>
          {detail.result && Object.keys(detail.result).length > 0 && (
            <div className="hint" style={{ marginTop: 12 }}>结果：{JSON.stringify(detail.result)}</div>
          )}
        </div>
      )}
    </div>
  )
}
