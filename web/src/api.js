// 统一后端请求封装：所有接口返回 {ok, data, error} 包络，这里拆包并在失败时抛错。
// 同源部署：前端与后端同端口（8080），无需配置 baseURL；dev 模式靠 vite proxy 转发。
const BASE = ''

async function request(path, options = {}) {
  const { method = 'GET', body, ...rest } = options
  const headers = { ...(rest.headers || {}) }
  let payload = body
  if (body !== undefined && !(body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'
    payload = JSON.stringify(body)
  }
  let resp
  try {
    resp = await fetch(BASE + path, { method, headers, body: payload, ...rest })
  } catch (e) {
    throw new Error('无法连接后端服务，请确认已启动 scripts/06_serve.py（默认 8080 端口）')
  }
  let json
  try {
    json = await resp.json()
  } catch (e) {
    throw new Error(`响应不是合法 JSON（HTTP ${resp.status}）`)
  }
  if (!resp.ok || json.ok === false) {
    throw new Error(json.error || `请求失败（HTTP ${resp.status}）`)
  }
  return json.data
}

export const api = {
  get: (path) => request(path),
  post: (path, body) => request(path, { method: 'POST', body }),
  patch: (path, body) => request(path, { method: 'PATCH', body }),
}

// 把文件读成 data URL（含前缀），供 /admin/upload 与 /identify 使用
export function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = () => reject(new Error('读取文件失败'))
    reader.readAsDataURL(file)
  })
}

// 轮询直到条件满足或超时：用于训练任务进度
export async function pollUntil(fn, { interval = 1500, timeout = 3600000 } = {}) {
  const start = Date.now()
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const result = await fn()
    if (result) return result
    if (Date.now() - start > timeout) throw new Error('轮询超时')
    await new Promise((r) => setTimeout(r, interval))
  }
}
