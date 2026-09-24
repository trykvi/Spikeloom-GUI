import type { Checkpoint, GraphDocument, Telemetry } from './types'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options)
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`
    try { const body = await response.json(); message = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body) } catch { /* no JSON */ }
    throw new Error(message)
  }
  return response.json() as Promise<T>
}
const json = (value: unknown) => ({ headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value) })
export const api = {
  telemetry: () => request<Telemetry>('/api/telemetry'),
  control: (action: string) => request<Telemetry>('/api/control', { method: 'POST', ...json({ action }) }),
  speed: (ticks_per_second: number) => request<Telemetry>('/api/tickspeed', { method: 'PUT', ...json({ ticks_per_second }) }),
  graphs: () => request<{ id: string; name: string; node_count: number }[]>('/api/graphs'),
  graph: (id: string) => request<GraphDocument>(`/api/graphs/${encodeURIComponent(id)}`),
  saveGraph: (graph: GraphDocument) => request<GraphDocument>(`/api/graphs/${encodeURIComponent(graph.id)}`, { method: 'PUT', ...json(graph) }),
  validateGraph: (graph: GraphDocument) => request<{ valid: boolean; errors: string[] }>('/api/graphs/validate', { method: 'POST', ...json(graph) }),
  checkpoints: () => request<Checkpoint[]>('/api/checkpoints'),
  uploadCheckpoint: async (file: File) => { const form = new FormData(); form.append('file', file); return request<Checkpoint>('/api/checkpoints', { method: 'POST', body: form }) }
}

export function connectTelemetry(onData: (data: Telemetry) => void, onConnection: (online: boolean) => void): () => void {
  let closed = false
  let socket: WebSocket | undefined
  let retry: number | undefined
  const connect = () => {
    if (closed) return
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    socket = new WebSocket(`${protocol}//${location.host}/ws/telemetry`)
    socket.onopen = () => onConnection(true)
    socket.onmessage = event => { try { onData(JSON.parse(event.data) as Telemetry) } catch { /* malformed frame */ } }
    socket.onclose = () => { onConnection(false); if (!closed) retry = window.setTimeout(connect, 2000) }
    socket.onerror = () => socket?.close()
  }
  connect()
  return () => { closed = true; if (retry) clearTimeout(retry); socket?.close() }
}

export function connectDetail<T>(channel: 'environment' | 'spikes', selection: { modelId?: string; iterationId?: string; agentId?: string }, onData: (frame: T) => void, onConnection: (online: boolean) => void): () => void {
  let closed = false
  let socket: WebSocket | undefined
  let retry: number | undefined
  const params = new URLSearchParams()
  if (selection.modelId) params.set('model_id', selection.modelId)
  if (selection.iterationId) params.set('iteration_id', selection.iterationId)
  if (selection.agentId) params.set('agent_id', selection.agentId)
  const connect = () => {
    if (closed) return
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    socket = new WebSocket(`${protocol}//${location.host}/ws/${channel}${params.size ? `?${params}` : ''}`)
    socket.onopen = () => onConnection(true)
    socket.onmessage = event => { try { onData(JSON.parse(event.data) as T) } catch { /* malformed frame */ } }
    socket.onclose = () => { onConnection(false); if (!closed) retry = window.setTimeout(connect, 2000) }
    socket.onerror = () => socket?.close()
  }
  connect()
  return () => { closed = true; if (retry) clearTimeout(retry); socket?.close() }
}
