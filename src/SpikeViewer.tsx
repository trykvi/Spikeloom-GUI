import { useEffect, useState } from 'react'
import { Activity, X, Zap } from 'lucide-react'
import { connectDetail } from './api'
import type { SpikeFrame } from './types'

type Props = { selection: { modelId?: string; iterationId?: string; agentId?: string }; onClose: () => void }
export function SpikeViewer({ selection, onClose }: Props) {
  const [frame, setFrame] = useState<SpikeFrame | null>(null)
  const [connected, setConnected] = useState(false)
  const [selected, setSelected] = useState(0)
  const [visible, setVisible] = useState(document.visibilityState === 'visible')
  useEffect(() => { const update = () => setVisible(document.visibilityState === 'visible'); document.addEventListener('visibilitychange', update); return () => document.removeEventListener('visibilitychange', update) }, [])
  useEffect(() => { if (!visible) { setConnected(false); return }; return connectDetail<SpikeFrame>('spikes', selection, setFrame, setConnected) }, [selection.modelId, selection.iterationId, selection.agentId, visible])
  const layer = frame?.layers[selected] ?? frame?.layers[0]
  return <div className="viewer-overlay spike-overlay" role="dialog" aria-modal="true" aria-label="Spike monitor"><div className="spike-window"><div className="spike-top"><div><span className="eyebrow">NEURAL ACTIVITY · ON DEMAND</span><h2>Spike monitor</h2><p>Bounded population samples for large networks</p></div><button className="viewer-close" onClick={onClose} aria-label="Close spike monitor"><X size={21}/></button></div><div className="spike-body"><aside className="spike-layers"><span className="eyebrow">LAYERS</span>{frame?.layers.map((item, index) => <button key={item.name} className={selected === index ? 'active' : ''} onClick={() => setSelected(index)}><Zap size={15}/><span>{item.name}<small>{item.unit_count.toLocaleString()} units</small></span><b>{(item.mean_rate * 100).toFixed(1)}%</b></button>)}</aside><div className="spike-detail"><div className="spike-detail-head"><div><span className="eyebrow">POPULATION ACTIVITY</span><h3>{layer?.name ?? 'Waiting for samples'}</h3></div><span className={`stream-state ${connected ? 'connected' : ''}`}><i/>{connected ? 'Sampling' : 'Connecting…'}</span></div><div className="spike-metrics"><div><Activity size={18}/><span>Mean firing rate</span><strong>{layer ? `${(layer.mean_rate * 100).toFixed(1)}%` : '—'}</strong></div><div><Zap size={18}/><span>Sampled units</span><strong>{layer?.sampled_units.toLocaleString() ?? '—'}</strong></div><div><span>Total units</span><strong>{layer?.unit_count.toLocaleString() ?? '—'}</strong></div></div><div className="spike-heatmap">{layer?.bins.map((value, index) => <div key={index} title={`Population bin ${index + 1}: ${(value * 100).toFixed(1)}%`} style={{ background: `rgba(166, 132, 255, ${.12 + Math.min(1, Math.max(0, value)) * .88})` }}/>)}</div><div className="spike-foot"><span>Aggregated spatial bins</span><span>Step {frame?.step.toLocaleString() ?? '—'}</span></div><p className="spike-note">This view displays a sampled approximation. It does not stream every neuron or parameter. Closing the monitor ends its subscription.</p></div></div></div></div>
}
