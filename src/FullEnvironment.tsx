import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { useEffect, useRef, useState } from 'react'
import { Color, InstancedMesh, Object3D, Vector3 } from 'three'
import { Crosshair, Maximize2, X } from 'lucide-react'
import { connectDetail } from './api'
import type { EnvironmentFrame } from './types'

type Selection = { modelId?: string; iterationId?: string; agentId?: string }
type Props = { selection: Selection; onClose: () => void }
const colors = ['#a994ff', '#69d7ce', '#f2aa91', '#f1d077', '#72dcae', '#e595cd']
function AgentInstances({ frame, selectedId }: { frame: EnvironmentFrame | null; selectedId?: string }) {
  const ref = useRef<InstancedMesh>(null)
  const agents = frame?.environment.agents ?? []
  useEffect(() => {
    if (!ref.current) return
    const dummy = new Object3D()
    agents.forEach((agent, index) => {
      const [x, y, z] = agent.position
      dummy.position.set(x ?? 0, Math.max(.22, y ?? 0), z ?? 0)
      dummy.scale.setScalar(agent.id === selectedId ? 1.7 : 1)
      dummy.updateMatrix()
      ref.current!.setMatrixAt(index, dummy.matrix)
      ref.current!.setColorAt(index, new Color(agent.id === selectedId ? '#ffffff' : colors[Math.abs(hash(agent.model_id ?? agent.id)) % colors.length]))
    })
    ref.current.count = agents.length
    ref.current.instanceMatrix.needsUpdate = true
    if (ref.current.instanceColor) ref.current.instanceColor.needsUpdate = true
  }, [agents, selectedId])
  return <instancedMesh ref={ref} args={[undefined, undefined, 200]} count={agents.length}><sphereGeometry args={[.19, 9, 7]}/><meshStandardMaterial roughness={.35} metalness={.15}/></instancedMesh>
}
function hash(value: string) { let result = 0; for (let i = 0; i < value.length; i++) result = (result * 31 + value.charCodeAt(i)) | 0; return result }

function FlyControls() {
  const { camera, gl } = useThree()
  const keys = useRef(new Set<string>())
  const look = useRef({ yaw: 0, pitch: -.35 })
  useEffect(() => {
    camera.position.set(0, 8, 14)
    camera.rotation.order = 'YXZ'
    camera.rotation.set(look.current.pitch, look.current.yaw, 0)
    const down = (event: KeyboardEvent) => { keys.current.add(event.code); if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.code)) event.preventDefault() }
    const up = (event: KeyboardEvent) => keys.current.delete(event.code)
    const move = (event: MouseEvent) => {
      if (document.pointerLockElement !== gl.domElement) return
      look.current.yaw -= event.movementX * .002
      look.current.pitch = Math.max(-1.52, Math.min(1.52, look.current.pitch - event.movementY * .002))
      camera.rotation.set(look.current.pitch, look.current.yaw, 0)
    }
    const lock = () => { if (document.pointerLockElement !== gl.domElement) keys.current.clear() }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    document.addEventListener('mousemove', move)
    document.addEventListener('pointerlockchange', lock)
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', up); document.removeEventListener('mousemove', move); document.removeEventListener('pointerlockchange', lock); if (document.pointerLockElement === gl.domElement) document.exitPointerLock() }
  }, [camera, gl])
  useFrame((_, delta) => {
    if (document.pointerLockElement !== gl.domElement) return
    const pressed = keys.current
    const direction = new Vector3(Number(pressed.has('KeyD')) - Number(pressed.has('KeyA')), Number(pressed.has('Space') || pressed.has('KeyE')) - Number(pressed.has('KeyQ') || pressed.has('ControlLeft')), Number(pressed.has('KeyS')) - Number(pressed.has('KeyW')))
    if (!direction.lengthSq()) return
    direction.normalize()
    const forward = new Vector3(Math.sin(look.current.yaw), 0, Math.cos(look.current.yaw))
    const right = new Vector3(Math.cos(look.current.yaw), 0, -Math.sin(look.current.yaw))
    const speed = pressed.has('ShiftLeft') ? 16 : 7
    const movement = new Vector3().addScaledVector(right, direction.x).addScaledVector(forward, direction.z).add(new Vector3(0, direction.y, 0)).multiplyScalar(Math.min(delta, .1) * speed)
    camera.position.add(movement)
  })
  return null
}

export function FullEnvironment({ selection, onClose }: Props) {
  const [frame, setFrame] = useState<EnvironmentFrame | null>(null)
  const [connected, setConnected] = useState(false)
  const [visible, setVisible] = useState(document.visibilityState === 'visible')
  useEffect(() => { const update = () => setVisible(document.visibilityState === 'visible'); document.addEventListener('visibilitychange', update); return () => document.removeEventListener('visibilitychange', update) }, [])
  useEffect(() => { if (!visible) { setConnected(false); return }; return connectDetail<EnvironmentFrame>('environment', selection, setFrame, setConnected) }, [selection.modelId, selection.iterationId, selection.agentId, visible])
  const target = frame?.environment.target ?? [0, .25, 0]
  const bounds = frame?.environment.bounds ?? [-8, 8]
  const span = Math.max(8, Math.abs((bounds[1] ?? 8) - (bounds[0] ?? -8)))
  return <div className="viewer-overlay environment-overlay" role="dialog" aria-modal="true" aria-label="Live 3D environment">
    <Canvas frameloop={visible ? 'always' : 'never'} camera={{ fov: 65, near: .01, far: 500 }} onCreated={({ gl }) => { gl.domElement.addEventListener('click', () => { if (!document.pointerLockElement) void gl.domElement.requestPointerLock() }) }}><color attach="background" args={['#0a1426']}/><fog attach="fog" args={['#0a1426', 32, 100]}/><ambientLight intensity={1.5}/><directionalLight position={[8, 13, 4]} intensity={2}/><gridHelper args={[span * 3, Math.round(span * 3), '#34445d', '#26334b']}/><mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -.035, 0]}><planeGeometry args={[span * 3, span * 3]}/><meshStandardMaterial color="#101d30" roughness={1}/></mesh><AgentInstances frame={frame} selectedId={selection.agentId}/><mesh position={new Vector3(target[0] ?? 0, .35, target[2] ?? 0)}><torusGeometry args={[.42, .07, 8, 24]}/><meshStandardMaterial color="#6ee1bd" emissive="#3bb28d" emissiveIntensity={.65}/></mesh><FlyControls/></Canvas>
    <div className="viewer-top"><div><span className="eyebrow">LIVE ENVIRONMENT · ON DEMAND</span><h2>Shared 3D arena</h2><p>{frame?.environment.agents?.length ?? 0} agents visible · step {frame?.step.toLocaleString() ?? '—'}</p></div><button className="viewer-close" onClick={onClose} aria-label="Close environment"><X size={21}/></button></div>
    <div className="viewer-bottom"><span className={`stream-state ${connected ? 'connected' : ''}`}><i/>{connected ? 'Detail stream active' : 'Connecting…'}</span><span><Crosshair size={15}/> Click scene for mouse look · WASD move · Space / Q fly · Shift sprint · Esc release</span><span><Maximize2 size={15}/> Full viewport</span></div>
  </div>
}
