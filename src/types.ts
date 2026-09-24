export type Port = { id: string; label: string; width: number }
export type GraphNode = {
  id: string; kind: string; label: string; position: { x: number; y: number }
  params: Record<string, string | number | boolean>; outputs: Port[]
  trainable: boolean; compute_gradients: boolean; checkpoint_id: string | null
}
export type GraphEdge = { id: string; source: string; target: string; source_handle: string; target_handle: string; feedback?: boolean; delay_ticks?: number }
export type GraphDocument = { schema_version: 1; id: string; name: string; description: string; updated_at?: string | null; nodes: GraphNode[]; edges: GraphEdge[] }
export type Checkpoint = { id: string; size_bytes: number }
export type MetricPoint = { step: number; reward: number; actor_loss: number; critic_loss: number; entropy: number; spike_rate: number }
export type IterationSummary = { id: string; name: string; status: string; step: number; metrics: Omit<MetricPoint, 'step'>; history: MetricPoint[]; agent_count: number }
export type ModelSummary = { id: string; name: string; iterations: IterationSummary[] }
export type AgentSummary = { id: string; model_id: string; iteration_id: string; status: string; episode: number; episode_step: number; reward: number }
export type EnvironmentFrame = { type: 'environment'; model_id?: string; iteration_id?: string; agent_id?: string; step: number; environment: { agent: number[]; agents?: { id: string; model_id?: string; iteration_id?: string; position: number[]; reward?: number }[]; target: number[]; velocity: number[]; instant_reward: number; bounds: number[] } }
export type SpikeFrame = { type: 'spikes'; model_id?: string; iteration_id?: string; agent_id?: string; step: number; sampling?: string; layers: { name: string; unit_count: number; sampled_units: number; mean_rate: number; bins: number[] }[] }
export type Telemetry = {
  type: 'telemetry'; mode: 'demo' | 'external'; status: 'idle' | 'running' | 'paused' | 'stopped'; ticks_per_second: number
  step: number; episode: number; episode_step: number; elapsed_seconds: number
  metrics: Omit<MetricPoint, 'step'>; history: MetricPoint[]
  events: { time: string; type: string; message: string }[]
  models?: ModelSummary[]; agents?: AgentSummary[]
  environment?: EnvironmentFrame['environment']
  neurons?: { name: string; values: number[] }[]
}

export const INITIAL_GRAPH: GraphDocument = {
  schema_version: 1, id: 'sac-spiking-actor', name: 'Spiking SAC actor', description: 'Continuous-control actor with a split action head.',
  nodes: [
    { id: 'input', kind: 'input', label: 'Observation', position: { x: 30, y: 190 }, params: { features: 8 }, outputs: [{ id: 'out', label: 'features', width: 8 }], trainable: false, compute_gradients: false, checkpoint_id: null },
    { id: 'linear1', kind: 'linear', label: 'Linear projection', position: { x: 285, y: 190 }, params: { input_features: 8, output_features: 64, bias: true }, outputs: [{ id: 'out', label: '64 channels', width: 64 }], trainable: true, compute_gradients: true, checkpoint_id: null },
    { id: 'lif1', kind: 'lif', label: 'LIF population', position: { x: 550, y: 190 }, params: { norse_module: 'norse.torch.LIFCell', neurons: 64, tau_mem_inv: 100, tau_syn_inv: 200, v_th: 0.5, v_reset: 0, v_leak: 0, dt: 0.001, method: 'super', alpha: 100 }, outputs: [{ id: 'out', label: '64 spikes', width: 64 }], trainable: true, compute_gradients: true, checkpoint_id: null },
    { id: 'linear2', kind: 'linear', label: 'Action projection', position: { x: 815, y: 190 }, params: { input_features: 64, output_features: 10, bias: true }, outputs: [{ id: 'out', label: '10 channels', width: 10 }], trainable: true, compute_gradients: true, checkpoint_id: null },
    { id: 'split', kind: 'split', label: 'Split action head', position: { x: 1080, y: 190 }, params: { input_width: 10 }, outputs: [{ id: 'mean', label: 'Mean', width: 5 }, { id: 'log_std', label: 'Log std', width: 5 }], trainable: false, compute_gradients: true, checkpoint_id: null },
    { id: 'mean', kind: 'output', label: 'Action mean', position: { x: 1375, y: 85 }, params: { features: 5 }, outputs: [], trainable: false, compute_gradients: true, checkpoint_id: null },
    { id: 'std', kind: 'output', label: 'Log standard deviation', position: { x: 1375, y: 310 }, params: { features: 5 }, outputs: [], trainable: false, compute_gradients: true, checkpoint_id: null }
  ],
  edges: [
    { id: 'e1', source: 'input', target: 'linear1', source_handle: 'out', target_handle: 'in' },
    { id: 'e2', source: 'linear1', target: 'lif1', source_handle: 'out', target_handle: 'in' },
    { id: 'e3', source: 'lif1', target: 'linear2', source_handle: 'out', target_handle: 'in' },
    { id: 'e4', source: 'linear2', target: 'split', source_handle: 'out', target_handle: 'in' },
    { id: 'e5', source: 'split', target: 'mean', source_handle: 'mean', target_handle: 'in' },
    { id: 'e6', source: 'split', target: 'std', source_handle: 'log_std', target_handle: 'in' }
  ]
}

export const LAYER_CATALOG = [
  { group: 'Essentials', items: [{ kind: 'input', label: 'Input', hint: 'Observation or tensor' }, { kind: 'output', label: 'Output', hint: 'Named network output' }, { kind: 'linear', label: 'Linear', hint: 'Dense projection' }, { kind: 'split', label: 'Split', hint: 'Route feature slices' }, { kind: 'concat', label: 'Concatenate', hint: 'Merge tensor branches' }] },
  { group: 'Norse · feedforward', items: [
    { kind: 'lif', label: 'LIFCell', hint: 'Leaky integrate and fire' },
    { kind: 'li', label: 'LICell', hint: 'Leaky integrator' },
    { kind: 'lsnn', label: 'LSNNCell', hint: 'Adaptive spiking threshold' },
    { kind: 'lif_refrac', label: 'LIFRefracCell', hint: 'Absolute refractory period' },
    { kind: 'lif_adex', label: 'LIFAdExCell', hint: 'Adaptive exponential LIF' },
    { kind: 'lif_box', label: 'LIFBoxCell', hint: 'Simplified LIF model' },
    { kind: 'iaf', label: 'IAFCell', hint: 'Integrate and fire' },
    { kind: 'izhikevich', label: 'IzhikevichCell', hint: 'Izhikevich dynamics' },
    { kind: 'coba_lif', label: 'CobaLIFCell', hint: 'Conductance based LIF' }
  ] },
  { group: 'Norse · recurrent', items: [
    { kind: 'lif_recurrent', label: 'LIFRecurrentCell', hint: 'Recurrent LIF population' },
    { kind: 'lsnn_recurrent', label: 'LSNNRecurrentCell', hint: 'Recurrent adaptive spikes' },
    { kind: 'lif_refrac_recurrent', label: 'LIFRefracRecurrentCell', hint: 'Recurrent refractory LIF' },
    { kind: 'izhikevich_recurrent', label: 'IzhikevichRecurrentCell', hint: 'Recurrent Izhikevich' }
  ] },
  { group: 'Standard layers', items: [{ kind: 'conv1d', label: 'Conv 1D', hint: 'Temporal convolution' }, { kind: 'conv2d', label: 'Conv 2D', hint: 'Spatial convolution' }, { kind: 'gru', label: 'GRU', hint: 'Recurrent sequence' }, { kind: 'lstm', label: 'LSTM', hint: 'Gated recurrent layer' }, { kind: 'attention', label: 'Attention', hint: 'Multi-head attention' }, { kind: 'embedding', label: 'Embedding', hint: 'Discrete inputs' }] },
  { group: 'Activation & utility', items: [{ kind: 'relu', label: 'ReLU', hint: 'Rectified activation' }, { kind: 'tanh', label: 'Tanh', hint: 'Bounded activation' }, { kind: 'softmax', label: 'Softmax', hint: 'Normalized scores' }, { kind: 'layernorm', label: 'LayerNorm', hint: 'Feature normalization' }, { kind: 'dropout', label: 'Dropout', hint: 'Regularization' }, { kind: 'flatten', label: 'Flatten', hint: 'Reshape tensor' }, { kind: 'add', label: 'Add', hint: 'Residual sum' }] },
  { group: 'Reusable', items: [{ kind: 'pretrained', label: 'Pretrained module', hint: 'Load a checkpoint' }, { kind: 'subgraph', label: 'Network module', hint: 'Reference a saved graph' }] }
]

export function createNode(kind: string, label: string, position: { x: number; y: number }): GraphNode {
  const id = `${kind}-${crypto.randomUUID().slice(0, 8)}`
  const norseModules: Record<string, string> = {
    lif: 'LIFCell', li: 'LICell', alif: 'LSNNCell', lsnn: 'LSNNCell', lif_refrac: 'LIFRefracCell',
    lif_adex: 'LIFAdExCell', lif_box: 'LIFBoxCell', iaf: 'IAFCell', izhikevich: 'IzhikevichCell',
    coba_lif: 'CobaLIFCell', lif_recurrent: 'LIFRecurrentCell', lsnn_recurrent: 'LSNNRecurrentCell',
    lif_refrac_recurrent: 'LIFRefracRecurrentCell', izhikevich_recurrent: 'IzhikevichRecurrentCell'
  }
  const recurrent = kind.endsWith('_recurrent')
  const norseParams: GraphNode['params'] | undefined = norseModules[kind] ? {
    norse_module: `norse.torch.${norseModules[kind]}`,
    ...(recurrent ? { input_size: 64, hidden_size: 64, autapses: false } : { neurons: 64 }),
    ...(kind !== 'li' ? { v_th: 1, v_reset: 0 } : {}),
    tau_mem_inv: 100, tau_syn_inv: 200, dt: 0.001,
    ...(kind === 'lsnn' || kind === 'alif' || kind === 'lsnn_recurrent' ? { tau_adapt_inv: 0.0012, beta: 1.8 } : {}),
    ...(kind === 'lif_refrac' || kind === 'lif_refrac_recurrent' ? { rho_reset: 5 } : {}),
    method: 'super', alpha: 100
  } : undefined
  const params: GraphNode['params'] = kind === 'split' ? { input_width: 10 } :
    kind === 'linear' ? { input_features: 64, output_features: 64, bias: true } :
    norseParams ??
    kind === 'pretrained' ? { output_features: 64, module_key: 'actor' } :
    kind === 'subgraph' ? { graph_id: '', output_features: 64 } :
    kind === 'input' || kind === 'output' ? { features: 64 } :
    kind === 'conv1d' || kind === 'conv2d' ? { in_channels: 16, out_channels: 32, kernel_size: 3, stride: 1 } :
    kind === 'gru' || kind === 'lstm' ? { input_size: 64, hidden_size: 64, layers: 1 } :
    kind === 'attention' ? { embed_dim: 64, heads: 4 } :
    kind === 'dropout' ? { probability: 0.1 } : {}
  const width = Number(params.output_features ?? params.neurons ?? params.features ?? params.out_channels ?? params.hidden_size ?? params.embed_dim ?? 64)
  return { id, kind, label, position, params, outputs: kind === 'output' ? [] : kind === 'split' ? [{ id: 'a', label: 'Part A', width: 5 }, { id: 'b', label: 'Part B', width: 5 }] : [{ id: 'out', label: `${width} features`, width }], trainable: !['input', 'output', 'split', 'concat', 'add', 'flatten'].includes(kind), compute_gradients: kind !== 'input', checkpoint_id: null }
}
