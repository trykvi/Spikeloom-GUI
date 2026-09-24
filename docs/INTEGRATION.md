# Connecting a trainer to Spikeloom GUI

Spikeloom GUI is a viewer and control bridge. It has no training code. The reference framework at `C:\Users\TrymB\Documents\Projects\SNNFramework` is read only; integrate from a separate adapter or your own copy.

## Base telemetry

Set external mode with `POST /api/bridge/mode` and `{"mode":"external"}`. Publish a complete, bounded snapshot to `POST /api/bridge/telemetry` at 2–10 Hz. `GET /api/telemetry` and `/ws/telemetry` expose its latest value. The base stream carries scalar summaries; **it must not contain environment positions or neuron arrays**.

```json
{
  "status": "running", "ticks_per_second": 30, "step": 1250,
  "episode": 8, "episode_step": 74, "elapsed_seconds": 91,
  "metrics": {"reward": 0.82, "actor_loss": 0.14, "critic_loss": 0.26, "entropy": 0.31, "spike_rate": 0.19},
  "history": [{"step": 1250, "reward": 0.82}],
  "events": [{"time": "12:34:56", "type": "info", "message": "Checkpoint saved"}],
  "models": [{"id": "actor-a", "name": "Actor A", "iterations": [
    {"id": "actor-a-v1", "name": "Iteration 1", "status": "running", "step": 1250,
     "metrics": {"reward": 0.82}, "history": [{"step": 1250, "reward": 0.82}], "agent_count": 100}
  ]}],
  "agents": [{"id": "agent-001", "model_id": "actor-a", "iteration_id": "actor-a-v1",
              "status": "running", "episode": 8, "episode_step": 74, "reward": 0.76}]
}
```

The server adds `type:"telemetry"` and `mode:"external"`. Limits: 64 models, 256 total iterations, 200 agents, 250 global chart points, 30 events, and 128 KiB per base snapshot. Keep each iteration's chart history short. All numbers must be finite. IDs should be stable and use letters, numbers, underscores, or hyphens. Root metrics are aggregate run metrics; iteration metrics let the dashboard compare models.

## Viewer demand and detail channels

Opening `/ws/environment?model_id=actor-a&iteration_id=actor-a-v1&agent_id=agent-001` subscribes that viewer until its WebSocket closes. `/ws/spikes` with the same optional selectors independently enables spike sampling. An omitted selector matches any ID. The UI closes these sockets when the corresponding viewer closes, so overnight training keeps only base telemetry.

An adapter polls `GET /api/bridge/subscriptions`, which has `environment` and `spikes` entries. Each contains `viewer_count`, `requested_hz` (10 and 4 respectively), and `selections` with IDs and per-selection viewer counts. Construct detail frames **only when a matching selection is active**. A detail POST with no matching viewer returns `{"accepted":false,"reason":"no_subscribers"}` and stores nothing. The helper below expires demand after three seconds without a successful poll. The server retains only 64 recent frames per channel.

Environment frame to `POST /api/bridge/environment` (at most 256 KiB and 200 agents):

```json
{
  "model_id": "actor-a", "iteration_id": "actor-a-v1", "agent_id": "agent-001", "step": 1250,
  "environment": {
    "agent": [0.1, 0.25, -0.2], "target": [0.3, 0.25, 0.1],
    "velocity": [0.0, 0.0, 0.02], "instant_reward": 0.76, "bounds": [-4, 4],
    "agents": [{"id": "agent-001", "model_id": "actor-a", "iteration_id": "actor-a-v1",
                "position": [0.1, 0.25, -0.2], "reward": 0.76}]
  }
}
```

Coordinates are `[x, height, z]`. Map planar simulation `(x, y)` to `(x, height, z=y)` if needed. The scene's `agents` can contain all 200 agents across models and iterations. Top-level IDs and `environment.agent` identify the selected agent. Keep geometry compact; avoid video frames.

Spike frame to `POST /api/bridge/spikes` (at most 64 KiB, 32 layers, and 128 aggregate bins per layer):

```json
{
  "model_id": "actor-a", "iteration_id": "actor-a-v1", "agent_id": "agent-001", "step": 1250,
  "sampling": "reservoir-64",
  "layers": [{"name": "Actor LIF 1", "unit_count": 1048576, "sampled_units": 64,
              "mean_rate": 0.19, "bins": [0.12, 0.28, 0.16, 0.20]}]
}
```

Bins are normalized aggregate activity, not one value per neuron. For large networks, maintain a stable small sample or running aggregate and publish at 4 Hz or less. Avoid synchronizing CUDA tensors solely for UI data on every optimizer step. Neither endpoint computes activations itself.

## Bridge helpers and controls

`backend/bridge_client.py` uses only the Python standard library. Create helpers inside the adapter process after spawning workers:

```python
from backend.bridge_client import TelemetryPublisher, CommandSubscriber, SubscriptionSubscriber

base = TelemetryPublisher(max_hz=10)
scene = TelemetryPublisher(endpoint="http://127.0.0.1:8000/api/bridge/environment", max_hz=10, max_payload_bytes=256*1024)
spikes = TelemetryPublisher(endpoint="http://127.0.0.1:8000/api/bridge/spikes", max_hz=4, max_payload_bytes=64*1024)
controls = CommandSubscriber()
demand = SubscriptionSubscriber()

# At an existing safe loop boundary:
for command in controls.drain():
    handle_control_at_safe_boundary(command.action, command.value)
base.publish(make_detached_base_snapshot())

# At separate detail sample clocks:
if demand.wants("environment", model_id="actor-a", iteration_id="actor-a-v1", agent_id="agent-001"):
    scene.publish(make_detached_scene_frame())
if demand.wants("spikes", model_id="actor-a", iteration_id="actor-a-v1", agent_id="agent-001"):
    spikes.publish(make_detached_aggregate_spike_frame())
```

These functions are application-specific placeholders. `publish` replaces one pending slot; JSON encoding and HTTP happen on its daemon thread. Do not mutate a published tree. Construct expensive samples **after** the demand check. If the server is down, demand expires and network failure never blocks training. For many viewers, rotate among requested selections or use separate bounded publishers.

The dashboard sends `POST /api/control` with `{"action":"start"|"pause"|"stop"|"reset"|"step"}` and `PUT /api/tickspeed` with `{"ticks_per_second":30}`. In external mode these become queued commands. Poll `GET /api/bridge/commands?after=N` or use `CommandSubscriber`; apply controls at a safe trainer boundary, then reflect actual state in the next base snapshot.

## Reference framework notes

The read-only framework uses `AsyncTrainingConfig`, `ModelSpec`, and multiprocessing workers in `src/training/async_training.py` and `src/workers.py`. Environment workers own `environment.step()`; trainer workers call `trainer.update(batch)` and receive losses. In a separate integration copy, aggregate these in one sidecar with fixed-size summaries. Avoid sending UI frames or activation arrays through training channels every step.

`src/agents/test_agent.py` builds a spiking actor with two `LIFCell` layers and a final `LICell`, plus a nonspiking twin-Q critic. The actor exposes no telemetry hook, so an adapter needs an optional read-only observation point for bounded spike summaries. `BaseAgent.save()` writes actor and critic state dictionaries to a `.pt` checkpoint. Graph `.json` files describe architecture, delayed feedback edges, and per-node `trainable` / `compute_gradients` flags; checkpoint files hold weights. The UI graph does not compile into runnable PyTorch/Norse modules. A future compiler must validate shapes, recurrence delays, and checkpoint keys.
