"""Spikeloom GUI API. Demo telemetry is synthetic; no training is performed here."""
from __future__ import annotations

import asyncio
import json
import math
import random
import re
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = ROOT / "data" / "graphs"
CHECKPOINT_DIR = ROOT / "data" / "checkpoints"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


class Port(BaseModel):
    id: str
    label: str
    width: int = Field(ge=1)


class GraphNode(BaseModel):
    id: str
    kind: str
    label: str
    position: dict[str, float]
    params: dict[str, Any] = Field(default_factory=dict)
    outputs: list[Port] = Field(default_factory=list)
    trainable: bool = True
    compute_gradients: bool = True
    checkpoint_id: str | None = None


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    source_handle: str = "out"
    target_handle: str = "in"
    feedback: bool = False
    delay_ticks: int | None = Field(default=None, ge=1)


class GraphDocument(BaseModel):
    schema_version: Literal[1] = 1
    id: str
    name: str
    description: str = ""
    updated_at: str | None = None
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class Command(BaseModel):
    action: Literal["start", "pause", "stop", "reset", "step"]


class SpeedRequest(BaseModel):
    ticks_per_second: int = Field(ge=1, le=240)


class ModeRequest(BaseModel):
    mode: Literal["demo", "external"]


class DemoDriver:
    """Replace this adapter with a trainer bridge; retain the public snapshot shape."""

    def __init__(self) -> None:
        self.status = "idle"
        self.ticks_per_second = 30
        self.step = 0
        self.episode = 1
        self.episode_step = 0
        self.started_at: float | None = None
        self.last_tick = time.monotonic()
        self.history: list[dict[str, Any]] = []
        self.events = [{"time": "--:--:--", "type": "info", "message": "Demo telemetry ready. Start a run to explore the workspace."}]
        self.rng = random.Random(43)
        self.position = [0.0, 0.25, 0.0]
        self.target = [2.45, 0.25, -1.65]
        self.velocity = [0.0, 0.0, 0.0]
        self.reward = 0.0

    def event(self, message: str, kind: str = "info") -> None:
        self.events.insert(0, {"time": time.strftime("%H:%M:%S"), "type": kind, "message": message})
        self.events = self.events[:30]

    @staticmethod
    def resolve_selection(selection: dict[str, str] | None) -> dict[str, str]:
        requested = selection or {}
        for model_index in range(4):
            for iteration_index in range(2):
                for agent_index in range(25):
                    candidate = {"model_id": f"demo-model-{model_index}",
                                 "iteration_id": f"demo-iteration-{model_index}-{iteration_index}",
                                 "agent_id": f"demo-agent-{model_index}-{iteration_index}-{agent_index}"}
                    if selection_matches(requested, candidate):
                        return candidate
        return {"model_id": "demo-model-0", "iteration_id": "demo-iteration-0-0", "agent_id": "demo-agent-0-0-0"}

    def command(self, action: str) -> None:
        if action == "start":
            if self.status != "running":
                self.status = "running"
                self.started_at = self.started_at or time.monotonic()
                self.last_tick = time.monotonic()
                self.event("Run started" if self.step == 0 else "Run resumed", "success")
        elif action == "pause":
            self.status = "paused"
            self.event("Run paused")
        elif action == "stop":
            self.status = "stopped"
            self.event("Run stopped", "warning")
        elif action == "reset":
            self.__init__()
            self.event("Demo state reset")
        elif action == "step":
            self.status = "paused"
            self.advance()
            self.event("Advanced one environment step")

    def advance(self) -> None:
        self.step += 1
        self.episode_step += 1
        phase = self.step * 0.035
        self.position = [2.2 * math.sin(phase * 0.52), 0.25, 1.7 * math.cos(phase * 0.38)]
        self.velocity = [0.04 * math.cos(phase * 0.52), 0, -0.03 * math.sin(phase * 0.38)]
        distance = math.dist((self.position[0], self.position[2]), (self.target[0], self.target[2]))
        self.reward = max(0.0, 1.4 - distance * 0.26) + self.rng.uniform(-0.04, 0.04)
        if self.episode_step >= 175:
            self.episode += 1
            self.episode_step = 0
            self.event(f"Episode {self.episode - 1} complete · return {self.reward * 83:.1f}", "success")
        if self.step % 5 == 0:
            progress = 1 - math.exp(-self.step / 820)
            self.history.append({
                "step": self.step,
                "reward": round(18 + 73 * progress + 9 * math.sin(self.step / 39) + self.rng.uniform(-5, 5), 2),
                "actor_loss": round(0.64 * math.exp(-self.step / 420) + 0.13 + self.rng.uniform(-0.035, 0.035), 3),
                "critic_loss": round(0.8 * math.exp(-self.step / 360) + 0.09 + self.rng.uniform(-0.04, 0.04), 3),
                "entropy": round(0.52 * math.exp(-self.step / 850) + 0.26 + self.rng.uniform(-0.025, 0.025), 3),
                "spike_rate": round(0.34 + 0.14 * math.sin(self.step / 48) + self.rng.uniform(-0.035, 0.035), 3),
            })
            self.history = self.history[-250:]

    def spike_snapshot(self, selection: dict[str, str] | None = None) -> dict[str, Any]:
        selected = self.resolve_selection(selection)
        spike_layers = []
        for layer_index, width in enumerate((1_024, 4_096, 8_192, 256)):
            bins = [round(max(0.02, min(1.0, 0.5 + 0.42 * math.sin(self.step * 0.12 + i * 1.7 + layer_index * 2.2))), 2) for i in range(16)]
            spike_layers.append({
                "name": ("Encoder", "LIF layer 1", "LIF layer 2", "Action head")[layer_index],
                "unit_count": width, "sampled_units": 64, "mean_rate": round(sum(bins) / len(bins), 3), "bins": bins,
            })
        return {"type": "spikes", "step": self.step, "layers": spike_layers, "sampling": "illustrative", **selected}

    def environment_snapshot(self, selection: dict[str, str] | None = None) -> dict[str, Any]:
        selected_ids = self.resolve_selection(selection)
        scene_agents = []
        for model_index in range(4):
            for iteration_index in range(2):
                for agent_index in range(25):
                    phase = self.step * (0.012 + model_index * 0.002) + agent_index * 0.37 + iteration_index * 1.1 + model_index * 0.63
                    radius = 1.2 + (agent_index % 7) * 0.34 + model_index * 0.48 + iteration_index * 0.21
                    scene_agents.append({
                        "id": f"demo-agent-{model_index}-{iteration_index}-{agent_index}",
                        "model_id": f"demo-model-{model_index}", "iteration_id": f"demo-iteration-{model_index}-{iteration_index}",
                        "position": [round(radius * math.sin(phase), 3), 0.25, round(radius * math.cos(phase), 3)],
                        "reward": round(max(0, 1.4 - radius * 0.2), 3),
                    })
        selected = next(item for item in scene_agents if item["id"] == selected_ids["agent_id"])
        return {"type": "environment", "step": self.step, **selected_ids,
                "environment": {"agent": selected["position"], "agents": scene_agents,
                                "target": self.target, "velocity": self.velocity, "instant_reward": round(self.reward, 3), "bounds": [-4, 4]}}

    def snapshot(self) -> dict[str, Any]:
        latest = self.history[-1] if self.history else None
        metrics = latest or {"reward": 0, "actor_loss": 0, "critic_loss": 0, "entropy": 0, "spike_rate": 0}
        models = []
        agents = []
        for model_index, name in enumerate(("Spiking actor", "Adaptive policy", "Recurrent control", "Visual encoder")):
            model_id = f"demo-model-{model_index}"
            iterations = []
            for iteration_index in range(2):
                iteration_id = f"demo-iteration-{model_index}-{iteration_index}"
                offset = model_index * 2.8 + iteration_index * 1.3
                iteration_metrics = {**metrics, "reward": round(metrics["reward"] - offset, 3)}
                iterations.append({"id": iteration_id, "name": f"Iteration {iteration_index + 1}", "status": self.status,
                                   "step": self.step, "metrics": iteration_metrics, "history": self.history[-24:], "agent_count": 25})
                for agent_index in range(25):
                    agents.append({"id": f"demo-agent-{model_index}-{iteration_index}-{agent_index}", "model_id": model_id,
                                   "iteration_id": iteration_id, "status": self.status, "episode": self.episode,
                                   "episode_step": self.episode_step, "reward": round(self.reward - offset * 0.01 + agent_index * 0.003, 3)})
            models.append({"id": model_id, "name": name, "iterations": iterations})
        return {
            "type": "telemetry", "mode": "demo", "status": self.status,
            "ticks_per_second": self.ticks_per_second, "step": self.step,
            "episode": self.episode, "episode_step": self.episode_step,
            "elapsed_seconds": int(time.monotonic() - self.started_at) if self.started_at else 0,
            "metrics": metrics,
            "history": self.history[-120:], "events": self.events[:8],
            "models": models, "agents": agents,
        }


driver = DemoDriver()
clients: set[WebSocket] = set()
detail_clients: dict[str, dict[WebSocket, dict[str, str]]] = {"environment": {}, "spikes": {}}
detail_sent: dict[WebSocket, int] = {}
detail_lock = threading.RLock()
detail_frames: dict[str, dict[tuple[str, str, str], tuple[int, dict[str, Any]]]] = {"environment": {}, "spikes": {}}
detail_sequence = 0
mode: Literal["demo", "external"] = "demo"
external_snapshot: dict[str, Any] | None = None
commands: list[dict[str, Any]] = []
command_sequence = 0


def snapshot() -> dict[str, Any]:
    if mode == "external":
        if external_snapshot is not None:
            return external_snapshot
        return {"type": "telemetry", "mode": "external", "status": "idle", "ticks_per_second": 30,
                "step": 0, "episode": 0, "episode_step": 0, "elapsed_seconds": 0,
                "metrics": {"reward": 0, "actor_loss": 0, "critic_loss": 0, "entropy": 0, "spike_rate": 0},
                "history": [], "events": [], "models": [], "agents": []}
    return driver.snapshot()


def enqueue(action: str, value: Any = None) -> None:
    global command_sequence
    command_sequence += 1
    commands.append({"sequence": command_sequence, "action": action, "value": value})
    del commands[:-128]


def selection_from_socket(websocket: WebSocket) -> dict[str, str]:
    selection = {}
    for field in ("model_id", "iteration_id", "agent_id"):
        value = websocket.query_params.get(field)
        if value:
            selection[field] = safe_id(value)
    return selection


def selection_matches(selection: dict[str, str], payload: dict[str, Any]) -> bool:
    return all(payload.get(field) == value for field, value in selection.items())


def active_subscriptions() -> dict[str, Any]:
    with detail_lock:
        result = {}
        for channel, subscribers in detail_clients.items():
            counts: dict[tuple[str, str, str], int] = {}
            for selection in subscribers.values():
                key = tuple(selection.get(field, "") for field in ("model_id", "iteration_id", "agent_id"))
                counts[key] = counts.get(key, 0) + 1
            result[channel] = {
                "viewer_count": len(subscribers),
                "requested_hz": 10 if channel == "environment" else 4,
                "selections": [{"model_id": key[0], "iteration_id": key[1], "agent_id": key[2], "viewer_count": count}
                               for key, count in counts.items()],
            }
        return result


def latest_matching_frame(channel: str, selection: dict[str, str]) -> tuple[int, dict[str, Any]] | None:
    with detail_lock:
        matching = [(version, frame) for version, frame in detail_frames[channel].values() if selection_matches(selection, frame)]
    return max(matching, key=lambda item: item[0]) if matching else None


async def broadcast() -> None:
    while True:
        now = time.monotonic()
        if mode == "demo" and driver.status == "running":
            due = min(24, int((now - driver.last_tick) * driver.ticks_per_second))
            for _ in range(due):
                driver.advance()
            if due:
                driver.last_tick += due / driver.ticks_per_second
        else:
            driver.last_tick = now
        if clients:
            payload = snapshot()
            for client in tuple(clients):
                try:
                    await client.send_json(payload)
                except Exception:
                    clients.discard(client)
        with detail_lock:
            detail_subscribers = [(channel, socket, selection.copy()) for channel, sockets in detail_clients.items()
                                  for socket, selection in sockets.items()]
        for channel, socket, selection in detail_subscribers:
            if mode == "demo":
                if channel == "spikes" and int(now * 10) % 3 != 0:
                    continue
                frame = driver.environment_snapshot(selection) if channel == "environment" else driver.spike_snapshot(selection)
                version = int(now * 10)
            else:
                latest = latest_matching_frame(channel, selection)
                if latest is None:
                    continue
                version, frame = latest
            if detail_sent.get(socket) == version:
                continue
            try:
                await socket.send_json(frame)
                detail_sent[socket] = version
            except Exception:
                with detail_lock:
                    detail_clients[channel].pop(socket, None)
                    if not detail_clients[channel]:
                        detail_frames[channel].clear()
                detail_sent.pop(socket, None)
        await asyncio.sleep(0.1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(broadcast())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Spikeloom GUI API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"ok": True, "mode": mode}


@app.get("/api/telemetry")
def telemetry():
    return snapshot()


@app.post("/api/control")
def control(command: Command):
    if mode == "external":
        enqueue(command.action)
    else:
        driver.command(command.action)
    return snapshot()


@app.put("/api/tickspeed")
def tickspeed(request: SpeedRequest):
    if mode == "external":
        enqueue("tickspeed", request.ticks_per_second)
    else:
        driver.ticks_per_second = request.ticks_per_second
        driver.event(f"Tick speed set to {request.ticks_per_second} Hz")
    return snapshot()


@app.websocket("/ws/telemetry")
async def telemetry_socket(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    await websocket.send_json(snapshot())
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients.discard(websocket)


async def detail_socket(websocket: WebSocket, channel: str) -> None:
    try:
        selection = selection_from_socket(websocket)
    except HTTPException:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    with detail_lock:
        detail_clients[channel][websocket] = selection
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        with detail_lock:
            detail_clients[channel].pop(websocket, None)
            if not detail_clients[channel]:
                detail_frames[channel].clear()
        detail_sent.pop(websocket, None)


@app.websocket("/ws/environment")
async def environment_socket(websocket: WebSocket):
    await detail_socket(websocket, "environment")


@app.websocket("/ws/spikes")
async def spikes_socket(websocket: WebSocket):
    await detail_socket(websocket, "spikes")


@app.get("/api/bridge/subscriptions")
def poll_subscriptions():
    return active_subscriptions()


@app.post("/api/bridge/mode")
def set_mode(request: ModeRequest):
    global mode, external_snapshot
    if request.mode == "external" and mode != "external":
        external_snapshot = None
        commands.clear()
        with detail_lock:
            detail_frames["environment"].clear()
            detail_frames["spikes"].clear()
    mode = request.mode
    return snapshot()


@app.get("/api/bridge/commands")
def poll_commands(after: int = 0):
    return {"sequence": command_sequence, "commands": [item for item in commands if item["sequence"] > after]}


@app.post("/api/bridge/telemetry")
def publish_external(payload: dict[str, Any]):
    global external_snapshot, mode
    required = {"status", "step", "elapsed_seconds", "ticks_per_second", "metrics", "history", "events"}
    if not required.issubset(payload):
        raise HTTPException(422, f"Missing snapshot fields: {', '.join(sorted(required - payload.keys()))}")
    if not isinstance(payload["history"], list) or not isinstance(payload["events"], list):
        raise HTTPException(422, "history and events must be arrays")
    if len(payload["history"]) > 250 or len(payload["events"]) > 30:
        raise HTTPException(413, "Telemetry exceeds bounded dashboard limits")
    models = payload.get("models", [])
    agents = payload.get("agents", [])
    if not isinstance(models, list) or not isinstance(agents, list) or len(models) > 64 or len(agents) > 200:
        raise HTTPException(413, "Model or agent summaries exceed dashboard limits")
    if any(not isinstance(model, dict) or not isinstance(model.get("iterations", []), list) or len(model.get("iterations", [])) > 64 for model in models):
        raise HTTPException(422, "Each model needs a bounded iterations array")
    if sum(len(model.get("iterations", [])) for model in models) > 256:
        raise HTTPException(413, "Iteration summaries exceed dashboard limits")
    if "environment" in payload or "neurons" in payload:
        raise HTTPException(422, "Publish environment and spikes only on opt-in detail endpoints")
    payload.setdefault("episode", 0)
    payload.setdefault("episode_step", 0)
    payload.setdefault("models", [])
    payload.setdefault("agents", [])
    payload["type"] = "telemetry"
    payload["mode"] = "external"
    try:
        body = json.dumps(payload, allow_nan=False)
    except (ValueError, TypeError):
        raise HTTPException(422, "Snapshot must contain only finite JSON values")
    if len(body.encode("utf-8")) > 128 * 1024:
        raise HTTPException(413, "Base snapshot exceeds 128 KiB")
    external_snapshot = payload
    mode = "external"
    return {"accepted": True}


def publish_detail(channel: str, payload: dict[str, Any]) -> dict[str, Any]:
    global detail_sequence
    for field in ("model_id", "iteration_id", "agent_id"):
        value = payload.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", value):
            raise HTTPException(422, f"{field} must be a safe nonempty ID")
    if not isinstance(payload.get("step"), int) or payload["step"] < 0:
        raise HTTPException(422, "step must be a nonnegative integer")
    with detail_lock:
        wanted = any(selection_matches(selection, payload) for selection in detail_clients[channel].values())
    if not wanted:
        return {"accepted": False, "reason": "no_subscribers"}
    payload["type"] = channel
    if channel == "environment":
        scene = payload.get("environment")
        if not isinstance(scene, dict) or not isinstance(scene.get("agents", []), list) or len(scene.get("agents", [])) > 200:
            raise HTTPException(413, "Environment frame needs at most 200 agents")
        if any(not isinstance(agent, dict) or not isinstance(agent.get("position"), list) or len(agent["position"]) != 3
               for agent in scene.get("agents", [])):
            raise HTTPException(422, "Each scene agent needs a 3D position")
        max_bytes = 256 * 1024
    else:
        layers = payload.get("layers")
        if not isinstance(layers, list) or len(layers) > 32 or any(not isinstance(layer, dict) or not isinstance(layer.get("bins"), list) or len(layer["bins"]) > 128 for layer in layers):
            raise HTTPException(413, "Spike frame needs at most 32 layers with 128 aggregate bins each")
        if any(any(not isinstance(value, (float, int)) or isinstance(value, bool) or not 0 <= value <= 1 for value in layer["bins"])
               for layer in layers):
            raise HTTPException(422, "Spike bins must be normalized to [0, 1]")
        max_bytes = 64 * 1024
    try:
        body = json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError):
        raise HTTPException(422, "Detail frame must contain only finite JSON values")
    if len(body.encode("utf-8")) > max_bytes:
        raise HTTPException(413, "Detail frame exceeds channel size limit")
    key = tuple(payload[field] for field in ("model_id", "iteration_id", "agent_id"))
    with detail_lock:
        detail_sequence += 1
        detail_frames[channel][key] = (detail_sequence, payload)
        if len(detail_frames[channel]) > 64:
            oldest = min(detail_frames[channel], key=lambda item: detail_frames[channel][item][0])
            del detail_frames[channel][oldest]
    return {"accepted": True}


@app.post("/api/bridge/environment")
def publish_environment(payload: dict[str, Any]):
    return publish_detail("environment", payload)


@app.post("/api/bridge/spikes")
def publish_spikes(payload: dict[str, Any]):
    return publish_detail("spikes", payload)


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", value):
        raise HTTPException(400, "ID must use letters, numbers, hyphens or underscores")
    return value


def validate_graph(graph: GraphDocument) -> list[str]:
    errors: list[str] = []
    ids = [node.id for node in graph.nodes]
    if len(ids) != len(set(ids)):
        errors.append("Node IDs must be unique")
    if len(graph.nodes) > 1000 or len(graph.edges) > 4000:
        errors.append("Graph exceeds the editor size limit")
    nodes = {node.id: node for node in graph.nodes}
    for node in graph.nodes:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", node.id):
            errors.append(f"{node.label}: invalid node ID")
        port_ids = [port.id for port in node.outputs]
        if len(port_ids) != len(set(port_ids)):
            errors.append(f"{node.label}: output port IDs must be unique")
        if any(not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", port.id) for port in node.outputs):
            errors.append(f"{node.label}: invalid output port ID")
        if node.kind == "split":
            input_width = node.params.get("input_width")
            if len(node.outputs) < 2 or not isinstance(input_width, int) or input_width < 1 or sum(port.width for port in node.outputs) != input_width:
                errors.append(f"{node.label}: split widths must sum to input width")
        if node.kind == "pretrained":
            if not node.checkpoint_id or Path(node.checkpoint_id).name != node.checkpoint_id or not (CHECKPOINT_DIR / node.checkpoint_id).is_file():
                errors.append(f"{node.label}: choose an uploaded checkpoint")
        if node.kind == "subgraph":
            graph_id = node.params.get("graph_id")
            if not isinstance(graph_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", graph_id) or not (GRAPH_DIR / f"{graph_id}.json").is_file():
                errors.append(f"{node.label}: choose a saved network")
    edge_ids = [edge.id for edge in graph.edges]
    if len(edge_ids) != len(set(edge_ids)):
        errors.append("Edge IDs must be unique")
    adjacency = {node.id: [] for node in graph.nodes}
    incoming: dict[str, list[int]] = {node.id: [] for node in graph.nodes}
    for edge in graph.edges:
        if edge.source not in nodes or edge.target not in nodes:
            errors.append(f"Edge {edge.id}: source or target is missing")
            continue
        if edge.source == edge.target and not edge.feedback:
            errors.append(f"Edge {edge.id}: a direct self connection needs delayed feedback")
        if edge.feedback and edge.delay_ticks is None:
            errors.append(f"Edge {edge.id}: feedback needs delay_ticks >= 1")
        if not edge.feedback and edge.delay_ticks is not None:
            errors.append(f"Edge {edge.id}: delay_ticks requires feedback")
        if nodes[edge.target].kind == "input":
            errors.append(f"Edge {edge.id}: input nodes cannot receive connections")
        output_ids = {port.id for port in nodes[edge.source].outputs}
        if edge.source_handle not in output_ids:
            errors.append(f"Edge {edge.id}: source port {edge.source_handle} is missing")
            continue
        source_port = next((port for port in nodes[edge.source].outputs if port.id == edge.source_handle), None)
        source_width = source_port.width if source_port else None
        if source_width is not None:
            incoming[edge.target].append(source_width)
            target = nodes[edge.target]
            expected = next((target.params.get(key) for key in ("input_width", "input_features", "input_size", "features", "neurons") if isinstance(target.params.get(key), int)), None)
            if expected and target.kind not in {"concat", "add", "subgraph", "pretrained"} and source_width != expected:
                errors.append(f"{edge.source} → {edge.target}: {source_width} features do not match expected {expected}")
        if not edge.feedback:
            adjacency[edge.source].append(edge.target)
    for node in graph.nodes:
        if node.kind == "add" and len(set(incoming[node.id])) > 1:
            errors.append(f"{node.label}: residual inputs must have equal widths")
        if node.kind == "split" and len(incoming[node.id]) != 1:
            errors.append(f"{node.label}: split needs exactly one input")
    visiting: set[str] = set()
    visited: set[str] = set()

    def cycle_from(node_id: str) -> bool:
        if node_id in visiting:
            return True
        if node_id in visited:
            return False
        visiting.add(node_id)
        if any(cycle_from(target) for target in adjacency[node_id]):
            return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    if any(cycle_from(node_id) for node_id in adjacency if node_id not in visited):
        errors.append("Instantaneous graph connections contain a cycle; mark a feedback edge with delay_ticks >= 1")
    return errors


@app.get("/api/graphs")
def list_graphs():
    result = []
    for path in sorted(GRAPH_DIR.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            result.append({"id": doc["id"], "name": doc["name"], "updated_at": doc.get("updated_at"), "node_count": len(doc.get("nodes", []))})
        except (ValueError, KeyError):
            continue
    return result


@app.get("/api/graphs/{graph_id}")
def get_graph(graph_id: str):
    path = GRAPH_DIR / f"{safe_id(graph_id)}.json"
    if not path.exists():
        raise HTTPException(404, "Graph not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/graphs/validate")
def graph_validation(graph: GraphDocument):
    errors = validate_graph(graph)
    return {"valid": not errors, "errors": errors}


@app.put("/api/graphs/{graph_id}")
def save_graph(graph_id: str, graph: GraphDocument):
    safe_id(graph_id)
    if graph.id != graph_id:
        raise HTTPException(400, "Graph ID mismatch")
    errors = validate_graph(graph)
    if errors:
        raise HTTPException(422, {"errors": errors})
    doc = graph.model_dump()
    doc["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = GRAPH_DIR / f"{graph_id}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    temporary.replace(path)
    return doc


@app.delete("/api/graphs/{graph_id}")
def delete_graph(graph_id: str):
    path = GRAPH_DIR / f"{safe_id(graph_id)}.json"
    if not path.exists():
        raise HTTPException(404, "Graph not found")
    path.unlink()
    return {"deleted": graph_id}


@app.get("/api/checkpoints")
def list_checkpoints():
    return [{"id": path.name, "size_bytes": path.stat().st_size} for path in sorted(CHECKPOINT_DIR.iterdir()) if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}]


@app.post("/api/checkpoints")
async def upload_checkpoint(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,120}\.(pt|pth|ckpt|safetensors)", filename, re.IGNORECASE):
        raise HTTPException(400, "Use a .pt, .pth, .ckpt or .safetensors checkpoint filename")
    destination = CHECKPOINT_DIR / filename
    if destination.exists():
        raise HTTPException(409, "Checkpoint filename already exists")
    with destination.open("xb") as output:
        shutil.copyfileobj(file.file, output, length=1024 * 1024)
    return {"id": filename, "size_bytes": destination.stat().st_size}
