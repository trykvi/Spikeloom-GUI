# Spikeloom GUI

A React + React Flow + FastAPI interface for designing spiking and conventional neural networks and observing continuous-control RL runs. **There is no training implementation in this repository.** Demo mode generates synthetic telemetry so every screen can be explored before connecting a trainer.

## Run locally

Open two PowerShell terminals in this directory:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```powershell
.\scripts\start-ui.ps1
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). FastAPI documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

For a new checkout, create the Python environment with `python -m venv .venv` and install `requirements.txt` with `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`. Install Node.js 20+ and run `npm install` (or `pnpm install`) before starting the UI. The UI script also detects the bundled Codex Node runtime on this computer if Node is absent from `PATH`.

## What is included

- **Dashboard:** start, pause, stop, reset, single step, and tick speed controls; model, iteration, and agent selectors; per-iteration plots; a visible aggregate spike trend and activity feed; and a scrollable 200-agent roster. The full-screen 3D arena and approximate population spike monitor each start their own stream only while open. Click the arena to capture the mouse, move with WASD, fly with Space/Q, sprint with Shift, and release the mouse with Esc.
- **Network editor:** viewport-filling React Flow canvas with mouse-wheel zoom, a searchable layer library with collapsible categories, Norse cell catalog, branching, split ports with independently editable widths, delayed feedback connections, pretrained modules, saved-network modules, per-node trainability and gradient switches, parameter inspector, undo/redo, templates, validation, JSON import/export, and server-side save/open.
- **Assets:** upload existing `.pt`, `.pth`, `.ckpt`, or `.safetensors` checkpoints. Graph JSON stores a checkpoint reference, not weight bytes.
- **Integration:** compact base telemetry for multiple models and up to 200 agents, demand-gated detail endpoints, WebSocket fan-out, command queue for a trainer adapter, and a standard-library publisher/subscriber with bounded queues and rate limits.

Saved graphs live in `data/graphs/*.json`. Checkpoint files live in `data/checkpoints/` and are intentionally ignored by Git. The graph schema is versioned with `schema_version: 1`.

## Connect a trainer

Read [docs/INTEGRATION.md](docs/INTEGRATION.md) for the snapshot schema, viewer subscription protocol, control protocol, low-overhead sampling advice, and a mapping to the read-only `C:\Users\TrymB\Documents\Projects\SNNFramework` reference. The reference project was inspected but not modified. The 3D view renders agent positions supplied by an adapter. Edited JSON graphs are architecture metadata; executing them requires a separate graph-to-PyTorch/Norse compiler in the training project.

The FastAPI service binds to `127.0.0.1` in the launch command. Keep it loopback-only unless adding authentication and access controls for external clients.
