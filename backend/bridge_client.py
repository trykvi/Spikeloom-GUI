"""Best-effort telemetry publisher for an external training process.

This module contains no trainer code and has no third-party dependencies. Create
one publisher per process, after multiprocessing workers have started. The
producer hands it a fresh JSON-compatible snapshot; a daemon thread serializes
and sends only the newest pending snapshot. Network failure never blocks a
training step.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class PublisherStats:
    accepted: int
    sent: int
    superseded: int
    rejected: int
    failed: int
    last_error: str | None


class TelemetryPublisher:
    """Send complete UI snapshots over HTTP with latest-wins backpressure.

    ``publish`` performs no JSON encoding, tensor transfer, or network I/O.
    Call it with a freshly built tree of ordinary Python scalars, lists and
    dictionaries. Do not mutate that tree after handing it to the publisher.
    A single pending slot bounds memory use when the UI is slow or offline.
    This is telemetry: dropped samples are intentional and not replayed.
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8000/api/bridge/telemetry",
        *,
        max_hz: float = 10.0,
        timeout_seconds: float = 0.25,
        max_payload_bytes: int = 128 * 1024,
    ) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must be an HTTP(S) URL")
        if max_hz <= 0 or timeout_seconds <= 0 or max_payload_bytes <= 0:
            raise ValueError("max_hz, timeout_seconds and max_payload_bytes must be positive")
        self.endpoint = endpoint
        self.max_hz = max_hz
        self.timeout_seconds = timeout_seconds
        self.max_payload_bytes = max_payload_bytes
        self._condition = threading.Condition()
        self._pending: Mapping[str, Any] | None = None
        self._closed = False
        self._accepted = 0
        self._sent = 0
        self._superseded = 0
        self._rejected = 0
        self._failed = 0
        self._last_error: str | None = None
        self._thread = threading.Thread(target=self._run, name="telemetry-publisher", daemon=True)
        self._thread.start()

    def publish(self, snapshot: Mapping[str, Any]) -> bool:
        """Queue a snapshot without waiting for the server; return False if closed."""
        with self._condition:
            if self._closed:
                return False
            if self._pending is not None:
                self._superseded += 1
            self._pending = snapshot
            self._accepted += 1
            self._condition.notify()
            return True

    def stats(self) -> PublisherStats:
        with self._condition:
            return PublisherStats(
                accepted=self._accepted,
                sent=self._sent,
                superseded=self._superseded,
                rejected=self._rejected,
                failed=self._failed,
                last_error=self._last_error,
            )

    def close(self, *, wait_seconds: float = 0.0) -> None:
        """Discard pending data and stop. An in-flight request may finish."""
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify()
        if wait_seconds > 0:
            self._thread.join(timeout=wait_seconds)

    def __enter__(self) -> TelemetryPublisher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _run(self) -> None:
        interval = 1.0 / self.max_hz
        next_send = 0.0
        while True:
            with self._condition:
                while not self._closed:
                    delay = next_send - time.monotonic()
                    if self._pending is not None and delay <= 0:
                        snapshot = self._pending
                        self._pending = None
                        break
                    self._condition.wait(timeout=max(0.0, delay) if self._pending is not None else None)
                else:
                    return

            next_send = time.monotonic() + interval
            try:
                body = json.dumps(snapshot, separators=(",", ":"), allow_nan=False).encode("utf-8")
                if len(body) > self.max_payload_bytes:
                    with self._condition:
                        self._rejected += 1
                        self._last_error = f"telemetry payload exceeds {self.max_payload_bytes} bytes"
                    continue
                request = Request(
                    self.endpoint,
                    data=body,
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    response.read(1)
                with self._condition:
                    self._sent += 1
                    self._last_error = None
            except (TypeError, ValueError) as exc:
                with self._condition:
                    self._rejected += 1
                    self._last_error = f"invalid telemetry: {exc}"
            except Exception as exc:  # HTTP, connection and timeout errors must never reach training.
                with self._condition:
                    self._failed += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class ControlCommand:
    sequence: int
    action: str
    value: int | float | str | None = None


class CommandSubscriber:
    """Fetch UI commands in a daemon thread; apply them at safe trainer boundaries.

    The internal queue is bounded and preserves order. If it fills, polling
    pauses until ``drain`` is called instead of silently losing stop commands.
    Network errors are retried at the configured interval.
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8000/api/bridge/commands",
        *,
        poll_seconds: float = 0.25,
        timeout_seconds: float = 0.25,
        max_pending: int = 64,
    ) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must be an HTTP(S) URL")
        if poll_seconds <= 0 or timeout_seconds <= 0 or max_pending <= 0:
            raise ValueError("poll_seconds, timeout_seconds and max_pending must be positive")
        self.endpoint = endpoint
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.max_pending = max_pending
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = False
        self._after = 0
        self._pending: deque[ControlCommand] = deque()
        self._thread = threading.Thread(target=self._run, name="control-subscriber", daemon=True)
        self._thread.start()

    def drain(self) -> list[ControlCommand]:
        """Return queued commands in sequence order; never performs network I/O."""
        with self._lock:
            commands = list(self._pending)
            self._pending.clear()
        if commands:
            self._wake.set()
        return commands

    def close(self, *, wait_seconds: float = 0.0) -> None:
        with self._lock:
            self._closed = True
        self._wake.set()
        if wait_seconds > 0:
            self._thread.join(timeout=wait_seconds)

    def _run(self) -> None:
        while True:
            with self._lock:
                if self._closed:
                    return
                free = self.max_pending - len(self._pending)
                after = self._after
            if free:
                try:
                    separator = "&" if "?" in self.endpoint else "?"
                    url = self.endpoint + separator + urlencode({"after": after})
                    with urlopen(url, timeout=self.timeout_seconds) as response:
                        payload = json.load(response)
                    commands = payload.get("commands", [])
                    if isinstance(commands, list):
                        with self._lock:
                            for item in commands[:free]:
                                if not isinstance(item, dict):
                                    continue
                                sequence = item.get("sequence")
                                action = item.get("action")
                                if not isinstance(sequence, int) or sequence <= self._after or not isinstance(action, str):
                                    continue
                                self._pending.append(ControlCommand(sequence, action, item.get("value")))
                                self._after = sequence
                except Exception:
                    # Controls are best-effort transport. The server retains the
                    # sequence and the next poll starts at the last accepted one.
                    pass
            self._wake.wait(self.poll_seconds)
            self._wake.clear()


class SubscriptionSubscriber:
    """Observe viewer demand off the training thread.

    Build scene positions or spike summaries only when ``wants`` is true.
    A stale or unavailable server means no detail sampling, avoiding surprise
    overhead during unattended runs. Base scalar telemetry is independent.
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8000/api/bridge/subscriptions",
        *,
        poll_seconds: float = 1.0,
        timeout_seconds: float = 0.25,
        stale_seconds: float = 3.0,
    ) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must be an HTTP(S) URL")
        if min(poll_seconds, timeout_seconds, stale_seconds) <= 0:
            raise ValueError("poll_seconds, timeout_seconds and stale_seconds must be positive")
        self.endpoint = endpoint
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.stale_seconds = stale_seconds
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._demand: dict[str, Any] = {}
        self._received_at = 0.0
        self._thread = threading.Thread(target=self._run, name="viewer-demand-subscriber", daemon=True)
        self._thread.start()

    def selections(self, channel: str) -> list[dict[str, str]]:
        """Return active selections; an empty list means do not sample."""
        if channel not in {"environment", "spikes"}:
            raise ValueError("channel must be environment or spikes")
        with self._lock:
            if time.monotonic() - self._received_at > self.stale_seconds:
                return []
            channel_demand = self._demand.get(channel, {})
            raw = channel_demand.get("selections", []) if isinstance(channel_demand, dict) else []
            return [{key: value for key, value in item.items() if key in {"model_id", "iteration_id", "agent_id"} and isinstance(value, str) and value}
                    for item in raw if isinstance(item, dict) and item.get("viewer_count", 0) > 0]

    def wants(self, channel: str, *, model_id: str, iteration_id: str, agent_id: str) -> bool:
        """Cheap local check before materializing an optional detail frame."""
        identifiers = {"model_id": model_id, "iteration_id": iteration_id, "agent_id": agent_id}
        return any(all(identifiers[key] == expected for key, expected in selection.items())
                   for selection in self.selections(channel))

    def close(self, *, wait_seconds: float = 0.0) -> None:
        self._closed.set()
        if wait_seconds > 0:
            self._thread.join(timeout=wait_seconds)

    def __enter__(self) -> SubscriptionSubscriber:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                with urlopen(self.endpoint, timeout=self.timeout_seconds) as response:
                    demand = json.load(response)
                if isinstance(demand, dict):
                    with self._lock:
                        self._demand = demand
                        self._received_at = time.monotonic()
            except Exception:
                # The last demand expires; no training step waits for HTTP.
                pass
            self._closed.wait(self.poll_seconds)
