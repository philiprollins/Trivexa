"""A single-GPU job queue: jobs run one at a time in a worker thread, with progress pushed to websocket clients."""

import asyncio
import logging
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import WebSocket

from .engines import Cancelled

log = logging.getLogger(__name__)

MAX_FINISHED = 100


@dataclass
class Job:
    kind: str
    params: dict
    run: Callable[["Job"], dict]  # executed on the worker thread; returns the saved asset
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "queued"  # queued | loading | running | done | error | cancelled
    step: int = 0
    total: int = 0
    preview: str | None = None
    asset: dict | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    cancel_requested: threading.Event = field(default_factory=threading.Event)

    def public(self, with_preview: bool = True) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "params": self.params,
            "status": self.status,
            "step": self.step,
            "total": self.total,
            "preview": self.preview if with_preview else None,
            "asset": self.asset,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class JobManager:
    def __init__(self):
        self.jobs: OrderedDict[str, Job] = OrderedDict()
        self.clients: set[WebSocket] = set()
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._worker: asyncio.Task | None = None
        self._gpu_lock = threading.Lock()

    @property
    def gpu_lock(self) -> threading.Lock:
        """Held while a job runs; anything else touching the GPU (e.g. unloading models) should take it."""
        return self._gpu_lock

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._worker = asyncio.create_task(self._work())

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()

    def submit(self, job: Job) -> Job:
        self.jobs[job.id] = job
        self._queue.put_nowait(job)
        self._emit("job", job.public())
        return job

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.status in ("done", "error", "cancelled"):
            return False
        job.cancel_requested.set()
        if job.status == "queued":
            self._finish(job, "cancelled")
        return True

    def list(self) -> list[dict]:
        return [j.public(with_preview=False) for j in self.jobs.values()]

    @property
    def pending(self) -> int:
        return sum(1 for j in self.jobs.values() if j.status in ("queued", "loading", "running"))

    # -- worker -------------------------------------------------------------

    async def _work(self) -> None:
        while True:
            job = await self._queue.get()
            if job.status != "queued":
                continue
            job.started_at = time.time()
            self._update(job, status="loading")
            try:
                asset = await asyncio.to_thread(self._run_locked, job)
            except Exception as e:  # noqa: BLE001 - surfaced to the UI
                if isinstance(e, Cancelled) or job.cancel_requested.is_set():
                    self._finish(job, "cancelled")
                else:
                    log.error("Job %s failed:\n%s", job.id, traceback.format_exc())
                    job.error = str(e) or type(e).__name__
                    self._finish(job, "error")
            else:
                job.asset = asset
                self._finish(job, "done")

    def _run_locked(self, job: Job) -> dict:
        with self._gpu_lock:
            return job.run(job)

    # -- called from the worker thread ---------------------------------------

    def report_step(self, job: Job, step: int, total: int, preview: str | None) -> None:
        job.status = "running"
        job.step, job.total, job.preview = step, total, preview
        self._emit_threadsafe("progress", {"id": job.id, "status": job.status, "step": step, "total": total, "preview": preview})

    # -- events ---------------------------------------------------------------

    def _update(self, job: Job, **fields) -> None:
        for k, v in fields.items():
            setattr(job, k, v)
        self._emit("job", job.public())

    def _finish(self, job: Job, status: str) -> None:
        job.status = status
        job.finished_at = time.time()
        job.preview = None
        self._emit("job", job.public())
        finished = [j for j in self.jobs.values() if j.finished_at]
        for old in finished[:-MAX_FINISHED]:
            self.jobs.pop(old.id, None)

    def _emit_threadsafe(self, event: str, data: dict) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(self._emit, event, data)

    def _emit(self, event: str, data: dict) -> None:
        msg = {"type": event, "data": data}
        for ws in list(self.clients):
            asyncio.create_task(self._send(ws, msg))

    async def _send(self, ws: WebSocket, msg: dict) -> None:
        try:
            await ws.send_json(msg)
        except Exception:  # noqa: BLE001 - client went away
            self.clients.discard(ws)
