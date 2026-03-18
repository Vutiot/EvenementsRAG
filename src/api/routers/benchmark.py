"""Benchmark run SSE endpoint."""

import asyncio
import queue
import threading

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.api.benchmark_service import BenchmarkService
from src.api.schemas import BenchmarkRunRequest

router = APIRouter()

_service = BenchmarkService()


@router.post("/benchmark/run")
async def run_benchmark(request: BenchmarkRunRequest):
    """Start a full benchmark run, streaming progress via SSE."""
    q: queue.Queue[str | None] = queue.Queue()

    def _worker():
        try:
            for event in _service.run_benchmark(request):
                q.put(event)
        except Exception as exc:
            import json
            q.put(f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n")
        finally:
            q.put(None)  # sentinel

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    async def _stream():
        while True:
            try:
                item = await asyncio.to_thread(q.get, timeout=300)
            except Exception:
                break
            if item is None:
                break
            yield item

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
