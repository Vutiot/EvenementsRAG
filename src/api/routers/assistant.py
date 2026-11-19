"""Assistant endpoints (E8/E9): dataset profile and the eval-dataset interview."""

import asyncio
import queue
import threading

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.assistant.dataset_profiler import DatasetProfile, get_profile
from src.assistant.eval_chat import EvalChatService

router = APIRouter()

_chat_service = EvalChatService()


@router.get("/assistant/profile/{collection}", response_model=DatasetProfile)
async def get_collection_profile(
    collection: str,
    refresh: bool = Query(False, description="Recompute instead of using the cached profile"),
):
    """Compact, LLM-ready profile of a collection (cached on disk)."""
    try:
        return await asyncio.to_thread(get_profile, collection, refresh)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")


class EvalChatRequest(BaseModel):
    message: str = ""
    session_id: str | None = None
    collection: str | None = None  # required when session_id is empty


@router.post("/assistant/eval/chat")
async def eval_chat(request: EvalChatRequest):
    """One interview turn, streamed as SSE.

    Events: ``session`` {session_id}, ``warning``, ``token`` {token},
    ``plan_update`` {draft, missing, complete, quick_replies},
    ``done`` {message, cost_summary}, ``error`` {message}.
    Start a session by sending ``collection`` and an empty ``message``.
    """
    q: queue.Queue[str | None] = queue.Queue()

    def _worker():
        try:
            for event in _chat_service.chat(
                request.message, session_id=request.session_id, collection=request.collection
            ):
                q.put(event)
        finally:
            q.put(None)

    threading.Thread(target=_worker, daemon=True).start()

    async def _stream():
        while True:
            item = await asyncio.to_thread(q.get)
            if item is None:
                break
            yield item

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/assistant/eval/sessions/{session_id}")
def get_eval_session(session_id: str):
    """Messages, draft plan, missing fields and cost of a session."""
    session = _chat_service.store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session")
    return session.state()


@router.delete("/assistant/eval/sessions/{session_id}")
def delete_eval_session(session_id: str):
    if not _chat_service.store.delete(session_id):
        raise HTTPException(status_code=404, detail="Unknown or expired session")
    return {"deleted": session_id}
