"""Eval-dataset assistant: guided interview backend (E8-F2-T1).

The assistant interviews the user about the eval dataset they want (question
types, personas of the askers, themes and periods, difficulty mix, answer
language, volume), grounded on the collection's ``DatasetProfile``. It never
generates the dataset itself: it only maintains a *draft plan* that the
structured-output step (E8-F3-T1) turns into the DatasetManager form.

Each assistant turn is streamed as text, followed by a hidden machine-readable
block::

    <draft>{"question_types": [...], ..., "quick_replies": ["...", "..."]}</draft>

The block is stripped from the streamed text, parsed, merged into the
session's draft and emitted as a ``plan_update`` event. Completeness is
decided by the server from the draft fields, not by the model.

Sessions live in memory (lost on restart), which is fine for one local user.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, Generator, List, Optional

from src.llm.usage import UsageTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_SESSIONS = 50
MAX_HISTORY_MESSAGES = 40  # user + assistant messages kept in the prompt

KNOWN_QUESTION_TYPES = [
    "factual",
    "temporal",
    "comparative",
    "entity_centric",
    "relationship",
    "analytical",
    "keywords",
    "telegraphic",
    "grammar_typos",
]

# Draft fields the interview must settle before the plan is complete
REQUIRED_FIELDS = (
    "question_types",
    "personas",
    "themes_include",
    "difficulty_mix",
    "language",
    "total_questions",
)
LIST_FIELDS = ("question_types", "personas", "themes_include", "themes_exclude")
SCALAR_FIELDS = ("period", "language", "total_questions", "notes")

DRAFT_OPEN, DRAFT_CLOSE = "<draft>", "</draft>"

SYSTEM_PROMPT = """\
You are the evaluation-dataset assistant of a RAG benchmarking platform. Through a \
short guided interview you help the user decide what evaluation questions to \
generate from a document collection. You do NOT generate questions yourself.

Collection profile (computed from the data, trust it):
{profile}

Settle these points, ONE question per turn, in roughly this order:
1. question_types: which kinds of questions. Known types: {types}. New types are allowed if the user needs them.
2. personas: who will ask these questions (e.g. student, historian, analyst, journalist).
3. themes_include / themes_exclude: themes, entities or periods to cover or leave out. \
Propose concrete options from the profile clusters, entities and period.
4. difficulty_mix: share of easy / medium / hard questions (sums to 1).
5. language: language of the questions and answers (default: the collection language).
6. total_questions: how many questions in total.

Rules:
- Ask one thing at a time and always propose a sensible default taken from the profile.
- Keep each message short (at most ~80 words), in the user's language.
- When the user accepts a default or answers, record it and move on.
- When every point is settled, summarise the plan in a few lines and ask the user to \
confirm or adjust. Do not ask further questions after confirmation.

After your message, ALWAYS append exactly one block, on its own line:
<draft>{{JSON}}</draft>
The JSON contains only the fields you can fill so far, among: \
"question_types" (list), "personas" (list), "themes_include" (list), "themes_exclude" (list), \
"period" (string), "difficulty_mix" ({{"easy": x, "medium": y, "hard": z}}), \
"language" (string), "total_questions" (int), "notes" (string), plus \
"quick_replies": 2 to 4 short answers the user could click for your question. \
The user never sees this block.
"""


# ---------------------------------------------------------------------------
# Draft plan handling
# ---------------------------------------------------------------------------


def normalize_draft(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the known draft fields with sane types; drop everything else."""
    out: Dict[str, Any] = {}
    for f in LIST_FIELDS:
        v = raw.get(f)
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            items = [str(x).strip() for x in v if str(x).strip()]
            if items:
                out[f] = items
    for f in ("period", "language", "notes"):
        v = raw.get(f)
        if isinstance(v, (str, int, float)) and str(v).strip():
            out[f] = str(v).strip()
    total = raw.get("total_questions")
    try:
        total = int(total) if total is not None else None
    except (TypeError, ValueError):
        total = None
    if total and total > 0:
        out["total_questions"] = total
    mix = raw.get("difficulty_mix")
    if isinstance(mix, dict):
        vals = {}
        for k in ("easy", "medium", "hard"):
            try:
                vals[k] = max(0.0, float(mix.get(k, 0) or 0))
            except (TypeError, ValueError):
                vals[k] = 0.0
        s = sum(vals.values())
        if s > 0:
            # Accept percentages or fractions; store fractions summing to 1
            out["difficulty_mix"] = {k: round(v / s, 3) for k, v in vals.items()}
    return out


def missing_fields(draft: Dict[str, Any]) -> List[str]:
    return [f for f in REQUIRED_FIELDS if not draft.get(f)]


def split_draft_block(text: str) -> tuple[str, Optional[Dict[str, Any]], List[str]]:
    """Return (visible text, parsed draft or None, quick replies)."""
    start = text.find(DRAFT_OPEN)
    if start == -1:
        return text.strip(), None, []
    visible = text[:start].strip()
    end = text.find(DRAFT_CLOSE, start)
    body = text[start + len(DRAFT_OPEN) : end if end != -1 else len(text)].strip()
    body = re.sub(r"^```(?:json)?|```$", "", body).strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", body, re.DOTALL)
        try:
            data = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            data = None
    if not isinstance(data, dict):
        logger.warning(f"assistant draft block unreadable: {body[:200]}")
        return visible, None, []
    quick = data.get("quick_replies") or []
    quick = [str(q).strip() for q in quick if str(q).strip()][:4] if isinstance(quick, list) else []
    return visible, data, quick


class _DraftFilter:
    """Streams text until the ``<draft>`` marker, holding back a possible partial marker."""

    def __init__(self) -> None:
        self._buf = ""
        self._hidden = False

    def feed(self, piece: str) -> str:
        if self._hidden:
            return ""
        self._buf += piece
        idx = self._buf.find(DRAFT_OPEN)
        if idx != -1:
            out, self._buf, self._hidden = self._buf[:idx], "", True
            return out
        # Keep the longest suffix that could be the start of the marker
        keep = 0
        for k in range(1, len(DRAFT_OPEN)):
            if self._buf.endswith(DRAFT_OPEN[:k]):
                keep = k
        out = self._buf[: len(self._buf) - keep]
        self._buf = self._buf[len(self._buf) - keep :]
        return out

    def flush(self) -> str:
        out = "" if self._hidden else self._buf
        self._buf = ""
        return out


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@dataclass
class ChatSession:
    session_id: str
    collection: str
    profile_json: Optional[str]
    messages: List[Dict[str, str]] = field(default_factory=list)
    draft: Dict[str, Any] = field(default_factory=dict)
    quick_replies: List[str] = field(default_factory=list)
    usage: UsageTracker = field(default_factory=UsageTracker)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def state(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "collection": self.collection,
            "has_profile": self.profile_json is not None,
            "messages": list(self.messages),
            "draft": dict(self.draft),
            "missing": missing_fields(self.draft),
            "complete": not missing_fields(self.draft),
            "quick_replies": list(self.quick_replies),
            "cost_summary": self.usage.summary(),
        }


class SessionStore:
    """In-memory sessions; the least recently used one is dropped past ``max_sessions``."""

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._sessions: Dict[str, ChatSession] = {}
        self._lock = threading.Lock()
        self._max = max_sessions

    def create(self, collection: str, profile_json: Optional[str]) -> ChatSession:
        s = ChatSession(uuid.uuid4().hex[:16], collection, profile_json)
        with self._lock:
            if len(self._sessions) >= self._max:
                oldest = min(self._sessions.values(), key=lambda x: x.updated_at)
                self._sessions.pop(oldest.session_id, None)
            self._sessions[s.session_id] = s
        return s

    def get(self, session_id: str) -> Optional[ChatSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None


# ---------------------------------------------------------------------------
# Chat turn
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _default_llm():
    """(client, model, extra request kwargs) for the assistant model."""
    from config.settings import settings
    from src.llm import get_llm_client, usage_request_kwargs

    provider = settings.ASSISTANT_PROVIDER
    return get_llm_client(provider), settings.ASSISTANT_MODEL, usage_request_kwargs(provider)


def _load_profile_json(collection: str) -> Optional[str]:
    from src.assistant.dataset_profiler import get_profile

    return get_profile(collection).to_prompt_json()


class EvalChatService:
    """Runs interview turns; one instance per API process."""

    def __init__(
        self,
        store: Optional[SessionStore] = None,
        llm_factory: Callable[[], tuple] = _default_llm,
        profile_loader: Callable[[str], Optional[str]] = _load_profile_json,
    ) -> None:
        self.store = store or SessionStore()
        self._llm_factory = llm_factory
        self._profile_loader = profile_loader

    def _system_prompt(self, session: ChatSession) -> str:
        return SYSTEM_PROMPT.format(
            profile=session.profile_json or "(profile unavailable: ask the user about the content)",
            types=", ".join(KNOWN_QUESTION_TYPES),
        )

    def chat(
        self, message: str, session_id: Optional[str] = None, collection: Optional[str] = None
    ) -> Generator[str, None, None]:
        """One interview turn as SSE: ``session``, ``token``*, ``plan_update``, ``done`` | ``error``.

        Without ``session_id`` a session is created for ``collection`` and an
        empty ``message`` lets the assistant open the interview.
        """
        if session_id:
            session = self.store.get(session_id)
            if session is None:
                yield _sse("error", {"message": f"Unknown or expired session '{session_id}'"})
                return
        else:
            if not collection:
                yield _sse("error", {"message": "collection is required to start a session"})
                return
            try:
                profile = self._profile_loader(collection)
            except KeyError:
                yield _sse("error", {"message": f"Collection '{collection}' not found"})
                return
            except Exception as exc:  # profile is helpful, not required
                logger.warning(f"profile unavailable for {collection}: {exc}")
                profile = None
                yield _sse("warning", {"message": f"Dataset profile unavailable: {exc}"})
            session = self.store.create(collection, profile)

        if not message.strip() and session.messages:
            yield _sse("error", {"message": "Empty message"})
            return

        if not session.lock.acquire(blocking=False):
            yield _sse("error", {"message": "A reply is already being generated for this session"})
            return
        try:
            yield _sse("session", {"session_id": session.session_id, "collection": session.collection})
            yield from self._turn(session, message.strip())
        finally:
            session.updated_at = time.time()
            session.lock.release()

    def _turn(self, session: ChatSession, message: str) -> Generator[str, None, None]:
        history = session.messages[-MAX_HISTORY_MESSAGES:]
        user_msg = message or "Start the interview: greet me in one sentence, then ask the first question."
        messages = [{"role": "system", "content": self._system_prompt(session)}]
        messages += history
        if session.draft:
            messages.append(
                {
                    "role": "system",
                    "content": "Current draft (keep fields unless the user changes them): "
                    + json.dumps(session.draft, ensure_ascii=False),
                }
            )
        messages.append({"role": "user", "content": user_msg})

        client, model, extra = self._llm_factory()
        filt = _DraftFilter()
        parts: List[str] = []
        usage = None
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=900,
                stream=True,
                stream_options={"include_usage": True},
                **extra,
            )
            for chunk in stream:
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                if not getattr(chunk, "choices", None):
                    continue
                piece = chunk.choices[0].delta.content or ""
                if not piece:
                    continue
                parts.append(piece)
                visible = filt.feed(piece)
                if visible:
                    yield _sse("token", {"token": visible})
            tail = filt.flush()
            if tail:
                yield _sse("token", {"token": tail})
        except Exception as exc:
            logger.error(f"assistant LLM call failed: {exc}")
            yield _sse("error", {"message": f"Assistant model call failed: {exc}"})
            return
        finally:
            if usage is not None:
                from src.llm import extract_usage

                session.usage.record_usage(
                    "assistant", model, extract_usage(SimpleNamespace(usage=usage))
                )

        visible, draft_update, quick = split_draft_block("".join(parts))
        if message:
            session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": visible})
        if draft_update is not None:
            session.draft.update(normalize_draft(draft_update))
        session.quick_replies = quick

        missing = missing_fields(session.draft)
        yield _sse(
            "plan_update",
            {
                "draft": session.draft,
                "missing": missing,
                "complete": not missing,
                "quick_replies": quick,
            },
        )
        yield _sse(
            "done",
            {"message": visible, "cost_summary": session.usage.summary()},
        )
