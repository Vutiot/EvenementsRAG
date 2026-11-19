"""Tests for the eval-dataset interview backend (E8-F2-T1)."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.assistant.eval_chat import (
    EvalChatService,
    SessionStore,
    _DraftFilter,
    missing_fields,
    normalize_draft,
    split_draft_block,
)


def _chunks(text, piece=7, usage=(120, 30)):
    out = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text[i : i + piece]))], usage=None)
        for i in range(0, len(text), piece)
    ]
    out.append(
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1], model_extra={}),
        )
    )
    return out


def _service(replies, profile='{"n_chunks": 10}'):
    client = MagicMock()
    client.chat.completions.create.side_effect = [iter(_chunks(r)) for r in replies]
    svc = EvalChatService(
        store=SessionStore(),
        llm_factory=lambda: (client, "test-model", {}),
        profile_loader=lambda c: profile,
    )
    return svc, client


def _events(stream):
    out = []
    for raw in stream:
        lines = raw.strip().split("\n")
        out.append((lines[0][len("event: "):], json.loads(lines[1][len("data: "):])))
    return out


REPLY_1 = (
    "Hello! Which question types do you want? I suggest factual and temporal.\n"
    '<draft>{"question_types": ["factual", "temporal"], "language": "en", '
    '"quick_replies": ["Factual + temporal", "Add analytical"]}</draft>'
)
REPLY_2 = (
    "Noted. Who will ask these questions?\n"
    '<draft>{"personas": ["student"], "difficulty_mix": {"easy": 30, "medium": 50, "hard": 20}}</draft>'
)


def test_first_turn_creates_session_and_hides_draft_block():
    svc, client = _service([REPLY_1])
    ev = _events(svc.chat("", collection="ww2"))
    names = [e for e, _ in ev]
    assert names[0] == "session" and names[-2:] == ["plan_update", "done"]
    streamed = "".join(d["token"] for e, d in ev if e == "token")
    assert "<draft>" not in streamed and "draft" not in streamed
    assert streamed.strip().startswith("Hello!")
    plan = dict(ev)["plan_update"]
    assert plan["draft"]["question_types"] == ["factual", "temporal"]
    assert plan["quick_replies"] == ["Factual + temporal", "Add analytical"]
    assert plan["complete"] is False and "personas" in plan["missing"]
    # system prompt carries the profile
    sent = client.chat.completions.create.call_args.kwargs["messages"]
    assert '"n_chunks": 10' in sent[0]["content"]
    assert dict(ev)["done"]["cost_summary"]["total_input_tokens"] == 120


def test_second_turn_merges_draft_and_keeps_history():
    svc, client = _service([REPLY_1, REPLY_2])
    sid = dict(_events(svc.chat("", collection="ww2")))["session"]["session_id"]
    ev = dict(_events(svc.chat("factual and temporal please", session_id=sid)))
    draft = ev["plan_update"]["draft"]
    assert draft["question_types"] == ["factual", "temporal"]
    assert draft["personas"] == ["student"]
    assert draft["difficulty_mix"] == {"easy": 0.3, "medium": 0.5, "hard": 0.2}
    sent = client.chat.completions.create.call_args.kwargs["messages"]
    roles = [m["role"] for m in sent]
    assert roles[0] == "system" and roles[-1] == "user"
    assert any("Current draft" in m["content"] for m in sent if m["role"] == "system")
    state = svc.store.get(sid).state()
    assert [m["role"] for m in state["messages"]] == ["assistant", "user", "assistant"]
    assert state["cost_summary"]["by_stage"]["assistant"]["models"]["test-model"]["calls"] == 2


def test_errors():
    svc, _ = _service([])
    assert _events(svc.chat("hi", session_id="nope"))[0][0] == "error"
    assert _events(svc.chat("hi"))[0][0] == "error"  # no collection

    def missing(c):
        raise KeyError(c)

    svc2 = EvalChatService(llm_factory=lambda: None, profile_loader=missing)
    assert _events(svc2.chat("", collection="x"))[0] == ("error", {"message": "Collection 'x' not found"})


def test_profile_failure_is_a_warning_not_an_error():
    def broken(c):
        raise RuntimeError("qdrant down")

    client = MagicMock()
    client.chat.completions.create.return_value = iter(_chunks(REPLY_1))
    svc = EvalChatService(llm_factory=lambda: (client, "m", {}), profile_loader=broken)
    names = [e for e, _ in _events(svc.chat("", collection="ww2"))]
    assert names[0] == "warning" and "done" in names


def test_llm_failure_reports_error_and_keeps_session_usable():
    client = MagicMock()
    client.chat.completions.create.side_effect = [RuntimeError("503"), iter(_chunks(REPLY_1))]
    svc = EvalChatService(llm_factory=lambda: (client, "m", {}), profile_loader=lambda c: None)
    ev = _events(svc.chat("", collection="ww2"))
    assert ev[-1][0] == "error"
    sid = ev[0][1]["session_id"]
    assert svc.store.get(sid).messages == []
    assert _events(svc.chat("", session_id=sid))[-1][0] == "done"


def test_draft_filter_holds_partial_marker():
    f = _DraftFilter()
    out = f.feed("Hello <dr") + f.feed("aft>{}") + f.feed("</draft>") + f.flush()
    assert out == "Hello "
    f2 = _DraftFilter()
    assert f2.feed("a <b") + f2.flush() == "a <b"


def test_split_draft_block_variants():
    assert split_draft_block("no block") == ("no block", None, [])
    vis, d, q = split_draft_block('Hi\n<draft>```json\n{"language": "fr"}\n```</draft>')
    assert (vis, d, q) == ("Hi", {"language": "fr"}, [])
    vis, d, q = split_draft_block("Hi <draft>not json")
    assert vis == "Hi" and d is None


def test_normalize_draft_and_completeness():
    d = normalize_draft(
        {
            "question_types": "factual",
            "personas": ["", "historian"],
            "total_questions": "50",
            "difficulty_mix": {"easy": 1, "medium": 1, "hard": 2},
            "language": "fr",
            "themes_include": ["Pacific war"],
            "unknown": 1,
        }
    )
    assert d["question_types"] == ["factual"] and d["personas"] == ["historian"]
    assert d["total_questions"] == 50 and d["difficulty_mix"]["hard"] == 0.5
    assert "unknown" not in d
    assert missing_fields(d) == []


def test_session_store_evicts_least_recently_used():
    store = SessionStore(max_sessions=2)
    a = store.create("c", None)
    b = store.create("c", None)
    a.updated_at, b.updated_at = 2.0, 1.0
    store.create("c", None)
    assert store.get(b.session_id) is None and store.get(a.session_id) is not None


def test_chat_endpoint_streams_sse(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.routers import assistant

    svc, _ = _service([REPLY_1])
    monkeypatch.setattr(assistant, "_chat_service", svc)
    app = FastAPI()
    app.include_router(assistant.router, prefix="/api")
    client = TestClient(app)
    r = client.post("/api/assistant/eval/chat", json={"collection": "ww2"})
    assert r.status_code == 200
    assert "event: plan_update" in r.text and "event: done" in r.text
    sid = json.loads(r.text.split("\n")[1][len("data: "):])["session_id"]
    state = client.get(f"/api/assistant/eval/sessions/{sid}").json()
    assert state["draft"]["language"] == "en"
    assert client.delete(f"/api/assistant/eval/sessions/{sid}").status_code == 200
    assert client.get(f"/api/assistant/eval/sessions/{sid}").status_code == 404
