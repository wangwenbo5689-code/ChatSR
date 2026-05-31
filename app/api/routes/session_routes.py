import uuid

from fastapi import APIRouter, Form, Request

from app.api.app_context import get_app_context
from app.api.route_utils import require_text_max_length, require_valid_identifier, run_action


router = APIRouter()


def _run_session_action(action, error_detail: str):
    return run_action(
        action,
        error_detail,
        key_error_detail="session not found",
    )


def _validate_session_id(context, session_id: str) -> str:
    return require_valid_identifier(
        session_id,
        context.session_store.validate_session_id,
        "无效的会话ID",
    )


@router.get("/api/sessions")
def get_sessions(request: Request):
    context = get_app_context(request)
    with context.session_lock:
        items = context.session_store.list_sessions(context.sessions)
    return {"sessions": items}


@router.get("/api/sessions/{session_id}")
def get_session_messages(request: Request, session_id: str):
    context = get_app_context(request)
    with context.session_lock:
        return _run_session_action(
            lambda: context.session_store.get_session_payload(context.sessions, session_id),
            "获取会话消息失败",
        )


@router.post("/api/sessions")
def create_session(request: Request):
    context = get_app_context(request)
    session_id = str(uuid.uuid4())
    with context.session_lock:
        context.session_store.create_session(context.sessions, session_id)
    return {"id": session_id}


@router.patch("/api/sessions/{session_id}")
def rename_session(request: Request, session_id: str, title: str = Form(...)):
    context = get_app_context(request)
    session_id = _validate_session_id(context, session_id)
    title = require_text_max_length(
        title,
        100,
        "title不能为空",
        too_long_detail="标题长度不能超过100个字符",
    )
    with context.session_lock:
        _run_session_action(
            lambda: context.session_store.rename_session(context.sessions, session_id, title),
            "重命名会话失败",
        )
    return {"ok": True}


@router.delete("/api/sessions/{session_id}")
def delete_session(request: Request, session_id: str):
    context = get_app_context(request)
    session_id = _validate_session_id(context, session_id)
    with context.session_lock:
        existed = context.session_store.delete_session(context.sessions, session_id)
    return {"ok": True, "deleted": existed}


@router.delete("/api/sessions")
def clear_sessions(request: Request):
    context = get_app_context(request)
    with context.session_lock:
        context.session_store.clear_sessions(context.sessions)
    return {"ok": True}
