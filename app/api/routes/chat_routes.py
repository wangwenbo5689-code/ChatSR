from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Form, Request
from fastapi.responses import StreamingResponse

from app.api.app_context import get_app_context
from app.api.route_utils import (
    ChatStreamForm,
    clamp_generation_params,
    parse_selected_doc_paths,
    require_non_empty_text,
    require_valid_identifier,
)


router = APIRouter()


def _build_chat_stream_form(
    message: str = Form(...),
    session_id: str = Form(...),
    max_length: Optional[int] = Form(None),
    context_len: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    selected_doc_paths_json: str = Form("[]"),
) -> ChatStreamForm:
    return ChatStreamForm(
        message=message,
        session_id=session_id,
        max_length=max_length,
        context_len=context_len,
        temperature=temperature,
        selected_doc_paths_json=selected_doc_paths_json,
    )


@router.post("/api/chat/stream")
def chat_stream(
    request: Request,
    background_tasks: BackgroundTasks,
    form: ChatStreamForm = Depends(_build_chat_stream_form),
):
    context = get_app_context(request)
    limits = context.config.generation_limits

    message = require_non_empty_text(form.message, "message不能为空")
    session_id = require_valid_identifier(
        form.session_id,
        context.session_store.validate_session_id,
        "无效的会话ID",
    )
    max_length, context_len, temperature = clamp_generation_params(
        max_length=form.max_length,
        context_len=form.context_len,
        temperature=form.temperature,
        limits=limits,
    )
    selected_doc_paths = parse_selected_doc_paths(
        form.selected_doc_paths_json, context.corpus_service
    )

    return StreamingResponse(
        context.chat_service.stream_events(
            model=context.model,
            sessions=context.sessions,
            session_lock=context.session_lock,
            background_tasks=background_tasks,
            message=message,
            session_id=session_id,
            max_length=max_length,
            context_len=context_len,
            temperature=temperature,
            selected_doc_paths=selected_doc_paths,
        ),
        media_type="text/event-stream",
    )
