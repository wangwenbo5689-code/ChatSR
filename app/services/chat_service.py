from typing import Any

from fastapi import BackgroundTasks

import copy
import json
import os
import uuid

from loguru import logger

class ChatService:
    def __init__(self, session_store):
        self.session_store = session_store

    @staticmethod
    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @staticmethod
    def _log(level: str, message: str, **fields):
        logger.bind(**fields).log(level.upper(), message)

    @staticmethod
    def _background_compress(model, sessions, session_lock, history_to_compress, current_summary, sid, store):
        try:
            updated_h, updated_s = model.compress_history_if_needed(
                history_to_compress, current_summary
            )
            if updated_h != history_to_compress or updated_s != current_summary:
                with session_lock:
                    store.apply_compression_result(
                        sessions=sessions,
                        session_id=sid,
                        history_to_compress=history_to_compress,
                        updated_history=updated_h,
                        updated_summary=updated_s,
                    )
        except Exception as exc:
            logger.error(f"Background compression failed: {exc}")

    def stream_events(
        self,
        model: Any,
        sessions: dict,
        session_lock: Any,
        background_tasks: BackgroundTasks,
        message: str,
        session_id: str,
        max_length: int,
        context_len: int,
        temperature: float,
        selected_doc_paths: list | None = None,
    ):
        query_id = str(uuid.uuid4())
        with session_lock:
            session = self.session_store.ensure_session(sessions, session_id)
            history_snapshot = copy.deepcopy(session.get("history", []))
            summary_snapshot = session.get("summary", "")

        assistant_full = ""
        latest_summary = summary_snapshot
        intent = "unknown"
        try:
            intent = model.query_expander.classify_intent(message)
        except Exception:
            pass
        self._log(
            "info",
            "chat_stream_started",
            session_id=session_id,
            query_id=query_id,
            intent=intent,
        )

        try:
            yield self._sse({"type": "start", "query_id": query_id, "intent": intent})
            for chunk in model.predict_stream(
                message,
                max_length=max_length,
                context_len=context_len,
                temperature=temperature,
                history=history_snapshot,
                history_summary=summary_snapshot,
                selected_doc_paths=selected_doc_paths,
                intent=intent if intent != "unknown" else None,
                trace_ctx={"session_id": session_id, "query_id": query_id},
            ):
                if isinstance(chunk, dict) and "history" in chunk:
                    latest_summary = chunk["history_summary"]
                    break
                assistant_full += chunk
                yield self._sse({"type": "delta", "content": chunk})

            with session_lock:
                final_history = self.session_store.store_chat_turn(sessions, session_id, message, assistant_full)

            background_tasks.add_task(
                self._background_compress,
                model,
                sessions,
                session_lock,
                final_history,
                latest_summary,
                session_id,
                self.session_store,
            )
            self._log(
                "info",
                "chat_stream_done",
                session_id=session_id,
                query_id=query_id,
                intent=intent,
                answer_length=len(assistant_full),
            )
            yield self._sse({"type": "done", "query_id": query_id, "intent": intent})
        except Exception as exc:
            logger.exception(exc)
            error_msg, error_code = self._classify_error(exc)
            self._log(
                "error",
                "chat_stream_error",
                session_id=session_id,
                query_id=query_id,
                intent=intent,
                error_code=error_code,
                error_message=error_msg,
            )
            yield self._sse(
                {
                    "type": "error",
                    "message": error_msg,
                    "error_code": error_code,
                    "query_id": query_id,
                    "intent": intent,
                }
            )

    @staticmethod
    def _classify_error(exc: Exception) -> tuple[str, str]:
        try:
            from app.core.rag_query_expander import IntentClassificationError

            if isinstance(exc, IntentClassificationError):
                msg = str(exc)
                detail = getattr(exc, "detail", None)
                if os.getenv("CHATSR_DEV_MODE", "0") == "1" and isinstance(detail, str) and detail:
                    msg = f"{msg}: {detail}"
                return msg, "INTENT_CLASSIFICATION_ERROR"
        except Exception:
            pass

        if os.getenv("CHATSR_DEV_MODE", "0") == "1":
            return f"处理请求时发生错误: {exc}", "INTERNAL_ERROR"
        return "处理请求时发生错误", "INTERNAL_ERROR"
