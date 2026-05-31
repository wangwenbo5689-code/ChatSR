from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from app.api.app_context import get_app_context


router = APIRouter()


@router.get("/")
def index_page(request: Request):
    context = get_app_context(request)
    return FileResponse(str(context.config.index_html_path))


@router.get("/api/health")
def health(request: Request):
    """
    健康检查

    返回内容：
    - ok: 整体健康状态（所有检查通过才为 true）
    - checks: 各组件详细状态
        - model: 模型加载状态
        - corpus_service: 语料服务状态
        - session_store: 会话存储状态
    - details: 额外详细信息
    """
    context = get_app_context(request)

    model_loaded = context.model is not None
    corpus_service_ok = context.corpus_service is not None
    session_store_ok = context.session_store is not None

    details = {}
    if model_loaded and context.model is not None:
        try:
            details["chunk_count"] = context.model.hierarchical_retriever.get_index_counts().get("element", 0)
        except Exception:
            details["chunk_count"] = "unknown"

    return {
        "ok": model_loaded and corpus_service_ok and session_store_ok,
        "checks": {
            "model_loaded": model_loaded,
            "corpus_service_ok": corpus_service_ok,
            "session_store_ok": session_store_ok,
        },
        "details": details,
    }


@router.get("/api/system/config")
def system_config(request: Request):
    context = get_app_context(request)
    return {
        "generation_limits": context.config.generation_limits,
        "retrieval_strategy": context.config.retrieval_strategy,
    }
