from fastapi import APIRouter, Request

from app.api.app_context import get_app_context
from app.api.route_utils import run_action


router = APIRouter()


@router.get("/api/embeddings/status")
def embeddings_status(request: Request):
    context = get_app_context(request)
    model = context.model
    if model is None:
        return {
            "corpus_file_count": 0,
            "chroma_persist_directory": "",
            "embedding_dir": "",
            "embedding_exists": False,
            "in_memory_chunk_count": 0,
            "persisted_index_exists": False,
        }
    state = context.embedding_service.resolve_local_embedding_state(model)
    return {
        "corpus_file_count": len(state["corpus_files"]),
        "chroma_persist_directory": state.get("chroma_persist_directory", ""),
        "embedding_dir": state["embedding_dir"],
        "embedding_exists": state["embedding_exists"],
        "in_memory_chunk_count": state.get("chunk_count", 0),
        "persisted_index_exists": state.get("persisted_index_exists", False),
    }


@router.post("/api/embeddings/save")
def save_embeddings(request: Request):
    context = get_app_context(request)
    return run_action(
        lambda: context.embedding_service.save_embeddings(context.model, context.rag_lock),
        "保存 embedding 索引时发生错误",
    )


@router.post("/api/embeddings/load")
def load_embeddings(request: Request):
    context = get_app_context(request)
    return run_action(
        lambda: context.embedding_service.load_embeddings(context.model, context.rag_lock),
        "加载 embedding 索引时发生错误",
    )
