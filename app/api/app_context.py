import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

from fastapi import Request

from app.api.app_config import ApiRuntimeConfig
from app.core.rag_runtime import RagRuntime
from app.services.chat_service import ChatService
from app.services.corpus_service import CorpusService
from app.services.embedding_lifecycle_service import EmbeddingLifecycleService
from app.services.session_store import SessionStore


@dataclass
class AppContext:
    """应用共享上下文

    锁设计说明：
    - 使用 threading.Lock 而非 asyncio.Lock，因为当前路由均为同步函数 (def)
    - 同步路由在 uvicorn 线程池中执行，threading.Lock 是正确选择
    - lifespan 是 async 函数，但模型加载为一次性操作，with 同步锁不阻塞主事件循环
    - 如需全部切换为 async，需将路由改为 async def 并替换为 asyncio.Lock
    """
    config: ApiRuntimeConfig
    session_lock: threading.Lock
    rag_lock: threading.Lock
    session_store: SessionStore
    corpus_service: CorpusService
    chat_service: ChatService
    embedding_service: EmbeddingLifecycleService
    sessions: Dict[str, Dict[str, Any]]
    model: Optional[RagRuntime] = None


def build_app_context(config: ApiRuntimeConfig) -> AppContext:
    session_store = SessionStore(config.session_store_path)
    corpus_service = CorpusService(
        local_corpus_dir=config.local_corpus_dir,
        allowed_exts=config.allowed_exts,
        max_files_per_upload=config.max_files_per_upload,
        max_file_size_mb=config.max_file_size_mb,
    )
    return AppContext(
        config=config,
        session_lock=threading.Lock(),
        rag_lock=threading.Lock(),
        session_store=session_store,
        corpus_service=corpus_service,
        chat_service=ChatService(session_store),
        embedding_service=EmbeddingLifecycleService(corpus_service),
        sessions=session_store.load(),
    )


def get_app_context(request: Request) -> AppContext:
    return request.app.state.context
