from app.api.routes.chat_routes import router as chat_router
from app.api.routes.corpus_routes import router as corpus_router
from app.api.routes.embedding_routes import router as embedding_router
from app.api.routes.session_routes import router as session_router
from app.api.routes.system_routes import router as system_router


__all__ = [
    "chat_router",
    "corpus_router",
    "embedding_router",
    "session_router",
    "system_router",
]
