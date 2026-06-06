import os

from loguru import logger


class RagPersistenceService:
    def __init__(self, rag):
        self.rag = rag

    def _current_persist_directory(self) -> str:
        return str(
            getattr(self.rag, "chroma_persist_directory", "")
            or getattr(self.rag.hierarchical_retriever, "persist_directory", "")
        )

    def save_corpus_emb(self):
        save_dir = self._current_persist_directory()
        self.rag.hierarchical_retriever.save_persist_directory(save_dir)
        self.rag._index_service.switch_persist_directory(save_dir)
        logger.debug(f"Saving corpus embeddings in active Chroma directory: {save_dir}")
        return save_dir

    def load_corpus_emb(self, emb_dir: str):
        active_dir = self._current_persist_directory()
        requested_dir = str(emb_dir or active_dir)
        if os.path.abspath(requested_dir) != os.path.abspath(active_dir):
            raise ValueError(
                f"Only the active Chroma index is supported: {active_dir}. "
                f"Refusing legacy index path: {requested_dir}"
            )
        logger.debug(f"Loading active Chroma embeddings from {active_dir}")
        self.rag.hierarchical_retriever.load_persist_directory(active_dir)
        self.rag._index_service.switch_persist_directory(active_dir)
        return active_dir
