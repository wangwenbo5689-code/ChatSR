from loguru import logger


class RagPersistenceService:
    def __init__(self, rag):
        self.rag = rag

    def save_corpus_emb(self):
        save_dir = self.rag.resolve_embedding_dir()
        self.rag.hierarchical_retriever.save_persist_directory(save_dir)
        self.rag._index_service.switch_persist_directory(save_dir)
        logger.debug(f"Saving corpus embeddings to {save_dir}")
        return save_dir

    def load_corpus_emb(self, emb_dir: str):
        logger.debug(f"Loading corpus embeddings from {emb_dir}")
        self.rag.hierarchical_retriever.load_persist_directory(emb_dir)
        self.rag._index_service.switch_persist_directory(emb_dir)
        return emb_dir
