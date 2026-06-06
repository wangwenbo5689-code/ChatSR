import os
import sqlite3
from urllib.parse import quote

from fastapi import HTTPException
from loguru import logger


class EmbeddingLifecycleService:
    def __init__(self, corpus_service):
        self.corpus_service = corpus_service

    @staticmethod
    def _chunk_count(model) -> int:
        return model.hierarchical_retriever.get_index_counts().get("element", 0)

    @staticmethod
    def _bind_corpus_files(model, corpus_files) -> None:
        model.corpus_files = corpus_files

    def _embedding_payload(self, embedding_dir: str, model) -> dict:
        return {
            "ok": True,
            "embedding_dir": embedding_dir,
            "chunk_count": self._chunk_count(model),
        }

    @staticmethod
    def _unpack_state(state: dict):
        return state["corpus_files"], state["embedding_dir"], state["embedding_exists"]

    @staticmethod
    def _ensure_corpus_files(corpus_files, detail: str) -> None:
        if not corpus_files:
            raise HTTPException(status_code=400, detail=detail)

    @staticmethod
    def _chroma_store_exists(persist_directory) -> bool:
        if not isinstance(persist_directory, str) or not persist_directory:
            return False
        sqlite_path = os.path.join(persist_directory, "chroma.sqlite3")
        if not os.path.isdir(persist_directory) or not os.path.isfile(sqlite_path):
            return False
        uri_path = quote(sqlite_path.replace("\\", "/"), safe="/:")
        uri = f"file:{uri_path}?mode=ro&immutable=1"
        try:
            with sqlite3.connect(uri, uri=True, timeout=1) as conn:
                return conn.execute("SELECT 1 FROM embeddings LIMIT 1").fetchone() is not None
        except sqlite3.Error as exc:
            logger.warning(f"failed to inspect Chroma persisted store: {sqlite_path}, error={exc}")
            return False

    def resolve_local_embedding_state(self, model) -> dict:
        corpus_files = self.corpus_service.list_local_corpus_files()
        chroma_dir = getattr(model, "chroma_persist_directory", "")
        chroma_exists = self._chroma_store_exists(chroma_dir)
        chunk_count = self._chunk_count(model)
        persisted_index_exists = chroma_exists
        embedding_dir = chroma_dir
        embedding_exists = persisted_index_exists
        return {
            "corpus_files": corpus_files,
            "embedding_dir": embedding_dir,
            "embedding_exists": embedding_exists,
            "snapshot_dir": None,
            "snapshot_exists": False,
            "chroma_persist_directory": chroma_dir,
            "chroma_store_exists": chroma_exists,
            "persisted_index_exists": persisted_index_exists,
            "chunk_count": chunk_count,
        }

    def bootstrap_local_corpus(self, model, rag_lock) -> None:
        state = self.resolve_local_embedding_state(model)
        corpus_files, embedding_dir, embedding_exists = self._unpack_state(state)
        if not corpus_files:
            logger.info("no local corpus files found on startup")
            return

        with rag_lock:
            self._bind_corpus_files(model, corpus_files)
            if embedding_exists:
                model.load_corpus_emb(embedding_dir)
                logger.info(
                    f"loaded local embedding index: {embedding_dir}, "
                    f"chunks={self._chunk_count(model)}"
                )
            else:
                model.add_corpus(corpus_files)
                logger.info(
                    f"loaded local corpus files: {len(corpus_files)}, "
                    f"chunks={self._chunk_count(model)}"
                )

    def save_embeddings(self, model, rag_lock) -> dict:
        state = self.resolve_local_embedding_state(model)
        corpus_files, _, _ = self._unpack_state(state)
        self._ensure_corpus_files(corpus_files, "本地语料库为空，无法落盘 embedding 索引")
        if not model.hierarchical_retriever.has_indexed_content():
            raise HTTPException(status_code=400, detail="当前内存中没有语料 chunk，请先上传或加载语料")

        with rag_lock:
            self._bind_corpus_files(model, corpus_files)
            embedding_dir = model.save_corpus_emb()
            return self._embedding_payload(embedding_dir, model)

    def load_embeddings(self, model, rag_lock) -> dict:
        state = self.resolve_local_embedding_state(model)
        corpus_files, embedding_dir, embedding_exists = self._unpack_state(state)
        self._ensure_corpus_files(corpus_files, "本地语料库为空，无法加载 embedding 索引")
        if not embedding_exists:
            raise HTTPException(status_code=404, detail="未找到 embedding 索引目录")

        with rag_lock:
            self._bind_corpus_files(model, corpus_files)
            model.load_corpus_emb(embedding_dir)
            return self._embedding_payload(embedding_dir, model)

    def shutdown(self) -> None:
        """应用关闭时的清理逻辑。

        当前版本以日志记录为主，后续可扩展：
        - 保存未落盘的 embedding 索引
        - 关闭 Chroma 连接
        - 释放 GPU 资源
        """
        logger.info("EmbeddingLifecycleService 清理完成")
