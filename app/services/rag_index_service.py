from typing import List, Union


class RagIndexService:
    def __init__(self, rag):
        self.rag = rag

    def sync_runtime_indexes(self):
        self.rag.hierarchical_retriever.refresh_indexes()
        self.rag.chunk_metadata = self.rag.hierarchical_retriever.build_element_metadata_index()

    def is_doc_indexed(self, doc_hash: str) -> bool:
        return any(
            isinstance(meta, dict) and meta.get('doc_hash') == doc_hash
            for meta in self.rag.chunk_metadata.values()
        )

    def merge_corpus_files(self, files: Union[str, List[str], None]) -> List[str]:
        incoming = self._normalize_files(files)
        existing = self._normalize_files(getattr(self.rag, "corpus_files", None))
        merged = list(dict.fromkeys(existing + incoming))
        self.rag.corpus_files = merged
        return merged

    def switch_persist_directory(self, persist_directory: str):
        self.rag.chroma_persist_directory = persist_directory
        self.rag.hierarchical_retriever.switch_persist_directory(persist_directory)
        self.sync_runtime_indexes()

    @staticmethod
    def _normalize_files(files: Union[str, List[str], None]) -> List[str]:
        if not files:
            return []
        if isinstance(files, str):
            files = [files]
        return [f for f in files if isinstance(f, str) and f]
