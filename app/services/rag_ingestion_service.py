from typing import List, Union

from loguru import logger

from app.core.rag_hash_utils import file_md5_hex_32


class RagIngestionService:
    def __init__(self, rag):
        self.rag = rag

    def add_corpus(self, files: Union[str, List[str]], persist_files: bool = True):
        if isinstance(files, str):
            files = [files]

        document_service = self.rag._document_service
        index_service = self.rag._index_service

        indexed_any = False
        for doc_file in files:
            doc_hash = file_md5_hex_32(doc_file)
            if index_service.is_doc_indexed(doc_hash):
                logger.info(f"Document already indexed, skip rebuilding: {doc_file}")
                continue

            chunk_items = document_service.build_doc_chunks(doc_file=doc_file, doc_hash=doc_hash)
            self.rag.hierarchical_retriever.index_document(
                doc_hash=doc_hash,
                doc_file=doc_file,
                chunk_items=chunk_items,
            )
            indexed_any = True

        if persist_files:
            index_service.merge_corpus_files(files)
        if indexed_any:
            index_service.sync_runtime_indexes()
        index_counts = self.rag.hierarchical_retriever.get_index_counts()
        logger.debug(
            f"files: {files}, paper count: {index_counts.get('paper', 0)}, "
            f"section count: {index_counts.get('section', 0)}, element count: {index_counts.get('element', 0)}"
        )
