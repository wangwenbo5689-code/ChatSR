import os
from typing import Any, Dict, List

from loguru import logger

from app.core.docling_academic_chunker import create_docling_academic_chunker
from app.core.rag_defaults import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, MIN_CHUNK_SIZE
from app.core.rag_common import ACADEMIC_TOKENIZER_LOCAL_DIRNAME, ACADEMIC_TOKENIZER_MODEL_NAME, make_chunk_item


class RagDocumentService:
    def __init__(self, rag):
        self.rag = rag

    def build_doc_chunks(self, doc_file: str, doc_hash: str) -> List[Dict[str, Any]]:
        self.ensure_docling_academic_chunker()
        chunk_items = self.build_chunks_from_docling(doc_file=doc_file, doc_hash=doc_hash)
        logger.info(f"HybridChunker chunks extracted: {len(chunk_items)}")
        return chunk_items

    def ensure_docling_academic_chunker(self) -> None:
        if self.rag.docling_academic_chunker is not None or not self.rag.docling_extractor:
            return
        try:
            max_tokens = max(int(getattr(self.rag, "chunk_size", DEFAULT_CHUNK_SIZE) or DEFAULT_CHUNK_SIZE), MIN_CHUNK_SIZE)
            overlap_tokens = max(int(getattr(self.rag, "chunk_overlap", DEFAULT_CHUNK_OVERLAP) or DEFAULT_CHUNK_OVERLAP), 0)
            self.rag.docling_academic_chunker = create_docling_academic_chunker({
                "max_tokens": max_tokens,
                "overlap_tokens": overlap_tokens,
                "tokenizer_model": ACADEMIC_TOKENIZER_MODEL_NAME,
                "tokenizer_local_dir": os.path.join(
                    self.rag.save_corpus_emb_dir,
                    "models",
                    ACADEMIC_TOKENIZER_LOCAL_DIRNAME,
                ),
            })
            if self.rag.docling_academic_chunker:
                logger.info("Docling academic chunker enabled (lazy init)")
        except Exception as e:
            logger.warning(f"Lazy init DoclingAcademicChunker failed: {e}")
            self.rag.docling_academic_chunker = None

    def build_chunks_from_docling(self, doc_file: str, doc_hash: str) -> List[Dict[str, Dict[str, object]]]:
        if not (self.rag.docling_academic_chunker and self.rag.docling_extractor):
            raise RuntimeError("Docling academic chunker is not initialized")

        doc = self.rag.docling_extractor.extract_document(doc_file)
        enhanced_chunks = self.rag.docling_academic_chunker.chunk_document(doc)
        items: List[Dict[str, Dict[str, object]]] = []
        for idx, (text, metadata) in enumerate(enhanced_chunks):
            clean_text = (text or "").strip()
            if not clean_text:
                continue
            chunk_meta = dict(metadata or {})
            chunk_meta.setdefault("doc_hash", doc_hash)
            chunk_meta.setdefault("doc_file", doc_file)
            chunk_meta.setdefault("chunk_index", idx)
            chunk_meta.setdefault("text_content", clean_text)
            items.append(make_chunk_item(text=clean_text, metadata=chunk_meta))
        if not items:
            raise RuntimeError(f"HybridChunker produced no chunks for {doc_file}")
        return items
