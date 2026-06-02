# -*- coding: utf-8 -*-
"""
混合检索器架构
多路并行召回 + RRF 融合 + 阈值过滤 + 重排聚合
"""

from collections import defaultdict
import math
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from rank_bm25 import BM25Okapi

from app.core.chroma_similarity import ProjectEmbeddingFunction
from app.core.hierarchical_context_builder import ContextBuilder
from app.core.hierarchical_payload_builder import build_hierarchical_document_payloads
from app.core.hierarchical_ranking import ResultFuser, ResultReranker
from app.core.hierarchical_types import FusedResult, RetrievalLevel, RetrievalResult
from app.core.rag_defaults import (
    DEFAULT_ELEMENT_CANDIDATE_POOL_SIZE,
    DEFAULT_ELEMENT_RETRIEVER_TOP_K,
    DEFAULT_HYBRID_RETRIEVE_TOP_N,
    DEFAULT_PARENT_RERANK_TOP_K,
    DEFAULT_PAPER_RETRIEVER_TOP_K,
    DEFAULT_RERANK_MODEL_NAME,
    default_hierarchical_retriever_config,
)
from app.core.section_filters import is_method_section_metadata, is_noise_section_metadata, is_noise_section_title


# 推荐配置（仅保留当前代码路径实际生效的参数）
RECOMMENDED_CONFIG = default_hierarchical_retriever_config()

class HybridRetriever:
    """混合检索器。

    检索主链路：
    1) 稠密 + 稀疏召回并做 RRF 融合；
    2) 按 intent 做相对阈值过滤；
    3) 可选交叉编码重排（分数按排名位次融合）；
    4) element 层重排后固定保留前 20 个子块，聚合到父章节；
    5) 构建父章节结果并交给 ContextBuilder。

    意图分流（三类）：
    - document_overview: 文档级概述，直接走 paper 层；
    - general_overview: 一般性概述，走 section/element 链路；
    - fact_qa: 事实性问答，走 element 链路。
    """

    def __init__(
        self,
        persist_directory: str,
        embedding_model_name_or_path: str,
        device: str = "cpu",
        rerank_model_name_or_path: str = DEFAULT_RERANK_MODEL_NAME,
        config: Optional[Dict[str, Any]] = None
    ):
        import chromadb
        from chromadb.config import Settings

        self.persist_directory = persist_directory
        self.embedding_model_name_or_path = embedding_model_name_or_path
        self.device = device
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedding_fn = ProjectEmbeddingFunction(
            model_name_or_path=embedding_model_name_or_path,
            device=device,
        )

        def get_or_create(name: str):
            return self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )

        self.paper_collection = get_or_create("paper_collection")
        self.section_collection = get_or_create("section_collection")
        self.element_collection = get_or_create("element_collection")

        self.config = RECOMMENDED_CONFIG.copy()
        if config:
            self.config.update(config)

        self.paper_retriever = None
        self.element_retriever = None
        self.result_fuser = ResultFuser(self.config)
        self.reranker = ResultReranker(
            model_name_or_path=rerank_model_name_or_path,
            device=device,
            config=self.config
        ) if self.config.get("enable_rerank", True) else None
        self.refresh_indexes()

    def _config_value(self, key: str, default: Any = None) -> Any:
        config = getattr(self, "config", {}) or {}
        if key in config:
            return config[key]
        if key in RECOMMENDED_CONFIG:
            return RECOMMENDED_CONFIG[key]
        return default

    def _config_int(self, key: str, default: int) -> int:
        try:
            return int(self._config_value(key, default))
        except (TypeError, ValueError):
            return int(default)

    def _config_float(self, key: str, default: float) -> float:
        try:
            return float(self._config_value(key, default))
        except (TypeError, ValueError):
            return float(default)

    def upsert_collection(self, collection, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]):
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=self.embedding_fn(documents),
        )

    def refresh_indexes(self):
        tokenizer = getattr(self.embedding_fn, "tokenizer", None)
        self.paper_retriever = PaperRetriever(self.paper_collection, self.embedding_fn, self.persist_directory, tokenizer)
        self.element_retriever = ElementRetriever(self.element_collection, self.embedding_fn, self.persist_directory, tokenizer)
        self.context_builder = ContextBuilder(
            element_collection=self.element_collection,
        )

    @staticmethod
    def _normalize_intent(intent: Optional[str]) -> str:
        """标准化意图文本。"""
        return (intent or "").strip().lower() if isinstance(intent, str) else ""

    @staticmethod
    def _empty_response() -> Dict[str, Any]:
        return {
            "context": "",
            "citations": [],
            "blocks": [],
            "result_count": 0,
            "total_length": 0,
            "structured": [],
        }



    @staticmethod
    def _flatten_collection_values(values):
        if values is None:
            return []
        if hasattr(values, "tolist"):
            return values.tolist()
        if isinstance(values, tuple):
            return list(values)
        return values

    def _fetch_collection_records(self, collection, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        record_ids = [record_id for record_id in dict.fromkeys(ids) if record_id]
        if not record_ids:
            return {}
        try:
            results = collection.get(ids=record_ids)
        except Exception as exc:
            logger.warning(f"Load parent records failed: {exc}")
            return {}

        result_ids = self._flatten_collection_values(results.get("ids"))
        documents = self._flatten_collection_values(results.get("documents"))
        metadatas = self._flatten_collection_values(results.get("metadatas"))
        payload = {}
        for index, record_id in enumerate(result_ids):
            payload[record_id] = {
                "document": documents[index] if index < len(documents) else "",
                "metadata": metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {},
            }
        return payload

    @staticmethod
    def _collection_get_with_optional_where(collection, where_clause: Optional[Dict[str, Any]]):
        if where_clause is None:
            return collection.get()
        return collection.get(where=where_clause)

    @staticmethod
    def _retrieve_with_optional_doc_filter(retriever, query: str, topn: int, allowed_doc_hashes: Optional[set[str]]):
        try:
            return retriever.retrieve(
                query,
                topn=topn,
                allowed_doc_hashes=allowed_doc_hashes,
            )
        except TypeError:
            try:
                return retriever.retrieve(query, topn=topn)
            except TypeError:
                return retriever.retrieve(query)

    @staticmethod
    def _normalize_allowed_doc_hashes(allowed_doc_hashes: Optional[set[str]]) -> Optional[set[str]]:
        if not allowed_doc_hashes:
            return None
        normalized = {
            str(doc_hash).strip().lower()
            for doc_hash in allowed_doc_hashes
            if isinstance(doc_hash, str) and str(doc_hash).strip()
        }
        return normalized or None

    @staticmethod
    def _doc_hash_from_meta(meta: Any) -> str:
        if not isinstance(meta, dict):
            return ""
        doc_hash = str(meta.get("doc_hash") or "").strip().lower()
        if doc_hash:
            return doc_hash
        paper_id = str(meta.get("paper_id") or "").strip().lower()
        if paper_id.endswith("_paper"):
            return paper_id[:-6]
        return ""

    @classmethod
    def _is_meta_allowed(cls, meta: Dict[str, Any], allowed_doc_hashes: Optional[set[str]]) -> bool:
        if not allowed_doc_hashes:
            return True
        return cls._doc_hash_from_meta(meta) in allowed_doc_hashes

    @staticmethod
    def _fusion_weights_for_intent(intent: str, level: str) -> Tuple[float, float]:
        if intent == "fact_qa":
            if level == RetrievalLevel.PAPER.value:
                return 0.9, 1.15
            return 1.0, 1.2
        if intent == "general_overview" and level != RetrievalLevel.PAPER.value:
            return 1.3, 0.9
        if level == RetrievalLevel.PAPER.value:
            return 1.15, 0.95
        return 1.2, 0.9

    @staticmethod
    def _is_noise_section(section_title: Optional[str]) -> bool:
        return is_noise_section_title(section_title)

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        cleaned = re.sub(r"[ \t]+", " ", (text or "").strip())
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit].rstrip() + "..."

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _collection_records_to_list(records) -> List[Dict[str, Any]]:
        if not records:
            return []
        record_ids = HybridRetriever._flatten_collection_values(records.get("ids"))
        documents = HybridRetriever._flatten_collection_values(records.get("documents"))
        metadatas = HybridRetriever._flatten_collection_values(records.get("metadatas"))
        payload: List[Dict[str, Any]] = []
        for index, record_id in enumerate(record_ids or []):
            payload.append({
                "id": record_id,
                "document": documents[index] if index < len(documents) else "",
                "metadata": metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {},
            })
        return payload

    @staticmethod
    def _where_clause_for_doc_hashes(allowed_doc_hashes: Optional[set[str]]) -> Optional[Dict[str, Any]]:
        normalized_hashes = HybridRetriever._normalize_allowed_doc_hashes(allowed_doc_hashes)
        if not normalized_hashes:
            return None
        return {"doc_hash": {"$in": sorted(normalized_hashes)}}

    def _collection_records_by_doc_hash(self, collection, doc_hash: str) -> List[Dict[str, Any]]:
        normalized_hash = str(doc_hash or "").strip().lower()
        if not normalized_hash:
            return []
        where_clause = self._where_clause_for_doc_hashes({normalized_hash})
        try:
            records = self._collection_get_with_optional_where(collection, where_clause)
        except Exception as exc:
            logger.warning(f"Load records by doc_hash failed: {exc}")
            return []
        return [
            record for record in self._collection_records_to_list(records)
            if self._doc_hash_from_meta(record.get("metadata") or {}) == normalized_hash
        ]

    @staticmethod
    def _sort_section_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            records,
            key=lambda record: (
                HybridRetriever._as_int((record.get("metadata") or {}).get("section_number"), 10**6),
                HybridRetriever._as_int((record.get("metadata") or {}).get("start_page"), 10**6),
                str((record.get("metadata") or {}).get("section_title") or ""),
            ),
        )

    @staticmethod
    def _sort_element_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            records,
            key=lambda record: (
                HybridRetriever._as_int((record.get("metadata") or {}).get("position"), 10**6),
                HybridRetriever._as_int((record.get("metadata") or {}).get("chunk_index"), 10**6),
            ),
        )

    def _section_representative_text(
        self,
        section_record: Dict[str, Any],
        elements_by_section: Dict[str, List[Dict[str, Any]]],
        limit: int = 900,
    ) -> str:
        section_doc = str(section_record.get("document") or "").strip()
        if section_doc:
            return self._truncate_text(section_doc, limit)
        section_id = str((section_record.get("metadata") or {}).get("section_id") or section_record.get("id") or "")
        element_texts = [
            str(element.get("document") or "").strip()
            for element in self._sort_element_records(elements_by_section.get(section_id, []))
            if str(element.get("document") or "").strip()
        ]
        return self._truncate_text("\n".join(element_texts), limit)

    def _section_full_text(
        self,
        section_record: Dict[str, Any],
        elements_by_section: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        section_id = str((section_record.get("metadata") or {}).get("section_id") or section_record.get("id") or "").strip()
        element_texts = [
            str(element.get("document") or "").strip()
            for element in self._sort_element_records(elements_by_section.get(section_id, []))
            if str(element.get("document") or "").strip()
        ]
        if element_texts:
            return "\n".join(element_texts)
        return str(section_record.get("document") or "").strip()

    def _render_document_overview_packet(
        self,
        paper_record: Dict[str, Any],
        section_records: List[Dict[str, Any]],
        elements_by_section: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        paper_meta = dict(paper_record.get("metadata") or {})
        paper_doc = str(paper_record.get("document") or "").strip()
        title = str(paper_meta.get("title") or paper_record.get("id") or "未知文档").strip()
        abstract = str(paper_meta.get("abstract") or "").strip()
        if not abstract and paper_doc:
            abstract = self._truncate_text(paper_doc, 1000)

        header_lines = [
            "【文档级概述资料包】",
            f"标题: {title}",
            f"文档ID: {paper_record.get('id', '')}",
            f"doc_hash: {paper_meta.get('doc_hash', '')}",
        ]
        if paper_meta.get("page_count"):
            header_lines.append(f"页数: {paper_meta.get('page_count')}")
        if paper_meta.get("chunk_count"):
            header_lines.append(f"正文块数: {paper_meta.get('chunk_count')}")
        if abstract:
            header_lines.extend(["", "【摘要】", self._truncate_text(abstract, 1200)])

        structure_lines = ["", "【章节结构】"]
        for index, section in enumerate(section_records, start=1):
            meta = dict(section.get("metadata") or {})
            title_text = str(meta.get("section_hierarchy") or meta.get("section_title") or section.get("id") or "").strip()
            start_page = meta.get("start_page", "?")
            end_page = meta.get("end_page", start_page)
            page_text = f"第{start_page}页" if start_page == end_page else f"第{start_page}-{end_page}页"
            structure_lines.append(f"{index}. {title_text}（{page_text}，{meta.get('chunk_count', 0)}块）")

        section_content_lines = ["", "【各章节全部内容】"]
        for index, section in enumerate(section_records, start=1):
            meta = dict(section.get("metadata") or {})
            title_text = str(meta.get("section_hierarchy") or meta.get("section_title") or section.get("id") or "").strip()
            start_page = meta.get("start_page", "?")
            end_page = meta.get("end_page", start_page)
            page_text = f"第{start_page}页" if start_page == end_page else f"第{start_page}-{end_page}页"
            section_text = self._section_full_text(section, elements_by_section)
            if not section_text:
                continue
            section_content_lines.extend([
                "",
                f"### {index}. {title_text}",
                f"页码: {page_text}",
                section_text,
            ])

        return "\n".join(header_lines + structure_lines + section_content_lines).strip()

    def build_document_overview_context(self, doc_hash: str, query: str = "") -> Dict[str, Any]:
        normalized_hash = str(doc_hash or "").strip().lower()
        if not normalized_hash:
            return self._empty_response()

        paper_records = self._collection_records_by_doc_hash(self.paper_collection, normalized_hash)
        if not paper_records:
            payload = self._empty_response()
            payload["context"] = "未找到选中文档的层级索引内容"
            return payload
        paper_record = paper_records[0]

        section_records = [
            record for record in self._collection_records_by_doc_hash(self.section_collection, normalized_hash)
            if (
                not is_noise_section_metadata(record.get("metadata"))
                and not is_method_section_metadata(record.get("metadata"))
            )
        ]
        section_records = self._sort_section_records(section_records)
        allowed_section_ids = {
            str((record.get("metadata") or {}).get("section_id") or record.get("id") or "").strip()
            for record in section_records
        }
        allowed_section_ids = {section_id for section_id in allowed_section_ids if section_id}

        element_records = [
            record for record in self._collection_records_by_doc_hash(self.element_collection, normalized_hash)
        ]
        elements_by_section: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for element in element_records:
            section_id = str((element.get("metadata") or {}).get("section_id") or "").strip()
            if section_id in allowed_section_ids:
                elements_by_section[section_id].append(element)

        block = self._render_document_overview_packet(
            paper_record=paper_record,
            section_records=section_records,
            elements_by_section=elements_by_section,
        )
        paper_meta = dict(paper_record.get("metadata") or {})
        citation = {
            "type": "paper",
            "paper_id": paper_record.get("id", ""),
            "paper_title": paper_meta.get("title", "未知文献"),
            "title": paper_meta.get("title", "未知文献"),
            "section_title": "全文概述",
            "page": 1,
            "doc_hash": normalized_hash,
        }
        structured_sections = []
        for section in section_records:
            meta = dict(section.get("metadata") or {})
            structured_sections.append({
                "id": section.get("id", ""),
                "title": meta.get("section_title", ""),
                "hierarchy": meta.get("section_hierarchy", ""),
                "start_page": meta.get("start_page", ""),
                "end_page": meta.get("end_page", ""),
                "chunk_count": meta.get("chunk_count", 0),
            })
        context = ContextBuilder._assemble_final_context(query=query, context_parts=[block], citations=[citation])
        return {
            "context": context,
            "citations": [citation],
            "blocks": [block],
            "result_count": 1,
            "total_length": len(block),
            "structured": [{
                "level": "paper",
                "id": paper_record.get("id", ""),
                "title": paper_meta.get("title", ""),
                "abstract": paper_meta.get("abstract", ""),
                "doc_hash": normalized_hash,
                "sections": structured_sections,
                "elements": [],
            }],
        }

    def _retrieve_parent_sections(
        self,
        query: str,
        topn: int,
        allowed_doc_hashes: Optional[set[str]] = None,
        intent: Optional[str] = None,
    ) -> List[FusedResult]:
        """基于 element 命中构建父章节候选。

        topn 语义：控制召回深度（用于扩大 element 双路召回候选池），
        不等同于最终返回条数。
        """
        resolved_intent = self._normalize_intent(intent)
        if resolved_intent not in {"document_overview", "general_overview", "fact_qa"}:
            raise ValueError("意图分类失败：未提供有效intent")
        dense_weight, sparse_weight = self._fusion_weights_for_intent(resolved_intent, RetrievalLevel.ELEMENT.value)
        recall_depth = max(self._config_int("element_base_recall", DEFAULT_ELEMENT_CANDIDATE_POOL_SIZE), int(topn))
        element_dense, element_sparse = self._retrieve_with_optional_doc_filter(
            self.element_retriever,
            query,
            topn=recall_depth * 3,
            allowed_doc_hashes=allowed_doc_hashes,
        )
        child_hits = self.result_fuser._rrf_fuse(
            element_dense,
            element_sparse,
            level=RetrievalLevel.ELEMENT.value,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        if not child_hits:
            return []

        # 1) 子块排名百分比过滤：保留前40%
        high_quality_children = self._filter_children_by_rank_percent(child_hits)
        if not high_quality_children:
            return []

        parent_buckets: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "child_score_sum": 0.0,
                "best_child_score": 0.0,
                "children": [],
                "paper_id": "",
            }
        )
        for child in high_quality_children:
            parent_id = child.metadata.get("parent_id") or child.metadata.get("section_id")
            if not parent_id:
                continue
            bucket = parent_buckets[parent_id]
            bucket["child_score_sum"] += child.score
            bucket["best_child_score"] = max(bucket["best_child_score"], child.score)
            bucket["children"].append(child)
            if not bucket["paper_id"]:
                bucket["paper_id"] = child.metadata.get("paper_id", "")

        if not parent_buckets:
            return []

        section_records = self._fetch_collection_records(self.section_collection, list(parent_buckets.keys()))
        paper_ids = []
        for parent_id, record in section_records.items():
            record_meta = record.get("metadata") or {}
            paper_id = record_meta.get("paper_id") or record_meta.get("parent_id") or parent_buckets[parent_id]["paper_id"]
            if paper_id:
                paper_ids.append(paper_id)
        paper_records = self._fetch_collection_records(self.paper_collection, paper_ids)

        fused_results = []
        for parent_id, bucket in parent_buckets.items():
            parent_record = section_records.get(parent_id)
            if not parent_record:
                continue
            parent_meta = dict(parent_record.get("metadata") or {})

            # 过滤噪声章节（References 等）
            section_title = parent_meta.get("section_title", "")
            if self._is_noise_section(section_title):
                continue

            paper_id = parent_meta.get("paper_id") or parent_meta.get("parent_id") or bucket.get("paper_id", "")
            paper_meta = dict(paper_records.get(paper_id, {}).get("metadata") or {})
            if paper_meta:
                parent_meta.setdefault("title", paper_meta.get("title", ""))
                parent_meta.setdefault("authors", paper_meta.get("authors", ""))
                parent_meta.setdefault("venue", paper_meta.get("venue", ""))
                parent_meta.setdefault("year", paper_meta.get("year", ""))
            parent_meta["paper_id"] = paper_id or parent_meta.get("paper_id", "")
            parent_meta["matched_children"] = [
                {
                    "id": child.id,
                    "content": child.metadata.get("text_content") or child.content,
                    "page": child.metadata.get("page", 0),
                    "score": child.score,
                }
                for child in bucket["children"]
            ]
            parent_meta["matched_child_count"] = len(bucket["children"])
            parent_meta["retrieval_mode"] = "parent_child"
            parent_meta["child_score_sum"] = bucket["child_score_sum"]
            # 2) 父块融合分数 = 最高子块分数 + 0.3 * 子块平均分数
            parent_score, score_breakdown = self._compute_parent_fusion_score(
                bucket["children"],
                intent=resolved_intent,
            )
            section_score = parent_score
            parent_meta["parent_score_breakdown"] = score_breakdown
            fused_results.append(
                FusedResult(
                    result=RetrievalResult(
                        id=parent_id,
                        content=parent_record.get("document", ""),
                        score=section_score,
                        level="section",
                        metadata=parent_meta,
                    ),
                    fused_score=section_score,
                    source_scores={
                        "element_children": bucket["child_score_sum"],
                        "section": parent_score,
                        "best_child_score": score_breakdown["best_child_score"],
                        "avg_child_score": score_breakdown["avg_child_score"],
                    },
                )
            )

        if not fused_results:
            return []
        fused_results.sort(key=lambda item: item.fused_score, reverse=True)

        # 2) 父块初筛：保留 >= 最大融合分数 * 0.7
        before_ratio = self._config_float("parent_prescreen_threshold", RECOMMENDED_CONFIG["parent_prescreen_threshold"])
        max_parent_score = max(item.fused_score for item in fused_results)
        prefiltered_parents = [
            item for item in fused_results
            if item.fused_score >= max_parent_score * before_ratio
        ]
        if not prefiltered_parents:
            prefiltered_parents = fused_results[:1]

        # 3) 父块重排 + 阈值筛选
        reranked_parents = self._rerank_parent_candidates(
            query=query,
            parent_candidates=prefiltered_parents,
            intent=resolved_intent,
        )
        if not reranked_parents:
            return []
        if resolved_intent == "general_overview":
            after_ratio = self._config_float("parent_rerank_threshold_overview", 0.78)
        elif resolved_intent == "fact_qa":
            after_ratio = self._config_float("parent_rerank_threshold", 0.7)
        top_rerank_score = max(item.fused_score for item in reranked_parents)
        final_parents = [
            item for item in reranked_parents
            if item.fused_score >= top_rerank_score * after_ratio
        ]
        return final_parents or reranked_parents[:1]

    def _retrieve_child_candidates_for_fact(
        self,
        query: str,
        topn: int,
        allowed_doc_hashes: Optional[set[str]] = None,
    ) -> List[FusedResult]:
        dense_weight, sparse_weight = self._fusion_weights_for_intent("fact_qa", RetrievalLevel.ELEMENT.value)
        recall_depth = max(self._config_int("element_base_recall", DEFAULT_ELEMENT_CANDIDATE_POOL_SIZE), int(topn))
        element_dense, element_sparse = self._retrieve_with_optional_doc_filter(
            self.element_retriever,
            query,
            topn=recall_depth * 3,
            allowed_doc_hashes=allowed_doc_hashes,
        )
        child_hits = self.result_fuser._rrf_fuse(
            element_dense,
            element_sparse,
            level=RetrievalLevel.ELEMENT.value,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        if not child_hits:
            return []

        # 先让 reranker 看完整候选池，避免过早按 RRF 排名丢掉语义更准的子块。
        candidate_children = [
            child for child in child_hits
            if not self._is_noise_section((child.metadata or {}).get("section_title", ""))
        ]
        if not candidate_children:
            return []

        child_candidates: List[FusedResult] = [
            FusedResult(
                result=RetrievalResult(
                    id=child.id,
                    content=child.content,
                    score=float(child.score),
                    level=child.level,
                    metadata=child.metadata,
                ),
                fused_score=float(child.score),
                source_scores={"element_rrf": float(child.score)},
            )
            for child in candidate_children
        ]
        if not child_candidates:
            return []

        # 2) 子块重排后阈值筛选：>= top * 0.7
        if self.reranker:
            try:
                child_candidates = self._call_reranker(
                    query=query,
                    results=child_candidates,
                    top_k=None,
                    original_score_weight=self._config_float("fact_original_weight", 0.5),
                    rerank_score_weight=self._config_float("fact_rerank_weight", 0.5),
                )
            except Exception as exc:
                logger.warning(f"子块重排失败，使用融合排序: {exc}")
        child_candidates = sorted(child_candidates, key=lambda item: float(item.fused_score), reverse=True)
        if not child_candidates:
            return []
        rank_keep_ratio = self._config_float("child_rank_percentile_keep", RECOMMENDED_CONFIG["child_rank_percentile_keep"])
        rank_keep_ratio = min(max(rank_keep_ratio, 0.05), 1.0)
        rank_keep_count = max(1, int(math.ceil(len(child_candidates) * rank_keep_ratio)))
        child_candidates = child_candidates[:rank_keep_count]
        keep_ratio = self._config_float("child_rerank_threshold", 0.7)
        top_score = float(child_candidates[0].fused_score)
        filtered = [item for item in child_candidates if float(item.fused_score) >= top_score * keep_ratio]
        return filtered or child_candidates[:1]

    def _filter_children_by_rank_percent(self, child_hits: List[RetrievalResult]) -> List[RetrievalResult]:
        if not child_hits:
            return []
        keep_ratio = self._config_float("child_rank_percentile_keep", RECOMMENDED_CONFIG["child_rank_percentile_keep"])
        keep_ratio = min(max(keep_ratio, 0.05), 1.0)
        ranked = sorted(child_hits, key=lambda item: item.score, reverse=True)
        keep_count = max(1, int(math.ceil(len(ranked) * keep_ratio)))
        return ranked[:keep_count]

    def _select_high_quality_children(
        self,
        query: str,
        child_hits: List[RetrievalResult],
        intent: Optional[str] = None,
    ) -> List[RetrievalResult]:
        if not child_hits:
            return []
        fused_children = [
            FusedResult(
                result=child,
                fused_score=float(child.score),
                source_scores={"element": float(child.score)},
            )
            for child in sorted(child_hits, key=lambda item: item.score, reverse=True)
        ]
        if hasattr(self, "_rerank_results"):
            reranked = self._rerank_results(query, fused_children, top_k=None, intent=intent)
        elif self.reranker:
            reranked = self._call_reranker(query=query, results=fused_children, top_k=None)
        else:
            reranked = fused_children
        return [item.result for item in reranked[:15]]

    @staticmethod
    def _is_document_summary_query(query: str) -> bool:
        text = (query or "").strip().lower()
        if not text:
            return False
        explicit_markers = ("整篇", "全文", "全篇", "整份", "主要贡献", "总体概述")
        local_markers = ("第", "页", "章节", "方法设计", "结果部分", "摘要里")
        if any(marker in text for marker in local_markers):
            return False
        return any(marker in text for marker in explicit_markers)

    def _compute_parent_section_score(
        self,
        children: List[RetrievalResult],
        intent: Optional[str] = None,
    ) -> Tuple[float, Dict[str, float]]:
        if not children:
            return 0.0, {
                "best_child_score": 0.0,
                "avg_child_score": 0.0,
                "support_bonus": 0.0,
                "count_bonus": 0.0,
            }
        config = getattr(self, "config", {}) or {}
        resolved_intent = self._normalize_intent(intent)
        child_scores = [max(float(child.score), 0.0) for child in children]
        best_child_score = max(child_scores)
        avg_child_score = sum(child_scores) / len(child_scores)
        support_weight = self._config_float("support_bonus_weight", 0.3)
        count_weight = self._config_float("child_count_bonus_weight", 0.15)
        if resolved_intent == "fact_qa":
            support_weight = self._config_float("support_bonus_weight_fact", support_weight)
            count_weight = self._config_float("child_count_bonus_weight_fact", count_weight)
        elif resolved_intent == "general_overview":
            support_weight = self._config_float("support_bonus_weight_overview", support_weight)
            count_weight = self._config_float("child_count_bonus_weight_overview", count_weight)
        support_bonus = support_weight * avg_child_score
        count_bonus = count_weight * max(len(child_scores) - 1, 0)
        parent_score = best_child_score + support_bonus + count_bonus
        return parent_score, {
            "best_child_score": best_child_score,
            "avg_child_score": avg_child_score,
            "support_bonus": support_bonus,
            "count_bonus": count_bonus,
        }

    def _compute_parent_fusion_score(
        self,
        children: List[RetrievalResult],
        intent: Optional[str] = None,
    ) -> Tuple[float, Dict[str, float]]:
        return self._compute_parent_section_score(children, intent=intent)

    def _rerank_parent_candidates(
        self,
        query: str,
        parent_candidates: List[FusedResult],
        intent: Optional[str],
    ) -> List[FusedResult]:
        if not parent_candidates:
            return []
        if not self.reranker:
            return sorted(parent_candidates, key=lambda item: item.fused_score, reverse=True)

        config = getattr(self, "config", {}) or {}
        resolved_intent = (intent or "").strip().lower() if isinstance(intent, str) else ""
        if resolved_intent == "fact_qa":
            original_weight = self._config_float("fact_original_weight", 0.5)
            model_weight = self._config_float("fact_rerank_weight", 0.5)
        else:
            original_weight = self._config_float("overview_original_weight", 0.3)
            model_weight = self._config_float("overview_rerank_weight", 0.7)
        try:
            reranked = self._call_reranker(
                query=query,
                results=parent_candidates,
                top_k=None,
                original_score_weight=original_weight,
                rerank_score_weight=model_weight,
            )
        except Exception as exc:
            logger.warning(f"父块重排失败，使用原排序: {exc}")
            reranked = parent_candidates
        reranked.sort(key=lambda item: item.fused_score, reverse=True)
        return reranked

    def _call_reranker(
        self,
        *,
        query: str,
        results: List[FusedResult],
        top_k: Optional[int],
        original_score_weight: float = 0.3,
        rerank_score_weight: float = 0.7,
    ) -> List[FusedResult]:
        if not self.reranker:
            return results[:top_k] if top_k else results
        try:
            return self.reranker.rerank(
                query=query,
                results=results,
                top_k=top_k,
                original_score_weight=original_score_weight,
                rerank_score_weight=rerank_score_weight,
            )
        except TypeError:
            return self.reranker.rerank(query, results, top_k=top_k)

    def retrieve_parent_candidates(
        self,
        query: str,
        topn: int,
        allowed_doc_hashes: Optional[set[str]],
        intent: str,
    ) -> List[FusedResult]:
        resolved_intent = self._normalize_intent(intent)
        if resolved_intent == "document_overview":
            raise ValueError("document_overview must use build_document_overview_context with exactly one doc_hash")
        return self._retrieve_parent_sections(
            query=query,
            topn=topn,
            allowed_doc_hashes=allowed_doc_hashes,
            intent=resolved_intent,
        )

    def retrieve_child_candidates(
        self,
        query: str,
        topn: int,
        allowed_doc_hashes: Optional[set[str]],
        intent: str,
    ) -> List[FusedResult]:
        resolved_intent = self._normalize_intent(intent)
        if resolved_intent != "fact_qa":
            return []
        return self._retrieve_child_candidates_for_fact(
            query=query,
            topn=topn,
            allowed_doc_hashes=allowed_doc_hashes,
        )

    def retrieve(self, query: str, topn: int = DEFAULT_HYBRID_RETRIEVE_TOP_N, allowed_doc_hashes: Optional[set[str]] = None, intent: Optional[str] = None) -> Dict[str, Any]:
        """统一检索入口。

        topn 语义：召回阶段的候选深度；最终输出数量由 parent_rerank_top_k 控制。
        """
        allowed_doc_hashes = self._normalize_allowed_doc_hashes(allowed_doc_hashes)
        parent_rerank_top_k = max(self._config_int("parent_rerank_top_k", DEFAULT_PARENT_RERANK_TOP_K), 1)
        resolved_intent = self._normalize_intent(intent)
        if resolved_intent not in {"document_overview", "general_overview", "fact_qa"}:
            raise ValueError("意图分类失败：未提供有效intent")

        if resolved_intent == "document_overview":
            if not allowed_doc_hashes or len(allowed_doc_hashes) != 1:
                raise ValueError("document_overview requires exactly one selected doc_hash")
            doc_hash = next(iter(allowed_doc_hashes))
            payload = self.build_document_overview_context(doc_hash=doc_hash, query=query)
            logger.info(f"Hybrid retrieve resolved by selected document: {doc_hash}")
            return payload

        if resolved_intent == "fact_qa":
            child_candidates = self.retrieve_child_candidates(
                query=query,
                topn=topn,
                allowed_doc_hashes=allowed_doc_hashes,
                intent=resolved_intent,
            )
            if child_candidates:
                child_candidates = sorted(child_candidates, key=lambda item: float(item.fused_score), reverse=True)
                keep_ratio = self._config_float("dynamic_score_threshold", 0.7)
                min_keep = self._config_int("dynamic_min_keep_fact", 1)
                max_keep = self._config_int("dynamic_max_keep_fact", 6)
                top_score = float(child_candidates[0].fused_score)
                selected = [item for item in child_candidates if float(item.fused_score) >= top_score * keep_ratio]
                if len(selected) < min_keep:
                    selected = child_candidates[: min(min_keep, len(child_candidates))]
                if len(selected) > max_keep:
                    selected = selected[:max_keep]
                logger.info(f"Hybrid retrieve resolved at child-element level: {len(selected)}")
                return self.context_builder.build(selected, query=query)
            return self._empty_response()

        parent_section_results = self.retrieve_parent_candidates(
            query=query,
            topn=topn,
            allowed_doc_hashes=allowed_doc_hashes,
            intent=resolved_intent,
        )
        if parent_section_results:
            parent_section_results.sort(key=lambda item: item.fused_score, reverse=True)
            parent_section_results = parent_section_results[:parent_rerank_top_k]
            logger.info(f"Hybrid retrieve resolved at parent-section level: {len(parent_section_results)}")
            return self.context_builder.build(parent_section_results, query=query)
        return self._empty_response()


class BaseCollectionRetriever:
    """通用检索器 稠密检索 + 稀疏检索(BM25)"""

    level = ""
    collection_label = ""
    default_top_k = DEFAULT_PAPER_RETRIEVER_TOP_K

    def __init__(self, collection, embedding_fn, enable_sparse: bool = True, persist_directory: Optional[str] = None, tokenizer: Any = None):
        self.collection = collection
        self.embedding_fn = embedding_fn
        self.enable_sparse = bool(enable_sparse)
        self.persist_directory = persist_directory
        self.tokenizer = tokenizer
        self.bm25: Any = None
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_metadatas: List[Dict[str, Any]] = []
        if self.enable_sparse:
            self._build_bm25_index()

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        if self._is_chinese_text(text):
            import jieba
            return list(jieba.cut(text))
        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(text)
            return [t for t in tokens if t.strip()]
        return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", text.lower())

    @staticmethod
    def _detect_chinese_ratio(text: str) -> float:
        if not text:
            return 0.0
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        total_chars = len(re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]", text))
        if total_chars == 0:
            return 0.0
        return len(chinese_chars) / total_chars

    def _is_chinese_text(self, text: str, threshold: float = 0.3) -> bool:
        return self._detect_chinese_ratio(text) >= threshold

    def _metadata_at(self, idx: int) -> Dict[str, Any]:
        if idx < len(self.doc_metadatas) and isinstance(self.doc_metadatas[idx], dict):
            return self.doc_metadatas[idx]
        return {}

    def _where_clause_for_doc_hashes(self, allowed_doc_hashes: Optional[set[str]]) -> Optional[Dict[str, Any]]:
        if not allowed_doc_hashes:
            return None
        normalized_hashes = sorted({
            str(doc_hash).strip().lower()
            for doc_hash in allowed_doc_hashes
            if isinstance(doc_hash, str) and str(doc_hash).strip()
        })
        if not normalized_hashes:
            return None
        return {"doc_hash": {"$in": normalized_hashes}}

    def _build_bm25_index(self):
        cache_path = None
        if self.persist_directory:
            cache_path = os.path.join(self.persist_directory, f"bm25_cache_{self.collection_label}.pkl")

        # 尝试从磁盘加载缓存
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)

                # 检查缓存有效性（通过文档总数对比）
                current_count = self.collection.count()
                if len(cache_data.get("doc_ids", [])) == current_count:
                    self.bm25 = cache_data.get("bm25")
                    self.doc_ids = cache_data.get("doc_ids", [])
                    self.doc_texts = cache_data.get("doc_texts", [])
                    self.doc_metadatas = cache_data.get("doc_metadatas", [])
                    if self.bm25:
                        logger.info(f"Loaded BM25 index from cache for {self.collection_label}: {len(self.doc_ids)} documents")
                        return
                else:
                    logger.info(f"BM25 cache for {self.collection_label} is stale (count {len(cache_data.get('doc_ids', []))} != {current_count}), rebuilding...")
            except Exception as e:
                logger.warning(f"Failed to load BM25 cache for {self.collection_label}: {e}")

        # 重新构建索引
        try:
            results = self.collection.get()
            if results and results.get("documents"):
                self.doc_ids = [str(item) for item in (results.get("ids") or []) if item is not None]
                self.doc_texts = [str(item) for item in (results.get("documents") or []) if item is not None]
                self.doc_metadatas = [
                    dict(item) for item in (results.get("metadatas") or []) if isinstance(item, dict)
                ]
                tokenized = [self._tokenize(text) for text in self.doc_texts]
                self.bm25 = BM25Okapi(tokenized)
                logger.info(f"Built BM25 index for {self.collection_label} collection: {len(self.doc_texts)} documents")

                # 保存到磁盘
                if cache_path:
                    try:
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        with open(cache_path, "wb") as f:
                            pickle.dump({
                                "bm25": self.bm25,
                                "doc_ids": self.doc_ids,
                                "doc_texts": self.doc_texts,
                                "doc_metadatas": self.doc_metadatas,
                            }, f)
                        logger.info(f"Saved BM25 index to cache for {self.collection_label}")
                    except Exception as e:
                        logger.warning(f"Failed to save BM25 cache for {self.collection_label}: {e}")
        except Exception as e:
            logger.warning(f"Failed to build BM25 index for {self.collection_label}s: {e}")

    def _dense_search(
        self,
        query: str,
        topn: int,
        allowed_doc_hashes: Optional[set[str]] = None,
    ) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        try:
            where_clause = self._where_clause_for_doc_hashes(allowed_doc_hashes)
            query_results = self.collection.query(
                query_embeddings=self.embedding_fn([query]),
                n_results=topn,
                where=where_clause,
            )
            ids = (query_results.get("ids") or [[]])[0]
            docs = (query_results.get("documents") or [[]])[0]
            metas = (query_results.get("metadatas") or [[]])[0]
            distances = (query_results.get("distances") or [[]])[0]
            allowed_hashes = None
            if allowed_doc_hashes:
                allowed_hashes = {
                    str(doc_hash).strip().lower()
                    for doc_hash in allowed_doc_hashes
                    if isinstance(doc_hash, str) and str(doc_hash).strip()
                }

            for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
                if not doc:
                    continue
                if is_noise_section_metadata(meta or {}):
                    continue
                if allowed_hashes is not None:
                    meta_hash = str((meta or {}).get("doc_hash") or "").strip().lower()
                    if meta_hash not in allowed_hashes:
                        continue
                score = 1.0 / (dist + 1e-8) if dist > 0 else 1.0
                results.append(
                    RetrievalResult(
                        id=doc_id,
                        content=doc,
                        score=score,
                        level=self.level,
                        metadata=meta or {},
                    )
                )
        except Exception as e:
            logger.warning(f"{self.collection_label.capitalize()} dense search failed: {e}")
        return results

    def _sparse_search(
        self,
        query: str,
        topn: int,
        allowed_doc_hashes: Optional[set[str]] = None,
    ) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        if not self.enable_sparse:
            return results
        if not self.bm25 or not self.doc_texts:
            return results

        try:
            query_tokens = self._tokenize(query)
            if not query_tokens:
                return results

            candidate_indices = list(range(len(self.doc_texts)))
            if allowed_doc_hashes:
                normalized_hashes = {
                    str(doc_hash).strip().lower()
                    for doc_hash in allowed_doc_hashes
                    if isinstance(doc_hash, str) and str(doc_hash).strip()
                }
                candidate_indices = [
                    idx
                    for idx in candidate_indices
                    if str(self._metadata_at(idx).get("doc_hash") or "").strip().lower() in normalized_hashes
                ]
            if not candidate_indices:
                return results

            subset_texts = [self.doc_texts[idx] for idx in candidate_indices]
            subset_tokenized = [self._tokenize(text) for text in subset_texts]
            subset_bm25 = BM25Okapi(subset_tokenized)
            subset_scores = subset_bm25.get_scores(query_tokens)
            top_local_indices = sorted(range(len(subset_scores)), key=lambda i: subset_scores[i], reverse=True)[:topn]
            for local_idx in top_local_indices:
                score = float(subset_scores[local_idx])
                if score <= 0:
                    continue
                global_idx = candidate_indices[local_idx]
                meta = self._metadata_at(global_idx)
                if is_noise_section_metadata(meta):
                    continue
                results.append(
                    RetrievalResult(
                        id=self.doc_ids[global_idx],
                        content=self.doc_texts[global_idx],
                        score=score,
                        level=self.level,
                        metadata=meta,
                    )
                )
        except Exception as e:
            logger.warning(f"{self.collection_label.capitalize()} sparse search failed: {e}")
        return results

    def retrieve(
        self,
        query: str,
        topn: Optional[int] = None,
        allowed_doc_hashes: Optional[set[str]] = None,
    ) -> Tuple[List[RetrievalResult], List[RetrievalResult]]:
        resolved_topn = topn or self.default_top_k
        return (
            self._dense_search(query, resolved_topn, allowed_doc_hashes=allowed_doc_hashes),
            self._sparse_search(query, resolved_topn, allowed_doc_hashes=allowed_doc_hashes),
        )


class PaperRetriever(BaseCollectionRetriever):
    """论文层检索器 - 仅稠密检索（不构建 BM25）"""

    level = RetrievalLevel.PAPER.value
    collection_label = "paper"
    default_top_k = DEFAULT_PAPER_RETRIEVER_TOP_K

    def __init__(self, collection, embedding_fn, persist_directory: Optional[str] = None, tokenizer: Any = None):
        super().__init__(collection, embedding_fn, enable_sparse=False, persist_directory=persist_directory, tokenizer=tokenizer)


class ElementRetriever(BaseCollectionRetriever):
    """要素层检索器 - 支持稠密检索 + 稀疏检索(BM25)"""

    level = RetrievalLevel.ELEMENT.value
    collection_label = "element"
    default_top_k = DEFAULT_ELEMENT_RETRIEVER_TOP_K

    def __init__(self, collection, embedding_fn, persist_directory: Optional[str] = None, tokenizer: Any = None):
        super().__init__(collection, embedding_fn, enable_sparse=True, persist_directory=persist_directory, tokenizer=tokenizer)


class HierarchicalChromaRetriever:
    """分层 Chroma 检索器。"""

    def __init__(
        self,
        persist_directory: str,
        embedding_model_name_or_path: str,
        device: str = "cpu",
        rerank_model_name_or_path: str = DEFAULT_RERANK_MODEL_NAME,
    ):
        self.persist_directory = persist_directory
        self.embedding_model_name_or_path = embedding_model_name_or_path
        self.device = device
        self.rerank_model_name_or_path = rerank_model_name_or_path
        self.retriever = self._create_retriever()

    def _create_retriever(self) -> HybridRetriever:
        return HybridRetriever(
            persist_directory=self.persist_directory,
            embedding_model_name_or_path=self.embedding_model_name_or_path,
            device=self.device,
            rerank_model_name_or_path=self.rerank_model_name_or_path,
        )

    def refresh_indexes(self):
        self.retriever.refresh_indexes()

    @staticmethod
    def _normalize_collection_values(values):
        if values is None:
            return []
        if hasattr(values, "tolist"):
            return values.tolist()
        if isinstance(values, tuple):
            return list(values)
        return values

    def _iter_named_collections(self):
        return (
            ("paper", self.retriever.paper_collection),
            ("section", self.retriever.section_collection),
            ("element", self.retriever.element_collection),
        )

    def _replace_collection_contents(self, collection, data: Dict[str, Any]):
        existing = collection.get()
        existing_ids = self._normalize_collection_values(existing.get("ids")) if existing else []
        if existing_ids:
            collection.delete(ids=existing_ids)
        ids = self._normalize_collection_values(data.get("ids"))
        docs = self._normalize_collection_values(data.get("documents"))
        metas = self._normalize_collection_values(data.get("metadatas"))
        embeddings = self._normalize_collection_values(data.get("embeddings"))
        if ids:
            resolved_embeddings = embeddings if embeddings else None
            collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=resolved_embeddings)

    def build_element_metadata_index(self) -> Dict[str, Dict[str, Any]]:
        results = self.retriever.element_collection.get(include=["metadatas"])
        ids = self._normalize_collection_values(results.get("ids"))
        metadatas = self._normalize_collection_values(results.get("metadatas"))
        return {
            str(record_id): dict(meta or {})
            for record_id, meta in zip(ids, metadatas)
            if record_id is not None
        }

    def get_index_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for level, collection in self._iter_named_collections():
            result = collection.get()
            counts[level] = len(self._normalize_collection_values(result.get("ids")))
        return counts

    def has_indexed_content(self) -> bool:
        return self.get_index_counts().get("element", 0) > 0

    def save_persist_directory(self, persist_directory: str):
        if not persist_directory:
            return
        target_dir = os.path.abspath(persist_directory)
        current_dir = os.path.abspath(self.persist_directory)
        if target_dir == current_dir:
            self.refresh_indexes()
            return
        snapshot = {
            level: collection.get(include=["documents", "metadatas", "embeddings"])
            for level, collection in self._iter_named_collections()
        }
        self.switch_persist_directory(target_dir)
        for level, collection in self._iter_named_collections():
            self._replace_collection_contents(collection, snapshot.get(level, {}))
        self.refresh_indexes()

    def load_persist_directory(self, persist_directory: str):
        if not persist_directory:
            return
        self.switch_persist_directory(persist_directory)

    def switch_persist_directory(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.retriever = self._create_retriever()

    def index_document(self, doc_hash: str, doc_file: str, chunk_items: List[Dict[str, Any]]):
        payloads = build_hierarchical_document_payloads(doc_hash=doc_hash, doc_file=doc_file, chunk_items=chunk_items)
        for level in ("paper", "section", "element"):
            level_payload = payloads[level]
            if not level_payload["ids"]:
                continue
            target_collection = getattr(self.retriever, f"{level}_collection", None)
            self.retriever.upsert_collection(
                target_collection,
                ids=level_payload["ids"],
                documents=level_payload["documents"],
                metadatas=level_payload["metadatas"],
            )
        self.refresh_indexes()

    def retrieve(self, query: str, topn: int = DEFAULT_HYBRID_RETRIEVE_TOP_N, allowed_doc_hashes: Optional[set[str]] = None, intent: Optional[str] = None) -> Dict[str, Any]:
        return self.retriever.retrieve(query, topn, allowed_doc_hashes=allowed_doc_hashes, intent=intent)

    def build_document_overview_context(self, doc_hash: str, query: str = "") -> Dict[str, Any]:
        return self.retriever.build_document_overview_context(doc_hash=doc_hash, query=query)

    def retrieve_parent_candidates(
        self,
        query: str,
        topn: int = DEFAULT_HYBRID_RETRIEVE_TOP_N,
        allowed_doc_hashes: Optional[set[str]] = None,
        intent: Optional[str] = None,
    ) -> List[FusedResult]:
        resolved_intent = (intent or "").strip().lower() if isinstance(intent, str) else ""
        return self.retriever.retrieve_parent_candidates(
            query=query,
            topn=topn,
            allowed_doc_hashes=allowed_doc_hashes,
            intent=resolved_intent,
        )

    def retrieve_child_candidates(
        self,
        query: str,
        topn: int = DEFAULT_HYBRID_RETRIEVE_TOP_N,
        allowed_doc_hashes: Optional[set[str]] = None,
        intent: Optional[str] = None,
    ) -> List[FusedResult]:
        resolved_intent = (intent or "").strip().lower() if isinstance(intent, str) else ""
        return self.retriever.retrieve_child_candidates(
            query=query,
            topn=topn,
            allowed_doc_hashes=allowed_doc_hashes,
            intent=resolved_intent,
        )

    def build_context_from_parents(self, parents: List[FusedResult], query: str) -> Dict[str, Any]:
        if not parents:
            return self.retriever._empty_response()
        return self.retriever.context_builder.build(parents, query=query)

    def build_context_from_fused_results(self, fused_results: List[FusedResult], query: str) -> Dict[str, Any]:
        if not fused_results:
            return self.retriever._empty_response()
        return self.retriever.context_builder.build(fused_results, query=query)
