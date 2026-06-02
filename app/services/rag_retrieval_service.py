from typing import Any, List, Optional, Set

from loguru import logger

from app.core.rag_defaults import (
    DEFAULT_ELEMENT_CANDIDATE_POOL_SIZE,
    DEFAULT_FINAL_CONTEXT_TOP_K,
    DEFAULT_PARENT_RERANK_TOP_K,
    DEFAULT_SUB_QUERY_SCORE_BOOST,
)
from app.core.rag_hash_utils import file_md5_hex_32
from app.core.rag_common import add_source_numbers
from app.core.hierarchical_types import FusedResult


class DocumentSelectionRequiredError(RuntimeError):
    DEFAULT_MESSAGE = "请先在右侧文档列表选择一篇要概述的文档，然后再让我总结整篇论文。"
    SINGLE_DOCUMENT_MESSAGE = "文档级概述一次只能选择一篇文档，请只保留一篇后再试。"

    def __init__(self, message: str | None = None):
        super().__init__(message or self.DEFAULT_MESSAGE)


class RagRetrievalService:
    HIERARCHICAL_RETRIEVE_TOP_N = DEFAULT_ELEMENT_CANDIDATE_POOL_SIZE
    FINAL_QUERY_TOP_K = DEFAULT_PARENT_RERANK_TOP_K
    FINAL_CONTEXT_TOP_K = DEFAULT_FINAL_CONTEXT_TOP_K
    SUB_QUERY_HIT_BOOST = DEFAULT_SUB_QUERY_SCORE_BOOST
    VALID_INTENTS = {"document_overview", "general_overview", "fact_qa"}

    def __init__(self, rag):
        self.rag = rag
        self._latest_hit_scores: dict[str, float] = {}
        self.strategy = getattr(rag, "retrieval_strategy", {}) or {}

    def _get_strategy_number(self, key: str, default: float) -> float:
        value = self.strategy.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _get_dynamic_selection(self, intent: str, default_keep: float, default_min: int, default_max: int) -> tuple[float, int, int]:
        dynamic = (self.strategy or {}).get("dynamic_selection", {}) or {}
        intent_cfg = dynamic.get(intent, {}) or {}
        keep_ratio = intent_cfg.get("keep_ratio", default_keep)
        min_keep = intent_cfg.get("min_keep", default_min)
        max_keep = intent_cfg.get("max_keep", default_max)
        try:
            return float(keep_ratio), int(min_keep), int(max_keep)
        except (TypeError, ValueError):
            return float(default_keep), int(default_min), int(default_max)

    @staticmethod
    def _extract_retrieval_data(ret_dict) -> List[dict]:
        blocks = ret_dict.get("blocks", [])
        citations = ret_dict.get("citations", [])
        results = []
        for i, block in enumerate(blocks):
            if isinstance(block, str) and block.strip():
                citation = citations[i] if i < len(citations) else {}
                results.append({
                    "text": block.strip(),
                    "metadata": citation
                })

        if results:
            return results

        context = ret_dict.get("context", "").strip()
        no_result_markers = ("未找到相关内容", "无该类文章")
        if (
            context
            and not any(marker in context for marker in no_result_markers)
            and "相关文献内容" not in context
        ):
            return [{"text": context, "metadata": {}}]
        return []

    def build_prompt_and_references(
        self,
        query: str,
        context_len: int,
        selected_doc_paths=None,
        trace_ctx=None,
        history=None,
        history_summary=None,
        intent=None,
    ) -> tuple[str, List[str]]:
        reference_data = self.get_reference_results(
            query,
            selected_doc_paths=selected_doc_paths,
            trace_ctx=trace_ctx,
            history=history,
            history_summary=history_summary,
            intent=intent,
        )
        if reference_data:
            formatted_references = add_source_numbers(reference_data)
            context_str = '\n'.join(formatted_references)[:(context_len - len(self.rag.prompt_template))]
        else:
            context_str = ''
            formatted_references = []
        return self.rag.prompt_template.format(context_str=context_str, query_str=query), formatted_references

    @staticmethod
    def _normalize_selected_doc_paths(selected_doc_paths) -> list[str]:
        if isinstance(selected_doc_paths, str):
            return [selected_doc_paths]
        if not selected_doc_paths:
            return []
        return [
            file_path for file_path in selected_doc_paths
            if isinstance(file_path, str) and file_path.strip()
        ]

    @staticmethod
    def _resolve_allowed_doc_hashes(selected_doc_paths) -> Optional[Set[str]]:
        selected_paths = RagRetrievalService._normalize_selected_doc_paths(selected_doc_paths)
        if not selected_paths:
            return None
        allowed_hashes: Set[str] = set()
        for file_path in selected_paths:
            try:
                allowed_hashes.add(file_md5_hex_32(file_path))
            except Exception as exc:
                logger.warning(f"skip selected file hash due to error: {file_path}, {exc}")
        return allowed_hashes or None

    @staticmethod
    def _count_selected_doc_paths(selected_doc_paths) -> int:
        return len(RagRetrievalService._normalize_selected_doc_paths(selected_doc_paths))

    @staticmethod
    def _require_single_document_selection(selected_count: int, allowed_doc_hashes: Optional[Set[str]]) -> str:
        if selected_count <= 0 or not allowed_doc_hashes:
            raise DocumentSelectionRequiredError()
        if selected_count != 1 or len(allowed_doc_hashes) != 1:
            raise DocumentSelectionRequiredError(DocumentSelectionRequiredError.SINGLE_DOCUMENT_MESSAGE)
        return next(iter(allowed_doc_hashes))

    def try_hierarchical_retrieve(self, query: str, allowed_doc_hashes: Optional[Set[str]] = None, intent: str | None = None) -> List[dict]:
        ret_dict = self.rag.hierarchical_retriever.retrieve(
            query=query,
            topn=int(self._get_strategy_number("element_candidate_pool_size", self.HIERARCHICAL_RETRIEVE_TOP_N)),
            allowed_doc_hashes=allowed_doc_hashes,
            intent=intent,
        )
        self._latest_hit_scores = self._extract_retrieval_scores(ret_dict)
        return self._extract_retrieval_data(ret_dict)

    @staticmethod
    def _extract_retrieval_scores(ret_dict) -> dict[str, float]:
        if not isinstance(ret_dict, dict):
            return {}
        structured = ret_dict.get("structured", []) or []
        section_scores: dict[str, float] = {}
        for paper in structured:
            if not isinstance(paper, dict):
                continue
            for section in (paper.get("sections") or []):
                if not isinstance(section, dict):
                    continue
                section_id = str(section.get("id", "")).strip()
                if section_id:
                    section_scores[section_id] = float(section.get("score", 0) or 0)

        score_map: dict[str, float] = {}
        blocks = ret_dict.get("blocks", []) or []
        citations = ret_dict.get("citations", []) or []
        for idx, block in enumerate(blocks):
            if not isinstance(block, str):
                continue
            text = block.strip()
            if not text:
                continue
            score = 0.0
            if idx < len(citations) and isinstance(citations[idx], dict):
                sec_id = str(citations[idx].get("section_id", "")).strip()
                score = section_scores.get(sec_id, 0.0)
            score_map[text] = max(score_map.get(text, 0.0), score)
        return score_map

    def _fuse_multi_query_hits(self, all_query_hits: List[List[dict]], all_query_scores: List[dict[str, float]]) -> List[dict]:
        merged: dict[str, dict[str, Any]] = {}
        first_seen_order = 0
        for hits, score_map in zip(all_query_hits, all_query_scores):
            seen_in_query: set[str] = set()
            query_top_k = int(self._get_strategy_number("parent_rerank_top_k", self.FINAL_QUERY_TOP_K))
            for hit in hits[:query_top_k]:
                text = hit["text"]
                metadata = hit["metadata"]
                dedupe_key = self._build_section_dedupe_key(metadata=metadata, text=text)

                normalized = text.strip()
                if not normalized:
                    continue
                if dedupe_key not in merged:
                    merged[dedupe_key] = {
                        "text": text,
                        "metadata": metadata,
                        "hit_count": 0,
                        "section_score": 0.0,
                        "first_order": first_seen_order,
                    }
                    first_seen_order += 1
                if dedupe_key not in seen_in_query:
                    merged[dedupe_key]["hit_count"] += 1
                    seen_in_query.add(dedupe_key)
                merged[dedupe_key]["section_score"] = max(
                    merged[dedupe_key]["section_score"],
                    score_map.get(normalized, 0.0),
                )

        final_context_top_k = int(self._get_strategy_number("final_context_top_k", self.FINAL_CONTEXT_TOP_K))
        if len(merged) <= final_context_top_k:
            ordered = sorted(merged.values(), key=lambda p: p["first_order"])
            return [{"text": p["text"], "metadata": p["metadata"]} for p in ordered]

        hit_boost = self._get_strategy_number("sub_query_score_boost", self.SUB_QUERY_HIT_BOOST)
        ranked = []
        for payload in merged.values():
            fused_score = payload["section_score"] * (1.0 + payload["hit_count"] * hit_boost)
            ranked.append((payload, fused_score))

        ranked.sort(key=lambda item: (item[1], -item[0]["first_order"]), reverse=True)
        return [{"text": p["text"], "metadata": p["metadata"]} for p, _ in ranked[:final_context_top_k]]

    @staticmethod
    def _build_section_dedupe_key(metadata: dict | None, text: str) -> str:
        meta = metadata if isinstance(metadata, dict) else {}
        section_id = str(meta.get("section_id") or "").strip()
        if section_id:
            return f"section_id:{section_id}"
        paper_title = str(meta.get("paper_title") or "").strip().lower()
        section_title = str(meta.get("section_title") or "").strip().lower()
        page = str(meta.get("page") or "").strip()
        if section_title:
            return f"section:{paper_title}|{section_title}|{page}"
        return f"text:{text.strip().lower()}"

    def _classify_intent_once(self, query: str, intent: str | None = None) -> str:
        normalized = (intent or "").strip().lower() if isinstance(intent, str) else ""
        if normalized in self.VALID_INTENTS:
            return normalized
        return self.rag.query_expander.classify_intent(query)

    def _expand_query_with_optional_history(self, query: str, history=None, history_summary=None) -> List[str]:
        try:
            return self.rag._expand_query(
                query,
                history=history,
                history_summary=history_summary,
            )
        except TypeError:
            return self.rag._expand_query(query)

    def get_reference_results(
        self,
        query: str,
        selected_doc_paths=None,
        trace_ctx=None,
        history=None,
        history_summary=None,
        intent=None,
    ):
        trace_ctx = trace_ctx or {}
        selected_count = self._count_selected_doc_paths(selected_doc_paths)
        allowed_doc_hashes = self._resolve_allowed_doc_hashes(selected_doc_paths)
        classified_intent = self._classify_intent_once(query, intent=intent)
        if classified_intent == "document_overview":
            try:
                doc_hash = self._require_single_document_selection(selected_count, allowed_doc_hashes)
            except DocumentSelectionRequiredError:
                logger.bind(
                    session_id=trace_ctx.get("session_id", ""),
                    query_id=trace_ctx.get("query_id", ""),
                    intent=classified_intent,
                ).info("document_selection_required")
                raise

        logger.bind(
            session_id=trace_ctx.get("session_id", ""),
            query_id=trace_ctx.get("query_id", ""),
            intent=classified_intent,
        ).info("retrieval_started")
        if classified_intent == "document_overview":
            return self._run_document_overview(query=query, doc_hash=doc_hash)

        expanded_queries = self._expand_query_with_optional_history(
            query,
            history=history,
            history_summary=history_summary,
        )

        if classified_intent == "fact_qa":
            return self._run_fused_pipeline(
                expanded_queries, allowed_doc_hashes, classified_intent,
                retriever="retrieve_child_candidates",
                builder="build_context_from_fused_results",
                intent_label="fact_qa",
            )

        return self._run_fused_pipeline(
            expanded_queries, allowed_doc_hashes, classified_intent,
            retriever="retrieve_parent_candidates",
            builder="build_context_from_parents",
            intent_label="general_overview",
        )

    def _run_document_overview(self, query: str, doc_hash: str):
        payload = self.rag.hierarchical_retriever.build_document_overview_context(
            doc_hash=doc_hash,
            query=query,
        )
        self._latest_hit_scores = {}
        return self._extract_retrieval_data(payload)

    def _run_fused_pipeline(self, expanded_queries, allowed_doc_hashes, intent, *, retriever, builder, intent_label):
        default_keep, min_k, max_k = {"fact_qa": (0.7, 1, 6), "general_overview": (0.65, 2, 6)}[intent_label]
        per_query_lists: List[List[FusedResult]] = []
        for eq in expanded_queries:
            candidates = self._safe_retrieve(eq, allowed_doc_hashes, intent, method=retriever)
            if candidates:
                candidates.sort(key=lambda item: float(item.fused_score), reverse=True)
                per_query_lists.append(candidates)

        if not per_query_lists:
            return []

        fused = self._fuse_fused_result_lists_rrf(
            fused_lists=per_query_lists,
            id_getter=lambda item: str(item.result.id or "").strip(),
        )
        if not fused:
            return []

        keep_ratio, min_keep, max_keep = self._get_dynamic_selection(intent_label, default_keep, min_k, max_k)
        final = self._select_dynamic_candidates(fused_items=fused, keep_ratio=keep_ratio, min_keep=min_keep, max_keep=max_keep)
        last_query = expanded_queries[-1] if expanded_queries else ""
        payload = getattr(self.rag.hierarchical_retriever, builder)(final, query=last_query)
        return self._extract_retrieval_data(payload)

    def _safe_retrieve(self, query, allowed_doc_hashes, intent, *, method):
        fn = getattr(self.rag.hierarchical_retriever, method)
        topn = int(self._get_strategy_number("element_candidate_pool_size", self.HIERARCHICAL_RETRIEVE_TOP_N))
        try:
            return fn(query=query, topn=topn, allowed_doc_hashes=allowed_doc_hashes, intent=intent)
        except TypeError:
            try:
                return fn(query, intent=intent)
            except TypeError:
                return fn(query)

    def _fuse_fused_result_lists_rrf(
        self,
        fused_lists: List[List[FusedResult]],
        id_getter,
    ) -> List[FusedResult]:
        if not fused_lists:
            return []
        rrf_k = self._get_strategy_number("cross_query_rrf_k", 60)
        rrf_scale = self._get_strategy_number("cross_query_rrf_scale", 100.0)
        score_map: dict[str, float] = {}
        result_map: dict[str, FusedResult] = {}

        for fused_list in fused_lists:
            for rank, item in enumerate(fused_list, start=1):
                item_id = str(id_getter(item) or "").strip()
                if not item_id:
                    continue
                score_map[item_id] = score_map.get(item_id, 0.0) + (rrf_scale / (rrf_k + rank))
                if item_id not in result_map:
                    result_map[item_id] = item

        fused: List[FusedResult] = []
        for item_id, score in sorted(score_map.items(), key=lambda kv: kv[1], reverse=True):
            base_item = result_map[item_id]
            fused.append(
                FusedResult(
                    result=base_item.result,
                    fused_score=float(score),
                    source_scores={"cross_query_rrf": float(score)},
                )
            )
        return fused

    @staticmethod
    def _select_dynamic_candidates(
        fused_items: List[FusedResult],
        keep_ratio: float,
        min_keep: int,
        max_keep: int,
    ) -> List[FusedResult]:
        if not fused_items:
            return []
        fused_sorted = sorted(fused_items, key=lambda item: float(item.fused_score), reverse=True)
        top_score = float(fused_sorted[0].fused_score)
        threshold = top_score * keep_ratio
        selected = [item for item in fused_sorted if float(item.fused_score) >= threshold]

        if len(selected) < min_keep:
            selected = fused_sorted[: min(min_keep, len(fused_sorted))]
        if len(selected) > max_keep:
            selected = selected[:max_keep]
        return selected
