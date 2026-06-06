import re
from typing import List, Optional, Set

from loguru import logger

from app.core.rag_defaults import (
    DEFAULT_ELEMENT_CANDIDATE_POOL_SIZE,
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
    VALID_INTENTS = {"document_overview", "general_overview", "fact_qa"}
    FACT_QUERY_MAX_VARIANTS = 5
    GENERAL_OVERVIEW_QUERY_MAX_VARIANTS = 4

    def __init__(self, rag):
        self.rag = rag
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

    @staticmethod
    def _dedupe_queries(queries: List[str], limit: int) -> List[str]:
        final: List[str] = []
        seen: set[str] = set()
        for query in queries:
            cleaned = re.sub(r"\s+", " ", str(query or "")).strip()
            if not cleaned:
                continue
            normalized = cleaned.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            final.append(cleaned)
            if len(final) >= limit:
                break
        return final

    @staticmethod
    def _split_fact_query(query: str) -> List[str]:
        parts = re.split(r"[？?。；;]+", query or "")
        splitters = ("相比之下", "作为对比", "同时", "以及", "此外", "并且")
        segments: List[str] = []
        for part in parts:
            candidates = [part]
            for splitter in splitters:
                next_candidates: List[str] = []
                for candidate in candidates:
                    next_candidates.extend(candidate.split(splitter))
                candidates = next_candidates
            for candidate in candidates:
                cleaned = candidate.strip(" ，,、：:")
                if len(cleaned) >= 8:
                    segments.append(cleaned)
        return segments or [query]

    @staticmethod
    def _split_fact_query_legacy(query: str) -> List[str]:
        parts = re.split(r"[？?。；;]+", query or "")
        splitters = ("相比之下", "作为对比", "同时", "以及", "此外", "并且")
        segments: List[str] = []
        for part in parts:
            candidates = [part]
            for splitter in splitters:
                next_candidates: List[str] = []
                for candidate in candidates:
                    next_candidates.extend(candidate.split(splitter))
                candidates = next_candidates
            for candidate in candidates:
                cleaned = candidate.strip(" ，,、：:")
                if len(cleaned) >= 8:
                    segments.append(cleaned)
        return segments or [query]

    @staticmethod
    def _build_fact_alias_query(query: str) -> str:
        alias_map = (
            ("训练数据", "training data dataset data source training mix weight"),
            ("数据集", "dataset"),
            ("来源", "data source source"),
            ("权重", "weight training mix"),
            ("参数", "parameters n params"),
            ("层数", "layers"),
            ("隐藏层", "hidden size"),
            ("注意力头", "attention heads"),
            ("错误率", "error rate top-1 top-5"),
            ("验证集", "validation set"),
            ("计算复杂度", "FLOPs complexity"),
            ("成功率", "success rate"),
            ("平均", "average"),
            ("分数", "score"),
            ("方差调度", "variance schedule beta"),
            ("扩散", "diffusion"),
            ("简化目标", "simplified training objective"),
            ("缩放定律", "scaling laws"),
            ("损失", "loss"),
            ("计算", "compute"),
            ("常数项", "constant term"),
            ("公式", "formula equation"),
            ("外部知识", "external knowledge"),
            ("适应性", "adaptability"),
            ("微调", "fine-tuning"),
            ("检索增强", "retrieval augmented generation RAG"),
            ("表", "table"),
        )
        aliases: List[str] = []
        for marker, alias in alias_map:
            if marker in query:
                aliases.append(alias)
        ascii_terms = re.findall(r"[A-Za-z][A-Za-z0-9_.+-]*|\d+(?:\.\d+)?%?", query or "")
        return " ".join(ascii_terms + aliases).strip()

    def _augment_fact_queries(self, original_query: str, expanded_queries: List[str]) -> List[str]:
        _ = expanded_queries
        return self._build_fact_english_queries(original_query)

    @staticmethod
    def _clean_generated_fact_query(text: str) -> str:
        for line in re.split(r"[\n\r]+", text or ""):
            cleaned = re.sub(r"^(\d+[\.\)]|[-*â€¢])\s*", "", line).strip(' \t"`\'"""\'\'\'')
            cleaned = re.sub(r"^(english\s+)?(retrieval\s+)?query\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned:
                return cleaned
        return ""

    @staticmethod
    def _clean_generated_query_line(text: str) -> str:
        cleaned = re.sub(r"^\s*(?:\d+[\.\)]|[-*])\s*", "", text or "")
        cleaned = cleaned.strip(" \t`\"'[]")
        cleaned = re.sub(
            r"^(english\s+)?(retrieval\s+)?(sub[-\s]?query|query|rewrite|variant)(?:\s+\d+)?\s*:\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = cleaned.strip(" \t`\"',;")
        return re.sub(r"\s+", " ", cleaned).strip()

    def _parse_generated_english_queries(self, generated: str, *, limit: int) -> List[str]:
        raw_lines = [line for line in re.split(r"[\n\r]+", generated or "") if line.strip()]
        if len(raw_lines) == 1:
            raw_lines = [line for line in re.split(r"\s*;\s*", raw_lines[0]) if line.strip()]
        cleaned = [self._clean_generated_query_line(line) for line in raw_lines]
        return self._dedupe_queries(cleaned, limit=limit)

    def _generate_english_queries(self, original_query: str, *, intent_label: str, prompt: str, limit: int) -> List[str]:
        generator = getattr(getattr(self.rag, "query_expander", None), "generate_summary", None)
        if not callable(generator):
            generator = getattr(self.rag, "_generate_summary", None)
        if not callable(generator):
            return self._dedupe_queries([original_query], limit=limit)

        try:
            generated = generator(prompt, max_new_tokens=160, temperature=0.0)
        except Exception as exc:
            logger.warning(f"{intent_label} query rewrite failed, using original query: {exc}")
            return self._dedupe_queries([original_query], limit=limit)

        parsed = self._parse_generated_english_queries(str(generated or ""), limit=limit)
        return parsed or self._dedupe_queries([original_query], limit=limit)

    def _build_fact_english_sub_queries_with_generator(self, original_query: str) -> List[str]:
        prompt = (
            "Split the following academic fact question into English retrieval sub-queries.\n"
            "Preserve paper titles, model names, dataset names, numbers, symbols, formulas, and table names.\n"
            f"Return at most {self.FACT_QUERY_MAX_VARIANTS} sub-queries. If the question is simple, return one sub-query.\n"
            "Use English only. Do not answer the question. Output one sub-query per line, with no explanation.\n\n"
            f"Question:\n{original_query}"
        )
        return self._generate_english_queries(
            original_query,
            intent_label="fact_qa",
            prompt=prompt,
            limit=self.FACT_QUERY_MAX_VARIANTS,
        )

    def _build_overview_english_rewrite_queries_with_generator(self, original_query: str) -> List[str]:
        prompt = (
            "Rewrite the following academic overview question into English retrieval query variants.\n"
            "Each variant should represent the full original question, not a separate sub-question.\n"
            "Focus on broad concepts, methods, contributions, comparisons, limitations, and section-level keywords.\n"
            f"Return at most {self.GENERAL_OVERVIEW_QUERY_MAX_VARIANTS} rewritten queries. If one query is enough, return one query.\n"
            "Use English only. Do not answer the question. Output one rewritten query per line, with no explanation.\n\n"
            f"Question:\n{original_query}"
        )
        return self._generate_english_queries(
            original_query,
            intent_label="general_overview",
            prompt=prompt,
            limit=self.GENERAL_OVERVIEW_QUERY_MAX_VARIANTS,
        )

    def _rewrite_sub_query_to_english(self, sub_query: str, *, intent_label: str) -> str:
        generator = getattr(getattr(self.rag, "query_expander", None), "generate_summary", None)
        if not callable(generator):
            generator = getattr(self.rag, "_generate_summary", None)
        if not callable(generator):
            return sub_query

        if intent_label == "general_overview":
            instruction = (
                "Rewrite the following academic overview question segment into ONE concise English retrieval query.\n"
                "Focus on broad concepts, methods, contributions, comparisons, limitations, and section-level keywords.\n"
            )
        else:
            instruction = (
                "Rewrite the following academic question segment into ONE concise English retrieval query.\n"
                "Preserve paper titles, model names, dataset names, numbers, symbols, formulas, and table names.\n"
            )
        prompt = (
            instruction +
            "Use English only, with keywords likely to appear in the paper. Output only the rewritten query.\n\n"
            f"Question segment:\n{sub_query}"
        )
        try:
            generated = generator(prompt, max_new_tokens=80, temperature=0.0)
        except Exception as exc:
            logger.warning(f"{intent_label} query rewrite failed, using original segment: {exc}")
            return sub_query
        return self._clean_generated_fact_query(str(generated or "")) or sub_query

    def _rewrite_fact_sub_query_to_english(self, sub_query: str) -> str:
        return self._rewrite_sub_query_to_english(sub_query, intent_label="fact_qa")

    def _build_fact_english_queries(self, original_query: str) -> List[str]:
        return self._build_fact_english_sub_queries_with_generator(original_query)

    @staticmethod
    def _split_overview_query(query: str) -> List[str]:
        return RagRetrievalService._split_fact_query(query)

    def _rewrite_overview_sub_query_to_english(self, sub_query: str) -> str:
        return self._rewrite_sub_query_to_english(sub_query, intent_label="general_overview")

    def _build_general_overview_english_queries(self, original_query: str) -> List[str]:
        return self._build_overview_english_rewrite_queries_with_generator(original_query)

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

        if classified_intent == "fact_qa":
            expanded_queries = self._build_fact_english_queries(query)
        elif classified_intent == "general_overview":
            expanded_queries = self._build_general_overview_english_queries(query)
        else:
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
        return self._extract_retrieval_data(payload)

    def _run_fused_pipeline(self, expanded_queries, allowed_doc_hashes, intent, *, retriever, builder, intent_label):
        default_keep, min_k, max_k = {"fact_qa": (0.6, 2, 6), "general_overview": (0.65, 2, 6)}[intent_label]
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
