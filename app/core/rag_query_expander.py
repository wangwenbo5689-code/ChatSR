import re
from typing import Callable, List, Optional
from loguru import logger
from app.core.rag_defaults import (
    DEFAULT_HISTORY_KEEP_TURNS,
    DEFAULT_HISTORY_SUMMARY_TURNS,
    DEFAULT_INTENT_CLASSIFICATION_MAX_NEW_TOKENS,
    DEFAULT_INTENT_CLASSIFICATION_TEMPERATURE,
    DEFAULT_INTENT_ERROR_DETAIL_MAX_LENGTH,
    DEFAULT_QUERY_CANDIDATE_MAX_LENGTH,
    DEFAULT_QUERY_EXPANSION_MAX_NEW_TOKENS,
    DEFAULT_QUERY_EXPANSION_TEMPERATURE,
    DEFAULT_QUERY_OPTIMIZATION_MAX_NEW_TOKENS,
    DEFAULT_SUMMARY_MAX_NEW_TOKENS,
    DEFAULT_SUMMARY_TEMPERATURE,
)
from app.core.rag_prompts import (
    INTENT_CLASSIFICATION_PROMPT_TEMPLATE,
    QUERY_OPTIMIZATION_PROMPT_TEMPLATE,
    QUERY_REWRITE_PROMPT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
)
from app.core.rag_history_manager import RagHistoryManager


class IntentClassificationError(RuntimeError):
    """意图分类错误"""



class RagQueryExpander:
    """RAG 查询扩展器"""
    MAX_SUB_QUERIES = 3

    def __init__(
        self,
        generate_summary: Callable[..., str],
        history_summary_turns: int = DEFAULT_HISTORY_SUMMARY_TURNS,
        history_keep_turns: int = DEFAULT_HISTORY_KEEP_TURNS,
    ):
        """初始化查询扩展器

        Args:
            generate_summary: 生成摘要的函数
        """
        self.generate_summary = generate_summary
        self.history_manager = RagHistoryManager(
            stream_generate_answer=self._stream_summary,
            summary_turns=history_summary_turns,
            keep_turns=history_keep_turns,
        )

    def _stream_summary(
        self,
        *,
        history=None,
        max_new_tokens=DEFAULT_SUMMARY_MAX_NEW_TOKENS,
        temperature=DEFAULT_SUMMARY_TEMPERATURE,
        **_kwargs,
    ):
        prompt = ""
        if history and isinstance(history, list) and history[0]:
            prompt = str(history[0][0] or "")
        if not prompt:
            return iter(())
        text = self.generate_summary(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return iter([text] if text else [])

    def _build_history_context(self, kept_history: List[List[str]], new_summary: str) -> str:
        """构建历史上下文文本。"""
        context_parts: List[str] = []
        if new_summary:
            context_parts.append(f"当前会话的压缩记忆：\n{new_summary}")
        recent_dialogue = self.history_manager.format_history_pairs(kept_history)
        if recent_dialogue:
            context_parts.append(f"最近2轮对话：\n{recent_dialogue}")
        return "\n\n".join(context_parts).strip()

    def _prepare_history_context(
        self,
        history: List[List[str]],
        history_summary: str,
        use_history: bool,
    ) -> tuple[List[List[str]], str]:
        if not use_history or not history:
            return [], history_summary or ""

        keep_turns = self.history_manager.keep_turns
        kept_history = history[-keep_turns:] if keep_turns else []
        older_history = history[:-keep_turns] if keep_turns else history
        summary = history_summary or ""
        if not older_history:
            return kept_history, summary

        summary_turns = older_history[-self.history_manager.summary_turns:] if self.history_manager.summary_turns else []
        dialogue_str = self.history_manager.format_history_pairs(summary_turns)
        if not dialogue_str:
            return kept_history, summary

        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            old_summary=summary or "（无）",
            dialogue_str=dialogue_str,
        )
        try:
            generated = (
                self.generate_summary(
                    prompt,
                    max_new_tokens=DEFAULT_SUMMARY_MAX_NEW_TOKENS,
                    temperature=DEFAULT_SUMMARY_TEMPERATURE,
                ) or ""
            ).strip()
            if generated:
                summary = generated
        except Exception as exc:
            logger.warning(f"历史摘要生成失败: {exc}, 使用已有摘要继续")
        return kept_history, summary

    def _optimize_query(self, query: str, history_context: str, should_optimize: bool) -> str:
        """根据历史上下文优化查询。"""
        if not should_optimize or not history_context:
            return query

        optimize_prompt = QUERY_OPTIMIZATION_PROMPT_TEMPLATE.format(
            history_context=history_context,
            query_str=query,
        )
        try:
            optimized_raw = (
                self.generate_summary(
                    optimize_prompt,
                    max_new_tokens=DEFAULT_QUERY_OPTIMIZATION_MAX_NEW_TOKENS,
                    temperature=DEFAULT_SUMMARY_TEMPERATURE,
                ) or ""
            ).strip()
            optimized_candidate = re.sub(r"\s+", " ", optimized_raw).strip(' \t"\'')
            if optimized_candidate:
                return optimized_candidate
        except Exception as exc:
            logger.warning(f"查询优化失败: {exc}, 使用原始查询继续生成子查询")
        return query

    def _parse_sub_queries(self, expansion_result: str, optimized_query: str) -> List[str]:
        """清洗并截断子查询结果。"""
        raw_lines = re.split(r"[\n\r;；]+", expansion_result or "")
        cleaned_queries: List[str] = []
        seen = set()
        for line in raw_lines:
            candidate = re.sub(r"^(\d+[\.\)]|[-*•])\s*", "", line).strip(' \t"\'"""\'\'\'')
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if len(candidate) < 2:
                continue
            if len(candidate) > DEFAULT_QUERY_CANDIDATE_MAX_LENGTH:
                continue
            if "：" in candidate and len(candidate.split("：", 1)[0]) <= 6:
                continue
            normalized = candidate.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            cleaned_queries.append(candidate)

        if optimized_query.lower() not in seen:
            cleaned_queries.insert(0, optimized_query)
        return cleaned_queries[: self.MAX_SUB_QUERIES]


    def expand(
        self,
        query: str,
        enabled: bool,
        history: Optional[List[List[str]]] = None,
        history_summary: Optional[str] = None,
        use_history: bool = True,
    ) -> List[str]:
        """扩展查询

        Args:
            query: 查询文本
            enabled: 是否启用查询扩展
            history: 历史对话记录
            history_summary: 历史对话摘要
            use_history: 是否启用历史上下文

        Returns:
            List[str]: 扩展后的查询列表
        """
        if not enabled:
            return [query]

        # 使用 RagHistoryManager 处理历史记录，获取最近2轮对话和之前5轮对话的摘要
        processed_history = history or []
        processed_summary = history_summary or ""

        kept_history, new_summary = self._prepare_history_context(
            history=processed_history,
            history_summary=processed_summary,
            use_history=use_history,
        )
        history_context = self._build_history_context(kept_history, new_summary)
        should_optimize = bool(processed_history) and use_history
        optimized_query = self._optimize_query(query, history_context, should_optimize)

        # 再根据优化后的问题生成子查询（最多3个）
        prompt = QUERY_REWRITE_PROMPT_TEMPLATE.format(optimized_query=optimized_query)
        logger.info(f"扩展查询: {query}")
        try:
            expansion_result = self.generate_summary(
                prompt,
                max_new_tokens=DEFAULT_QUERY_EXPANSION_MAX_NEW_TOKENS,
                temperature=DEFAULT_QUERY_EXPANSION_TEMPERATURE,
            )
            final_queries = self._parse_sub_queries(expansion_result, optimized_query)
            logger.info(f"生成子查询: {final_queries}")
            return final_queries
        except Exception as exc:
            logger.warning(f"查询扩展失败: {exc}, 回退到原始查询")
            return [query]

    def classify_intent(self, query: str) -> str:
        """分类意图

        Args:
            query: 查询文本

        Returns:
            str: 意图类别

        Raises:
            IntentClassificationError: 意图分类失败
        """
        prompt = INTENT_CLASSIFICATION_PROMPT_TEMPLATE.format(query_str=query)
        logger.info(f"分类查询意图: {query}")
        try:
            result = (
                self.generate_summary(
                    prompt,
                    max_new_tokens=DEFAULT_INTENT_CLASSIFICATION_MAX_NEW_TOKENS,
                    temperature=DEFAULT_INTENT_CLASSIFICATION_TEMPERATURE,
                ) or ""
            ).strip()
        except Exception as exc:
            err = IntentClassificationError("意图分类失败：模型调用异常")
            err.detail = str(exc)
            raise err from exc

        cleaned = re.sub(r"[`\"'“”‘’]", "", result or "").strip().lower()
        match = re.match(r"^(document_overview|general_overview|fact_qa)\s*$", cleaned)
        if match:
            return match.group(1)
        err = IntentClassificationError("意图分类失败：模型未返回有效类别")
        err.detail = (result or "")[:DEFAULT_INTENT_ERROR_DETAIL_MAX_LENGTH]
        raise err
