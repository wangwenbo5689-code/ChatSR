from collections.abc import Iterable
from typing import Callable, List, Tuple

from loguru import logger

from app.core.rag_defaults import (
    DEFAULT_HISTORY_KEEP_TURNS,
    DEFAULT_HISTORY_SUMMARY_TURNS,
    DEFAULT_SUMMARY_MAX_NEW_TOKENS,
    DEFAULT_SUMMARY_TEMPERATURE,
)
from app.core.rag_prompts import SUMMARY_PROMPT_TEMPLATE


class RagHistoryManager:
    """RAG 历史记录管理器"""

    RECENT_SUMMARY_TURNS = DEFAULT_HISTORY_SUMMARY_TURNS
    RECENT_KEEP_TURNS = DEFAULT_HISTORY_KEEP_TURNS
    DEFAULT_MAX_NEW_TOKENS = DEFAULT_SUMMARY_MAX_NEW_TOKENS
    DEFAULT_TEMPERATURE = DEFAULT_SUMMARY_TEMPERATURE

    def __init__(
        self,
        stream_generate_answer: Callable[..., Iterable[str]],
        summary_turns: int = DEFAULT_HISTORY_SUMMARY_TURNS,
        keep_turns: int = DEFAULT_HISTORY_KEEP_TURNS,
    ):
        """初始化历史记录管理器

        Args:
            stream_generate_answer: 流式生成回答的函数
        """
        self.stream_generate_answer = stream_generate_answer
        resolved_summary_turns = DEFAULT_HISTORY_SUMMARY_TURNS if summary_turns is None else summary_turns
        resolved_keep_turns = DEFAULT_HISTORY_KEEP_TURNS if keep_turns is None else keep_turns
        self.summary_turns = max(int(resolved_summary_turns), 0)
        self.keep_turns = max(int(resolved_keep_turns), 0)

    @staticmethod
    def format_history_pairs(pairs: List[List[str]]) -> str:
        """格式化历史对话对

        Args:
            pairs: 历史对话对列表

        Returns:
            str: 格式化后的历史对话
        """
        if not pairs:
            return ""

        lines = []
        for index, pair in enumerate(pairs, start=1):
            user_text = pair[0] if len(pair) > 0 else ""
            assistant_text = pair[1] if len(pair) > 1 else ""
            lines.append(f"第{index}轮用户：{user_text}")
            lines.append(f"第{index}轮助手：{assistant_text}")
        return "\n".join(lines)

    def generate_summary(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """生成摘要

        Args:
            prompt: 提示文本
            max_new_tokens: 最大新令牌数
            temperature: 温度

        Returns:
            str: 生成的摘要
        """
        if not prompt:
            return ""

        temp_history = [[prompt, ""]]
        result_parts = []

        try:
            for token in self.stream_generate_answer(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                history=temp_history,
            ):
                if token:
                    result_parts.append(token)
        except Exception as e:
            # 记录错误但不中断主流程
            logger.warning(f"生成摘要失败: {e}")
            return ""

        return "".join(result_parts).strip()

    def compress_if_needed(
        self,
        history: List[List[str]],
        history_summary: str,
        enable_history: bool,
        **_compat_kwargs,
    ) -> Tuple[List[List[str]], str]:
        """处理历史记录，确保每轮对话都有适当的上下文

        Args:
            history: 历史记录
            history_summary: 历史摘要
            enable_history: 是否启用历史记录

        Returns:
            处理后的历史记录和摘要
        """
        # 快速返回条件
        if not enable_history or not history:
            return history, history_summary

        # 生成新摘要（使用最近两轮对话之前的5轮对话）
        new_summary = self._generate_dialogue_summary(history, history_summary)

        # 始终保留最近 2 轮对话
        keep_turns = self.keep_turns

        if keep_turns == 0:
            return [], new_summary

        return history[-keep_turns:], new_summary

    def _generate_dialogue_summary(self, history: List[List[str]], old_summary: str = "") -> str:
        """生成对话摘要

        Args:
            history: 历史记录

        Returns:
            str: 生成的摘要
        """
        # 获取最近两轮对话之前的5轮对话用于生成摘要
        # 例如：历史记录有10轮，那么取轮次 3-7（索引 2-6）
        if len(history) <= self.keep_turns:
            # 历史不足以产生新摘要时，保留已有压缩记忆。
            return old_summary or ""

        # 计算需要摘要的对话范围
        if self.summary_turns == 0:
            return old_summary or ""
        summary_start = max(0, len(history) - self.keep_turns - self.summary_turns)
        summary_end = len(history) - self.keep_turns
        summary_turns = history[summary_start:summary_end]

        dialogue_str = self.format_history_pairs(summary_turns)

        if not dialogue_str:
            return old_summary or ""

        # 构建提示
        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            old_summary=old_summary or "（总结最近两轮对话之前的5轮对话的主要内容和主题）",
            dialogue_str=dialogue_str
        )

        generated = self.generate_summary(prompt)
        return generated or (old_summary or "")
