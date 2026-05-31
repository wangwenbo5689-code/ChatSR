import unittest

from app.core.rag_history_manager import RagHistoryManager
from app.core.rag_query_expander import RagQueryExpander


class RagHistoryManagerTests(unittest.TestCase):
    def test_format_history_pairs_keeps_turn_order(self):
        formatted = RagHistoryManager.format_history_pairs([["问题1", "答案1"], ["问题2", "答案2"]])

        self.assertIn("第1轮用户：问题1", formatted)
        self.assertIn("第1轮助手：答案1", formatted)
        self.assertIn("第2轮用户：问题2", formatted)
        self.assertIn("第2轮助手：答案2", formatted)

    def test_compress_if_needed_always_keeps_recent_two_turns_and_updates_summary(self):
        manager = RagHistoryManager(lambda **_: iter(["新的摘要"]))

        history, summary = manager.compress_if_needed(
            history=[["q1", "a1"], ["q2", "a2"], ["q3", "a3"]],
            history_summary="旧摘要",
            enable_history=True,
            history_max_turns=2,
            history_keep_last_turns=1,  # 当前策略固定保留最近 2 轮，此参数仅为兼容保留
        )

        self.assertEqual(history, [["q2", "a2"], ["q3", "a3"]])
        self.assertEqual(summary, "新的摘要")


class RagQueryExpanderTests(unittest.TestCase):
    def test_expand_returns_original_query_when_disabled(self):
        expander = RagQueryExpander(lambda *args, **kwargs: "不会被调用")

        self.assertEqual(expander.expand("残差连接是什么", enabled=False), ["残差连接是什么"])

    def test_expand_deduplicates_and_limits_queries(self):
        expander = RagQueryExpander(
            lambda *args, **kwargs: "残差连接是什么\n1. 残差连接的作用\n残差连接是什么\n残差连接与跳连的区别\n残差连接对训练稳定性的影响"
        )

        queries = expander.expand("残差连接是什么", enabled=True)

        self.assertEqual(
            queries,
            ["残差连接是什么", "残差连接的作用", "残差连接与跳连的区别"],
        )


if __name__ == "__main__":
    unittest.main()
