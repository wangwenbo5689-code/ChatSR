import unittest

from app.core.hierarchical_context_builder import ContextBuilder
from app.core.hierarchical_payload_builder import build_hierarchical_document_payloads, normalize_chunk_items
from app.core.hierarchical_retriever import HybridRetriever
from app.core.hierarchical_ranking import ResultFuser, ResultReranker
from app.core.hierarchical_types import FusedResult, RetrievalResult


class StaticRetriever:
    def __init__(self, dense_results=None, sparse_results=None):
        self.dense_results = dense_results or []
        self.sparse_results = sparse_results or []

    def retrieve(self, _query: str, topn=None, _topn=None):
        return self.dense_results, self.sparse_results


class FakeCollection:
    def __init__(self, records):
        self.records = records

    def get(self, ids=None, where=None, _where=None, **_kwargs):
        if ids is not None:
            selected_ids = [record_id for record_id in ids if record_id in self.records]
        else:
            selected_ids = list(self.records.keys())
        where_clause = where or _where
        if where_clause:
            allowed_hashes = None
            doc_hash_clause = where_clause.get("doc_hash") if isinstance(where_clause, dict) else None
            if isinstance(doc_hash_clause, dict) and "$in" in doc_hash_clause:
                allowed_hashes = {str(item).lower() for item in doc_hash_clause["$in"]}
            elif isinstance(doc_hash_clause, str):
                allowed_hashes = {doc_hash_clause.lower()}
            if allowed_hashes is not None:
                selected_ids = [
                    record_id for record_id in selected_ids
                    if str(self.records[record_id]["metadata"].get("doc_hash") or "").lower() in allowed_hashes
                ]
        return {
            "ids": selected_ids,
            "documents": [self.records[record_id]["document"] for record_id in selected_ids],
            "metadatas": [self.records[record_id]["metadata"] for record_id in selected_ids],
        }


class HierarchicalPayloadBuilderTests(unittest.TestCase):
    def test_normalize_chunk_items_centralizes_text_and_metadata_cleanup(self):
        normalized = normalize_chunk_items(
            chunk_items=[
                {"text": "  attention residual block  ", "metadata": {"section_title": "Intro"}},
                {"text": "", "metadata": {"section_title": "Skip"}},
                {"text": "valid", "metadata": None},
                "invalid",
            ],
            doc_hash="doc-1",
        )

        self.assertEqual([item["text"] for item in normalized], ["attention residual block", "valid"])
        self.assertEqual(normalized[0]["metadata"]["section_title"], "Intro")
        self.assertEqual(normalized[0]["metadata"]["doc_hash"], "doc-1")
        self.assertEqual(normalized[0]["metadata"]["text_content"], "attention residual block")
        self.assertEqual(normalized[1]["metadata"]["doc_hash"], "doc-1")
        self.assertEqual(normalized[1]["metadata"]["section_hierarchy"], "Document")

    def test_build_payloads_keeps_section_and_element_links(self):
        payloads = build_hierarchical_document_payloads(
            doc_hash="doc-1",
            doc_file="/tmp/demo.pdf",
            chunk_items=[
                {
                    "text": "attention residual block",
                    "metadata": {"section_title": "Intro", "section_hierarchy": "Intro", "source_type": "text", "page": 2},
                },
                {
                    "text": "reference entry",
                    "metadata": {"section_title": "References", "section_hierarchy": "References", "section_type": "references", "source_type": "text", "page": 10},
                },
            ],
        )

        self.assertEqual(set(payloads.keys()), {"paper", "section", "element"})
        self.assertEqual(payloads["paper"]["ids"], ["doc-1_paper"])
        self.assertNotIn("parent_id", payloads["paper"]["metadatas"][0])
        self.assertEqual(payloads["section"]["ids"], ["doc-1_paper_sec_0"])
        self.assertEqual(payloads["section"]["metadatas"][0]["parent_id"], "doc-1_paper")
        self.assertEqual(payloads["element"]["metadatas"][0]["section_id"], "doc-1_paper_sec_0")
        self.assertEqual(payloads["element"]["metadatas"][0]["parent_id"], "doc-1_paper_sec_0")
        self.assertEqual(payloads["element"]["documents"][0], "attention residual block")
        self.assertEqual(payloads["element"]["ids"][0], "doc-1_paper_sec_0_elem_0")
        self.assertNotIn("reference entry", payloads["paper"]["documents"][0])
        self.assertNotIn("reference entry", payloads["element"]["documents"])
        self.assertEqual(payloads["paper"]["metadatas"][0]["page_count"], 10)
        self.assertNotIn("year", payloads["paper"]["metadatas"][0])
        self.assertNotIn("body_section_count", payloads["paper"]["metadatas"][0])
        self.assertNotIn("page_span", payloads["section"]["metadatas"][0])
        self.assertNotIn("global_position", payloads["element"]["metadatas"][0])

    def test_build_payloads_preserves_all_sections_with_content(self):
        payloads = build_hierarchical_document_payloads(
            doc_hash="doc-1",
            doc_file="/tmp/demo.pdf",
            chunk_items=[
                {
                    "text": "intro",
                    "metadata": {"section_title": "Introduction", "section_hierarchy": "Introduction", "page": 1},
                },
                {
                    "text": "motivation",
                    "metadata": {"section_title": "Motivation", "section_hierarchy": "Introduction > Motivation", "page": 2},
                },
                {
                    "text": "method",
                    "metadata": {"section_title": "Method", "section_hierarchy": "Method", "page": 3},
                },
            ],
        )

        section_metas = payloads["section"]["metadatas"]
        # All sections with content are preserved, including parent sections.
        self.assertEqual(
            [meta["section_hierarchy"] for meta in section_metas],
            ["Introduction", "Introduction > Motivation", "Method"],
        )
        self.assertEqual(section_metas[2]["section_number"], "3")

    def test_build_payloads_filters_global_noise_sections(self):
        payloads = build_hierarchical_document_payloads(
            doc_hash="doc-1",
            doc_file="/tmp/demo.pdf",
            chunk_items=[
                {
                    "text": "method details",
                    "metadata": {"section_title": "Method", "section_hierarchy": "Method", "page": 3},
                },
                {
                    "text": "all authors contributed equally",
                    "metadata": {"section_title": "Author Contributions", "section_hierarchy": "Author Contributions", "page": 9},
                },
            ],
        )

        self.assertEqual([meta["section_title"] for meta in payloads["section"]["metadatas"]], ["Method"])
        self.assertEqual([meta["section_title"] for meta in payloads["element"]["metadatas"]], ["Method"])
        self.assertNotIn("all authors", payloads["paper"]["documents"][0])

    def test_build_payloads_keeps_parent_chunks_in_parent_section(self):
        payloads = build_hierarchical_document_payloads(
            doc_hash="doc-1",
            doc_file="/tmp/demo.pdf",
            chunk_items=[
                {
                    "text": "intro overview",
                    "metadata": {"section_title": "Introduction", "section_hierarchy": "Introduction", "page": 1},
                },
                {
                    "text": "motivation",
                    "metadata": {"section_title": "Motivation", "section_hierarchy": "Introduction > Motivation", "page": 2},
                },
            ],
        )

        # Both parent and leaf sections are preserved.
        self.assertEqual(
            [meta["section_hierarchy"] for meta in payloads["section"]["metadatas"]],
            ["Introduction", "Introduction > Motivation"],
        )
        # Parent content stays in parent section.
        intro_doc = payloads["section"]["documents"][0]
        self.assertIn("intro overview", intro_doc)
        # Leaf content stays in leaf section.
        motivation_doc = payloads["section"]["documents"][1]
        self.assertIn("motivation", motivation_doc)

    def test_build_payloads_preserves_section_order_and_no_remapping(self):
        payloads = build_hierarchical_document_payloads(
            doc_hash="doc-1",
            doc_file="/tmp/demo.pdf",
            chunk_items=[
                {
                    "text": "background details",
                    "metadata": {"section_title": "Background", "section_hierarchy": "Introduction > Background", "page": 1},
                },
                {
                    "text": "bridge paragraph",
                    "metadata": {"section_title": "Introduction", "section_hierarchy": "Introduction", "page": 2},
                },
                {
                    "text": "motivation details",
                    "metadata": {"section_title": "Motivation", "section_hierarchy": "Introduction > Motivation", "page": 3},
                },
            ],
        )

        # All sections appear in original order; no remapping to nearest leaf.
        self.assertEqual(
            [meta["section_hierarchy"] for meta in payloads["section"]["metadatas"]],
            ["Introduction > Background", "Introduction", "Introduction > Motivation"],
        )
        # Element section_id matches its original section.
        self.assertEqual(payloads["element"]["metadatas"][1]["section_hierarchy"], "Introduction")
        self.assertEqual(payloads["element"]["metadatas"][1]["section_id"], "doc-1_paper_sec_1")

    def test_build_payloads_use_leaf_section_title_and_keep_full_hierarchy(self):
        payloads = build_hierarchical_document_payloads(
            doc_hash="doc-1",
            doc_file="/tmp/demo.pdf",
            chunk_items=[
                {
                    "text": "motivation details",
                    "metadata": {
                        "section_title": "Introduction",
                        "section_hierarchy": "Introduction > Motivation",
                        "page": 2,
                    },
                },
            ],
        )

        section_meta = payloads["section"]["metadatas"][0]
        element_meta = payloads["element"]["metadatas"][0]
        self.assertEqual(section_meta["section_title"], "Motivation")
        self.assertEqual(section_meta["section_hierarchy"], "Introduction > Motivation")
        self.assertEqual(element_meta["section_title"], "Motivation")
        self.assertEqual(element_meta["section_hierarchy"], "Introduction > Motivation")


class ResultFuserTests(unittest.TestCase):
    def test_rrf_fuse_merges_dense_and_sparse_rankings(self):
        fuser = ResultFuser({"similarity_threshold": 0.0, "enable_deduplication": True})
        dense_results = [
            RetrievalResult(
                id="sec-1",
                content="section",
                score=0.8,
                level="section",
                metadata={"paper_id": "paper-1", "section_title": "Intro"},
            )
        ]
        sparse_results = [
            RetrievalResult(
                id="sec-1",
                content="section",
                score=1.2,
                level="section",
                metadata={"paper_id": "paper-1", "section_title": "Intro"},
            ),
            RetrievalResult(
                id="sec-2",
                content="other section",
                score=0.4,
                level="section",
                metadata={"paper_id": "paper-2", "section_title": "Method"},
            ),
        ]
        fused = fuser._rrf_fuse(dense_results, sparse_results, level="section")

        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0].id, "sec-1")
        self.assertEqual(fused[0].level, "section")


class ResultRerankerTests(unittest.TestCase):
    def test_rerank_without_model_keeps_existing_fused_order(self):
        reranker = ResultReranker.__new__(ResultReranker)
        reranker.model_name_or_path = "dummy"
        reranker.device = "cpu"
        reranker.rerank_tokenizer = None
        reranker.rerank_model = None

        results = [
            FusedResult(
                result=RetrievalResult(
                    id="sec-2",
                    content="classification details",
                    score=0.2,
                    level="section",
                    metadata={"section_title": "Classification"},
                ),
                fused_score=0.8,
                source_scores={"section": 0.8},
            ),
            FusedResult(
                result=RetrievalResult(
                    id="sec-1",
                    content="overview",
                    score=0.3,
                    level="section",
                    metadata={"section_title": "Overview"},
                ),
                fused_score=0.6,
                source_scores={"section": 0.6},
            ),
        ]

        reranked = reranker.rerank("classification result", results, top_k=2)

        self.assertEqual([item.result.id for item in reranked], ["sec-2", "sec-1"])
        self.assertEqual([item.fused_score for item in reranked], [0.8, 0.6])

    def test_min_max_normalization_aligns_different_score_scales(self):
        normalized = ResultReranker._normalize_scores_min_max([10.0, 20.0, 30.0])
        self.assertEqual(normalized, [0.0, 0.5, 1.0])

    def test_min_max_normalization_returns_neutral_when_all_scores_equal(self):
        normalized = ResultReranker._normalize_scores_min_max([3.0, 3.0, 3.0])
        self.assertEqual(normalized, [0.5, 0.5, 0.5])

    def test_rank_positions_assigns_one_based_rank_by_desc_score(self):
        ranks = ResultReranker._rank_positions([0.8, 0.2, 0.5])
        self.assertEqual(ranks, [1, 3, 2])

    def test_rank_fusion_uses_reciprocal_rank_and_is_scale_invariant(self):
        fused_a = ResultReranker._fuse_scores_by_rank(
            original_scores=[1000.0, 900.0, 800.0],
            model_scores=[-3.0, 2.0, 1.0],
            original_weight=0.45,
            model_weight=0.55,
            rank_bias=60,
        )
        fused_b = ResultReranker._fuse_scores_by_rank(
            original_scores=[1.0, 0.0, -1.0],
            model_scores=[0.01, 10.0, 2.0],
            original_weight=0.45,
            model_weight=0.55,
            rank_bias=60,
        )

        # 两组输入只改变了绝对分值尺度与偏移，排名关系相同，融合结果应一致。
        self.assertEqual(fused_a, fused_b)
        self.assertGreater(fused_a[1], fused_a[0])


class ContextBuilderTests(unittest.TestCase):
    def test_build_returns_expected_context_shape(self):
        builder = ContextBuilder(max_context_length=2000)
        fused_results = [
            FusedResult(
                result=RetrievalResult(
                    id="elem-1",
                    content="残差连接可以改善深层网络训练。",
                    score=1.0,
                    level="element",
                    metadata={"paper_id": "paper-1", "section_title": "Intro", "page": 3, "element_type": "text"},
                ),
                fused_score=1.0,
                source_scores={"element": 1.0},
            )
        ]

        payload = builder.build(fused_results, query="残差连接的作用")

        self.assertIn("用户问题: 残差连接的作用", payload["context"])
        self.assertIn("=" * 50, payload["context"])
        self.assertEqual(payload["result_count"], 1)
        self.assertEqual(len(payload["blocks"]), 1)
        self.assertIn("残差连接可以改善深层网络训练。", payload["blocks"][0])
        self.assertEqual(payload["citations"][0]["type"], "element")
        self.assertEqual(payload["structured"][0]["id"], "paper-1")
        self.assertEqual(payload["structured"][0]["elements"][0]["id"], "elem-1")

    def test_build_keeps_first_result_when_single_section_block_exceeds_limit(self):
        builder = ContextBuilder(max_context_length=200)
        fused_results = [
            FusedResult(
                result=RetrievalResult(
                    id="sec-1",
                    content="残差连接" * 300,
                    score=1.0,
                    level="section",
                    metadata={
                        "paper_id": "paper-1",
                        "title": "Residual Paper",
                        "section_title": "Introduction",
                        "section_type": "introduction",
                        "start_page": 1,
                    },
                ),
                fused_score=1.0,
                source_scores={"section": 1.0},
            )
        ]

        payload = builder.build(fused_results, query="残差连接有什么作用")

        self.assertEqual(payload["result_count"], 1)
        self.assertEqual(len(payload["blocks"]), 1)
        self.assertEqual(payload["citations"][0]["type"], "section")
        self.assertNotEqual(payload["context"], "未找到相关内容。")
        self.assertIn("...[内容截断]", payload["blocks"][0])
        self.assertLessEqual(payload["total_length"], 200)


class QueryRoutingTests(unittest.TestCase):
    def test_document_level_query_avoids_local_focus_questions(self):
        self.assertFalse(HybridRetriever._is_document_summary_query("摘要里是怎么描述方法设计的"))
        self.assertTrue(HybridRetriever._is_document_summary_query("请概述这篇论文的主要贡献"))

    def test_only_explicit_document_summary_queries_route_to_paper(self):
        self.assertFalse(HybridRetriever._is_document_summary_query("请概述这篇论文的方法设计"))
        self.assertFalse(HybridRetriever._is_document_summary_query("请总结第3页讲了什么"))
        self.assertFalse(HybridRetriever._is_document_summary_query("这篇论文的结果部分说了什么"))
        self.assertTrue(HybridRetriever._is_document_summary_query("请总结整篇论文"))
        self.assertTrue(HybridRetriever._is_document_summary_query("请给我全文概述"))


class ParentDocumentRetrieverTests(unittest.TestCase):
    def test_select_high_quality_children_reranks_all_filtered_and_keeps_top_15(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        captured = {"top_k": "unset"}

        def fake_rerank_results(query, fused_results, top_k, intent=None):
            _ = query, intent
            captured["top_k"] = top_k
            fused_results.sort(key=lambda item: item.fused_score, reverse=True)
            return fused_results

        retriever._rerank_results = fake_rerank_results

        child_hits = [
            RetrievalResult(
                id=f"elem-{idx}",
                content=f"chunk {idx}",
                score=float(100 - idx),
                level="element",
                metadata={"parent_id": f"sec-{idx}", "section_id": f"sec-{idx}", "paper_id": "paper-1"},
            )
            for idx in range(15)
        ]

        selected = retriever._select_high_quality_children(
            query="test query",
            child_hits=child_hits,
            intent="fact_qa",
        )

        self.assertIsNone(captured["top_k"])
        self.assertEqual(len(selected), 15)
        self.assertEqual([item.id for item in selected], [f"elem-{idx}" for idx in range(15)])

    def test_child_hits_are_promoted_to_parent_sections(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.config = {
            "paper_top_k": 2,
            "element_base_recall": 4,
            "element_rerank_top_k": 4,
            "max_children_per_section": 3,
            "child_score_ratio": 0.0,
            "parent_rerank_top_k": 3,
            "paper_weight": 0.2,
            "similarity_threshold": 0.0,
        }
        retriever.result_fuser = ResultFuser({"similarity_threshold": 0.0, "enable_deduplication": False})
        retriever.reranker = None
        retriever.paper_retriever = StaticRetriever()
        retriever.element_retriever = StaticRetriever(
            dense_results=[
                RetrievalResult(
                    id="elem-1",
                    content="残差连接可以改善深层网络训练。",
                    score=0.8,
                    level="element",
                    metadata={
                        "parent_id": "sec-1",
                        "paper_id": "paper-1",
                        "section_id": "sec-1",
                        "section_title": "Intro",
                        "page": 3,
                        "text_content": "残差连接可以改善深层网络训练。",
                    },
                )
            ]
        )
        retriever.section_collection = FakeCollection(
            {
                "sec-1": {
                    "document": "这是引言章节的完整父块内容。",
                    "metadata": {
                        "section_id": "sec-1",
                        "paper_id": "paper-1",
                        "parent_id": "paper-1",
                        "section_title": "Intro",
                        "section_type": "introduction",
                        "start_page": 2,
                    },
                }
            }
        )
        retriever.paper_collection = FakeCollection(
            {
                "paper-1": {
                    "document": "整篇论文主体内容。",
                    "metadata": {
                        "paper_id": "paper-1",
                        "title": "demo.pdf",
                        "authors": "",
                        "venue": "",
                        "year": 2024,
                    },
                }
            }
        )
        retriever.context_builder = ContextBuilder(max_context_length=4000)

        payload = retriever.retrieve("残差连接有什么作用", topn=2, intent="fact_qa")

        self.assertEqual(payload["citations"][0]["type"], "element")
        self.assertIn("残差连接可以改善深层网络训练。", payload["context"])
        self.assertEqual(payload["structured"][0]["elements"][0]["id"], "elem-1")

    def test_child_rerank_happens_before_parent_section_promotion(self):
        class FakeReranker:
            def rerank(self, _query, results, top_k=None):
                score_map = {"elem-1": 0.3, "elem-2": 0.95}
                for item in results:
                    item.fused_score = score_map.get(item.result.id, item.fused_score)
                results.sort(key=lambda item: item.fused_score, reverse=True)
                return results[:top_k] if top_k else results

        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.config = {
            "paper_top_k": 2,
            "element_base_recall": 4,
            "element_rerank_top_k": 4,
            "max_children_per_section": 3,
            "child_score_ratio": 0.0,
            "parent_rerank_top_k": 3,
            "paper_weight": 0.2,
            "similarity_threshold": 0.0,
        }
        retriever.result_fuser = ResultFuser({"similarity_threshold": 0.0, "enable_deduplication": False})
        retriever.reranker = FakeReranker()
        retriever.paper_retriever = StaticRetriever()
        retriever.element_retriever = StaticRetriever(
            dense_results=[
                RetrievalResult(
                    id="elem-1",
                    content="较弱命中",
                    score=0.9,
                    level="element",
                    metadata={"parent_id": "sec-1", "paper_id": "paper-1", "section_id": "sec-1", "page": 2},
                ),
                RetrievalResult(
                    id="elem-2",
                    content="更强命中",
                    score=0.7,
                    level="element",
                    metadata={"parent_id": "sec-2", "paper_id": "paper-1", "section_id": "sec-2", "page": 5},
                ),
            ]
        )
        retriever.section_collection = FakeCollection(
            {
                "sec-1": {
                    "document": "章节一",
                    "metadata": {"section_id": "sec-1", "paper_id": "paper-1", "parent_id": "paper-1", "section_title": "Section One", "section_type": "other", "start_page": 2},
                },
                "sec-2": {
                    "document": "章节二",
                    "metadata": {"section_id": "sec-2", "paper_id": "paper-1", "parent_id": "paper-1", "section_title": "Section Two", "section_type": "other", "start_page": 5},
                },
            }
        )
        retriever.paper_collection = FakeCollection(
            {
                "paper-1": {
                    "document": "整篇论文主体内容。",
                    "metadata": {"paper_id": "paper-1", "title": "demo.pdf", "authors": "", "venue": "", "year": 2024},
                }
            }
        )
        retriever.context_builder = ContextBuilder(max_context_length=4000)

        payload = retriever.retrieve("哪个章节更相关", topn=2, intent="fact_qa")

        self.assertEqual(payload["structured"][0]["elements"][0]["id"], "elem-2")

    def test_parent_section_score_rewards_consistent_multiple_children(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        section_one_score, _ = retriever._compute_parent_section_score(
            [
                RetrievalResult(id="elem-1", content="a", score=0.92, level="element", metadata={}),
            ]
        )
        section_two_score, _ = retriever._compute_parent_section_score(
            [
                RetrievalResult(id="elem-2", content="b", score=0.82, level="element", metadata={}),
                RetrievalResult(id="elem-3", content="c", score=0.79, level="element", metadata={}),
            ]
        )

        self.assertGreater(section_two_score, section_one_score)

    def test_parent_section_score_uses_intent_specific_bonus_weights(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.config = {
            "support_bonus_weight": 0.2,
            "child_count_bonus_weight": 0.1,
            "support_bonus_weight_fact": 0.25,
            "child_count_bonus_weight_fact": 0.05,
            "support_bonus_weight_overview": 0.15,
            "child_count_bonus_weight_overview": 0.15,
        }
        children = [
            RetrievalResult(id="elem-1", content="a", score=0.9, level="element", metadata={}),
            RetrievalResult(id="elem-2", content="b", score=0.8, level="element", metadata={}),
            RetrievalResult(id="elem-3", content="c", score=0.7, level="element", metadata={}),
        ]

        fact_score, fact_breakdown = retriever._compute_parent_section_score(children, intent="fact_qa")
        overview_score, overview_breakdown = retriever._compute_parent_section_score(children, intent="general_overview")

        self.assertGreater(overview_score, fact_score)
        self.assertGreater(fact_breakdown["support_bonus"], overview_breakdown["support_bonus"])
        self.assertLess(fact_breakdown["count_bonus"], overview_breakdown["count_bonus"])

    def test_document_overview_builds_selected_document_packet(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.config = {"parent_rerank_top_k": 3}
        retriever.paper_collection = FakeCollection({
            "doc-1_paper": {
                "document": "full paper body",
                "metadata": {
                    "paper_id": "doc-1_paper",
                    "title": "demo residual paper",
                    "abstract": "paper abstract",
                    "doc_hash": "doc-1",
                    "page_count": 9,
                    "chunk_count": 4,
                },
            },
        })
        retriever.section_collection = FakeCollection({
            "doc-1_paper_sec_0": {
                "document": "introduction representative content",
                "metadata": {
                    "section_id": "doc-1_paper_sec_0",
                    "paper_id": "doc-1_paper",
                    "section_title": "Introduction",
                    "section_hierarchy": "Introduction",
                    "section_number": "1",
                    "start_page": 1,
                    "end_page": 2,
                    "chunk_count": 1,
                    "doc_hash": "doc-1",
                },
            },
            "doc-1_paper_sec_1": {
                "document": "all authors contributed equally",
                "metadata": {
                    "section_id": "doc-1_paper_sec_1",
                    "paper_id": "doc-1_paper",
                    "section_title": "Author Contributions",
                    "section_hierarchy": "Author Contributions",
                    "section_number": "2",
                    "start_page": 9,
                    "end_page": 9,
                    "chunk_count": 1,
                    "doc_hash": "doc-1",
                },
            },
        })
        retriever.element_collection = FakeCollection({})

        payload = retriever.retrieve(
            "summarize the whole paper",
            topn=2,
            allowed_doc_hashes={"doc-1"},
            intent="document_overview",
        )

        self.assertEqual(payload["citations"][0]["type"], "paper")
        self.assertEqual(payload["structured"][0]["level"], "paper")
        self.assertEqual(payload["structured"][0]["sections"][0]["title"], "Introduction")
        self.assertIn("paper abstract", payload["blocks"][0])
        self.assertIn("introduction representative content", payload["blocks"][0])
        self.assertNotIn("all authors contributed equally", payload["blocks"][0])

    def test_document_overview_uses_selected_doc_hash_without_retrieval(self):
        class ExplodingRetriever:
            def retrieve(self, *_args, **_kwargs):
                raise AssertionError("retrieval should not run")

        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.config = {"parent_rerank_top_k": 3}
        retriever.paper_retriever = ExplodingRetriever()
        retriever.element_retriever = ExplodingRetriever()
        retriever.reranker = ExplodingRetriever()
        retriever.paper_collection = FakeCollection({
            "doc-1_paper": {
                "document": "residual paper body",
                "metadata": {"paper_id": "doc-1_paper", "title": "Residual Paper", "doc_hash": "doc-1"},
            },
            "doc-2_paper": {
                "document": "attention paper body",
                "metadata": {"paper_id": "doc-2_paper", "title": "Attention Paper", "doc_hash": "doc-2"},
            },
        })
        retriever.section_collection = FakeCollection({})
        retriever.element_collection = FakeCollection({})

        payload = retriever.retrieve(
            "summarize residual paper",
            topn=2,
            allowed_doc_hashes={"doc-2"},
            intent="document_overview",
        )

        self.assertEqual(payload["citations"][0]["paper_id"], "doc-2_paper")
        self.assertIn("Attention Paper", payload["blocks"][0])
        self.assertNotIn("Residual Paper", payload["blocks"][0])


if __name__ == "__main__":
    unittest.main()
