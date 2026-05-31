import argparse
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.api.app_config import build_runtime_config
from app.core.docling_academic_chunker import DoclingAcademicChunker
from app.core.docling_extractor import DoclingExtractor
from app.core.hierarchical_retriever import HierarchicalChromaRetriever, build_hierarchical_document_payloads
from app.core.hierarchical_types import FusedResult, RetrievalResult
from app.core.rag_common import ACADEMIC_TOKENIZER_LOCAL_DIRNAME, ACADEMIC_TOKENIZER_MODEL_NAME
from app.core.rag import EMBEDDING_MODEL_NAME, Rag
from app.core.rag_hash_utils import file_md5_hex_32
from app.services.embedding_lifecycle_service import EmbeddingLifecycleService
from app.services.rag_document_service import RagDocumentService
from app.services.rag_generation_service import DOCUMENT_SELECTION_REQUIRED_MESSAGE, RagGenerationService
from app.services.rag_index_service import RagIndexService
from app.services.rag_retrieval_service import DocumentSelectionRequiredError, RagRetrievalService
from app.services.session_store import SessionStore

try:
    from fastapi.testclient import TestClient
    from app.api.fastapi_app import create_app
except ModuleNotFoundError:
    TestClient = None
    create_app = None


class DummyEmbeddingFunction:
    def name(self):
        return "dummy_embedding_function"

    def is_legacy(self):
        return False

    def __call__(self, input):
        return [[0.1, 0.2, 0.3] for _ in input]


class RagRegressionTests(unittest.TestCase):
    def test_generation_window_clamps_when_max_new_tokens_exceeds_context(self):
        safe_new_tokens, max_src_len = RagGenerationService._resolve_generation_window(
            context_len=1024,
            max_new_tokens=4096,
        )

        self.assertEqual(safe_new_tokens, 1016)
        self.assertEqual(max_src_len, 1)

    def test_predict_stream_yields_incremental_deltas(self):
        rag = Rag.__new__(Rag)
        rag.enable_history = True
        rag.use_ollama = True
        rag.logger = MagicMock()
        rag._retrieval_service = MagicMock()
        rag._retrieval_service.build_prompt_and_references.return_value = ("prompt", [])
        gen = RagGenerationService(rag)
        gen.stream_generate_answer = lambda **kwargs: iter(["你", "好"])

        chunks = list(gen.predict_stream("问题", history=[], history_summary=""))

        self.assertEqual(chunks[0], "你")
        self.assertEqual(chunks[1], "好")
        self.assertEqual(chunks[-1]["history"][-1][0], "问题")
        self.assertTrue(chunks[-1]["history"][-1][1].startswith("你好"))
        self.assertIn("回答引用内容", chunks[-1]["history"][-1][1])

    def test_document_overview_without_selected_doc_prompts_without_retrieval(self):
        rag = Rag.__new__(Rag)
        rag.enable_history = True
        rag.use_ollama = True
        rag.logger = MagicMock()
        rag._retrieval_service = MagicMock()
        gen = RagGenerationService(rag)
        gen.stream_generate_answer = MagicMock(return_value=iter(["bad"]))

        chunks = list(gen.predict_stream(
            "summarize the whole paper",
            history=[],
            history_summary="",
            intent="document_overview",
            selected_doc_paths=[],
        ))

        self.assertEqual(chunks[0], DOCUMENT_SELECTION_REQUIRED_MESSAGE)
        self.assertEqual(chunks[-1]["history"], [["summarize the whole paper", DOCUMENT_SELECTION_REQUIRED_MESSAGE]])
        rag._retrieval_service.build_prompt_and_references.assert_not_called()
        gen.stream_generate_answer.assert_not_called()

    def test_document_overview_with_multiple_selected_docs_prompts_without_retrieval(self):
        rag = Rag.__new__(Rag)
        rag.enable_history = True
        rag.use_ollama = True
        rag.logger = MagicMock()
        rag._retrieval_service = MagicMock()
        gen = RagGenerationService(rag)
        gen.stream_generate_answer = MagicMock(return_value=iter(["bad"]))

        chunks = list(gen.predict_stream(
            "summarize the whole paper",
            history=[],
            history_summary="",
            intent="document_overview",
            selected_doc_paths=["/tmp/a.pdf", "/tmp/b.pdf"],
        ))

        self.assertEqual(chunks[0], DocumentSelectionRequiredError.SINGLE_DOCUMENT_MESSAGE)
        rag._retrieval_service.build_prompt_and_references.assert_not_called()
        gen.stream_generate_answer.assert_not_called()

    def test_predict_uses_public_history_compression_method(self):
        rag = Rag.__new__(Rag)
        rag.enable_history = True
        rag.use_ollama = True
        rag.logger = MagicMock()
        rag.history = []
        rag.history_summary = ""
        rag._retrieval_service = MagicMock()
        rag._retrieval_service.build_prompt_and_references.return_value = ("prompt", ["ref"])
        rag.compress_history_if_needed = lambda history, history_summary: (history + [["summary", "done"]], "compressed")
        gen = RagGenerationService(rag)
        gen.stream_generate_answer = lambda **kwargs: iter(["答", "案"])

        response, refs = gen.predict("问题")

        self.assertTrue(response.startswith("答案"))
        self.assertIn("回答引用内容", response)
        self.assertEqual(refs, ["ref"])
        self.assertEqual(rag.history_summary, "compressed")
        self.assertEqual(rag.history[-1], ["summary", "done"])

    @patch("app.services.rag_ingestion_service.file_md5_hex_32", return_value="doc-1")
    def test_add_corpus_skips_already_indexed_document(self, _hash_patch):
        rag = Rag.__new__(Rag)
        rag.chunk_metadata = {"0": {"doc_hash": "doc-1"}}
        rag.hierarchical_retriever = MagicMock()
        rag.hierarchical_retriever.get_index_counts.return_value = {"paper": 0, "section": 0, "element": 0}
        rag._document_service = MagicMock()
        rag._index_service = MagicMock()
        rag._index_service.is_doc_indexed.return_value = True

        rag.add_corpus(["/tmp/demo.pdf"])

        rag._document_service.build_doc_chunks.assert_not_called()
        rag._index_service.merge_corpus_files.assert_called_once_with(["/tmp/demo.pdf"])
        rag._index_service.sync_runtime_indexes.assert_not_called()

    @patch("app.services.rag_ingestion_service.file_md5_hex_32", return_value="doc-1")
    def test_add_corpus_does_not_persist_temp_files_when_disabled(self, _hash_patch):
        rag = Rag.__new__(Rag)
        rag.chunk_metadata = {}
        rag.hierarchical_retriever = MagicMock()
        rag.hierarchical_retriever.get_index_counts.return_value = {"paper": 1, "section": 1, "element": 1}
        rag._document_service = MagicMock()
        rag._document_service.build_doc_chunks.return_value = [{"text": "t", "metadata": {"doc_hash": "doc-1"}}]
        rag._index_service = MagicMock()
        rag._index_service.is_doc_indexed.return_value = False

        rag.add_corpus(["/tmp/upload.pdf"], persist_files=False)

        rag._index_service.merge_corpus_files.assert_not_called()
        rag._index_service.sync_runtime_indexes.assert_called_once()

    def test_hash_uses_full_file_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            same_prefix = b"A" * (1024 * 1024)
            first_path = str(Path(temp_dir) / "a.bin")
            second_path = str(Path(temp_dir) / "b.bin")
            with open(first_path, "wb") as first:
                first.write(same_prefix + b"FIRST")
            with open(second_path, "wb") as second:
                second.write(same_prefix + b"SECOND")

            first_hash = file_md5_hex_32(first_path)
            second_hash = file_md5_hex_32(second_path)

        self.assertNotEqual(first_hash, second_hash)

    def test_save_and_load_embedding_state_keep_runtime_indexes_in_sync(self):
        rag = Rag.__new__(Rag)
        rag.hierarchical_retriever = MagicMock()
        rag.corpus_files = ["/tmp/demo.pdf"]
        rag.save_corpus_emb_dir = "/tmp/cache"
        rag._index_service = MagicMock()
        rag.resolve_embedding_dir = lambda corpus_files=None: "/tmp/cache/abc123"

        save_dir = rag.save_corpus_emb()

        self.assertEqual(save_dir, "/tmp/cache/abc123")
        rag.hierarchical_retriever.save_persist_directory.assert_called_once_with("/tmp/cache/abc123")
        rag._index_service.switch_persist_directory.assert_called_once_with("/tmp/cache/abc123")

        rag.hierarchical_retriever.load_persist_directory.reset_mock()
        rag._index_service.switch_persist_directory.reset_mock()

        rag.load_corpus_emb("/tmp/cache/loaded")

        rag.hierarchical_retriever.load_persist_directory.assert_called_once_with("/tmp/cache/loaded")
        rag._index_service.switch_persist_directory.assert_called_once_with("/tmp/cache/loaded")

    def test_get_reference_results_uses_hierarchical_main_path(self):
        rag = Rag.__new__(Rag)
        rag.rerank_top_k = 2
        rag._expand_query = lambda query: [query]
        rag.query_expander = SimpleNamespace(classify_intent=lambda _query: "fact_qa")
        rag.hierarchical_retriever = MagicMock()
        rag.hierarchical_retriever.retrieve_child_candidates.return_value = [
            FusedResult(
                result=RetrievalResult(
                    id="elem-a",
                    content="层级结果A",
                    score=1.0,
                    level="element",
                    metadata={"section_title": "Intro"},
                ),
                fused_score=1.0,
                source_scores={},
            )
        ]
        rag.hierarchical_retriever.build_context_from_fused_results.return_value = {
            "blocks": ["层级结果A"],
            "citations": [{"section_title": "Intro"}],
        }

        retrieval = RagRetrievalService(rag)

        refs = retrieval.get_reference_results("残差连接")

        self.assertEqual(refs, [{"text": "层级结果A", "metadata": {"section_title": "Intro"}}])

    def test_get_reference_results_requires_selected_doc_for_document_overview(self):
        rag = Rag.__new__(Rag)
        rag.query_expander = SimpleNamespace(classify_intent=lambda _query: "document_overview")
        rag.hierarchical_retriever = MagicMock()

        retrieval = RagRetrievalService(rag)

        with self.assertRaises(DocumentSelectionRequiredError):
            retrieval.get_reference_results("summarize the whole paper", intent="document_overview")

        rag.hierarchical_retriever.retrieve.assert_not_called()

    @patch("app.services.rag_retrieval_service.file_md5_hex_32", return_value="hash-a")
    def test_get_reference_results_uses_document_overview_packet_without_expansion(self, _hash_patch):
        rag = Rag.__new__(Rag)
        rag._expand_query = MagicMock(side_effect=AssertionError("query expansion should not run"))
        rag.query_expander = SimpleNamespace(classify_intent=lambda _query: "document_overview")
        rag.hierarchical_retriever = MagicMock()
        rag.hierarchical_retriever.build_document_overview_context.return_value = {
            "blocks": ["document overview packet"],
            "citations": [{"paper_title": "Paper A", "section_title": "全文概述"}],
        }

        retrieval = RagRetrievalService(rag)

        refs = retrieval.get_reference_results(
            "summarize the whole paper",
            selected_doc_paths=["/tmp/a.pdf"],
            intent="document_overview",
        )

        self.assertEqual(refs[0]["text"], "document overview packet")
        rag.hierarchical_retriever.build_document_overview_context.assert_called_once_with(
            doc_hash="hash-a",
            query="summarize the whole paper",
        )
        rag.hierarchical_retriever.retrieve.assert_not_called()
        rag._expand_query.assert_not_called()

    @patch("app.services.rag_retrieval_service.file_md5_hex_32")
    def test_get_reference_results_requires_exactly_one_doc_for_document_overview(self, hash_patch):
        rag = Rag.__new__(Rag)
        rag.query_expander = SimpleNamespace(classify_intent=lambda _query: "document_overview")
        rag.hierarchical_retriever = MagicMock()
        hash_patch.side_effect = lambda path: {"a.pdf": "hash-a", "b.pdf": "hash-b"}[path]

        retrieval = RagRetrievalService(rag)

        with self.assertRaises(DocumentSelectionRequiredError) as ctx:
            retrieval.get_reference_results(
                "summarize the whole paper",
                selected_doc_paths=["a.pdf", "b.pdf"],
                intent="document_overview",
            )

        self.assertEqual(str(ctx.exception), DocumentSelectionRequiredError.SINGLE_DOCUMENT_MESSAGE)
        rag.hierarchical_retriever.build_document_overview_context.assert_not_called()

    @patch("app.services.rag_retrieval_service.file_md5_hex_32")
    def test_get_reference_results_passes_selected_doc_hashes(self, hash_patch):
        rag = Rag.__new__(Rag)
        rag.rerank_top_k = 2
        rag._expand_query = lambda query: [query]
        rag.query_expander = SimpleNamespace(classify_intent=lambda _query: "fact_qa")
        rag.hierarchical_retriever = MagicMock()
        hash_patch.side_effect = lambda path: {
            "/tmp/a.pdf": "hash-a",
            "/tmp/b.pdf": "hash-b",
        }[path]

        retrieval = RagRetrievalService(rag)
        captured = {}

        def fake_retrieve_child_candidates(query, topn, allowed_doc_hashes=None, intent=None):
            captured["query"] = query
            captured["topn"] = topn
            captured["allowed_doc_hashes"] = allowed_doc_hashes
            captured["intent"] = intent
            return [
                FusedResult(
                    result=RetrievalResult(
                        id="elem-a",
                        content="命中A",
                        score=1.0,
                        level="element",
                        metadata={"section_title": "Intro"},
                    ),
                    fused_score=1.0,
                    source_scores={},
                )
            ]

        rag.hierarchical_retriever.retrieve_child_candidates.side_effect = fake_retrieve_child_candidates
        rag.hierarchical_retriever.build_context_from_fused_results.return_value = {
            "blocks": ["命中A"],
            "citations": [{"section_title": "Intro"}],
        }

        refs = retrieval.get_reference_results(
            "残差连接",
            selected_doc_paths=["/tmp/a.pdf", "/tmp/b.pdf"],
        )

        self.assertEqual(captured["query"], "残差连接")
        self.assertEqual(captured["allowed_doc_hashes"], {"hash-a", "hash-b"})
        self.assertEqual(captured["intent"], "fact_qa")
        self.assertEqual(refs, [{"text": "命中A", "metadata": {"section_title": "Intro"}}])

    def test_get_reference_results_reranks_multi_query_hits_with_count_and_section_score(self):
        rag = Rag.__new__(Rag)
        rag.rerank_top_k = 2
        rag._expand_query = lambda query: ["q1", "q2", "q3"]
        rag.query_expander = SimpleNamespace(classify_intent=lambda _query: "fact_qa")
        rag.hierarchical_retriever = MagicMock()

        retrieval = RagRetrievalService(rag)

        def make_item(item_id, score):
            return FusedResult(
                result=RetrievalResult(
                    id=item_id,
                    content=item_id,
                    score=score,
                    level="element",
                    metadata={"section_title": item_id},
                ),
                fused_score=score,
                source_scores={},
            )

        def fake_retrieve_child_candidates(query, topn, allowed_doc_hashes=None, intent=None):
            _ = topn, allowed_doc_hashes, intent
            if query == "q1":
                return [make_item("A", 0.9), make_item("B", 0.8), make_item("C", 0.7)]
            if query == "q2":
                return [make_item("A", 0.7), make_item("D", 0.6)]
            return [make_item("A", 0.5)]

        captured_final = {}

        def fake_build_context(final, query):
            captured_final["ids"] = [item.result.id for item in final]
            return {
                "blocks": [item.result.content for item in final],
                "citations": [item.result.metadata for item in final],
            }

        rag.hierarchical_retriever.retrieve_child_candidates.side_effect = fake_retrieve_child_candidates
        rag.hierarchical_retriever.build_context_from_fused_results.side_effect = fake_build_context

        refs = retrieval.get_reference_results("残差连接")

        self.assertEqual(captured_final["ids"][0], "A")
        self.assertEqual(refs[0], {"text": "A", "metadata": {"section_title": "A"}})

    def test_try_hierarchical_retrieve_prefers_structured_blocks_over_display_context(self):
        rag = Rag.__new__(Rag)
        rag.rerank_top_k = 2
        rag.hierarchical_retriever = MagicMock()
        rag.hierarchical_retriever.retrieve.return_value = {
            "context": "用户问题: q\n\n相关文献内容:\n" + "=" * 50 + "\n包装文本",
            "blocks": ["块A", "块B"],
        }

        retrieval = RagRetrievalService(rag)

        refs = retrieval.try_hierarchical_retrieve("残差连接")

        self.assertEqual(refs, [{"text": "块A", "metadata": {}}, {"text": "块B", "metadata": {}}])


class HierarchicalRetrieverPersistenceTests(unittest.TestCase):
    @patch("app.core.hierarchical_ranking.ResultReranker._load_model", return_value=None)
    @patch("app.core.hierarchical_retriever.ProjectEmbeddingFunction", return_value=DummyEmbeddingFunction())
    def test_save_corpus_embeddings_preserves_hierarchical_collections(self, _embedding_patch, _reranker_patch):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            retriever = HierarchicalChromaRetriever(
                persist_directory=source_dir,
                embedding_model_name_or_path="dummy",
                device="cpu",
            )
            retriever.index_document(
                doc_hash="doc-1",
                doc_file="/tmp/demo.pdf",
                chunk_items=[{
                    "text": "attention residual block",
                    "metadata": {"section_title": "Intro", "source_type": "text", "page": 2},
                }],
            )
            retriever.save_persist_directory(target_dir)

            reloaded = HierarchicalChromaRetriever(
                persist_directory=target_dir,
                embedding_model_name_or_path="dummy",
                device="cpu",
            )
            self.assertIn("doc-1_paper", reloaded.retriever.paper_collection.get().get("ids", []))
            self.assertIn("doc-1_paper_sec_0", reloaded.retriever.section_collection.get().get("ids", []))
            self.assertIn("doc-1_paper_sec_0_elem_0", reloaded.retriever.element_collection.get().get("ids", []))


class RagDocumentServiceRefactorTests(unittest.TestCase):
    @patch("app.core.docling_academic_chunker.load_hybrid_chunker")
    def test_docling_academic_chunker_counts_tokens_with_tokenizer(self, hybrid_chunker_patch):
        hybrid_chunker_patch.return_value = MagicMock()

        class FakeTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [101, 11, 22, 33, 102]}

        with patch(
            "app.core.docling_academic_chunker.DoclingAcademicChunker._load_tokenizer_local_first",
            return_value=FakeTokenizer(),
        ):
            chunker = DoclingAcademicChunker(config={"max_tokens": 256, "overlap_tokens": 32})

        self.assertEqual(chunker._count_tokens("任意文本"), 5)

    @patch("app.core.docling_academic_chunker.DoclingAcademicChunker._load_tokenizer_local_first", return_value=None)
    @patch("app.core.docling_academic_chunker.load_hybrid_chunker")
    def test_docling_academic_chunker_explicitly_enables_merge_peers(self, hybrid_chunker_patch, _tokenizer_patch):
        hybrid_chunker_cls = MagicMock()
        hybrid_chunker_patch.return_value = hybrid_chunker_cls

        DoclingAcademicChunker(config={"max_tokens": 256, "overlap_tokens": 32})

        hybrid_chunker_cls.assert_called_once()
        self.assertTrue(hybrid_chunker_cls.call_args.kwargs["merge_peers"])

    @patch("app.services.rag_document_service.create_docling_academic_chunker")
    def test_docling_chunker_uses_embedding_tokenizer_configuration(self, create_chunker_patch):
        rag = MagicMock()
        rag.docling_academic_chunker = None
        rag.docling_extractor = object()
        rag.chunk_size = 256
        rag.chunk_overlap = 32
        rag.save_corpus_emb_dir = "/tmp/cache"

        service = RagDocumentService(rag)
        service.ensure_docling_academic_chunker()

        create_chunker_patch.assert_called_once()
        config = create_chunker_patch.call_args.args[0]
        self.assertEqual(config["tokenizer_model"], EMBEDDING_MODEL_NAME)
        self.assertEqual(config["tokenizer_model"], ACADEMIC_TOKENIZER_MODEL_NAME)
        self.assertEqual(config["tokenizer_local_dir"], "/tmp/cache/models/all-MiniLM-L12-v2")
        self.assertEqual(config["tokenizer_local_dir"].split("/")[-1], ACADEMIC_TOKENIZER_LOCAL_DIRNAME)

    def test_build_doc_chunks_uses_hybrid_chunks_directly(self):
        rag = MagicMock()
        rag.chunk_size = 256
        rag.chunk_overlap = 32
        rag.save_corpus_emb_dir = "/tmp/cache"
        rag.docling_extractor = MagicMock()
        rag.docling_extractor.extract_document.return_value = object()
        rag.docling_academic_chunker = MagicMock()
        rag.docling_academic_chunker.chunk_document.return_value = [
            ("attention residual block", {"section_title": "Intro", "section_hierarchy": "Intro", "source_type": "text", "page": 2})
        ]

        service = RagDocumentService(rag)
        items = service.build_doc_chunks("/tmp/demo.pdf", "doc-1")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["text"], "attention residual block")
        self.assertEqual(items[0]["metadata"]["doc_hash"], "doc-1")
        self.assertEqual(items[0]["metadata"]["doc_file"], "/tmp/demo.pdf")

class DoclingMetadataExtractionTests(unittest.TestCase):
    def test_docling_extractor_safe_page_reads_prov_page_number(self):
        item = SimpleNamespace(prov=[{"page_no": 7}])

        self.assertEqual(DoclingExtractor._safe_page(item), 7)

    @patch("app.core.docling_academic_chunker.DoclingAcademicChunker._load_tokenizer_local_first", return_value=None)
    @patch("app.core.docling_academic_chunker.load_hybrid_chunker")
    def test_docling_academic_chunker_reads_section_and_page_from_docling_meta(self, hybrid_chunker_patch, _tokenizer_patch):
        chunk = SimpleNamespace(
            text="Residual learning eases optimization.",
            meta=SimpleNamespace(
                headings=["1. Introduction"],
                doc_items=[{"prov": [{"page_no": 3}]}],
            ),
        )

        class FakeHybridChunker:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def chunk(self, _doc):
                return [chunk]

        hybrid_chunker_patch.return_value = FakeHybridChunker

        chunker = DoclingAcademicChunker(config={"max_tokens": 256, "overlap_tokens": 32})
        chunks = chunker.chunk_document(object())

        self.assertEqual(len(chunks), 1)
        text, metadata = chunks[0]
        self.assertEqual(text, "Residual learning eases optimization.")
        self.assertEqual(metadata["section_title"], "1. Introduction")
        self.assertEqual(metadata["section_hierarchy"], "1. Introduction")
        self.assertEqual(metadata["page"], 3)

    @patch("app.core.docling_academic_chunker.DoclingAcademicChunker._load_tokenizer_local_first", return_value=None)
    @patch("app.core.docling_academic_chunker.load_hybrid_chunker")
    def test_docling_academic_chunker_detects_table_from_doc_items_labels(self, hybrid_chunker_patch, _tokenizer_patch):
        chunk = SimpleNamespace(
            text="table row",
            meta=SimpleNamespace(
                headings=["2. Results"],
                doc_items=[{"label": "table", "prov": [{"page_no": 4}]}],
            ),
        )

        class FakeHybridChunker:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def chunk(self, _doc):
                return [chunk]

        hybrid_chunker_patch.return_value = FakeHybridChunker
        chunker = DoclingAcademicChunker(config={"max_tokens": 256, "overlap_tokens": 32})
        chunks = chunker.chunk_document(object())

        self.assertEqual(len(chunks), 1)
        _, metadata = chunks[0]
        self.assertEqual(metadata["source_type"], "table")
        self.assertEqual(metadata["page"], 4)

    @patch("app.core.docling_academic_chunker.DoclingAcademicChunker._load_tokenizer_local_first", return_value=None)
    @patch("app.core.docling_academic_chunker.load_hybrid_chunker")
    def test_docling_academic_chunker_keeps_table_chunk_independent(self, hybrid_chunker_patch, _tokenizer_patch):
        table_chunk = SimpleNamespace(
            text="r1 c1",
            meta=SimpleNamespace(
                headings=["2. Results"],
                doc_items=[{"label": "table"}],
            ),
        )
        text_chunk = SimpleNamespace(
            text="short explanation",
            meta=SimpleNamespace(
                headings=["2. Results"],
                doc_items=[{"label": "text"}],
            ),
        )

        class FakeHybridChunker:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def chunk(self, _doc):
                return [table_chunk, text_chunk]

        hybrid_chunker_patch.return_value = FakeHybridChunker
        chunker = DoclingAcademicChunker(config={"max_tokens": 256, "overlap_tokens": 32})
        chunks = chunker.chunk_document(object())

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0][1]["source_type"], "table")
        self.assertEqual(chunks[1][1]["source_type"], "text")


class RagServiceLocatorRefactorTests(unittest.TestCase):
    def test_document_service_is_initialized(self):
        rag = Rag.__new__(Rag)
        rag.logger = None

        from app.services.rag_document_service import RagDocumentService
        rag._document_service = RagDocumentService(rag)

        self.assertIsInstance(rag._document_service, RagDocumentService)


class SessionStoreRefactorTests(unittest.TestCase):
    def test_session_store_centralizes_session_listing_and_payload_building(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(str(Path(temp_dir) / "sessions.db"))
            sessions = {}
            session_id = "11111111-1111-1111-1111-111111111111"
            store.create_session(sessions, session_id)
            store.store_chat_turn(sessions, session_id, "你好", "你好呀")
            store.rename_session(sessions, session_id, "测试会话")

            listing = store.list_sessions(sessions)
            payload = store.get_session_payload(sessions, session_id)

            self.assertEqual(listing[0]["title"], "测试会话")
            self.assertEqual(listing[0]["message_count"], 2)
            self.assertEqual(payload["messages"][0], {"role": "user", "content": "你好"})
            self.assertEqual(payload["messages"][1], {"role": "assistant", "content": "你好呀"})

    def test_session_store_applies_background_compression_only_when_history_matches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(str(Path(temp_dir) / "sessions.db"))
            sessions = {}
            session_id = "11111111-1111-1111-1111-111111111111"
            store.create_session(sessions, session_id)
            original_history = store.store_chat_turn(sessions, session_id, "问题", "答案")

            applied = store.apply_compression_result(
                sessions=sessions,
                session_id=session_id,
                history_to_compress=original_history,
                updated_history=[["summary", "done"]],
                updated_summary="compressed",
            )

            self.assertTrue(applied)
            self.assertEqual(sessions[session_id]["history"], [["summary", "done"]])
            self.assertEqual(sessions[session_id]["summary"], "compressed")

    def test_session_store_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session_path = Path(temp_dir) / "sessions.db"
            store_a = SessionStore(str(session_path))
            sessions = {}
            session_id = "11111111-1111-1111-1111-111111111111"
            store_a.create_session(sessions, session_id)
            store_a.store_chat_turn(sessions, session_id, "问题", "答案")
            store_a.rename_session(sessions, session_id, "测试")

            store_b = SessionStore(str(session_path))
            loaded = store_b.load()
            self.assertIn(session_id, loaded)
            self.assertEqual(loaded[session_id]["title"], "测试")
            self.assertEqual(loaded[session_id]["history"], [["问题", "答案"]])

    def test_session_store_normalize_invalid_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(str(Path(temp_dir) / "sessions.db"))
            sessions = {}
            session_id = "11111111-1111-1111-1111-111111111111"
            store.create_session(sessions, session_id)
            sessions[session_id]["history"] = [["问题", "回答"], ["only-question"], "bad"]

            payload = store.get_session_payload(sessions, session_id)
            self.assertGreater(len(payload["messages"]), 0)

    def test_session_store_rejects_legacy_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(str(Path(temp_dir) / "sessions.db"))
            sessions = {}
            session_id = "11111111-1111-1111-1111-111111111111"
            store.create_session(sessions, session_id)
            sessions[session_id] = [["旧问题", "旧回答"]]

            payload = store.get_session_payload(sessions, session_id)
            self.assertIn("messages", payload)

    def test_session_store_normalize_history_types(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(str(Path(temp_dir) / "sessions.db"))
            sessions = {}
            session_id = "11111111-1111-1111-1111-111111111111"
            store.create_session(sessions, session_id)
            sessions[session_id]["history"] = [[{"q": 1}, ["a"]]]

            payload = store.get_session_payload(sessions, session_id)
            self.assertGreater(len(payload["messages"]), 0)


class EmbeddingLifecycleServiceRefactorTests(unittest.TestCase):
    def test_resolve_local_embedding_state_reuses_single_path_rule(self):
        corpus_service = MagicMock()
        corpus_service.list_local_corpus_files.return_value = ["/tmp/a.pdf", "/tmp/b.pdf"]
        model = MagicMock()
        model.resolve_embedding_dir.return_value = "/tmp/cache/abc123"

        service = EmbeddingLifecycleService(corpus_service)
        with patch("app.services.embedding_lifecycle_service.os.path.isdir", return_value=True):
            state = service.resolve_local_embedding_state(model)

        self.assertEqual(state["corpus_files"], ["/tmp/a.pdf", "/tmp/b.pdf"])
        self.assertEqual(state["embedding_dir"], "/tmp/cache/abc123")
        self.assertTrue(state["embedding_exists"])
        model.resolve_embedding_dir.assert_called_once_with(["/tmp/a.pdf", "/tmp/b.pdf"])

    def test_resolve_local_embedding_state_accepts_persisted_chroma_store(self):
        corpus_service = MagicMock()
        corpus_service.list_local_corpus_files.return_value = ["/tmp/a.pdf"]
        model = MagicMock()
        model.resolve_embedding_dir.return_value = "/tmp/cache/missing"
        model.hierarchical_retriever.get_index_counts.return_value = {"element": 0}

        with tempfile.TemporaryDirectory() as tmp_dir:
            chroma_dir = Path(tmp_dir) / "chroma"
            chroma_dir.mkdir()
            with sqlite3.connect(chroma_dir / "chroma.sqlite3") as conn:
                conn.execute("CREATE TABLE embeddings (id TEXT)")
                conn.execute("INSERT INTO embeddings VALUES ('chunk-1')")

            model.chroma_persist_directory = str(chroma_dir)
            service = EmbeddingLifecycleService(corpus_service)
            state = service.resolve_local_embedding_state(model)

        self.assertTrue(state["embedding_exists"])
        self.assertTrue(state["persisted_index_exists"])
        self.assertFalse(state["snapshot_exists"])
        self.assertEqual(state["embedding_dir"], str(chroma_dir))
        self.assertEqual(state["chunk_count"], 0)

    def test_bootstrap_local_corpus_prefers_existing_embedding_dir(self):
        corpus_service = MagicMock()
        corpus_service.list_local_corpus_files.return_value = ["/tmp/a.pdf"]
        model = MagicMock()
        model.resolve_embedding_dir.return_value = "/tmp/cache/abc123"
        model.hierarchical_retriever.get_index_counts.return_value = {"element": 1}

        service = EmbeddingLifecycleService(corpus_service)
        with patch("app.services.embedding_lifecycle_service.os.path.isdir", return_value=True):
            service.bootstrap_local_corpus(model, MagicMock())

        model.load_corpus_emb.assert_called_once_with("/tmp/cache/abc123")
        model.add_corpus.assert_not_called()

    def test_save_embeddings_runs_under_same_lock_guard(self):
        corpus_service = MagicMock()
        corpus_service.list_local_corpus_files.return_value = ["/tmp/a.pdf"]
        model = MagicMock()
        model.hierarchical_retriever.has_indexed_content.return_value = True
        model.hierarchical_retriever.get_index_counts.return_value = {"element": 1}
        model.save_corpus_emb.return_value = "/tmp/cache/abc123"
        rag_lock = MagicMock()

        service = EmbeddingLifecycleService(corpus_service)
        payload = service.save_embeddings(model, rag_lock)

        self.assertEqual(payload["embedding_dir"], "/tmp/cache/abc123")
        self.assertEqual(payload["chunk_count"], 1)
        model.save_corpus_emb.assert_called_once()
        self.assertEqual(model.corpus_files, ["/tmp/a.pdf"])
        rag_lock.__enter__.assert_called_once()
        rag_lock.__exit__.assert_called_once()


class RagIndexServiceRefactorTests(unittest.TestCase):
    def test_merge_corpus_files_deduplicates_and_updates_rag_state(self):
        rag = MagicMock()
        rag.corpus_files = ["/tmp/a.pdf"]

        service = RagIndexService(rag)
        merged = service.merge_corpus_files(["/tmp/a.pdf", "/tmp/b.pdf"])

        self.assertEqual(merged, ["/tmp/a.pdf", "/tmp/b.pdf"])
        self.assertEqual(rag.corpus_files, ["/tmp/a.pdf", "/tmp/b.pdf"])

    def test_switch_persist_directory_keeps_reload_hook_compatible(self):
        rag = MagicMock()
        rag.corpus_files = ["/tmp/a.pdf"]
        service = RagIndexService(rag)
        service.sync_runtime_indexes = MagicMock()

        service.switch_persist_directory("/tmp/cache")

        rag.hierarchical_retriever.switch_persist_directory.assert_called_once_with("/tmp/cache")
        service.sync_runtime_indexes.assert_called_once()

    def test_sync_runtime_indexes_reads_metadata_from_hierarchical_retriever(self):
        rag = MagicMock()
        rag.hierarchical_retriever.build_element_metadata_index.return_value = {"elem-1": {"doc_hash": "doc-1"}}

        service = RagIndexService(rag)
        service.sync_runtime_indexes()

        rag.hierarchical_retriever.refresh_indexes.assert_called_once()
        self.assertEqual(rag.chunk_metadata, {"elem-1": {"doc_hash": "doc-1"}})


class HierarchicalRetrieverRefactorTests(unittest.TestCase):
    def test_build_hierarchical_document_payloads_reuses_single_schema(self):
        payloads = build_hierarchical_document_payloads(
            doc_hash="doc-1",
            doc_file="/tmp/demo.pdf",
            chunk_items=[{
                "text": "attention residual block",
                "metadata": {"section_title": "Intro", "section_hierarchy": "Intro", "source_type": "text", "page": 2},
            }],
        )

        self.assertEqual(set(payloads.keys()), {"paper", "section", "element"})
        self.assertEqual(payloads["paper"]["ids"], ["doc-1_paper"])
        self.assertNotIn("parent_id", payloads["paper"]["metadatas"][0])
        self.assertEqual(payloads["section"]["ids"], ["doc-1_paper_sec_0"])
        self.assertEqual(payloads["element"]["metadatas"][0]["section_id"], "doc-1_paper_sec_0")
        self.assertEqual(payloads["element"]["documents"], ["attention residual block"])
        self.assertEqual(payloads["element"]["ids"], ["doc-1_paper_sec_0_elem_0"])

    def test_build_hierarchical_document_payloads_use_leaf_section_title(self):
        payloads = build_hierarchical_document_payloads(
            doc_hash="doc-1",
            doc_file="/tmp/demo.pdf",
            chunk_items=[{
                "text": "motivation details",
                "metadata": {
                    "section_title": "Introduction",
                    "section_hierarchy": "Introduction > Motivation",
                    "source_type": "text",
                    "page": 2,
                },
            }],
        )

        self.assertEqual(payloads["section"]["metadatas"][0]["section_title"], "Motivation")
        self.assertEqual(payloads["section"]["metadatas"][0]["section_hierarchy"], "Introduction > Motivation")
        self.assertEqual(payloads["element"]["metadatas"][0]["section_title"], "Motivation")


@unittest.skipIf(TestClient is None or create_app is None, "fastapi not installed")
class FastApiArchitectureRefactorTests(unittest.TestCase):
    @staticmethod
    def _build_args(**overrides):
        base = {
            "gen_model_type": "ollama",
            "gen_model_name": "qwen2.5:3b",
            "lora_model": None,
            "rerank_model_name": "BAAI/bge-reranker-base",
            "device": None,
            "corpus_files": "",
            "int4": False,
            "int8": False,
            "chunk_size": 256,
            "chunk_overlap": 50,
            "history_max_turns": 6,
            "history_keep_last_turns": 2,
            "ollama_host": "http://127.0.0.1:11434",
            "docling_use_ocr": True,
            "docling_table_structure": True,
            "query_expansion": False,
            "chroma_persist_directory": "/tmp/chroma",
        }
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_build_runtime_config_centralizes_api_runtime_settings(self):
        config = build_runtime_config(self._build_args(corpus_files="a.pdf, b.pdf"))

        self.assertTrue(str(config.local_corpus_dir).endswith("data/local_corpus"))
        self.assertTrue(str(config.index_html_path).endswith("app/utils/index.html"))
        self.assertEqual(config.model_init_kwargs["corpus_files"], ["a.pdf", "b.pdf"])
        self.assertTrue(config.model_init_kwargs["docling_table_structure"])
        self.assertEqual(config.cors_origins, ("http://localhost:8000", "http://127.0.0.1:8000"))

    def test_create_app_registers_routes_around_shared_context(self):
        config = build_runtime_config(self._build_args())
        fake_model = MagicMock()
        fake_model.hierarchical_retriever.get_index_counts.return_value = {"element": 0}

        with patch("os.path.isdir", return_value=True), patch("urllib.request.urlopen"), patch(
            "app.api.fastapi_app.RagBootstrap.create_runtime",
            return_value=fake_model,
        ) as create_runtime:
            app = create_app(config)
            app.state.context.embedding_service.bootstrap_local_corpus = MagicMock()
            app.state.context.sessions["11111111-1111-1111-1111-111111111111"] = {
                "title": "测试会话",
                "updated_at": 1,
                "created_at": 1,
                "history": [["你好", "你好呀"]],
                "summary": "",
            }

            with TestClient(app) as client:
                health_response = client.get("/api/health")
                sessions_response = client.get("/api/sessions")

        self.assertEqual(health_response.status_code, 200)
        self.assertTrue(health_response.json()["ok"])
        self.assertEqual(sessions_response.status_code, 200)
        self.assertIn(
            "测试会话",
            [session["title"] for session in sessions_response.json()["sessions"]],
        )
        create_runtime.assert_called_once_with(config.model_init_kwargs)
        app.state.context.embedding_service.bootstrap_local_corpus.assert_called_once_with(
            fake_model,
            app.state.context.rag_lock,
        )


if __name__ == "__main__":
    unittest.main()
