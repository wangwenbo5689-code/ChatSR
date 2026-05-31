# -*- coding: utf-8 -*-
"""
RAG 主类
"""

import os
from typing import Union, List, Dict, Any, Optional

import torch
from loguru import logger

from app.core.docling_extractor import create_docling_extractor
from app.core.hierarchical_retriever import HierarchicalChromaRetriever
from app.core.rag_defaults import EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_DIRNAME
from app.core.rag_common import (
    ACADEMIC_TOKENIZER_MODEL_NAME, ACADEMIC_TOKENIZER_LOCAL_DIRNAME,
    DEFAULT_GENERATION_LIMITS, DEFAULT_RERANK_MODEL_NAME,
    DEFAULT_RETRIEVAL_STRATEGY, _merge_generation_limits,
    _merge_retrieval_strategy, add_source_numbers, make_chunk_item,
)
from app.core.rag_generation_model import init_generation_model, resolve_default_device
from app.core.rag_history_manager import RagHistoryManager
from app.core.rag_hash_utils import resolve_embedding_dir as resolve_embedding_dir_from_files
from app.core.rag_prompts import PROMPT_TEMPLATE
from app.core.rag_query_expander import RagQueryExpander
from app.services.rag_document_service import RagDocumentService
from app.services.rag_generation_service import RagGenerationService
from app.services.rag_index_service import RagIndexService
from app.services.rag_ingestion_service import RagIngestionService
from app.services.rag_persistence_service import RagPersistenceService
from app.services.rag_retrieval_service import RagRetrievalService


class Rag:
    def __init__(
            self,
            generate_model_type: str = "ollama",
            generate_model_name_or_path: str = "qwen2.5:3b",
            lora_model_name_or_path: Optional[str] = None,
            corpus_files: Optional[Union[str, List[str]]] = None,
            save_corpus_emb_dir: str = "./corpus_embs/",
            device: Optional[str] = None,
            int8: bool = False,
            int4: bool = False,
            chunk_size: int = 256,
            chunk_overlap: int = 50,
            rerank_model_name_or_path: Optional[str] = None,
            enable_history: bool = True,
            rerank_top_k: int = 3,
            ollama_host: Optional[str] = None,
            history_summary: str = "",
            docling_use_ocr: bool = True,
            docling_table_structure: bool = True,
            query_expansion: bool = True,
            chroma_persist_directory: str = "./corpus_embs/chroma",
            generation_limits: Optional[Dict[str, Any]] = None,
            retrieval_strategy: Optional[Dict[str, Any]] = None,
    ):
        """初始化 RAG 模型

        Args:
            generate_model_type: 生成模型类型，默认为 "ollama"
            generate_model_name_or_path: 生成模型名称或路径，默认为 "qwen2.5:3b"（Ollama）
            lora_model_name_or_path: LoRA 模型名称或路径
            corpus_files: 语料文件
            save_corpus_emb_dir: 保存语料嵌入的目录，默认为 ./corpus_embs/
            device: 设备，默认为 None，自动选择 GPU 或 CPU
            int8: 是否使用 int8 量化，默认为 False
            int4: 是否使用 int4 量化，默认为 False
            chunk_size: 分块大小，默认为 256
            chunk_overlap: 分块重叠，默认为 50
            rerank_model_name_or_path: 重排模型名称或路径，默认为 'BAAI/bge-reranker-base'
            enable_history: 是否启用历史记录，默认为 True
            rerank_top_k: 层次检索和重排后保留的 top-k 参考
            ollama_host: Ollama 主机地址，默认为 "http://127.0.0.1:11434"
            history_summary: 当前会话的压缩记忆
            docling_use_ocr: 是否在 Docling 提取器中启用 OCR
            docling_table_structure: 是否启用 Docling 表格结构提取（TableFormer）
            query_expansion: 是否启用查询扩展
            chroma_persist_directory: Chroma 持久化目录
        """

        self.device: Any = device or resolve_default_device()
        self.chunk_metadata: Dict[str, Dict[str, Any]] = {}
        self.logger = logger
        self.prompt_template = PROMPT_TEMPLATE
        self.save_corpus_emb_dir = save_corpus_emb_dir
        self.rerank_model_name_or_path = rerank_model_name_or_path or DEFAULT_RERANK_MODEL_NAME
        self.chroma_persist_directory = chroma_persist_directory
        self.use_docling = True
        self.docling_academic_chunker: Any = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._init_docling(docling_use_ocr, docling_table_structure)
        self._init_retriever(rerank_model_name_or_path)
        self._init_generator(
            generate_model_type, generate_model_name_or_path,
            lora_model_name_or_path, ollama_host, int8, int4,
        )
        self._init_config(
            enable_history, rerank_top_k, history_summary,
            query_expansion, generation_limits, retrieval_strategy,
        )
        self._init_services()

        self._index_service.sync_runtime_indexes()
        self.corpus_files = corpus_files
        if corpus_files:
            self.add_corpus(corpus_files)

    def _init_docling(self, docling_use_ocr: bool, docling_table_structure: bool):
        self.docling_extractor = create_docling_extractor(
            use_ocr=docling_use_ocr,
            ocr_language=None,
            enable_table_structure=docling_table_structure,
            table_structure_mode="accurate",
        )
        if not self.docling_extractor:
            raise RuntimeError("Docling is required but not available.")
        logger.info("Docling document extractor enabled")

    def _init_retriever(self, rerank_model_name_or_path):
        self.hierarchical_retriever = HierarchicalChromaRetriever(
            persist_directory=self.chroma_persist_directory,
            embedding_model_name_or_path=EMBEDDING_MODEL_NAME,
            device=str(self.device),
            rerank_model_name_or_path=rerank_model_name_or_path or DEFAULT_RERANK_MODEL_NAME,
        )
        logger.info(f"Using hierarchical Chroma backend: {self.chroma_persist_directory}")

    def _init_generator(self, gen_model_type, gen_model_name_or_path, lora_name, ollama_host, int8, int4):
        if gen_model_type != "ollama":
            raise ValueError("当前版本仅支持 gen_model_type='ollama'")
        self.use_ollama = gen_model_type == "ollama"
        self.ollama_host: Optional[str] = None
        self.ollama_model: Optional[str] = None
        self.gen_model: Any = None
        self.tokenizer: Any = None

        if self.use_ollama:
            self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
            self.ollama_model = gen_model_name_or_path
            logger.info(f"Using Ollama model: {self.ollama_model} at {self.ollama_host}")
        else:
            self.gen_model, self.tokenizer = self._init_gen_model(
                gen_model_type, gen_model_name_or_path,
                peft_name=lora_name, int8=int8, int4=int4,
            )

    def _init_services(self):
        self.history_manager = RagHistoryManager(self.stream_generate_answer)
        self.query_expander = RagQueryExpander(self._generate_summary)
        self._index_service = RagIndexService(self)
        self._document_service = RagDocumentService(self)
        self._ingestion_service = RagIngestionService(self)
        self._retrieval_service = RagRetrievalService(self)
        self._generation_service = RagGenerationService(self)
        self._persistence_service = RagPersistenceService(self)

    def _init_config(self, enable_history, rerank_top_k, history_summary, query_expansion, generation_limits, retrieval_strategy):
        self.history: List[List[str]] = []
        self.enable_history = enable_history
        self.rerank_top_k = rerank_top_k
        self.history_summary = history_summary or ""
        self.query_expansion = query_expansion
        self.generation_limits = _merge_generation_limits(generation_limits)
        self.retrieval_strategy = _merge_retrieval_strategy(retrieval_strategy)

    def __str__(self):
        """字符串表示

        Returns:
            str: 字符串表示
        """
        if self.use_ollama:
            return f"Hierarchical retriever: {self.hierarchical_retriever}, Generate model: Ollama({self.ollama_model})"
        return f"Hierarchical retriever: {self.hierarchical_retriever}, Generate model: {self.gen_model}"


    def _init_gen_model(
            self,
            gen_model_type: str,
            gen_model_name_or_path: str,
            peft_name: Optional[str] = None,
            int8: bool = False,
            int4: bool = False,
    ):
        """初始化生成模型

        Args:
            gen_model_type: 模型类型
            gen_model_name_or_path: 模型名称或路径
            peft_name: PEFT 模型名称
            int8: 是否使用 int8 量化
            int4: 是否使用 int4 量化

        Returns:
            模型和分词器
        """
        return init_generation_model(
            device=self.device,
            gen_model_type=gen_model_type,
            gen_model_name_or_path=gen_model_name_or_path,
            peft_name=peft_name,
            int8=int8,
            int4=int4,
        )


    def _get_chat_input(self, history=None, history_summary=None):
        """获取聊天输入

        Args:
            history: 历史记录
            history_summary: 历史摘要

        Returns:
            聊天输入
        """
        messages = []
        if history_summary:
            messages.append({
                'role': 'system',
                'content': f"以下是当前会话的压缩记忆，请在回答时遵循：\n{history_summary}"
            })
        for conv in (history or []):
            if conv and conv[0]:
                messages.append({'role': 'user', 'content': conv[0]})
            if conv and conv[1]:
                messages.append({'role': 'assistant', 'content': conv[1]})

        # 如果使用 Ollama，直接返回消息
        if self.use_ollama:
            return messages

        input_ids = self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt'
        )
        return input_ids.to(self.gen_model.device)


    def _expand_query(
            self,
            query: str,
            history: Optional[List[List[str]]] = None,
            history_summary: Optional[str] = None,
    ) -> List[str]:
        """扩展查询

        Args:
            query: 查询文本

        Returns:
            List[str]: 扩展后的查询列表
        """
        return self.query_expander.expand(
            query=query,
            enabled=self.query_expansion,
            history=self.history if history is None else history,
            history_summary=self.history_summary if history_summary is None else history_summary,
            use_history=self.enable_history,
        )


    def _generate_summary(
            self,
            prompt: str,
            max_new_tokens: int = 256,
            temperature: float = 0.2,
    ) -> str:
        """生成摘要

        Args:
            prompt: 提示文本
            max_new_tokens: 最大新令牌数
            temperature: 温度

        Returns:
            str: 生成的摘要
        """
        return self.history_manager.generate_summary(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )


    def compress_history_if_needed(self, history, history_summary):
        """如果需要，压缩历史记录

        Args:
            history: 历史记录
            history_summary: 历史摘要

        Returns:
            压缩后的历史记录和摘要
        """
        return self.history_manager.compress_if_needed(
            history=history,
            history_summary=history_summary,
            enable_history=self.enable_history,
        )


    @torch.inference_mode()
    def stream_generate_answer(
            self,
            max_new_tokens=1024,
            temperature=0.7,
            repetition_penalty=1.0,
            context_len=2048,
            history=None,
            history_summary=None,
    ):
        """流式生成回答

        Args:
            max_new_tokens: 最大新令牌数
            temperature: 温度
            repetition_penalty: 重复惩罚
            context_len: 上下文长度
            history: 历史记录
            history_summary: 历史摘要

        Yields:
            生成的回答片段
        """
        yield from self._generation_service.stream_generate_answer(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            context_len=context_len,
            history=history,
            history_summary=history_summary,
        )


    def add_corpus(self, files: Union[str, List[str]], persist_files: bool = True):
        """添加语料

        Args:
            files: 语料文件
            persist_files: 是否持久化文件
        """
        if not hasattr(self, "_ingestion_service"):
            self._ingestion_service = RagIngestionService(self)
        self._ingestion_service.add_corpus(files, persist_files=persist_files)


    def resolve_embedding_dir(self, corpus_files=None):
        """解析嵌入目录

        Args:
            corpus_files: 语料文件

        Returns:
            嵌入目录路径
        """
        target_files = self.corpus_files if corpus_files is None else corpus_files
        return resolve_embedding_dir_from_files(self.save_corpus_emb_dir, target_files or [])


    def predict_stream(
            self,
            query: str,
            max_length: int = 768,
            context_len: int = 8192,
            temperature: float = 0.2,
            history=None,
            history_summary=None,
            selected_doc_paths=None,
            trace_ctx=None,
            intent=None,
    ):
        """流式预测

        Args:
            query: 查询文本
            max_length: 最大长度
            context_len: 上下文长度
            temperature: 温度
            history: 历史记录
            history_summary: 历史摘要
            selected_doc_paths: 选定的文档路径

        Yields:
            预测结果
        """
        yield from self._generation_service.predict_stream(
            query=query,
            max_length=max_length,
            context_len=context_len,
            temperature=temperature,
            history=history,
            history_summary=history_summary,
            selected_doc_paths=selected_doc_paths,
            trace_ctx=trace_ctx,
            intent=intent,
        )


    def save_corpus_emb(self):
        """保存语料嵌入

        Returns:
            保存结果
        """
        if not hasattr(self, "_persistence_service"):
            self._persistence_service = RagPersistenceService(self)
        return self._persistence_service.save_corpus_emb()


    def load_corpus_emb(self, emb_dir: str):
        """加载语料嵌入

        Args:
            emb_dir: 嵌入目录

        Returns:
            加载结果
        """
        if not hasattr(self, "_persistence_service"):
            self._persistence_service = RagPersistenceService(self)
        return self._persistence_service.load_corpus_emb(emb_dir)
