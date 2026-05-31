# -*- coding: utf-8 -*-
"""Docling Document 专用学术分块器（HybridChunker + 学术增强）。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from app.core.rag_defaults import EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_DIRNAME


def load_hybrid_chunker():
    """加载 Docling HybridChunker 类"""
    try:
        from docling.chunking import HybridChunker
    except Exception as exc:
        raise ImportError("Docling HybridChunker 不可用") from exc
    return HybridChunker


@dataclass
class DoclingEnhancedMetadata:
    """Docling 增强元数据类"""
    chunk_id: str  # 分块唯一标识
    token_count: int  # 令牌数量
    level: str = "element"  # 层级
    section_title: str = ""  # 章节标题
    page: Optional[int] = None  # 页码
    source_type: str = "text"  # 源类型
    section_hierarchy: List[str] = field(default_factory=list)  # 章节层次结构

    def to_chunk_metadata(self) -> Dict[str, Any]:
        """将元数据转换为分块元数据字典"""
        return {
            "chunk_id": self.chunk_id,
            "level": self.level,
            "section_title": self.section_title,
            "page": self.page,
            "source_type": self.source_type,
            "content_length": self.token_count,
            "section_hierarchy": " > ".join(self.section_hierarchy),
        }


class DoclingAcademicChunker:
    """Docling 学术分块器"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化学术分块器

        Args:
            config: 配置字典，包含分块参数
        """
        try:
            hybrid_chunker_cls = load_hybrid_chunker()
        except ImportError:
            raise ImportError("Docling HybridChunker 不可用")

        # 默认配置
        self.config = config or {
            "max_tokens": 256,
            "overlap_tokens": 50,
            "merge_peers": True,
            "inject_parent_headings": True,
            "tokenizer_model": EMBEDDING_MODEL_NAME,
            "tokenizer_local_dir": EMBEDDING_MODEL_NAME,
        }

        tokenizer = self._load_tokenizer_local_first(
            model_name=self.config.get("tokenizer_model", EMBEDDING_MODEL_NAME),
            local_dir=self.config.get("tokenizer_local_dir", EMBEDDING_MODEL_NAME),
        )
        self.tokenizer = tokenizer

        # 初始化混合分块器
        chunker_kwargs = {
            "max_tokens": int(self.config.get("max_tokens", 256)),
            "overlap_tokens": int(self.config.get("overlap_tokens", 50)),
            "merge_peers": bool(self.config.get("merge_peers", True)),
        }
        if tokenizer is not None:
            chunker_kwargs["tokenizer"] = tokenizer

        self.hybrid_chunker = hybrid_chunker_cls(**chunker_kwargs)

    @staticmethod
    def _load_tokenizer_local_first(model_name: str, local_dir: str):
        """优先加载本地 tokenizer；本地不存在则尝试下载并保存。"""
        from transformers import AutoTokenizer

        try:
            if local_dir and os.path.isdir(local_dir):
                logger.info(f"从本地缓存加载分词器: {local_dir}")
                return AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True, local_files_only=True)
        except Exception as e:
            logger.warning(f"加载本地分词器失败: {e}")

        try:
            logger.info(f"下载分词器: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)
                tokenizer.save_pretrained(local_dir)
                logger.info(f"将分词器保存到本地缓存: {local_dir}")
            return tokenizer
        except Exception as e:
            logger.warning(f"下载分词器失败: {e}")
            return None

    def chunk_document(self, doc: Any) -> List[Tuple[str, Dict[str, Any]]]:
        """对文档进行分块

        Args:
            doc: Docling 文档对象

        Returns:
            分块列表，每个元素为 (文本, 元数据) 元组
        """
        chunks: List[Tuple[str, Dict[str, Any]]] = []
        for i, chunk in enumerate(self.hybrid_chunker.chunk(doc)):
            text = (getattr(chunk, "text", "") or "").strip()
            if not text:
                continue

            # 提取章节层次结构
            section_hierarchy = self._extract_section_hierarchy(chunk)
            # 获取章节标题
            section_title = section_hierarchy[-1] if section_hierarchy else ""
            # 提取页码
            page = self._extract_page(chunk)
            # 计算令牌数
            token_count = self._count_tokens(text)

            # 创建增强元数据
            metadata = DoclingEnhancedMetadata(
                chunk_id=f"chunk_{i}",
                token_count=token_count,
                level="element",
                section_title=section_title,
                page=page,
                source_type=self._detect_source_type(chunk, text),
                section_hierarchy=section_hierarchy,
            )

            # 注入父级标题
            if self.config.get("inject_parent_headings", True) and len(section_hierarchy) > 1:
                parent = " > ".join(section_hierarchy[:-1])
                text = f"[{parent}]\n{text}"

            chunks.append((text, metadata.to_chunk_metadata()))

        return chunks

    @staticmethod
    def _chunk_meta(chunk: Any) -> Any:
        """获取分块的元数据"""
        return getattr(chunk, "meta", None) or getattr(chunk, "metadata", None)

    @staticmethod
    def _normalize_heading_values(values: Any) -> List[str]:
        """标准化标题值"""
        if values is None:
            return []
        if isinstance(values, str):
            text = values.strip()
            return [text] if text else []
        normalized: List[str] = []
        if isinstance(values, (list, tuple)):
            for value in values:
                text = str(value or "").strip()
                if text:
                    normalized.append(text)
        return normalized

    @staticmethod
    def _extract_page_from_prov_entries(prov_entries: Any) -> Optional[int]:
        """从提供条目提取页码"""
        if not isinstance(prov_entries, (list, tuple)):
            return None
        for prov in prov_entries:
            if isinstance(prov, dict):
                page_no = prov.get("page_no")
            else:
                page_no = getattr(prov, "page_no", None)
            if isinstance(page_no, int):
                return page_no
        return None

    def _extract_section_hierarchy(self, chunk: Any) -> List[str]:
        """提取章节层次结构"""
        md = self._chunk_meta(chunk)
        if md is not None:
            for attr in ("headings", "parent_headings", "section_path"):
                values = self._normalize_heading_values(getattr(md, attr, None))
                if values:
                    return values
        values = self._normalize_heading_values(getattr(chunk, "heading_path", None))
        if values:
            return values
        return []

    def _extract_page(self, chunk: Any) -> Optional[int]:
        """提取页码"""
        # 尝试从分块直接获取页码
        for attr in ("page_no", "page"):
            page = getattr(chunk, attr, None)
            if isinstance(page, int):
                return page

        # 尝试从元数据获取页码
        md = self._chunk_meta(chunk)
        if md is None:
            return None

        for attr in ("page_no", "page"):
            page = getattr(md, attr, None)
            if isinstance(page, int):
                return page

        # 尝试从提供条目获取页码
        page = self._extract_page_from_prov_entries(getattr(md, "prov", None))
        if page is not None:
            return page

        # 尝试从文档项目获取页码
        doc_items = getattr(md, "doc_items", None)
        if isinstance(doc_items, (list, tuple)):
            for item in doc_items:
                prov_entries = item.get("prov") if isinstance(item, dict) else getattr(item, "prov", None)
                page = self._extract_page_from_prov_entries(prov_entries)
                if page is not None:
                    return page
        return None

    def _extract_doc_item_labels(self, chunk: Any) -> List[str]:
        """提取文档项目标签"""
        labels: List[str] = []
        md = self._chunk_meta(chunk)
        if md is None:
            return labels
        doc_items = getattr(md, "doc_items", None)
        if not isinstance(doc_items, (list, tuple)):
            return labels
        for item in doc_items:
            if isinstance(item, dict):
                label = item.get("label")
            else:
                label = getattr(item, "label", None)
            if not label:
                continue
            labels.append(str(label).strip().lower())
        return labels

    def _detect_source_type(self, chunk: Any, text: str) -> str:
        """检测源类型"""
        labels = self._extract_doc_item_labels(chunk)
        if any("table" in label for label in labels):
            return "table"
        if any(
            "figure" in label or "picture" in label or "image" in label
            for label in labels
        ):
            return "figure"
        return "text"

    def _count_tokens(self, text: str) -> int:
        """计算令牌数"""
        if self.tokenizer is not None:
            try:
                encoded = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", None)
                if isinstance(input_ids, list):
                    if input_ids and isinstance(input_ids[0], list):
                        return len(input_ids[0])
                    return len(input_ids)
            except Exception as e:
                logger.warning(f"分词器令牌计数失败，使用启发式计数: {e}")
        # 启发式计数：中文字符 + 英文单词
        zh = len(re.findall(r"[\u4e00-\u9fff]", text))
        en = len(re.findall(r"\b[a-zA-Z]+\b", text))
        return zh + en


def create_docling_academic_chunker(config: Optional[Dict[str, Any]] = None) -> Optional[DoclingAcademicChunker]:
    """创建 Docling 学术分块器

    Args:
        config: 配置字典

    Returns:
        DoclingAcademicChunker 实例或 None
    """
    try:
        return DoclingAcademicChunker(config=config)
    except ImportError:
        logger.warning("Docling HybridChunker 不可用")
        return None
