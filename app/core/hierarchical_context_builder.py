from typing import Any, Dict, List

from app.core.hierarchical_types import FusedResult, RetrievalLevel, RetrievalResult

logger: Any
try:
    from loguru import logger as _logger
    logger = _logger
except ModuleNotFoundError:
    import logging

    logger = logging.getLogger(__name__)


class ContextBuilder:
    """上下文构建器

    用于构建检索结果的上下文，包括论文、章节和元素级别的信息
    """
    def __init__(
        self,
        element_collection=None,
        max_context_length: int = 25000,
    ):
        """初始化上下文构建器

        Args:
            element_collection: 元素集合
            max_context_length: 最大上下文长度
        """
        self.element_collection = element_collection
        self.max_context_length = max_context_length

    def build(
        self,
        fused_results: List[FusedResult],
        query: str = "",
        include_citations: bool = True,
    ) -> Dict[str, Any]:
        """构建上下文

        Args:
            fused_results: 融合结果列表
            query: 查询文本
            include_citations: 是否包含引用

        Returns:
            Dict[str, Any]: 上下文信息
        """
        context_parts: List[str] = []
        citations: List[Dict[str, Any]] = []
        current_length = 0
        for fused in fused_results:
            result = fused.result
            context_block = self._build_context_block(result, include_citations, query)
            block_text = context_block["text"]
            block_length = len(block_text)
            available_length = self.max_context_length - current_length
            if available_length <= 0:
                break
            if block_length > available_length:
                if context_parts:
                    continue
                block_text = self._truncate_block_text(block_text, available_length)
                block_length = len(block_text)
                if not block_text:
                    continue
            context_parts.append(block_text)
            current_length += block_length
            if include_citations:
                citations.append(context_block["citation"])
        final_context = self._assemble_final_context(query=query, context_parts=context_parts, citations=citations)
        return {
            "context": final_context,
            "citations": citations,
            "blocks": context_parts,
            "result_count": len(context_parts),
            "total_length": current_length,
            "structured": self._build_structured_context(fused_results),
        }

    @staticmethod
    def _truncate_block_text(text: str, max_length: int) -> str:
        """截断文本块

        Args:
            text: 文本内容
            max_length: 最大长度

        Returns:
            str: 截断后的文本
        """
        if max_length <= 0:
            return ""
        if len(text) <= max_length:
            return text
        suffix = "\n...[内容截断]"
        if max_length <= len(suffix):
            return text[:max_length]
        return text[: max_length - len(suffix)].rstrip() + suffix

    def _build_context_block(self, result: RetrievalResult, include_citations: bool, query: str) -> Dict[str, Any]:
        """构建上下文块

        Args:
            result: 检索结果
            include_citations: 是否包含引用

        Returns:
            Dict[str, Any]: 上下文块信息
        """
        if result.level == "paper":
            return self._build_paper_block(result, include_citations)
        if result.level == "section":
            return self._build_section_block(result, include_citations)
        return self._build_element_block(result, include_citations, query)

    def _build_paper_block(self, result: RetrievalResult, include_citations: bool) -> Dict[str, Any]:
        """构建论文块

        Args:
            result: 检索结果
            include_citations: 是否包含引用

        Returns:
            Dict[str, Any]: 论文块信息
        """
        meta = result.metadata
        content = result.content
        info_lines = ["【论文信息】", f"标题: {meta.get('title', '未知')}"]
        authors = str(meta.get("authors") or "").strip()
        if authors:
            info_lines.append(f"作者: {authors}")
        publish_parts = [str(meta.get("venue") or "").strip(), str(meta.get("year") or "").strip()]
        publish_text = " ".join([part for part in publish_parts if part])
        if publish_text:
            info_lines.append(f"发表: {publish_text}")
        citation_count = meta.get("citation_count")
        if citation_count not in (None, ""):
            info_lines.append(f"引用数: {citation_count}")
        block_text = "\n".join(info_lines) + f"\n\n【文档内容】\n{content}\n"
        citation = {
            "type": "paper",
            "paper_id": result.id,
            "title": meta.get("title", "未知"),
            "authors": meta.get("authors", "未知"),
            "year": meta.get("year", "未知"),
        } if include_citations else {}
        return {"text": block_text, "citation": citation}

    def _build_section_block(self, result: RetrievalResult, include_citations: bool) -> Dict[str, Any]:
        """构建章节块

        Args:
            result: 检索结果
            include_citations: 是否包含引用

        Returns:
            Dict[str, Any]: 章节块信息
        """
        meta = result.metadata
        content = result.content
        matched_children = meta.get("matched_children") or []
        matched_text = ""
        if matched_children:
            matched_parts = []
            for child in matched_children[:3]:
                if not isinstance(child, dict):
                    continue
                child_content = (child.get("content") or "").strip()
                if not child_content:
                    continue
                child_page = child.get("page", meta.get("start_page", "?"))
                matched_parts.append(f"第{child_page}页: {child_content[:220]}")
            if matched_parts:
                matched_text = "\n【命中片段】\n" + "\n...\n".join(matched_parts)
        block_text = f"""
【章节信息】
论文: {meta.get('title') or meta.get('paper_id', '未知')}
章节: {meta.get('section_title', '未知')}
页码: 第{meta.get('start_page', '?')}页
{matched_text}

【内容】
{content}
"""
        citation = {
            "type": "section",
            "section_id": result.id,
            "paper_id": meta.get("paper_id", "未知"),
            "paper_title": meta.get("title") or meta.get("paper_title") or "未知文献",
            "section_title": meta.get("section_title", "未知"),
            "page": meta.get("start_page", 0),
        } if include_citations else {}
        return {"text": block_text, "citation": citation}

    def _build_element_block(self, result: RetrievalResult, include_citations: bool, query: str) -> Dict[str, Any]:
        """构建元素块

        Args:
            result: 检索结果
            include_citations: 是否包含引用

        Returns:
            Dict[str, Any]: 元素块信息
        """
        meta = result.metadata
        content = result.content
        element_type = meta.get("element_type", "text")
        paper_title = meta.get("title") or meta.get("paper_title") or meta.get("paper_id", "未知")
        element_type_cn = {
            "text": "文本段落",
            "figure": "图片",
            "table": "表格",
            "equation": "公式",
            "citation": "引用",
        }.get(element_type, "内容")
        extended = self._get_extended_context(result, query)
        block_text = f"""
【{element_type_cn}信息】
来源论文: {paper_title}
所属章节: {meta.get('section_title', '未知')}
位置: 第{meta.get('page', '?')}页
类型: {element_type}

【内容】
{content}

【上下文】
{extended}
"""
        citation = {
            "type": "element",
            "element_id": result.id,
            "paper_id": meta.get("paper_id", "未知"),
            "paper_title": meta.get("title") or meta.get("paper_title") or "未知文献",
            "section_title": meta.get("section_title", "未知"),
            "page": meta.get("page", 0),
            "element_type": element_type,
        } if include_citations else {}
        return {"text": block_text, "citation": citation}

    def _get_extended_context(self, result: RetrievalResult, query: str) -> str:
        """获取扩展上下文

        Args:
            result: 检索结果

        Returns:
            str: 扩展上下文
        """
        metadata = result.metadata
        section_id = metadata.get("section_id")
        position = metadata.get("position", 0)
        if not self.element_collection:
            return "无"
        if not section_id:
            prev_id = metadata.get("previous_element", "")
            next_id = metadata.get("next_element", "")
            extended_parts = []
            if prev_id:
                try:
                    prev_result = self.element_collection.get(ids=[prev_id])
                    if prev_result and prev_result.get("documents"):
                        extended_parts.append(f"前文: {prev_result['documents'][0][:200]}")
                except Exception as exc:
                    logger.debug(f"加载前一个元素失败: {exc}")
            if next_id:
                try:
                    next_result = self.element_collection.get(ids=[next_id])
                    if next_result and next_result.get("documents"):
                        extended_parts.append(f"后文: {next_result['documents'][0][:200]}")
                except Exception as exc:
                    logger.debug(f"加载后一个元素失败: {exc}")
            return "\n".join(extended_parts) if extended_parts else "无"
        try:
            results = self.element_collection.query(
                query_texts=[query],
                n_results=5, # 扩展上下文的固定结果数量
                where={
                    "$and": [
                        {"section_id": section_id},
                        {"position": {"$gte": max(0, position - 2)}},
                        {"position": {"$lte": position + 2}},
                    ]
                },
                include=['documents', 'metadatas', 'distances']
            )
            if results and results.get("documents"):
                # 将嵌套列表展平为文档和元数据
                documents_flat = [item for sublist in results.get("documents", []) for item in sublist]
                metadatas_flat = [item for sublist in results.get("metadatas", []) for item in sublist]

                positions = [meta.get("position", 0) for meta in metadatas_flat]
                sorted_indices = sorted(range(len(positions)), key=lambda index: positions[index])
                context_parts = []
                for index in sorted_indices:
                    if index < len(documents_flat):
                        doc = documents_flat[index]
                        if doc and doc != result.content:
                            context_parts.append(doc[:200])
                return "\n...\n".join(context_parts[:2])
        except Exception as exc:
            logger.warning(f"获取扩展上下文失败: {exc}")
        return "无"

    @staticmethod
    def _assemble_final_context(query: str, context_parts: List[str], citations: List[Dict]) -> str:
        """组装最终上下文

        Args:
            query: 查询文本
            context_parts: 上下文部分列表
            citations: 引用列表

        Returns:
            str: 最终上下文
        """
        if not context_parts:
            return "未找到相关内容。"
        citation_list = []
        for index, citation in enumerate(citations, 1):
            if not citation:
                continue
            if citation.get("type") == "paper":
                citation_list.append(f"[{index}] {citation.get('title', '')}, {citation.get('authors', '')}, {citation.get('year', '')}")
            elif citation.get("type") == "section":
                citation_list.append(f"[{index}] {citation.get('section_title', '')} (第{citation.get('page', '?')}页)")
            else:
                citation_list.append(f"[{index}] 第{citation.get('page', '?')}页, {citation.get('section_title', '')}")
        separator = "\n" + "=" * 50 + "\n"
        return f"""
用户问题: {query}

相关文献内容:
{separator}
{separator.join(context_parts)}

参考文献:
{"\n".join(citation_list)}
"""

    def _build_structured_context(self, fused_results: List[FusedResult]) -> List[Dict[str, Any]]:
        """构建结构化上下文

        Args:
            fused_results: 融合结果列表

        Returns:
            List[Dict[str, Any]]: 结构化上下文
        """
        paper_map: Dict[str, Dict[str, Any]] = {}
        paper_order: List[str] = []
        for fused in fused_results:
            result = fused.result
            meta = result.metadata
            paper_id = result.id if result.level == RetrievalLevel.PAPER.value else meta.get("paper_id", "")
            if not paper_id:
                continue
            if paper_id not in paper_map:
                paper_map[paper_id] = {
                    "level": "paper",
                    "id": paper_id,
                    "title": meta.get("title", ""),
                    "authors": meta.get("authors", ""),
                    "year": meta.get("year", ""),
                    "venue": meta.get("venue", ""),
                    "score": 0.0,
                    "sections": [],
                    "elements": [],
                }
                paper_order.append(paper_id)
            paper_entry = paper_map[paper_id]
            paper_entry["score"] = max(paper_entry["score"], fused.fused_score)
            if meta.get("title") and not paper_entry.get("title"):
                paper_entry["title"] = meta.get("title", "")
            if meta.get("authors") and not paper_entry.get("authors"):
                paper_entry["authors"] = meta.get("authors", "")
            if meta.get("year") and not paper_entry.get("year"):
                paper_entry["year"] = meta.get("year", "")
            if meta.get("venue") and not paper_entry.get("venue"):
                paper_entry["venue"] = meta.get("venue", "")

            if result.level == RetrievalLevel.PAPER.value:
                continue
            if result.level == RetrievalLevel.SECTION.value:
                paper_entry["sections"].append({
                    "id": result.id,
                    "title": meta.get("section_title", ""),
                    "score": fused.fused_score,
                })
                continue
            if result.level == RetrievalLevel.ELEMENT.value:
                paper_entry["elements"].append({
                    "id": result.id,
                    "type": meta.get("element_type", ""),
                    "page": meta.get("page", ""),
                    "score": fused.fused_score,
                })
        context = [paper_map[paper_id] for paper_id in paper_order]
        for item in context:
            item["sections"] = sorted(item["sections"], key=lambda section: section.get("score", 0), reverse=True)[:5]
            item["elements"] = sorted(item["elements"], key=lambda element: element.get("score", 0), reverse=True)[:10]
        return context
