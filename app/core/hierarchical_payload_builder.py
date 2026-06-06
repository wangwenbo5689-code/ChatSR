import os
import re
from collections import defaultdict
from typing import Any, Dict, List

from app.core.section_filters import (
    is_noise_section_metadata,
    normalize_section_title as normalize_filter_section_title,
)


ABSTRACT_TITLES = {"abstract", "\u6458\u8981"}


def _split_section_hierarchy(meta: Dict[str, Any]) -> List[str]:
    """分割章节层次结构

    Args:
        meta: 元数据字典

    Returns:
        List[str]: 层次结构列表
    """
    hierarchy = meta.get("section_hierarchy")
    if isinstance(hierarchy, str):
        parts = [part.strip() for part in hierarchy.split(">")]
        return [part for part in parts if part]
    if isinstance(hierarchy, list):
        parts = [str(part).strip() for part in hierarchy]
        return [part for part in parts if part]
    return []


def _normalize_section_title(meta: Dict[str, Any]) -> str:
    """标准化章节标题

    Args:
        meta: 元数据字典

    Returns:
        str: 标准化的章节标题
    """
    section_title = str(meta.get("section_title") or "").strip()
    return section_title or "Document"


def _section_key_from_meta(meta: Dict[str, Any]) -> str:
    """从元数据生成章节键

    Args:
        meta: 元数据字典

    Returns:
        str: 章节键
    """
    hierarchy_parts = _split_section_hierarchy(meta)
    if hierarchy_parts:
        return " > ".join(hierarchy_parts)
    return _normalize_section_title(meta)


def _leaf_title_from_section_key(section_key: str) -> str:
    """从章节键获取叶节点标题

    Args:
        section_key: 章节键

    Returns:
        str: 叶节点标题
    """
    if not section_key:
        return "Document"
    return section_key.split(" > ")[-1].strip() or "Document"


def _build_element_retrieval_text(text: str, section_title: str) -> str:
    body = str(text or "").strip()
    title = str(section_title or "").strip()
    if not title or title.lower() == "document":
        return body
    if body.lower().startswith(title.lower()):
        return body
    return f"Section: {title}\n\n{body}".strip()


def _is_abstract_title(value: Any) -> bool:
    normalized = normalize_filter_section_title(value)
    if not normalized:
        return False
    return any(
        normalized == title
        or normalized.startswith(f"{title} ")
        or normalized.startswith(f"{title}:")
        for title in ABSTRACT_TITLES
    )


def _is_abstract_metadata(meta: Dict[str, Any]) -> bool:
    if _is_abstract_title(meta.get("section_title")):
        return True
    return any(_is_abstract_title(part) for part in _split_section_hierarchy(meta))


def _text_starts_with_abstract_label(text: str) -> bool:
    prefix = str(text or "").strip()[:80].lower()
    return (
        prefix == "abstract"
        or prefix.startswith("abstract\n")
        or prefix.startswith("abstract:")
        or prefix.startswith("abstract.")
        or prefix.startswith("abstract -")
        or prefix.startswith("abstract \u2013")
        or prefix.startswith("abstract \u2014")
        or prefix.startswith("\u6458\u8981")
    )


def _is_abstract_item(item: Dict[str, Any]) -> bool:
    meta = item.get("metadata") or {}
    text = str(item.get("text") or "")
    return _is_abstract_metadata(meta) or _text_starts_with_abstract_label(text)


def _strip_abstract_heading(text: str) -> str:
    body = str(text or "").strip()
    stripped = re.sub(r"^(abstract|\u6458\u8981)\s*[:.\-]?\s*", "", body, flags=re.IGNORECASE).strip()
    return stripped or body


def _extract_abstract_text(normalized_items: List[Dict[str, Any]]) -> str:
    pieces = [
        _strip_abstract_heading(item["text"])
        for item in normalized_items
        if _is_abstract_item(item)
    ]
    return "\n\n".join(piece for piece in pieces if piece).strip()


def _coerce_page(value: Any) -> int:
    """强制转换页码为整数

    Args:
        value: 页码值

    Returns:
        int: 页码整数
    """
    try:
        page = int(value)
    except (TypeError, ValueError):
        return 1
    return page if page > 0 else 1


def _as_int(value: Any, default: int = 0) -> int:
    """转换值为整数

    Args:
        value: 要转换的值
        default: 默认值

    Returns:
        int: 转换后的整数
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_reference_section(meta: Dict[str, Any]) -> bool:
    """检查是否为参考文献章节

    Args:
        meta: 元数据字典

    Returns:
        bool: 是否为参考文献章节
    """
    return is_noise_section_metadata(meta)


def normalize_chunk_items(
    chunk_items: List[Dict[str, Any]],
    doc_hash: str,
) -> List[Dict[str, Any]]:
    """标准化分块项目

    清理文本、补全元数据、过滤空内容。

    Args:
        chunk_items: 原始分块项目列表
        doc_hash: 文档哈希

    Returns:
        List[Dict[str, Any]]: 标准化后的项目列表
    """
    normalized = []
    for idx, item in enumerate(chunk_items):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        meta = dict(item.get("metadata") or {})
        meta.setdefault("section_title", meta.get("section_title", "Document"))
        meta.setdefault("section_hierarchy", meta.get("section_hierarchy", meta.get("section_title", "Document")))
        meta.setdefault("page", 1)
        meta.setdefault("doc_hash", doc_hash)
        meta.setdefault("text_content", text)
        meta.setdefault("chunk_index", idx)
        normalized.append({"text": text, "metadata": meta})
    return normalized


def _collect_sections(
    normalized_items: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], List[str]]:
    """按原始章节归属收集所有有内容的章节。

    保留所有有内容的章节（包括父章节和叶子章节），内容归属不变：
    - 属于父章节的内容保留在父章节
    - 属于子章节的内容保留在子章节
    不做强制下沉或映射到最近叶子章节。

    Args:
        normalized_items: 标准化的分块项目列表

    Returns:
        tuple: (重新映射的项目, 章节组, 章节顺序)
    """
    section_order: List[str] = []
    seen_sections = set()
    item_section_keys: List[str] = []
    for item in normalized_items:
        section_key = _section_key_from_meta(item["metadata"])
        item_section_keys.append(section_key)
        if section_key not in seen_sections:
            seen_sections.add(section_key)
            section_order.append(section_key)

    remapped_items: List[Dict[str, Any]] = []
    section_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    section_order_after_remap: List[str] = []
    seen_after_remap = set()

    for index, item in enumerate(normalized_items):
        section_key = item_section_keys[index]
        remapped_item = {
            "text": item["text"],
            "metadata": dict(item["metadata"]),
        }
        remapped_item["metadata"]["section_hierarchy"] = section_key
        remapped_item["metadata"]["section_title"] = _leaf_title_from_section_key(section_key)
        remapped_items.append(remapped_item)
        section_groups[section_key].append(remapped_item)
        if section_key not in seen_after_remap:
            seen_after_remap.add(section_key)
            section_order_after_remap.append(section_key)

    effective_section_order = [
        section_key for section_key in section_order_after_remap
        if section_groups.get(section_key)
    ]
    return remapped_items, section_groups, effective_section_order


def _build_paper_document(file_name: str, abstract_text: str, section_names: List[str]) -> str:
    lines = [f"Title: {file_name}"]
    if abstract_text:
        lines.extend(["", "Abstract:", abstract_text])
    if section_names:
        lines.extend(["", "Sections:"])
        lines.extend(f"- {_leaf_title_from_section_key(name)}" for name in section_names)
    return "\n".join(lines).strip() or file_name


def _build_paper_payload(
    doc_hash: str,
    doc_file: str,
    paper_id: str,
    abstract_text: str,
    page_count: int,
    section_names: List[str],
    child_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """构建论文 payload

    Args:
        doc_hash: 文档哈希
        doc_file: 文档文件路径
        paper_id: 论文 ID
        abstract_text: 专门抽取的摘要文本
        page_count: 页数
        section_names: 章节名称列表
        child_items: 子项目列表

    Returns:
        Dict[str, Any]: 论文 payload
    """
    file_name = os.path.basename(doc_file)
    paper_doc = _build_paper_document(
        file_name=file_name,
        abstract_text=abstract_text,
        section_names=section_names,
    )
    metadata = {
        "level": "paper",
        "paper_id": paper_id,
        "title": file_name,
        "abstract": abstract_text,
        "pdf_path": file_name if os.path.basename(os.path.dirname(doc_file)).startswith("chatsr_upload_") else doc_file,
        "corpus_name": "default",
        "doc_hash": doc_hash,
        "section_count": len(section_names),
        "chunk_count": len(child_items),
        "section_titles": " | ".join(_leaf_title_from_section_key(name) for name in section_names),
    }
    if page_count > 0:
        metadata["page_count"] = page_count
    return {
        "ids": [paper_id],
        "documents": [paper_doc],
        "metadatas": [metadata],
    }


def _build_section_payloads(
    doc_hash: str,
    paper_id: str,
    paper_title: str,
    section_groups: Dict[str, List[Dict[str, Any]]],
    section_order: List[str],
) -> Dict[str, Any]:
    """构建章节 payloads

    Args:
        doc_hash: 文档哈希
        paper_id: 论文 ID
        paper_title: 论文标题
        section_groups: 章节组
        section_order: 章节顺序

    Returns:
        Dict[str, Any]: 章节 payloads
    """
    section_ids = [f"{paper_id}_sec_{idx}" for idx, _ in enumerate(section_order)]
    section_id_map = {section_name: section_ids[idx] for idx, section_name in enumerate(section_order)}
    section_docs = []
    section_metas = []

    for idx, section_name in enumerate(section_order):
        section_items = section_groups.get(section_name, [])
        section_doc = "\n\n".join(item["text"] for item in section_items).strip()
        section_docs.append(section_doc)
        pages = [_coerce_page(item["metadata"].get("page")) for item in section_items]
        section_title = _leaf_title_from_section_key(section_name)
        is_noise_section = is_noise_section_metadata({
            "section_title": section_title,
            "section_hierarchy": section_name,
        })
        section_meta = {
            "level": "section",
            "section_id": section_ids[idx],
            "parent_id": paper_id,
            "paper_id": paper_id,
            "title": paper_title,
            "section_title": section_title,
            "section_hierarchy": section_name,
            "section_number": str(idx + 1),
            "start_page": min(pages) if pages else 1,
            "end_page": max(pages) if pages else 1,
            "chunk_count": len(section_items),
            "doc_hash": doc_hash,
            "is_noise_section": is_noise_section,
        }
        section_metas.append(section_meta)

    return {
        "names": section_order,
        "ids": section_ids,
        "id_map": section_id_map,
        "documents": section_docs,
        "metadatas": section_metas,
    }


def _build_element_payloads(
    doc_hash: str,
    paper_id: str,
    paper_title: str,
    child_items: List[Dict[str, Any]],
    section_names: List[str],
) -> Dict[str, Any]:
    """构建元素 payloads

    Args:
        doc_hash: 文档哈希
        paper_id: 论文 ID
        paper_title: 论文标题
        child_items: 子项目列表
        section_names: 章节名称列表

    Returns:
        Dict[str, Any]: 元素 payloads
    """
    section_index_map = {name: idx for idx, name in enumerate(section_names)}
    per_section_position: Dict[str, int] = defaultdict(int)
    element_ids: List[str] = []
    element_docs: List[str] = []
    element_metas: List[Dict[str, Any]] = []

    for global_idx, item in enumerate(child_items):
        text = item["text"]
        meta = item["metadata"]
        section_key = _section_key_from_meta(meta)
        section_title = _leaf_title_from_section_key(section_key)
        is_noise_section = is_noise_section_metadata({
            "section_title": section_title,
            "section_hierarchy": section_key,
        })
        section_idx = section_index_map.get(section_key, -1)
        section_id = f"{paper_id}_sec_{section_idx}" if section_idx >= 0 else ""
        per_section_position[section_key] += 1
        position_in_section = per_section_position[section_key]
        element_id = f"{section_id}_elem_{position_in_section - 1}" if section_id else f"{paper_id}_elem_{global_idx}"
        element_meta = {
            "level": "element",
            "element_id": element_id,
            "parent_id": section_id,
            "paper_id": paper_id,
            "paper_title": paper_title,
            "section_id": section_id,
            "section_title": section_title,
            "section_hierarchy": section_key,
            "element_type": str(meta.get("source_type") or "text"),
            "position": position_in_section,
            "page": _coerce_page(meta.get("page", 1)),
            "chunk_index": _as_int(meta.get("chunk_index"), global_idx),
            "doc_hash": doc_hash,
            "text_content": text,
            "is_noise_section": is_noise_section,
        }
        element_ids.append(element_id)
        element_docs.append(_build_element_retrieval_text(text, section_title))
        element_metas.append(element_meta)

    return {
        "ids": element_ids,
        "documents": element_docs,
        "metadatas": element_metas,
    }


def build_hierarchical_document_payloads(
    doc_hash: str,
    doc_file: str,
    chunk_items: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """构建分层文档 payloads

    Args:
        doc_hash: 文档哈希
        doc_file: 文档文件路径
        chunk_items: 分块项目列表

    Returns:
        Dict[str, Dict[str, Any]]: 分层文档 payloads
    """
    normalized_items = normalize_chunk_items(chunk_items=chunk_items, doc_hash=doc_hash)
    paper_id = f"{doc_hash}_paper"
    paper_title = os.path.basename(doc_file)
    abstract_text = _extract_abstract_text(normalized_items)
    body_items = [
        item for item in normalized_items
        if not _is_reference_section(item["metadata"]) and not _is_abstract_item(item)
    ]
    if not body_items and not abstract_text:
        body_items = normalized_items
    child_items, section_groups, section_order = _collect_sections(body_items)
    page_count = max((_coerce_page(item["metadata"].get("page")) for item in normalized_items), default=0)
    section_payload = _build_section_payloads(
        doc_hash=doc_hash,
        paper_id=paper_id,
        paper_title=paper_title,
        section_groups=section_groups,
        section_order=section_order,
    )
    element_payload = _build_element_payloads(
        doc_hash=doc_hash,
        paper_id=paper_id,
        paper_title=paper_title,
        child_items=child_items,
        section_names=section_payload["names"],
    )

    return {
        "paper": _build_paper_payload(
            doc_hash=doc_hash,
            doc_file=doc_file,
            paper_id=paper_id,
            abstract_text=abstract_text,
            page_count=page_count,
            section_names=section_payload["names"],
            child_items=child_items,
        ),
        "section": {
            "ids": section_payload["ids"],
            "documents": section_payload["documents"],
            "metadatas": section_payload["metadatas"],
        },
        "element": {
            "ids": element_payload["ids"],
            "documents": element_payload["documents"],
            "metadatas": element_payload["metadatas"],
        },
    }
