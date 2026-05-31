from typing import Any, Dict, List, Optional

from app.core.rag_defaults import (
    DEFAULT_GENERATION_LIMITS,
    DEFAULT_RERANK_MODEL_NAME,
    DEFAULT_RETRIEVAL_STRATEGY,
    EMBEDDING_MODEL_DIRNAME,
    EMBEDDING_MODEL_NAME,
    default_generation_limits,
    default_retrieval_strategy,
)

ACADEMIC_TOKENIZER_MODEL_NAME = EMBEDDING_MODEL_NAME
ACADEMIC_TOKENIZER_LOCAL_DIRNAME = EMBEDDING_MODEL_DIRNAME


def _merge_generation_limits(override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    limits = default_generation_limits()
    if isinstance(override, dict):
        limits.update(override)
    return limits


def _merge_retrieval_strategy(override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = default_retrieval_strategy()
    if not isinstance(override, dict):
        return merged

    merged.update({k: v for k, v in override.items() if k != "dynamic_selection"})
    base_dynamic = default_retrieval_strategy().get("dynamic_selection", {})
    override_dynamic = override.get("dynamic_selection", {})
    if isinstance(override_dynamic, dict):
        for intent_key, intent_cfg in override_dynamic.items():
            if isinstance(intent_cfg, dict):
                merged_cfg = dict(base_dynamic.get(intent_key, {}))
                merged_cfg.update(intent_cfg)
                base_dynamic[intent_key] = merged_cfg
    merged["dynamic_selection"] = base_dynamic
    return merged


def make_chunk_item(
    text: str,
    level: str = 'paragraph',
    section_title: str = '',
    page: Any = None,
    source_type: str = 'text',
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    item_metadata = {
        'level': level,
        'section_title': section_title,
        'page': page,
        'source_type': source_type,
    }
    if metadata:
        item_metadata.update(metadata)
    return {'text': text, 'metadata': item_metadata}


def add_source_numbers(lst: List[dict]) -> List[str]:
    results = []
    for idx, item in enumerate(lst):
        text = item.get("text", "")
        meta = item.get("metadata", {})

        paper_title = meta.get("paper_title") or meta.get("title") or "\u672a\u77e5\u6587\u732e"
        section_title = meta.get("section_title") or "\u672a\u77e5\u7ae0\u8282"
        page = meta.get("page") or meta.get("start_page")

        source_info = f"{paper_title} - {section_title}"
        if page and page != "?":
            source_info += f" (\u7b2c {page} \u9875)"

        results.append(f'[{idx + 1}] (\u6765\u6e90: {source_info})\n"{text}"')
    return results
