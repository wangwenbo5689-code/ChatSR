import re
from typing import Any, Dict


NOISE_SECTION_KEYWORDS = {
    "references",
    "bibliography",
    "reference",
    "acknowledgments",
    "acknowledgements",
    "acknowledgment",
    "acknowledgement",
    "appendix",
    "appendices",
    "biography",
    "biographies",
    "author contributions",
    "authors contributions",
    "author contribution",
    "contribution statement",
    "conflict of interest",
    "conflicts of interest",
    "competing interests",
    "declaration of interests",
    "funding",
    "financial support",
    "ethics statement",
    "ethics approval",
    "consent",
    "data availability",
    "code availability",
    "correspondence",
    "supplementary material",
    "supplementary information",
    "supplemental material",
    "abstract",
    "参考文献",
    "致谢",
    "作者贡献",
    "利益冲突",
    "资金支持",
    "基金资助",
    "伦理声明",
    "数据可用性",
    "代码可用性",
    "补充材料",
    "附录",
    "摘要",
}


def normalize_section_title(title: Any) -> str:
    text = str(title or "").strip().lower()
    text = re.sub(r"^\s*[\divx]+[\.\)\-: ]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .:-_()[]{}【】（）")


def is_noise_section_title(title: Any) -> bool:
    normalized = normalize_section_title(title)
    if not normalized:
        return False
    for keyword in NOISE_SECTION_KEYWORDS:
        keyword = keyword.lower()
        if (
            keyword == normalized
            or normalized.startswith(f"{keyword} ")
            or normalized.endswith(f" {keyword}")
            or f" {keyword} " in normalized
            or keyword in normalized
        ):
            return True
    return False


def is_noise_section_metadata(meta: Dict[str, Any] | None) -> bool:
    if not isinstance(meta, dict):
        return False
    if is_noise_section_title(meta.get("section_title")):
        return True
    hierarchy = meta.get("section_hierarchy")
    if isinstance(hierarchy, str):
        return any(is_noise_section_title(part) for part in hierarchy.split(">"))
    if isinstance(hierarchy, list):
        return any(is_noise_section_title(part) for part in hierarchy)
    return False
