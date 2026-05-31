from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


@dataclass
class RetrievalResult:
    """检索结果

    Attributes:
        id: 结果唯一标识
        content: 结果内容
        score: 相似度分数
        level: 检索层级
        metadata: 元数据字典
    """
    id: str
    content: str
    score: float
    level: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理

        确保分数为正数，将负分数转换为正数
        """
        if self.score < 0:
            self.score = 1.0 / (1.0 - self.score + 1e-8)


@dataclass
class FusedResult:
    """融合结果

    Attributes:
        result: 检索结果
        fused_score: 融合后的分数
        source_scores: 各来源分数
    """
    result: RetrievalResult
    fused_score: float
    source_scores: Dict[str, float]


class RetrievalLevel(Enum):
    """检索层级枚举

    Members:
        PAPER: 论文层级
        SECTION: 章节层级
        ELEMENT: 元素层级
    """
    PAPER = "paper"
    SECTION = "section"
    ELEMENT = "element"
