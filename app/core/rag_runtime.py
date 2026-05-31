from dataclasses import dataclass
from typing import Any, Dict

from app.core.rag import Rag


@dataclass
class RagRuntime:
    """RAG 运行态容器，承载已初始化的 Rag 实例。"""

    rag: Rag

    def __getattr__(self, item: str):
        # 透传到底层 Rag，保持现有调用兼容。
        return getattr(self.rag, item)

    def __str__(self) -> str:
        return str(self.rag)


class RagBootstrap:
    """负责构建 RagRuntime，隔离初始化流程。"""

    @staticmethod
    def create_runtime(model_init_kwargs: Dict[str, Any]) -> RagRuntime:
        rag = Rag(**model_init_kwargs)
        return RagRuntime(rag=rag)
