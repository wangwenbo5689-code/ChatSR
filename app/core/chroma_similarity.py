# -*- coding: utf-8 -*-

from functools import lru_cache
from typing import List, Any, Tuple

import torch

from app.core.rag_defaults import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=8)
def _load_embedding_components(model_name_or_path: str, device: str) -> Tuple[Any, Any]:
    """加载嵌入模型组件

    Args:
        model_name_or_path: 模型名称或路径
        device: 运行设备

    Returns:
        Tuple[Any, Any]: 分词器和模型
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
    model.to(device)
    model.eval()
    return tokenizer, model


class ProjectEmbeddingFunction:
    """Chroma EmbeddingFunction: 使用本地 all-MiniLM-L12-v2 模型生成向量"""

    def __init__(
        self,
        model_name_or_path: str = EMBEDDING_MODEL_NAME,
        device: str = "cpu",
        max_length: int = 512,
    ):
        """初始化嵌入函数

        Args:
            model_name_or_path: 模型名称或路径
            device: 运行设备
            max_length: 最大序列长度
        """
        self.model_name_or_path = model_name_or_path
        self.device = str(device)
        self.max_length = max_length

        self.tokenizer, self.model = _load_embedding_components(self.model_name_or_path, self.device)

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """均值池化

        Args:
            last_hidden_state: 最后一层隐藏状态
            attention_mask: 注意力掩码

        Returns:
            torch.Tensor: 池化后的向量
        """
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = torch.sum(masked, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / denom

    def name(self) -> str:
        """获取嵌入函数名称

        Returns:
            str: 嵌入函数名称
        """
        return "project_embedding_function"

    @torch.inference_mode()
    def __call__(self, input: List[str]) -> List[List[float]]:
        """生成嵌入向量

        Args:
            input: 输入文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not input:
            return []

        batch = self.tokenizer(
            input,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        outputs = self.model(**batch)
        pooled = self._mean_pool(outputs.last_hidden_state, batch["attention_mask"])
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized.detach().cpu().tolist()
