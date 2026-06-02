from typing import Any, Dict, List, Optional

from app.core.rag_defaults import DEFAULT_RERANK_MODEL_NAME, DEFAULT_TEXT_ENCODER_MAX_LENGTH
from app.core.hierarchical_types import FusedResult, RetrievalResult

logger: Any
try:
    from loguru import logger as _logger
    logger = _logger
except ModuleNotFoundError:
    import logging

    logger = logging.getLogger(__name__)


class ResultFuser:
    """结果融合器

    用于融合不同检索方法的结果，支持 RRF 融合策略
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化结果融合器

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.similarity_threshold = self.config.get("similarity_threshold", 0.5)
        self.rrf_k = self.config.get("rrf_k", 60)
        self.score_scale_factor = self.config.get("score_scale_factor", 100.0)

    def _rrf_fuse(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        level: str,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> List[RetrievalResult]:
        """使用 RRF 融合稠密和稀疏检索结果

        标准 RRF：按排序位置累积分，支持 dense/sparse 的可调权重

        Args:
            dense_results: 稠密检索结果
            sparse_results: 稀疏检索结果
            level: 检索层级
            dense_weight: 稠密结果权重
            sparse_weight: 稀疏结果权重

        Returns:
            List[RetrievalResult]: 融合后的结果
        """
        score_map: Dict[str, float] = {}
        result_map: Dict[str, RetrievalResult] = {}
        sorted_dense = sorted(dense_results, key=lambda item: item.score, reverse=True)
        for rank, result in enumerate(sorted_dense, start=1):
            doc_id = result.id
            # 引入 score_scale_factor 使分数量级更大 (默认 100x)
            score_map[doc_id] = score_map.get(doc_id, 0.0) + dense_weight * (self.score_scale_factor / (self.rrf_k + rank))
            if doc_id not in result_map:
                result_map[doc_id] = result

        sorted_sparse = sorted(sparse_results, key=lambda item: item.score, reverse=True)
        for rank, result in enumerate(sorted_sparse, start=1):
            doc_id = result.id
            # 引入 score_scale_factor 使分数量级更大 (默认 100x)
            score_map[doc_id] = score_map.get(doc_id, 0.0) + sparse_weight * (self.score_scale_factor / (self.rrf_k + rank))
            if doc_id not in result_map:
                result_map[doc_id] = result

        fused_results = []
        for doc_id, rrf_score in sorted(score_map.items(), key=lambda item: item[1], reverse=True):
            if doc_id in result_map:
                result = result_map[doc_id]
                fused_results.append(
                    RetrievalResult(
                        id=result.id,
                        content=result.content,
                        score=rrf_score,
                        level=level,
                        metadata=result.metadata,
                    )
                )
        return fused_results

    def _filter_low_score(self, results: List[FusedResult], threshold: Optional[float] = None) -> List[FusedResult]:
        """过滤低分数结果

        Args:
            results: 融合结果列表
            threshold: 相对阈值（score/max_score），不传则使用默认配置

        Returns:
            List[FusedResult]: 过滤后的结果
        """
        if not results:
            return results
        max_score = max(item.fused_score for item in results)
        if max_score <= 0:
            return results
        effective_threshold = self.similarity_threshold if threshold is None else float(threshold)
        return [item for item in results if item.fused_score / max_score >= effective_threshold]


class ResultReranker:
    """结果重排器

    使用重排模型对检索结果进行重排
    """
    def __init__(
        self,
        model_name_or_path: str = DEFAULT_RERANK_MODEL_NAME,
        device: str = "cpu",
        config: Optional[Dict[str, Any]] = None,
    ):
        """初始化结果重排器

        Args:
            model_name_or_path: 重排模型名称或路径
            device: 运行设备
            config: 配置字典
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.config = config or {}
        self.score_scale_factor = self.config.get("score_scale_factor", 100.0)
        self.rerank_tokenizer: Any = None
        self.rerank_model: Any = None
        self._load_model()

    def _load_model(self):
        """加载重排模型"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.rerank_tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.rerank_model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
            self.rerank_model.to(self.device)
            self.rerank_model.eval()
            logger.info(f"加载重排模型: {self.model_name_or_path}")
        except Exception as exc:
            logger.warning(f"加载重排模型失败: {exc}")
            self.rerank_tokenizer = None
            self.rerank_model = None

    def rerank(
        self,
        query: str,
        results: List[FusedResult],
        top_k: Optional[int] = None,
        original_score_weight: float = 0.3,
        rerank_score_weight: float = 0.7,
    ) -> List[FusedResult]:
        """重排结果

        Args:
            query: 查询文本
            results: 融合结果列表
            top_k: 保留前 k 个结果
            original_score_weight: 原始分数权重
            rerank_score_weight: 重排分数权重

        Returns:
            List[FusedResult]: 重排后的结果
        """
        if not results:
            return results
        if self.rerank_model is not None and self.rerank_tokenizer is not None:
            return self._rerank_with_model(
                query,
                results,
                top_k,
                original_score_weight=original_score_weight,
                rerank_score_weight=rerank_score_weight,
            )
        return results[:top_k] if top_k else results

    def _rerank_with_model(
        self,
        query: str,
        results: List[FusedResult],
        top_k: Optional[int],
        original_score_weight: float,
        rerank_score_weight: float,
    ) -> List[FusedResult]:
        """使用模型重排结果

        Args:
            query: 查询文本
            results: 融合结果列表
            top_k: 保留前 k 个结果
            original_score_weight: 原始分数权重
            rerank_score_weight: 重排分数权重

        Returns:
            List[FusedResult]: 重排后的结果
        """
        contents = [item.result.content for item in results]
        try:
            import torch

            tokenizer = self.rerank_tokenizer
            model = self.rerank_model
            if tokenizer is None or model is None:
                return results[:top_k] if top_k else results

            pairs = [[query, content] for content in contents]
            with torch.no_grad():
                inputs = tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=DEFAULT_TEXT_ENCODER_MAX_LENGTH,
                )
                inputs_on_device = {key: value.to(model.device) for key, value in inputs.items()}
                scores = model(**inputs_on_device, return_dict=True).logits.view(-1,).float()
            scores = scores.cpu().numpy()
            normalized_original_weight, normalized_rerank_weight = self._normalize_blend_weights(
                original_score_weight=original_score_weight,
                rerank_score_weight=rerank_score_weight,
            )
            rank_fusion_scores = self._fuse_scores_by_rank(
                [float(item.fused_score) for item in results],
                [float(score) for score in scores.tolist()],
                original_weight=normalized_original_weight,
                model_weight=normalized_rerank_weight,
                scale_factor=self.score_scale_factor,
            )
            for index, result in enumerate(results):
                result.fused_score = rank_fusion_scores[index]
            results.sort(key=lambda item: item.fused_score, reverse=True)
            return results[:top_k] if top_k else results
        except Exception as exc:
            logger.warning(f"重排模型失败: {exc}, 保持融合排序")
            return results[:top_k] if top_k else results

    @staticmethod
    def _normalize_blend_weights(
        original_score_weight: float,
        rerank_score_weight: float,
    ) -> tuple[float, float]:
        """归一化权重

        Args:
            original_score_weight: 原始分数权重
            rerank_score_weight: 重排分数权重

        Returns:
            tuple[float, float]: 归一化后的权重
        """
        total_weight = max(original_score_weight + rerank_score_weight, 1e-8)
        return original_score_weight / total_weight, rerank_score_weight / total_weight

    @staticmethod
    def _normalize_scores_min_max(scores: List[float]) -> List[float]:
        """将分数归一化到 [0, 1] 区间。"""
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        span = max_score - min_score
        if span <= 1e-8:
            return [0.5 for _ in scores]
        return [(score - min_score) / span for score in scores]

    @staticmethod
    def _rank_positions(scores: List[float]) -> List[int]:
        """返回每个分数在降序排序下的名次（1-based）。"""
        if not scores:
            return []
        ranked_indices = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))
        ranks = [0] * len(scores)
        for rank, idx in enumerate(ranked_indices):
            ranks[idx] = rank + 1
        return ranks

    @staticmethod
    def _fuse_scores_by_rank(
        original_scores: List[float],
        model_scores: List[float],
        original_weight: float,
        model_weight: float,
        rank_bias: int = 60,
        scale_factor: float = 100.0,
    ) -> List[float]:
        """按位次做融合：w_orig/(rank_orig + bias) + w_model/(rank_model + bias)。"""
        if not original_scores or not model_scores:
            return []
        if len(original_scores) != len(model_scores):
            raise ValueError("original_scores 和 model_scores 长度不一致")
        original_ranks = ResultReranker._rank_positions(original_scores)
        model_ranks = ResultReranker._rank_positions(model_scores)
        return [
            ((original_weight / (rank_orig + rank_bias)) + (model_weight / (rank_model + rank_bias))) * scale_factor
            for rank_orig, rank_model in zip(original_ranks, model_ranks)
        ]
