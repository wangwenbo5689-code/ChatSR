from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_MODEL_PATH = PROJECT_ROOT / "models" / "all-MiniLM-L12-v2"
EMBEDDING_MODEL_NAME = str(EMBEDDING_MODEL_PATH)
EMBEDDING_MODEL_DIRNAME = EMBEDDING_MODEL_PATH.name

RERANK_MODEL_PATH = PROJECT_ROOT / "models" / "bge-reranker-base"
DEFAULT_RERANK_MODEL_NAME = str(RERANK_MODEL_PATH)

DEFAULT_GENERATION_LIMITS: Dict[str, Any] = {
    "max_length_default": 768,
    "max_length_min": 64,
    "max_length_max": 1024,
    "context_len_default": 8192,
    "context_len_min": 1024,
    "context_len_max": 32768,
    "temperature_default": 0.2,
    "temperature_min": 0.0,
    "temperature_max": 1.0,
}

DEFAULT_RETRIEVAL_STRATEGY: Dict[str, Any] = {
    "element_candidate_pool_size": 18,
    "parent_rerank_top_k": 5,
    "final_context_top_k": 4,
    "sub_query_score_boost": 0.15,
    "cross_query_rrf_k": 60.0,
    "cross_query_rrf_scale": 100.0,
    "dynamic_selection": {
        "fact_qa": {"keep_ratio": 0.7, "min_keep": 1, "max_keep": 6},
        "general_overview": {"keep_ratio": 0.78, "min_keep": 2, "max_keep": 4},
    },
}

DEFAULT_HIERARCHICAL_RETRIEVER_CONFIG: Dict[str, Any] = {
    "element_base_recall": 18,
    "parent_rerank_top_k": 5,
    "enable_rerank": True,
    "score_scale_factor": 100.0,
    "fact_original_weight": 0.5,
    "fact_rerank_weight": 0.5,
    "overview_original_weight": 0.6,
    "overview_rerank_weight": 0.4,
    "child_rank_percentile_keep": 0.5,
    "child_rerank_threshold": 0.7,
    "parent_prescreen_threshold": 0.55,
    "parent_rerank_threshold": 0.7,
    "parent_rerank_threshold_overview": 0.78,
    "cross_query_rrf_k": 60,
    "cross_query_rrf_scale": 100.0,
    "dynamic_score_threshold": 0.7,
    "dynamic_min_keep": 2,
    "dynamic_max_keep": 6,
    "dynamic_min_keep_fact": 1,
    "dynamic_max_keep_fact": 6,
    "support_bonus_weight": 0.3,
    "child_count_bonus_weight": 0.15,
    "support_bonus_weight_fact": 0.3,
    "child_count_bonus_weight_fact": 0.1,
    "support_bonus_weight_overview": 0.18,
    "child_count_bonus_weight_overview": 0.08,
}


def default_generation_limits() -> Dict[str, Any]:
    return deepcopy(DEFAULT_GENERATION_LIMITS)


def default_retrieval_strategy() -> Dict[str, Any]:
    return deepcopy(DEFAULT_RETRIEVAL_STRATEGY)


def default_hierarchical_retriever_config() -> Dict[str, Any]:
    return deepcopy(DEFAULT_HIERARCHICAL_RETRIEVER_CONFIG)
