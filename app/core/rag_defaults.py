from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_MODEL_PATH = PROJECT_ROOT / "models" / "all-MiniLM-L12-v2"
EMBEDDING_MODEL_NAME = str(EMBEDDING_MODEL_PATH)
EMBEDDING_MODEL_DIRNAME = EMBEDDING_MODEL_PATH.name

RERANK_MODEL_PATH = PROJECT_ROOT / "models" / "bge-reranker-base"
DEFAULT_RERANK_MODEL_NAME = str(RERANK_MODEL_PATH)
DEFAULT_GENERATE_MODEL_TYPE = "ollama"
DEFAULT_GENERATE_MODEL_NAME = "qwen2.5:3b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 30
MIN_CHUNK_SIZE = 64
DEFAULT_TEXT_ENCODER_MAX_LENGTH = 512

DEFAULT_MAX_NEW_TOKENS = 1024
MIN_MAX_NEW_TOKENS = 64
MAX_MAX_NEW_TOKENS = 1024
DEFAULT_CONTEXT_LEN = 10240
MIN_CONTEXT_LEN = 1024
MAX_CONTEXT_LEN = 32768
DEFAULT_TEMPERATURE = 0.2
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 1.0

DEFAULT_SUMMARY_MAX_NEW_TOKENS = 256
DEFAULT_SUMMARY_TEMPERATURE = DEFAULT_TEMPERATURE
DEFAULT_QUERY_OPTIMIZATION_MAX_NEW_TOKENS = 120
DEFAULT_QUERY_EXPANSION_MAX_NEW_TOKENS = 200
DEFAULT_QUERY_EXPANSION_TEMPERATURE = 0.3
DEFAULT_INTENT_CLASSIFICATION_MAX_NEW_TOKENS = 20
DEFAULT_INTENT_CLASSIFICATION_TEMPERATURE = 0.0
DEFAULT_QUERY_CANDIDATE_MAX_LENGTH = 120
DEFAULT_INTENT_ERROR_DETAIL_MAX_LENGTH = 200

DEFAULT_HISTORY_SUMMARY_TURNS = 5
DEFAULT_HISTORY_KEEP_TURNS = 2

DEFAULT_ALLOWED_EXTS = (".pdf",)
DEFAULT_MAX_FILES_PER_UPLOAD = 10
DEFAULT_MAX_FILE_SIZE_MB = 20
DEFAULT_CORS_ORIGINS = ("http://localhost:8000", "http://127.0.0.1:8000")
FILE_READ_CHUNK_SIZE_BYTES = 1024 * 1024

DEFAULT_ELEMENT_CANDIDATE_POOL_SIZE = 18
DEFAULT_PARENT_RERANK_TOP_K = 5
DEFAULT_FINAL_CONTEXT_TOP_K = 4
DEFAULT_SUB_QUERY_SCORE_BOOST = 0.15
DEFAULT_CROSS_QUERY_RRF_K = 60.0
DEFAULT_CROSS_QUERY_RRF_SCALE = 100.0
DEFAULT_ELEMENT_BASE_RECALL = DEFAULT_ELEMENT_CANDIDATE_POOL_SIZE
DEFAULT_HYBRID_RETRIEVE_TOP_N = 8
DEFAULT_PAPER_RETRIEVER_TOP_K = 10
DEFAULT_ELEMENT_RETRIEVER_TOP_K = 20

DEFAULT_GENERATION_LIMITS: Dict[str, Any] = {
    "max_length_default": DEFAULT_MAX_NEW_TOKENS,
    "max_length_min": MIN_MAX_NEW_TOKENS,
    "max_length_max": MAX_MAX_NEW_TOKENS,
    "context_len_default": DEFAULT_CONTEXT_LEN,
    "context_len_min": MIN_CONTEXT_LEN,
    "context_len_max": MAX_CONTEXT_LEN,
    "temperature_default": DEFAULT_TEMPERATURE,
    "temperature_min": MIN_TEMPERATURE,
    "temperature_max": MAX_TEMPERATURE,
}

DEFAULT_RETRIEVAL_STRATEGY: Dict[str, Any] = {
    "element_candidate_pool_size": DEFAULT_ELEMENT_CANDIDATE_POOL_SIZE,
    "parent_rerank_top_k": DEFAULT_PARENT_RERANK_TOP_K,
    "final_context_top_k": DEFAULT_FINAL_CONTEXT_TOP_K,
    "sub_query_score_boost": DEFAULT_SUB_QUERY_SCORE_BOOST,
    "cross_query_rrf_k": DEFAULT_CROSS_QUERY_RRF_K,
    "cross_query_rrf_scale": DEFAULT_CROSS_QUERY_RRF_SCALE,
    "dynamic_selection": {
        "fact_qa": {"keep_ratio": 0.7, "min_keep": 1, "max_keep": 6},
        "general_overview": {"keep_ratio": 0.78, "min_keep": 2, "max_keep": 4},
    },
}

DEFAULT_HIERARCHICAL_RETRIEVER_CONFIG: Dict[str, Any] = {
    "element_base_recall": DEFAULT_ELEMENT_BASE_RECALL,
    "parent_rerank_top_k": DEFAULT_PARENT_RERANK_TOP_K,
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
    "cross_query_rrf_k": DEFAULT_CROSS_QUERY_RRF_K,
    "cross_query_rrf_scale": DEFAULT_CROSS_QUERY_RRF_SCALE,
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
