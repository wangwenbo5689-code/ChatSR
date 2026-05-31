# ============================================================
# API 运行时配置模块
#
# 功能：
# - 定义 CLI 参数解析
# - 构建不可变的运行时配置对象
# - 集中管理所有可配置参数
# ============================================================
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from app.core.rag_defaults import (
    EMBEDDING_MODEL_DIRNAME,
    EMBEDDING_MODEL_NAME,
    default_generation_limits,
    default_retrieval_strategy,
)

# 设置 HuggingFace 镜像站点，加速模型下载
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ============================================================
# ApiRuntimeConfig - 运行时配置数据类
#
# 使用 @dataclass(frozen=True) 使其不可变
# 所有字段在创建后无法修改，保证配置一致性
# ============================================================
@dataclass(frozen=True)
class ApiRuntimeConfig:
    """API 运行时配置（不可变数据类）"""

    # -------- 路径配置 --------
    project_root: Path           # 项目根目录
    data_dir: Path             # 数据目录 (data/)
    local_corpus_dir: Path     # 本地语料目录 (data/local_corpus/)
    session_store_path: str     # 会话存储文件路径
    index_html_path: Path       # 前端首页路径
    static_dir: Path           # 静态文件目录

    # -------- 文件上传配置 --------
    allowed_exts: Tuple[str, ...]      # 允许上传的文件扩展名
    max_files_per_upload: int           # 单次上传最大文件数
    max_file_size_mb: int              # 单个文件最大大小 (MB)

    # -------- CORS 配置 --------
    cors_origins: Tuple[str, ...]       # 允许的跨域来源

    # -------- 生成与检索配置 --------
    generation_limits: Dict[str, Any]    # 生成参数限制
    retrieval_strategy: Dict[str, Any]   # 检索策略配置

    # -------- 模型初始化参数 --------
    model_init_kwargs: Dict[str, object]  # 传递给模型的参数


# ============================================================
# parse_runtime_args - CLI 参数解析
#
# 使用 argparse 解析命令行参数
# 支持通过 --前缀传递参数给应用程序
#
# 返回: (namespace, unknown_args) 元组
# ============================================================
def parse_runtime_args():
    """解析 CLI 参数"""
    # 获取项目根目录 (向上两级)
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="ChatSR - 基于 FastAPI 的本地 RAG 问答服务"
    )

    # -------- 模型配置 --------
    parser.add_argument(
        "--gen_model_type",
        type=str,
        default="ollama",
        help="生成模型类型 (默认: ollama)",
    )
    parser.add_argument(
        "--gen_model_name",
        type=str,
        default="qwen2.5:3b",
        help="生成模型名称 (默认: qwen2.5:3b)",
    )
    parser.add_argument(
        "--lora_model",
        type=str,
        default=None,
        help="LoRA 模型路径 (可选)",
    )
    parser.add_argument(
        "--rerank_model_name",
        type=str,
        default="BAAI/bge-reranker-base",
        help="重排模型名称 (默认: BAAI/bge-reranker-base)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="运行设备，如 'cuda' 或 'cpu' (默认: 自动选择)",
    )

    # -------- 语料配置 --------
    parser.add_argument(
        "--corpus_files",
        type=str,
        default="",
        help="预加载的语料文件路径，多个用逗号分隔 (默认: 空)",
    )
    parser.add_argument(
        "--int4",
        action="store_true",
        help="启用 INT4 量化 (减小模型体积，降低显存)",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="启用 INT8 量化 (平衡模型体积和性能)",
    )

    # -------- 分块配置 --------
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="文档分块大小，按 token 计 (默认: 256)",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=50,
        help="相邻分块之间的重叠 token 数 (默认: 50)",
    )

    # -------- 历史记录配置 --------
    parser.add_argument(
        "--history_max_turns",
        type=int,
        default=6,
        help="保留的最大对话轮次 (默认: 6)",
    )
    parser.add_argument(
        "--history_keep_last_turns",
        type=int,
        default=2,
        help="压缩后保留的最后几轮对话 (默认: 2)",
    )

    # -------- Ollama 配置 --------
    parser.add_argument(
        "--ollama_host",
        type=str,
        default="http://127.0.0.1:11434",
        help="Ollama 服务地址 (默认: http://127.0.0.1:11434)",
    )

    # -------- Docling PDF 解析配置 --------
    # OCR 配置：启用/禁用光学字符识别
    parser.add_argument(
        "--docling_use_ocr",
        dest="docling_use_ocr",
        action="store_true",
        help="启用 Docling OCR 识别 (默认: 启用)",
    )
    parser.add_argument(
        "--no_docling_use_ocr",
        dest="docling_use_ocr",
        action="store_false",
        help="禁用 Docling OCR 识别",
    )
    parser.set_defaults(docling_use_ocr=True)

    # 表格结构提取配置
    parser.add_argument(
        "--docling_table_structure",
        dest="docling_table_structure",
        action="store_true",
        help="启用 Docling 表格结构提取 (TableFormer) (默认: 启用)",
    )
    parser.add_argument(
        "--no_docling_table_structure",
        dest="docling_table_structure",
        action="store_false",
        help="禁用 Docling 表格结构提取",
    )
    parser.set_defaults(docling_table_structure=True)

    # -------- 查询扩展配置 --------
    parser.add_argument(
        "--query_expansion",
        dest="query_expansion",
        action="store_true",
        help="启用查询扩展 (子查询生成、意图分类) (默认: 启用)",
    )
    parser.add_argument(
        "--no_query_expansion",
        dest="query_expansion",
        action="store_false",
        help="禁用查询扩展",
    )
    parser.set_defaults(query_expansion=True)

    # -------- Chroma 向量数据库配置 --------
    parser.add_argument(
        "--chroma_persist_directory",
        type=str,
        default=str(project_root / "corpus_embs" / "chroma"),
        help="Chroma 向量数据库持久化目录 (默认: corpus_embs/chroma)",
    )

    return parser.parse_known_args()


# ============================================================
# build_runtime_config - 构建运行时配置
#
# 将解析后的 CLI 参数组装成 ApiRuntimeConfig 对象
# 定义默认的生成/检索参数
#
# 参数:
#     args: argparse.Namespace 对象，包含解析后的 CLI 参数
#
# 返回:
#     ApiRuntimeConfig: 不可变的运行时配置对象
# ============================================================
def build_runtime_config(args) -> ApiRuntimeConfig:
    """构建运行时配置"""
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"

    # 解析预加载语料文件
    corpus_files = [
        item.strip()
        for item in (args.corpus_files or "").split(",")
        if item.strip()
    ]

    # ============================================================
    # 生成参数限制
    #
    # 用于校验用户请求的生成参数是否在合理范围内
    # ============================================================
    generation_limits: Dict[str, Any] = default_generation_limits()

    # ============================================================
    # 检索策略配置
    #
    # 控制分层检索的各个环节参数
    # ============================================================
    retrieval_strategy: Dict[str, Any] = default_retrieval_strategy()

    # ============================================================
    # 构建并返回配置对象
    # ============================================================
    return ApiRuntimeConfig(
        # -------- 路径配置 --------
        project_root=project_root,
        data_dir=data_dir,
        local_corpus_dir=data_dir / "local_corpus",
        session_store_path=str(data_dir / "sessions" / "sessions.db"),
        index_html_path=project_root / "app" / "utils" / "index.html",
        static_dir=project_root / "app" / "utils",

        # -------- 文件上传配置 --------
        allowed_exts=(".pdf",),              # 目前仅支持 PDF
        max_files_per_upload=10,             # 单次最多上传 10 个文件
        max_file_size_mb=20,                 # 单个文件最大 20MB

        # -------- CORS 配置 --------
        cors_origins=("http://localhost:8000", "http://127.0.0.1:8000"),

        # -------- 生成与检索配置 --------
        generation_limits=generation_limits,
        retrieval_strategy=retrieval_strategy,

        # -------- 模型初始化参数 --------
        model_init_kwargs={
            # 模型类型与名称
            "generate_model_type": args.gen_model_type,
            "generate_model_name_or_path": args.gen_model_name,
            "lora_model_name_or_path": args.lora_model,

            # 语料配置
            "corpus_files": corpus_files,
            "save_corpus_emb_dir": str(project_root / "corpus_embs"),

            # 设备配置
            "device": args.device,

            # 量化配置
            "int4": args.int4,
            "int8": args.int8,

            # 分块配置
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,

            # 重排模型
            "rerank_model_name_or_path": args.rerank_model_name,

            # 历史记录
            "enable_history": True,

            # Ollama 配置
            "ollama_host": args.ollama_host,

            # Docling 配置
            "docling_use_ocr": args.docling_use_ocr,
            "docling_table_structure": args.docling_table_structure,

            # 查询扩展
            "query_expansion": args.query_expansion,

            # Chroma 配置
            "chroma_persist_directory": args.chroma_persist_directory,

            # 传递给服务的配置
            "generation_limits": generation_limits,
            "retrieval_strategy": retrieval_strategy,
        },
    )
