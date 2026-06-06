# ============================================================
# 上下文管理：asynccontextmanager 用于处理 FastAPI 的 lifespan 事件
# 应用启动时加载模型，关闭时执行清理（yield 之后的代码）
# ============================================================
from contextlib import asynccontextmanager

# FastAPI 核心
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
# CORS 中间件：允许跨域请求
from fastapi.middleware.cors import CORSMiddleware
# 静态文件服务：挂载前端页面
from fastapi.staticfiles import StaticFiles

# 日志
from loguru import logger

# ============================================================
# 内部模块导入
# - app_config: CLI 参数解析和运行时配置构建
# - app_context: 应用共享上下文
# - routes: 5 个路由模块（chat, corpus, embedding, session, system）
# - rag_runtime: RAG 运行时引导（模型加载）
# ============================================================
from app.api.app_config import build_runtime_config, parse_runtime_args
from app.api.app_context import build_app_context
from app.api.routes import (
    chat_router,
    corpus_router,
    embedding_router,
    session_router,
    system_router,
)
from app.core.rag_runtime import RagBootstrap
from app.core.rag_defaults import API_GENERATE_MODEL_TYPES, API_KEY_ENV_NAMES, DEFAULT_OLLAMA_HOST, EMBEDDING_MODEL_NAME


def create_app(config=None):
    """
    创建并配置 FastAPI 应用的工厂函数。

    参数:
        config: 可选的 ApiRuntimeConfig 实例。
               如果为 None，则从 CLI 参数解析。
    返回:
        配置好的 FastAPI 应用实例。
    """
    # --------------------------------------------------------
    # 1. 运行时配置
    #    - 如果未传入 config，通过 parse_runtime_args() 解析 CLI 参数
    #    - 再通过 build_runtime_config() 构建不可变的 ApiRuntimeConfig
    # --------------------------------------------------------
    runtime_config = config
    if runtime_config is None:
        args, _ = parse_runtime_args()
        runtime_config = build_runtime_config(args)

    # --------------------------------------------------------
    # 1.5. Preflight 环境检查
    #    在加载模型前验证关键依赖是否可用，失败则提前报错
    # --------------------------------------------------------
    def _run_preflight_checks(context, rt_config):
        """启动前的环境检查，任一失败将阻止启动"""
        import os
        import urllib.request

        model_kwargs = rt_config.model_init_kwargs
        errors = []

        generate_model_type = str(model_kwargs.get("generate_model_type") or "").strip().lower()
        if generate_model_type == "ollama":
            ollama_host = model_kwargs.get("ollama_host") or os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
            try:
                req = urllib.request.Request(f"{ollama_host.rstrip('/')}/api/tags", method="GET")
                urllib.request.urlopen(req, timeout=5)
                logger.info(f"Ollama 服务可达: {ollama_host}")
            except Exception:
                errors.append(f"Ollama 服务不可达 ({ollama_host})，请确认 ollama serve 已启动")

        elif generate_model_type in API_GENERATE_MODEL_TYPES:
            api_key = model_kwargs.get("api_key") or next(
                (os.getenv(env_name) for env_name in API_KEY_ENV_NAMES if os.getenv(env_name)),
                "",
            )
            api_base_url = str(model_kwargs.get("api_base_url") or "").strip()
            if not api_key:
                errors.append(f"LLM API key is not configured. Set one of: {', '.join(API_KEY_ENV_NAMES)}")
            if not api_base_url.startswith(("http://", "https://")):
                errors.append(f"Invalid LLM API base URL: {api_base_url}")

        embedding_model_dir = EMBEDDING_MODEL_NAME
        if not os.path.isdir(embedding_model_dir):
            errors.append(f"Embedding 模型目录不存在: {embedding_model_dir}")

        chroma_dir = rt_config.model_init_kwargs.get("chroma_persist_directory", "./corpus_embs/chroma")
        try:
            os.makedirs(chroma_dir, exist_ok=True)
            test_file = os.path.join(chroma_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except OSError:
            errors.append(f"Chroma 持久化目录不可写: {chroma_dir}")

        data_dir = rt_config.data_dir
        try:
            os.makedirs(data_dir, exist_ok=True)
        except OSError:
            errors.append(f"数据目录不可写: {data_dir}")

        if errors:
            error_summary = "\n  • ".join(errors)
            raise RuntimeError(f"启动预检失败:\n  • {error_summary}")

        logger.info("启动预检通过")

    # --------------------------------------------------------
    # 2. Lifespan 事件管理器
    #    - on_startup → preflight 检查 → 加载 LLM 模型 + 初始化本地语料
    #    - on_shutdown → yield 后的代码（当前为空）
    # --------------------------------------------------------
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        context = app.state.context

        # preflight 检查：在加载模型前验证环境
        _run_preflight_checks(context, runtime_config)

        # 模型加载需要持有 rag_lock，防止并发加载
        with context.rag_lock:
            context.model = RagBootstrap.create_runtime(
                runtime_config.model_init_kwargs
            )
        logger.info(f"model loaded: {context.model}")

        # 引导本地语料：检查磁盘 embedding 状态，必要时加载
        context.embedding_service.bootstrap_local_corpus(
            context.model, context.rag_lock
        )
        yield

        logger.info("应用关闭中，执行清理...")
        context.embedding_service.shutdown()
        if context.model is not None:
            try:
                del context.model
            except Exception:
                pass
            context.model = None
        logger.info("应用清理完成")

    # --------------------------------------------------------
    # 3. 创建 FastAPI 实例
    #    - title: API 文档中的标题
    #    - lifespan: 替代 deprecated 的 on_event("startup"/"shutdown")
    # --------------------------------------------------------
    app = FastAPI(title="ChatSR API", lifespan=lifespan)

    # --------------------------------------------------------
    # 4. 全局异常处理器
    #    统一 JSON 错误响应格式，避免异常泄漏内部细节
    # --------------------------------------------------------
    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "status_code": exc.status_code},
        )

    @app.exception_handler(Exception)
    async def _generic_exception_handler(request: Request, exc: Exception):
        logger.exception(f"未处理的异常: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "服务器内部错误", "status_code": 500},
        )

    # --------------------------------------------------------
    # 5. 注册 CORS 中间件
    #    - allow_origins: 允许的来源列表（从配置读取）
    #    - allow_credentials: 允许携带 cookie / Authorization
    #    - allow_methods/headers: 允许的 HTTP 方法和请求头
    # --------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(runtime_config.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["Content-Type", "Authorization"],
    )

    # --------------------------------------------------------
    # 6. 注入应用级状态
    #    - app.state.context: 共享的 AppContext（含服务实例、锁等）
    # --------------------------------------------------------
    app.state.context = build_app_context(runtime_config)

    # --------------------------------------------------------
    # 7. 注册路由（按领域拆分）
    #    - system_router: 首页、健康检查、系统配置
    #    - embedding_router: embedding 状态/保存/加载
    #    - corpus_router: 文件上传与语料管理
    #    - session_router: 会话 CRUD
    #    - chat_router: 流式问答（核心端点）
    # --------------------------------------------------------
    app.include_router(system_router)
    app.include_router(embedding_router)
    app.include_router(corpus_router)
    app.include_router(session_router)
    app.include_router(chat_router)

    # --------------------------------------------------------
    # 8. 挂载静态文件目录
    #    - /static 路径映射到配置的 static_dir
    #    - 用于服务前端 index.html 及相关资源
    # --------------------------------------------------------
    app.mount(
        "/static",
        StaticFiles(directory=str(runtime_config.static_dir)),
        name="static",
    )

    return app


# ============================================================
# 模块级实例：当作为入口直接运行（如 uvicorn app.api.fastapi_app:app）
# 时，创建默认应用。
# ============================================================
app = create_app()
