# ChatSR

ChatSR 是一个基于 FastAPI 的本地 RAG 问答服务，支持 Ollama 模型调用（默认 32K 上下文），核心能力包括文档导入、分层检索、查询扩展、会话管理和流式回答。

## 目录结构

- `app/api`：HTTP 接口入口，负责应用装配、运行时配置和路由拆分
  - `app_config.py`：CLI 参数解析与运行时配置构建
  - `app_context.py`：应用共享上下文
  - `fastapi_app.py`：FastAPI 应用创建与依赖注入
  - `routes/`：路由拆分
    - `system_routes.py`：首页和健康检查
    - `embedding_routes.py`：embedding 状态、保存、加载
    - `corpus_routes.py`：文件上传与导入
    - `session_routes.py`：会话列表、重命名、删除
    - `chat_routes.py`：流式问答
- `app/core`：RAG 核心能力
  - `rag.py`：统一编排检索、生成和索引操作
  - `rag_common.py`：共享常量、默认配置与工具函数（避免循环导入）
  - `rag_history_manager.py`：对话历史压缩管理
  - `rag_query_expander.py`：查询扩展（子查询生成、意图分类）
  - `rag_prompts.py`：提示词模板
  - `hierarchical_retriever.py`：分层检索器
  - `hierarchical_context_builder.py`：分层上下文构建
  - `hierarchical_ranking.py`：分层排序
  - `hierarchical_payload_builder.py`：分层载荷构建
  - `docling_academic_chunker.py`：学术文档分块
  - `docling_extractor.py`：Docling 文档提取
  - `chroma_similarity.py`：Chroma 相似度计算
- `app/services`：面向业务流程的服务层
  - `chat_service.py`：聊天服务（含后台历史压缩）
  - `rag_generation_service.py`：RAG 生成服务
  - `rag_retrieval_service.py`：RAG 检索服务
  - `corpus_service.py`：语料服务
  - `embedding_lifecycle_service.py`：Embedding 生命周期服务
  - `rag_document_service.py`：RAG 文档服务
  - `rag_index_service.py`：RAG 索引服务
  - `rag_ingestion_service.py`：RAG 摄取服务
  - `rag_persistence_service.py`：RAG 持久化服务
  - `session_store.py`：会话存储
- `app/utils`：前端静态资源（index.html）
- `tests`：关键回归测试
  - `test_rag_regressions.py`：RAG 回归测试
  - `test_hierarchical_helpers.py`：分层检索辅助测试
  - `test_rag_helpers.py`：RAG 辅助测试
- `data`：本地语料和会话数据
  - `local_corpus/`：本地语料
  - `sessions/`：会话数据
- `corpus_embs`：embedding 与 Chroma 持久化目录
- `models`：本地模型（embedding、rerank 等）

## 请求链路

1. `app/api/fastapi_app.py` 创建应用并注入共享上下文
2. `app/api/routes` 负责按领域拆分接口
3. `app/services` 负责衔接接口与核心能力
4. `app/core/rag.py` 统一编排检索、生成和索引操作

## 当前架构重点

- 入口层已拆分为配置、上下文、限流和路由模块，降低单文件耦合
- 路由层只处理协议与参数校验，核心业务交由服务层完成
- 应用共享资源统一收敛到 `AppContext`，避免 `app.state` 分散膨胀
- 运行时配置统一收敛到 `ApiRuntimeConfig`，减少路径与参数散落
- 支持 Ollama 模型调用，默认 32K 上下文窗口
- 支持查询扩展（子查询生成、意图分类、历史压缩）

## Section 层规则

- `section` collection 保留所有有内容的章节（含父章节和子章节）
- 每个 chunk 保留其原始章节归属（`section_hierarchy` 与 `section_title` 直接来自原始章节）
- `element` collection 中的 `section_id` 和 `parent_id` 与 section 层保持一致，用于 parent-child 检索提升
- `paper` 层中的 `section_count` 与 `section_titles` 按全部章节统计

## 启动方式

安装依赖：

```bash
.venv/bin/pip install -r requirements.txt
```

启动服务（Ollama 模式）：

```bash
uvicorn app.api.fastapi_app:app --host 0.0.0.0 --port 8000
```

### 主要 CLI 参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--gen_model_type` | `ollama` | 生成模型类型 |
| `--gen_model_name` | `qwen2.5:3b` | Ollama 模型名称 |
| `--ollama_host` | `http://127.0.0.1:11434` | Ollama 服务地址 |
| `--rerank_model_name` | `BAAI/bge-reranker-base` | Rerank 模型名称 |
| `--chunk_size` | `256` | 分块大小 |
| `--chunk_overlap` | `50` | 分块重叠 |
| `--query_expansion` | `True` | 启用查询扩展 |
| `--docling_use_ocr` | `True` | 启用 OCR |
| `--docling_table_structure` | `True` | 启用表格结构提取 |

## 测试

运行关键回归测试：

```bash
.venv/bin/python -m unittest tests.test_rag_regressions
```
