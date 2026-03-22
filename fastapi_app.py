# -*- coding: utf-8 -*-
"""
ChatPDF FastAPI backend.
- File upload + corpus ingestion
- Session management (persisted)
- SSE streaming chat
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from rag import Rag

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

ALLOWED_EXTS = {".pdf", ".docx", ".md", ".txt", ".jsonl"}
MAX_FILES_PER_UPLOAD = 10
MAX_FILE_SIZE_MB = 20
SESSION_STORE_PATH = os.path.join("data", "sessions", "sessions.json")


# ---------- Session storage ----------
# in-memory structure:
# {
#   "session_id": {
#      "title": str,
#      "updated_at": int(ms),
#      "history": [[user, assistant], ...]
#   }
# }
sessions: Dict[str, Dict] = {}
session_lock = threading.Lock()
rag_lock = threading.Lock()


def now_ms() -> int:
    return int(time.time() * 1000)


def normalize_user_message(text: str) -> str:
    """Strip RAG prompt wrapper and keep only original user question if present."""
    if not text:
        return ""
    marker = "\n\n问题:\n"
    if text.startswith("基于以下已知信息") and marker in text:
        return text.split(marker, 1)[1].strip()
    return text


def _migrate_legacy_sessions(data):
    """Support old formats gracefully."""
    if not isinstance(data, dict):
        return {}

    # New format already
    if all(isinstance(v, dict) and "history" in v for v in data.values()):
        cleaned = {}
        for sid, item in data.items():
            history = item.get("history", [])
            normalized_history = []
            for p in history:
                if isinstance(p, list) and len(p) == 2:
                    normalized_history.append([normalize_user_message(p[0]), p[1]])
            cleaned[sid] = {
                "title": (item.get("title") or "新对话").strip() or "新对话",
                "updated_at": int(item.get("updated_at") or now_ms()),
                "history": normalized_history,
                "summary": (item.get("summary") or "").strip(),
            }
        return cleaned

    # Legacy format: {sid: [[q,a], ...]}
    cleaned = {}
    for sid, value in data.items():
        if isinstance(value, list):
            history = [[normalize_user_message(p[0]), p[1]] for p in value if isinstance(p, list) and len(p) == 2]
            title = "新对话"
            if history and history[0][0]:
                title = str(history[0][0])[:24]
            cleaned[sid] = {
                "title": title,
                "updated_at": now_ms(),
                "history": history,
                "summary": "",
            }
    return cleaned


def load_sessions() -> Dict[str, Dict]:
    os.makedirs(os.path.dirname(SESSION_STORE_PATH), exist_ok=True)
    if not os.path.exists(SESSION_STORE_PATH):
        return {}
    try:
        with open(SESSION_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _migrate_legacy_sessions(data)
    except Exception as e:
        logger.warning(f"failed to load sessions: {e}")
        return {}


def save_sessions() -> None:
    os.makedirs(os.path.dirname(SESSION_STORE_PATH), exist_ok=True)
    tmp = SESSION_STORE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)
    os.replace(tmp, SESSION_STORE_PATH)


def ensure_session(session_id: str) -> Dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "title": "新对话",
            "updated_at": now_ms(),
            "history": [],
            "summary": "",
        }
    else:
        sessions[session_id].setdefault("summary", "")
    return sessions[session_id]


# ---------- RAG init ----------
parser = argparse.ArgumentParser()
parser.add_argument("--gen_model_type", type=str, default="ollama")
parser.add_argument("--gen_model_name", type=str, default="qwen2.5:3b")
parser.add_argument("--lora_model", type=str, default=None)
parser.add_argument("--rerank_model_name", type=str, default="BAAI/bge-reranker-base")
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--corpus_files", type=str, default="")
parser.add_argument("--int4", action="store_true")
parser.add_argument("--int8", action="store_true")
parser.add_argument("--chunk_size", type=int, default=220)
parser.add_argument("--chunk_overlap", type=int, default=0)
parser.add_argument("--num_expand_context_chunk", type=int, default=2)
parser.add_argument("--history_max_turns", type=int, default=6)
parser.add_argument("--history_keep_last_turns", type=int, default=2)
parser.add_argument("--ollama_host", type=str, default="http://127.0.0.1:11434")
args, _ = parser.parse_known_args()

corpus_files_list = []
if args.corpus_files:
    corpus_files_list = [f.strip() for f in args.corpus_files.split(",") if f.strip()]

model = Rag(
    generate_model_type=args.gen_model_type,
    generate_model_name_or_path=args.gen_model_name,
    lora_model_name_or_path=args.lora_model,
    corpus_files=corpus_files_list,
    device=args.device,
    int4=args.int4,
    int8=args.int8,
    chunk_size=args.chunk_size,
    chunk_overlap=args.chunk_overlap,
    num_expand_context_chunk=args.num_expand_context_chunk,
    rerank_model_name_or_path=args.rerank_model_name,
    enable_history=True,
    history_max_turns=args.history_max_turns,
    history_keep_last_turns=args.history_keep_last_turns,
    ollama_host=args.ollama_host,
)
logger.info(f"model loaded: {model}")

sessions = load_sessions()


# ---------- App ----------
app = FastAPI(title="ChatPDF API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_local_corpus_files() -> List[str]:
    corpus_dir = os.path.join("data", "local_corpus")
    if not os.path.isdir(corpus_dir):
        return []
    files = []
    for name in os.listdir(corpus_dir):
        p = os.path.join(corpus_dir, name)
        if os.path.isfile(p) and Path(p).suffix.lower() in ALLOWED_EXTS:
            files.append(p)
    files.sort()
    return files


def bootstrap_local_corpus() -> None:
    """Load local corpus/embeddings on service startup."""
    corpus_files = list_local_corpus_files()
    if not corpus_files:
        logger.info("no local corpus files found on startup")
        return

    with rag_lock:
        model.corpus_files = corpus_files
        hash_name = model.get_file_hash(corpus_files)
        emb_dir = os.path.join(model.save_corpus_emb_dir, hash_name)
        if os.path.isdir(emb_dir):
            model.load_corpus_emb(emb_dir)
            logger.info(f"loaded local embedding index: {emb_dir}, chunks={len(model.sim_model.corpus)}")
        else:
            model.add_corpus(corpus_files)
            logger.info(f"loaded local corpus files: {len(corpus_files)}, chunks={len(model.sim_model.corpus)}")


bootstrap_local_corpus()


@app.get("/")
def index_page():
    return FileResponse("static/index.html")


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/embeddings/status")
def embeddings_status():
    corpus_files = list_local_corpus_files()
    emb_dir = None
    loaded = False
    if corpus_files:
        hash_name = model.get_file_hash(corpus_files)
        emb_dir = os.path.join(model.save_corpus_emb_dir, hash_name)
        loaded = os.path.isdir(emb_dir)
    return {
        "corpus_file_count": len(corpus_files),
        "embedding_dir": emb_dir,
        "embedding_exists": loaded,
        "in_memory_chunk_count": len(getattr(model.sim_model, "corpus", {})),
    }


@app.post("/api/embeddings/save")
def save_embeddings():
    corpus_files = list_local_corpus_files()
    if not corpus_files:
        raise HTTPException(status_code=400, detail="本地语料库为空，无法落盘 embedding 索引")
    if not getattr(model.sim_model, "corpus", None):
        raise HTTPException(status_code=400, detail="当前内存中没有语料 chunk，请先上传或加载语料")

    model.corpus_files = corpus_files
    save_dir = model.save_corpus_emb()
    return {"ok": True, "embedding_dir": save_dir, "chunk_count": len(model.sim_model.corpus)}


@app.post("/api/embeddings/load")
def load_embeddings():
    corpus_files = list_local_corpus_files()
    if not corpus_files:
        raise HTTPException(status_code=400, detail="本地语料库为空，无法加载 embedding 索引")

    hash_name = model.get_file_hash(corpus_files)
    emb_dir = os.path.join(model.save_corpus_emb_dir, hash_name)
    if not os.path.isdir(emb_dir):
        raise HTTPException(status_code=404, detail=f"未找到 embedding 索引目录: {emb_dir}")

    with rag_lock:
        model.corpus_files = corpus_files
        model.load_corpus_emb(emb_dir)

    return {"ok": True, "embedding_dir": emb_dir, "chunk_count": len(model.sim_model.corpus)}


@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    save_to_local_corpus: bool = Form(True),
):
    if not files:
        return {"message": "未选择文件。"}
    if len(files) > MAX_FILES_PER_UPLOAD:
        return {"message": f"单次最多上传 {MAX_FILES_PER_UPLOAD} 个文件。"}

    corpus_dir = os.path.join("data", "local_corpus")
    os.makedirs(corpus_dir, exist_ok=True)

    existing_hash_to_path = {}
    if save_to_local_corpus:
        for name in os.listdir(corpus_dir):
            p = os.path.join(corpus_dir, name)
            if os.path.isfile(p):
                try:
                    existing_hash_to_path[md5_file(p)] = p
                except Exception as e:
                    logger.warning(f"skip hashing existing file {p}: {e}")

    ingest_paths = []
    saved_paths = []
    temp_paths = []
    skipped_dup, skipped_type, skipped_size = 0, 0, 0

    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTS:
            skipped_type += 1
            continue

        raw = await f.read()
        if len(raw) > MAX_FILE_SIZE_MB * 1024 * 1024:
            skipped_size += 1
            continue

        if save_to_local_corpus:
            temp_path = os.path.join(corpus_dir, f".tmp_{uuid.uuid4().hex}{ext}")
            with open(temp_path, "wb") as out:
                out.write(raw)

            file_hash = md5_file(temp_path)
            if file_hash in existing_hash_to_path:
                skipped_dup += 1
                os.remove(temp_path)
                continue

            final_name = os.path.basename(f.filename or f"upload{ext}")
            final_path = os.path.join(corpus_dir, final_name)
            if os.path.exists(final_path):
                stem = Path(final_name).stem
                suffix = Path(final_name).suffix
                final_path = os.path.join(corpus_dir, f"{stem}_{file_hash[:8]}{suffix}")

            shutil.move(temp_path, final_path)
            existing_hash_to_path[file_hash] = final_path
            saved_paths.append(final_path)
            ingest_paths.append(final_path)
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(raw)
                temp_path = tmp.name
            temp_paths.append(temp_path)
            ingest_paths.append(temp_path)

    try:
        if ingest_paths:
            model.add_corpus(ingest_paths)
    finally:
        for p in temp_paths:
            try:
                os.remove(p)
            except Exception as e:
                logger.warning(f"failed to remove temp file {p}: {e}")

    msg = [f"成功导入: {len(ingest_paths)}"]
    if save_to_local_corpus:
        msg.append(f"已保存到本地语料库: {len(saved_paths)}")
    else:
        msg.append("未保存到本地语料库")
    if skipped_dup:
        msg.append(f"重复跳过: {skipped_dup}")
    if skipped_type:
        msg.append(f"类型不支持: {skipped_type}")
    if skipped_size:
        msg.append(f"超出 {MAX_FILE_SIZE_MB}MB: {skipped_size}")

    return {
        "message": " | ".join(msg),
        "paths": saved_paths,
        "save_to_local_corpus": save_to_local_corpus,
        "ingested": len(ingest_paths),
    }


@app.get("/api/sessions")
def get_sessions():
    with session_lock:
        items = []
        for sid, info in sessions.items():
            items.append(
                {
                    "id": sid,
                    "title": info.get("title", "新对话"),
                    "updated_at": info.get("updated_at", 0),
                    "message_count": len(info.get("history", [])) * 2,
                }
            )
        items.sort(key=lambda x: x["updated_at"], reverse=True)
    return {"sessions": items}


@app.get("/api/sessions/{session_id}")
def get_session_messages(session_id: str):
    with session_lock:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="session not found")
        history = sessions[session_id].get("history", [])
        title = sessions[session_id].get("title", "新对话")
        summary = sessions[session_id].get("summary", "")

    messages = []
    for q, a in history:
        messages.append({"role": "user", "content": q or ""})
        messages.append({"role": "assistant", "content": a or ""})

    return {"id": session_id, "title": title, "summary": summary, "messages": messages}


@app.post("/api/sessions")
def create_session():
    sid = uuid.uuid4().hex
    with session_lock:
        ensure_session(sid)
        save_sessions()
    return {"id": sid}


@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: str, title: str = Form(...)):
    title = (title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title不能为空")

    with session_lock:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="session not found")
        sessions[session_id]["title"] = title[:100]
        sessions[session_id]["updated_at"] = now_ms()
        save_sessions()
    return {"ok": True}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    with session_lock:
        existed = session_id in sessions
        if existed:
            del sessions[session_id]
            save_sessions()
    return {"ok": True, "deleted": existed}


@app.delete("/api/sessions")
def clear_sessions():
    with session_lock:
        sessions.clear()
        save_sessions()
    return {"ok": True}


@app.post("/api/chat/stream")
def chat_stream(
    message: str = Form(...),
    session_id: str = Form(...),
    max_length: int = Form(1024),
    context_len: int = Form(8192),
    temperature: float = Form(0.7),
):
    message = (message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message不能为空")

    with session_lock:
        session = ensure_session(session_id)
        history_snapshot = [pair[:] for pair in session.get("history", [])]
        summary_snapshot = session.get("summary", "")

    def event_gen():
        assistant_full = ""
        try:
            # model has mutable history; lock to avoid cross-session race
            with rag_lock:
                model.history = [[normalize_user_message(p[0]), p[1]] for p in history_snapshot]
                model.history_summary = summary_snapshot
                safe_max_length = max(64, min(int(max_length), 4096))
                safe_context_len = max(1024, min(int(context_len), 200000))
                safe_temperature = max(0.0, min(float(temperature), 2.0))
                for chunk in model.predict_stream(
                    message,
                    max_length=safe_max_length,
                    context_len=safe_context_len,
                    temperature=safe_temperature,
                ):
                    assistant_full = chunk
                    yield f"data: {json.dumps({'type': 'delta', 'content': chunk}, ensure_ascii=False)}\n\n"
                latest_summary = model.history_summary

            with session_lock:
                session = ensure_session(session_id)
                session["history"] = history_snapshot + [[message, assistant_full]]
                session["summary"] = latest_summary
                if session["title"] == "新对话":
                    session["title"] = message[:24]
                session["updated_at"] = now_ms()
                save_sessions()

            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception(e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


app.mount("/static", StaticFiles(directory="static"), name="static")
