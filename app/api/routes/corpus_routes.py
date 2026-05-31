from typing import List

from fastapi import APIRouter, File, Request, UploadFile

from app.api.app_context import get_app_context
from app.api.route_utils import run_action


router = APIRouter()


@router.get("/api/corpus/files")
def list_local_corpus_files(request: Request):
    context = get_app_context(request)
    files = context.corpus_service.list_local_corpus_items()
    return {
        "count": len(files),
        "files": files,
    }


@router.post("/api/upload")
def upload_files(
    request: Request,
    files: List[UploadFile] = File(...),
):
    context = get_app_context(request)
    return run_action(
        lambda: context.corpus_service.ingest_uploads(
            files=files,
            model=context.model,
            rag_lock=context.rag_lock,
        ),
        "上传文件失败",
    )
