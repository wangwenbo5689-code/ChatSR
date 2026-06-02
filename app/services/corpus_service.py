import hashlib
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from fastapi import HTTPException, UploadFile
from loguru import logger

from app.core.rag_defaults import (
    DEFAULT_MAX_FILE_SIZE_MB,
    DEFAULT_MAX_FILES_PER_UPLOAD,
    FILE_READ_CHUNK_SIZE_BYTES,
)


class CorpusService:
    def __init__(
        self,
        local_corpus_dir: Path,
        allowed_exts: Iterable[str],
        max_files_per_upload: int = DEFAULT_MAX_FILES_PER_UPLOAD,
        max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
    ):
        self.local_corpus_dir = Path(local_corpus_dir)
        self.allowed_exts = set(allowed_exts)
        self.max_files_per_upload = max_files_per_upload
        self.max_file_size_mb = max_file_size_mb

    @staticmethod
    def md5_file(path: str) -> str:
        hasher = hashlib.md5()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(FILE_READ_CHUNK_SIZE_BYTES), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _write_upload_to_temp(upload_file: UploadFile, temp_path: str) -> Tuple[bytes, int]:
        try:
            upload_file.file.seek(0)
        except Exception:
            pass

        first_bytes = b""
        total_size = 0
        with open(temp_path, "wb") as out:
            while True:
                chunk = upload_file.file.read(FILE_READ_CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                if not first_bytes:
                    first_bytes = chunk[:8]
                total_size += len(chunk)
                out.write(chunk)
        return first_bytes, total_size

    @staticmethod
    def _is_safe_filename(filename: str) -> bool:
        return bool(filename) and ".." not in filename and "/" not in filename and "\\" not in filename

    def _validate_upload(self, filename: str, ext: str, first_bytes: bytes, total_size: int) -> str | None:
        if not self._is_safe_filename(filename):
            return "type"
        if ext not in self.allowed_exts:
            return "type"
        if total_size > self.max_file_size_mb * FILE_READ_CHUNK_SIZE_BYTES:
            return "size"
        if ext == ".pdf" and not first_bytes.startswith(b"%PDF-"):
            return "type"
        return None

    def _handle_invalid_upload(
        self,
        temp_path: str,
        filename: str,
        invalid_reason: str,
        skipped_type: int,
        skipped_size: int,
    ) -> tuple[int, int]:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if invalid_reason == "size":
            logger.warning(f"Skipped {filename}: exceeds {self.max_file_size_mb}MB")
            return skipped_type, skipped_size + 1
        return skipped_type + 1, skipped_size

    def _persist_upload(
        self,
        temp_path: str,
        filename: str,
        ext: str,
        existing_hash_to_path: dict,
    ) -> tuple[str | None, bool]:
        file_hash = self.md5_file(temp_path)
        if file_hash in existing_hash_to_path:
            os.remove(temp_path)
            return None, True

        final_name = os.path.basename(filename or f"upload{ext}")
        final_name = "".join(char for char in final_name if char.isalnum() or char in "_- .")
        final_path = str(self.local_corpus_dir / final_name)
        if os.path.exists(final_path):
            stem = Path(final_name).stem
            suffix = Path(final_name).suffix
            final_path = str(self.local_corpus_dir / f"{stem}_{file_hash[:8]}{suffix}")

        shutil.move(temp_path, final_path)
        existing_hash_to_path[file_hash] = final_path
        return final_path, False

    @staticmethod
    def _index_ingested_paths(ingest_paths: Sequence[str], model, rag_lock) -> None:
        if not ingest_paths:
            return
        try:
            with rag_lock:
                model.add_corpus(ingest_paths, persist_files=True)
        except Exception as exc:
            logger.exception(exc)
            raise HTTPException(status_code=500, detail="处理文件时发生错误")

    def _build_ingest_result(
        self,
        ingested_count: int,
        saved_paths: Sequence[str],
        skipped_dup: int,
        skipped_type: int,
        skipped_size: int,
    ) -> dict:
        message_parts = [f"成功导入: {ingested_count}"]
        message_parts.append(f"已保存到本地语料库: {len(saved_paths)}")
        if skipped_dup:
            message_parts.append(f"重复跳过: {skipped_dup}")
        if skipped_type:
            message_parts.append(f"类型不支持: {skipped_type}")
        if skipped_size:
            message_parts.append(f"超出 {self.max_file_size_mb}MB: {skipped_size}")
        return {
            "message": " | ".join(message_parts),
            "paths": list(saved_paths),
            "save_to_local_corpus": True,
            "ingested": ingested_count,
        }

    def _build_existing_hash_index(self) -> dict:
        existing_hash_to_path = {}
        for file_path in self.list_local_corpus_files():
            try:
                existing_hash_to_path[self.md5_file(file_path)] = file_path
            except Exception as exc:
                logger.warning(f"skip hashing existing file {file_path}: {exc}")
        return existing_hash_to_path

    def list_local_corpus_files(self) -> List[str]:
        if not self.local_corpus_dir.is_dir():
            return []
        return sorted(
            str(self.local_corpus_dir / name)
            for name in os.listdir(self.local_corpus_dir)
            if os.path.isfile(str(self.local_corpus_dir / name))
            and Path(name).suffix.lower() in self.allowed_exts
        )

    def list_local_corpus_items(self) -> List[dict]:
        items = []
        for file_path in self.list_local_corpus_files():
            path_obj = Path(file_path)
            try:
                stat_info = path_obj.stat()
                items.append(
                    {
                        "name": path_obj.name,
                        "path": str(path_obj),
                        "size_bytes": int(stat_info.st_size),
                        "updated_at": int(stat_info.st_mtime * 1000),
                    }
                )
            except OSError as exc:
                logger.warning(f"skip local corpus file metadata {file_path}: {exc}")
        def sort_key(item: dict) -> int:
            value = item.get("updated_at")
            return value if isinstance(value, int) else 0

        items.sort(key=sort_key, reverse=True)
        return items

    def ingest_uploads(self, files: Sequence[UploadFile], model, rag_lock) -> dict:
        if not files:
            return {"message": "未选择文件。"}
        if len(files) > self.max_files_per_upload:
            return {"message": f"单次最多上传 {self.max_files_per_upload} 个文件。"}

        os.makedirs(str(self.local_corpus_dir), exist_ok=True)

        existing_hash_to_path = self._build_existing_hash_index()

        ingest_paths = []
        saved_paths = []
        skipped_dup, skipped_type, skipped_size = 0, 0, 0

        with tempfile.TemporaryDirectory(prefix="chatsr_upload_") as temp_dir:
            for upload_file in files:
                filename = upload_file.filename or ""
                ext = Path(filename).suffix.lower()
                temp_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}{ext or '.tmp'}")
                first_bytes, total_size = self._write_upload_to_temp(upload_file, temp_path)

                invalid_reason = self._validate_upload(
                    filename=filename,
                    ext=ext,
                    first_bytes=first_bytes,
                    total_size=total_size,
                )
                if invalid_reason is not None:
                    skipped_type, skipped_size = self._handle_invalid_upload(
                        temp_path=temp_path,
                        filename=filename,
                        invalid_reason=invalid_reason,
                        skipped_type=skipped_type,
                        skipped_size=skipped_size,
                    )
                    continue

                saved_path, is_dup = self._persist_upload(
                    temp_path=temp_path,
                    filename=filename,
                    ext=ext,
                    existing_hash_to_path=existing_hash_to_path,
                )
                if is_dup:
                    skipped_dup += 1
                    continue
                if saved_path is None:
                    continue
                saved_paths.append(saved_path)
                ingest_path = saved_path
                ingest_paths.append(ingest_path)

            self._index_ingested_paths(ingest_paths, model, rag_lock)

        return self._build_ingest_result(
            ingested_count=len(ingest_paths),
            saved_paths=saved_paths,
            skipped_dup=skipped_dup,
            skipped_type=skipped_type,
            skipped_size=skipped_size,
        )
