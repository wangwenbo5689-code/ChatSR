import hashlib
import os
from typing import Iterable

from app.core.rag_defaults import FILE_READ_CHUNK_SIZE_BYTES


def file_md5_hex_32(file_path: str) -> str:
    """计算文件的 MD5 哈希值（32位）

    Args:
        file_path: 文件路径

    Returns:
        str: MD5 哈希值（32位）
    """
    hasher = hashlib.md5()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(FILE_READ_CHUNK_SIZE_BYTES), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:32]


def files_md5_hex_32(file_paths: Iterable[str]) -> str:
    """计算多个文件的 MD5 哈希值（32位）

    Args:
        file_paths: 文件路径列表

    Returns:
        str: MD5 哈希值（32位）
    """
    hasher = hashlib.md5()
    for file_path in sorted(file_paths):
        with open(file_path, 'rb') as file:
            for chunk in iter(lambda: file.read(FILE_READ_CHUNK_SIZE_BYTES), b""):
                hasher.update(chunk)
    return hasher.hexdigest()[:32]


def resolve_embedding_dir(save_corpus_emb_dir: str, corpus_files: Iterable[str]) -> str | None:
    """解析嵌入目录

    Args:
        save_corpus_emb_dir: 保存嵌入的目录
        corpus_files: 语料文件列表

    Returns:
        str | None: 嵌入目录路径
    """
    files = [file_path for file_path in corpus_files if file_path]
    if not files:
        return None
    return os.path.join(save_corpus_emb_dir, files_md5_hex_32(files))
