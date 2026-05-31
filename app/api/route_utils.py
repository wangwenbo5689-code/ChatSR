from fastapi import HTTPException
from loguru import logger
from pydantic import BaseModel, Field


def require_non_empty_text(value: str, detail: str) -> str:
    text = (value or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail=detail)
    return text


def require_valid_identifier(value: str, validator, detail: str) -> str:
    if not validator(value):
        raise HTTPException(status_code=400, detail=detail)
    return value


def require_text_max_length(
    value: str,
    max_length: int,
    empty_detail: str,
    *,
    too_long_detail: str | None = None,
) -> str:
    text = require_non_empty_text(value, empty_detail)
    if len(text) > max_length:
        raise HTTPException(
            status_code=400,
            detail=too_long_detail or f"长度不能超过{max_length}个字符",
        )
    return text


def run_action(
    action,
    error_detail: str,
    *,
    status_code: int = 500,
    key_error_detail: str | None = None,
):
    try:
        return action()
    except HTTPException:
        raise
    except KeyError:
        if key_error_detail is not None:
            raise HTTPException(status_code=404, detail=key_error_detail)
        raise
    except Exception as exc:
        logger.exception(exc)
        raise HTTPException(status_code=status_code, detail=error_detail)


def clamp_generation_params(
    *,
    max_length,
    context_len,
    temperature,
    limits,
):
    """将生成参数钳制在配置限制范围内。

    对 None 值用默认值填充，超出边界时裁剪到合法范围内。
    不抛出异常，由 Pydantic 模型在前置层完成校验。
    """
    defaults = {
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
    defaults.update(limits)

    max_length = int(max_length or defaults["max_length_default"])
    context_len = int(context_len or defaults["context_len_default"])
    temperature = float(temperature if temperature is not None else defaults["temperature_default"])

    max_length = max(int(defaults["max_length_min"]), min(max_length, int(defaults["max_length_max"])))
    context_len = max(int(defaults["context_len_min"]), min(context_len, int(defaults["context_len_max"])))
    temperature = max(float(defaults["temperature_min"]), min(temperature, float(defaults["temperature_max"])))
    return max_length, context_len, temperature


def parse_selected_doc_paths(doc_paths_json, corpus_service):
    """解析并验证用户选择的文档路径列表。

    从 JSON 字符串解析出文档路径，并与本地语料文件交叉比对，
    过滤掉不存在的路径。
    """
    import json

    try:
        selected = json.loads(doc_paths_json or "[]")
    except json.JSONDecodeError:
        return []

    if not isinstance(selected, list):
        return []

    local_files = set(corpus_service.list_local_corpus_files())
    return [
        path
        for path in selected
        if isinstance(path, str) and path in local_files
    ]


class ChatStreamForm(BaseModel):
    """聊天流式请求的表单参数模型"""

    message: str = Field(..., min_length=1, description="用户消息")
    session_id: str = Field(..., min_length=1, description="会话ID")
    max_length: int | None = Field(None, ge=1, description="最大生成长度")
    context_len: int | None = Field(None, ge=1, description="上下文长度")
    temperature: float | None = Field(None, ge=0, description="温度参数")
    selected_doc_paths_json: str = Field("[]", description="选中文档路径 JSON")
