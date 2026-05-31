import torch
from loguru import logger


def resolve_default_device():
    """解析默认设备

    Returns:
        设备对象，优先选择 GPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")  # NVIDIA GPU
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    return torch.device("cpu")  # CPU


def init_generation_model(
    device,
    gen_model_type: str,
    gen_model_name_or_path: str,
    peft_name: str | None = None,
    int8: bool = False,
    int4: bool = False,
):
    """初始化生成模型

    Args:
        device: 设备对象
        gen_model_type: 模型类型
        gen_model_name_or_path: 模型名称或路径
        peft_name: PEFT 模型名称
        int8: 是否使用 int8 量化
        int4: 是否使用 int4 量化

    Returns:
        模型和分词器对象，Ollama 模式返回 (None, None)
    """
    if gen_model_type == "ollama":
        return None, None  # Ollama 模式不需要本地模型
    logger.warning(f"模型类型 {gen_model_type} 不支持，仅支持 ollama")
    return None, None
