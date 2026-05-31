# -*- coding: utf-8 -*-
"""
基于 Docling 的文档提取器
- 统一结构化提取接口
- 兼容不同 Docling 版本的属性差异
- 支持按参数启用/禁用 OCR
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from loguru import logger

def load_docling_components():
    """加载 Docling 组件

    Returns:
        tuple: DocumentConverter, PdfFormatOption, InputFormat, PdfPipelineOptions

    Raises:
        ImportError: 当 Docling 未安装时
    """
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
    except ImportError as exc:
        raise ImportError("Docling 未安装") from exc
    return DocumentConverter, PdfFormatOption, InputFormat, PdfPipelineOptions


class DoclingExtractor:
    """基于 Docling 的文档提取器"""

    def __init__(
        self,
        use_ocr: bool = True,
        ocr_language: str = "zh-CN",
        enable_table_structure: bool = True,
        table_structure_mode: str = "accurate",
    ):
        """初始化提取器

        Args:
            use_ocr: 是否启用 OCR
            ocr_language: OCR 识别语言
            enable_table_structure: 是否启用表格结构识别
            table_structure_mode: 表格结构识别模式
        """
        self.use_ocr = bool(use_ocr)
        self.ocr_language = ocr_language
        self.converter: Optional[Any] = None
        self.enable_table_structure = bool(enable_table_structure)
        self.table_structure_mode = (table_structure_mode or "").strip().lower()
        self._last_convert_cache_key: Optional[str] = None
        self._last_convert_result: Any = None

        try:
            document_converter_cls, pdf_format_option_cls, input_format_cls, pdf_pipeline_options_cls = load_docling_components()
        except ImportError:
            logger.warning("Docling 未可用，提取器将无法工作")
            return

        try:
            pipeline_options = pdf_pipeline_options_cls()
            pipeline_options.do_ocr = self.use_ocr

            # 设置 OCR 语言，支持列表或单个字符串
            if hasattr(pipeline_options, "ocr_options"):
                # 新版本 Docling 使用 ocr_options
                if self.ocr_language:
                    langs = [self.ocr_language] if isinstance(self.ocr_language, str) else self.ocr_language
                    pipeline_options.ocr_options.lang = langs
            elif hasattr(pipeline_options, "ocr_language"):
                pipeline_options.ocr_language = self.ocr_language

            # 默认关闭强制全文 OCR 模式，使 Docling 仅在必要时（如图片、表格或无文本层页面）触发 OCR
            if hasattr(pipeline_options, "force_full_page_ocr"):
                pipeline_options.force_full_page_ocr = False

            # 设置 OCR 选项，确保对图像和表格进行文字提取
            if self.use_ocr and hasattr(pipeline_options, "ocr_options"):
                # 如果存在 ocr_options，可以在此处进一步配置具体行为
                pass

            if hasattr(pipeline_options, "do_table_structure"):
                pipeline_options.do_table_structure = self.enable_table_structure
            elif hasattr(pipeline_options, "enable_table_structure"):
                pipeline_options.enable_table_structure = self.enable_table_structure

            if self.enable_table_structure and hasattr(pipeline_options, "table_structure_options"):
                table_structure_options = getattr(pipeline_options, "table_structure_options", None)
                if table_structure_options is not None:
                    if hasattr(table_structure_options, "do_cell_matching"):
                        table_structure_options.do_cell_matching = True
                    if self.table_structure_mode and hasattr(table_structure_options, "mode"):
                        mode_value: Any = self.table_structure_mode
                        try:
                            from docling.datamodel.pipeline_options import TableFormerMode

                            upper_mode = self.table_structure_mode.upper()
                            if hasattr(TableFormerMode, upper_mode):
                                mode_value = getattr(TableFormerMode, upper_mode)
                        except Exception:
                            mode_value = self.table_structure_mode
                        table_structure_options.mode = mode_value

            self.converter = document_converter_cls(
                format_options={
                    input_format_cls.PDF: pdf_format_option_cls(pipeline_options=pipeline_options)
                }
            )
            logger.info(
                "Docling 提取器初始化完成 "
                f"(OCR: {'启用' if self.use_ocr else '禁用'}, "
                f"表格结构: {'启用' if self.enable_table_structure else '禁用'}, "
                f"表格模式: {self.table_structure_mode or '默认'})"
            )
        except Exception as e:
            self.converter = None
            logger.warning(f"Docling 转换器初始化失败: {e}")

    def _ensure_ready(self):
        """确保 Docling 组件已准备就绪

        Raises:
            ImportError: 当 Docling 未安装时
            RuntimeError: 当转换器未初始化时
        """
        try:
            load_docling_components()
        except ImportError as exc:
            raise ImportError("Docling 未安装。请通过以下命令安装: pip install docling") from exc
        if not self.converter:
            raise RuntimeError("Docling 转换器未初始化")

    @staticmethod
    def _safe_page(item: Any) -> Optional[int]:
        """安全获取页码信息

        Args:
            item: 要提取页码的对象

        Returns:
            Optional[int]: 页码
        """
        for attr in ("page_no", "page"):
            value = getattr(item, attr, None)
            if isinstance(value, int):
                return value
        prov_entries = getattr(item, "prov", None)
        if isinstance(prov_entries, (list, tuple)):
            for prov in prov_entries:
                if isinstance(prov, dict):
                    value = prov.get("page_no")
                else:
                    value = getattr(prov, "page_no", None)
                if isinstance(value, int):
                    return value
        return None

    def _convert(self, file_path: str):
        """转换文件

        Args:
            file_path: 文件路径

        Returns:
            Any: 转换结果

        Raises:
            FileNotFoundError: 当文件不存在时
            RuntimeError: 当转换器未初始化时
        """
        self._ensure_ready()
        converter = self.converter
        if converter is None:
            raise RuntimeError("Docling 转换器未初始化")
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        stat = path_obj.stat()
        cache_key = f"{path_obj.resolve()}::{stat.st_size}::{stat.st_mtime_ns}"
        if cache_key == self._last_convert_cache_key and self._last_convert_result is not None:
            return self._last_convert_result
        result = converter.convert(path_obj)
        self._last_convert_cache_key = cache_key
        self._last_convert_result = result
        return result

    def extract_document(self, file_path: str):
        """提取文档

        返回 Docling Document 对象（用于高级分块器）

        Args:
            file_path: 文件路径

        Returns:
            Any: Docling Document 对象

        Raises:
            RuntimeError: 当文档为空时
        """
        result = self._convert(file_path)
        if not result or not getattr(result, "document", None):
            raise RuntimeError(f"空文档: {file_path}")
        return result.document


def create_docling_extractor(
    use_ocr: bool = True,
    ocr_language: Any = None,
    enable_table_structure: bool = True,
    table_structure_mode: str = "accurate",
) -> Optional[DoclingExtractor]:
    """创建 Docling 提取器工厂函数

    Args:
        use_ocr: 是否启用 OCR
        ocr_language: OCR 识别语言，默认为 ["en", "zh"]
        enable_table_structure: 是否启用表格结构识别
        table_structure_mode: 表格结构识别模式

    Returns:
        Optional[DoclingExtractor]: Docling 提取器实例，若 Docling 不可用则返回 None
    """
    try:
        load_docling_components()
    except ImportError:
        logger.warning("Docling 不可用，返回 None")
        return None

    # 默认支持中英文
    if ocr_language is None:
        ocr_language = ["en", "zh"]

    return DoclingExtractor(
        use_ocr=use_ocr,
        ocr_language=ocr_language,
        enable_table_structure=enable_table_structure,
        table_structure_mode=table_structure_mode,
    )
