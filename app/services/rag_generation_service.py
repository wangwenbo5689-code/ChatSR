from threading import Thread
import re

from loguru import logger
from transformers import TextIteratorStreamer

from app.api.route_utils import clamp_generation_params
from app.core.rag_defaults import DEFAULT_CONTEXT_LEN, DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE
from app.services.rag_retrieval_service import DocumentSelectionRequiredError

DEFAULT_MAX_LENGTH = DEFAULT_MAX_NEW_TOKENS
DOCUMENT_SELECTION_REQUIRED_MESSAGE = DocumentSelectionRequiredError.DEFAULT_MESSAGE
DOCUMENT_SINGLE_SELECTION_REQUIRED_MESSAGE = DocumentSelectionRequiredError.SINGLE_DOCUMENT_MESSAGE


class RagGenerationService:
    def __init__(self, rag):
        self.rag = rag

    @staticmethod
    def _document_selection_prompt(intent, selected_doc_paths) -> str:
        normalized = (intent or "").strip().lower() if isinstance(intent, str) else ""
        if normalized != "document_overview":
            return ""
        if isinstance(selected_doc_paths, str):
            selected_doc_paths = [selected_doc_paths]
        selected_count = len([
            file_path for file_path in (selected_doc_paths or [])
            if isinstance(file_path, str) and file_path.strip()
        ])
        if selected_count <= 0:
            return DOCUMENT_SELECTION_REQUIRED_MESSAGE
        if selected_count != 1:
            return DOCUMENT_SINGLE_SELECTION_REQUIRED_MESSAGE
        return ""

    def _prepare_generation(
        self,
        query: str,
        context_len: int,
        history,
        history_summary="",
        selected_doc_paths=None,
        trace_ctx=None,
        intent=None,
    ):
        prompt, reference_results = self.rag._retrieval_service.build_prompt_and_references(
            query=query,
            context_len=context_len,
            selected_doc_paths=selected_doc_paths,
            trace_ctx=trace_ctx,
            history=history,
            history_summary=history_summary,
            intent=intent,
        )
        logger.debug(f"prompt: {prompt}")
        model_history = [list(pair) for pair in history]
        model_history.append([prompt, ""])
        return model_history, reference_results

    def _get_limit(self, key: str, default_value):
        limits = getattr(self.rag, "generation_limits", {}) or {}
        return limits.get(key, default_value)

    @staticmethod
    def _extract_source_line(reference_item: str) -> str:
        first_line = (reference_item.splitlines() or [""])[0].strip()
        match = re.match(r"^\[(\d+)\]\s*\(来源:\s*(.*?)\)\s*$", first_line)
        if match:
            idx = match.group(1).strip()
            source = match.group(2).strip()
            if idx and source:
                return f"[{idx}] {source}"
        return ""

    @staticmethod
    def _strip_source_prefix(source_line: str) -> str:
        stripped = re.sub(r"^\[\d+\]\s*", "", (source_line or "").strip())
        return stripped.strip()

    def _collect_source_lines(self, reference_results) -> list[str]:
        source_lines: list[str] = []
        seen = set()
        for item in (reference_results or []):
            source_line = self._extract_source_line(item)
            if not source_line:
                continue
            key = source_line.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            source_lines.append(source_line)
        return source_lines

    def _build_reference_tail(self, response: str, reference_results) -> str:
        source_lines = self._collect_source_lines(reference_results)
        ordered_sources: list[str] = []
        for source_line in source_lines:
            source_text = self._strip_source_prefix(source_line)
            if source_text:
                ordered_sources.append(source_text)
        if not ordered_sources:
            ordered_sources = ["无可用来源"]
        rendered = [f"[{index}] {item}" for index, item in enumerate(ordered_sources, start=1)]
        return "\n\n回答引用内容：\n" + "\n".join(rendered)

    def _ensure_reference_tail(self, response: str, reference_results) -> str:
        text = re.sub(
            r"\n{2,}(回答引用内容|参考来源)\s*[：:][\s\S]*$",
            "",
            (response or "").strip(),
            flags=re.IGNORECASE,
        )
        return (text + self._build_reference_tail(text, reference_results)).strip()

    @staticmethod
    def _sanitize_generation_params(max_new_tokens: int, context_len: int, temperature: float, limits: dict | None = None) -> tuple[int, int, float]:
        return clamp_generation_params(
            max_length=max_new_tokens,
            context_len=context_len,
            temperature=temperature,
            limits=limits or {},
        )

    @staticmethod
    def _resolve_generation_window(context_len: int, max_new_tokens: int) -> tuple[int, int]:
        safe_ctx = max(16, int(context_len or 16))
        safe_tokens = max(1, int(max_new_tokens or 1))
        max_tokens = max(safe_ctx - 8, 1)
        if safe_tokens > max_tokens:
            safe_tokens = max_tokens
        return safe_tokens, max(safe_ctx - safe_tokens - 8, 1)

    def _iter_answer_chunks(
        self,
        *,
        max_length: int,
        temperature: float,
        context_len: int,
        history,
        history_summary,
    ):
        for chunk in self.stream_generate_answer(
            max_new_tokens=max_length,
            temperature=temperature,
            context_len=context_len,
            history=history,
            history_summary=history_summary,
        ):
            if not self.rag.use_ollama and chunk == "</s>":
                continue
            yield chunk

    def stream_generate_answer(
        self,
        max_new_tokens=DEFAULT_MAX_LENGTH,
        temperature=DEFAULT_TEMPERATURE,
        repetition_penalty=1.0,
        context_len=DEFAULT_CONTEXT_LEN,
        history=None,
        history_summary=None,
    ):
        safe_tokens, safe_ctx, safe_temp = self._sanitize_generation_params(
            max_new_tokens=max_new_tokens,
            context_len=context_len,
            temperature=temperature,
            limits=getattr(self.rag, "generation_limits", {}),
        )
        if self.rag.use_ollama:
            yield from self._stream_ollama(safe_tokens, safe_ctx, safe_temp, history, history_summary)
            return
        yield from self._stream_local(safe_tokens, safe_ctx, safe_temp, repetition_penalty, history, history_summary)

    def _stream_ollama(self, max_tokens: int, context_len: int, temperature: float, history, history_summary):
        import ollama

        client = ollama.Client(host=self.rag.ollama_host)
        messages = self.rag._get_chat_input(history=history, history_summary=history_summary)
        try:
            stream = client.chat(
                model=self.rag.ollama_model,
                messages=messages,
                stream=True,
                options={'num_ctx': context_len, 'num_predict': max_tokens, 'temperature': temperature},
            )
            for chunk in stream:
                yield chunk['message']['content']
        except Exception as exc:
            logger.error(f"Ollama API error: {exc}")
            raise RuntimeError(f"Ollama API error: {exc}") from exc

    def _stream_local(self, max_tokens: int, context_len: int, temperature: float, repetition_penalty: float, history, history_summary):
        streamer = TextIteratorStreamer(self.rag.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        input_ids = self.rag._get_chat_input(history=history, history_summary=history_summary)
        max_tokens, max_src_len = self._resolve_generation_window(
            context_len=context_len,
            max_new_tokens=max_tokens,
        )
        input_ids = input_ids[-max_src_len:]
        thread = Thread(target=self.rag.gen_model.generate, kwargs=dict(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
        ))
        thread.start()
        yield from streamer

    def predict_stream(
        self,
        query: str,
        max_length: int = DEFAULT_MAX_LENGTH,
        context_len: int = DEFAULT_CONTEXT_LEN,
        temperature: float = DEFAULT_TEMPERATURE,
        history=None,
        history_summary=None,
        selected_doc_paths=None,
        trace_ctx=None,
        intent=None,
    ):
        history = history or []
        history_summary = history_summary or ""

        if not self.rag.enable_history:
            history, history_summary = [], ""

        selection_prompt = self._document_selection_prompt(intent, selected_doc_paths)
        if selection_prompt:
            yield selection_prompt
            yield {"history": history + [[query, selection_prompt]], "history_summary": history_summary}
            return

        try:
            model_history, reference_results = self._prepare_generation(
                query=query,
                context_len=context_len,
                history=history,
                history_summary=history_summary,
                selected_doc_paths=selected_doc_paths,
                trace_ctx=trace_ctx,
                intent=intent,
            )
        except DocumentSelectionRequiredError as exc:
            selection_prompt = str(exc) or DOCUMENT_SELECTION_REQUIRED_MESSAGE
            yield selection_prompt
            yield {"history": history + [[query, selection_prompt]], "history_summary": history_summary}
            return

        response = ""
        for chunk in self._iter_answer_chunks(
            max_length=max_length,
            temperature=temperature,
            context_len=context_len,
            history=model_history,
            history_summary=history_summary,
        ):
            response += chunk
            yield chunk

        final_response = self._ensure_reference_tail(response, reference_results)
        suffix = final_response[len(response):]
        if suffix:
            yield suffix
        response = final_response

        yield {"history": history + [[query, response]], "history_summary": history_summary}

    def predict(
        self,
        query: str,
        max_length: int = DEFAULT_MAX_LENGTH,
        context_len: int = DEFAULT_CONTEXT_LEN,
        temperature: float = DEFAULT_TEMPERATURE,
        selected_doc_paths=None,
        trace_ctx=None,
        intent=None,
    ):
        if not self.rag.enable_history:
            self.rag.history, self.rag.history_summary = [], ""

        selection_prompt = self._document_selection_prompt(intent, selected_doc_paths)
        if selection_prompt:
            response = selection_prompt
            self.rag.history.append([query, response])
            self.rag.history, self.rag.history_summary = self.rag.compress_history_if_needed(
                self.rag.history,
                self.rag.history_summary,
            )
            return response, []

        try:
            model_history, reference_results = self._prepare_generation(
                query=query,
                context_len=context_len,
                history=self.rag.history,
                history_summary=self.rag.history_summary,
                selected_doc_paths=selected_doc_paths,
                trace_ctx=trace_ctx,
                intent=intent,
            )
        except DocumentSelectionRequiredError as exc:
            response = str(exc) or DOCUMENT_SELECTION_REQUIRED_MESSAGE
            self.rag.history.append([query, response])
            self.rag.history, self.rag.history_summary = self.rag.compress_history_if_needed(
                self.rag.history,
                self.rag.history_summary,
            )
            return response, []

        response = "".join(
            self._iter_answer_chunks(
                max_length=max_length,
                temperature=temperature,
                context_len=context_len,
                history=model_history,
                history_summary=self.rag.history_summary,
            )
        ).strip()
        response = self._ensure_reference_tail(response, reference_results)
        self.rag.history.append([query, response])
        self.rag.history, self.rag.history_summary = self.rag.compress_history_if_needed(self.rag.history, self.rag.history_summary)
        return response, reference_results
