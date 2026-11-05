"""Lớp bọc Ollama dùng chung cho trích xuất KG và các agent.

- structured(): ép LLM nộp kết quả qua một tool `submit_*` có schema Pydantic. Ollama chưa hỗ trợ
  `format=` (structured output) cho qwen3.5, còn tool calling thì ổn định.
- run_tools(): vòng gọi tool tự viết, tối đa max_steps bước.
- Mọi lượt gọi được ghi vào Usage (số lượt, token, thời gian) để đưa vào trace của agent.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import ollama
from pydantic import BaseModel, ValidationError

from config import EMBED_DIM, EMBED_MODEL, LLM_MODEL, NUM_CTX, OLLAMA_HOST

# Gửi embedding theo lô nhỏ: một request cả nghìn văn bản lúc máy thiếu RAM làm runner của Ollama bị dừng giữa chừng
EMBED_BATCH = 64
EMBED_RETRIES = 3


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def add(self, response, seconds: float) -> None:
        self.calls += 1
        self.prompt_tokens += response.prompt_eval_count or 0
        self.completion_tokens += response.eval_count or 0
        self.seconds += seconds

    def as_dict(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "seconds": round(self.seconds, 2)}


@dataclass
class ToolRun:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


def inline_refs(schema: dict) -> dict:
    """Thay các $ref trỏ vào $defs bằng định nghĩa đầy đủ (model nhỏ đọc schema phẳng tốt hơn)."""
    defs = schema.get("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def submit_tool(schema: type[BaseModel]) -> tuple[str, dict]:
    name = "submit_" + re.sub(r"(?<!^)(?=[A-Z])", "_", schema.__name__).lower()
    return name, {
        "type": "function",
        "function": {"name": name, "description": (schema.__doc__ or "Nộp kết quả").strip(),
                     "parameters": inline_refs(schema.model_json_schema())},
    }


class LLM:
    def __init__(self, model: str = LLM_MODEL, host: str = OLLAMA_HOST, num_ctx: int = NUM_CTX):
        self.model = model
        self.num_ctx = num_ctx
        self.client = ollama.Client(host=host)

    def chat(self, messages: list[dict], *, tools: list | None = None, think: bool = False,
             usage: Usage | None = None, temperature: float = 0.0):
        start = time.perf_counter()
        response = self.client.chat(model=self.model, messages=messages, tools=tools, think=think,
                                    options={"temperature": temperature, "num_ctx": self.num_ctx},
                                    keep_alive="30m")
        if usage is not None:
            usage.add(response, time.perf_counter() - start)
        return response

    def stream(self, messages: list[dict], *, tools: list | None = None, think: bool = False,
               usage: Usage | None = None, temperature: float = 0.0, on_text: Callable[[str], None] | None = None,
               stop_on_text: bool = False):
        """Như chat nhưng nhận kết quả dần. on_text(chữ đã sinh) được gọi mỗi khi có chữ mới; stop_on_text=True thì
        ngắt ngay khi model bắt đầu viết chữ thay vì gọi công cụ (Ollama dừng sinh khi kết nối bị đóng)."""
        start = time.perf_counter()
        text, tool_calls, last = "", [], None
        chunks = self.client.chat(model=self.model, messages=messages, tools=tools, think=think, stream=True,
                                  options={"temperature": temperature, "num_ctx": self.num_ctx}, keep_alive="30m")
        try:
            for chunk in chunks:
                last = chunk
                tool_calls += chunk.message.tool_calls or []
                if chunk.message.content:
                    text += chunk.message.content
                    if stop_on_text and not tool_calls and len(text.strip()) >= 20:
                        break
                    if on_text is not None:
                        on_text(text)
        finally:
            if hasattr(chunks, "close"):
                chunks.close()
        if usage is not None:
            usage.add(last or SimpleNamespace(prompt_eval_count=0, eval_count=0), time.perf_counter() - start)
        return SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=tool_calls or None))

    def structured(self, schema: type[BaseModel], system: str, user: str, *, think: bool = False,
                   retries: int = 2, usage: Usage | None = None) -> BaseModel:
        name, tool = submit_tool(schema)
        messages = [{"role": "system", "content": f"{system}\n\nBắt buộc gọi công cụ `{name}` đúng một lần để nộp kết quả."},
                    {"role": "user", "content": user}]
        last_error = "không gọi công cụ"
        for _ in range(retries + 1):
            response = self.chat(messages, tools=[tool], think=think, usage=usage)
            calls = [c for c in (response.message.tool_calls or []) if c.function.name == name]
            payload = dict(calls[0].function.arguments) if calls else _json_from_text(response.message.content)
            if payload is not None:
                try:
                    return schema.model_validate(_coerce_json_strings(payload))
                except ValidationError as error:
                    last_error = str(error)[:500]
            messages.append({"role": "assistant", "content": response.message.content or "",
                             "tool_calls": response.message.tool_calls})
            messages.append({"role": "user", "content": f"Kết quả chưa hợp lệ ({last_error}). Hãy gọi lại `{name}` "
                                                         "với dữ liệu đúng schema."})
        raise ValueError(f"LLM không trả về {schema.__name__} hợp lệ: {last_error}")

    def run_tools(self, system: str, user: str, tools: list[Callable], *, think: bool = False,
                  max_steps: int = 6, usage: Usage | None = None, need_text: bool = True) -> ToolRun:
        """Vòng gọi công cụ. need_text=False: chỉ cần kết quả công cụ, nên khi model chuyển sang viết đoạn tóm tắt
        cuối thì ngắt luôn (tiết kiệm 5–15 giây mỗi lượt với model 9B)."""
        usage = usage or Usage()
        registry = {fn.__name__: fn for fn in tools}
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        log: list[dict] = []
        for _ in range(max_steps):
            if need_text:
                response = self.chat(messages, tools=tools, think=think, usage=usage)
            else:
                response = self.stream(messages, tools=tools, think=think, usage=usage, stop_on_text=True)
            calls = response.message.tool_calls or []
            if not calls:
                return ToolRun(response.message.content or "", log, usage)
            messages.append({"role": "assistant", "content": response.message.content or "", "tool_calls": calls})
            for call in calls:
                fn = registry.get(call.function.name)
                args = dict(call.function.arguments)
                try:
                    result = fn(**args) if fn else {"error": f"không có công cụ {call.function.name}"}
                except Exception as error:  # trả lỗi về cho model tự sửa tham số
                    result = {"error": f"{type(error).__name__}: {error}"}
                log.append({"tool": call.function.name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_name": call.function.name,
                                 "content": json.dumps(result, ensure_ascii=False, default=str)[:6000]})
        return ToolRun("", log, usage)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        rows: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start:start + EMBED_BATCH]
            for attempt in range(1, EMBED_RETRIES + 1):
                try:
                    rows += self.client.embed(model=EMBED_MODEL, input=batch, keep_alive="30m").embeddings
                    break
                except (ollama.ResponseError, ConnectionError):
                    if attempt == EMBED_RETRIES:
                        raise
                    time.sleep(3 * attempt)   # Ollama tự khởi động lại runner ở request sau
        vectors = np.asarray(rows, dtype=np.float32).reshape(len(texts), -1)
        return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)


def _json_from_text(text: str | None) -> dict | None:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


NULL_STRINGS = {"none", "null", "n/a", "na", "", "không có", "không rõ"}


def _coerce_json_strings(payload: Any) -> Any:
    """Model đôi khi trả mảng/đối tượng dưới dạng chuỗi JSON, hoặc chuỗi 'None' thay cho null; chuẩn hóa lại."""
    if isinstance(payload, str) and payload.strip().lower() in NULL_STRINGS:
        return None
    if isinstance(payload, dict):
        return {k: _coerce_json_strings(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_coerce_json_strings(v) for v in payload]
    if isinstance(payload, str) and payload[:1] in "[{":
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload
    return payload
