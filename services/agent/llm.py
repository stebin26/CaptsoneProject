# services/agent/llm.py
"""
Ollama tool-calling client — the agent's brain.

A transparent, framework-independent wrapper around Ollama's /api/chat endpoint.
Given a message history and a set of tool schemas, it asks the local model to
either (a) call a tool or (b) return a final answer, and hands back a normalized
response object that the agent graph can consume without knowing Ollama's format.

Design goals for a small (3B) local model:
  - use Ollama's native tool-calling, with a JSON-in-content fallback
  - strict, defensive parsing so malformed model output never crashes the loop
  - transport retries with backoff (model load / server-busy tolerance)
  - keep the raw response for full visibility while debugging the spike
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

# Use the project logger if ops_common is importable; fall back to stdlib so
# this file also runs standalone during the spike (terminal, no full stack).
try:
    from ops_common.logging import get_logger

    logger = get_logger(__name__)
except Exception:  # pragma: no cover - fallback only
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


# --- Configuration -----------------------------------------------------------
# Read from OPS_-prefixed env vars (the project's standard) with defaults that
# match the current setup: Ollama on the Windows host, llama3.2:3b.

def _env(key: str, default: str) -> str:
    return os.getenv(f"OPS_{key}", default)


@dataclass
class LLMConfig:
    host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://host.docker.internal:11434"))
    model: str = field(default_factory=lambda: _env("AGENT_MODEL", "llama3.2:3b"))
    temperature: float = field(default_factory=lambda: float(_env("AGENT_TEMPERATURE", "0.1")))
    request_timeout: int = field(default_factory=lambda: int(_env("AGENT_LLM_TIMEOUT", "120")))
    max_retries: int = field(default_factory=lambda: int(_env("AGENT_LLM_RETRIES", "3")))
    backoff_seconds: float = field(default_factory=lambda: float(_env("AGENT_LLM_BACKOFF", "1.5")))
    num_ctx: int = field(default_factory=lambda: int(_env("AGENT_NUM_CTX", "4096")))


# --- Normalized response types -----------------------------------------------
# The rest of the agent only ever sees these two shapes, never raw Ollama JSON.

@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall]
    raw: dict[str, Any]

    @property
    def is_final(self) -> bool:
        # A final answer is any response that did not request a tool.
        return len(self.tool_calls) == 0


class LLMTransportError(RuntimeError):
    """Raised when Ollama cannot be reached / satisfied after all retries."""


# --- Client ------------------------------------------------------------------

class OllamaToolClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        base = self.config.host.rstrip("/")
        self._chat_url = f"{base}/api/chat"
        self._tags_url = f"{base}/api/tags"

    # ---- public API ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        # One decision step: send history + tool schemas, get tool-call OR answer.
        payload = self._build_payload(messages, tools)
        raw = self._post_with_retry(payload)
        response = self._parse(raw)
        logger.info(
            "LLM step -> %s",
            "final answer" if response.is_final else f"tool={response.tool_calls[0].name}",
        )
        return response

    def health_check(self) -> dict[str, Any]:
        # Confirm Ollama is up and whether the target model is actually pulled.
        try:
            resp = requests.get(self._tags_url, timeout=10)
            resp.raise_for_status()
            models = [m.get("name", "") for m in resp.json().get("models", [])]
        except requests.RequestException as exc:
            return {"reachable": False, "model_present": False, "models": [], "error": str(exc)}
        present = any(m == self.config.model or m.startswith(self.config.model) for m in models)
        return {"reachable": True, "model_present": present, "models": models}

    # ---- request building ---------------------------------------------------

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        # Non-streaming so we get a single complete message to parse tool calls from.
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_ctx": self.config.num_ctx,
            },
        }
        if tools:
            payload["tools"] = tools
        return payload

    # ---- transport with retry ----------------------------------------------

    def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = requests.post(self._chat_url, json=payload, timeout=self.config.request_timeout)
                resp.raise_for_status()
                return resp.json()
            except (requests.ConnectionError, requests.Timeout) as exc:
                # Transient transport issue — retry with growing backoff.
                last_err = exc
                logger.warning("Ollama transport error (%d/%d): %s", attempt, self.config.max_retries, exc)
            except requests.HTTPError as exc:
                # Retry 5xx (model loading / busy); fail fast on 4xx (bad request).
                status = exc.response.status_code if exc.response is not None else 0
                last_err = exc
                if status < 500:
                    logger.error("Ollama client error %s: %s", status, exc)
                    raise LLMTransportError(f"Ollama returned {status}") from exc
                logger.warning("Ollama server error %s (%d/%d)", status, attempt, self.config.max_retries)
            time.sleep(self.config.backoff_seconds * attempt)
        raise LLMTransportError(
            f"Ollama unreachable after {self.config.max_retries} attempts"
        ) from last_err

    # ---- response parsing ---------------------------------------------------

    def _parse(self, raw: dict[str, Any]) -> LLMResponse:
        message = raw.get("message", {}) or {}
        content = (message.get("content") or "").strip()

        # 1) Preferred: the model used Ollama's native tool_calls field.
        native = self._extract_native_tool_calls(message)
        if native:
            return LLMResponse(content=content, tool_calls=native, raw=raw)

        # 2) Fallback: small models sometimes emit the tool call as JSON in text.
        embedded = self._extract_embedded_tool_call(content)
        if embedded:
            return LLMResponse(content="", tool_calls=[embedded], raw=raw)

        # 3) Otherwise it's a plain final answer.
        return LLMResponse(content=content, tool_calls=[], raw=raw)

    @staticmethod
    def _extract_native_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = (tc or {}).get("function", {}) or {}
            name = fn.get("name")
            if not name:
                continue
            args = fn.get("arguments", {})
            # Ollama usually returns a dict; tolerate a JSON string just in case.
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(name=str(name), arguments=args if isinstance(args, dict) else {}))
        return calls

    @staticmethod
    def _extract_embedded_tool_call(content: str) -> ToolCall | None:
        if not content:
            return None
        block = OllamaToolClient._find_json_object(content)
        if not block:
            return None
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        # Accept the few shapes a small model tends to produce.
        name = data.get("tool") or data.get("name") or data.get("tool_name")
        if not name:
            return None
        args = data.get("arguments") or data.get("args") or data.get("parameters") or {}
        return ToolCall(name=str(name), arguments=args if isinstance(args, dict) else {})

    @staticmethod
    def _find_json_object(text: str) -> str | None:
        # Strip ```json fences if present, then return the first balanced {...}.
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidate = fenced.group(1) if fenced else text
        start = candidate.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(candidate)):
            ch = candidate[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return candidate[start : i + 1]
        return None


# --- Module-level convenience ------------------------------------------------

_default_client: OllamaToolClient | None = None


def get_llm() -> OllamaToolClient:
    # Lazy singleton so the whole agent shares one configured client.
    global _default_client
    if _default_client is None:
        _default_client = OllamaToolClient()
    return _default_client