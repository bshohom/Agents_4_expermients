from __future__ import annotations

import os
from typing import Any

import requests


class LocalLLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: int = 600,
    ) -> None:
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "Qwen/Qwen3.5-9B")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "dummy")
        self.timeout_s = timeout_s

    def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout_s,
        )

        if not resp.ok:
            raise RuntimeError(
                f"Local LLM server returned {resp.status_code}:\n{resp.text}\n\nPayload was:\n{payload}"
            )

        data = resp.json()

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected response format from local LLM server: {data}") from e