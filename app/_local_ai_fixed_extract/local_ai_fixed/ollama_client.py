from __future__ import annotations

import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv

from app.config import load_settings

logger = logging.getLogger(__name__)

_OLLAMA_ERROR = "Khong goi duoc Ollama."


class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        load_dotenv()
        settings = load_settings()
        self._base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or settings.local_ai.ollama_base_url or "").rstrip("/")
        self._model = model or os.getenv("OLLAMA_MODEL") or settings.local_ai.ollama_model or ""
        self._timeout_seconds = int(timeout_seconds or os.getenv("OLLAMA_TIMEOUT_SECONDS") or 180)

    def generate(self, prompt: str) -> str:
        if not self._base_url or not self._model:
            raise RuntimeError(
                "Missing Ollama configuration. Set OLLAMA_BASE_URL and OLLAMA_MODEL before running the chatbot."
            )

        try:
            response = requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            body = exc.response.text[:500] if exc.response is not None else ""
            logger.warning(
                "Ollama request failed model=%s base_url=%s status=%s body=%s",
                self._model,
                self._base_url,
                status,
                body,
                exc_info=True,
            )
            if status == 404:
                raise RuntimeError(
                    f"{_OLLAMA_ERROR} Model hoac endpoint khong ton tai: model={self._model}. "
                    f"Hay chay: ollama pull {self._model} va ollama serve"
                ) from exc
            raise RuntimeError(f"{_OLLAMA_ERROR} HTTP status={status} model={self._model}") from exc
        except requests.Timeout as exc:
            logger.warning(
                "Ollama request timed out model=%s base_url=%s timeout_seconds=%s prompt_chars=%s",
                self._model,
                self._base_url,
                self._timeout_seconds,
                len(prompt),
                exc_info=True,
            )
            raise RuntimeError(
                f"{_OLLAMA_ERROR} timeout sau {self._timeout_seconds}s "
                f"model={self._model} base_url={self._base_url} prompt_chars={len(prompt)}"
            ) from exc
        except requests.ConnectionError as exc:
            logger.warning(
                "Ollama connection failed model=%s base_url=%s prompt_chars=%s",
                self._model,
                self._base_url,
                len(prompt),
                exc_info=True,
            )
            raise RuntimeError(
                f"{_OLLAMA_ERROR} khong ket noi duoc Ollama "
                f"model={self._model} base_url={self._base_url}"
            ) from exc
        except Exception as exc:
            logger.warning(
                "Ollama request failed type=%s message=%s model=%s base_url=%s prompt_chars=%s",
                type(exc).__name__,
                str(exc),
                self._model,
                self._base_url,
                len(prompt),
                exc_info=True,
            )
            raise RuntimeError(
                f"{_OLLAMA_ERROR} type={type(exc).__name__} message={str(exc)} "
                f"model={self._model} base_url={self._base_url} prompt_chars={len(prompt)}"
            ) from exc

        answer = str(payload.get("response") or "").strip()
        if not answer:
            raise RuntimeError(f"{_OLLAMA_ERROR} empty response model={self._model}")
        return answer

    def runtime_config(self) -> dict[str, Any]:
        return {
            "base_url": self._base_url,
            "model": self._model,
            "timeout_seconds": self._timeout_seconds,
        }
