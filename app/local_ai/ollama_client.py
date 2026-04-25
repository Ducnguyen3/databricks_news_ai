from __future__ import annotations

import logging
import os

import requests
from dotenv import load_dotenv

from app.config import load_settings

logger = logging.getLogger(__name__)

_OLLAMA_ERROR = "Khong goi duoc Ollama. Hay chay: ollama pull mistral:7b va ollama serve"


class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        load_dotenv()
        settings = load_settings()
        self._base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or settings.local_ai.ollama_base_url or "").rstrip("/")
        self._model = model or os.getenv("OLLAMA_MODEL") or settings.local_ai.ollama_model or ""
        self._timeout_seconds = timeout_seconds

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
        except Exception as exc:
            logger.warning("Ollama request failed", exc_info=True)
            raise RuntimeError(_OLLAMA_ERROR) from exc

        answer = str(payload.get("response") or "").strip()
        if not answer:
            raise RuntimeError(_OLLAMA_ERROR)
        return answer
