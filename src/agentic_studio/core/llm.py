from __future__ import annotations

import hashlib
from dataclasses import dataclass

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None

from agentic_studio.core.config import settings


@dataclass(slots=True)
class LLMResponse:
    text: str
    provider: str
    model: str


class LLM:
    def __init__(self) -> None:
        self._model = None
        if settings.gemini_api_key and genai is not None:
            genai.configure(api_key=settings.gemini_api_key)
            self._model = genai.GenerativeModel(model_name=settings.gemini_model)

    def generate(self, prompt: str, system: str = "You are a precise AI assistant.") -> LLMResponse:
        if self._model is None:
            return self._mock_response(prompt)

        response = self._model.generate_content(
            [f"System instruction: {system}", prompt],
            generation_config={"temperature": settings.temperature},
        )
        text = (response.text or "").strip()
        return LLMResponse(text=text, provider="gemini", model=settings.gemini_model)

    def _mock_response(self, prompt: str) -> LLMResponse:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        brief = prompt[:300].replace("\n", " ").strip()
        text = (
            "[MOCK MODE] This response is deterministic because GEMINI_API_KEY is not set.\n\n"
            f"Prompt hash: {digest}\n"
            f"Prompt preview: {brief}\n\n"
            "Actionable output:\n"
            "1. Use retrieved evidence first.\n"
            "2. Highlight uncertainty and risks.\n"
            "3. Recommend next experiments with measurable KPIs."
        )
        return LLMResponse(text=text, provider="mock", model="rules-v1")