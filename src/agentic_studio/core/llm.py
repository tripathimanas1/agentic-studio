from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

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


@dataclass(slots=True)
class ToolCallResponse:
    """Response from LLM requesting tool calls."""
    tool_name: str
    arguments: dict[str, Any]
    reasoning: str = ""


@dataclass(slots=True)
class LLMToolResponse:
    """Response from LLM with tool calling."""
    text: str | None  # Final text if reasoning ends
    tool_calls: list[ToolCallResponse]  # Tool calls to make
    provider: str
    model: str
    raw_response: Any = None  # Store raw API response for debugging


class LLM:
    def __init__(self) -> None:
        self._model = None
        print(f"[LLM] Initializing LLM")
        print(f"[LLM] GEMINI_API_KEY set: {bool(settings.gemini_api_key)}")
        print(f"[LLM] GEMINI_API_KEY length: {len(settings.gemini_api_key) if settings.gemini_api_key else 0}")
        print(f"[LLM] genai available: {genai is not None}")
        
        if settings.gemini_api_key and genai is not None:
            print(f"[LLM] ✓ Configuring Gemini API")
            genai.configure(api_key=settings.gemini_api_key)
            self._model = genai.GenerativeModel(model_name=settings.gemini_model)
            print(f"[LLM] ✓ Model initialized: {settings.gemini_model}")
        else:
            print(f"[LLM] ✗ Using MOCK mode (no API key or genai unavailable)")

    def generate(self, prompt: str, system: str = "You are a precise AI assistant.") -> LLMResponse:
        if self._model is None:
            return self._mock_response(prompt)

        response = self._model.generate_content(
            [f"System instruction: {system}", prompt],
            generation_config={"temperature": settings.temperature},
        )
        text = (response.text or "").strip()
        return LLMResponse(text=text, provider="gemini", model=settings.gemini_model)

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        system: str = "You are a reasoning AI assistant. Use tools to accomplish your goal.",
    ) -> LLMToolResponse:
        """Generate response with tool calling capability."""
        if self._model is None:
            return self._mock_tool_response(prompt, tools)

        try:
            response = self._model.generate_content(
                [f"System instruction: {system}", prompt],
                generation_config={"temperature": settings.temperature},
                tools=[{"google_search_retrieval": {"disable_out_of_domain_filter": False}}]
                if not tools
                else [{"function_declarations": tools}],
            )

            # Parse tool calls from response
            tool_calls = []
            if response.function_calls:
                for func_call in response.function_calls:
                    tool_calls.append(
                        ToolCallResponse(
                            tool_name=func_call.name,
                            arguments=dict(func_call.args),
                            reasoning="",
                        )
                    )

            text = None
            if response.text:
                text = response.text.strip()

            return LLMToolResponse(
                text=text,
                tool_calls=tool_calls,
                provider="gemini",
                model=settings.gemini_model,
                raw_response=response,
            )
        except Exception as e:
            # Fallback to mock if tool calling fails
            return self._mock_tool_response(prompt, tools)

    def generate_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        system: str = "You are a precise AI assistant.",
    ) -> dict[str, Any]:
        """Generate a response with structured output matching the schema."""
        if self._model is None:
            return {"mock": True, "schema_applied": True}

        try:
            response = self._model.generate_content(
                [f"System instruction: {system}", prompt],
                generation_config={"temperature": settings.temperature},
            )

            text = response.text or "{}"
            # Try to parse as JSON
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # If not valid JSON, return as text
                return {"raw_text": text, "parsed": False}
        except Exception:
            return {"mock": True, "error": "Structured generation failed"}

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

    def _mock_tool_response(
        self, prompt: str, tools: list[dict[str, Any]]
    ) -> LLMToolResponse:
        """Mock tool response for testing."""
        tool_names = [t.get("name", "unknown") for t in tools]

        # Simulate deciding which tool to call based on prompt
        if "retrieve" in prompt.lower() or "search" in prompt.lower():
            tool_name = "retrieve" if "retrieve" in tool_names else tool_names[0] if tool_names else "plan"
        else:
            tool_name = tool_names[0] if tool_names else "plan"

        return LLMToolResponse(
            text=None,
            tool_calls=[
                ToolCallResponse(
                    tool_name=tool_name,
                    arguments={},
                    reasoning="[MOCK] Determined tool to call based on prompt analysis.",
                )
            ],
            provider="mock",
            model="rules-v1",
        )