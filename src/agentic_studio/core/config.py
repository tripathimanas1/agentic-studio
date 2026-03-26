from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    max_chunks: int = int(os.getenv("MAX_RAG_CHUNKS", "6"))
    temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.1"))


settings = Settings()