"""Application settings loaded from environment variables.

A single module-level ``settings`` singleton is created at import time.
All components that need configuration should import ``settings`` from
this module rather than reading ``os.environ`` directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    embedding_model: str


settings: Settings = Settings(
    embedding_model=os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
)
