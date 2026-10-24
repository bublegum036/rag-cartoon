"""Small .env loader: literal values only, no shell execution/interpolation."""

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


MODEL_REVISION = "fd1525a9fd15316a2d503bf26ab031a61d056e98"


def read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"Invalid .env entry on line {number}")
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Unclosed .env quote on line {number}")
            value = value[1:-1]
        values[key] = value
    return values


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_alias: str = "cartoons"
    qdrant_api_key: str = field(default="", repr=False)
    timeout_seconds: float = 10.0
    model_revision: str = MODEL_REVISION
    model_cache: Path = Path(".cache/models")
    embedding_threads: int = 2
    chunk_chars: int = 1200

    def __post_init__(self) -> None:
        url = urlsplit(self.qdrant_url)
        if url.scheme not in ("http", "https") or not url.hostname or url.query or url.fragment:
            raise ValueError("QDRANT_URL must be an HTTP(S) URL without query/fragment")
        if url.username or url.password:
            raise ValueError("Use QDRANT_API_KEY instead of credentials in QDRANT_URL")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", self.qdrant_alias):
            raise ValueError("QDRANT_ALIAS must contain 1–100 letters, digits, underscores or hyphens")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("QDRANT_TIMEOUT_SECONDS must be positive and finite")
        if self.embedding_threads < 1 or self.chunk_chars < 1:
            raise ValueError("Thread count and chunk size must be positive")
        if not re.fullmatch(r"[0-9a-f]{40}", self.model_revision):
            raise ValueError("EMBEDDING_REVISION must be an immutable 40-character commit SHA")

    @classmethod
    def load(cls, env_file: Path = Path(".env")) -> "Settings":
        values = {**read_env(env_file), **os.environ}
        return cls(
            qdrant_url=values.get("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/"),
            qdrant_alias=values.get("QDRANT_ALIAS", "cartoons"),
            qdrant_api_key=values.get("QDRANT_API_KEY", ""),
            timeout_seconds=float(values.get("QDRANT_TIMEOUT_SECONDS", "10")),
            model_revision=values.get("EMBEDDING_REVISION", MODEL_REVISION),
            model_cache=Path(values.get("MODEL_CACHE", ".cache/models")),
            embedding_threads=int(values.get("EMBEDDING_THREADS", "2")),
            chunk_chars=int(values.get("CHUNK_CHARS", "1200")),
        )
