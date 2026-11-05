"""Đọc danh sách trang Long Châu và chuẩn hóa URL."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from config import RAW_HTML_DIR, SEED_URLS_PATH


@dataclass(frozen=True)
class Seed:
    url: str
    type: str
    topic: str

    @property
    def slug(self) -> str:
        return slug_for(self.url)

    @property
    def html_path(self) -> Path:
        return RAW_HTML_DIR / f"{self.slug}.html"


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/") or "/"
    return f"https://{host}{path}"


def slug_for(url: str) -> str:
    path = urlsplit(normalize_url(url)).path.strip("/")
    path = re.sub(r"\.html?$", "", path)
    return re.sub(r"[^a-z0-9]+", "_", path.lower()).strip("_")


def load_seeds(path: Path = SEED_URLS_PATH) -> list[Seed]:
    with open(path, encoding="utf-8") as f:
        return [Seed(row["url"], row["type"], row["topic"]) for row in csv.DictReader(f)]
