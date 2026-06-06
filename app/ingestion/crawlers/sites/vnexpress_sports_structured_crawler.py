from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.ingestion.crawlers.sites.vnexpress_crawler import VNEXPRESS_STRUCTURED_PATHS
from app.processing.canonicalizer import normalize_url


@dataclass(frozen=True, slots=True)
class StructuredSportsPage:
    source: str
    url: str
    path: str
    html: str


class VnExpressSportsStructuredCrawler:
    source_name = "vnexpress"

    def __init__(self, source_config: Any) -> None:
        self.config = source_config

    def build_structured_page_urls(self) -> list[str]:
        base_url = str(getattr(self.config, "base_url", "")).rstrip("/")
        paths = tuple(getattr(self.config, "structured_paths", ())) or tuple(VNEXPRESS_STRUCTURED_PATHS)
        return [normalize_url(f"{base_url}/{path.strip('/')}") for path in paths if path.strip("/")]

    def parse_page(self, url: str, html: str) -> StructuredSportsPage:
        path = url.replace(str(getattr(self.config, "base_url", "")).rstrip("/") + "/", "", 1)
        return StructuredSportsPage(source=self.source_name, url=url, path=path, html=html)
