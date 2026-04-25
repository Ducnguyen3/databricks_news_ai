from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Iterable

from app.config import CrawlSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SourceConfig:
    source_name: str
    base_url: str
    homepage_url: str
    rss_url: str | None
    category_paths: tuple[str, ...]
    enabled: bool
    discover_categories: bool
    max_pages_per_category: int
    stop_after_empty_pages: int
    stop_after_duplicate_pages: int
    request_delay_seconds: float
    timeout_seconds: int
    retry_count: int
    user_agent: str
    crawl_homepage: bool = True


class SourceManagementService:
    def __init__(self, settings: CrawlSettings) -> None:
        self._settings = settings
        self._source_configs = self._build_default_source_configs(settings)

    def list_registered_sources(self) -> list[str]:
        return sorted(self._source_configs)

    def get_source_config(self, source_name: str) -> SourceConfig:
        config = self._source_configs.get(source_name)
        if config is None:
            supported = ", ".join(self.list_registered_sources())
            logger.error("Unsupported source=%s supported_sources=%s", source_name, supported)
            raise ValueError(f"Unsupported source '{source_name}'. Supported sources: {supported}")
        return config

    def get_enabled_sources(self, source_names: Iterable[str] | None = None) -> list[SourceConfig]:
        requested_sources = tuple(source_names or self._settings.sources)
        configs: list[SourceConfig] = []
        for source_name in requested_sources:
            config = self.get_source_config(source_name)
            if not config.enabled:
                logger.info("[CRAWL] Skip disabled source=%s", source_name)
                continue
            configs.append(config)
        return configs

    def enabled_crawlers(self) -> list[object]:
        from app.ingestion.crawlers.registry import default_crawlers

        return default_crawlers(self.get_enabled_sources())

    @staticmethod
    def _build_default_source_configs(settings: CrawlSettings) -> dict[str, SourceConfig]:
        def config(
            source_name: str,
            base_url: str,
            category_paths: tuple[str, ...],
            rss_url: str | None = None,
            homepage_url: str | None = None,
        ) -> SourceConfig:
            return SourceConfig(
                source_name=source_name,
                base_url=base_url,
                homepage_url=homepage_url or base_url,
                rss_url=rss_url,
                category_paths=category_paths,
                enabled=True,
                discover_categories=settings.discover_categories,
                max_pages_per_category=settings.max_pages_per_category,
                stop_after_empty_pages=settings.stop_after_empty_pages,
                stop_after_duplicate_pages=settings.stop_after_duplicate_pages,
                request_delay_seconds=settings.request_delay_seconds,
                timeout_seconds=settings.request_timeout_seconds,
                retry_count=settings.retry_count,
                user_agent=settings.user_agent,
            )

        return {
            "vnexpress": config(
                source_name="vnexpress",
                base_url="https://vnexpress.net",
                homepage_url="https://vnexpress.net",
                rss_url="https://vnexpress.net/rss/tin-moi-nhat.rss",
                category_paths=(
                    "thoi-su",
                    "kinh-doanh",
                    "the-gioi",
                    "khoa-hoc-cong-nghe",
                ),
            ),
            "cafef": config(
                source_name="cafef",
                base_url="https://cafef.vn",
                rss_url="https://cafef.vn/thi-truong-chung-khoan.rss",
                category_paths=(
                    "thi-truong-chung-khoan.chn",
                    "bat-dong-san.chn",
                    "doanh-nghiep.chn",
                    "tai-chinh-ngan-hang.chn",
                ),
            ),
            "genk": config(
                source_name="genk",
                base_url="https://genk.vn",
                rss_url="https://genk.vn/rss/home.rss",
                category_paths=(
                    "mobile.chn",
                    "internet.chn",
                    "kham-pha.chn",
                    "ai.chn",
                ),
            ),
            "diendandoanhnghiep": config(
                source_name="diendandoanhnghiep",
                base_url="https://diendandoanhnghiep.vn",
                category_paths=(
                    "kinh-te",
                    "tai-chinh",
                    "bat-dong-san",
                    "khoi-nghiep",
                ),
            ),
        }


def with_limits(config: SourceConfig, max_pages_per_category: int | None = None) -> SourceConfig:
    if max_pages_per_category is None:
        return config
    return replace(config, max_pages_per_category=max_pages_per_category)
