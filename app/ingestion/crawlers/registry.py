from __future__ import annotations

import logging
from typing import Iterable

from app.ingestion.crawlers.base import BaseCrawler
from app.ingestion.crawlers.sites.cafef_crawler import CafeFCrawler
from app.ingestion.crawlers.sites.diendandoanhnghiep_crawler import DienDanDoanhNghiepCrawler
from app.ingestion.crawlers.sites.genk_crawler import GenKCrawler
from app.ingestion.crawlers.sites.vnexpress_crawler import VnExpressCrawler

logger = logging.getLogger(__name__)

_REGISTERED_CRAWLERS: dict[str, type[BaseCrawler]] = {
    VnExpressCrawler.source_name: VnExpressCrawler,
    CafeFCrawler.source_name: CafeFCrawler,
    GenKCrawler.source_name: GenKCrawler,
    DienDanDoanhNghiepCrawler.source_name: DienDanDoanhNghiepCrawler,
}


class CrawlerRegistry:
    def __init__(self, registry: dict[str, type[BaseCrawler]] | None = None) -> None:
        self._registry = registry or _REGISTERED_CRAWLERS

    def get(self, source_name: str) -> type[BaseCrawler]:
        crawler_cls = self._registry.get(source_name)
        if crawler_cls is None:
            supported = ", ".join(self.list_registered_sources())
            logger.error("Unsupported crawler source=%s supported_sources=%s", source_name, supported)
            raise ValueError(f"Unsupported crawler source '{source_name}'. Supported sources: {supported}")
        return crawler_cls

    def get_crawler(self, source_name: str, source_config: object | None = None) -> BaseCrawler:
        crawler_cls = self.get(source_name)
        if source_config is None:
            source_config = _default_source_config(source_name)
        return crawler_cls(source_config)

    def list_registered_sources(self) -> list[str]:
        return sorted(self._registry)

    def default_crawlers(self, source_configs: Iterable[object] | None = None) -> list[BaseCrawler]:
        configs = list(source_configs) if source_configs is not None else _default_source_configs()
        return [self.get_crawler(str(getattr(config, "source_name")), config) for config in configs]


_DEFAULT_REGISTRY = CrawlerRegistry()


def crawler_registry() -> dict[str, type[BaseCrawler]]:
    return dict(_REGISTERED_CRAWLERS)


def get(source_name: str) -> type[BaseCrawler]:
    return _DEFAULT_REGISTRY.get(source_name)


def get_crawler(source_name: str, source_config: object | None = None) -> BaseCrawler:
    return _DEFAULT_REGISTRY.get_crawler(source_name, source_config)


def list_registered_sources() -> list[str]:
    return _DEFAULT_REGISTRY.list_registered_sources()


def default_crawlers(source_configs: Iterable[object] | None = None) -> list[BaseCrawler]:
    return _DEFAULT_REGISTRY.default_crawlers(source_configs)


def build_crawlers(source_names: tuple[str, ...] | list[str]) -> list[BaseCrawler]:
    configs_by_name = {config.source_name: config for config in _default_source_configs()}
    return [get_crawler(source_name, configs_by_name.get(source_name)) for source_name in source_names]


def _default_source_configs() -> list[object]:
    from app.config import load_settings
    from app.ingestion.services.source_management_service import SourceManagementService

    settings = load_settings()
    return SourceManagementService(settings.crawl).get_enabled_sources()


def _default_source_config(source_name: str) -> object:
    from app.config import load_settings
    from app.ingestion.services.source_management_service import SourceManagementService

    settings = load_settings()
    return SourceManagementService(settings.crawl).get_source_config(source_name)

