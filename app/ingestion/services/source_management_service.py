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
    max_concurrent_requests: int
    timeout_seconds: int
    retry_count: int
    user_agent: str
    crawl_homepage: bool = True
    structured_paths: tuple[str, ...] = ()
    special_paths: tuple[str, ...] = ()


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
            structured_paths: tuple[str, ...] = (),
            special_paths: tuple[str, ...] = (),
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
                max_concurrent_requests=settings.max_concurrent_requests,
                timeout_seconds=settings.request_timeout_seconds,
                retry_count=settings.retry_count,
                user_agent=settings.user_agent,
                structured_paths=structured_paths,
                special_paths=special_paths,
            )

        return {
            "vnexpress": config(
                source_name="vnexpress",
                base_url="https://vnexpress.net",
                homepage_url="https://vnexpress.net",
                rss_url="https://vnexpress.net/rss/tin-moi-nhat.rss",
                category_paths=(
                    "thoi-su",
                    "thoi-su/chinh-tri",
                    "thoi-su/huong-toi-ky-nguyen-moi",
                    "thoi-su/dan-sinh",
                    "thoi-su/lao-dong-viec-lam",
                    "thoi-su/giao-thong",
                    "thoi-su/quy-hy-vong",
                    "the-gioi",
                    "the-gioi/phan-tich",
                    "the-gioi/tu-lieu",
                    "the-gioi/quan-su",
                    "the-gioi/cuoc-song-do-day",
                    "the-gioi/nguoi-viet-5-chau",
                    "the-gioi/bac-my",
                    "kinh-doanh",
                    "kinh-doanh/net-zero",
                    "kinh-doanh/quoc-te",
                    "kinh-doanh/doanh-nghiep",
                    "kinh-doanh/chung-khoan",
                    "kinh-doanh/ebank",
                    "kinh-doanh/vi-mo",
                    "kinh-doanh/tien-cua-toi",
                    "kinh-doanh/hang-hoa",
                    "kinh-doanh/kinh-te-vung",
                    "kinh-doanh/doanh-nghiep-vuon-minh",
                    "khoa-hoc-cong-nghe",
                    "khoa-hoc-cong-nghe/bo-khoa-hoc-va-cong-nghe",
                    "khoa-hoc-cong-nghe/chuyen-doi-so",
                    "khoa-hoc-cong-nghe/doi-moi-sang-tao",
                    "khoa-hoc-cong-nghe/ai",
                    "khoa-hoc-cong-nghe/vu-tru",
                    "khoa-hoc-cong-nghe/the-gioi-tu-nhien",
                    "khoa-hoc-cong-nghe/thiet-bi",
                    "khoa-hoc-cong-nghe/cua-so-tri-thuc",
                    "khoa-hoc-cong-nghe/cuoc-thi-sang-kien-khoa-hoc",
                    "goc-nhin",
                    "goc-nhin/chinh-tri-chinh-sach",
                    "goc-nhin/y-te-suc-khoe",
                    "goc-nhin/kinh-doanh-quan-tri",
                    "goc-nhin/giao-duc-tri-thuc",
                    "goc-nhin/moi-truong",
                    "goc-nhin/van-hoa-loi-song",
                    "goc-nhin/tac-gia",
                    "bat-dong-san",
                    "bat-dong-san/chinh-sach",
                    "bat-dong-san/thi-truong",
                    "bat-dong-san/du-an",
                    "bat-dong-san/khong-gian-song",
                    "bat-dong-san/tu-van",
                    "suc-khoe",
                    "suc-khoe/tin-tuc",
                    "suc-khoe/cac-benh",
                    "suc-khoe/song-khoe",
                    "suc-khoe/vaccine",
                    "giai-tri",
                    "giai-tri/gioi-sao",
                    "giai-tri/sach",
                    "giai-tri/phim",
                    "giai-tri/nhac",
                    "giai-tri/thoi-trang",
                    "giai-tri/lam-dep",
                    "giai-tri/san-khau-my-thuat",
                    "the-thao",
                    "the-thao/world-cup-2026",
                    "bong-da",
                    "the-thao/marathon",
                    "the-thao/tennis",
                    "the-thao/cac-mon-khac",
                    "the-thao/hau-truong",
                    "phap-luat",
                    "phap-luat/ho-so-pha-an",
                    "phap-luat/tu-van",
                    "giao-duc",
                    "giao-duc/tin-tuc",
                    "giao-duc/tuyen-sinh",
                    "giao-duc/chan-dung",
                    "giao-duc/du-hoc",
                    "giao-duc/thao-luan",
                    "giao-duc/hoc-tieng-anh",
                    "giao-duc/giao-duc-40",
                    "doi-song",
                    "doi-song/nhip-song",
                    "doi-song/to-am",
                    "doi-song/bai-hoc-song",
                    "doi-song/cooking",
                    "doi-song/tieu-dung",
                    "oto-xe-may",
                    "oto-xe-may/thi-truong",
                    "oto-xe-may/xe-dien",
                    "oto-xe-may/dien-dan",
                    "oto-xe-may/v-car",
                    "oto-xe-may/v-bike",
                    "oto-xe-may/cam-lai",
                    "du-lich",
                    "du-lich/diem-den",
                    "du-lich/am-thuc",
                    "du-lich/dau-chan",
                    "du-lich/tu-van",
                    "du-lich/cam-nang",
                    "y-kien",
                    "y-kien/thoi-su",
                    "y-kien/doi-song",
                    "tam-su",
                    "tam-su/hen-ho",
                    "thu-gian",
                    "thu-gian/cuoi",
                    "thu-gian/do-vui",
                    "thu-gian/chuyen-la",
                    "thu-gian/crossword",
                    "thu-gian/thu-cung",
                    "thu-gian/tro-choi",
                ),
                structured_paths=("the-thao/du-lieu-bong-da",),
            ),
            "cafef": config(
                source_name="cafef",
                base_url="https://cafef.vn",
                rss_url="https://cafef.vn/thi-truong-chung-khoan.rss",
                category_paths=(
                    "xa-hoi.chn",
                    "thi-truong-chung-khoan.chn",
                    "bat-dong-san.chn",
                    "doanh-nghiep.chn",
                    "tai-chinh-ngan-hang.chn",
                    "smart-money.chn",
                    "tai-chinh-quoc-te.chn",
                    "vi-mo-dau-tu.chn",
                    "kinh-te-so.chn",
                    "thi-truong.chn",
                    "song.chn",
                    "lifestyle.chn",
                ),
            ),
            "genk": config(
                source_name="genk",
                base_url="https://genk.vn",
                rss_url="https://genk.vn/rss/home.rss",
                category_paths=(
                    "mobile.chn",
                    "ai.chn",
                    "tin-ict.chn",
                    "internet.chn",
                    "kham-pha.chn",
                    "xem-mua-luon.chn",
                    "xe.chn",
                    "apps-games.chn",
                    "do-choi-so.chn",
                    "mobile/dien-thoai.chn",
                    "mobile/may-tinh-bang.chn",
                    "internet/digital-marketing.chn",
                    "internet/media.chn",
                    "kham-pha/lich-su.chn",
                    "kham-pha/tri-thuc.chn",
                    "tra-da-cong-nghe.chn",
                    "tra-da-cong-nghe/tan-man.chn",
                    "tra-da-cong-nghe/y-tuong-sang-tao.chn",
                    "blockchain.chn",
                    "blockchain/xu-huong.chn",
                    "blockchain/cong-nghe.chn",
                    "blockchain/nhan-vat.chn",
                    "thu-thuat.chn",
                    "song.chn",
                    "nhom-chu-de/emagazine.chn",
                    "gia-dung.chn",
                ),
            ),
            "diendandoanhnghiep": config(
                source_name="diendandoanhnghiep",
                base_url="https://diendandoanhnghiep.vn",
                category_paths=(
                    "tin-moi-nhat",
                    "chinh-tri-xa-hoi",
                    "chinh-tri-xa-hoi/chinh-tri",
                    "chinh-tri-xa-hoi/tam-diem",
                    "chinh-tri-xa-hoi/mat-tran",
                    "chinh-tri-xa-hoi/kinh-te",
                    "chinh-tri-xa-hoi/xa-hoi",
                    "vcci",
                    "vcci/phat-trien-ben-vung",
                    "vcci/tieng-noi-cua-hiep-hoi-doanh-nghiep",
                    "vcci/doanh-nghiep-hang-dau-viet-nam",
                    "vcci/dai-hoi-vcci-lan-thu-vii",
                    "vcci/xuc-tien-dau-tu-thuong-mai",
                    "vcci/tham-muu-chinh-sach",
                    "doanh-nghiep",
                    "doanh-nghiep/quan-tri",
                    "doanh-nghiep/trach-nhiem-xa-hoi",
                    "doanh-nghiep/chuyen-dong",
                    "doanh-nghiep/giao-thuong",
                    "doanh-nhan",
                    "doanh-nhan/chuyen-lam-an",
                    "doanh-nhan/ca-phe-doanh-nhan",
                    "doanh-nhan/phong-cach-song",
                    "doanh-nhan/suc-khoe",
                    "doanh-nhan/khoa-hoc",
                    "doanh-nhan/phong-thuy",
                    "doanh-nhan/chat-luong-song",
                    "khoi-nghiep",
                    "khoi-nghiep/khoi-nghiep-quoc-gia",
                    "khoi-nghiep/y-tuong-kinh-doanh",
                    "khoi-nghiep/co-van-huan-luyen",
                    "khoi-nghiep/so-tay-khoi-nghiep",
                    "du-lich",
                    "du-lich/trai-nghiem",
                    "du-lich/hoat-dong-du-lich",
                    "du-lich/hoi-nhap",
                    "kinh-te-dia-phuong",
                    "cong-nghe",
                    "cong-nghe/kinh-te-so",
                    "cong-nghe/ung-dung",
                    "cong-nghe/chuyen-doi-so",
                    "o-to-xe-may",
                    "o-to-xe-may/dien-dan",
                    "o-to-xe-may/thong-tin-thi-truong",
                    "o-to-xe-may/san-pham",
                    "o-to-xe-may/tu-van-ky-thuat",
                    "doanh-nghiep-thi-truong",
                    "doanh-nghiep-thi-truong/thong-tin-doanh-nghiep",
                    "doanh-nghiep-thi-truong/san-pham-thi-truong",
                    "ngan-hang-chung-khoan",
                    "ngan-hang-chung-khoan/chung-khoan",
                    "ngan-hang-chung-khoan/tin-dung-ngan-hang",
                    "ngan-hang-chung-khoan/tai-chinh-doanh-nghiep",
                    "ngan-hang-chung-khoan/thi-truong-vang",
                    "ngan-hang-chung-khoan/dich-vu-tai-chinh",
                    "ngan-hang-chung-khoan/tai-chinh-so",
                    "ngan-hang-chung-khoan/chuyen-de",
                    "bat-dong-san",
                    "bat-dong-san/thi-truong",
                    "bat-dong-san/doanh-nghiep-du-an",
                    "bat-dong-san/chinh-sach-quy-hoach",
                    "bat-dong-san/cafe-dia-oc",
                    "bat-dong-san/tien-do-du-an",
                    "quoc-te",
                    "quoc-te/doi-ngoai",
                    "quoc-te/kinh-te-the-gioi",
                    "quoc-te/phan-tich-binh-luan",
                    "phap-luat",
                    "phap-luat/nghien-cuu-trao-doi",
                    "phap-luat/ban-doc",
                    "phap-luat/kien-nghi",
                    "phap-luat/24h",
                    "phap-luat/nhin-thang-noi-that",
                    "phap-luat/chong-hang-gia",
                    "phap-luat/ho-so",
                    "phap-luat/phap-dinh",
                ),
            ),
        }


def with_limits(config: SourceConfig, max_pages_per_category: int | None = None) -> SourceConfig:
    if max_pages_per_category is None:
        return config
    return replace(config, max_pages_per_category=max_pages_per_category)
