from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)

NEWS_TOPICS: dict[str, dict[str, object]] = {
    "tech_ai_internet": {
        "name": "Cong nghe - AI - Internet",
        "keywords": [
            "ai",
            "tri tue nhan tao",
            "cong nghe",
            "internet",
            "chuyen doi so",
            "chip",
            "cloud",
            "an ninh mang",
            "phan mem",
            "robot",
            "startup cong nghe",
        ],
    },
    "economy_finance_stock": {
        "name": "Kinh te - Tai chinh - Chung khoan",
        "keywords": [
            "kinh te",
            "tai chinh",
            "chung khoan",
            "vn-index",
            "co phieu",
            "ngan hang",
            "lai suat",
            "ty gia",
            "trai phieu",
            "thanh khoan",
            "khoi ngoai",
            "tu doanh",
            "vi mo",
            "cpi",
            "gdp",
            "lam phat",
        ],
    },
    "politics_society": {
        "name": "Thoi su - Chinh tri - Xa hoi",
        "keywords": [
            "thoi su",
            "chinh tri",
            "xa hoi",
            "chinh sach",
            "quoc hoi",
            "chinh phu",
            "phap luat",
            "dan sinh",
            "giao thong",
            "viec lam",
            "an sinh",
        ],
    },
    "world_geopolitics": {
        "name": "Quoc te - Dia chinh tri - The gioi",
        "keywords": [
            "the gioi",
            "quoc te",
            "dia chinh tri",
            "my",
            "trung quoc",
            "nga",
            "ukraine",
            "israel",
            "trung dong",
            "chien tranh",
            "xung dot",
            "ngoai giao",
            "nato",
            "lien hop quoc",
        ],
    },
    "business_startup": {
        "name": "Doanh nghiep - Khoi nghiep",
        "keywords": [
            "doanh nghiep",
            "startup",
            "khoi nghiep",
            "ceo",
            "loi nhuan",
            "doanh thu",
            "m&a",
            "goi von",
            "dau tu",
            "ket qua kinh doanh",
            "chien luoc",
        ],
    },
    "real_estate": {
        "name": "Bat dong san",
        "keywords": [
            "bat dong san",
            "nha dat",
            "chung cu",
            "du an",
            "quy hoach",
            "dat nen",
            "khu do thi",
            "nha o",
            "mat bang",
            "van phong",
            "khu cong nghiep",
        ],
    },
    "lifestyle_education_health_entertainment": {
        "name": "Doi song - Giao duc - Suc khoe - Giai tri",
        "keywords": [
            "doi song",
            "giao duc",
            "suc khoe",
            "giai tri",
            "du lich",
            "the thao",
            "phim",
            "am nhac",
            "truong hoc",
            "tuyen sinh",
            "y te",
            "benh",
            "lifestyle",
            "am thuc",
        ],
    },
}

GENERAL_NEWS_TOPIC = {
    "key": "general_news",
    "name": "Tin tong hop",
}

GLOBAL_CATEGORY_ALIASES: dict[str, str] = {
    "chung khoan": "economy_finance_stock",
    "ngan hang": "economy_finance_stock",
    "tai chinh": "economy_finance_stock",
    "kinh te": "economy_finance_stock",
    "ai": "tech_ai_internet",
    "cong nghe": "tech_ai_internet",
    "internet": "tech_ai_internet",
    "mobile": "tech_ai_internet",
    "blockchain": "tech_ai_internet",
    "thoi su": "politics_society",
    "chinh tri": "politics_society",
    "xa hoi": "politics_society",
    "phap luat": "politics_society",
    "the gioi": "world_geopolitics",
    "quoc te": "world_geopolitics",
    "doanh nghiep": "business_startup",
    "khoi nghiep": "business_startup",
    "doanh nhan": "business_startup",
    "bat dong san": "real_estate",
    "nha dat": "real_estate",
    "giao duc": "lifestyle_education_health_entertainment",
    "suc khoe": "lifestyle_education_health_entertainment",
    "giai tri": "lifestyle_education_health_entertainment",
    "du lich": "lifestyle_education_health_entertainment",
    "the thao": "lifestyle_education_health_entertainment",
    "doi song": "lifestyle_education_health_entertainment",
}

SOURCE_CATEGORY_ALIASES: dict[str, dict[str, str]] = {
    "cafef": {
        "thi-truong-chung-khoan": "economy_finance_stock",
        "thi truong chung khoan": "economy_finance_stock",
        "thi-truong-chung-khoan.chn": "economy_finance_stock",
        "tai-chinh-ngan-hang": "economy_finance_stock",
        "tai chinh ngan hang": "economy_finance_stock",
        "tai-chinh-ngan-hang.chn": "economy_finance_stock",
        "smart-money": "economy_finance_stock",
        "smart money": "economy_finance_stock",
        "smart-money.chn": "economy_finance_stock",
        "vi-mo-dau-tu": "economy_finance_stock",
        "vi mo dau tu": "economy_finance_stock",
        "vi-mo-dau-tu.chn": "economy_finance_stock",
        "thi-truong": "economy_finance_stock",
        "thi truong": "economy_finance_stock",
        "thi-truong.chn": "economy_finance_stock",
        "tai-chinh-quoc-te": "economy_finance_stock",
        "tai chinh quoc te": "economy_finance_stock",
        "tai-chinh-quoc-te.chn": "economy_finance_stock",
        "doanh-nghiep": "business_startup",
        "doanh nghiep": "business_startup",
        "doanh-nghiep.chn": "business_startup",
        "bat-dong-san": "real_estate",
        "bat dong san": "real_estate",
        "bat-dong-san.chn": "real_estate",
        "kinh-te-so": "tech_ai_internet",
        "kinh te so": "tech_ai_internet",
        "kinh-te-so.chn": "tech_ai_internet",
        "xa-hoi": "politics_society",
        "xa hoi": "politics_society",
        "xa-hoi.chn": "politics_society",
        "song": "lifestyle_education_health_entertainment",
        "song.chn": "lifestyle_education_health_entertainment",
        "lifestyle": "lifestyle_education_health_entertainment",
        "lifestyle.chn": "lifestyle_education_health_entertainment",
    },
    "vnexpress": {
        "thoi-su": "politics_society",
        "thoi-su/chinh-tri": "politics_society",
        "thoi-su/huong-toi-ky-nguyen-moi": "politics_society",
        "thoi-su/dan-sinh": "politics_society",
        "thoi-su/lao-dong-viec-lam": "politics_society",
        "thoi-su/giao-thong": "politics_society",
        "thoi-su/quy-hy-vong": "politics_society",
        "the-gioi": "world_geopolitics",
        "the-gioi/phan-tich": "world_geopolitics",
        "the-gioi/tu-lieu": "world_geopolitics",
        "the-gioi/quan-su": "world_geopolitics",
        "the-gioi/cuoc-song-do-day": "world_geopolitics",
        "the-gioi/nguoi-viet-5-chau": "world_geopolitics",
        "the-gioi/bac-my": "world_geopolitics",
        "kinh-doanh/doanh-nghiep": "business_startup",
        "kinh-doanh/doanh-nghiep-vuon-minh": "business_startup",
        "kinh-doanh/chung-khoan": "economy_finance_stock",
        "kinh-doanh/ebank": "economy_finance_stock",
        "kinh-doanh/vi-mo": "economy_finance_stock",
        "kinh-doanh/tien-cua-toi": "economy_finance_stock",
        "kinh-doanh/hang-hoa": "economy_finance_stock",
        "kinh-doanh/kinh-te-vung": "economy_finance_stock",
        "kinh-doanh/quoc-te": "economy_finance_stock",
        "kinh-doanh/net-zero": "economy_finance_stock",
        "kinh-doanh": "economy_finance_stock",
        "khoa-hoc-cong-nghe/bo-khoa-hoc-va-cong-nghe": "tech_ai_internet",
        "khoa-hoc-cong-nghe/chuyen-doi-so": "tech_ai_internet",
        "khoa-hoc-cong-nghe/doi-moi-sang-tao": "tech_ai_internet",
        "khoa-hoc-cong-nghe/ai": "tech_ai_internet",
        "khoa-hoc-cong-nghe/vu-tru": "tech_ai_internet",
        "khoa-hoc-cong-nghe/the-gioi-tu-nhien": "tech_ai_internet",
        "khoa-hoc-cong-nghe/thiet-bi": "tech_ai_internet",
        "khoa-hoc-cong-nghe/cua-so-tri-thuc": "tech_ai_internet",
        "khoa-hoc-cong-nghe/cuoc-thi-sang-kien-khoa-hoc": "tech_ai_internet",
        "khoa-hoc-cong-nghe": "tech_ai_internet",
        "goc-nhin/chinh-tri-chinh-sach": "politics_society",
        "goc-nhin/y-te-suc-khoe": "lifestyle_education_health_entertainment",
        "goc-nhin/kinh-doanh-quan-tri": "business_startup",
        "goc-nhin/giao-duc-tri-thuc": "lifestyle_education_health_entertainment",
        "goc-nhin/moi-truong": "politics_society",
        "goc-nhin/van-hoa-loi-song": "lifestyle_education_health_entertainment",
        "goc-nhin/tac-gia": "politics_society",
        "goc-nhin": "politics_society",
        "bat-dong-san/chinh-sach": "real_estate",
        "bat-dong-san/thi-truong": "real_estate",
        "bat-dong-san/du-an": "real_estate",
        "bat-dong-san/khong-gian-song": "real_estate",
        "bat-dong-san/tu-van": "real_estate",
        "bat-dong-san": "real_estate",
        "suc-khoe/tin-tuc": "lifestyle_education_health_entertainment",
        "suc-khoe/cac-benh": "lifestyle_education_health_entertainment",
        "suc-khoe/song-khoe": "lifestyle_education_health_entertainment",
        "suc-khoe/vaccine": "lifestyle_education_health_entertainment",
        "suc-khoe": "lifestyle_education_health_entertainment",
        "giai-tri/gioi-sao": "lifestyle_education_health_entertainment",
        "giai-tri/sach": "lifestyle_education_health_entertainment",
        "giai-tri/phim": "lifestyle_education_health_entertainment",
        "giai-tri/nhac": "lifestyle_education_health_entertainment",
        "giai-tri/thoi-trang": "lifestyle_education_health_entertainment",
        "giai-tri/lam-dep": "lifestyle_education_health_entertainment",
        "giai-tri/san-khau-my-thuat": "lifestyle_education_health_entertainment",
        "giai-tri": "lifestyle_education_health_entertainment",
        "the-thao/world-cup-2026": "lifestyle_education_health_entertainment",
        "the-thao/marathon": "lifestyle_education_health_entertainment",
        "the-thao/tennis": "lifestyle_education_health_entertainment",
        "the-thao/cac-mon-khac": "lifestyle_education_health_entertainment",
        "the-thao/hau-truong": "lifestyle_education_health_entertainment",
        "the-thao": "lifestyle_education_health_entertainment",
        "bong-da": "lifestyle_education_health_entertainment",
        "phap-luat/ho-so-pha-an": "politics_society",
        "phap-luat/tu-van": "politics_society",
        "phap-luat": "politics_society",
        "giao-duc/tin-tuc": "lifestyle_education_health_entertainment",
        "giao-duc/tuyen-sinh": "lifestyle_education_health_entertainment",
        "giao-duc/chan-dung": "lifestyle_education_health_entertainment",
        "giao-duc/du-hoc": "lifestyle_education_health_entertainment",
        "giao-duc/thao-luan": "lifestyle_education_health_entertainment",
        "giao-duc/hoc-tieng-anh": "lifestyle_education_health_entertainment",
        "giao-duc/giao-duc-40": "lifestyle_education_health_entertainment",
        "giao-duc": "lifestyle_education_health_entertainment",
        "doi-song/nhip-song": "lifestyle_education_health_entertainment",
        "doi-song/to-am": "lifestyle_education_health_entertainment",
        "doi-song/bai-hoc-song": "lifestyle_education_health_entertainment",
        "doi-song/cooking": "lifestyle_education_health_entertainment",
        "doi-song/tieu-dung": "lifestyle_education_health_entertainment",
        "doi-song": "lifestyle_education_health_entertainment",
        "oto-xe-may/thi-truong": "economy_finance_stock",
        "oto-xe-may/xe-dien": "tech_ai_internet",
        "oto-xe-may/dien-dan": "lifestyle_education_health_entertainment",
        "oto-xe-may/v-car": "lifestyle_education_health_entertainment",
        "oto-xe-may/v-bike": "lifestyle_education_health_entertainment",
        "oto-xe-may/cam-lai": "lifestyle_education_health_entertainment",
        "oto-xe-may": "lifestyle_education_health_entertainment",
        "du-lich/diem-den": "lifestyle_education_health_entertainment",
        "du-lich/am-thuc": "lifestyle_education_health_entertainment",
        "du-lich/dau-chan": "lifestyle_education_health_entertainment",
        "du-lich/tu-van": "lifestyle_education_health_entertainment",
        "du-lich/cam-nang": "lifestyle_education_health_entertainment",
        "du-lich": "lifestyle_education_health_entertainment",
        "y-kien/thoi-su": "politics_society",
        "y-kien/doi-song": "lifestyle_education_health_entertainment",
        "y-kien": "politics_society",
        "tam-su/hen-ho": "lifestyle_education_health_entertainment",
        "tam-su": "lifestyle_education_health_entertainment",
        "thu-gian/cuoi": "lifestyle_education_health_entertainment",
        "thu-gian/do-vui": "lifestyle_education_health_entertainment",
        "thu-gian/chuyen-la": "lifestyle_education_health_entertainment",
        "thu-gian/crossword": "lifestyle_education_health_entertainment",
        "thu-gian/thu-cung": "lifestyle_education_health_entertainment",
        "thu-gian/tro-choi": "lifestyle_education_health_entertainment",
        "thu-gian": "lifestyle_education_health_entertainment",
        "thoi su": "politics_society",
        "the gioi": "world_geopolitics",
        "khoa hoc cong nghe": "tech_ai_internet",
        "kinh doanh": "economy_finance_stock",
    },
    "genk": {
        "ai.chn": "tech_ai_internet",
        "tin-ict.chn": "tech_ai_internet",
        "mobile.chn": "tech_ai_internet",
        "mobile/dien-thoai.chn": "tech_ai_internet",
        "mobile/may-tinh-bang.chn": "tech_ai_internet",
        "internet.chn": "tech_ai_internet",
        "internet/digital-marketing.chn": "tech_ai_internet",
        "internet/media.chn": "tech_ai_internet",
        "kham-pha.chn": "tech_ai_internet",
        "kham-pha/lich-su.chn": "tech_ai_internet",
        "kham-pha/tri-thuc.chn": "tech_ai_internet",
        "tra-da-cong-nghe.chn": "tech_ai_internet",
        "tra-da-cong-nghe/tan-man.chn": "tech_ai_internet",
        "tra-da-cong-nghe/y-tuong-sang-tao.chn": "tech_ai_internet",
        "blockchain.chn": "tech_ai_internet",
        "blockchain/xu-huong.chn": "tech_ai_internet",
        "blockchain/cong-nghe.chn": "tech_ai_internet",
        "blockchain/nhan-vat.chn": "tech_ai_internet",
        "thu-thuat.chn": "tech_ai_internet",
        "apps-games.chn": "tech_ai_internet",
        "do-choi-so.chn": "tech_ai_internet",
        "xem-mua-luon.chn": "tech_ai_internet",
        "gia-dung.chn": "tech_ai_internet",
        "xe.chn": "tech_ai_internet",
        "song.chn": "lifestyle_education_health_entertainment",
        "nhom-chu-de/emagazine.chn": "tech_ai_internet",
        "ai": "tech_ai_internet",
        "mobile": "tech_ai_internet",
        "internet": "tech_ai_internet",
        "kham-pha": "tech_ai_internet",
        "kham pha": "tech_ai_internet",
    },
    "diendandoanhnghiep": {
        "chinh-tri-xa-hoi/kinh-te": "economy_finance_stock",
        "chinh-tri-xa-hoi/chinh-tri": "politics_society",
        "chinh-tri-xa-hoi/tam-diem": "politics_society",
        "chinh-tri-xa-hoi/mat-tran": "politics_society",
        "chinh-tri-xa-hoi/xa-hoi": "politics_society",
        "chinh-tri-xa-hoi": "politics_society",
        "vcci/tham-muu-chinh-sach": "politics_society",
        "vcci/phat-trien-ben-vung": "business_startup",
        "vcci/tieng-noi-cua-hiep-hoi-doanh-nghiep": "business_startup",
        "vcci/doanh-nghiep-hang-dau-viet-nam": "business_startup",
        "vcci/dai-hoi-vcci-lan-thu-vii": "business_startup",
        "vcci/xuc-tien-dau-tu-thuong-mai": "business_startup",
        "vcci": "business_startup",
        "doanh-nghiep/quan-tri": "business_startup",
        "doanh-nghiep/trach-nhiem-xa-hoi": "business_startup",
        "doanh-nghiep/chuyen-dong": "business_startup",
        "doanh-nghiep/giao-thuong": "business_startup",
        "doanh-nghiep": "business_startup",
        "khoi-nghiep/khoi-nghiep-quoc-gia": "business_startup",
        "khoi-nghiep/y-tuong-kinh-doanh": "business_startup",
        "khoi-nghiep/co-van-huan-luyen": "business_startup",
        "khoi-nghiep/so-tay-khoi-nghiep": "business_startup",
        "khoi-nghiep": "business_startup",
        "ngan-hang-chung-khoan/chung-khoan": "economy_finance_stock",
        "ngan-hang-chung-khoan/tin-dung-ngan-hang": "economy_finance_stock",
        "ngan-hang-chung-khoan/tai-chinh-doanh-nghiep": "economy_finance_stock",
        "ngan-hang-chung-khoan/thi-truong-vang": "economy_finance_stock",
        "ngan-hang-chung-khoan/dich-vu-tai-chinh": "economy_finance_stock",
        "ngan-hang-chung-khoan/tai-chinh-so": "economy_finance_stock",
        "ngan-hang-chung-khoan/chuyen-de": "economy_finance_stock",
        "ngan-hang-chung-khoan": "economy_finance_stock",
        "bat-dong-san/thi-truong": "real_estate",
        "bat-dong-san/doanh-nghiep-du-an": "real_estate",
        "bat-dong-san/chinh-sach-quy-hoach": "real_estate",
        "bat-dong-san/cafe-dia-oc": "real_estate",
        "bat-dong-san/tien-do-du-an": "real_estate",
        "bat-dong-san": "real_estate",
        "quoc-te/kinh-te-the-gioi": "world_geopolitics",
        "quoc-te/doi-ngoai": "world_geopolitics",
        "quoc-te/phan-tich-binh-luan": "world_geopolitics",
        "quoc-te": "world_geopolitics",
        "o-to-xe-may/thong-tin-thi-truong": "economy_finance_stock",
        "o-to-xe-may/dien-dan": "lifestyle_education_health_entertainment",
        "o-to-xe-may/san-pham": "lifestyle_education_health_entertainment",
        "o-to-xe-may/tu-van-ky-thuat": "lifestyle_education_health_entertainment",
        "o-to-xe-may": "lifestyle_education_health_entertainment",
        "phap-luat/nghien-cuu-trao-doi": "politics_society",
        "phap-luat/ban-doc": "politics_society",
        "phap-luat/kien-nghi": "politics_society",
        "phap-luat/24h": "politics_society",
        "phap-luat/nhin-thang-noi-that": "politics_society",
        "phap-luat/chong-hang-gia": "politics_society",
        "phap-luat/ho-so": "politics_society",
        "phap-luat/phap-dinh": "politics_society",
        "phap-luat": "politics_society",
        "kinh-te": "economy_finance_stock",
        "kinh te": "economy_finance_stock",
        "tai-chinh": "economy_finance_stock",
        "tai chinh": "economy_finance_stock",
        "bat dong san": "real_estate",
        "khoi nghiep": "business_startup",
    },
}

SOURCE_CATEGORY_PREFIX_TOPIC_MAP: dict[str, dict[str, str]] = {
    "genk": {
        "mobile/": "tech_ai_internet",
        "internet/": "tech_ai_internet",
        "kham-pha/": "tech_ai_internet",
        "tra-da-cong-nghe/": "tech_ai_internet",
        "blockchain/": "tech_ai_internet",
    },
    "diendandoanhnghiep": {
        "chinh-tri-xa-hoi/": "politics_society",
        "vcci/": "business_startup",
        "doanh-nghiep/": "business_startup",
        "khoi-nghiep/": "business_startup",
        "ngan-hang-chung-khoan/": "economy_finance_stock",
        "bat-dong-san/": "real_estate",
        "quoc-te/": "world_geopolitics",
        "o-to-xe-may/": "lifestyle_education_health_entertainment",
        "phap-luat/": "politics_society",
    }
}

SOURCE_CATEGORY_SECONDARY_TOPICS: dict[str, dict[str, list[str]]] = {
    "cafef": {
        "tai-chinh-quoc-te.chn": ["world_geopolitics"],
        "tai-chinh-quoc-te": ["world_geopolitics"],
        "tai chinh quoc te": ["world_geopolitics"],
        "kinh-te-so.chn": ["economy_finance_stock"],
        "kinh-te-so": ["economy_finance_stock"],
        "kinh te so": ["economy_finance_stock"],
        "doanh-nghiep.chn": ["economy_finance_stock"],
        "doanh-nghiep": ["economy_finance_stock"],
        "doanh nghiep": ["economy_finance_stock"],
        "bat-dong-san.chn": ["economy_finance_stock"],
        "bat-dong-san": ["economy_finance_stock"],
        "bat dong san": ["economy_finance_stock"],
    },
    "genk": {
        "ai.chn": ["business_startup"],
        "tin-ict.chn": ["business_startup"],
        "internet/digital-marketing.chn": ["business_startup"],
        "blockchain.chn": ["economy_finance_stock"],
        "blockchain/xu-huong.chn": ["economy_finance_stock"],
        "xem-mua-luon.chn": ["lifestyle_education_health_entertainment"],
        "gia-dung.chn": ["lifestyle_education_health_entertainment"],
        "xe.chn": ["lifestyle_education_health_entertainment", "economy_finance_stock"],
        "song.chn": ["tech_ai_internet"],
    },
    "diendandoanhnghiep": {
        "vcci": ["politics_society", "economy_finance_stock"],
        "vcci/tham-muu-chinh-sach": ["business_startup"],
        "doanh-nghiep": ["economy_finance_stock"],
        "doanh-nghiep/quan-tri": ["economy_finance_stock"],
        "khoi-nghiep": ["tech_ai_internet"],
        "ngan-hang-chung-khoan/tai-chinh-so": ["tech_ai_internet"],
        "bat-dong-san": ["economy_finance_stock"],
        "quoc-te/kinh-te-the-gioi": ["economy_finance_stock"],
        "phap-luat": ["business_startup"],
    },
}

_MIN_TOPIC_SCORE = 3.0
CATEGORY_ALIAS_WEIGHT = 10.0
CATEGORY_PRIMARY_LOCK_WEIGHT = 100.0
_SHORT_KEYWORD_RE_CACHE: dict[str, re.Pattern[str]] = {}

SOURCE_CATEGORY_PRIMARY_TOPIC_LOCKS: dict[str, dict[str, str]] = {
    "vnexpress": {
        "thoi-su": "politics_society",
        "thoi-su/chinh-tri": "politics_society",
        "thoi-su/huong-toi-ky-nguyen-moi": "politics_society",
        "thoi-su/dan-sinh": "politics_society",
        "thoi-su/lao-dong-viec-lam": "politics_society",
        "thoi-su/giao-thong": "politics_society",
        "thoi-su/quy-hy-vong": "politics_society",
        "the-thao": "lifestyle_education_health_entertainment",
        "the-thao/world-cup-2026": "lifestyle_education_health_entertainment",
        "the-thao/marathon": "lifestyle_education_health_entertainment",
        "the-thao/tennis": "lifestyle_education_health_entertainment",
        "the-thao/cac-mon-khac": "lifestyle_education_health_entertainment",
        "the-thao/hau-truong": "lifestyle_education_health_entertainment",
        "bong-da": "lifestyle_education_health_entertainment",
        "khoa-hoc-cong-nghe": "tech_ai_internet",
        "khoa-hoc-cong-nghe/bo-khoa-hoc-va-cong-nghe": "tech_ai_internet",
        "khoa-hoc-cong-nghe/chuyen-doi-so": "tech_ai_internet",
        "khoa-hoc-cong-nghe/doi-moi-sang-tao": "tech_ai_internet",
        "khoa-hoc-cong-nghe/ai": "tech_ai_internet",
        "khoa-hoc-cong-nghe/vu-tru": "tech_ai_internet",
        "khoa-hoc-cong-nghe/the-gioi-tu-nhien": "tech_ai_internet",
        "khoa-hoc-cong-nghe/thiet-bi": "tech_ai_internet",
        "khoa-hoc-cong-nghe/cua-so-tri-thuc": "tech_ai_internet",
        "khoa-hoc-cong-nghe/cuoc-thi-sang-kien-khoa-hoc": "tech_ai_internet",
        "giai-tri": "lifestyle_education_health_entertainment",
        "giai-tri/gioi-sao": "lifestyle_education_health_entertainment",
        "giai-tri/sach": "lifestyle_education_health_entertainment",
        "giai-tri/phim": "lifestyle_education_health_entertainment",
        "giai-tri/nhac": "lifestyle_education_health_entertainment",
        "giai-tri/thoi-trang": "lifestyle_education_health_entertainment",
        "giai-tri/lam-dep": "lifestyle_education_health_entertainment",
        "giai-tri/san-khau-my-thuat": "lifestyle_education_health_entertainment",
        "kinh-doanh/quoc-te": "economy_finance_stock",
    },
    "cafef": {
        "tai-chinh-quoc-te.chn": "economy_finance_stock",
        "thi-truong.chn": "economy_finance_stock",
    },
    "diendandoanhnghiep": {
        "o-to-xe-may/thong-tin-thi-truong": "economy_finance_stock",
        "o-to-xe-may/dien-dan": "lifestyle_education_health_entertainment",
        "o-to-xe-may/san-pham": "lifestyle_education_health_entertainment",
        "o-to-xe-may/tu-van-ky-thuat": "lifestyle_education_health_entertainment",
        "o-to-xe-may": "lifestyle_education_health_entertainment",
    },
}


def normalize_topic(
    source: str,
    source_category: str | None,
    title: str,
    summary: str | None = None,
    content: str | None = None,
) -> dict[str, Any]:
    scores = {topic_key: 0.0 for topic_key in NEWS_TOPICS}
    category_text = normalize_text(source_category)
    title_text = normalize_text(title)
    summary_text = normalize_text(summary)
    content_text = normalize_text(content)

    alias_topic = map_category_to_topic(source, source_category)
    if alias_topic:
        scores[alias_topic] += CATEGORY_ALIAS_WEIGHT

    locked_topic = locked_primary_topic(source, source_category)
    if locked_topic:
        scores[locked_topic] += CATEGORY_PRIMARY_LOCK_WEIGHT

    for topic_key, config in NEWS_TOPICS.items():
        keywords = tuple(str(keyword) for keyword in config.get("keywords", ()))
        scores[topic_key] += score_topic(category_text, keywords) * 5.0
        scores[topic_key] += score_topic(title_text, keywords) * 3.0
        scores[topic_key] += score_topic(summary_text, keywords) * 2.0
        scores[topic_key] += score_topic(content_text, keywords) * 1.0

    best_topic, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score < _MIN_TOPIC_SCORE:
        return topic_result(str(GENERAL_NEWS_TOPIC["key"]), 0.0, [])

    total_score = sum(score for score in scores.values() if score > 0)
    confidence = best_score / total_score if total_score > 0 else 0.0
    secondary_topics = [
        topic_key
        for topic_key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if topic_key != best_topic and score > 0 and (score >= _MIN_TOPIC_SCORE or score >= best_score * 0.5)
    ][:3]
    for secondary_topic in _secondary_topics_for_category(source, source_category):
        if secondary_topic != best_topic and secondary_topic not in secondary_topics:
            secondary_topics.append(secondary_topic)

    logger.info(
        "[TAXONOMY] source=%s source_category=%s primary_topic=%s confidence=%s",
        source,
        source_category,
        best_topic,
        round(max(0.0, min(1.0, confidence)), 4),
    )
    return topic_result(best_topic, confidence, secondary_topics)


def map_category_to_topic(source: str | None, category: str | None) -> str | None:
    category_text = normalize_text(category)
    if not category_text:
        return None
    normalized_source = normalize_text(source).replace(" ", "")
    source_aliases = SOURCE_CATEGORY_ALIASES.get(normalized_source, {})
    for alias, topic_key in source_aliases.items():
        if _keyword_matches(category_text, normalize_text(alias)):
            return topic_key
    prefix_topic = _source_prefix_topic(normalized_source, category_text)
    if prefix_topic:
        return prefix_topic
    for alias, topic_key in GLOBAL_CATEGORY_ALIASES.items():
        if _keyword_matches(category_text, normalize_text(alias)):
            return topic_key
    return None


def locked_primary_topic(source: str | None, category: str | None) -> str | None:
    category_text = normalize_text(category)
    if not category_text:
        return None
    normalized_source = normalize_text(source).replace(" ", "")
    source_locks = SOURCE_CATEGORY_PRIMARY_TOPIC_LOCKS.get(normalized_source, {})
    for alias, topic_key in source_locks.items():
        if _keyword_matches(category_text, normalize_text(alias)):
            return topic_key
    return None


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    stripped = " ".join(str(text).casefold().strip().split())
    return remove_vietnamese_accents(stripped)


def remove_vietnamese_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    without_marks = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return without_marks.replace("đ", "d").replace("Đ", "D")


def score_topic(text: str, keywords: tuple[str, ...]) -> float:
    if not text:
        return 0.0
    score = 0.0
    for keyword in keywords:
        normalized_keyword = normalize_text(keyword)
        if not normalized_keyword:
            continue
        if _keyword_matches(text, normalized_keyword):
            score += 1.0
    return score


def get_topic_name(topic_key: str) -> str:
    if topic_key == GENERAL_NEWS_TOPIC["key"]:
        return str(GENERAL_NEWS_TOPIC["name"])
    topic = NEWS_TOPICS.get(topic_key)
    if topic is None:
        return str(GENERAL_NEWS_TOPIC["name"])
    return str(topic["name"])


def topic_result(primary_topic: str, confidence: float, secondary_topics: list[str]) -> dict[str, Any]:
    return {
        "primary_topic": primary_topic,
        "primary_topic_name": get_topic_name(primary_topic),
        "topic_confidence": round(max(0.0, min(1.0, confidence)), 4),
        "secondary_topics": secondary_topics,
    }


def _secondary_topics_for_category(source: str | None, category: str | None) -> list[str]:
    category_text = normalize_text(category)
    if not category_text:
        return []
    normalized_source = normalize_text(source).replace(" ", "")
    source_rules = SOURCE_CATEGORY_SECONDARY_TOPICS.get(normalized_source, {})
    for alias, topics in source_rules.items():
        if _keyword_matches(category_text, normalize_text(alias)):
            return list(topics)
    return []


def _source_prefix_topic(source: str, category: str) -> str | None:
    for prefix, topic_key in SOURCE_CATEGORY_PREFIX_TOPIC_MAP.get(source, {}).items():
        if category.startswith(normalize_text(prefix)):
            return topic_key
    return None


def _keyword_matches(text: str, keyword: str) -> bool:
    if len(keyword) <= 3:
        pattern = _SHORT_KEYWORD_RE_CACHE.get(keyword)
        if pattern is None:
            pattern = re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)")
            _SHORT_KEYWORD_RE_CACHE[keyword] = pattern
        return bool(pattern.search(text))
    return keyword in text
