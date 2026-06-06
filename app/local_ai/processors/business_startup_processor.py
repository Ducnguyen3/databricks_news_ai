from __future__ import annotations

from app.local_ai.processors.default_processor import DefaultNewsProcessor


class BusinessStartupProcessor(DefaultNewsProcessor):
    topic_id = "business_startup"
    domain_instructions = (
        "Tap trung vao cong ty, lanh dao, goi von, ket qua kinh doanh, M&A va chien luoc. "
        "Neu thieu so lieu tai chinh thi khong tu bia. "
        "Phan biet thong tin da cong bo voi nhan dinh trong retrieved context."
    )
