from __future__ import annotations

from app.local_ai.processors.default_processor import DefaultNewsProcessor


class PoliticsSocietyProcessor(DefaultNewsProcessor):
    topic_id = "politics_society"
    domain_instructions = (
        "Tap trung vao su kien, chinh sach, tac dong xa hoi va nhom nguoi dan bi anh huong. "
        "Phan biet thong tin chinh thuc va nhan dinh neu retrieved context co du lieu do. "
        "Khong suy dien y dinh hoac tac dong xa hoi neu context khong ho tro."
    )
