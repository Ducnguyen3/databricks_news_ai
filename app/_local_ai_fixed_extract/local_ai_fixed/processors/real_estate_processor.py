from __future__ import annotations

from app.local_ai.processors.default_processor import DefaultNewsProcessor


class RealEstateProcessor(DefaultNewsProcessor):
    topic_id = "real_estate"
    domain_instructions = (
        "Tap trung vao dia diem, du an, chinh sach, quy hoach, gia va thi truong. "
        "Khong bia gia nha hoac gia dat neu context khong co. "
        "Neu lien quan quy hoach/phap ly, nhan manh can doi chieu nguon chinh thuc."
    )
