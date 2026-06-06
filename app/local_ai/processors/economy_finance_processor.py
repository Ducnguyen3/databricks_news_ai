from __future__ import annotations

from app.local_ai.processors.default_processor import DefaultNewsProcessor


class EconomyFinanceProcessor(DefaultNewsProcessor):
    topic_id = "economy_finance_stock"
    domain_instructions = (
        "Tap trung vao thi truong, lai suat, co phieu, chi so, dong tien va rui ro. "
        "Khong bia du lieu gia realtime neu chua co stock API. "
        "Neu khong co du lieu gia realtime, noi ro rang chi phan tich theo tin tuc da crawl. "
        "Khong dua khuyen nghi mua/ban chac chan."
    )
