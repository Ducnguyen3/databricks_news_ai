from __future__ import annotations

from app.local_ai.processors.default_processor import DefaultNewsProcessor


class TechAIProcessor(DefaultNewsProcessor):
    topic_id = "tech_ai_internet"
    domain_instructions = (
        "Tap trung vao cong nghe, cong ty, san pham, tac dong va xu huong. "
        "Neu context co nhieu nguon, so sanh diem giong va khac. "
        "Khong bia thong so ky thuat, model benchmark hoac timeline san pham neu context khong co."
    )
