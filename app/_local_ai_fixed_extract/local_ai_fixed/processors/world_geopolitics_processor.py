from __future__ import annotations

from app.local_ai.processors.default_processor import DefaultNewsProcessor


class WorldGeopoliticsProcessor(DefaultNewsProcessor):
    topic_id = "world_geopolitics"
    domain_instructions = (
        "Tap trung vao quoc gia, xung dot, ngoai giao, timeline va cac ben lien quan. "
        "Phan biet ro su kien da xac nhan, tuyen bo tu cac ben, va phan tich/nhan dinh. "
        "Khong suy doan qua muc neu nguon chua du."
    )
