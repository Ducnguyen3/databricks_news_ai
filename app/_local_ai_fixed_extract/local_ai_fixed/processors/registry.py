from __future__ import annotations

from app.local_ai.processors.base_processor import BaseNewsProcessor
from app.local_ai.processors.business_startup_processor import BusinessStartupProcessor
from app.local_ai.processors.default_processor import DefaultNewsProcessor
from app.local_ai.processors.economy_finance_processor import EconomyFinanceProcessor
from app.local_ai.processors.lifestyle_processor import LifestyleProcessor
from app.local_ai.processors.politics_society_processor import PoliticsSocietyProcessor
from app.local_ai.processors.real_estate_processor import RealEstateProcessor
from app.local_ai.processors.tech_ai_processor import TechAIProcessor
from app.local_ai.processors.world_geopolitics_processor import WorldGeopoliticsProcessor

PROCESSOR_REGISTRY: dict[str, type[BaseNewsProcessor]] = {
    "tech_ai_internet": TechAIProcessor,
    "economy_finance_stock": EconomyFinanceProcessor,
    "politics_society": PoliticsSocietyProcessor,
    "world_geopolitics": WorldGeopoliticsProcessor,
    "business_startup": BusinessStartupProcessor,
    "real_estate": RealEstateProcessor,
    "lifestyle_education_health_entertainment": LifestyleProcessor,
}


def get_processor(topic_id: str | None) -> BaseNewsProcessor:
    processor_class = PROCESSOR_REGISTRY.get(str(topic_id or ""))
    if processor_class is None:
        return DefaultNewsProcessor()
    return processor_class()
