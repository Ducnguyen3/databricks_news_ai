from app.local_ai.processors.base_processor import BaseNewsProcessor
from app.local_ai.processors.business_startup_processor import BusinessStartupProcessor
from app.local_ai.processors.default_processor import DefaultNewsProcessor
from app.local_ai.processors.economy_finance_processor import EconomyFinanceProcessor
from app.local_ai.processors.lifestyle_processor import LifestyleProcessor
from app.local_ai.processors.politics_society_processor import PoliticsSocietyProcessor
from app.local_ai.processors.real_estate_processor import RealEstateProcessor
from app.local_ai.processors.registry import PROCESSOR_REGISTRY, get_processor
from app.local_ai.processors.tech_ai_processor import TechAIProcessor
from app.local_ai.processors.world_geopolitics_processor import WorldGeopoliticsProcessor

__all__ = [
    "BaseNewsProcessor",
    "BusinessStartupProcessor",
    "DefaultNewsProcessor",
    "EconomyFinanceProcessor",
    "LifestyleProcessor",
    "PoliticsSocietyProcessor",
    "PROCESSOR_REGISTRY",
    "RealEstateProcessor",
    "TechAIProcessor",
    "WorldGeopoliticsProcessor",
    "get_processor",
]
