"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

from .resolver import PronounResolver, ResolutionConfig
from .llm import LLMProvider, PydanticLLMProvider
from .models import Replacement, ResolutionOutput

all = [
    "PronounResolver",
    "ResolutionConfig",
    "LLMProvider",
    "PydanticLLMProvider",
    "Replacement",
    "ResolutionOutput",
]