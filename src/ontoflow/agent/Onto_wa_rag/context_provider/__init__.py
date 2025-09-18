"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# context_providers/__init__.py
from .entity_index import EntityIndex
from .local_context_provider import LocalContextProvider
from .global_context_provider import GlobalContextProvider
from .semantic_context_provider import SemanticContextProvider
from .smart_context_provider import SmartContextProvider
from .contextual_text_generator import ContextualTextGenerator

__all__ = [
    'EntityIndex',
    'LocalContextProvider',
    'GlobalContextProvider',
    'SemanticContextProvider',
    'SmartContextProvider',
    'ContextualTextGenerator'
]
