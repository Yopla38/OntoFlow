"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# src/factory/provider_factory.py
from typing import Dict, Any
from agent.src.types.interfaces import LLMProvider, MemoryProvider
from agent.src.providers.llm_providers import OpenAIProvider, AnthropicProvider, LocalDeepSeek_R1_Provider
from agent.src.providers.memory_providers import MongoDBMemory, PineconeMemory, LocalFileMemory, SQLiteMemory, InMemoryStorage


class LLMProviderFactory:
    _model_cache = {}  # Cache pour le modèle lui-même

    @staticmethod
    def create_provider(provider_type: str, model: str, api_key: str, system_prompt: str = None,
                        structured_response_for_local_model: str = None, **kwargs) -> LLMProvider:
        """
        Crée une instance de LLMProvider avec gestion du cache pour les modèles locaux.
        Chaque instance a ses propres paramètres system_prompt et structured_response.
        """
        default_kwargs = {
            "anthropic": {"max_tokens": 8000},
        }

        if provider_type in default_kwargs:
            kwargs = {**default_kwargs[provider_type], **kwargs}

        providers = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "local": LocalDeepSeek_R1_Provider,
        }

        if provider_type not in providers:
            raise ValueError(f"Provider type '{provider_type}' not supported")

        # Création d'une nouvelle instance dans tous les cas
        provider_class = providers[provider_type]

        if provider_type == "local":
            cache_key = f"{provider_type}-{model}"

            # Si le modèle est déjà chargé, on le réutilise
            if cache_key in LLMProviderFactory._model_cache:
                cached_model = LLMProviderFactory._model_cache[cache_key]
                # Création d'une nouvelle instance avec le modèle en cache
                provider = provider_class(model=model, api_key=api_key, cached_model=cached_model)
            else:
                # Première création du provider avec chargement du modèle
                provider = provider_class(model=model, api_key=api_key)
                # Stockage du modèle en cache
                LLMProviderFactory._model_cache[cache_key] = provider.get_model()
        else:
            provider = provider_class(model=model, api_key=api_key)

        # Configuration des paramètres spécifiques à l'instance
        if system_prompt and hasattr(provider, 'set_system_prompt'):
            provider.set_system_prompt(system_prompt)

        if structured_response_for_local_model and hasattr(provider, 'set_structured_local_model'):
            provider.set_structured_local_model(structured_response_for_local_model)

        return provider


class MemoryProviderFactory:
    @staticmethod
    def create_provider(provider_type: str, config: Dict[str, Any]) -> MemoryProvider:
        """
        Crée une instance de MemoryProvider en fonction du type spécifié

        Args:
            provider_type: Type de provider (mongodb, pinecone, local, etc.)
            config: Configuration du provider
        """
        providers = {
            "mongodb": MongoDBMemory,
            "pinecone": PineconeMemory,
            "local": LocalFileMemory,
            "sqlite": SQLiteMemory,
            "memory": InMemoryStorage
        }

        if provider_type not in providers:
            raise ValueError(f"Memory provider type '{provider_type}' not supported")

        provider_class = providers[provider_type]
        return provider_class(**config)
