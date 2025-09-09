"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

import asyncio
from typing import Type, List, Any, TypeVar, Optional

from pydantic import BaseModel

from document_analysis.src.pronoun_resolver.llm import LLMError
from provider.llm_providers import LLMProvider, OpenAIProvider

T = TypeVar("T", bound=BaseModel)

class LLMAdapter:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    # CORRECTION : Ajout de Optional[Type[T]] pour accepter `None`
    def __call__(self, messages: List[str], pydantic_model: Optional[Type[T]], **gen_kwargs: Any) -> Any:
        prompt = messages[0]
        formatted_messages = [{"role": "user", "content": prompt}]

        async def run_async():
            # CORRECTION : On gère séparément le cas Pydantic et le cas texte simple
            if pydantic_model is not None:
                # On attend un modèle Pydantic
                response = await self.provider.generate_response(
                    messages=formatted_messages,
                    pydantic_model=pydantic_model,
                    **gen_kwargs
                )
                if response is None:
                    raise LLMError(f"{type(self.provider).__name__} returned None")
                if isinstance(response, pydantic_model):
                    return response
                return pydantic_model.model_validate(response)
            else:
                # On attend une chaîne de caractères
                response_str = await self.provider.generate_response(
                    messages=formatted_messages,
                    **gen_kwargs
                )
                return response_str

        if self.loop.is_running():
            future = asyncio.run_coroutine_threadsafe(run_async(), self.loop)
            return future.result()
        else:
            return self.loop.run_until_complete(run_async())