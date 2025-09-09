"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

from __future__ import annotations
import logging
from typing import Any, List, Type, TypeVar, Callable
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# CORRIGÉ : Utilisation de __name__
logger = logging.getLogger(__name__)

T = TypeVar("T")

class LLMError(RuntimeError):
    pass

class LLMProvider:
    """Abstract provider interface expecting structured outputs via Pydantic models."""
    def create_message(self, messages: List[str], pydantic_model: Type[T], **gen_kwargs: Any) -> T:  # pragma: no cover
        raise NotImplementedError

class PydanticLLMProvider(LLMProvider):
    """Wrap a lower-level client that returns structured pydantic via its own API.

    Provide a callable backend(messages, pydantic_model, **kwargs) -> pydantic_model.
    """
    # CORRIGÉ : Ajout de __init__
    def __init__(self, backend: Callable):
        self.backend = backend

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        retry=retry_if_exception_type(Exception),
    )
    def create_message(self, messages: List[str], pydantic_model: Type[T], **gen_kwargs: Any) -> T:
        try:
            # CORRIGÉ : appel avec les bons arguments
            return self.backend(messages, pydantic_model, **gen_kwargs)
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            raise LLMError(str(e))

