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
from pydantic import BaseModel, Field
from typing import List, Literal

# On utilise Literal pour forcer le LLM à choisir parmi une liste définie.
PronounType = Literal['subject', 'indirect_object']


class Replacement(BaseModel):
    pos: int = Field(..., description="Index de début absolu du pronom à remplacer.")
    length: int = Field(..., description="Longueur EXACTE du pronom original.")
    text: str = Field(..., description="Texte du référent (ex: 'Jean', 'Marie').")
    pronoun_type: PronounType = Field(..., description="Le type grammatical du pronom.")
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class ResolutionOutput(BaseModel):
    replacements: List[Replacement] = []
