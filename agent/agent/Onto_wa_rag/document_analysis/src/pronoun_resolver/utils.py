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
import re
from typing import Iterable

logger = logging.getLogger(__name__)
APOS = "'’"

_def_ws = re.compile(r"\s+")


def normalize_ws(s: str) -> str:
    return _def_ws.sub(" ", s).strip()


def is_all_upper(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def is_title(s: str) -> bool:
    return s == s.title() and s not in {s.upper(), s.lower()}


def adjust_casing(original: str, replacement: str) -> str:
    if not replacement:
        return replacement
    if is_all_upper(original):
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    if is_title(original):
        return replacement.title()
    return replacement


def safe_join_text(chunks: Iterable[str]) -> str:
    return "".join(chunks)
