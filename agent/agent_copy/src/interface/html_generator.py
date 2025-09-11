"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# src/interface/html_generator.py
from typing import Any, Optional


def has_editable_field(data: Any) -> bool:
    if isinstance(data, dict):
        if data.get('editable'):
            return True
        for value in data.values():
            if has_editable_field(value):
                return True
    elif isinstance(data, list):
        for item in data:
            if has_editable_field(item):
                return True
    return False


