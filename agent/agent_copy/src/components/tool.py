"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

from typing import Callable, List, Dict, Any
from dataclasses import dataclass


@dataclass
class Tool:
    name: str
    description: str
    function: Callable
    required_params: List[Dict[str, Any]]
