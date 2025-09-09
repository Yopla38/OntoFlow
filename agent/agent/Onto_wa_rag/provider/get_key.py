"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

import os
from pathlib import Path


def get_openai_key(api_key_path: Path | str) -> str:
    api_key_path = os.path.join(api_key_path, "openAI_key.txt")
    # Lire la clé d'API depuis le fichier
    with open(api_key_path, "r") as f:
        api_key = f.read().strip()
    return api_key


def get_anthropic_key(api_key_path: Path | str) -> str:
    api_key_path = os.path.join(api_key_path, "anthropicAI_key.txt")
    # Lire la clé d'API depuis le fichier
    with open(api_key_path, "r") as f:
        api_key = f.read().strip()
    return api_key