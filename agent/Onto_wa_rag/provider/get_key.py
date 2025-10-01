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
    api_key_path = os.path.join(os.path.expanduser(api_key_path), "openAI_key.txt")
    print(f"Collecting openAI key from '{api_key_path}'...", end=" ")
    # Lire la clé d'API depuis le fichier
    api_key = ""
    if os.path.exists(api_key_path):
        print("File exists, reading.")
        with open(api_key_path, "r") as f:
            api_key = f.read().strip()
    else:
        print("File not found, defaulting to empty key.")
    return api_key


def get_anthropic_key(api_key_path: Path | str) -> str:
    api_key_path = os.path.join(os.path.expanduser(api_key_path), "anthropicAI_key.txt")
    print(f"Collecting Anthropic key from '{api_key_path}'...", end=" ")
    # Lire la clé d'API depuis le fichier
    api_key = ""
    if os.path.exists(api_key_path):
        print("File exists, reading.")
        with open(api_key_path, "r") as f:
            api_key = f.read().strip()
    else:
        print("File not found, defaulting to empty key.")
    return api_key
