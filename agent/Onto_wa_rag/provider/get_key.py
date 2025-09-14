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

def _get_key(name, api_key_path: Path | str) -> str:
    api_key_path = os.path.join(api_key_path, name+"_key.txt")
    # Lire la clé d'API depuis le fichier
    api_key = ""
    envvar = name.upper()+'_API_KEY'
    if os.path.exists(api_key_path):
        with open(api_key_path, "r") as f:
            api_key = f.read().strip()
        print(envvar+' read from file')
    else:
        api_key = os.environ.get(envvar, '')
        if len(api_key) > 0:
            print(envvar+' read from environment')
    return api_key

def get_openai_key(api_key_path: Path | str) -> str:
    return _get_key('openAI', api_key_path)


def get_anthropic_key(api_key_path: Path | str) -> str:
    return _get_key('anthropicAI', api_key_path)
