"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# anthropic_setup.py
import os.path


def get_anthropic_key():
    return os.getenv('ANTHROPIC_SMALL_API_KEY', "")
