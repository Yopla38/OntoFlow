"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# openai_setup.py
import os.path

import openai as opai
from openai import OpenAI


def get_openai_key():
    return os.getenv('OPENAI_SMALL_API_KEY', "")


openai_error = opai.APIError
openai = OpenAI(api_key=get_openai_key())
