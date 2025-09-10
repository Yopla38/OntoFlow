"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

from utils.anthropic_setup import get_anthropic_key
from utils.openai_setup import get_openai_key

CLAUDE_KEY = get_anthropic_key()
OPENAI_KEY = get_openai_key()
LOCAL_MODEL_PATH = "/home/yopla/Documents/llm_models/python/models/txt2txt/DeepSeek-R1-Distill-Qwen-14B-Q5_K_S/DeepSeek-R1-Distill-Qwen-14B-Q5_K_S.gguf"
