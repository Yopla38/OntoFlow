"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# Paramètres communs
API_KEY_PATH = "/home/yopla/Documents/keys/"

# Pour Integration_fortran_RAG
STORAGE_DIR = "/home/yopla/test_agent/onto_RAG"  # Zone de stockage du RAG (chunks, hopfields network, ...)
CHUNK_SIZE = 2000  # Taille de découpage des fichiers ou entités
CHUNK_OVERLAP = 0  # Pour le code: pas de recouvrement des chunks
ONTOLOGY_PATH_TTL = "/home/yopla/PycharmProjects/llm-hackathon-2025/2-aiengine/OntoFlow/agent/Onto_wa_rag/Bibliotheque_d_ontologie/bigdft_ontologie_ipynb.ttl"
MAX_RESULTS = 20  # Nombre de passages retrounés pour une recherche
MAX_CONCURRENT = 5  # Nombre thread utilisés pour l'ajout de document

LANGUAGE = "français"

# ---------------- FORTRAN AGENT--------------
FORTRAN_AGENT_NB_STEP = 15

# ----------------- VISION --------------------
VISION_AGENT_MODEL = "claude-sonnet-4-20250514"
VISION_NB_STEP_AGENT = 8

# ---------------- HOPFIELD PARAMETER ---------
BETA = 20.0  # valeur de séparation des motifs appris. Valeur plus élevée pour une classification plus précise
NORMALIZED_PATTERN = True
TOP_K_CONCEPT = 10

# Pour plus tard...
LOCAL_EMBEDDING_PATH = "/home/yopla/Documents/llm_models/python/embedding/SPECTER"
MAX_CONCEPT_TO_DETECT = 8
CONFIANCE = 0.35

# Configuration des couleurs pour le terminal
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

RELATION_MODEL_TYPE = "TransE"
RELATION_CONFIDENCE = 0.45

USE_SEMANTIC_CHUNKING = True  # Activer le chunking sémantique
