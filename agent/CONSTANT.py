"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
"""


NB_ITERATION_IMPROVMENT = 1
MODEL_SMALL = 'claude-3-5-haiku-20241022'
MODEL_MEDIUM = 'claude-3-5-haiku-20241022'
MODEL_HIGH = 'claude-sonnet-4-20250514' #  "claude-sonnet-4-20250514"
MODEL_AGREGATEUR = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKEN = 10000

# FORTRAN
PERFORM_CLEANING = True  # Nettoyage du code fortran (à la première indexation)
