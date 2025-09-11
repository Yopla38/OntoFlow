"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

import logging
import os

from CONSTANT import API_KEY_PATH, LLM_MODEL
from document_analysis.src.pronoun_resolver import PydanticLLMProvider, ResolutionConfig, PronounResolver

# Configuration du logging pour voir ce qu'il se passe
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from provider.llm_providers import OpenAIProvider
from provider.get_key import get_openai_key # Assurez-vous que ce chemin est correct

# Imports depuis le module d'analyse de document
from .llm_adapter import LLMAdapter


def run_pronoun_resolution_tests():
    """
    Fonction principale qui orchestre la résolution de pronoms sur une suite de tests.
    """
    # 1. Instanciation du Provider LLM
    try:
        api_key = os.environ.get("OPENAI_API_KEY") or get_openai_key(API_KEY_PATH)
        if not api_key:
            raise ValueError("Clé API OpenAI non trouvée.")
    except Exception as e:
        logging.error(f"Erreur lors de la récupération de la clé API : {e}")
        return

    openai_provider = OpenAIProvider(model=LLM_MODEL, api_key=api_key)
    adapter = LLMAdapter(provider=openai_provider)
    llm_wrapper = PydanticLLMProvider(backend=adapter)
    config = ResolutionConfig(lang="fr", context_window=1000)
    resolver = PronounResolver(llm_provider=llm_wrapper, config=config)

    # 2. Définition de la suite de tests
    test_cases = [
        {
            "name": "CAS 1: Le bug original (Sujet + Objet Indirect au Passé Composé)",
            "input": "Marie a parlé à Jean de son projet. Il est enthousiaste. Elle lui a demandé son avis.",
            "expected": "Marie a parlé à Jean de son projet. Jean est enthousiaste. Marie a demandé à Jean son avis."
        },
        {
            "name": "CAS 2: Objet Indirect au Présent (pas d'auxiliaire)",
            "input": "Le professeur voit les étudiants. Il leur parle calmement.",
            "expected": "Le professeur voit les étudiants. Le professeur parle calmement à les étudiants."
        },
        {
            "name": "CAS 3: Uniquement des Sujets",
            "input": "Le chat dort sur le tapis. Il est gris. La souris le regarde. Elle est petite.",
            "expected": "Le chat dort sur le tapis. Le chat est gris. La souris le regarde. La souris est petite."
        },
        {
            "name": "CAS 4: Pluriel et Passé Composé",
            "input": "Les directeurs ont vu les employés. Ils leur ont donné de nouvelles instructions.",
            "expected": "Les directeurs ont vu les employés. Les directeurs ont donné de nouvelles instructions à les employés."
        }
    ]

    # 3. Exécution des tests
    all_passed = True
    for i, case in enumerate(test_cases):
        print(f"\n--- EXÉCUTION DU TEST {i + 1}: {case['name']} ---")
        logging.info(f"Texte original :\n{case['input']}")

        resolved_text = resolver.resolve(case['input'])

        logging.info(f"Texte résolu :\n{resolved_text}")
        logging.info(f"Texte attendu:\n{case['expected']}")

        if resolved_text == case['expected']:
            print(f"✅ RÉSULTAT: SUCCÈS")
        else:
            print(f"❌ RÉSULTAT: ÉCHEC")
            all_passed = False

    print("\n--- RÉSUMÉ DES TESTS ---")
    if all_passed:
        print("✅ Tous les tests ont réussi !")
    else:
        print("❌ Au moins un test a échoué.")


if __name__ == "__main__":
    # TODO python -m spacy download fr_core_news_md
    run_pronoun_resolution_tests()
