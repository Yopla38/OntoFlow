"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# test_agent_fortran.py
import argparse
import asyncio
import os
import sys
from pathlib import Path


def lire_fichier_texte(chemin_fichier):
    """
    Lit un fichier texte et retourne son contenu

    Args:
        chemin_fichier (str): Le chemin vers le fichier à lire

    Returns:
        str: Le contenu du fichier
    """
    try:
        with open(chemin_fichier, 'r', encoding='utf-8') as fichier:
            contenu = fichier.read()
        return contenu
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{chemin_fichier}' n'a pas été trouvé.")
        return None
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return None


async def main():
    from agent.Agent_fortran import Deployer_agent_fortran

    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Déployer l'agent d'amélioration")
    parser.add_argument(
        "--program-path",
        required=True,
        help="Chemin du programme à améliorer"
    )
    parser.add_argument(
        "--idea",
        required=True,
        help="Idée d'amélioration à implémenter contenu dans un fichier texte"
    )

    # Parse des arguments
    args = parser.parse_args()

    # Chemin absolu vers le répertoire où se trouve ce script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Ajouter le répertoire parent au chemin Python
    sys.path.insert(0, script_dir)

    # Maintenant importer le module
    try:

        # Utiliser les arguments passés en ligne de commande
        program_path = args.program_path
        idea = args.idea

        idea = lire_fichier_texte(idea)
        print(f"Déploiement de l'agent pour {program_path}")
        print(f"Idée: {idea}")

        # Exécuter l'agent
        agent = await Deployer_agent_fortran(program_path, idea)

        print("Agent exécuté avec succès")

    except Exception as e:
        import traceback
        print(f"Erreur: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
