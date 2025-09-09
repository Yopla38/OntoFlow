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
from typing import Optional, Set, List


def generate_tree_llm(path: str = '', included_dirs: Optional[List[str]] = None,
                  excluded_dirs: Optional[Set[str]] = None) -> str:
    """
    Génère l'arborescence des dossiers et fichiers pour un chemin donné.

    Arguments:
    path : str : Le chemin du répertoire racine.
    included_dirs : list : Liste des noms de dossiers à inclure dans l'arborescence.
    excluded_dirs : set : Ensemble des noms de dossiers ou fichiers à exclure.

    Retourne:
    str : Une chaîne de caractères représentant l'arborescence des dossiers et fichiers.
    """

    if included_dirs is None:
        included_dirs = os.listdir(path)
    if excluded_dirs is None:
        excluded_dirs = set()

    tree_str = ""

    def should_include(entry_path: str, entry: str) -> bool:
        """Détermine si une entrée doit être incluse dans l'arborescence"""
        # Vérifier si le chemin relatif est dans les dossiers exclus
        relative_path = os.path.relpath(entry_path, path)

        # Si nous sommes à la racine, vérifier si le dossier est dans included_dirs
        if os.path.dirname(relative_path) == '':
            return entry in included_dirs and entry not in excluded_dirs

        # Pour les sous-dossiers, vérifier uniquement les exclusions
        return (entry not in excluded_dirs and
                not entry.startswith('.') and
                not any(part in excluded_dirs for part in relative_path.split(os.sep)))

    def generate_sub_tree(current_path: str, prefix: str):
        nonlocal tree_str
        try:
            entries = sorted(os.listdir(current_path))
        except PermissionError:
            return

        if entries:
            # Filtrer les entrées
            filtered_entries = [
                entry for entry in entries
                if should_include(os.path.join(current_path, entry), entry)
            ]

            for index, entry in enumerate(filtered_entries):
                entry_path = os.path.join(current_path, entry)
                connector = "└──" if index == len(filtered_entries) - 1 else "├──"
                tree_str += f"{prefix}{connector} {entry}\n"

                if os.path.isdir(entry_path):
                    extension = "    " if index == len(filtered_entries) - 1 else "│   "
                    generate_sub_tree(entry_path, prefix + extension)

    generate_sub_tree(path, "")
    return tree_str



