from pathlib import Path
from typing import List, Dict, Any


def scan_directory(
        directory_path: str,
        file_filters: List[str] = None,
        recursive: bool = True,
        exclude_patterns: List[str] = None,
        project_name: str = None,
        version: str = "1.0"
) -> List[Dict[str, Any]]:
    """
    Scanne un répertoire et retourne la liste des fichiers à traiter

    Args:
        directory_path: Chemin du répertoire à scanner
        file_filters: Extensions à inclure (ex: ['f90', 'md', 'txt'])
        recursive: Si True, scan récursif des sous-répertoires
        exclude_patterns: Patterns à exclure (ex: ['.git', '__pycache__', '.pyc'])
        project_name: Nom du projet (auto-détecté si None)
        version: Version du projet

    Returns:
        Liste de dictionnaires compatibles avec add_documents_batch
    """

    directory_path = Path(directory_path).resolve()

    if not directory_path.exists():
        print(f"❌ Répertoire non trouvé : {directory_path}")
        return []

    if not directory_path.is_dir():
        print(f"❌ Le chemin n'est pas un répertoire : {directory_path}")
        return []

    # Filtres par défaut
    if file_filters is None:
        file_filters = ['f90', 'f95']
        """
        'f03', 'f08', 'f', 'for', 'ftn',  # Fortran
                        'md', 'rst', 'txt',  # Documentation
                        'py', 'c', 'cpp', 'h', 'hpp',  # Autres codes
                        'yaml', 'yml', 'json', 'xml']  # Configs
        """

    # Exclusions par défaut
    if exclude_patterns is None:
        exclude_patterns = [
            '.git', '.svn', '.hg',  # Contrôle de version
            '__pycache__', '.pytest_cache', 'node_modules',  # Cache
            '.vscode', '.idea', '.DS_Store',  # IDE/Système
            'build', 'dist', 'target', 'out',  # Build
            '.pyc', '.pyo', '.pyd', '.so', '.dll',  # Binaires
            'CMakeFiles', 'CMakeCache.txt',  # CMake
            'venv', ".venv"

        ]

    # Auto-détecter le nom du projet si pas fourni
    if project_name is None:
        project_name = directory_path.name

    print(f"📁 Scan du répertoire : {directory_path}")
    print(f"   Filtres : {file_filters}")
    print(f"   Récursif : {recursive}")
    print(f"   Projet : {project_name}")

    found_files = []

    # Fonction de scan
    def scan_folder(folder_path: Path, current_depth: int = 0):
        try:
            for item in folder_path.iterdir():

                # Vérifier les exclusions
                if any(pattern in str(item) for pattern in exclude_patterns):
                    continue

                if item.is_file():
                    # Vérifier l'extension
                    extension = item.suffix.lower().lstrip('.')
                    if extension in file_filters:
                        # Calculer le chemin relatif pour les métadonnées
                        relative_path = item.relative_to(directory_path)

                        file_info = {
                            "filepath": str(item),
                            "project_name": project_name,
                            "version": version,
                            "additional_metadata": {
                                "relative_path": str(relative_path),
                                "directory_depth": current_depth,
                                "parent_directory": str(item.parent.name),
                                "scanned_from": str(directory_path),
                                "file_size": item.stat().st_size,
                                "scan_method": "directory_scan"
                            }
                        }
                        found_files.append(file_info)

                elif item.is_dir() and recursive:
                    # Scan récursif des sous-répertoires
                    scan_folder(item, current_depth + 1)

        except PermissionError:
            print(f"⚠️ Accès refusé : {folder_path}")
        except Exception as e:
            print(f"⚠️ Erreur scan {folder_path} : {e}")

    # Lancer le scan
    scan_folder(directory_path)

    # Statistiques
    file_stats = {}
    for file_info in found_files:
        ext = Path(file_info["filepath"]).suffix.lower().lstrip('.')
        file_stats[ext] = file_stats.get(ext, 0) + 1

    print(f"✅ Scan terminé : {len(found_files)} fichiers trouvés")
    for ext, count in sorted(file_stats.items()):
        print(f"   .{ext}: {count} fichiers")

    return found_files


def add_header_to_files(
    directory_path: str,
    entete: str,
    file_filters: List[str] = None,
    recursive: bool = True,
    exclude_patterns: List[str] = None,
    project_name: str = None,
    version: str = "1.0"
) -> None:
    """
    Ajoute un en-tête au début de chaque fichier filtré trouvé dans un répertoire.

    Args:
        directory_path: Répertoire racine à scanner
        entete: Chaîne de texte à insérer au début de chaque fichier
        file_filters: Extensions ciblées (par défaut uniquement .py)
        recursive: Si True, scan récursif
        exclude_patterns: Patterns de fichiers/dossiers à exclure
        project_name: Nom du projet (optionnel)
        version: Version du projet
    """

    # Importer ta fonction existante
    found_files = scan_directory(
        directory_path=directory_path,
        file_filters=file_filters or ["py"],
        recursive=recursive,
        exclude_patterns=exclude_patterns,
        project_name=project_name,
        version=version
    )

    # Nettoyage entete (s'assurer qu'il finit par 2 sauts de ligne)
    entete_clean = entete.strip() + "\n\n"

    for file_info in found_files:
        filepath = Path(file_info["filepath"])

        try:
            # Lire le contenu du fichier
            with filepath.open("r", encoding="utf-8") as f:
                content = f.read()

            # Vérifier si l’entête est déjà présent
            if content.startswith(entete_clean.strip()):
                print(f"⏩ Déjà présent : {filepath}")
                continue

            # Réécrire le fichier avec l’entête au début
            with filepath.open("w", encoding="utf-8") as f:
                f.write(entete_clean + content)

            print(f"✅ Entête ajouté : {filepath}")

        except Exception as e:
            print(f"⚠️ Erreur avec {filepath} : {e}")


def remove_header_from_files(
    directory_path: str,
    entete: str,
    file_filters: List[str] = None,
    recursive: bool = True,
    exclude_patterns: List[str] = None,
    project_name: str = None,
    version: str = "1.0"
) -> None:
    """
    Supprime un en-tête au début de chaque fichier filtré trouvé dans un répertoire.

    Args:
        directory_path: Répertoire racine à scanner
        entete: Chaîne de texte à retirer si trouvée en début de fichier
        file_filters: Extensions ciblées (par défaut uniquement .py)
        recursive: Si True, scan récursif
        exclude_patterns: Patterns de fichiers/dossiers à exclure
        project_name: Nom du projet (optionnel)
        version: Version du projet
    """

    found_files = scan_directory(
        directory_path=directory_path,
        file_filters=file_filters or ["py"],
        recursive=recursive,
        exclude_patterns=exclude_patterns,
        project_name=project_name,
        version=version
    )

    entete_clean = entete.strip() + "\n\n"

    for file_info in found_files:
        filepath = Path(file_info["filepath"])

        try:
            # Lire le contenu du fichier
            with filepath.open("r", encoding="utf-8") as f:
                content = f.read()

            # Vérifier si l’entête est présent au tout début
            if content.startswith(entete_clean):
                new_content = content[len(entete_clean):]

                with filepath.open("w", encoding="utf-8") as f:
                    f.write(new_content)

                print(f"🗑️ Entête supprimé : {filepath}")
            else:
                print(f"⏩ Aucun entête à supprimer : {filepath}")

        except Exception as e:
            print(f"⚠️ Erreur avec {filepath} : {e}")

if __name__ == "__main__":
    entete = '''"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """'''

    #remove_header_from_files("/home/yopla/PycharmProjects/ci-agent/agent/venv", entete, file_filters=["py"])
    add_header_to_files("/home/yopla/PycharmProjects/ci-agent/agent/", entete=entete, file_filters=["py"])

