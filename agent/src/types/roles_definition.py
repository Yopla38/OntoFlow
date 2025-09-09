"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

import enum
import json
import os
from pathlib import Path
from typing import Dict, Any

from agent.src.utils.utilitaires import generate_tree_llm

from CONSTANT import MODEL_HIGH

from ROLES import mes_roles


def get_files_content(base_dir: str, tree: str) -> str:
    """Récupère et formate le contenu de tous les fichiers pertinents dans l'arborescence"""
    formatted_content = []

    for line in tree.split('\n'):
        if line.strip() and "──" in line:
            filename = line.split("──")[-1].strip()
            filepath = os.path.join(base_dir, filename)
            if os.path.isfile(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        formatted_content.append(f"Nom du fichier: {filename}\nContenu du fichier:\n{content}\n")
                except Exception as e:
                    formatted_content.append(f"Nom du fichier: {filename}\nErreur de lecture: {str(e)}\n")

    return "\n".join(formatted_content) if formatted_content else "Aucun fichier trouvé"


# claude-3-5-sonnet-20241022
def generate_folder_context_for_role(base_dir: Path | str = None, excluded_dir=None) -> str:
    if excluded_dir is None:
        excluded_dir = {".agent_workspace"}
    tree = generate_tree_llm(base_dir, included_dirs=os.listdir(base_dir), excluded_dirs=excluded_dir)
    return tree


def generate_file_content_for_role(base_dir: Path | str = None, excluded_dir=None) -> str:
    if excluded_dir is None:
        excluded_dir = {".agent_workspace"}
    return get_files_content(base_dir,
                             generate_tree_llm(
                                 base_dir,
                                 included_dirs=os.listdir(base_dir),
                                 excluded_dirs=excluded_dir))


#  JINJA est tres pratique
def select_role(role_name: str = None, base_dir: Path | str = None, excluded_dir: enum = None) -> Dict:

    tree = generate_folder_context_for_role(base_dir, excluded_dir)
    roles = mes_roles(tree)

    if role_name in roles:
        return roles[role_name]


def read_json_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as fichier:
            return json.load(fichier)
    except FileNotFoundError:
        print("Erreur : Le fichier n'a pas été trouvé")
    except json.JSONDecodeError:
        print("Erreur : Le fichier JSON n'est pas valide")
    except Exception as e:
        print(f"Erreur inattendue : {e}")

def get_base_communication_config(api_key: str) -> Dict[str, Any]:
    """Configuration de base pour la communication"""
    return {
        'provider_type': 'openai',
        'model': 'gpt-3.5-turbo',
        'api_key': api_key,
        'system_prompt': """Je suis un assistant de communication qui peut vous informer sur:
- L'état actuel du projet et des agents
- Les tâches en cours et terminées
- Le contexte et les décisions prises"""
    }


def get_super_agent_communication_config(api_key: str) -> Dict[str, Any]:
    """Configuration spécifique pour le SuperAgent"""
    base_config = get_base_communication_config(api_key)
    base_config.update({
        'system_prompt': """Je suis le SuperAgent, responsable de la gestion des projets et des agents.
Je peux :
1. Configurer de nouveaux projets
2. Recruter et gérer des agents
3. Créer et exécuter des workflows
4. Superviser l'ensemble des opérations"""
    })
    return base_config


def get_agent_specific_config(agent_type: str, api_key: str, base_dir: Path = None) -> Dict[str, Any]:
    """Configuration spécifique selon le type d'agent"""
    base_config = get_base_communication_config(api_key)

    specific_prompts = {
        'generateur_idees': f"""Je suis l'interface de communication pour le générateur d'idées.
Je peux vous informer sur :
- Les idées en cours d'analyse
- Les améliorations proposées
- L'état d'avancement des suggestions
Contexte du projet: {base_dir if base_dir else 'Non spécifié'}""",

        'developpeur': """Je suis l'interface de communication pour le développeur.
Je peux vous informer sur :
- Les modifications en cours
- Les implémentations réalisées
- Les difficultés techniques rencontrées""",

        # Ajoutez d'autres types d'agents selon besoin
    }

    if agent_type in specific_prompts:
        base_config['system_prompt'] = specific_prompts[agent_type]

    return base_config
