"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

import asyncio
import logging
import os
import traceback
from pathlib import Path

import socketio

from agent.KEY import OPENAI_KEY
from agent.src.agent import setup_workspace
from agent.src.components.communicative_server import communication_server
from agent.src.factory.development_agent_factory import DevelopmentAgentFactory
from agent.src.components.file_manager import FileManager
from agent.src.components.task_manager import TaskManager
from agent.src.factory.simple_agent_factory import SimpleAgentFactory
from agent.src.interface.html_manager import HTMLManager
from agent.src.types.roles_definition import get_files_content, select_role, generate_file_content_for_role
from agent.src.utils.utilitaires import generate_tree_llm

from CONSTANT import NB_ITERATION_IMPROVMENT

from CONSTANT import MODEL_AGREGATEUR
from agent.src.agents.agregateur_agent import AiderCodeAggregator


class deploy_improvment_agent_MIA:

    def __init__(self, library_path: Path | str = None, idea: str = None):
        self.library_path = library_path
        self.idea = idea
        self.SUPPORTED_ROLES = ["generateur_idees", "developpeur"]
        self.base_dir = None
        self.exclude_dir = {".agent_workspace", "auto", "log", "__pycache__"}

    @classmethod
    async def create(cls, library_path: Path | str = None, idea: str = None):
        instance = cls(library_path, idea)
        await instance.run()
        return instance

    async def auto_improve_cycle(self,
                                 idea,
                                 idea_generator,
                                 developpeur,
                                 file_manager: FileManager,
                                 directory,
                                 logger: logging.Logger
                                 ) -> None:
        """
        Exécute un cycle d'amélioration simple
        """

        try:
            # 1. Obtenir une proposition d'amélioration
            logger.info("Génération d'une proposition d'amélioration...")
            if idea != "":
                idea = " pour " + idea
            content_files = generate_file_content_for_role(self.base_dir, self.exclude_dir)
            improvement = await idea_generator.process_message(
                f"""Voici le contenu des fichiers du projet:
CONTENU DES FICHIERS:
    
{content_files}

Mission:
Analysez le projet et proposez une amélioration concrète{idea}."""
            )

            # Récupérer l'idée principale
            idea = improvement.improvement_proposal.idea
            logger.info(f"Idée lumineuse: {idea}")

            async def process_task(tasks):

                if not isinstance(tasks, list):
                    logger.error("La variable 'tasks' n'est pas une liste. Type réel : %s", type(tasks))
                    return  # Fin prématurée si tasks n'est pas une liste

                # Parcourir les tâches avec gestion des cas inattendus
                for i, task in enumerate(tasks):
                    try:
                        # Conversion en dictionnaire si nécessaire
                        if hasattr(task, "dict"):  # Cas d'un modèle Pydantic
                            task_dict = task.model_dump()
                        elif isinstance(task, dict):  # Si c'est déjà un dictionnaire
                            task_dict = task
                        else:
                            logger.warning(f"Tâche inattendue au format inconnu : {task}")
                            continue  # Ignorer les tâches au format inattendu

                        # Extraire les informations pour le log
                        titre = task_dict.get("title", "Titre manquant")
                        description = task_dict.get("description", "Description manquante")

                        # Log de la tâche avant de la traiter
                        logger.info(f"Traitement de la tâche {i + 1}: Titre = {titre}, Description = {description}")
                        logger.info(f"Données brutes de la tâche : {task_dict}")

                        # Envoi de la tâche au développeur
                        logger.info(f"Envoi de la tâche {i + 1} à 'developpeur.process_task'...")
                        implementation = await developpeur.process_task(task_dict)
                        logger.info(f"Tâche {i + 1} traitée par le développeur. Résultat : {implementation}")

                    except Exception as e:
                        logger.error(f"Erreur lors du traitement de la tâche {i + 1} : {e}")
                        continue  # Passer à la tâche suivante en cas d'erreur

            # Parcourir les tâches avec gestion des cas inattendus
            await process_task(improvement.improvement_proposal.tasks)

            #  Retourner l'ensemble au générateur d'idées pour validation
            logger.info("Vérification des modification...")
            improvement = await idea_generator.process_message(
                f"""
    CONTEXTE DU PROJET:
    Arborescence du projet: 
    {generate_tree_llm(directory, included_dirs=os.listdir(directory), excluded_dirs=self.exclude_dir)}

    CONTENU DES FICHIERS:
    {get_files_content(directory, generate_tree_llm(directory, included_dirs=os.listdir(directory), excluded_dirs={".agent_workspace", "auto", "log", "__pycache__"}))}
    Veuillez vérifier que le projet actuel réponde à votre précédente demande et est executable en l'état. Si ce n'est pas le cas, générez des tâches de correction.\n
ATTENTION : si l'état actuel satisfait la demande, vous ne devez pas proposer d'autres idées."""
            )
            # Parcourir les tâches avec gestion des cas inattendus
            await process_task(improvement.improvement_proposal.tasks)

            return improvement

        except AttributeError as e:
            logger.error(f"Erreur dans la structure des données : {e}")
        except Exception as e:
            logger.error(f"Erreur inattendue : {e}")

    async def run(self):
        try:
            await self.main(directory=Path(self.library_path), idea=self.idea)
        except KeyboardInterrupt:
            print("\nArrêt du programme...")
        except Exception as e:
            print(f"Erreur fatale: {str(e)}")

    async def main(self, directory: Path = None, idea: str = ""):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        try:
            workspace_dir, projects_dir, templates_dir, venvs_dir = await setup_workspace(directory)
            project_name = "generateur_idees_evolutif"
            venv_dir = venvs_dir / project_name
            venv_dir.mkdir(exist_ok=True)
            logger.info(f"Projet: {projects_dir}")

            # Initialisation des managers
            file_manager = FileManager(
                project_path=str(directory),
                venv_path=str(venv_dir)
            )
            await file_manager.initialize_venv()

            task_manager = TaskManager()
            html_manager = HTMLManager(str(workspace_dir))

            # Mise à jour des rôles supportés
            DevelopmentAgentFactory.SUPPORTED_ROLES = self.SUPPORTED_ROLES

            self.base_dir = directory

            # Création des agents
            idea_generator = SimpleAgentFactory.create_agent_from_role(
                role_config=select_role("GenerateurIdees", base_dir=directory, excluded_dir={".agent_workspace", "auto", "log", "__pycache__"}),
                project_path=str(projects_dir),
                project_name=project_name,
                workspace_root=str(workspace_dir),
                file_manager=file_manager,
                task_manager=task_manager
            )

            """
            # Configuration de la communication (uniquement pour SimpleAgent)
            communication_config = {
                'provider_type': 'openai',
                'model': 'gpt-4o',
                'api_key': OPENAI_KEY,
                'system_prompt': f"Je suis un assistant de communication d'agent autonome qui peut vous informer sur:
                                               - L'état actuel du projet et des agents
                                               - Les tâches en cours et terminées
                                               - Le contexte et les décisions prises
                                               Je peux communiquer pendant que les agents travaillent.
                                               Paramètres et contexte de l'agent autonome que je représente : {idea_generator.system_prompt}
                                               "
            }

            # Activer la communication pour cet agent
            idea_generator.setup_communication(communication_config, socketio=socketio)

            # Enregistrement explicite de l'agent
            communication_server.register_agent('idea_generator', idea_generator)
            logger.info("Agent enregistré dans le serveur de communication")
            # Démarrer le serveur de communication
            communication_server.start(host='0.0.0.0', port=5000)
            """

            developpeur = DevelopmentAgentFactory.create_agent_from_role(
                role_config=select_role("Developpeur", directory, excluded_dir={".agent_workspace", "auto", "log", "__pycache__"}),
                project_path=str(projects_dir),
                project_name=project_name,
                workspace_root=str(workspace_dir),
                file_manager=file_manager,
                task_manager=task_manager,
                agregateur=AiderCodeAggregator(MODEL_AGREGATEUR)
            )

            # Boucle d'auto-amélioration
            iterations = NB_ITERATION_IMPROVMENT  # Nombre d'itérations d'amélioration
            for i in range(iterations):
                logger.info(f"\nDémarrage de l'itération {i + 1}/{iterations}")

                results = await self.auto_improve_cycle(
                    idea,
                    idea_generator,
                    developpeur,
                    file_manager,
                    directory,
                    logger
                )

                # Pause entre les itérations
                await asyncio.sleep(2)

            logger.info("Processus d'auto-amélioration terminé avec succès!")
            # Nettoyage des ressources
            idea_generator.cleanup()
            developpeur.cleanup()
            idea_generator.is_running = False
            #communication_server.stop()
            return "Auto-amélioration terminée"

        except Exception as e:
            logger.error(f"Erreur lors de l'exécution: {str(e)}")
            traceback.print_exc()
            raise


async def Deployer_agent_bibliotheque(library_path, idea):
    try:
        # Créer et exécuter l'agent
        print(f"Création de l'agent avec l'idée: {idea}")
        agent_MIA = await deploy_improvment_agent_MIA.create(library_path, idea)
        print("Agent créé avec succès")
        return agent_MIA
    except Exception as e:
        print(f"Erreur lors du déploiement de l'agent: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    agent_MIA = asyncio.run(Deployer_agent_bibliotheque("/home/yopla/test_agent", "un code efficace de calcul de nombres premiers en python"))