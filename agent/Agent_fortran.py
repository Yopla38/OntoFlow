"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# agent_fortran.py
import asyncio
import logging
import os
import traceback
from pathlib import Path
from typing import Optional, List, Dict

from agent.src.agent import setup_workspace
from agent.src.factory.development_agent_factory import DevelopmentAgentFactory
from agent.src.factory.rag_agent_factory import RagAgentFactory  # Nouvelle factory
from agent.src.components.file_manager import FileManager
from agent.src.components.task_manager import TaskManager
from agent.src.factory.simple_agent_factory import SimpleAgentFactory
from agent.src.types.roles_definition import select_role
from agent.src.utils.utilitaires import generate_tree_llm

from agent.CONSTANT import NB_ITERATION_IMPROVMENT, MODEL_AGREGATEUR

from .src.agents.agregateur_agent import AiderCodeAggregator

from agent.Onto_wa_rag.CONSTANT import ONTOLOGY_PATH_TTL
from agent.Onto_wa_rag.Integration_fortran_RAG import OntoRAG


class deploy_improvment_agent_fortran:

    def __init__(self, library_path: Path | str = None, idea: str = None):
        self.library_path = Path(library_path) if library_path else None
        self.idea = idea
        self.SUPPORTED_ROLES = ["generateur_idees", "developpeur", "agent_rag"]  # Ajout du rôle RAG
        self.base_dir = self.library_path
        self.exclude_dir = {".agent_workspace", "auto", "log", "__pycache__", "rag_storage"}

        # Agent RAG utilisant la nouvelle architecture
        self.rag_agent = None
        self.logger = logging.getLogger(__name__)

    @classmethod
    async def create(cls, library_path: Path | str = None, idea: str = None):
        instance = cls(library_path, idea)
        await instance.run()
        return instance

    async def auto_improve_cycle(self,
                                 user_idea: str,
                                 idea_generator,
                                 developpeur,
                                 rag_agent,
                                 file_manager: FileManager,
                                 directory: Path,
                                 logger: logging.Logger
                                 ) -> Optional[dict]:
        """
        Cycle d'amélioration complet avec planification, correction agentique, enrichissement et exécution.
        """
        try:
            # --- ÉTAPE 1: PLAN PRÉLIMINAIRE ---
            self.logger.info("Génération d'un plan d'amélioration préliminaire...")
            project_tree_str = generate_tree_llm(directory, excluded_dirs=self.exclude_dir)

            improvement_prompt = f"""DEMANDE UTILISATEUR: {user_idea}

            ARBORESCENCE DU PROJET:
            {project_tree_str}

            TA MISSION : Agir comme un chef de projet technique très efficace.
            1.  Analyse la demande de l'utilisateur pour définir un objectif clair.
            2.  Décompose cet objectif en une séquence de tâches **ACTIONNABLES** pour un agent développeur.
            3.  Une tâche doit **TOUJOURS** résulter en une **modification de code** ou la **création d'un fichier**.
            4.  **N'ÉCRIS PAS** de tâches purement analytiques comme "Analyser X" ou "Étudier Y". L'analyse est le travail du RAG ou est implicite dans le travail du développeur.

            COMMENT FORMULER LES CONSULTATIONS RAG :
            Tu dois déléguer le travail d'analyse au système RAG.

            **POUR LES TÂCHES DE MODIFICATION DE CODE (ex: ajouter un docstring, refactoriser) :**
            -   Demande au RAG d'**EXTRAIRE LE CODE SOURCE PRÉCIS** de l'entité à modifier.
            -   Exemple de consultation : `"Veuillez extraire le code complet de la subroutine 'fourtrans_isf'."`

            **POUR LES TÂCHES DE RÉDACTION (ex: créer un README.md) :**
            -   Demande au RAG de faire une **ANALYSE ou une SYNTHÈSE**.
            -   Exemple de consultation : `"Fournis une synthèse de l'architecture globale du projet en vue de rédiger un README."`

            INSTRUCTIONS FINALES :
            -   Crée une liste de tâches qui modifient ou créent des fichiers.
            -   Associe à chaque tâche la bonne consultation RAG (analyse ou extraction).
            -   Ne spécifie pas de nom de fichier dans les tâches si tu ne le connais pas. La phase de correction le fera.
            """

            preliminary_plan_response = await idea_generator.process_message(improvement_prompt)

            if not preliminary_plan_response or not hasattr(preliminary_plan_response, 'improvement_proposal'):
                self.logger.error("Le plan préliminaire n'a pas pu être généré.")
                return None

            preliminary_proposal = preliminary_plan_response.improvement_proposal
            generated_idea_summary = getattr(preliminary_proposal, 'idea', "Idée générale non spécifiée.")
            rag_consultations = getattr(preliminary_proposal, 'rag_consultations', [])
            preliminary_tasks = getattr(preliminary_proposal, 'tasks', [])

            if not preliminary_tasks:
                self.logger.warning("Le plan préliminaire ne contient aucune tâche.")
                return None

            # --- ÉTAPE 2 & 3: COLLECTE DE CONNAISSANCES ET CORRECTION DU PLAN ---

            final_tasks = []  # Initialisation de la variable pour tous les chemins

            if not rag_consultations:
                self.logger.info("Aucune consultation RAG requise. Le plan est considéré comme final.")
                consultation_results = {}
                final_tasks = [t.model_dump() for t in preliminary_tasks]
            else:
                # CHEMIN AVEC RAG ET CORRECTION
                self.logger.info(f"Exécution de {len(rag_consultations)} consultations RAG...")
                consultation_results = await self._process_rag_consultations(rag_consultations, preliminary_tasks,
                                                                             rag_agent)

                self.logger.info("Lancement de la phase de correction du plan par l'agent...")

                correction_context = "\n".join(
                    f"--- Contexte pour la clé '{key}' ---\n{value}\n---\n"
                    for key, value in consultation_results.items()
                )
                preliminary_tasks_str = "\n".join([str(t.model_dump()) for t in preliminary_tasks])

                correction_prompt = f"""MISSION DE CORRECTION DE TÂCHES

                CONTEXTE : Tu as généré un plan de tâches préliminaire. Un système RAG a ensuite été consulté pour obtenir des informations précises sur le code source.

                TA TÂCHE : Ton unique objectif est de corriger la liste de fichiers (`files`) pour chaque tâche ci-dessous en te basant sur le contexte RAG fourni.

                --- PLAN PRÉLIMINAIRE ---
                {preliminary_tasks_str}
                ---

                --- CONTEXTE FOURNI PAR LE RAG (SOURCE DE VÉRITÉ) ---
                {correction_context}
                ---

                INSTRUCTIONS PRÉCISES :
                1. Pour chaque tâche du plan préliminaire, regarde sa `rag_context_key`.
                2. Trouve le contexte correspondant dans "CONTEXTE FOURNI PAR LE RAG".
                3. Identifie le nom du fichier source mentionné dans ce contexte.
                4. **Modifie la clé `files` de la tâche pour qu'elle contienne UNIQUEMENT ce nom de fichier.**
                5. Si le contexte RAG est un message d'erreur ou de clarification (ex: CLARIFICATION_NEEDED), laisses la liste `files` de la tâche vide `[]` pour indiquer que l'information est manquante.
                6. Pour les tâches qui créent un nouveau fichier (ex: README.md), conserve le nom de fichier original.
                7. **Ne modifie AUCUN autre champ (title, description, dependencies, etc.).**

                FORMAT DE SORTIE ATTENDU :
                Retourne **UNIQUEMENT L'OBJET `Improvement_proposal`** contenant la liste des tâches corrigées dans le champ `tasks`. Ne génère PAS de champ `Idea` ou `Rag_consultations`. Ta réponse doit commencer directement par `Improvement_proposal`.
                """

                corrected_plan_response = await idea_generator.process_message(correction_prompt)

                if (corrected_plan_response and hasattr(corrected_plan_response, 'improvement_proposal') and
                        corrected_plan_response.improvement_proposal.tasks):
                    final_tasks_pydantic = corrected_plan_response.improvement_proposal.tasks
                    final_tasks = [t.model_dump() for t in final_tasks_pydantic]
                    self.logger.info(f"Plan corrigé avec succès. {len(final_tasks)} tâches finales prêtes.")
                else:
                    self.logger.warning(
                        "L'agent n'a pas retourné un plan corrigé valide. Utilisation du plan préliminaire.")
                    final_tasks = [t.model_dump() for t in preliminary_tasks]

                # Nous nous assurons ici que tous les chemins sont relatifs à la racine du projet.
                self.logger.info("Normalisation des chemins de fichiers dans les tâches finales...")

                for task_dict in final_tasks:
                    if "files" in task_dict and isinstance(task_dict["files"], list):
                        normalized_files = []
                        for file_path in task_dict["files"]:
                            try:
                                # os.path.abspath est nécessaire si le chemin est déjà relatif mais mal formé
                                abs_path = os.path.abspath(file_path)
                                project_root_abs = os.path.abspath(directory)

                                # Vérifier si le chemin commence bien par la racine du projet
                                if abs_path.startswith(project_root_abs):
                                    relative_path = os.path.relpath(abs_path, project_root_abs)
                                    normalized_files.append(relative_path)
                                else:
                                    # Le chemin est en dehors du projet (ex: /tmp/file) ou un nom simple (README.md)
                                    # Pour un nom simple, abspath le placera dans le répertoire courant, qui est la racine du projet
                                    # donc relpath fonctionnera. On garde ce cas.
                                    self.logger.warning(f"Le chemin '{file_path}' semble être en dehors du projet.")
                                    normalized_files.append(file_path)  # On le garde tel quel
                            except Exception as e:
                                self.logger.error(f"Erreur de normalisation pour le chemin '{file_path}': {e}")
                                normalized_files.append(file_path)  # Sécurité : on garde le chemin original

                        if task_dict["files"] != normalized_files:
                            self.logger.info(
                                f"Chemins normalisés pour '{task_dict['title']}': {task_dict['files']} -> {normalized_files}")
                            task_dict["files"] = normalized_files

            # --- ÉTAPE 4: ENRICHISSEMENT FINAL DU PLAN CORRIGÉ ---
            self.logger.info("Enrichissement final des descriptions des tâches...")
            for task_dict in final_tasks:
                rag_key = task_dict.get("rag_context_key")
                if rag_key and rag_key in consultation_results:
                    context_to_add = consultation_results[rag_key]
                    if not context_to_add.strip().startswith("// ERREUR"):
                        original_description = task_dict.get("description", "")
                        task_dict[
                            "description"] = f"{original_description}\n\n--- CONTEXTE ... ---\n{context_to_add}\n--- FIN DU CONTEXTE ---"
                        task_dict["enriched_with_rag"] = True
                    else:
                        self.logger.warning(
                            f"Pas d'enrichissement pour '{task_dict['title']}' car le RAG a retourné une erreur.")

            # --- ÉTAPE 5: EXÉCUTION DU PLAN FINAL ET ENRICHI ---
            self.logger.info("Exécution du plan de tâches final...")
            for i, task_dict in enumerate(final_tasks):
                try:
                    titre = task_dict.get("title", "Titre manquant")
                    self.logger.info(f"Traitement de la tâche finale {i + 1}: {titre}")
                    await developpeur.process_task(task_dict)
                    self.logger.info(f"Tâche {i + 1} traitée")
                except Exception as e:
                    self.logger.error(f"Erreur lors de l'exécution de la tâche {i + 1}: {e}", exc_info=True)
                    continue

            # --- ÉTAPE 6: VÉRIFICATION ---
            self.logger.info("Phase de vérification...")
            verification_prompt = f"""DEMANDE INITIALE: {user_idea} ... (votre prompt de vérification complet)"""

            verification_result = await idea_generator.process_message(verification_prompt)

            # --- ÉTAPE 7: CORRECTIONS POST-VÉRIFICATION ---
            if (verification_result and hasattr(verification_result, 'improvement_proposal') and
                    verification_result.improvement_proposal.tasks):

                correction_tasks_pydantic = verification_result.improvement_proposal.tasks
                if correction_tasks_pydantic:
                    self.logger.info("Traitement des corrections post-vérification...")
                    # On doit aussi enrichir ces nouvelles tâches avant de les exécuter
                    correction_tasks = [t.model_dump() for t in correction_tasks_pydantic]
                    for task_dict in correction_tasks:
                        rag_key = task_dict.get("rag_context_key")
                        if rag_key and rag_key in consultation_results:  # On réutilise les contextes déjà collectés
                            context_to_add = consultation_results[rag_key]
                            if not context_to_add.strip().startswith("// ERREUR"):
                                task_dict[
                                    "description"] += f"\n\n--- CONTEXTE ... ---\n{context_to_add}\n--- FIN DU CONTEXTE ---"

                    # Exécution des tâches de correction enrichies
                    for task in correction_tasks:
                        await developpeur.process_task(task)

            return verification_result

        except Exception as e:
            self.logger.error(f"Erreur dans auto_improve_cycle: {e}", exc_info=True)
            return None

    async def _process_rag_consultations(
            self,
            rag_consultations: List,
            preliminary_tasks: List,  # Nouvel argument
            rag_agent: OntoRAG
    ) -> Dict[str, str]:
        """
        Traite chaque consultation RAG comme une conversation isolée et contextuelle.
        1. Isole chaque conversation en vidant la mémoire de l'agent.
        2. Enrichit la requête RAG avec le contexte de la tâche du développeur associée.
        3. Permet à l'agent d'utiliser sa mémoire interne pour résoudre la requête.
        4. Gère les cas où une clarification est tout de même nécessaire.
        """
        consultation_results = {}

        # Créer un mapping rapide de clé RAG vers la tâche correspondante
        key_to_task_map = {
            task.rag_context_key: task
            for task in preliminary_tasks if task.rag_context_key
        }

        for consultation in rag_consultations:
            try:
                # Normaliser la consultation
                consultation_dict = consultation.model_dump() if hasattr(consultation, 'model_dump') else dict(
                    consultation)
                query = consultation_dict.get("query")
                storage_key = consultation_dict.get("storage_key")

                if not query or not storage_key:
                    self.logger.warning(f"Consultation RAG invalide ignorée: {consultation_dict}")
                    continue

                # --- 1. ISOLATION DE LA CONVERSATION ---
                self.logger.info(f"--- Nouvelle session de consultation pour la clé '{storage_key}' ---")
                rag_agent.agent_fortran.clear_memory()

                # --- 2. ENRICHISSEMENT PROACTIF DU CONTEXTE ---
                associated_task = key_to_task_map.get(storage_key)
                enriched_query = query  # Par défaut, on utilise la requête de base

                if associated_task:
                    self.logger.info(f"Contexte trouvé pour '{storage_key}': Tâche '{associated_task.title}'")
                    # On construit un prompt riche pour l'agent RAG
                    enriched_query = f"""Contexte de la mission : Un agent développeur doit accomplir la tâche suivante :
- **Titre de la tâche :** {associated_task.title}
- **Description de la tâche :** {associated_task.description}

Pour l'aider, j'ai besoin que tu répondes à cette question de manière aussi complète et précise que possible. Ta réponse servira de contexte principal pour le développeur.

**Ma question précise est :** "{query}"

Fournis le code, le nom du fichier, et toute analyse pertinente que tu peux extraire.
"""
                else:
                    self.logger.warning(
                        f"Aucune tâche associée trouvée pour la clé RAG '{storage_key}'. Utilisation de la requête simple.")

                # --- 3. APPEL DE L'AGENT AVEC MÉMOIRE ACTIVÉE ---
                self.logger.info(f"Envoi de la requête enrichie à l'agent RAG...")

                # On passe la requête enrichie comme argument positionnel
                agent_response = await rag_agent.agent_fortran.run(
                    enriched_query,
                    use_memory=True  # Permet à l'agent de raisonner en interne
                )

                # --- 4. GESTION DE LA RÉPONSE ---
                if agent_response:
                    if agent_response.startswith("CLARIFICATION_NEEDED:"):
                        clarification_question = agent_response.replace("CLARIFICATION_NEEDED:", "").strip()
                        error_message = (
                            f"// ERREUR DE CONTEXTE: Même avec un contexte enrichi, le RAG a besoin d'une clarification.\n"
                            f"// Question de l'agent: '{clarification_question}'")
                        consultation_results[storage_key] = error_message
                        self.logger.error(
                            f"Clarification toujours nécessaire pour '{storage_key}': {clarification_question}")
                    else:
                        consultation_results[storage_key] = agent_response
                        self.logger.info(f"Réponse complète obtenue pour la clé '{storage_key}'.")
                else:
                    msg = f"// ERREUR DE CONTEXTE: L'agent RAG n'a retourné aucune réponse pour la requête enrichie."
                    consultation_results[storage_key] = msg
                    self.logger.error(f"Réponse vide de l'agent pour la clé '{storage_key}'.")

            except Exception as e:
                self.logger.error(
                    f"Erreur majeure lors du traitement de la consultation pour la clé '{storage_key}': {e}",
                    exc_info=True)
                consultation_results[storage_key] = f"// ERREUR SYSTÈME: Une exception a eu lieu: {e}"

        return consultation_results

    async def run(self):
        """Point d'entrée principal"""
        try:
            if not self.library_path or not self.idea:
                self.logger.error("library_path ou idea manquant")
                print("Erreur: Chemin et idée requis.")
                return

            await self.main(directory=self.library_path, idea_str=self.idea)

        except KeyboardInterrupt:
            print("\nArrêt du programme...")
        except Exception as e:
            print(f"Erreur fatale: {str(e)}")
            self.logger.error(f"Erreur fatale: {str(e)}", exc_info=True)
        finally:
            await self._cleanup()

    async def _cleanup(self):
        """Nettoie les ressources"""
        try:
            if self.rag_agent:
                self.rag_agent.cleanup()
                self.logger.info("Agent RAG nettoyé")
        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage: {e}")

    async def main(self, directory: Path, idea_str: str):
        """Méthode principale d'exécution"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        try:
            # 1. Configuration de l'espace de travail
            workspace_dir, projects_dir, templates_dir, venvs_dir = await setup_workspace(directory)
            project_name = "generateur_idees_evolutif"
            venv_dir = venvs_dir / project_name
            venv_dir.mkdir(exist_ok=True)

            self.logger.info(f"Espace de travail: {projects_dir}")

            # 2. Initialisation des gestionnaires
            file_manager = FileManager(project_path=str(directory), venv_path=str(venv_dir))
            await file_manager.initialize_venv()
            task_manager = TaskManager()

            # 3. Configuration des factories
            self.logger.info("Création et initialisation du nouvel agent RAG Ontologique...")

            # Instanciez la classe OntoRAG
            self.rag_agent = OntoRAG(
                storage_dir=str(directory / "rag_storage"),  # Le stockage RAG se fera ici
                ontology_path=ONTOLOGY_PATH_TTL  # Assurez-vous que cette constante est accessible
            )
            # Initialisation
            await self.rag_agent.initialize()

            processing_options = {
                "clean_fortran": True,  # Activer le nettoyage pour les fichiers Fortran
                "overwrite_cleaned_file": True  # Écraser le fichier original
            }

            # --- 4. Indexation des fichiers (maintenant propres) ---
            self.logger.info(f"Scan du répertoire du projet pour l'indexation: {directory}")
            documents_to_index = self.rag_agent.scan_directory(
                directory_path=str(directory),
                exclude_patterns=list(self.exclude_dir) + [".fortran_backup"]
                # IMPORTANT: Exclure le répertoire de backup !
            )

            self.logger.info("Agent RAG initialisé. Lancement de l'indexation...")

            self.logger.info(f"Scan du répertoire du projet: {directory}")
            documents_to_index = self.rag_agent.scan_directory(
                directory_path=str(directory),
                # Les exclusions par défaut de scan_directory sont probablement suffisantes,
                # mais vous pouvez les surcharger ici si besoin.
                exclude_patterns=list(self.exclude_dir)
            )
            if documents_to_index:
                self.logger.info(f"Indexation de {len(documents_to_index)} fichiers...")
                # preserve_order=True est une bonne pratique pour le Fortran
                await self.rag_agent.add_documents_batch(documents_to_index, preserve_order=True) # processing_options=processing_options
                self.logger.info("Indexation terminée.")
            else:
                self.logger.warning("Aucun fichier à indexer n'a été trouvé via scan_directory.")

            # 5. Création des autres agents
            idea_generator = SimpleAgentFactory.create_agent_from_role(
                role_config=select_role("GenerateurIdees", base_dir=directory, excluded_dir=self.exclude_dir),
                project_path=str(projects_dir),
                project_name=project_name,
                workspace_root=str(workspace_dir),
                file_manager=file_manager,
                task_manager=task_manager
            )

            developpeur = DevelopmentAgentFactory.create_agent_from_role(
                role_config=select_role("Developpeur", directory, excluded_dir=self.exclude_dir),
                project_path=str(projects_dir),
                project_name=project_name,
                workspace_root=str(workspace_dir),
                file_manager=file_manager,
                task_manager=task_manager,
                agregateur=AiderCodeAggregator(MODEL_AGREGATEUR)
            )

            # Donner accès à l'agent RAG au générateur d'idées
            #idea_generator.rag_agent = self.rag_agent

            self.logger.info("Tous les agents créés.")

            # 6. Boucle d'amélioration continue
            for iteration in range(NB_ITERATION_IMPROVMENT):
                self.logger.info(f"\n=== ITÉRATION {iteration + 1}/{NB_ITERATION_IMPROVMENT} ===")

                try:
                    result = await self.auto_improve_cycle(
                        user_idea=idea_str,
                        idea_generator=idea_generator,
                        developpeur=developpeur,
                        rag_agent=self.rag_agent,  # Passer l'agent RAG
                        file_manager=file_manager,
                        directory=directory,
                        logger=self.logger
                    )

                    if result:
                        self.logger.info(f"Itération {iteration + 1} terminée avec succès")
                    else:
                        self.logger.warning(f"Itération {iteration + 1} avec problèmes")

                    await asyncio.sleep(2)

                except Exception as e:
                    self.logger.error(f"Erreur itération {iteration + 1}: {e}", exc_info=True)
                    continue

            self.logger.info("=== PROCESSUS TERMINÉ ===")

            # 7. Nettoyage
            try:
                if hasattr(idea_generator, 'cleanup'):
                    idea_generator.cleanup()
                if hasattr(developpeur, 'cleanup'):
                    developpeur.cleanup()
                self.logger.info("Agents nettoyés")
            except Exception as e:
                self.logger.error(f"Erreur nettoyage agents: {e}")

            return "Auto-amélioration terminée avec succès"

        except Exception as e:
            self.logger.error(f"Erreur dans main: {str(e)}", exc_info=True)
            traceback.print_exc()
            raise


async def Deployer_agent_fortran(library_path: Path | str, idea: str):
    """Fonction utilitaire pour déployer l'agent Fortran"""
    try:
        print(f"Déploiement de l'agent Fortran...")
        print(f"Projet: {library_path}")
        print(f"Idée: {idea}")

        agent_fortran = await deploy_improvment_agent_fortran.create(library_path, idea)
        print("Agent déployé avec succès")
        return agent_fortran

    except Exception as e:
        print(f"Erreur lors du déploiement: {e}")
        traceback.print_exc()
        raise


# --- EXEMPLE D'UTILISATION ---
async def example_run():
    test_library_path = "/home/yopla/test_agent"
    test_idea = ("Veuillez écrire le docstring complet de la fonction inspect_rototranslation. "
                 "Écrivez également le fichier README.md du module reformatting")

    if not Path(test_library_path).exists():
        print(f"Le répertoire {test_library_path} n'existe pas.")
        return

    print("=" * 60)
    print("DÉMARRAGE DE L'EXEMPLE")
    print("=" * 60)

    try:
        await Deployer_agent_fortran(test_library_path, test_idea)
        print("\n" + "=" * 60)
        print("EXEMPLE TERMINÉ AVEC SUCCÈS")
        print("=" * 60)
    except Exception as e:
        print(f"\nErreur: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(example_run())