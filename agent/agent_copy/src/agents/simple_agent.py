"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# src/agents/simple_agent.py
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from pydantic import BaseModel

from agent.src.agent import Agent
from agent.src.interface.html_manager import HTMLManager
from agent.src.server.form_server import FormServer
from agent.src.types.enums import AgentRole, AgentCapability
from agent.src.components.file_manager import FileManager
from agent.src.components.task_manager import TaskManager, Tache

from agent.src.types.interfaces import LLMProvider, MemoryProvider


# TODO Simplifier cet agent qui est une copie de developmentagent
class SimpleAgent(Agent):
    """
    Agent spécialisé pour les tâches de développement.

    Capacités :
    - Traitement des tâches selon le rôle
    - Gestion des fichiers
    - Génération d'interfaces HTML
    - Gestion des dépendances
    - Validation de code
    """

    def __init__(
            self,
            name: str,
            role_name: str,
            llm_provider: LLMProvider,
            memory_provider: MemoryProvider,
            system_prompt: str,
            project_path: str,
            html_manager: HTMLManager,
            project_name: str,
            file_manager: Optional[FileManager] = None,
            task_manager: Optional[TaskManager] = None,
            memory_size: int = 1000,
            pydantic_model: Optional[BaseModel] = None,
            response_format: str = "",
            actionable_keywords_files: Optional[List] = None,
            communication_config: Optional[Dict] = None,
            streaming: Optional[bool] = False
    ):
        # Conversion du rôle et initialisation de la classe parent

        self.a_k_files = ['files']
        if isinstance(actionable_keywords_files, list):
            self.a_k_files.extend(actionable_keywords_files)

        role = self._convert_role(role_name)
        super().__init__(
            name=name,
            role=role,
            llm_provider=llm_provider,
            memory_provider=memory_provider,
            system_prompt=system_prompt,
            memory_size=memory_size,
        )

        # Composants de développement
        self.pydantic_model = pydantic_model
        self.html_manager = html_manager
        self.project_name = project_name
        self.project_path = project_path
        self.response_format = response_format

        # Gestionnaires
        self.file_manager = file_manager or FileManager(project_path)
        self.task_manager = task_manager or TaskManager()

        # Formulaire interactif
        self.form_server = FormServer()
        self.form_response = None
        self.form_event = asyncio.Event()



        # Configuration de la communication si fournie
        if communication_config:
            self.setup_communication(communication_config)

        # Configuration du logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.name}")
        self.logger.setLevel(logging.INFO)

    async def process_message(self, message: Union[str, List[Dict[str, str]]], pydantic_model=None, streaming_callback=None, streaming=False) -> Any:
        """
        Traite un message et gère les interactions si nécessaire.

        Args:
            :param message:
            :param streaming:
            :param streaming_callback:

        Returns:
            Résultat du traitement (peut être structuré via Pydantic)

        """
        try:
            self.logger.info(f"Traitement du message par {self.name}")

            if self.pydantic_model or pydantic_model:
                # Obtenir la réponse structurée du LLM
                response = await self.llm_provider.generate_response(
                    message,
                    pydantic_model=pydantic_model if pydantic_model else self.pydantic_model
                )
                #return response  # TODO c'est la misère ici !!!
                # Normaliser les clés de la réponse
                normalized_response = self._normalize_response_keys(response)

                await self.knowledge_base.store_pydantic_result(self.name, normalized_response)

                # Générer le HTML pour visualisation/interaction
                html_file = self.html_manager.generate_html(
                    instance=normalized_response,
                    project_name=self.project_name,
                    agent_name=self.name,
                    prompt=self.system_prompt,
                    input_message=message
                )

                # Traiter les fichiers si présents
                for attr in self.a_k_files:
                    if hasattr(normalized_response, attr):
                        await self._process_files(getattr(normalized_response, attr))

                # Retourner le résultat normalisé
                return normalized_response

            return await super().process_message(message, streaming=streaming)

        except Exception as e:
            self.logger.error(f"Erreur dans process_message: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def consult_rag_agent(self, query: str) -> Dict[str, str]:
        """Consulte l'agent RAG et obtient une clé de stockage"""
        if not self.rag_agent:
            return {
                "storage_key": None,
                "summary": "Agent RAG non disponible",
                "status": "error"
            }

        try:
            self.logger.info(f"Consultation de l'agent RAG: '{query}'")

            # Demander à l'agent RAG de traiter et stocker
            result = await self.rag_agent.process_rag_request_with_key(query)

            # Stocker dans l'historique local
            if result.get("status") == "success":
                self.rag_consultations.append({
                    "query": query,
                    "storage_key": result.get("storage_key"),
                    "summary": result.get("summary"),
                    "timestamp": datetime.now().isoformat()
                })

            return result

        except Exception as e:
            self.logger.error(f"Erreur lors de la consultation RAG: {e}")
            return {
                "storage_key": None,
                "summary": f"Erreur: {str(e)}",
                "status": "error"
            }

    async def get_rag_context(self, storage_key: str) -> str:
        """Récupère le contexte RAG via la clé"""
        if not self.rag_agent or not storage_key:
            return "Contexte RAG non disponible"

        try:
            data = await self.rag_agent.get_stored_rag_data(storage_key)

            if data:
                # Formater les données pour affichage
                return self._format_rag_data_for_context(data)
            else:
                return "Aucune donnée trouvée pour cette clé"

        except Exception as e:
            self.logger.error(f"Erreur récupération contexte RAG: {e}")
            return f"Erreur: {str(e)}"

    def _format_rag_data_for_context(self, data: Any) -> str:
        """Formate les données RAG pour inclusion dans les tâches"""
        try:
            if isinstance(data, dict) and 'rag_analysis' in data:
                return data['rag_analysis'].get('consolidated_context', str(data))
            elif hasattr(data, 'rag_analysis'):
                return getattr(data.rag_analysis, 'consolidated_context', str(data))
            else:
                return str(data)
        except Exception as e:
            return f"Données RAG: {str(data)}"

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Surcharge pour ajouter la validation des fichiers"""
        try:
            # Utiliser le traitement de base de l'agent
            result = await super().process_task(task)

            # Ajouter la validation spécifique aux fichiers
            if any(word in result for word in self.a_k_files) in result and not self.can_perform_action(AgentCapability.MODIFY_FILES):
                raise PermissionError(f"Agent {self.name} non autorisé à modifier des fichiers")

            return result

        except Exception as e:
            self.logger.error(f"Erreur lors du traitement de la tâche: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def _process_files(self, files: List[Dict[str, Any]]) -> None:
        """Traite les fichiers à créer ou modifier"""
        try:
            for file_info in files:
                if not self.can_perform_action(AgentCapability.MODIFY_FILES):
                    self.logger.warning(f"Agent {self.name} tente de modifier des fichiers sans autorisation")
                    continue

                await self.file_manager.create_or_update_file(
                    file_info.file_path,
                    file_info.code_field.content
                )
                self.logger.info(f"Fichier traité: {file_info.file_path}")

        except Exception as e:
            self.logger.error(f"Erreur lors du traitement des fichiers: {str(e)}")
            raise

    async def handle_dependencies(self, dependencies: List[str]) -> tuple[bool, str]:
        """
        Gère l'installation des dépendances Python.

        Args:
            dependencies: Liste des packages à installer

        Returns:
            (succès, message)
        """
        try:
            # Création ou mise à jour du requirements.txt
            requirements_path = Path(self.project_path) / "requirements.txt"
            requirements_content = "\n".join(dependencies)

            await self.file_manager.create_or_update_file(
                str(requirements_path),
                requirements_content
            )

            # Installation des dépendances
            success, output = await self.file_manager.executor.install_requirements(
                str(requirements_path)
            )

            if not success:
                raise Exception(f"Erreur lors de l'installation des dépendances: {output}")

            return True, "Dépendances installées avec succès"

        except Exception as e:
            self.logger.error(f"Erreur lors de la gestion des dépendances: {str(e)}")
            return False, str(e)

    @staticmethod
    def _convert_role(role_name: str) -> AgentRole:
        """Convertit les noms de rôles en AgentRole enum"""
        role_mapping = {
            # Coordinateurs
            "architecte": AgentRole.COORDINATOR,
            "generateur_idees": AgentRole.EXECUTOR,
            "SuperAgent_BIGDFT": AgentRole.EXECUTOR,
            "convertisseur_dial2tec": AgentRole.EXECUTOR,
            "developpeur": AgentRole.EXECUTOR,
            # Spécialistes
            "ingenieur": AgentRole.SPECIALIST,

            # Exécuteurs
            "frontend_dev": AgentRole.EXECUTOR,
            "backend_dev": AgentRole.EXECUTOR,
            "database_dev": AgentRole.EXECUTOR,
            "designer_interface": AgentRole.EXECUTOR,
            "codeur": AgentRole.EXECUTOR,

            # Critiques
            "testeur": AgentRole.CRITIC,
            "inspecteur": AgentRole.CRITIC
        }

        if role_name not in role_mapping:
            raise ValueError(f"Rôle non supporté: {role_name}")

        return role_mapping[role_name]

    def _normalize_response_keys(self, response: Any) -> Any:
        """Normalise les clés de la réponse"""
        if hasattr(response, '__dict__'):
            data = response.model_dump()
            normalized_data = self._normalize_dict_keys(data)
            return self.pydantic_model(**normalized_data)
        return response

    def _normalize_dict_keys(self, data: Any) -> Any:
        """Normalise récursivement les clés d'un dictionnaire"""
        if isinstance(data, dict):
            return {
                k.lower(): self._normalize_dict_keys(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._normalize_dict_keys(item) for item in data]
        return data

    async def _wait_for_form_submission(self, html_file: str) -> Dict[str, Any]:
        """Attend la soumission du formulaire HTML"""
        try:
            server_task = asyncio.create_task(self.form_server.start())

            # Ouvrir le navigateur
            import webbrowser
            webbrowser.open(f'file://{html_file}')

            # Attendre la soumission
            response = await self.form_server.wait_for_submission()

            # Arrêter le serveur
            await self.form_server.stop()
            await server_task

            if not response:
                raise ValueError("Aucune réponse reçue du formulaire")

            return response

        except Exception as e:
            await self.form_server.stop()
            raise Exception(f"Erreur lors de l'attente du formulaire: {str(e)}")

    async def _create_enhanced_task_message(self, task_data: Dict[str, Any]) -> str:
        """Crée un message enrichi pour une tâche avec tout le contexte nécessaire"""
        try:
            # Récupération du contenu des fichiers existants
            file_contents = {}
            for file_path in task_data.get('files', []):
                if await self.file_manager.file_exists(file_path):
                    content = await self.file_manager.read_file(file_path)
                    file_contents[file_path] = content

            # Construction du contexte enrichi
            enhanced_context = {
                "task": task_data,
                "context": {
                    "project_description": self.project_description if hasattr(self, 'project_description') else None,
                    "existing_files": file_contents,
                    "dependent_tasks": await self._get_dependent_tasks(task_data)
                },
                "environment": {
                    "project_structure": self.file_manager.get_project_structure()
                }
            }

            # Conversion en texte formaté pour le LLM
            # TODO pour l'instant, on ne fournit que le minimum
            #return self._format_context_for_llm(enhanced_context)
            return str(enhanced_context['task'])

        except Exception as e:
            logging.error(f"Erreur lors de la création du message enrichi: {str(e)}")
            raise

    async def _get_dependent_tasks(self, task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Récupère les détails des tâches dont dépend la tâche actuelle"""
        dependent_tasks = []
        if not hasattr(self, 'task_manager'):
            return dependent_tasks

        for dep_id in task_data.get('dependencies', []):
            if task := self.task_manager.get_task(dep_id):
                dependent_tasks.append({
                    "id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "status": task.status
                })
        return dependent_tasks

    def _format_context_for_llm(self, context: Dict[str, Any]) -> str:
        """Formate le contexte enrichi en texte structuré pour le LLM"""
        sections = []

        # Section Tâche
        task = context['task']
        sections.extend([
            "=== DÉTAILS DE LA TÂCHE ===",
            f"ID: {task.get('id', 'N/A')}",
            f"Titre: {task.get('title', 'N/A')}",
            f"Description: {task.get('description', 'N/A')}"
        ])

        # Critères d'acceptation
        if task.get('acceptance_criteria'):
            sections.append("\nCritères d'acceptation:")
            sections.extend([f"- {criterion}" for criterion in task['acceptance_criteria']])

        # Section Fichiers
        if task.get('files'):
            sections.append("\n=== FICHIERS À TRAITER ===")
            sections.extend([f"- {file}" for file in task['files']])

        # Fichiers existants
        existing_files = context.get('context', {}).get('existing_files', {})
        if existing_files:
            sections.append("\n=== CONTENU DES FICHIERS EXISTANTS ===")
            for file_path, content in existing_files.items():
                sections.extend([
                    f"\nFichier: {file_path}",
                    "```",
                    content,
                    "```"
                ])

        # Tâches dépendantes
        dependent_tasks = context.get('context', {}).get('dependent_tasks', [])
        if dependent_tasks:
            sections.append("\n=== TÂCHES DÉPENDANTES ===")
            for dep_task in dependent_tasks:
                sections.extend([
                    f"\nTâche #{dep_task['id']}: {dep_task['title']}",
                    f"Status: {dep_task['status']}"
                ])

        # Structure du projet
        project_structure = context.get('environment', {}).get('project_structure')
        if project_structure:
            sections.append("\n=== STRUCTURE DU PROJET ===")
            sections.append(json.dumps(project_structure, indent=2))

        return "\n".join(sections)



