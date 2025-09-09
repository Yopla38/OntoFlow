"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# src/workflow/orchestrator.py
import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional

from agent.src.components.file_manager import FileManager
from agent.src.components.task_manager import TaskManager, Tache
from agent.src.types.enums import AgentCapability, AgentRole


class WorkflowState(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    WAITING = "waiting"


class WorkflowOrchestrator:
    """
    Orchestrateur de workflow gérant l'exécution des tâches entre différents agents.

    Responsabilités :
    - Gestion du flux de travail
    - Coordination des agents
    - Suivi des états et résultats
    - Gestion des erreurs
    """

    def __init__(
            self,
            workflow_config: Dict[str, Any],
            agents: Dict[str, Any],
            project_description: str,
            task_manager: TaskManager,
            file_manager: FileManager
    ):
        # Configuration de base
        self.workflow_config = workflow_config
        self.agents = agents
        self.project_description = project_description
        self.task_manager = task_manager
        self.file_manager = file_manager

        # État du workflow
        self.current_state = workflow_config["initial_state"]
        self.workflow_state = WorkflowState.WAITING
        self.state_results = {}

        # Tâches en attente d'informations
        self.waiting_tasks = []

        # Configuration du logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    async def execute(self) -> Dict[str, Any]:
        """
        Exécute le workflow complet en suivant la configuration définie.

        Returns:
            Dict[str, Any]: Résultats de l'exécution du workflow
        """
        try:
            self.workflow_state = WorkflowState.RUNNING
            self.logger.info(f"Démarrage du workflow depuis l'état: {self.current_state}")

            while self.current_state not in ["complete", "error"]:
                state_config = self.workflow_config["states"][self.current_state]

                # Vérification des capacités requises pour l'état
                if not await self._verify_state_capabilities(state_config):
                    self.current_state = "error"
                    continue

                # Exécution de l'état
                result = await self._execute_state(state_config)

                # Stockage du résultat
                self.state_results[self.current_state] = result

                # Détermination du prochain état
                next_state = await self._determine_next_state(state_config, result)

                self.logger.info(f"Transition: {self.current_state} -> {next_state}")
                self.current_state = next_state

            self.workflow_state = (
                WorkflowState.COMPLETED
                if self.current_state == "complete"
                else WorkflowState.ERROR
            )

            return self.state_results

        except Exception as e:
            self.logger.error(f"Erreur dans l'exécution du workflow: {str(e)}")
            self.workflow_state = WorkflowState.ERROR
            raise

    async def _execute_state(self, state_config: Dict[str, Any]) -> Any:
        """Exécute un état spécifique du workflow"""
        state_type = state_config["type"]
        self.logger.info(f"Exécution de l'état type: {state_type}")

        try:
            if state_type == "processing":
                return await self._execute_processing_state(state_config)
            elif state_type == "parallel_workflow":
                return await self._execute_parallel_workflow(state_config)
            else:
                raise ValueError(f"Type d'état non supporté: {state_type}")

        except Exception as e:
            self.logger.error(f"Erreur dans _execute_state: {str(e)}")
            raise

    async def _execute_processing_state(self, state_config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            agent_name = state_config["agent"]
            agent = self.agents.get(agent_name)

            if not agent:
                raise ValueError(f"Agent {agent_name} non trouvé")

            # Préparation du contexte pour l'agent
            input_data = await self._prepare_agent_input(state_config)

            # Si nous avons des tâches à traiter
            if "input_from" in state_config:
                tasks = self.state_results.get(state_config["input_from"])
                if tasks:
                    results = []
                    for task in tasks:
                        # Utiliser process_task au lieu de process_message
                        task_result = await agent.process_task(task)
                        results.append(task_result)
                    return {"status": "success", "results": results}

            # Sinon, utiliser process_message comme avant
            result = await agent.process_message(input_data)
            processed_result = await self._process_agent_result(agent, result)
            return processed_result

        except Exception as e:
            logging.error(f"Erreur dans _execute_processing_state: {str(e)}")
            raise

    async def _execute_parallel_workflow(self, state_config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.logger.info("Démarrage du workflow parallèle")
            available_tasks = self.task_manager.get_available_tasks()
            self.logger.info(f"Tâches disponibles : {len(available_tasks)}")

            waiting_tasks = await self._process_waiting_tasks()
            tasks_by_role = self._group_tasks_by_role(available_tasks)

            async_tasks = []
            for role, tasks in tasks_by_role.items():
                self.logger.info(f"Traitement des tâches pour le rôle : {role}")
                if agent := self.agents.get(role):
                    # Créer une coroutine pour le traitement des tâches
                    task = asyncio.create_task(self._process_role_tasks(agent, tasks))
                    async_tasks.append(task)

            if not async_tasks:
                self.logger.warning("Aucune tâche à exécuter")
                return {
                    "status": "success",
                    "results": {},
                    "waiting_tasks": waiting_tasks
                }

            # Attendre tous les résultats
            task_results = await asyncio.gather(*async_tasks, return_exceptions=True)
            self.logger.info(f"Résultats obtenus : {len(task_results)}")

            # Traiter les résultats
            return await self._process_parallel_results(task_results, waiting_tasks)

        except Exception as e:
            self.logger.error(f"Erreur dans _execute_parallel_workflow: {str(e)}")
            raise

    async def _process_role_tasks(self, agent: Any, tasks: List[Tache]) -> Dict[str, Any]:
        results = {}
        new_waiting_tasks = []
        for task in tasks:
            try:
                # Traitement de la tâche actuelle
                task_dict = task.__dict__
                result = await agent.process_task(task_dict)

                # Mettez à jour l'état de la tâche si elle est complétée
                if result.get("status") == "success":
                    self.task_manager.update_task_status(task.id, "completed")
                    # Stocker le résultat Pydantic original
                    results[task.id] = result
                    # Stocker dans la base de connaissances
                    await agent.knowledge_base.store_pydantic_result(agent.name, result)
                elif result.get("status") == "waiting_for_info":
                    new_waiting_tasks.append({
                        "task": task,
                        "missing_info": result.get("missing_information", []),
                        "agent_name": agent.name
                    })
                results[task.id] = result
            except Exception as e:
                logging.error(f"Erreur lors du traitement de la tâche {task.id}: {str(e)}")
                results[task.id] = {"status": "error", "error": str(e)}

        return {"completed_tasks": results, "waiting_tasks": new_waiting_tasks}

    async def _process_waiting_tasks(self) -> List[Dict[str, Any]]:
        """Tente de réexécuter les tâches en attente et retourne celles toujours bloquées"""
        still_waiting = []
        completed_tasks = []

        async_tasks = []
        for waiting_task in self.waiting_tasks:
            agent = self.agents.get(waiting_task["agent_name"])
            if agent:
                async_tasks.append(
                    self._retry_waiting_task(agent, waiting_task)
                )

        if async_tasks:
            retry_results = await asyncio.gather(*async_tasks, return_exceptions=True)

            for result in retry_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Erreur lors de la reprise d'une tâche: {str(result)}")
                    continue

                if result["status"] == "waiting_for_info":
                    still_waiting.append(result["task_info"])
                else:
                    completed_tasks.append(result)

        self.waiting_tasks = still_waiting
        return completed_tasks

    async def _retry_waiting_task(self, agent: Any, waiting_task: Dict[str, Any]) -> Dict[str, Any]:
        """Tente de réexécuter une tâche en attente"""
        try:
            result = await agent.process_task(waiting_task["task"])
            return {
                "status": result.get("status"),
                "task_info": waiting_task,
                "result": result
            }
        except Exception as e:
            raise Exception(f"Erreur lors de la reprise de la tâche {waiting_task['task'].id}: {str(e)}")

    async def _process_parallel_results(self, task_results: List[Any], waiting_results: List[Dict[str, Any]]) -> Dict[
        str, Any]:
        try:
            self.logger.info("Traitement des résultats parallèles")
            processed_results = {
                "status": "success", "results": {}, "waiting_tasks": [],
                "errors": [], "completed_tasks": 0, "failed_tasks": 0
            }
            for result in task_results:
                if isinstance(result, dict):
                    # Détection correcte du type résultat et accès à son contenu
                    completed_tasks = result.get("completed_tasks", {})
                    for task_id, task_result in completed_tasks.items():
                        processed_results["results"][task_id] = task_result
                        if task_result.get("status") == "success":
                            processed_results["completed_tasks"] += 1
                        else:
                            processed_results["failed_tasks"] += 1
                    processed_results["waiting_tasks"].extend(result.get("waiting_tasks", []))
                else:
                    self.logger.error(f"Type de résultat inattendu: {type(result)}")
            if processed_results["failed_tasks"] > 0:
                processed_results["status"] = "partial"
            if len(processed_results["results"]) == 0:
                processed_results["status"] = "error"
            self.logger.info(f"Résultats traités: {processed_results['status']}")
            return processed_results
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement des résultats parallèles: {str(e)}")
            raise

    async def old_process_parallel_results(self, task_results: List[Any], waiting_results: List[Dict[str, Any]]) -> Dict[
        str, Any]:
        try:
            self.logger.info("Traitement des résultats parallèles")
            processed_results = {
                "status": "success",
                "results": {},
                "waiting_tasks": [],
                "errors": [],
                "completed_tasks": 0,
                "failed_tasks": 0
            }

            for result in task_results:
                self.logger.info(f"Traitement du résultat : {type(result)}")

                if isinstance(result, Exception):
                    processed_results["status"] = "partial"
                    processed_results["errors"].append(str(result))
                    processed_results["failed_tasks"] += 1
                    continue

                if not isinstance(result, dict):
                    self.logger.error(f"Type de résultat inattendu : {type(result)}")
                    continue

                # Traiter les tâches complétées
                completed_tasks = result.get("completed_tasks", {})
                for task_id, task_result in completed_tasks.items():
                    processed_results["results"][task_id] = task_result
                    if task_result.get("status") == "success":
                        processed_results["completed_tasks"] += 1
                    else:
                        processed_results["failed_tasks"] += 1

                # Ajouter les tâches en attente
                processed_results["waiting_tasks"].extend(result.get("waiting_tasks", []))

            # Mise à jour du statut final
            if processed_results["failed_tasks"] > 0:
                processed_results["status"] = "partial"
            if len(processed_results["results"]) == 0:
                processed_results["status"] = "error"

            self.logger.info(f"Résultats traités : {processed_results['status']}")
            return processed_results

        except Exception as e:
            self.logger.error(f"Erreur lors du traitement des résultats parallèles: {str(e)}")
            raise
    def _group_tasks_by_role(self, tasks: List[Tache]) -> Dict[str, List[Tache]]:
        """Groupe les tâches par rôle assigné"""
        tasks_by_role = {}
        for task in tasks:
            role = task.assigned_role
            if role not in tasks_by_role:
                tasks_by_role[role] = []
            tasks_by_role[role].append(task)
        return tasks_by_role

    async def _prepare_agent_input(self, state_config: Dict[str, Any]) -> str:
        """Prépare les données d'entrée pour un agent"""
        if "input_from" in state_config:
            source = state_config["input_from"]
            return self.state_results.get(source, "")
        return self.project_description

    async def _process_agent_result(self, agent: Any, result: Any) -> Dict[str, Any]:
        """Traite le résultat d'un agent selon son rôle"""
        try:
            logging.info(f"Processing result from agent {agent.name}")

            # Stocker le résultat dans la base de connaissances
            await agent.knowledge_base.store_pydantic_result(agent.name, result)
            logging.info("Result stored in knowledge base")

            if hasattr(result, 'model_dump'):
                result_dict = result.model_dump()
            else:
                result_dict = result

            # Traitement des tâches si présentes
            if isinstance(result_dict, dict) and 'tasks' in result_dict:
                await self.task_manager.add_tasks(result_dict['tasks'], agent)

            return result_dict

        except Exception as e:
            logging.error(f"Error in _process_agent_result: {str(e)}")
            raise

    async def _verify_state_capabilities(self, state_config: Dict[str, Any]) -> bool:
        """Vérifie que l'agent a les capacités requises pour l'état"""
        if "required_capabilities" not in state_config:
            return True

        agent_name = state_config["agent"]
        agent = self.agents.get(agent_name)

        if not agent:
            return False

        required_capabilities = state_config["required_capabilities"]
        return all(agent.can_perform_action(cap) for cap in required_capabilities)

    async def _determine_next_state(self, state_config: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Détermine l'état suivant en fonction du résultat"""
        transitions = state_config["transitions"]
        state_type = state_config["type"]

        try:
            if state_type == "parallel_workflow":
                return await self._determine_parallel_workflow_next_state(result, transitions)
            elif self.current_state == "analysis":
                return await self._determine_analysis_next_state(result, transitions)
            else:
                return transitions.get(result.get("status", "error"), "error")

        except Exception as e:
            self.logger.error(f"Erreur lors de la détermination du prochain état: {str(e)}")
            return transitions.get("error", "error")

    async def _determine_analysis_next_state(self, result: Dict[str, Any], transitions: Dict[str, str]) -> str:
        """Détermine l'état suivant après l'analyse"""
        tasks = result.get("tasks", [])

        # Filtrer les tâches nécessitant une décomposition
        tasks_needing_decomposition = [
            task for task in tasks
            if task.get("needs_decomposition", False)
        ]

        # Filtrer les tâches prêtes pour le développement
        tasks_for_development = [
            task for task in tasks
            if not task.get("needs_decomposition", False)
        ]

        # Stocker les tâches dans les résultats d'état
        if tasks_needing_decomposition:
            self.state_results["tasks_for_engineering"] = tasks_needing_decomposition
            self.state_results["tasks_for_development"] = tasks_for_development
            self.logger.info(f"Trouvé {len(tasks_needing_decomposition)} tâches pour l'ingénieur")
            return "engineering"
        else:
            self.state_results["tasks_for_development"] = tasks
            self.logger.info("Toutes les tâches sont prêtes pour le développement")
            return "development"

    async def _determine_parallel_workflow_next_state(self, result: Dict[str, Any], transitions: Dict[str, str]) -> str:
        """Détermine l'état suivant pour un workflow parallèle"""
        all_tasks_completed = all(
            task.status == "completed"
            for task in self.task_manager.get_all_tasks()
        )

        if result.get("status") == "error":
            return transitions["error"]
        elif all_tasks_completed:
            return transitions["success"]
        else:
            return self.current_state
