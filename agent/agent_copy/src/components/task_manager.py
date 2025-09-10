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
from datetime import datetime
from typing import List, Dict, Any, Optional

from agent.src.agent import Agent
from agent.src.types.enums import AgentCapability


class Revision:
    def __init__(self, content, timestamp=None, author=None):
        self.content = content
        self.timestamp = timestamp if timestamp else datetime.now()
        self.author = author


class Tache:
    def __init__(self,
                 id: int,
                 title: str,
                 description: str,
                 assigned_role: str,
                 priority: int,
                 needs_decomposition: bool = False,
                 dependencies: List[int] = None,
                 files: List[str] = None,
                 acceptance_criteria: List[str] = None,
                 status: str = "pending",
                 type: str = None):
        self.id = id
        self.title = title
        self.description = description
        self.assigned_role = assigned_role
        self.priority = priority
        self.needs_decomposition = needs_decomposition
        self.dependencies = dependencies or []
        self.files = files or []
        self.acceptance_criteria = acceptance_criteria or []
        self.status = status
        self.type = type  # 'execution', 'decomposition', 'validation', etc.


class TaskManager:
    def __init__(self):
        self.tasks: Dict[int, Tache] = {}
        self.current_id = 0

    async def add_task(self, task_data: Dict[str, Any], agent: 'Agent') -> Optional[Tache]:
        """Ajoute une nouvelle tâche avec vérification des capacités"""
        try:
            if not agent.can_perform_action(AgentCapability.CREATE_TASKS):
                logging.warning(f"Agent {agent.name} non autorisé à créer des tâches")
                return None

            self.current_id += 1
            task = Tache(
                id=self.current_id,
                title=task_data["title"],
                description=task_data["description"],
                assigned_role=task_data["assigned_role"],
                priority=task_data.get("priority", 1),
                needs_decomposition=task_data.get("needs_decomposition", False),
                dependencies=task_data.get("dependencies", []),
                files=task_data.get("files", []),
                acceptance_criteria=task_data.get("acceptance_criteria", []),
                type=task_data.get("type", "execution")
            )

            self.tasks[task.id] = task
            return task

        except Exception as e:
            logging.error(f"Erreur lors de la création de la tâche: {str(e)}")
            return None

    async def add_tasks(self, tasks_data: List[Dict[str, Any]], agent: 'Agent'):
        """Ajoute plusieurs tâches"""
        added_tasks = []
        for task_data in tasks_data:
            task = await self.add_task(task_data, agent)
            if task:
                added_tasks.append(task)
        return added_tasks

    def get_task(self, task_id: int) -> Optional[Tache]:
        """Récupère une tâche par son ID"""
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> List[Tache]:
        """Récupère toutes les tâches"""
        return list(self.tasks.values())

    def get_available_tasks(self) -> List[Tache]:
        """Retourne les tâches dont toutes les dépendances sont satisfaites"""
        available_tasks = []
        for task in self.tasks.values():
            if task.status == "pending" and self._are_dependencies_met(task):
                available_tasks.append(task)
        return available_tasks

    def get_tasks_by_role(self, role: str) -> List[Tache]:
        """Récupère toutes les tâches assignées à un rôle spécifique"""
        return [task for task in self.tasks.values() if task.assigned_role == role]

    def get_tasks_by_status(self, status: str) -> List[Tache]:
        """Récupère toutes les tâches ayant un statut spécifique"""
        return [task for task in self.tasks.values() if task.status == status]

    def update_task_status(self, task_id: int, status: str):
        """Met à jour le statut d'une tâche"""
        if task := self.tasks.get(task_id):
            task.status = status
        else:
            raise ValueError(f"Tâche {task_id} non trouvée")

    def _are_dependencies_met(self, task: Tache) -> bool:
        """Vérifie si toutes les dépendances d'une tâche sont complétées"""
        return all(
            self.tasks[dep_id].status == "completed"
            for dep_id in task.dependencies
            if dep_id in self.tasks
        )

    def get_dependency_chain(self, task_id: int) -> List[int]:
        """Récupère la chaîne de dépendances pour une tâche"""
        chain = []
        if task := self.tasks.get(task_id):
            chain.append(task_id)
            for dep_id in task.dependencies:
                chain.extend(self.get_dependency_chain(dep_id))
        return chain

    def validate_dependencies(self) -> bool:
        """Vérifie qu'il n'y a pas de dépendances circulaires"""
        for task_id in self.tasks:
            visited = set()
            stack = [task_id]
            while stack:
                current = stack.pop()
                if current in visited:
                    return False
                visited.add(current)
                if task := self.tasks.get(current):
                    stack.extend(task.dependencies)
        return True