"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# src/types/enums.py
from enum import Enum, auto
from typing import FrozenSet, Dict


class AgentCapability(Enum):
    CREATE_TASKS = auto()           # Peut créer des tâches principales
    ASSIGN_TASKS = auto()           # Peut assigner des tâches
    DECOMPOSE_TASKS = auto()        # Peut décomposer des tâches
    EXECUTE_TASKS = auto()          # Peut exécuter des tâches
    VALIDATE_TASKS = auto()         # Peut valider des tâches
    MODIFY_FILES = auto()           # Peut modifier des fichiers
    CREATE_CORRECTIONS = auto()      # Peut créer des tâches de correction
    DEFINE_ARCHITECTURE = auto()     # Peut définir l'architecture
    ASSIGN_TO_ALL_ROLES = auto()     # Peut assigner à tous les rôles
    ASSIGN_TO_EXECUTORS = auto()     # Peut assigner uniquement aux executors
    COMMUNICATE = auto()


ROLE_CAPABILITIES: Dict[str, FrozenSet[AgentCapability]] = {
    "coordinator": frozenset({
        AgentCapability.CREATE_TASKS,
        AgentCapability.ASSIGN_TASKS,
        AgentCapability.DEFINE_ARCHITECTURE,
        AgentCapability.ASSIGN_TO_ALL_ROLES,
        AgentCapability.COMMUNICATE
    }),
    "specialist": frozenset({
        AgentCapability.DECOMPOSE_TASKS,
        AgentCapability.CREATE_TASKS,
        AgentCapability.ASSIGN_TASKS,
        AgentCapability.ASSIGN_TO_EXECUTORS
    }),
    "executor": frozenset({
        AgentCapability.EXECUTE_TASKS,
        AgentCapability.MODIFY_FILES,
        AgentCapability.COMMUNICATE

    }),
    "critic": frozenset({
        AgentCapability.VALIDATE_TASKS,
        AgentCapability.CREATE_CORRECTIONS
    })
}


class AgentRole(Enum):
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    EXECUTOR = "executor"
    CRITIC = "critic"

    def __init__(self, value: str):
        self._value_ = value
        self._capabilities: FrozenSet[AgentCapability] = ROLE_CAPABILITIES.get(value, frozenset())

    @property
    def capabilities(self) -> FrozenSet[AgentCapability]:
        return self._capabilities


class AgentState(Enum):
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    FINISHED = "finished"
