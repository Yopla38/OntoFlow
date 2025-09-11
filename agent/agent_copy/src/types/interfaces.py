"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# src/types/interfaces.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List

#from agent.src.agent import Agent


class ProjectManager:
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.workspace_dir = None
        self.projects_dir = None
        self.templates_dir = None
        self.venvs_dir = None
        self.active_agents: Dict[str, 'Agent'] = {}
        self.current_workflow: List[str] = []


@dataclass
class Message:
    """Classe représentant un message dans l'historique"""
    role: str  # 'user' ou 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


class ConversationHistory:
    """Classe gérant l'historique des conversations"""

    def __init__(self, max_messages: int = 10):
        self.messages: List[Message] = []
        self.max_messages = max_messages

    def add_message(self, role: str, content: str):
        """Ajoute un message à l'historique"""
        self.messages.append(Message(role=role, content=content))
        # Garde uniquement les n derniers messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_context_string(self, format_type: str = "anthropic") -> str:
        """Retourne l'historique formaté selon le provider"""
        if format_type == "anthropic":
            return "\n".join([
                f"{msg.role}: {msg.content}"
                for msg in self.messages
            ])
        elif format_type == "deepseek":
            return "\n".join([
                f"<｜{msg.role.capitalize()}｜>{msg.content}"
                for msg in self.messages
            ])

    def clear(self):
        """Efface l'historique"""
        self.messages.clear()


class LLMProvider(ABC):
    def __init__(self):
        self.history = ConversationHistory()

    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    async def generate_response_for_humain(self, messages: List[Dict[str, str]], stream=None) -> Dict[str, Any]:
        pass


class MemoryProvider(ABC):
    @abstractmethod
    async def add(self, data: Dict[str, Any]):
        pass

    @abstractmethod
    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def clear(self):
        pass


@dataclass
class EnrichmentPath:
    """Définit un chemin d'accès à une information dans un modèle Pydantic"""
    source_role: str          # Rôle source de l'information
    pydantic_path: List[str]  # Chemin dans le modèle Pydantic
    category: str            # Catégorie dans la base de connaissances
    description: str         # Description de l'information
