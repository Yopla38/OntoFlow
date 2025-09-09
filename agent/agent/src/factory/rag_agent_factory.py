"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# src/factory/rag_agent_factory.py
import os
from typing import Dict, Any, Optional, Set
from pathlib import Path

from agent.KEY import CLAUDE_KEY, OPENAI_KEY, LOCAL_MODEL_PATH
from agent.src.agents.agent_rag import RagAgent
from agent.src.components.file_manager import FileManager
from agent.src.components.task_manager import TaskManager
from agent.src.factory.provider_factory import MemoryProviderFactory, LLMProviderFactory
from agent.src.interface.html_manager import HTMLManager
from agent.src.models.pydantic_models import ModelGenerator
from agent.src.types.interfaces import MemoryProvider


class RagAgentFactory:
    SUPPORTED_ROLES = ["agent_rag"]

    @staticmethod
    def create_agent_from_role(
            role_config: Dict[str, Any],
            project_path: str,
            project_name: str,
            workspace_root: str,
            file_manager: Optional[FileManager] = None,
            task_manager: Optional[TaskManager] = None,
            memory_provider: Optional[MemoryProvider] = None,
            exclude_dirs: Optional[Set[str]] = None,
            index_directory: Optional[str] = None  # Répertoire à indexer
    ) -> RagAgent:
        """
        Crée un agent RAG à partir d'une configuration de rôle
        """
        # Création du modèle Pydantic si spécifié
        pydantic_model = None
        if "pydantic_model" in role_config:
            pydantic_model = ModelGenerator.create_model(
                role_config["name"],
                role_config["pydantic_model"]
            )

        # Création ou réutilisation du FileManager
        if file_manager is None:
            file_manager = FileManager(project_path)

        # Création ou réutilisation du TaskManager
        if task_manager is None:
            task_manager = TaskManager()

        # Création du HTML Manager
        html_manager = HTMLManager(workspace_root)

        # Détection du type de provider et modèle
        if "gpt" in role_config["model"]:
            provider_type = "openai"
            model = role_config["model"]
            key = OPENAI_KEY
        elif "claude" in role_config["model"]:
            provider_type = "anthropic"
            model = role_config["model"]
            key = CLAUDE_KEY
        else:
            provider_type = "local"
            model = LOCAL_MODEL_PATH
            key = ""

        # Création du LLM provider
        llm_provider = LLMProviderFactory.create_provider(
            provider_type=provider_type,
            model=model,
            api_key=key,
            system_prompt=role_config["prompt"],
            structured_response_for_local_model=role_config.get(
                "structured_response_for_local_model", ""
            )
        )

        # Utiliser le memory_provider fourni ou en créer un nouveau
        if memory_provider is None:
            memory_config = {
                "file_path": f"{project_path}/memory/shared_memory.json"
            }
            memory_provider = MemoryProviderFactory.create_provider(
                provider_type="local",
                config=memory_config
            )

        # Définir les répertoires à exclure
        if exclude_dirs is None:
            exclude_dirs = {"rag_storage", ".agent_workspace", "auto", "log", "__pycache__"}

        # Création de l'agent avec tous les composants
        return RagAgent(
            name=role_config["name"],
            role_name=role_config["name"],
            llm_provider=llm_provider,
            memory_provider=memory_provider,
            system_prompt=role_config["prompt"],
            project_path=project_path,
            html_manager=html_manager,
            project_name=project_name,
            response_format=role_config.get("response_format", ""),
            file_manager=file_manager,
            task_manager=task_manager,
            pydantic_model=pydantic_model,
            exclude_dirs=exclude_dirs,
            index_directory=index_directory
        )