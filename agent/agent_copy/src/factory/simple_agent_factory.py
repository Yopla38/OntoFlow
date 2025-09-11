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
import os
from asyncio import Queue
from socket import SocketIO
from typing import Dict, Any, Optional

from agent.KEY import CLAUDE_KEY, OPENAI_KEY, LOCAL_MODEL_PATH
from agent.src.components.file_manager import FileManager
from agent.src.components.task_manager import TaskManager
from agent.src.factory.provider_factory import MemoryProviderFactory, LLMProviderFactory
from agent.src.interface.html_manager import HTMLManager
from agent.src.models.pydantic_models import ModelGenerator
from agent.src.types.interfaces import MemoryProvider

from typing import Optional, Dict, Any, List
import asyncio
from queue import Queue
from flask_socketio import SocketIO
from agent.src.agents.simple_agent import SimpleAgent
from agent.src.factory.provider_factory import LLMProviderFactory


class SimpleAgentFactory:
    @staticmethod
    def create_agent_from_role(
            role_config: Dict[str, Any],
            project_path: str,
            project_name: str,
            workspace_root: str,
            file_manager: Optional[FileManager] = None,
            task_manager: Optional[TaskManager] = None,
            memory_provider: Optional[MemoryProvider] = None,
            streaming: Optional[bool] = False
    ) -> SimpleAgent:
        """
        Crée un agent simple à partir d'une configuration de rôle
        """
        # Création du modèle Pydantic si spécifié
        pydantic_model = None
        if "pydantic_model" in role_config:
            # todo try ici ???
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
            structured_response_for_local_model=role_config["structured_response_for_local_model"] if "structured_response_for_local_model" in role_config else ""
        )

        # Utiliser le memory_provider fourni ou en créer un nouveau si non fourni
        if memory_provider is None:
            memory_config = {
                "file_path": f"{project_path}/memory/shared_memory.json"
            }
            memory_provider = MemoryProviderFactory.create_provider(
                provider_type="local",
                config=memory_config
            )

        # Création de l'agent avec tous les composants
        return SimpleAgent(
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
            streaming=streaming
        )



"""
# Configuration
communication_config = {
    'provider_type': 'openai',
    'model': 'gpt-3.5-turbo',
    'api_key': 'your_api_key',
    'system_prompt': 'Custom prompt for communication'
}

# Création de l'agent
agent = CommunicativeAgent(
    # ... autres paramètres du SimpleAgent ...
    communication_config=communication_config
)

# Démarrage du serveur de communication
agent.start_communication_server()

# Dans un autre thread pour les tâches en arrière-plan
asyncio.run(agent.process_background_tasks())
"""