"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# src/agent.py
import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Union
import uuid
from datetime import datetime

import socketio

from agent.src.components.enrichment_config import EnrichmentConfig, ROLE_ENRICHMENTS
from agent.src.factory.provider_factory import LLMProviderFactory
from agent.src.types.enums import AgentRole, AgentState, AgentCapability
from agent.src.types.interfaces import LLMProvider, MemoryProvider
from agent.src.components.tool import Tool
from agent.src.components.knowledge_base import KnowledgeBase


class Agent:
    def __init__(
            self,
            name: str,
            role: AgentRole,
            llm_provider: LLMProvider,
            memory_provider: MemoryProvider,
            system_prompt: str,
            memory_size: int = 1000,
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.role = role
        self.llm_provider = llm_provider
        self.memory_provider = memory_provider
        self.system_prompt = system_prompt

        # État et mémoire
        self.state = AgentState.IDLE
        self.memory_size = memory_size
        self.conversation_history = []

        # Outils et base de connaissances
        self.tools: Dict[str, Tool] = {}
        self.knowledge_base = KnowledgeBase(memory_provider)

        # Pipeline d'enrichissement de texte
        self.message_enrichment_pipeline = MessageEnrichmentPipeline()

        # Relations avec d'autres agents
        self.collaborators: Dict[str, 'Agent'] = {}
        self.superior: Optional['Agent'] = None

        # Capacités et contraintes
        self.capabilities: Set[AgentCapability] = set(role.capabilities)  # Utiliser un set d'AgentCapability
        self.constraints: Dict[str, Any] = {}

        # Métriques de performance
        self.performance_metrics = {
            "tasks_completed": 0,
            "success_rate": 0.0,
            "average_response_time": 0.0
        }

        # Configuration de la communication
        self.communication_llm: Optional[LLMProvider] = None
        self.communication_lock = asyncio.Lock()  # Pour la thread-safety
        self.communication_task: Optional[asyncio.Task] = None
        self.communication_history: List[Dict[str, str]] = []
        self.socketio = None

        # Gestion des appels de fonctions
        self.fc_buffer = {"name": None, "arguments": ""}

    def setup_communication(self, config: Dict[str, Any], socketio=None) -> None:
        """Configure la capacité de communication de l'agent"""
        if not self.can_perform_action(AgentCapability.COMMUNICATE):
            raise PermissionError(f"Agent {self.name} n'a pas la capacité de communiquer")

        self.socketio = socketio
        self.communication_llm = LLMProviderFactory.create_provider(
            provider_type=config.get('provider_type', 'openai'),
            model=config.get('model', 'gpt-4o-2024-08-06'),
            api_key=config.get('api_key'),
            system_prompt=config.get('system_prompt',
                                     f"Assistant de communication pour l'agent {self.name}")
        )

        self.is_running = True

    async def _run_communication_loop(self):
        """Boucle de communication qui fonctionne en parallèle des tâches principales"""
        while True:
            try:
                if not hasattr(self, 'is_running') or not self.is_running:
                    break
                await asyncio.sleep(0.1)
            except Exception as e:
                logging.error(f"Erreur dans la boucle de communication: {e}")
            except asyncio.CancelledError:
                break

    def streaming_decompose_text(self, completion):
        """Décompose le texte streamé et gère les function calls"""
        content = ""
        fc_bool = False

        try:
            if hasattr(completion, 'choices') and completion.choices:
                delta = completion.choices[0].delta

                if delta.content:
                    content = delta.content

                if delta.function_call:
                    if not hasattr(self, "fc_buffer"):
                        self.fc_buffer = {"name": None, "arguments": ""}

                    if delta.function_call.name and not self.fc_buffer["name"]:
                        self.fc_buffer["name"] = delta.function_call.name

                    if delta.function_call.arguments:
                        self.fc_buffer["arguments"] += delta.function_call.arguments

                    fc_bool = True

            return content, fc_bool

        except Exception as e:
            logging.error(f"Error in streaming_decompose_text: {e}")
            return '', False

    async def communicate(self, message: str) -> None | str:
        if not self.communication_llm or not self.socketio:
            return "Communication non configurée pour cet agent"

        try:
            # Ajouter le message utilisateur à l'historique
            self.communication_history.append({"role": "user", "content": message})

            # Préparer le contexte
            context = {
                "agent_status": self.state.value,
                "current_task": "Aucune" if self.state == AgentState.IDLE else "En cours",
                "recent_memory": "",
                "agent_role": self.role.value,
                "agent_name": self.name
            }

            # Créer le message système avec le contexte
            system_message = {
                "role": "system",
                "content": f"""
    {self.communication_llm.system_prompt}

    Contexte actuel:
    {json.dumps(context, indent=2)}
    """
            }

            # Préparer tous les messages pour le LLM
            messages = [system_message] + self.communication_history

            # Émettre le début de la réponse
            self.socketio.emit('message', {'type': 'start', 'content': f"{self.name}: "})

            # Générer la réponse en streaming
            response_stream = await self.communication_llm.generate_response(
                messages=messages,
                stream=True
            )

            current_message = ""
            async for chunk in response_stream:
                if hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        current_message += content
                        self.socketio.emit('message', {'type': 'stream', 'content': content})

                elif hasattr(chunk.choices[0].delta, 'function_call'):
                    fc_data = chunk.choices[0].delta.function_call
                    if hasattr(fc_data, 'name'):
                        self.fc_buffer = {"name": fc_data.name, "arguments": ""}
                    if hasattr(fc_data, 'arguments'):
                        self.fc_buffer["arguments"] += fc_data.arguments

            if current_message:
                self.communication_history.append({"role": "assistant", "content": current_message})
                await self.learn_from_interaction(
                    {"user_message": message, "agent_response": current_message},
                    category="communications"
                )

            # Signaler la fin de la réponse
            self.socketio.emit('message', {'type': 'end'})

        except Exception as e:
            logging.error(f"Erreur lors de la communication: {e}")
            self.socketio.emit('error', str(e))

    def cleanup(self) -> None:
        """Nettoie les ressources de l'agent"""
        self.is_running = False
        if hasattr(self, 'communication_task') and self.communication_task:
            self.communication_task.cancel()

    async def handle_function_call(self, response: Dict) -> str:
        """Gère l'exécution des function calls"""
        try:
            function_call = response["choices"][0]["message"]["function_call"]
            function_name = function_call["name"]
            arguments = json.loads(function_call["arguments"])

            # Vérifier si la fonction existe dans les outils de l'agent
            if function_name in self.tools:
                tool = self.tools[function_name]
                result = await tool.function(**arguments)
                return result
            else:
                return f"Fonction {function_name} non trouvée"

        except Exception as e:
            logging.error(f"Erreur dans handle_function_call: {e}")
            return f"Erreur lors de l'exécution de la fonction: {str(e)}"

    def add_tool(self, tool: Tool):
        """Ajoute un outil à l'agent"""
        self.tools[tool.name] = tool

    async def use_tool(self, tool_name: str, **params) -> Any:
        """Utilise un outil spécifique"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")

        tool = self.tools[tool_name]
        return await tool.function(**params)

    async def process_message(self, message: Union[str, List[Dict[str, str]]], pydantic_model=None, streaming: bool=False) -> str:
        """Traitement des tâches principales, indépendant de la communication"""
        self.state = AgentState.WORKING
        try:
            response = await self.llm_provider.generate_response(message, pydantic_model=pydantic_model, stream=streaming)
            await self.knowledge_base.store_pydantic_result(self.name, response)
            return response
        finally:
            self.state = AgentState.IDLE

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Traite une tâche selon les capacités de l'agent"""
        try:
            logging.info(f"Starting process_task for task: {task.get('id')}")

            # Vérification des capacités nécessaires
            if task.get("needs_decomposition"):
                logging.info("Task needs decomposition, checking DECOMPOSE_TASKS capability")
                if not self.can_perform_action(AgentCapability.DECOMPOSE_TASKS):
                    logging.error("Agent not authorized to decompose tasks")
                    raise PermissionError(f"Agent {self.name} non autorisé à décomposer des tâches")
                logging.info("Agent authorized to decompose tasks")
            elif not self.can_perform_action(AgentCapability.EXECUTE_TASKS):
                logging.error("Agent not authorized to execute tasks")
                raise PermissionError(f"Agent {self.name} non autorisé à exécuter des tâches")

            logging.info("Checking required information")
            # Vérifier si toutes les informations nécessaires sont disponibles
            missing_info = await self._check_required_information()
            if missing_info:
                logging.info(f"Missing information: {missing_info}")
                # TODO trouver un mécanisme ici car enrichment_config, développeur depend de generateur_idée et de convertisseur_dial2tec
                # todo du coup je supprime le retour attendant l'information . Comment faire !!!
                """
                return {
                    "status": "waiting_for_info",
                    "missing_information": missing_info,
                    "task_id": task.get("id")
                }
                """
            logging.info("Creating enriched message")

            # Toutes les informations sont disponibles, continuer le traitement normal
            # Création du message enrichi via le pipeline
            task_message = await self.message_enrichment_pipeline.enrich_message(agent=self, task=task)

            # Traitement de la tâche
            result = await self.process_message(task_message)

            # Stockage automatique du résultat dans la base de connaissances
            await self.knowledge_base.store_pydantic_result(self.name, result)

            return result

        except Exception as e:
            logging.error(f"Erreur lors du traitement de la tâche: {str(e)}")
            raise

    async def _check_required_information(self) -> List[Dict[str, str]]:
        """Vérifie si toutes les informations nécessaires sont disponibles"""
        missing_info = []

        # Récupérer les enrichissements requis pour ce rôle
        required_enrichments = ROLE_ENRICHMENTS.get(self.name, [])

        logging.info(f"Checking required information for role {self.name}")
        logging.info(f"Required enrichments: {required_enrichments}")

        for enrichment in required_enrichments:
            # Vérifier si l'information existe en utilisant le chemin Pydantic
            knowledge = await self.knowledge_base.query_knowledge(
                query=enrichment.pydantic_path,  # Utiliser le chemin Pydantic
                category=enrichment.source_role  # Utiliser le rôle source
            )

            logging.info(
                f"Checking enrichment path {enrichment.pydantic_path} from {enrichment.source_role}: {knowledge}")

            if not knowledge:
                missing_info.append({
                    "source_role": enrichment.source_role,
                    "category": enrichment.category,
                    "description": enrichment.description
                })

        return missing_info

    def can_perform_action(self, capability: AgentCapability) -> bool:
        """Vérifie si l'agent a la capacité demandée"""
        logging.info(f"Checking capability {capability} for agent {self.name}")
        logging.info(f"Agent role: {self.role}")
        logging.info(f"Agent capabilities: {self.role.capabilities}")
        result = capability in self.capabilities
        logging.info(f"Can perform action: {result}")
        return result

    async def validate_and_perform_action(self, capability: AgentCapability, action_name: str) -> bool:
        """Valide et enregistre une action"""
        if not self.can_perform_action(capability):
            logging.warning(f"Agent {self.name} a tenté d'effectuer {action_name} sans autorisation")
            return False
        logging.info(f"Agent {self.name} effectue {action_name}")
        return True

    def _build_context(self, message: str, knowledge: List[Dict], memory: List[Dict]) -> str:
        context = f"System: {self.system_prompt}\n\n"

        if knowledge:
            context += "Relevant Knowledge:\n"
            for k in knowledge:
                context += f"- {k['content']}\n"

        if memory:
            context += "\nRelevant Memory:\n"
            for m in memory:
                context += f"- {m['content']}\n"

        if self.tools:
            context += "\nAvailable Tools:\n"
            for tool in self.tools.values():
                context += f"- {tool.name}: {tool.description}\n"

        context += f"\nUser: {message}"
        return context

    async def learn_from_interaction(self, interaction: Dict[str, str], category: str):
        """Apprend de l'interaction et met à jour la base de connaissances"""
        await self.knowledge_base.add_knowledge(category, str(interaction))

    async def _update_memory(self, input_text: str, response: str):
        """Met à jour la mémoire de l'agent"""
        await self.memory_provider.add({
            "input": input_text,
            "response": response,
            "timestamp": datetime.now()
        })

    def add_collaborator(self, agent: 'Agent'):
        """Ajoute un agent collaborateur"""
        self.collaborators[agent.id] = agent

    def remove_collaborator(self, agent_id: str):
        """Supprime un agent collaborateur"""
        if agent_id in self.collaborators:
            del self.collaborators[agent_id]

    def set_superior(self, agent: 'Agent'):
        """Définit l'agent supérieur"""
        self.superior = agent

    def get_status_report(self) -> Dict[str, Any]:
        """Génère un rapport d'état de l'agent"""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "state": self.state.value,
            "tools": list(self.tools.keys()),
            "performance_metrics": self.performance_metrics,
            "capabilities": self.capabilities,
            "constraints": self.constraints
        }


class MessageEnrichmentPipeline:
    """Pipeline générique d'enrichissement des messages"""

    async def enrich_message(self, agent: 'Agent', task: Dict[str, Any]) -> str:
        """
        Enrichit le message avec les informations pertinentes de la base de connaissances.
        """
        enriched_sections = []

        # Informations de base de la tâche
        enriched_sections.extend(self._get_task_base_info(task))

        if task.get("has_rag_data") and task.get("rag_context_key"):
            enriched_sections.append("\n=== CONTEXTE RAG AUTOMATIQUE ===")

            # Récupérer les données RAG via la clé
            try:
                rag_data = await agent.knowledge_base.query_knowledge(
                    query="",
                    category=task["rag_context_key"]
                )
                if rag_data:
                    enriched_sections.append(str(rag_data))
            except Exception as e:
                enriched_sections.append(f"Erreur récupération contexte RAG: {e}")

        # Récupération des règles d'enrichissement pour ce rôle
        role_enrichments = EnrichmentConfig.get_role_enrichments(agent.name)

        # Enrichissement à partir de la base de connaissances
        if role_enrichments:
            enriched_sections.append("\n=== CONTEXTE ENRICHI ===")

            for enrichment in role_enrichments:
                knowledge = await agent.knowledge_base.query_knowledge(
                    query=enrichment.pydantic_path,
                    category=enrichment.source_role
                )

                if knowledge:
                    enriched_sections.extend([
                        f"\n{enrichment.description}:",
                        self._format_knowledge(knowledge)
                    ])

        # Section Tâche

        enriched_sections.extend([
            "=== DÉTAILS DE LA TÂCHE ===",
            f"ID: {task.get('id', 'N/A')}",
            f"Titre: {task.get('title', 'N/A')}",
            f"Description: {task.get('description', 'N/A')}"]
        )

        # Critères d'acceptation
        if task.get('acceptance_criteria'):
            enriched_sections.append("\nCritères d'acceptation:")
            enriched_sections.extend([f"- {criterion}" for criterion in task['acceptance_criteria']])

        """
        # Section Fichiers
        if task.get('files'):
            enriched_sections.append("\n=== FICHIERS À TRAITER ===")
            for file in task['files']:
                if await agent.file_manager.file_exists(file):
                    content = await agent.file_manager.read_file(file)
                else:
                    content = f"Pas de contenu"
                enriched_sections.extend([
                    f"\nFichier: {file}",
                    "```",
                    content,
                    "```"
                ])
        """
        return "\n".join(enriched_sections)

    def _get_task_base_info(self, task: Dict[str, Any]) -> List[str]:
        """Retourne les informations de base de la tâche de manière générique"""
        sections = ["=== TÂCHE ACTUELLE ==="]

        # Parcourir tous les champs de la tâche de manière générique
        for key, value in task.items():
            # Traitement spécial pour les listes
            if isinstance(value, list):
                sections.append(f"\n{key.capitalize()}:")
                sections.extend([f"- {item}" for item in value])
            # Traitement normal pour les autres types
            else:
                sections.append(f"{key.capitalize()}: {value}")

        return sections

    def _format_knowledge(self, knowledge: Any) -> str:
        """Formate les connaissances pour l'affichage"""
        if isinstance(knowledge, (dict, list)):
            return json.dumps(knowledge, indent=2)
        return str(knowledge)


async def setup_workspace(base_dir: Path) -> tuple[Path, Path, Path, Path]:
    """
    Configure l'espace de travail et les dossiers nécessaires
    """
    # Création des chemins
    workspace_dir = Path(base_dir) / ".agent_workspace"
    projects_dir = workspace_dir  # Ici, projects_dir est le même que workspace_dir
    templates_dir = workspace_dir / "templates"
    venvs_dir = workspace_dir / "venv"

    # Création des dossiers
    for directory in [workspace_dir, projects_dir, templates_dir, venvs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Copie du template s'il n'existe pas
    template_file = templates_dir / "template.html"
    if not template_file.exists():
        source_template = Path(__file__).parent.absolute() / "templates" / "template.html"
        if source_template.exists():
            shutil.copy(source_template, template_file)
        else:
            raise FileNotFoundError(f"Template source non trouvé: {source_template}")

    return workspace_dir, projects_dir, templates_dir, venvs_dir
