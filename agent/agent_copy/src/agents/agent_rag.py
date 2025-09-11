"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# src/agents/rag_agent.py
import asyncio
import json
import logging
import uuid
from datetime import datetime

import openai
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

from agent.src.agent import Agent
from agent.src.interface.html_manager import HTMLManager
from agent.src.server.form_server import FormServer
from agent.src.types.enums import AgentRole, AgentCapability
from agent.src.components.file_manager import FileManager
from agent.src.components.task_manager import TaskManager
from agent.src.types.interfaces import LLMProvider, MemoryProvider

from ...agent.Onto_wa_rag.RAG_context_provider import RagTools, rag_fc_jarvis
from ...agent.Onto_wa_rag.provider.get_key import get_openai_key
from ...agent.Onto_wa_rag.CONSTANT import API_KEY_PATH, ONTOLOGY_PATH_TTL


class RagAgent(Agent):
    """
    Agent spécialisé pour la récupération et l'enrichissement de contexte via RAG.

    Capacités :
    - Analyse et décomposition de requêtes
    - Extraction de contexte via RAG
    - Enrichissement d'informations
    - Stockage dans la knowledge base partagée
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
            pydantic_model: Optional[Any] = None,
            response_format: str = "",
            exclude_dirs: Optional[Set[str]] = None,
            index_directory: Optional[str] = None  # répertoire à indexer
    ):
        # Conversion du rôle et initialisation de la classe parent
        role = self._convert_role(role_name)
        super().__init__(
            name=name,
            role=role,
            llm_provider=llm_provider,
            memory_provider=memory_provider,
            system_prompt=system_prompt,
            memory_size=memory_size
        )

        # Composants de base
        self.pydantic_model = pydantic_model
        self.html_manager = html_manager
        self.project_name = project_name
        self.project_path = Path(project_path)
        self.response_format = response_format

        # Répertoire à indexer (différent du project_path pour FileManager)
        self.index_directory = Path(index_directory) if index_directory else self.project_path

        # Gestionnaires
        self.file_manager = file_manager or FileManager(project_path)
        self.task_manager = task_manager or TaskManager()

        # Configuration RAG
        self.exclude_dirs = exclude_dirs or {"rag_storage", ".agent_workspace", "auto", "log", "__pycache__"}
        self.ontology_path = ONTOLOGY_PATH_TTL
        self.storage_dir = self.index_directory / "rag_storage"
        self.storage_dir.mkdir(exist_ok=True)

        # Outils RAG
        self.rag_tools: Optional[RagTools] = None
        self.query_planner_client: Optional[openai.Client] = None
        self.query_planner_model = "gpt-4o-mini"

        # Cache et état
        self.query_cache: Dict[str, Any] = {}
        self.last_indexed_state: Optional[str] = None
        self.is_initialized = False

        # Formulaire interactif (si nécessaire)
        self.form_server = FormServer()
        self.form_response = None
        self.form_event = asyncio.Event()

        # Configuration du logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.name}")
        self.logger.setLevel(logging.INFO)

    async def _ensure_rag_initialized(self) -> None:
        """S'assure que le système RAG est initialisé"""
        if self.is_initialized:
            return

        try:
            self.logger.info("Initialisation du système RAG...")

            # Client OpenAI pour la planification de requêtes
            try:
                self.query_planner_client = openai.Client(api_key=get_openai_key(api_key_path=API_KEY_PATH))
            except Exception as e:
                self.logger.error(f"Erreur initialisation client OpenAI: {e}")
                self.query_planner_client = None

            # Vérification de l'ontologie
            if not Path(self.ontology_path).exists():
                self.logger.warning(f"Fichier d'ontologie non trouvé: {self.ontology_path}")

            # Initialisation de RagTools
            self.rag_tools = RagTools(
                storage_dir=self.index_directory, #  TODO tout pourri
                ontology_ttl_path=self.ontology_path
            )
            await self.rag_tools._ensure_init()

            # Indexation initiale
            await self._reindex_project()

            self.is_initialized = True
            self.logger.info("Système RAG initialisé avec succès")

        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation RAG: {e}")
            raise

    async def _reindex_project(self) -> None:
        """Réindexe le projet pour mettre à jour la base RAG"""
        try:
            if not self.rag_tools:
                raise ValueError("RagTools non initialisé")

            exclude_dirs_paths = set()

            for dirname in self.exclude_dirs:
                # Chemins absolus à exclure
                exclude_path = self.index_directory / dirname
                if exclude_path.exists():
                    exclude_dirs_paths.add(exclude_path)
                    self.logger.info(f"Exclusion du répertoire: {exclude_path}")

                # CORRECTION: Ajouter des exclusions spécifiques problématiques
            additional_excludes = {
                self.index_directory / ".git",
                self.index_directory / ".venv",
                self.index_directory / "venv",
                self.index_directory / "env",
                self.index_directory / "node_modules",
                self.index_directory / ".pytest_cache",
                self.index_directory / "__pycache__",
                self.index_directory / ".agent_workspace" / "venv",  # Spécifiquement le venv dans workspace
            }

            for path in additional_excludes:
                if path.exists():
                    exclude_dirs_paths.add(path)
                    self.logger.info(f"Exclusion supplémentaire: {path}")

            self.logger.info(f"Répertoires exclus: {len(exclude_dirs_paths)}")
            self.logger.info(f"Extensions incluses: f90, md")

            # Indexation avec extensions pertinentes - CORRECTION: utiliser index_directory
            await self.rag_tools.rag.index_directory(
                root=self.index_directory,  # CORRECTION
                exclude_dirs=exclude_dirs_paths,
                include_exts={"f90", "md"}
            )

            # Mise à jour de l'état d'indexation - CORRECTION: utiliser index_directory
            self.last_indexed_state = str(self.index_directory.stat().st_mtime)
            self.logger.info("Réindexation terminée avec succès")

        except Exception as e:
            self.logger.error(f"Erreur lors de la réindexation: {e}")
            raise

    def _generate_storage_key(self, query: str) -> str:
        """Génère une clé unique pour stocker les résultats RAG"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_hash = hash(query) % 10000  # Hash simple pour éviter les clés trop longues
        unique_id = str(uuid.uuid4())[:8]
        return f"rag_{timestamp}_{query_hash}_{unique_id}"

    async def process_rag_request_with_key(self, query: str) -> Dict[str, str]:
        """
        Traite une demande RAG et retourne une clé pour accéder aux résultats
        """
        try:
            self.logger.info(f"Traitement demande RAG avec stockage: '{query}'")

            # Traiter la demande normalement
            result = await self.process_message(query)

            # Générer une clé unique
            storage_key = self._generate_storage_key(query)

            # Stocker dans la knowledge base avec la clé
            await self.knowledge_base.store_pydantic_result(storage_key, result)

            # Aussi stocker avec le nom de l'agent pour l'historique
            await self.knowledge_base.store_pydantic_result(self.name, result)

            # Retourner la clé et un résumé
            summary = self._create_result_summary(result)

            response = {
                "storage_key": storage_key,
                "summary": summary,
                "status": "success",
                "query": query
            }

            self.logger.info(f"Résultats RAG stockés avec la clé: {storage_key}")
            return response

        except Exception as e:
            self.logger.error(f"Erreur lors du traitement RAG avec clé: {e}")
            return {
                "storage_key": None,
                "summary": f"Erreur: {str(e)}",
                "status": "error",
                "query": query
            }

    def _create_result_summary(self, result: Any) -> str:
        """Crée un résumé des résultats pour le retour"""
        try:
            if isinstance(result, dict) and 'rag_analysis' in result:
                rag_analysis = result['rag_analysis']
                queries = rag_analysis.get('rag_queries', [])

                summary_parts = []
                for query_info in queries:
                    query_text = query_info.get('query', '')

                    # CORRECTION: Utiliser get_rag_response_from_query
                    response = self.get_rag_response_from_query(query_info)

                    if response and 'entity' in response and 'error' not in response:
                        entity = response.get('entity', 'N/A')
                        entity_type = response.get('entity_type', 'N/A')
                        file = response.get('file', 'N/A')
                        summary_parts.append(f"- {entity} ({entity_type}) dans {file}")
                    else:
                        summary_parts.append(f"- Aucun résultat pour: {query_text}")

                return f"Trouvé {len(queries)} éléments:\n" + "\n".join(summary_parts)

            elif hasattr(result, 'rag_analysis'):
                # Même logique pour objet Pydantic
                rag_analysis = result.rag_analysis
                if hasattr(rag_analysis, 'rag_queries'):
                    queries = rag_analysis.rag_queries
                    summary_parts = []
                    for query_info in queries:
                        # CORRECTION: Utiliser get_rag_response_from_query
                        response = self.get_rag_response_from_query(query_info)
                        entity = response.get('entity', 'N/A') if response else 'N/A'
                        summary_parts.append(f"- {entity}")
                    return f"Trouvé {len(queries)} éléments:\n" + "\n".join(summary_parts)

            return "Résultats RAG disponibles"

        except Exception as e:
            return f"Résumé indisponible: {str(e)}"

    async def process_message(self, message: str, pydantic_model=None, streaming: bool = False) -> Any:
        """
        Traite un message comme une demande RAG et retourne le contexte enrichi.
        """
        try:
            self.logger.info(f"Traitement de la demande RAG par {self.name}")

            # S'assurer que RAG est initialisé
            await self._ensure_rag_initialized()

            # Vérifier si c'est une demande de réindexation
            if "reindex" in message.lower() or "mise à jour index" in message.lower():
                await self._reindex_project()
                return {"status": "Index mis à jour avec succès"}

            # Utiliser le modèle Pydantic fourni ou celui par défaut
            model_to_use = pydantic_model or self.pydantic_model

            if model_to_use:
                # Enrichir le message avec des instructions RAG
                enriched_message = self._enrich_message_for_rag(message)

                # Obtenir la réponse structurée du LLM (sans les réponses RAG encore)
                response = await self.llm_provider.generate_response(
                    enriched_message,
                    pydantic_model=model_to_use,
                    stream=streaming
                )

                # Normaliser les clés de la réponse
                normalized_response = self._normalize_response_keys(response)

                # Extraire et exécuter les requêtes RAG
                queries = self._extract_queries_from_response(normalized_response)
                if queries:
                    self.logger.info(f"Exécution de {len(queries)} requêtes RAG")

                    # Exécuter les requêtes et enrichir avec les réponses
                    enriched_queries = await self._execute_rag_queries_with_responses(queries)

                    # Créer le contexte consolidé formaté
                    consolidated_context = self._format_queries_with_responses(enriched_queries)

                    # Reconstruire la réponse avec les données enrichies
                    final_response_dict = self._rebuild_response_with_enriched_queries(
                        normalized_response,
                        enriched_queries,
                        consolidated_context
                    )

                    # Convertir en objet Pydantic avant de stocker
                    try:
                        final_response = model_to_use(**final_response_dict)
                    except Exception as e:
                        self.logger.warning(f"Impossible de créer l'objet Pydantic: {e}")
                        final_response = final_response_dict

                    # Stocker dans la knowledge base
                    await self.knowledge_base.store_pydantic_result(self.name, final_response)

                    return final_response
                else:
                    # Pas de requêtes à exécuter
                    await self.knowledge_base.store_pydantic_result(self.name, normalized_response)
                    return normalized_response
            else:
                # Traitement simple sans structure
                context = await self._get_rag_context_simple(message)
                result = {"rag_context": context}

                await self.knowledge_base.store_pydantic_result(self.name, result)
                return result

        except Exception as e:
            self.logger.error(f"Erreur dans process_message: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _enrich_message_for_rag(self, message: str) -> str:
        """Enrichit le message avec des instructions spécifiques au RAG"""
        return f"""{message}

INSTRUCTIONS SUPPLÉMENTAIRES POUR L'ANALYSE RAG:
1. Analysez attentivement la demande pour identifier les éléments spécifiques du code
2. Décomposez la demande en requêtes précises pour le système RAG
3. Chaque requête doit cibler un élément spécifique (fonction, module, classe, etc.)
4. Formulez les requêtes de manière à maximiser la pertinence des résultats

Contexte du projet: {self.project_path}
Extensions supportées: Fortran 90, Python, Markdown, JSON, YAML
"""

    async def _get_rag_context_simple(self, query: str) -> str:
        """Récupère le contexte RAG de manière simple (sans structure Pydantic)"""
        try:
            await self._ensure_rag_initialized()

            # Méthode simple : utiliser directement les outils RAG
            self.rag_tools.my_list_of_response = []

            # CORRECTION: utiliser 'question' au lieu de 'data'
            await self.rag_tools.add_to_list(question=query)

            # Récupérer les résultats
            results = self.rag_tools.my_list_of_response.copy()

            # Formater et retourner
            return self._format_rag_results(results)

        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération simple du contexte: {e}")
            return f"Erreur lors de la récupération du contexte: {str(e)}"

    async def _decompose_query_with_llm(self, user_request: str) -> List[Dict[str, Any]]:
        """Décompose une requête utilisateur en sous-requêtes RAG via LLM"""
        if not self.query_planner_client:
            return [{"query": user_request, "target_type": "general", "priority": 1}]

        try:
            # Réinitialiser la liste des réponses RAG
            if self.rag_tools:
                self.rag_tools.my_list_of_response = []

            messages = [
                {
                    "role": "system",
                    "content": """Vous êtes un agent IA créateur de contexte pour un système RAG. 

    Analysez la demande utilisateur et décomposez-la en requêtes précises pour extraire le contexte pertinent du code.

    RÈGLES DE DÉCOMPOSITION:
    - Pour une fonction: "Veuillez me fournir la fonction [nom_fonction]"
    - Pour un module: "Veuillez me fournir le module [nom_module]"  
    - Pour une classe: "Veuillez me fournir la classe [nom_classe]"
    - Soyez précis et concis
    - Une requête = un élément spécifique du code

    Ne répondez que par des appels de fonction avec l'outil add_to_list."""
                },
                {"role": "user", "content": user_request}
            ]

            queries = []
            max_turns = 5

            for turn in range(max_turns):
                response = self.query_planner_client.chat.completions.create(
                    model=self.query_planner_model,
                    messages=messages,
                    tools=rag_fc_jarvis(),
                    tool_choice="auto"
                )

                msg = response.choices[0].message

                if msg.tool_calls:
                    messages.append({
                        "role": "assistant",
                        "tool_calls": [tc.model_dump() for tc in msg.tool_calls]
                    })

                    for tool_call in msg.tool_calls:
                        name = tool_call.function.name
                        args = json.loads(tool_call.function.arguments)

                        if hasattr(self.rag_tools, name):
                            # CORRECTION: utiliser le bon nom de paramètre
                            if name == "add_to_list":
                                # La méthode add_to_list prend 'question', pas 'data'
                                question = args.get("question", args.get("data", ""))
                                tool_result = await self.rag_tools.add_to_list(question=question)

                                # Extraire la requête pour notre liste
                                queries.append({
                                    "query": question,
                                    "target_type": self._infer_target_type(question),
                                    "priority": 1
                                })
                            else:
                                tool_result = await getattr(self.rag_tools, name)(**args)

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": name,
                                "content": json.dumps(tool_result)
                            })
                        else:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": name,
                                "content": json.dumps({"error": f"Outil {name} non trouvé"})
                            })

                    if not msg.content:
                        continue
                    else:
                        break
                else:
                    break

            return queries if queries else [{"query": user_request, "target_type": "general", "priority": 1}]

        except Exception as e:
            self.logger.error(f"Erreur lors de la décomposition: {e}")
            return [{"query": user_request, "target_type": "general", "priority": 1}]

    def _infer_target_type(self, query: str) -> str:
        """Infère le type de cible basé sur la requête"""
        query_lower = query.lower()

        if "fonction" in query_lower or "function" in query_lower:
            return "function"
        elif "module" in query_lower:
            return "module"
        elif "classe" in query_lower or "class" in query_lower:
            return "class"
        elif "fichier" in query_lower or "file" in query_lower:
            return "file"
        else:
            return "general"

    async def _execute_rag_queries_with_responses(self, queries: List[Any]) -> List[Dict[str, Any]]:
        """
        Exécute les requêtes RAG et stocke directement les réponses dans chaque requête
        """
        if not self.rag_tools:
            return []

        try:
            # Normaliser les queries
            enriched_queries = []

            for query_info in queries:
                # Convertir en dictionnaire
                if hasattr(query_info, 'model_dump'):
                    query_dict = query_info.model_dump()
                elif hasattr(query_info, 'dict'):
                    query_dict = query_info.dict()
                elif isinstance(query_info, dict):
                    query_dict = query_info.copy()
                else:
                    query_dict = {
                        "query": getattr(query_info, 'query', str(query_info)),
                        "target_type": getattr(query_info, 'target_type', 'general'),
                        "priority": getattr(query_info, 'priority', 1)
                    }

                # S'assurer que le champ response_json existe
                if "response_json" not in query_dict:
                    query_dict["response_json"] = "{}"

                enriched_queries.append(query_dict)

            # Trier par priorité
            sorted_queries = sorted(enriched_queries, key=lambda x: x.get("priority", 1))

            # Exécuter chaque requête et stocker sa réponse
            for query_dict in sorted_queries:
                query = query_dict.get("query", "")
                target_type = query_dict.get("target_type", "general")
                priority = query_dict.get("priority", 1)

                if not query:
                    continue

                self.logger.info(f"Exécution requête RAG: '{query}' (type: {target_type}, priorité: {priority})")

                # Vérifier le cache
                cache_key = f"{query}_{self.last_indexed_state}"
                if cache_key in self.query_cache:
                    cached_response = self.query_cache[cache_key]
                    query_dict["response_json"] = json.dumps(cached_response)  # CORRECTION: stocker en JSON
                    self.logger.debug(f"Résultat récupéré du cache pour: {query}")
                    continue

                try:
                    # Exécuter la requête RAG et obtenir la réponse structurée
                    rag_response = await self.rag_tools.rag_function_json(question=query)

                    # CORRECTION: Stocker la réponse en JSON string
                    query_dict["response_json"] = json.dumps(rag_response)

                    # Mettre en cache
                    self.query_cache[cache_key] = rag_response

                    self.logger.info(f"Requête exécutée avec succès: {query}")

                except Exception as e:
                    self.logger.error(f"Erreur lors de l'exécution de la requête '{query}': {e}")
                    error_response = {"error": str(e)}
                    query_dict["response_json"] = json.dumps(error_response)  # CORRECTION: stocker erreur en JSON

            self.logger.info(f"Toutes les requêtes RAG exécutées: {len(sorted_queries)} requêtes")
            return sorted_queries

        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution des requêtes RAG: {e}")
            return []

    async def _execute_rag_queries(self, queries: List[Any]) -> List[Dict[str, Any]]:
        """Exécute les requêtes RAG et retourne les résultats"""
        if not self.rag_tools:
            return []

        # Réinitialiser la liste des réponses
        self.rag_tools.my_list_of_response = []

        try:
            # Normaliser les queries en dictionnaires pour le traitement
            normalized_queries = []

            for query_info in queries:
                # Convertir l'objet Pydantic en dictionnaire
                if hasattr(query_info, 'model_dump'):
                    query_dict = query_info.model_dump()
                elif hasattr(query_info, 'dict'):
                    query_dict = query_info.dict()
                elif isinstance(query_info, dict):
                    query_dict = query_info
                else:
                    # Fallback : essayer d'accéder aux attributs directement
                    query_dict = {
                        "query": getattr(query_info, 'query', str(query_info)),
                        "target_type": getattr(query_info, 'target_type', 'general'),
                        "priority": getattr(query_info, 'priority', 1)
                    }

                normalized_queries.append(query_dict)

            # Trier par priorité
            sorted_queries = sorted(normalized_queries, key=lambda x: x.get("priority", 1))

            for query_dict in sorted_queries:
                query = query_dict.get("query", "")
                target_type = query_dict.get("target_type", "general")
                priority = query_dict.get("priority", 1)

                if not query:
                    continue

                self.logger.info(f"Exécution requête RAG: '{query}' (type: {target_type}, priorité: {priority})")

                # Vérifier le cache
                cache_key = f"{query}_{self.last_indexed_state}"
                if cache_key in self.query_cache:
                    cached_result = self.query_cache[cache_key]
                    self.rag_tools.my_list_of_response.extend(cached_result)
                    self.logger.debug(f"Résultat récupéré du cache pour: {query}")
                    continue

                # Exécuter la requête
                await self.rag_tools.add_to_list(question=query)

            # Récupérer les résultats
            results = self.rag_tools.my_list_of_response.copy()
            self.logger.info(f"Requêtes RAG exécutées: {len(results)} résultats obtenus")
            return results

        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution des requêtes RAG: {e}")
            return []

    def _normalize_response_keys(self, response: Any) -> Any:
        """Normalise les clés de la réponse"""
        if hasattr(response, 'model_dump'):
            return response
        elif hasattr(response, '__dict__'):
            # Si c'est un objet Pydantic, le retourner tel quel
            return response
        elif isinstance(response, dict) and self.pydantic_model:
            # Si c'est un dict et qu'on a un modèle Pydantic, essayer de l'instancier
            try:
                return self.pydantic_model(**response)
            except Exception as e:
                self.logger.warning(f"Impossible de créer l'instance Pydantic: {e}")
                return response
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

    def get_rag_data_for_query(self, result: Any, query_text: str) -> Dict[str, Any]:
        """
        Récupère les données RAG brutes pour une requête spécifique
        """
        try:
            if hasattr(result, 'rag_analysis'):
                rag_analysis = result.rag_analysis
                if hasattr(rag_analysis, 'rag_queries'):
                    for query_info in rag_analysis.rag_queries:
                        if hasattr(query_info, 'query') and query_info.query == query_text:
                            # CORRECTION: Utiliser get_rag_response_from_query
                            return self.get_rag_response_from_query(query_info)

            # Fallback pour format dict
            if isinstance(result, dict) and 'rag_analysis' in result:
                rag_queries = result['rag_analysis'].get('rag_queries', [])
                for query_info in rag_queries:
                    if query_info.get('query') == query_text:
                        # CORRECTION: Utiliser get_rag_response_from_query
                        return self.get_rag_response_from_query(query_info)

            return {}

        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données RAG: {e}")
            return {}

    def get_all_rag_responses(self, result: Any) -> List[Dict[str, Any]]:
        """
        Récupère toutes les réponses RAG d'un résultat
        """
        responses = []
        try:
            if hasattr(result, 'rag_analysis'):
                rag_analysis = result.rag_analysis
                if hasattr(rag_analysis, 'rag_queries'):
                    for query_info in rag_analysis.rag_queries:
                        # CORRECTION: Utiliser get_rag_response_from_query
                        response = self.get_rag_response_from_query(query_info)
                        if response:
                            responses.append(response)

            # Fallback pour format dict
            elif isinstance(result, dict) and 'rag_analysis' in result:
                rag_queries = result['rag_analysis'].get('rag_queries', [])
                for query_info in rag_queries:
                    # CORRECTION: Utiliser get_rag_response_from_query
                    response = self.get_rag_response_from_query(query_info)
                    if response:
                        responses.append(response)

            return responses

        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération de toutes les réponses RAG: {e}")
            return []

    def _format_single_rag_result(self, result: Dict[str, Any]) -> str:
        """Formate un seul résultat RAG en contexte lisible"""
        if "code" in result and "error" not in result:
            file = result.get('file', 'N/A')
            entity = result.get('entity', 'N/A')
            entity_type = result.get('entity_type', 'N/A')
            start_line = result.get('start_line', 'N/A')
            end_line = result.get('end_line', 'N/A')
            code = result.get('code', '')

            return f"""-- EXTRAIT PERTINENT --
Fichier: {file}
Entité: {entity} ({entity_type})
Lignes: {start_line}-{end_line}

{code}
"""
        else:
            return json.dumps(result, indent=2)

    async def get_stored_rag_data(self, storage_key: str) -> Any:
        """Récupère les données RAG stockées via la clé"""
        try:
            # CORRECTION : Utiliser storage_key comme catégorie ET comme query
            data = await self.knowledge_base.query_knowledge(
                query="",  # Query vide pour récupérer tout le contenu
                category=storage_key  # La clé est utilisée comme catégorie
            )

            if data:
                self.logger.info(f"Données RAG récupérées pour la clé: {storage_key}")
                return data
            else:
                # FALLBACK : Essayer avec une approche alternative
                self.logger.warning(f"Tentative de récupération alternative pour: {storage_key}")

                # Rechercher dans toutes les entrées de la knowledge base
                all_memories = await self.memory_provider.get_all()
                for memory in all_memories:
                    if memory.get('category') == storage_key:
                        self.logger.info(f"Données trouvées via fallback pour: {storage_key}")
                        return memory.get('content')

                self.logger.warning(f"Aucune donnée trouvée pour la clé: {storage_key}")
                return None

        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données RAG: {e}")
            return None

    def _format_rag_results(self, results: List[Dict[str, Any]]) -> str:
        """Formate les résultats RAG en contexte lisible"""
        if not results:
            return "Aucun contexte pertinent trouvé."

        formatted_blocks = []

        for i, result in enumerate(results):
            if isinstance(result, dict) and "code" in result:
                file = result.get('file', 'N/A')
                entity = result.get('entity', 'N/A')
                entity_type = result.get('entity_type', 'N/A')
                start_line = result.get('start_line', 'N/A')
                end_line = result.get('end_line', 'N/A')
                code = result.get('code', '')

                block = f"""-- EXTRAIT PERTINENT {i + 1} --
Fichier: {file}
Entité: {entity} ({entity_type})
Lignes: {start_line}-{end_line}

{code}
"""
                formatted_blocks.append(block)

            elif isinstance(result, str):
                formatted_blocks.append(f"-- INFORMATION {i + 1} --\n{result}\n")
            else:
                formatted_blocks.append(f"-- DONNÉE {i + 1} --\n{json.dumps(result, indent=2)}\n")

        return "\n".join(formatted_blocks)

    def _extract_queries_from_response(self, response: Any) -> List[Any]:
        """Extrait les requêtes RAG de la réponse du LLM"""
        try:
            queries = []

            # Cas 1: Réponse avec attribut rag_analysis
            if hasattr(response, 'rag_analysis'):
                rag_analysis = response.rag_analysis
                if hasattr(rag_analysis, 'rag_queries'):
                    queries = rag_analysis.rag_queries

            # Cas 2: Réponse en dictionnaire
            elif isinstance(response, dict) and 'rag_analysis' in response:
                rag_analysis = response['rag_analysis']
                queries = rag_analysis.get('rag_queries', [])

            self.logger.info(f"Extracted {len(queries)} queries from response")
            return queries

        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction des requêtes: {e}")
            return []

    def _format_queries_with_responses(self, queries_with_responses: List[Dict[str, Any]]) -> str:
        """Formate les requêtes enrichies en contexte lisible pour consolidated_context"""
        if not queries_with_responses:
            return "Aucun contexte pertinent trouvé."

        formatted_blocks = []

        for i, query_info in enumerate(queries_with_responses, 1):
            query = query_info.get("query", "")
            target_type = query_info.get("target_type", "")

            # Récupérer la réponse RAG
            response = self.get_rag_response_from_query(query_info)

            # En-tête de la section
            header = f"-- REQUÊTE {i}: {query} (Type: {target_type}) --"
            formatted_blocks.append(header)

            # Formater la réponse
            if isinstance(response, dict) and "error" not in response and "code" in response:
                file = response.get('file', 'N/A')
                entity = response.get('entity', 'N/A')
                entity_type = response.get('entity_type', 'N/A')
                start_line = response.get('start_line', 'N/A')
                end_line = response.get('end_line', 'N/A')
                code = response.get('code', '')

                content = f"""Fichier: {file}
    Entité: {entity} ({entity_type})
    Lignes: {start_line}-{end_line}

    {code}
    """
            elif isinstance(response, dict) and "error" in response:
                content = f"ERREUR: {response['error']}"
            else:
                content = json.dumps(response, indent=2)

            formatted_blocks.append(content)
            formatted_blocks.append("")  # Ligne vide entre les sections

        return "\n".join(formatted_blocks)

    def _rebuild_response_with_enriched_queries(self, original_response: Any, enriched_queries: List[Dict],
                                                consolidated_context: str) -> Dict[str, Any]:
        """Reconstruit la réponse avec les requêtes enrichies"""
        try:
            # Extraire les données de base
            if hasattr(original_response, 'rag_analysis'):
                original_analysis = original_response.rag_analysis

                return {
                    "rag_analysis": {
                        "original_request": getattr(original_analysis, 'original_request', ''),
                        "analysis": getattr(original_analysis, 'analysis', ''),
                        "rag_queries": enriched_queries,  # Les requêtes avec leurs réponses
                        "consolidated_context": consolidated_context
                    }
                }
            else:
                # Fallback
                return {
                    "rag_analysis": {
                        "original_request": "Request processed",
                        "analysis": "Queries executed and responses collected",
                        "rag_queries": enriched_queries,
                        "consolidated_context": consolidated_context
                    }
                }
        except Exception as e:
            self.logger.error(f"Erreur lors de la reconstruction de la réponse: {e}")
            return {
                "rag_analysis": {
                    "original_request": "Error during processing",
                    "analysis": f"Error: {str(e)}",
                    "rag_queries": enriched_queries,
                    "consolidated_context": consolidated_context
                }
            }

    def get_rag_response_from_query(self, query_info):
        """Récupère la réponse RAG d'une requête (gère response_json et response)"""
        try:
            if hasattr(query_info, 'response_json'):
                return json.loads(query_info.response_json)
            elif hasattr(query_info, 'response'):
                return query_info.response
            elif isinstance(query_info, dict):
                if 'response_json' in query_info:
                    return json.loads(query_info['response_json'])
                elif 'response' in query_info:
                    return query_info['response']
            return {}
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"Erreur décodage réponse RAG: {e}")
            return {"error": "Impossible de décoder la réponse RAG"}

    async def get_context_for_agent(self, agent_name: str, query: str) -> str:
        """API publique pour que d'autres agents récupèrent du contexte RAG"""
        try:
            self.logger.info(f"Récupération de contexte pour {agent_name}: '{query}'")

            # Traiter la demande complète
            result = await self.process_message(query)

            # Extraire le contexte formaté
            if isinstance(result, dict) and 'rag_analysis' in result:
                context = result['rag_analysis'].get('consolidated_context', '')
            elif hasattr(result, 'rag_analysis'):
                context = getattr(result.rag_analysis, 'consolidated_context', '')
            else:
                context = str(result)

            # Stocker l'interaction
            await self.learn_from_interaction(
                {
                    "requesting_agent": agent_name,
                    "query": query,
                    "context_provided": context,
                    "full_result": result  # NOUVEAU: stocker aussi le résultat complet
                },
                category="agent_interactions"
            )

            return context

        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération de contexte pour {agent_name}: {e}")
            return f"Erreur lors de la récupération du contexte: {str(e)}"

    def get_structured_data_for_agent(self, agent_name: str, last_result: Any) -> List[Dict[str, Any]]:
        """
        API pour que d'autres agents accèdent aux données RAG structurées
        """
        try:
            return self.get_all_rag_responses(last_result)
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données structurées: {e}")
            return []

    @staticmethod
    def _convert_role(role_name: str) -> AgentRole:
        """Convertit les noms de rôles en AgentRole enum"""
        role_mapping = {
            "agent_rag": AgentRole.SPECIALIST,
            "rag_agent": AgentRole.SPECIALIST,
        }

        if role_name not in role_mapping:
            # Par défaut, considérer comme spécialiste
            return AgentRole.SPECIALIST

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

    def cleanup(self) -> None:
        """Nettoie les ressources de l'agent RAG"""
        super().cleanup()
        if self.rag_tools:
            # Nettoyer les ressources RAG si nécessaire
            pass
        self.query_cache.clear()
        self.is_initialized = False