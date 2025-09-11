"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue - Agent Unifié
    ------------------------------------------
    """

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Literal, Union

from pydantic import BaseModel, Field, ValidationError

from ..fortran_analysis.core.entity_manager import UnifiedEntity
from ..fortran_analysis.providers.consult import FortranEntityExplorer
from .entity_explorer_jupyter import JupyterEntityExplorer
from ..provider.llm_providers import LLMProvider
from ..retriever_adapter import SimpleRetriever

logger = logging.getLogger(__name__)


@dataclass
class SourceReference:
    """Structure pour tracker les sources utilisées par l'agent."""
    entity_name: str
    entity_type: str
    filepath: str
    filename: str
    start_line: int
    end_line: int
    source_type: str  # "fortran" ou "jupyter"
    tool_used: str  # L'outil qui a fourni cette info
    reference_id: str  # ID unique pour citation

    def get_citation(self) -> str:
        """Génère une citation formatée."""
        return f"[{self.reference_id}] {self.filename} (lignes {self.start_line}-{self.end_line}) - {self.entity_type}: {self.entity_name}"


@dataclass
class AgentStep:
    """Représente une étape dans le processus de raisonnement de l'agent."""
    step_number: int
    thought: str
    plan: Optional[List[str]]
    working_memory: Optional[List[str]]
    tool_name: str
    tool_arguments: Dict[str, Any]
    tool_result_summary: str
    sources_discovered: List[str] = field(default_factory=list)
    execution_time_ms: Optional[float] = None


@dataclass
class EntitySummary:
    """Résumé d'une entité découverte pendant l'analyse."""
    name: str
    type: str
    source_type: str  # "fortran" ou "jupyter"
    filepath: str
    filename: str
    start_line: int
    end_line: int
    relevance_score: Optional[float] = None
    role_in_answer: str = ""  # "primary", "supporting", "context"


@dataclass
class AgentResponse:
    """Réponse structurée complète de l'agent."""

    # Réponse principale
    answer: str
    status: str  # "success", "clarification_needed", "timeout", "error"

    # Métadonnées de session
    query: str
    session_id: str
    timestamp: datetime
    execution_time_total_ms: float
    steps_taken: int
    max_steps: int

    # Processus de raisonnement
    reasoning_steps: List[AgentStep] = field(default_factory=list)
    final_plan_executed: Optional[List[str]] = None

    # Découvertes
    entities_discovered: List[EntitySummary] = field(default_factory=list)
    sources_consulted: List[SourceReference] = field(default_factory=list)

    # Analyse des outils
    tools_used: Dict[str, int] = field(default_factory=dict)
    explorers_used: List[str] = field(default_factory=list)

    # Métadonnées pour chaînage avec d'autres agents
    key_findings: List[str] = field(default_factory=list)
    suggested_followup_queries: List[str] = field(default_factory=list)
    confidence_level: float = 1.0

    # Message d'erreur ou de clarification
    clarification_question: Optional[str] = None
    error_details: Optional[str] = None

    # Méthode d'extension de AgentResponse
    def _analyze_entities_and_findings(self):
        """Analyse les sources pour identifier les entités clés et les insights."""

        # Convertir les sources en entités
        entity_counts = {}
        for source in self.sources_consulted:
            key = f"{source.entity_name}:{source.entity_type}"
            entity_counts[key] = entity_counts.get(key, 0) + 1

            entity_summary = EntitySummary(
                name=source.entity_name,
                type=source.entity_type,
                source_type=source.source_type,
                filepath=source.filepath,
                filename=source.filename,
                start_line=source.start_line,
                end_line=source.end_line,
                relevance_score=None,
                role_in_answer="primary" if entity_counts[key] > 1 else "supporting"
            )

            # Éviter les doublons
            if not any(e.name == entity_summary.name and e.filepath == entity_summary.filepath
                       for e in self.entities_discovered):
                self.entities_discovered.append(entity_summary)

        # Générer des insights clés
        self.key_findings = []
        if self.entities_discovered:
            fortran_entities = [e for e in self.entities_discovered if e.source_type == "fortran"]
            jupyter_entities = [e for e in self.entities_discovered if e.source_type == "jupyter"]

            if fortran_entities:
                self.key_findings.append(f"Analysé {len(fortran_entities)} entité(s) Fortran")
            if jupyter_entities:
                self.key_findings.append(f"Analysé {len(jupyter_entities)} entité(s) Jupyter")

        # Générer des questions de suivi
        self.suggested_followup_queries = []
        primary_entities = self.get_entities_by_role("primary")
        if primary_entities:
            for entity in primary_entities[:2]:  # Top 2
                self.suggested_followup_queries.append(f"Analyse détaillée de {entity.name}")
                self.suggested_followup_queries.append(f"Qui appelle {entity.name} ?")

        # Calculer un niveau de confiance basique
        if self.status == "success" and self.sources_consulted:
            self.confidence_level = min(1.0, len(self.sources_consulted) / 3.0)  # Plus de sources = plus de confiance
        else:
            self.confidence_level = 0.5

    def to_human_readable(self) -> str:
        """Convertit en format lisible pour un humain."""
        output = [self.answer]

        if self.sources_consulted:
            output.append("\n## 📚 Sources consultées :")
            for source in self.sources_consulted:
                output.append(f"\n{source.get_citation()}")

        return "".join(output)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation JSON."""
        return {
            "answer": self.answer,
            "status": self.status,
            "query": self.query,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "execution_time_total_ms": self.execution_time_total_ms,
            "steps_taken": self.steps_taken,
            "max_steps": self.max_steps,
            "reasoning_steps": [
                {
                    "step_number": step.step_number,
                    "thought": step.thought,
                    "plan": step.plan,
                    "working_memory": step.working_memory,
                    "tool_name": step.tool_name,
                    "tool_arguments": step.tool_arguments,
                    "tool_result_summary": step.tool_result_summary,
                    "sources_discovered": step.sources_discovered,
                    "execution_time_ms": step.execution_time_ms
                } for step in self.reasoning_steps
            ],
            "final_plan_executed": self.final_plan_executed,
            "entities_discovered": [
                {
                    "name": entity.name,
                    "type": entity.type,
                    "source_type": entity.source_type,
                    "filepath": entity.filepath,
                    "filename": entity.filename,
                    "start_line": entity.start_line,
                    "end_line": entity.end_line,
                    "relevance_score": entity.relevance_score,
                    "role_in_answer": entity.role_in_answer
                } for entity in self.entities_discovered
            ],
            "sources_consulted": [
                {
                    "reference_id": source.reference_id,
                    "entity_name": source.entity_name,
                    "entity_type": source.entity_type,
                    "filepath": source.filepath,
                    "filename": source.filename,
                    "start_line": source.start_line,
                    "end_line": source.end_line,
                    "source_type": source.source_type,
                    "tool_used": source.tool_used
                } for source in self.sources_consulted
            ],
            "tools_used": self.tools_used,
            "explorers_used": self.explorers_used,
            "key_findings": self.key_findings,
            "suggested_followup_queries": self.suggested_followup_queries,
            "confidence_level": self.confidence_level,
            "clarification_question": self.clarification_question,
            "error_details": self.error_details
        }

    def to_json(self, indent: int = 2) -> str:
        """Convertit en JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def get_entities_by_role(self, role: str) -> List[EntitySummary]:
        """Filtre les entités par rôle."""
        return [e for e in self.entities_discovered if e.role_in_answer == role]

    def get_primary_sources(self) -> List[SourceReference]:
        """Retourne les sources des entités primaires."""
        primary_entities = {e.name for e in self.get_entities_by_role("primary")}
        return [s for s in self.sources_consulted if s.entity_name in primary_entities]


class AgentAskClarificationArgs(BaseModel):
    """Arguments pour poser une question de clarification à l'utilisateur."""
    question: str = Field(..., description="La question précise à poser à l'utilisateur.")


class AgentFindEntityByNameArgs(BaseModel):
    """Arguments pour trouver une entité par son nom (Fortran ou Jupyter)."""
    entity_name: str = Field(..., description="The exact or approximate name of the entity to find.")


class AgentListEntitiesArgs(BaseModel):
    """Arguments pour trouver des entités par attributs ou par concept (Fortran ou Jupyter)."""
    entity_type: Optional[str] = Field(None,
                                       description="The type of entity (e.g., 'subroutine', 'function', 'notebook', 'code_cell', 'class').")
    entity_name: Optional[str] = Field(None, description="An approximate name of the entity.")
    filename: Optional[str] = Field(None, description="The name of the file to search within.")
    parent_entity: Optional[str] = Field(None, description="The name of the parent entity.")
    dependencies: Optional[str] = Field(None, description="Name of a module/import used.")
    entity_role: Optional[str] = Field(None, description="For Jupyter: role like 'summary', 'documentation', etc.")

    detected_concept: Optional[str] = Field(None,
                                            description="**Use this for any conceptual, semantic, or 'how-to' question.** Describe the logic or purpose to find (e.g., 'memory allocation', 'data visualization', 'machine learning').")


class AgentGetEntityReportArgs(BaseModel):
    """Arguments to get a detailed report for a single entity (Fortran or Jupyter)."""
    entity_name: str = Field(..., description="The exact name of the entity to get a full report for.")
    include_source_code: bool = Field(False, description="Whether to include source code. Default to false.")


class AgentGetRelationsArgs(BaseModel):
    """Arguments to get relationships for a single entity (Fortran or Jupyter)."""
    entity_name: str = Field(..., description="The name of the central entity for the relation query.")
    relation_type: Literal['callers', 'callees', 'references', 'imports'] = Field(...,
                                                                                  description="The type of relation: 'callers'/'references' (who uses this), 'callees' (what this uses), 'imports' (for Jupyter).")


class AgentGetNotebookOverviewArgs(BaseModel):
    """Arguments to get a complete overview of a Jupyter notebook."""
    notebook_name: str = Field(..., description="The name of the notebook to analyze.")


class AgentFinalAnswerArgs(BaseModel):
    """Arguments for the final answer with automatic source citation."""
    text: str = Field(...,
                      description="The final answer content WITHOUT manual source citations. The system will automatically add source references.")


class AgentDecision(BaseModel):
    """Définit la pensée et l'action structurée de l'agent unifié."""
    thought: str = Field(...,
                         description="Ma réflexion sur l'état actuel, ce que je viens d'apprendre, et ce que je dois faire.")
    plan: Optional[List[str]] = Field(None,
                                      description="[À DÉFINIR AU 1ER TOUR SEULEMENT] La liste des étapes pour répondre à la requête.")
    working_memory_candidates: Optional[List[str]] = Field(None,
                                                           description="La liste des noms d'entités qu'il reste à examiner.")
    tool_name: Literal[
        "find_entity_by_name",
        "list_entities",
        "get_entity_report",
        "get_relations",
        "get_notebook_overview",
        "ask_for_clarification",
        "final_answer"
    ] = Field(...)

    arguments: Union[
        AgentFindEntityByNameArgs,
        AgentListEntitiesArgs,
        AgentGetEntityReportArgs,
        AgentGetRelationsArgs,
        AgentGetNotebookOverviewArgs,
        AgentAskClarificationArgs,
        AgentFinalAnswerArgs
    ] = Field(...)


class CodeAnalysisAgent:
    """Agent unifié pour analyser le code Fortran ET les notebooks Jupyter avec citations automatiques."""

    def __init__(self,
                 llm_provider: LLMProvider,
                 fortran_explorer: Optional[FortranEntityExplorer] = None,
                 jupyter_explorer: Optional[JupyterEntityExplorer] = None,
                 max_steps: int = 7):
        """
        Initialise l'agent unifié avec système de citations.
        """
        if not fortran_explorer and not jupyter_explorer:
            raise ValueError("Au moins un explorateur (Fortran ou Jupyter) doit être fourni.")

        self.llm = llm_provider
        self.fortran_explorer = fortran_explorer
        self.jupyter_explorer = jupyter_explorer
        self.max_steps = max_steps
        self.system_prompt = self.build_unified_system_prompt()

        # Historique persistant
        self.conversation_history: List[Dict[str, str]] = []

        # ✅ NOUVEAU : Système de tracking des sources
        self.sources_used: Dict[str, SourceReference] = {}
        self.reference_counter = 0
        self.semantic_retriever = SimpleRetriever()

    def build_unified_system_prompt(self) -> str:
        """Construit le prompt système pour l'agent unifié avec emphasis sur les citations."""

        available_languages = []
        if self.fortran_explorer:
            available_languages.append("Fortran")
        if self.jupyter_explorer:
            available_languages.append("Jupyter notebooks (Python)")

        lang_description = " et ".join(available_languages)

        mission_and_process = f"""Tu es un agent expert autonome, spécialisé dans l'analyse de code {lang_description}. Ton travail doit être systématique, rigoureux et complet.

**OBLIGATION CRITIQUE : TRAÇABILITÉ DES SOURCES**
- Chaque information que tu utilises provient d'outils qui analysent des fichiers spécifiques
- Tu DOIS traquer les sources de tes informations pour pouvoir les citer
- Dans ta réponse finale, tu citeras automatiquement toutes les sources consultées
- Ne jamais inventer ou supposer des informations non vérifiées par tes outils

**TYPES D'ENTITÉS GÉRÉS :**
"""

        if self.fortran_explorer:
            mission_and_process += """
- **Fortran :** module, subroutine, function, program, type_definition, variable, parameter
"""

        if self.jupyter_explorer:
            mission_and_process += """
- **Jupyter :** notebook, code_cell, markdown_cell, function, class, async function
"""

        mission_and_process += """
**PROCESSUS DE RAISONNEMENT OBLIGATOIRE :**

1. **Rigueur et Exhaustivité :** Fournir des réponses COMPLÈTES et DOCUMENTÉES.
2. **Workflow de Mémoire de Travail :** Pour les recherches exhaustives.
3. **Robustesse :** Utiliser `ask_for_clarification` pour les requêtes ambiguës.
4. **Traçabilité :** Chaque affirmation doit être basée sur des données obtenues via tes outils.
"""

        tool_descriptions = f"""
<outils>
    - `semantic_search`: **🔍 OUTIL DE RECHERCHE SÉMANTIQUE** - Recherche par similarité dans le contenu des notebooks. Idéal pour "comment faire X", "exemples de Y", concepts techniques.
    - `find_entity_by_name`: **OUTIL DE DÉMARRAGE RAPIDE.** Fonctionne avec {lang_description}.
    - `list_entities`: **OUTIL DE DÉCOUVERTE STRUCTURELLE.** Recherche par attributs structurels (type, nom exact, parent).
    - `get_entity_report`: **OUTIL D'INSPECTION DÉTAILLÉE.** Rapport complet d'une entité.
    - `get_relations`: **OUTIL D'ENQUÊTE.** Relations/références d'une entité.
"""

        if self.jupyter_explorer:
            tool_descriptions += """
    - `get_notebook_overview`: **OUTIL SPÉCIALISÉ JUPYTER.** Vue d'ensemble d'un notebook complet.
"""

        tool_descriptions += """
    - `ask_for_clarification`: **OUTIL DE DIALOGUE.** Pour clarifier les requêtes ambiguës.
    - `final_answer`: **OUTIL DE CONCLUSION.** Le système ajoutera automatiquement les citations des sources consultées.

**STRATÉGIE DE CHOIX D'OUTIL :**
- Pour "comment faire X", "exemples de Y", questions conceptuelles → `semantic_search`
- Pour "quelle est l'entité X", recherche par nom → `find_entity_by_name`
- Pour "lister les entités de type Y" → `list_entities`
- Pour analyser une entité précise → `get_entity_report`
</outils>

**INSTRUCTIONS POUR LA RÉPONSE FINALE :**
- Concentre-toi sur le CONTENU de ta réponse dans `final_answer`
- Ne cite PAS manuellement les sources dans le texte
- Le système ajoutera automatiquement une section "Sources consultées" avec toutes les références
- Structure ta réponse de manière claire et logique
"""

        return f"{mission_and_process}\n\n{tool_descriptions}\n\nMaintenant, commence."

    def _add_source_reference(self, entity_info: Dict[str, Any], source_type: str, tool_used: str) -> str:
        """Ajoute une référence de source et retourne son ID."""

        # ✅ CORRECTION : Gérer les objets UnifiedEntity
        if isinstance(entity_info, dict):
            if 'entity' in entity_info:
                # Cas des résultats de list_entities
                entity = entity_info['entity']

                # Vérifier si entity est un objet UnifiedEntity ou un dict
                if hasattr(entity, 'entity_name'):  # C'est un objet UnifiedEntity
                    entity_name = entity.entity_name
                    entity_type = entity.entity_type
                    filepath = entity.filepath or 'Unknown'
                    filename = entity.filename or (filepath.split('/')[-1] if '/' in filepath else filepath)
                    start_line = entity.start_line or 0
                    end_line = entity.end_line or 0
                else:  # C'est un dictionnaire
                    entity_name = entity.get('entity_name', entity.get('name', 'Unknown'))
                    entity_type = entity.get('entity_type', entity.get('type', 'Unknown'))
                    filepath = entity.get('filepath', 'Unknown')
                    filename = entity.get('filename', filepath.split('/')[-1] if '/' in filepath else filepath)
                    start_line = entity.get('start_line', 0)
                    end_line = entity.get('end_line', 0)
            else:
                # Cas des rapports d'entité directs
                entity_name = entity_info.get('entity_name', entity_info.get('name', 'Unknown'))
                entity_type = entity_info.get('entity_type', entity_info.get('type', 'Unknown'))

                # Chercher dans summary si disponible
                summary = entity_info.get('summary', {})
                filepath = summary.get('filepath', entity_info.get('filepath', 'Unknown'))
                filename = filepath.split('/')[-1] if '/' in filepath else filepath
                start_line = summary.get('start_line', entity_info.get('start_line', 0))
                end_line = summary.get('end_line', entity_info.get('end_line', 0))
        elif hasattr(entity_info, 'entity_name'):
            # C'est directement un objet UnifiedEntity
            entity_name = entity_info.entity_name
            entity_type = entity_info.entity_type
            filepath = entity_info.filepath or 'Unknown'
            filename = entity_info.filename or (filepath.split('/')[-1] if '/' in filepath else filepath)
            start_line = entity_info.start_line or 0
            end_line = entity_info.end_line or 0
        else:
            # Fallback pour formats non reconnus
            entity_name = str(entity_info)
            entity_type = 'Unknown'
            filepath = 'Unknown'
            filename = 'Unknown'
            start_line = 0
            end_line = 0

        # Créer un ID unique pour cette source
        self.reference_counter += 1
        ref_id = f"S{self.reference_counter}"

        # Créer la référence
        source_ref = SourceReference(
            entity_name=entity_name,
            entity_type=entity_type,
            filepath=filepath,
            filename=filename,
            start_line=start_line,
            end_line=end_line,
            source_type=source_type,
            tool_used=tool_used,
            reference_id=ref_id
        )

        self.sources_used[ref_id] = source_ref
        return ref_id

    def _determine_explorer_type(self, entity_name: str = None, entity_type: str = None) -> str:
        """Détermine quel explorateur utiliser basé sur le contexte."""

        # Types spécifiquement Jupyter
        jupyter_types = {'notebook', 'code_cell', 'markdown_cell', 'class', 'async function'}

        # Types spécifiquement Fortran
        fortran_types = {'module', 'subroutine', 'program', 'type_definition', 'parameter'}

        if entity_type:
            if entity_type in jupyter_types and self.jupyter_explorer:
                return "jupyter"
            elif entity_type in fortran_types and self.fortran_explorer:
                return "fortran"

        # Heuristiques sur le nom
        if entity_name:
            if any(keyword in entity_name.lower() for keyword in ['cell', 'notebook', '.ipynb']):
                if self.jupyter_explorer:
                    return "jupyter"
            elif any(keyword in entity_name.lower() for keyword in ['.f90', '.f95', 'module_', 'subroutine_']):
                if self.fortran_explorer:
                    return "fortran"

        # Par défaut, utiliser le premier disponible
        if self.fortran_explorer:
            return "fortran"
        elif self.jupyter_explorer:
            return "jupyter"

        return "unknown"

    async def run(self, user_query: str, use_memory: bool = True) -> AgentResponse:
        """Exécute l'agent et retourne une réponse complètement structurée."""

        start_time = time.time()
        session_id = str(uuid.uuid4())[:8]

        print("=" * 80)
        print(f"🚀 DÉMARRAGE DE L'AGENT UNIFIÉ - Session {session_id}")
        print(f"📝 Requête: {user_query}")
        print("=" * 80)

        # Initialisation de la réponse structurée
        response = AgentResponse(
            answer="",
            status="in_progress",
            query=user_query,
            session_id=session_id,
            timestamp=datetime.now(),
            execution_time_total_ms=0.0,
            steps_taken=0,
            max_steps=self.max_steps,
            explorers_used=[]
        )

        # Déterminer quels explorateurs sont disponibles
        if self.fortran_explorer:
            response.explorers_used.append("fortran")
        if self.jupyter_explorer:
            response.explorers_used.append("jupyter")

        # Reset des sources pour chaque nouvelle requête (sauf si mémoire)
        if not use_memory:
            self.sources_used = {}
            self.reference_counter = 0
            print("🔄 Sources réinitialisées (nouvelle session)")
        else:
            print(f"🧠 Session continue - Sources déjà trackées: {len(self.sources_used)}")

        # Gestion de la mémoire
        if use_memory and self.conversation_history:
            self.conversation_history.append({"role": "user", "content": user_query})
            history = self.conversation_history.copy()
            print(f"📚 Mémoire persistante: {len(history)} messages dans l'historique")
        else:
            history = [{"role": "user", "content": user_query}]
            self.conversation_history = history.copy()
            print("🆕 Nouvelle session démarrée")

        for i in range(self.max_steps):
            step_start = time.time()

            print(f"\n" + "─" * 80)
            print(f"🔄 TOUR {i + 1}/{self.max_steps}")
            print("─" * 80)

            messages_for_llm = [{"role": "system", "content": self.system_prompt}] + history

            print("🤖 Envoi de la requête au LLM...")
            print(f"📊 Contexte: {len(messages_for_llm)} messages")

            decision_dict = await self.llm.generate_response(
                messages=messages_for_llm,
                pydantic_model=AgentDecision
            )

            if not decision_dict:
                response.status = "error"
                response.error_details = "Le LLM n'a pas retourné de décision."
                response.execution_time_total_ms = (time.time() - start_time) * 1000
                response.steps_taken = i
                print(f"❌ {response.error_details}")
                return response

            try:
                decision = AgentDecision.model_validate(decision_dict)

                # 🔍 ENREGISTRER L'ÉTAPE DE RAISONNEMENT
                print(f"🧠 RÉFLEXION DE L'AGENT:")
                print(f"   💭 Pensée: {decision.thought}")

                if decision.plan:
                    print(f"   📋 Plan défini:")
                    for idx, step in enumerate(decision.plan, 1):
                        print(f"      {idx}. {step}")

                    # Enregistrer le plan final s'il est défini pour la première fois
                    if not response.final_plan_executed:
                        response.final_plan_executed = decision.plan.copy()

                if decision.working_memory_candidates:
                    print(f"   🎯 Mémoire de travail: {len(decision.working_memory_candidates)} candidats")
                    for idx, candidate in enumerate(decision.working_memory_candidates, 1):
                        print(f"      {idx}. {candidate}")

                print(f"   🛠️  Outil choisi: {decision.tool_name}")
                print(f"   ⚙️  Arguments: {decision.arguments.model_dump(exclude_none=True)}")

            except ValidationError as e:
                response.status = "error"
                response.error_details = f"Erreur de validation: {e}"
                response.execution_time_total_ms = (time.time() - start_time) * 1000
                response.steps_taken = i
                print(f"❌ {response.error_details}")
                return response

            history.append({"role": "assistant", "content": decision.model_dump_json()})

            # Cas spéciaux
            if decision.tool_name == "ask_for_clarification":
                response.status = "clarification_needed"
                response.clarification_question = decision.arguments.question
                response.execution_time_total_ms = (time.time() - start_time) * 1000
                response.steps_taken = i + 1

                print("❓ L'agent demande une clarification")
                print(f"   Question: {decision.arguments.question}")
                return response

            if decision.tool_name == "final_answer":
                response.status = "success"
                response.answer = decision.arguments.text
                response.execution_time_total_ms = (time.time() - start_time) * 1000
                response.steps_taken = i + 1

                # Enrichir la réponse avec les informations collectées
                response.sources_consulted = list(self.sources_used.values())
                response._analyze_entities_and_findings()

                print("✅ L'agent génère sa réponse finale")
                print(f"📚 Sources ajoutées: {len(response.sources_consulted)} références")

                if use_memory:
                    self.conversation_history = history.copy()

                print("=" * 80)
                print("✅ RÉPONSE STRUCTURÉE GÉNÉRÉE")
                print("=" * 80)
                return response

            # Exécution de l'outil
            print(f"\n🔧 EXÉCUTION DE L'OUTIL: {decision.tool_name}")
            print("─" * 50)

            sources_before = len(self.sources_used)
            tool_result = await self._execute_tool_detailed(decision.tool_name, decision.arguments)
            sources_after = len(self.sources_used)

            step_execution_time = (time.time() - step_start) * 1000

            # Créer le résumé du résultat de l'outil
            tool_result_summary = self._create_tool_result_summary(tool_result)

            # Sources découvertes dans cette étape
            new_sources = []
            if sources_after > sources_before:
                new_sources_count = sources_after - sources_before
                recent_sources = list(self.sources_used.values())[-new_sources_count:]
                new_sources = [s.reference_id for s in recent_sources]
                print(f"📝 Nouvelles sources trackées: {new_sources_count}")
                for source in recent_sources:
                    print(f"   ➕ {source.get_citation()}")

            # Enregistrer l'étape
            reasoning_step = AgentStep(
                step_number=i + 1,
                thought=decision.thought,
                plan=decision.plan,
                working_memory=decision.working_memory_candidates,
                tool_name=decision.tool_name,
                tool_arguments=decision.arguments.model_dump(exclude_none=True),
                tool_result_summary=tool_result_summary,
                sources_discovered=new_sources,
                execution_time_ms=step_execution_time
            )

            response.reasoning_steps.append(reasoning_step)

            # Mettre à jour les statistiques des outils
            response.tools_used[decision.tool_name] = response.tools_used.get(decision.tool_name, 0) + 1

            formatted_result = self._format_tool_result_for_llm(tool_result)
            print(f"📤 Résultat formaté pour le LLM ({len(formatted_result)} caractères)")

            history.append({"role": "user", "content": formatted_result})

            print(f"📊 État actuel:")
            print(f"   💬 Messages dans l'historique: {len(history)}")
            print(f"   📚 Sources trackées: {len(self.sources_used)}")

        # Timeout
        response.status = "timeout"
        response.answer = "Je n'ai pas pu aboutir à une réponse finale dans le nombre d'étapes imparti."
        response.execution_time_total_ms = (time.time() - start_time) * 1000
        response.steps_taken = self.max_steps
        response.sources_consulted = list(self.sources_used.values())
        response._analyze_entities_and_findings()

        if use_memory:
            self.conversation_history = history.copy()

        print("⏰ TIMEOUT ATTEINT")
        print(f"📚 Sources consultées malgré le timeout: {len(response.sources_consulted)}")

        return response

    def _create_tool_result_summary(self, tool_result: Any) -> str:
        """Crée un résumé textuel du résultat d'un outil."""
        if isinstance(tool_result, list):
            return f"Liste de {len(tool_result)} élément(s)"
        elif isinstance(tool_result, dict):
            if tool_result.get('error'):
                return f"Erreur: {tool_result['error']}"
            elif 'entity_name' in tool_result:
                return f"Rapport pour {tool_result['entity_name']}"
            else:
                return f"Dictionnaire avec {len(tool_result)} clé(s)"
        else:
            summary = str(tool_result)[:100]
            return f"Résultat textuel: {summary}{'...' if len(str(tool_result)) > 100 else ''}"

    async def _execute_tool_detailed(self, tool_name: str, args: BaseModel) -> Any:
        """Exécute l'outil avec logs détaillés."""
        args_dict = args.model_dump(exclude_none=True)

        print(f"🎯 Outil: {tool_name}")
        print(f"📋 Arguments: {args_dict}")

        try:
            if tool_name == "semantic_search":
                return await self._execute_semantic_search(**args_dict)

            if tool_name == "get_notebook_overview":
                if not self.jupyter_explorer:
                    error = "Erreur: Aucun explorateur Jupyter disponible."
                    print(f"❌ {error}")
                    return error

                print("📓 Exécution sur explorateur Jupyter...")
                result = await self.jupyter_explorer.get_notebook_overview(**args_dict)

                # Analyser le résultat
                if isinstance(result, dict):
                    if result.get('error'):
                        print(f"❌ Erreur: {result['error']}")
                    else:
                        print(f"✅ Vue d'ensemble générée pour: {result.get('notebook_name', 'N/A')}")
                        stats = result.get('statistics', {})
                        print(f"   📊 Statistiques: {stats}")
                        self._add_source_reference(result, "jupyter", "get_notebook_overview")

                return result

            # Pour les autres outils, déterminer l'explorateur à utiliser
            entity_name = args_dict.get('entity_name')
            entity_type = args_dict.get('entity_type')

            print(f"🔍 Détermination de l'explorateur...")
            print(f"   🎯 Entité cible: {entity_name or 'N/A'}")
            print(f"   📝 Type: {entity_type or 'N/A'}")

            explorer_type = self._determine_explorer_type(entity_name, entity_type)
            print(f"   🤖 Explorateur choisi: {explorer_type}")

            if explorer_type == "jupyter" and self.jupyter_explorer:
                print("📓 Exécution sur explorateur Jupyter...")
                result = await self._execute_tool_on_explorer_detailed(tool_name, args_dict, self.jupyter_explorer,
                                                                       "jupyter")
                self._track_sources_from_result(result, "jupyter", tool_name)
                return result

            elif explorer_type == "fortran" and self.fortran_explorer:
                print("🔧 Exécution sur explorateur Fortran...")
                result = await self._execute_tool_on_explorer_detailed(tool_name, args_dict, self.fortran_explorer,
                                                                       "fortran")
                self._track_sources_from_result(result, "fortran", tool_name)
                return result

            else:
                # Essayer les deux si disponibles
                print("🔄 Tentative sur les deux explorateurs...")
                results = []

                if self.fortran_explorer:
                    try:
                        print("   🔧 Test sur Fortran...")
                        result = await self._execute_tool_on_explorer_detailed(tool_name, args_dict,
                                                                               self.fortran_explorer, "fortran")
                        if result and result != "Entité non trouvée":
                            print(f"   ✅ Résultat Fortran obtenu")
                            self._track_sources_from_result(result, "fortran", tool_name)
                            results.append({"type": "fortran", "result": result})
                        else:
                            print(f"   ❌ Aucun résultat Fortran")
                    except Exception as e:
                        print(f"   ❌ Erreur Fortran: {e}")

                if self.jupyter_explorer:
                    try:
                        print("   📓 Test sur Jupyter...")
                        result = await self._execute_tool_on_explorer_detailed(tool_name, args_dict,
                                                                               self.jupyter_explorer, "jupyter")
                        if result and result != "Entité non trouvée":
                            print(f"   ✅ Résultat Jupyter obtenu")
                            self._track_sources_from_result(result, "jupyter", tool_name)
                            results.append({"type": "jupyter", "result": result})
                        else:
                            print(f"   ❌ Aucun résultat Jupyter")
                    except Exception as e:
                        print(f"   ❌ Erreur Jupyter: {e}")

                if results:
                    print(f"📋 Résultats combinés: {len(results)} types de résultats")
                    return results
                else:
                    no_result = "Aucune entité trouvée dans les deux types de code."
                    print(f"❌ {no_result}")
                    return no_result

        except Exception as e:
            error_msg = f"Erreur lors de l'exécution de l'outil: {e}"
            logger.error(f"Erreur lors de l'exécution de l'outil '{tool_name}': {e}", exc_info=True)
            print(f"❌ {error_msg}")
            return error_msg

    async def _execute_semantic_search(self, query: str, max_results: int = 5, min_confidence: float = 0.3) -> Dict[
        str, Any]:
        """Exécute une recherche sémantique dans les notebooks."""
        print(f"🔍 Recherche sémantique pour: '{query}'")
        print(f"   📊 Paramètres: max_results={max_results}, min_confidence={min_confidence}")

        if len(self.semantic_retriever.chunks) == 0:
            error_msg = "Index sémantique vide. Aucun notebook indexé."
            print(f"   ❌ {error_msg}")
            return {"error": error_msg}

        results = self.semantic_retriever.query(query, k=max_results, min_score=min_confidence)

        if not results:
            print(f"   ❌ Aucun résultat au-dessus du seuil de confiance {min_confidence}")
            return {
                "query": query,
                "results": [],
                "total_indexed_chunks": len(self.semantic_retriever.chunks),
                "message": f"Aucun contenu pertinent trouvé (seuil: {min_confidence})"
            }

        print(f"   ✅ {len(results)} résultats trouvés")
        for i, result in enumerate(results, 1):
            score = result["similarity_score"]
            source = result["source_filename"]
            tokens = result.get("tokens", "?")
            print(f"      {i}. {source} - Score: {score:.3f} ({tokens} tokens)")

        # Créer des SourceReference pour le tracking
        sources_created = []
        for result in results:
            source_ref = SourceReference(
                entity_name=f"semantic_chunk_{len(self.sources_used) + 1}",
                entity_type="semantic_content",
                filepath=result.get("source_file", "unknown"),
                filename=result.get("source_filename", "unknown"),
                start_line=1,
                end_line=1,  # Les chunks n'ont pas de lignes précises
                source_type="jupyter",
                tool_used="semantic_search",
                reference_id=f"S{self.reference_counter + len(sources_created) + 1}"
            )
            sources_created.append(source_ref.reference_id)
            self.sources_used[source_ref.reference_id] = source_ref

        self.reference_counter += len(sources_created)

        return {
            "query": query,
            "results": results,
            "total_indexed_chunks": len(self.semantic_retriever.chunks),
            "sources_tracked": sources_created
        }

    async def _execute_tool_on_explorer_detailed(self, tool_name: str, args_dict: Dict, explorer,
                                                 explorer_type: str) -> Any:
        """Exécute un outil sur un explorateur spécifique avec logs détaillés."""

        print(f"⚙️  Exécution de '{tool_name}' sur explorateur {explorer_type}")

        try:
            if tool_name == "find_entity_by_name":
                entity_name = args_dict['entity_name']
                print(f"   🔍 Recherche de l'entité: {entity_name}")
                result = await explorer.find_entity_by_name(**args_dict)

                if isinstance(result, dict) and result.get('entity_name'):
                    print(f"   ✅ Entité trouvée: {result['entity_name']} ({result.get('entity_type', 'N/A')})")
                else:
                    print(f"   ❌ Entité non trouvée")

                return result

            elif tool_name == "list_entities":
                print(f"   📋 Recherche d'entités par critères...")
                for key, value in args_dict.items():
                    if value:
                        print(f"      {key}: {value}")

                result = await explorer.find_entities_by_criteria(**args_dict)

                if isinstance(result, list):
                    print(f"   📊 {len(result)} entité(s) trouvée(s)")
                    for i, item in enumerate(result[:5]):  # Afficher les 5 premières
                        # ✅ CORRECTION : Gérer correctement les objets UnifiedEntity
                        if isinstance(item, dict) and 'entity' in item:
                            entity = item['entity']
                            score = item.get('score', 0)

                            # Accéder aux attributs de l'objet UnifiedEntity
                            if hasattr(entity, 'entity_name'):
                                entity_name = entity.entity_name
                                entity_type = entity.entity_type
                            else:
                                # Fallback pour les dictionnaires
                                entity_name = entity.get('entity_name', entity.get('name', 'N/A'))
                                entity_type = entity.get('entity_type', entity.get('type', 'N/A'))

                            print(f"      {i + 1}. {entity_name} ({entity_type}) - score: {score:.1f}")
                        else:
                            # Format inattendu
                            print(f"      {i + 1}. Format inattendu: {type(item)}")

                    if len(result) > 5:
                        print(f"      ... et {len(result) - 5} autres")
                else:
                    print(f"   ❌ Résultat inattendu: {type(result)}")

                return result

            elif tool_name == "get_entity_report":
                entity_name = args_dict['entity_name']
                include_source = args_dict.get('include_source_code', False)
                print(f"   📄 Génération du rapport pour: {entity_name}")
                print(f"   📝 Code source inclus: {include_source}")

                result = await explorer.get_full_report(**args_dict)

                if isinstance(result, dict):
                    if result.get('error'):
                        print(f"   ❌ Erreur: {result['error']}")
                    else:
                        print(f"   ✅ Rapport généré pour: {result.get('entity_name', 'N/A')}")
                        summary = result.get('summary', {})
                        print(f"      Type: {summary.get('type', 'N/A')}")
                        print(f"      Fichier: {summary.get('filepath', 'N/A')}")
                        print(f"      Lignes: {summary.get('start_line', 'N/A')}-{summary.get('end_line', 'N/A')}")

                        # Informations sur les relations
                        outgoing = result.get('outgoing_relations', {})
                        incoming = result.get('incoming_relations', [])

                        if explorer_type == "fortran":
                            calls = outgoing.get('called_functions_or_subroutines', [])
                            deps = outgoing.get('module_dependencies (USE)', [])
                            print(f"      Appelle: {len(calls)} fonctions/subroutines")
                            print(f"      Utilise: {len(deps)} modules")
                        else:  # jupyter
                            imports = outgoing.get('imports', [])
                            calls = outgoing.get('function_calls', [])
                            print(f"      Imports: {len(imports)}")
                            print(f"      Appels: {len(calls)} fonctions")

                        print(f"      Référencé par: {len(incoming)} entité(s)")

                return result

            elif tool_name == "get_relations":
                entity_name = args_dict["entity_name"]
                relation_type = args_dict["relation_type"]
                print(f"   🔗 Analyse des relations '{relation_type}' pour: {entity_name}")

                if relation_type in ["callers", "references"]:
                    if explorer_type == "jupyter":
                        result = await explorer.get_references(entity_name)
                    else:  # Fortran
                        result = await explorer.get_callers(entity_name)

                elif relation_type in ["callees", "imports"]:
                    entity = await explorer.em.find_entity(entity_name)
                    if not entity:
                        error = f"Entité '{entity_name}' non trouvée."
                        print(f"   ❌ {error}")
                        return error

                    if explorer_type == "jupyter":
                        result = explorer.get_imports_and_calls(entity)
                    else:  # Fortran
                        result = explorer.get_callees_and_dependencies(entity)

                # Analyser le résultat
                if isinstance(result, list):
                    print(f"   📊 {len(result)} relation(s) trouvée(s)")
                    for i, item in enumerate(result[:3]):  # Afficher les 3 premières
                        if isinstance(item, dict):
                            name = item.get('name', 'N/A')
                            item_type = item.get('type', 'N/A')
                            print(f"      {i + 1}. {name} ({item_type})")
                    if len(result) > 3:
                        print(f"      ... et {len(result) - 3} autres")
                elif isinstance(result, dict):
                    print(f"   📊 Relations structurées trouvées")
                    for key, value in result.items():
                        if isinstance(value, list):
                            print(f"      {key}: {len(value)} élément(s)")
                        else:
                            print(f"      {key}: {value}")

                return result

            else:
                error = f"Erreur : Outil inconnu '{tool_name}'."
                print(f"   ❌ {error}")
                return error

        except Exception as e:
            error_msg = f"Erreur dans l'explorateur {explorer_type}: {e}"
            print(f"   ❌ {error_msg}")
            logger.error(f"Erreur {explorer_type} pour '{tool_name}': {e}", exc_info=True)
            return error_msg

    def _track_sources_from_result(self, result: Any, source_type: str, tool_name: str):
        """Tracke les sources depuis le résultat d'un outil."""
        try:
            if isinstance(result, list):
                # Cas des listes d'entités (list_entities, get_relations, etc.)
                for item in result:
                    if isinstance(item, dict):
                        self._add_source_reference(item, source_type, tool_name)
                    elif hasattr(item, 'entity_name'):  # ✅ NOUVEAU : Objet UnifiedEntity direct
                        self._add_source_reference(item, source_type, tool_name)
            elif isinstance(result, dict) and not result.get('error'):
                # Cas des rapports d'entité ou résultats uniques
                self._add_source_reference(result, source_type, tool_name)
            elif hasattr(result, 'entity_name'):  # ✅ NOUVEAU : Objet UnifiedEntity direct
                self._add_source_reference(result, source_type, tool_name)
        except Exception as e:
            logger.debug(f"Erreur tracking sources: {e}")

    def _format_tool_result_for_llm(self, result: Any) -> str:
        """Formate le résultat d'un outil pour le LLM."""
        processed_result = result

        # Gestion des listes de résultats (fortran + jupyter)
        if isinstance(result, list) and result and isinstance(result[0], dict) and 'type' in result[0]:
            # C'est un résultat mixte
            formatted_parts = []
            for part in result:
                part_type = part['type']
                part_result = part['result']
                formatted_parts.append(f"=== Résultats {part_type.upper()} ===")
                formatted_parts.append(self._format_single_result(part_result))
            return "\n".join(formatted_parts)

        return self._format_single_result(result)

    def _format_single_result(self, result: Any) -> str:
        """Formate un résultat unique."""
        processed_result = result

        # Traitement des entités UnifiedEntity
        if isinstance(result, list) and result and isinstance(result[0], dict) and 'entity' in result[0]:
            if isinstance(result[0]['entity'], UnifiedEntity):
                processed_result = [
                    {**item, "entity": item["entity"].to_dict()}
                    for item in result
                ]
        elif isinstance(result, dict) and 'entity' in result and isinstance(result['entity'], UnifiedEntity):
            processed_result = {**result, "entity": result["entity"].to_dict()}

        try:
            json_str = json.dumps(processed_result, indent=2)
            if len(json_str) > 8000:
                return f"Tool Result (tronqué): {json_str[:8000]}..."
            return f"Tool Result: {json_str}"
        except TypeError as e:
            logger.error(f"Erreur de sérialisation: {e}")
            return f"Tool Result: {str(result)}"

    def _generate_sources_section(self) -> str:
        """Génère la section des sources consultées."""
        if not self.sources_used:
            return "\n\n**Aucune source spécifique consultée.**"

        sources_section = "\n\n## 📚 Sources consultées :\n"

        for ref_id, source_ref in self.sources_used.items():
            sources_section += f"\n{source_ref.get_citation()}"

        return sources_section

    def clear_memory(self):
        """Efface la mémoire de conversation et les sources."""
        self.conversation_history = []
        self.sources_used = {}
        self.reference_counter = 0
        print("🧠 Mémoire de l'agent et sources effacées.")

    def get_memory_summary(self) -> str:
        """Retourne un résumé de la mémoire avec info sur les sources."""
        if not self.conversation_history:
            return "Aucune mémoire conservée."

        user_messages = [msg["content"] for msg in self.conversation_history if msg["role"] == "user"]
        sources_info = f", {len(self.sources_used)} sources trackées" if self.sources_used else ""
        return f"Mémoire : {len(self.conversation_history)} messages{sources_info}, dernières requêtes : {user_messages[-3:]}"

    def get_sources_used(self) -> List[SourceReference]:
        """Retourne la liste des sources utilisées dans la session courante."""
        return list(self.sources_used.values())