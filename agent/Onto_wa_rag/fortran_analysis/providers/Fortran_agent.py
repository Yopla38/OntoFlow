"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

import json
import logging
import re
from typing import Dict, Any, List, Optional, Literal, Union

from pydantic import BaseModel, Field, ValidationError

from ..core.entity_manager import UnifiedEntity
from .consult import FortranEntityExplorer
from ...provider.llm_providers import LLMProvider

logger = logging.getLogger(__name__)


class AgentAskClarificationArgs(BaseModel):
    """Arguments pour poser une question de clarification à l'utilisateur afin de lever une ambiguïté."""
    question: str = Field(..., description="La question précise à poser à l'utilisateur.")


class AgentFindEntityByNameArgs(BaseModel):
    """Arguments pour trouver une entité par son nom (exact ou approximatif). C'est l'outil le plus direct."""
    entity_name: str = Field(..., description="The exact or approximate name of the entity to find.")


class AgentListEntitiesArgs(BaseModel):
    """Arguments pour trouver des entités par attributs ou par concept."""
    entity_type: Optional[str] = Field(None, description="The type of entity (e.g., 'subroutine', 'function').")
    entity_name: Optional[str] = Field(None, description="An approximate name of the entity.")
    filename: Optional[str] = Field(None, description="The name of the file to search within.")
    parent_entity: Optional[str] = Field(None, description="The name of the parent entity.")
    dependencies: Optional[str] = Field(None, description="Name of a module used via 'USE'.")
    # ... autres filtres si nécessaire ...

    # Le paramètre le plus important pour les questions sémantiques
    detected_concept: Optional[str] = Field(None,
                                            description="**Use this for any conceptual, semantic, or 'how-to' question.** Describe the logic or purpose to find (e.g., 'memory allocation', 'error handling logic').")


class AgentGetEntityReportArgs(BaseModel):
    """Arguments to get a detailed report for a single entity."""
    entity_name: str = Field(..., description="The exact name of the entity to get a full report for.")
    include_source_code: bool = Field(False, description="Whether to include source code. Default to false.")


class AgentGetRelationsArgs(BaseModel):
    """Arguments to get relationships for a single entity."""
    entity_name: str = Field(..., description="The name of the central entity for the relation query.")
    relation_type: Literal['callers', 'callees'] = Field(...,
                                                         description="The type of relation: 'callers' or 'callees'.")


class AgentFinalAnswerArgs(BaseModel):
    """Arguments for the final answer, which concludes the task."""
    text: str = Field(..., description="The final, comprehensive answer to the user's original query.")


class AgentDecision(BaseModel):
    """Définit la pensée et l'action structurée de l'agent."""
    """Définit la pensée, le plan et l'action structurée de l'agent."""
    thought: str = Field(...,
                         description="Ma réflexion sur l'état actuel, ce que je viens d'apprendre, et ce que je dois faire pour exécuter la prochaine étape de mon plan.")
    plan: Optional[List[str]] = Field(None,
                                      description="[À DÉFINIR AU 1ER TOUR SEULEMENT pour les requêtes complexes] La liste numérotée des étapes de haut niveau pour répondre à la requête. Ne pas inclure cette clé dans les tours suivants.")
    # Mémoire de travail
    working_memory_candidates: Optional[List[str]] = Field(None,
                                                           description="La liste des noms d'entités qu'il reste à examiner. Je dois peupler cette liste après une découverte, puis la consommer un par un à chaque étape.")
    tool_name: Literal[
        "find_entity_by_name",
        "list_entities",
        "get_entity_report",
        "get_relations",
        "ask_for_clarification",
        "final_answer"
    ] = Field(...)

    arguments: Union[
        AgentFindEntityByNameArgs,
        AgentListEntitiesArgs,
        AgentGetEntityReportArgs,
        AgentGetRelationsArgs,
        AgentAskClarificationArgs,
        AgentFinalAnswerArgs
    ] = Field(...)


class FortranAgent:
    def __init__(self, llm_provider: LLMProvider, explorer: FortranEntityExplorer, max_steps: int = 7):
        """
        Initialise l'agent.

        Args:
            llm_provider: Le fournisseur de LLM pour générer les décisions.
            explorer: L'explorateur d'entités pour exécuter les outils.
            max_steps: Nombre maximum de tours pour éviter les boucles infinies.
        """
        self.llm = llm_provider
        self.explorer = explorer
        self.max_steps = max_steps
        self.system_prompt = self.build_agent_system_prompt()

        # ← AJOUT : Historique persistant
        self.conversation_history: List[Dict[str, str]] = []

    def build_agent_system_prompt(self) -> str:
        """
        Construit le prompt système optimisé pour un agent analyste, systématique, robuste
        et doté d'une mémoire de travail explicite.
        """

        mission_and_process = """Tu es un agent expert autonome, spécialisé dans l'analyse de code jupyter. Ton travail doit être systématique, rigoureux et complet. Tu ne dois jamais te contenter d'une réponse partielle si une réponse complète est possible.

    **PROCESSUS DE RAISONNEMENT OBLIGATOIRE :**

    1.  **Rigueur et Exhaustivité :** Ta mission principale est de fournir des réponses COMPLÈTES. Si une question implique de trouver "les" entités, tu dois toutes les trouver et les analyser avant de conclure.
    2.  **Le Workflow de la Mémoire de Travail :** C'est ta stratégie centrale pour les recherches exhaustives.
        a. **Planifier :** Pour toute requête complexe, crée un `plan` initial.
        b. **Découvrir et Stocker :** Utilise `list_entities` pour découvrir une liste de candidats. Dès que tu obtiens cette liste, tu DOIS la stocker dans le champ `working_memory_candidates`.
        c. **Traiter Systématiquement :** À chaque tour, prends le PREMIER élément de `working_memory_candidates`, exécute une action dessus (ex: `get_entity_report`), et dans ta réponse suivante, renvoie la liste MISE À JOUR (l'élément traité en moins).
        d. **Conclure :** Ne génère une `final_answer` que lorsque `working_memory_candidates` est vide ou que tu as traité tous les candidats pertinents.
    3.  **Robustesse (Clarification) :** Si la requête d'un utilisateur est ambiguë (ex: "la routine principale", "la partie importante"), ne fais PAS de supposition. Utilise l'outil `ask_for_clarification` pour demander des précisions.
    4.  **Efficacité :** Pour les questions simples et directes sur une entité nommée (ex: "que fait 'Periodic_Kernel' ?"), utilise la voie la plus rapide : `find_entity_by_name` -> `get_entity_report` -> `final_answer`. N'utilise pas la mémoire de travail si ce n'est pas nécessaire."""

        tool_descriptions = """
    <outils>
        - `find_entity_by_name`: **OUTIL DE DÉMARRAGE RAPIDE.** À utiliser en premier pour toute question sur une entité nommée. C'est le moyen le plus direct.
        - `get_entity_report`: **OUTIL D'INSPECTION.** Une fois une entité localisée, utilise cet outil pour lire son rapport détaillé (code, relations, etc.).
        - `list_entities`: **OUTIL DE DÉCOUVERTE EXHAUSTIVE.** À utiliser pour obtenir une liste complète de candidats basée sur un concept sémantique (`detected_concept`) ou des filtres. Le résultat de cet outil doit peupler la `working_memory_candidates`.
        - `get_relations`: **OUTIL D'ENQUÊTE.** Pour comprendre les connexions (appels) d'une entité spécifique.
        - `ask_for_clarification`: **OUTIL DE DIALOGUE.** Si la requête est vague ou incomplète, utilise cet outil pour poser une question précise à l'utilisateur.
        - `final_answer`: **OUTIL DE CONCLUSION.** À n'utiliser qu'en toute fin de processus, lorsque toutes les étapes du plan sont complétées et que la mémoire de travail est traitée.
    </outils>
    """

        strategic_examples = """
    <exemples_strategiques>

        --- EXEMPLE 1: Recherche Exhaustive (Le cas FFT) ---
        ## User Query: "quelles entités de code gèrent le calcul de la FFT ?"

        ## Tour 1 (Planification et Découverte)
        {
            "thought": "La requête est exhaustive. Je vais établir un plan pour lister tous les candidats, puis les analyser un par un en utilisant ma mémoire de travail. Je commence par lister les sous-routines liées au concept de FFT.",
            "plan": ["1. Lister les candidats liés à la FFT.", "2. Stocker les candidats en mémoire.", "3. Analyser chaque candidat de la mémoire.", "4. Synthétiser la réponse finale."],
            "working_memory_candidates": null,
            "tool_name": "list_entities",
            "arguments": { "detected_concept": "Fast Fourier Transform", "entity_type": "subroutine" }
        }

        ## Tool Result: (Résultat de l'outil contenant "G_PoissonSolver", "Periodic_Kernel")

        ## Tour 2 (Stockage en Mémoire et Traitement du 1er Candidat)
        {
            "thought": "J'ai obtenu la liste des candidats. Conformément à mon plan, je la stocke dans ma mémoire de travail et je commence l'analyse du premier : 'G_PoissonSolver'.",
            "plan": null,
            "working_memory_candidates": ["G_PoissonSolver", "Periodic_Kernel"],
            "tool_name": "get_entity_report",
            "arguments": { "entity_name": "G_PoissonSolver" }
        }

        ## Tool Result: (Rapport sur G_PoissonSolver, montrant des appels à 'fftstp_sg')

        ## Tour 3 (Traitement du 2ème Candidat de la Mémoire)
        {
            "thought": "'G_PoissonSolver' est bien pertinent. Je le retire de ma mémoire et je passe au candidat suivant : 'Periodic_Kernel'. Je ne me laisse pas distraire par les autres routines mentionnées dans le rapport précédent, je suis ma liste.",
            "plan": null,
            "working_memory_candidates": ["Periodic_Kernel"],
            "tool_name": "get_entity_report",
            "arguments": { "entity_name": "Periodic_Kernel" }
        }

        ## Tool Result: (Rapport sur Periodic_Kernel, montrant des appels à 'fourtrans_isf')

        ## Tour 4 (Synthèse Finale)
        {
            "thought": "'Periodic_Kernel' est aussi un acteur majeur. Ma mémoire de travail est maintenant vide, j'ai analysé tous les candidats initiaux. Je peux synthétiser la réponse finale et complète.",
            "plan": null,
            "working_memory_candidates": [],
            "tool_name": "final_answer",
            "arguments": { "text": "Deux entités principales gèrent les calculs de FFT : 1) 'G_PoissonSolver' (dans PSolver_Core.f90) qui utilise la routine 'fftstp_sg'. 2) 'Periodic_Kernel' (dans Build_Kernel.f90) qui utilise la routine 'fourtrans_isf'." }
        }

        --- EXEMPLE 2: Question Simple et Directe ---
        ## User Query: "donne-moi un rapport sur la fonction 'f_free'"

        ## Tour 1 (Pas de plan, action directe)
        {
            "thought": "La requête est simple et porte sur une entité nommée. La voie la plus rapide est d'utiliser `find_entity_by_name`.",
            "plan": null,
            "working_memory_candidates": null,
            "tool_name": "find_entity_by_name",
            "arguments": { "entity_name": "f_free" }
        }

        ## Tool Result: (Résultat de l'outil trouvant l'entité 'f_free')

        ## Tour 2 (Inspection)
        {
            "thought": "L'entité a été trouvée. Je vais maintenant récupérer son rapport détaillé.",
            "plan": null,
            "working_memory_candidates": null,
            "tool_name": "get_entity_report",
            "arguments": { "entity_name": "f_free", "include_source_code": true }
        }

        --- EXEMPLE 3: Question Ambiguë ---
        ## User Query: "analyse la routine principale"

        ## Tour 1 (Détection d'ambiguïté)
        {
            "thought": "La requête est ambiguë. 'routine principale' peut signifier le programme principal, la routine la plus appelée, ou la plus coûteuse. Au lieu de deviner, je dois demander des précisions à l'utilisateur.",
            "plan": null,
            "working_memory_candidates": null,
            "tool_name": "ask_for_clarification",
            "arguments": { "question": "Le terme 'routine principale' n'est pas clair. Cherchez-vous le point d'entrée du programme (un 'PROGRAM'), la routine la plus utilisée, ou celle qui consomme le plus de ressources ?" }
        }

        --- EXEMPLE 4: Requête de Haut Niveau (Résumé de projet) ---
        ## User Query: "Fais-moi un résumé du projet."

        ## Tour 1 (Planification et découverte des modules)
        {
            "thought": "C'est une requête de haut niveau. Mon plan est d'identifier les modules clés, les analyser, puis synthétiser. Je commence par lister tous les modules pour avoir une vue d'ensemble.",
            "plan": ["1. Lister tous les modules.", "2. Identifier et stocker les 2-3 modules les plus importants en mémoire.", "3. Analyser chaque module en mémoire.", "4. Synthétiser un résumé."],
            "working_memory_candidates": null,
            "tool_name": "list_entities",
            "arguments": { "entity_type": "module" }
        }

        ## Tool Result: (Liste des modules, ex: 'PSbase', 'psolver_workarrays', 'PScore')

        ## Tour 2 (Stockage en mémoire des modules clés)
        {
            "thought": "J'ai la liste des modules. 'PScore' et 'PSbase' semblent centraux. Je les place dans ma mémoire de travail pour les analyser.",
            "plan": null,
            "working_memory_candidates": ["PScore", "PSbase"],
            "tool_name": "get_entity_report",
            "arguments": { "entity_name": "PScore" }
        }

    </exemples_strategiques>
    """

        return f"{mission_and_process}\n\n{tool_descriptions}\n\n{strategic_examples}\n\nMaintenant, commence."

    async def _execute_tool(self, tool_name: str, args: BaseModel) -> Any:
        """Exécute l'outil avec des arguments Pydantic déjà validés."""
        print(f"🛠️ Exécution de l'outil '{tool_name}'")
        args_dict = args.model_dump(exclude_none=True)  # exclude_none est une bonne pratique
        try:
            if tool_name == "find_entity_by_name":
                return await self.explorer.find_entity_by_name(**args_dict)

            elif tool_name == "list_entities":
                # Cet appel unique gère maintenant la recherche par attributs ET sémantique
                return await self.explorer.find_entities_by_criteria(**args_dict)

            elif tool_name == "get_entity_report":
                return await self.explorer.get_full_report(**args_dict)

            elif tool_name == "get_relations":
                entity_name = args_dict["entity_name"]
                relation_type = args_dict["relation_type"]
                if relation_type == "callers":
                    return await self.explorer.get_callers(entity_name)
                else:  # 'callees'
                    entity = await self.explorer.em.find_entity(entity_name)
                    if not entity: return f"Entité '{entity_name}' non trouvée."
                    return self.explorer.get_callees_and_dependencies(entity)
            else:
                return f"Erreur : Outil inconnu '{tool_name}'."

        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de l'outil '{tool_name}': {e}", exc_info=True)
            return f"Erreur lors de l'exécution de l'outil: {e}"

    def _format_tool_result_for_llm(self, result: Any) -> str:
        """
        Formate le résultat d'un outil pour qu'il soit compréhensible par le LLM
        et garantit qu'il est sérialisable en JSON.
        """
        processed_result = result

        # Si le résultat est une liste (cas de find_entities_by_criteria),
        # on parcourt la liste pour convertir chaque UnifiedEntity.
        if isinstance(result, list) and result and 'entity' in result[0] and isinstance(result[0]['entity'],
                                                                                        UnifiedEntity):
            processed_result = [
                {**item, "entity": item["entity"].to_dict()}
                for item in result
            ]
        # Si le résultat est un dict contenant directement une UnifiedEntity
        elif isinstance(result, dict) and 'entity' in result and isinstance(result['entity'], UnifiedEntity):
            processed_result = {**result, "entity": result["entity"].to_dict()}

        # Maintenant, on peut sérialiser
        try:
            json_str = json.dumps(processed_result, indent=2)
            if len(json_str) > 8000:  # Tronquer les résultats très longs
                return f"Tool Result (tronqué car trop long): {json_str[:8000]}..."
            return f"Tool Result: {json_str}"
        except TypeError as e:
            logger.error(f"Erreur de sérialisation finale non gérée : {e}")
            return f"Tool Result: Erreur de conversion du résultat en JSON. {str(result)}"

    async def run(self, user_query: str, use_memory: bool = True) -> str:
        # Si on utilise la mémoire, on continue l'historique existant
        if use_memory and self.conversation_history:
            # Ajouter la nouvelle requête à l'historique existant
            self.conversation_history.append({"role": "user", "content": user_query})
            history = self.conversation_history.copy()
        else:
            # Nouvelle session
            history = [{"role": "user", "content": user_query}]
            self.conversation_history = history.copy()

        for i in range(self.max_steps):
            print(f"--- Tour de l'Agent {i + 1}/{self.max_steps} ---")

            messages_for_llm = [{"role": "system", "content": self.system_prompt}] + history

            decision_dict = await self.llm.generate_response(
                messages=messages_for_llm,
                pydantic_model=AgentDecision
            )

            if not decision_dict:
                return "Erreur: Le LLM n'a pas retourné de décision."

            try:
                decision = AgentDecision.model_validate(decision_dict)
                print(f"🤔 Décision validée du LLM: {decision.model_dump_json(indent=2)}")
            except ValidationError as e:
                return f"Erreur de validation de la décision du LLM : {e}"

            history.append({"role": "assistant", "content": decision.model_dump_json()})

            # Intercepter la demande de clarification AVANT d'appeler _execute_tool
            if decision.tool_name == "ask_for_clarification":
                logger.info("❓ L'agent demande une clarification.")
                # pour demander une nouvelle entrée à l'utilisateur.
                return f"CLARIFICATION_NEEDED: {decision.arguments.question}"

            if decision.tool_name == "final_answer":
                logger.info("✅ Réponse finale générée.")
                # ← METTRE À JOUR la mémoire persistante
                if use_memory:
                    self.conversation_history = history.copy()
                return decision.arguments.text

            tool_result = await self._execute_tool(decision.tool_name, decision.arguments)
            formatted_result = self._format_tool_result_for_llm(tool_result)
            history.append({"role": "user", "content": formatted_result})

        # ← METTRE À JOUR la mémoire même en cas de timeout
        if use_memory:
            self.conversation_history = history.copy()

        return "Désolé, je n'ai pas pu aboutir à une réponse finale dans le nombre d'étapes imparti."

    def clear_memory(self):
        """Efface la mémoire de conversation."""
        self.conversation_history = []
        print("🧠 Mémoire de l'agent effacée.")

    def get_memory_summary(self) -> str:
        """Retourne un résumé de la mémoire."""
        if not self.conversation_history:
            return "Aucune mémoire conservée."

        user_messages = [msg["content"] for msg in self.conversation_history if msg["role"] == "user"]
        return f"Mémoire : {len(self.conversation_history)} messages, dernières requêtes : {user_messages[-3:]}"
