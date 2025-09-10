import logging
from enum import Enum
from typing import Dict, Any, Optional, Union, Literal

from pydantic import Field, BaseModel

from ..fortran_analysis.providers.consult import FortranEntityExplorer
from ..provider.llm_providers import LLMProvider


# Configuration du logger
logger = logging.getLogger(__name__)


class ToolName(str, Enum):
    """Les noms des outils uniques que le routeur peut appeler."""
    LIST_ENTITIES = "list_entities"
    GET_ENTITY_REPORT = "get_entity_report"
    GET_RELATIONS = "get_relations"
    UNKNOWN = "unknown_query"

class ListEntitiesArgs(BaseModel):
    """Arguments to filter and list code entities."""
    entity_type: Optional[str] = Field(None, description="The type of entity (e.g., 'subroutine', 'function').")
    filename: Optional[str] = Field(None, description="The name of the file to search within. Can be partial.")
    entity_name: Optional[str] = Field(None, description="An approximate name of the entity to search for.")
    detected_concept: Optional[str] = Field(None,
                                            description="**Utilisez ce champ pour toute question sémantique ou conceptuelle.** Décrivez la fonctionnalité ou la logique recherchée (ex: 'gestion des entrées/sorties', 'communication MPI').")


class GetEntityReportArgs(BaseModel):
    """Arguments to get a detailed report for a single entity."""
    entity_name: str = Field(..., description="The exact name of the entity to get a full report for.")


class GetRelationsArgs(BaseModel):
    """Arguments to get relationships for a single entity."""
    entity_name: str = Field(..., description="The name of the central entity for the relation query.")
    relation_type: str = Field(..., description="The type of relation to find. Must be 'callers' (who calls this entity) or 'callees' (what this entity calls).")


class UnknownQueryArgs(BaseModel):
    """Used when the query's intent is unclear."""
    reason: str = Field(..., description="A brief explanation of why the query is considered unknown.")


class QueryPlan(BaseModel):
    """
    Le plan d'action décidé par le LLM.
    """
    thought: str = Field(
        ...,
        description="A short sentence explaining your reasoning for choosing the tool and arguments."
    )
    tool_name: ToolName = Field(
        ...,
        description="The most appropriate tool to respond to the user's request."
    )
    # Le champ 'arguments' est maintenant une Union de nos modèles spécifiques.
    # C'est beaucoup plus strict et guide le LLM.
    arguments: Union[
        ListEntitiesArgs,
        GetEntityReportArgs,
        GetRelationsArgs,
        UnknownQueryArgs
    ]


class IntelligentQueryRouter:
    """
    Analyse la requête de l'utilisateur avec un LLM pour choisir et exécuter le bon outil.
    """

    def __init__(self, llm_provider: LLMProvider, entity_explorer: FortranEntityExplorer):
        """
        Initialise le routeur.

        Args:
            llm_provider: Une instance d'un provider LLM (ex: OpenAIProvider).
            entity_explorer: Une instance du FortranEntityExplorer.
        """
        self.llm_provider = llm_provider
        self.explorer = entity_explorer
        self.system_prompt = self._build_system_prompt()
        self.llm_provider.set_system_prompt(self.system_prompt)

    def _build_system_prompt(self) -> str:
        """Construit un prompt système robuste et sans ambiguïté."""
        return """Tu es un système expert analysant des requêtes sur une base de code Fortran. Ta seule tâche est de choisir le bon outil et de remplir correctement ses arguments au format JSON.

    Voici les outils disponibles :

    1. `GET_ENTITY_REPORT`: Pour un rapport détaillé sur **UNE SEULE** entité **spécifique**.
       - `entity_name`: Le nom exact de l'entité.
       - Exemple: "donne-moi le rapport de CALCULATE_FORCE" -> tool: get_entity_report, args: {'entity_name': 'CALCULATE_FORCE'}

    2. `GET_RELATIONS`: Pour trouver les relations (appelants/appelés) d'**UNE SEULE** entité **spécifique**.
       - `entity_name`: Le nom exact de l'entité.
       - `relation_type`: 'callers' ou 'callees'.
       - Exemple: "Qui appelle Free_Kernel ?" -> tool: get_relations, args: {'entity_name': 'Free_Kernel', 'relation_type': 'callers'}

    3. `LIST_ENTITIES`: **TON OUTIL DE RECHERCHE PRINCIPAL ET UNIQUE.** Utilise-le pour **TOUTES** les requêtes qui demandent de trouver ou lister une ou plusieurs entités.
       
       --- DÉTECTION AUTOMATIQUE DU TYPE D'ENTITÉ ---
       **ATTENTION :** Détecte automatiquement le type d'entité depuis la requête utilisateur :
       
       Correspondances OBLIGATOIRES français -> anglais :
       - "subroutine", "sous-routine", "procédure" -> `entity_type='subroutine'`
       - "fonction", "function" -> `entity_type='function'`  
       - "module" -> `entity_type='module'`
       - "programme", "program" -> `entity_type='program'`
       - "type", "structure" -> `entity_type='type_definition'`
       - "interface" -> `entity_type='interface'`
       - "paramètre", "parameter" -> `entity_type='parameter'`
       - "variable", "var" -> `entity_type='variable'`
       
       **RÈGLE IMPORTANTE :** Si un type est mentionné dans la requête, tu DOIS le spécifier. Ne JAMAIS laisser `entity_type=None` si un type est détectable !

       --- ARGUMENTS DISPONIBLES ---
       - **Arguments par attributs** : `entity_type`, `filename`, `entity_name`, `parent_entity`, `dependencies`, etc.
       - **Argument sémantique** : `detected_concept`. Utilise-le pour toutes les questions sur la **fonctionnalité**, la **logique** ou le **rôle** du code.

       --- EXEMPLES CRUCIAUX POUR LIST_ENTITIES ---

       a) Détection de type + recherche par attributs :
       - "Montre-moi les fonctions dans 'utils.f90'"
         -> `{'tool_name': 'list_entities', 'arguments': {'entity_type': 'function', 'filename': 'utils.f90'}}`
       
       - "Quelles subroutines sont dans le fichier main.f90 ?"
         -> `{'tool_name': 'list_entities', 'arguments': {'entity_type': 'subroutine', 'filename': 'main.f90'}}`

       b) Recherche sémantique simple :
       - "Où est gérée l'allocation de mémoire ?"
         -> `{'tool_name': 'list_entities', 'arguments': {'detected_concept': 'allocation de mémoire'}}`

       c) **Recherche COMBINÉE (le cas le plus important) :**
       - "Quelle fonction dans le fichier 'scaling_function.f90' fait référence au concept de 'fft' ?"
         -> `{'tool_name': 'list_entities', 'arguments': {'entity_type': 'function', 'filename': 'scaling_function.f90', 'detected_concept': 'fft'}}`
       
       - "Quel subroutine du fichier scaling_function fait référence au concept de Exchange-Correlation ?"
         -> `{'tool_name': 'list_entities', 'arguments': {'entity_type': 'subroutine', 'filename': 'scaling_function', 'detected_concept': 'Exchange-Correlation'}}`

       d) Autres exemples combinés :
       - "Trouve les subroutines publiques qui utilisent le module 'mpi_lib' et qui concernent l'initialisation."
         -> `{'tool_name': 'list_entities', 'arguments': {'entity_type': 'subroutine', 'access_level': 'public', 'dependencies': 'mpi_lib', 'detected_concept': 'initialisation'}}`
       
       - "Quelles procédures dans kernel.f90 s'occupent du calcul matriciel ?"
         -> `{'tool_name': 'list_entities', 'arguments': {'entity_type': 'subroutine', 'filename': 'kernel.f90', 'detected_concept': 'calcul matriciel'}}`

       --- VALEURS POSSIBLES pour entity_type ---
       `module`, `subroutine`, `function`, `program`, `type_definition`, `interface`, `parameter`, `variable`
       
       --- FIN DES EXEMPLES ---

    4. `UNKNOWN_QUERY`: À utiliser **SEULEMENT** si la requête est totalement inintelligible.

    **PROCESSUS DE RAISONNEMENT OBLIGATOIRE :**
    1. Lis la requête et identifie les mots-clés pour le type d'entité
    2. Identifie les noms de fichiers mentionnés  
    3. Identifie les concepts sémantiques recherchés
    4. Combine tous ces éléments dans l'appel list_entities

    Tu DOIS répondre UNIQUEMENT avec un objet JSON valide correspondant au format demandé.
    """

    async def route_query(self, user_query: str) -> Dict[str, Any]:
        """
        Analyse la requête, choisit un outil, l'exécute et retourne le résultat.
        """
        logger.info(f"🧠 Routage de la requête: '{user_query}'")

        try:
            # Le LLM génère le plan d'action au format Pydantic
            # Note: J'utilise OpenAIProvider comme exemple, adaptez si besoin
            plan_dict = await self.llm_provider.generate_response(
                messages=[{"role": "user", "content": user_query}],
                pydantic_model=QueryPlan
            )
            plan = QueryPlan.model_validate(plan_dict)

            print(f"🤔 Plan du LLM: {plan.thought}")
            print(f"🛠️ Outil choisi: {plan.tool_name}, Arguments: {plan.arguments}")

            # On transforme le modèle Pydantic en dictionnaire pour le 'splatting' (**)
            args_dict = plan.arguments.model_dump()  # ou .dict() en v1

            # Exécuter l'outil choisi
            if plan.tool_name == ToolName.LIST_ENTITIES:
                return {
                    "type": "entity_list",
                    "data": await self.explorer.find_entities_by_criteria(**args_dict)
                }

            elif plan.tool_name == ToolName.GET_ENTITY_REPORT:
                return {
                    "type": "entity_report",
                    "data": await self.explorer.get_full_report(entity_name=plan.arguments.entity_name)
                }

            elif plan.tool_name == ToolName.GET_RELATIONS:
                if not isinstance(plan.arguments, GetRelationsArgs):
                    logger.error(f"Incohérence: outil={plan.tool_name} mais type d'args={type(plan.arguments)}\n{args_dict}")
                    return {"type": "error", "message": "Le LLM a produit des arguments incohérents."}

                # On récupère l'entité pour avoir plus de contexte
                entity = await self.explorer.em.find_entity(plan.arguments.entity_name)

                if not entity:
                    return {"type": "error", "message": f"Entité '{plan.arguments.entity_name}' non trouvée."}

                relation_data = None
                if plan.arguments.relation_type == 'callers':
                    # On utilise la méthode existante pour trouver les appelants !
                    relation_data = await self.explorer.get_callers(entity.entity_name)

                elif plan.arguments.relation_type == 'callees':
                    # On utilise la méthode existante pour les appelés !
                    relation_data = self.explorer.get_callees_and_dependencies(entity)

                return {
                    "type": "entity_relations",
                    "entity": entity.to_dict(),
                    "relation_type": plan.arguments.relation_type,
                    "data": relation_data
                }

            else:  # UNKNOWN_QUERY
                return {
                    "type": "unknown",
                    "message": "Je ne suis pas sûr de comprendre votre demande. Pouvez-vous reformuler ?"
                }

        except Exception as e:
            logger.error(f"Erreur durant le routage de la requête: {e}", exc_info=True)
            return {"type": "error", "message": f"Une erreur est survenue: {e}"}