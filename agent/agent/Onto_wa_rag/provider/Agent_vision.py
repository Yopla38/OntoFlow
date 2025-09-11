"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

"""
🔍 VISION AGENT - Agent autonome pour l'analyse d'images multimodales
Inspiré de fortran_analysis/providers/Fortran_agent.py
"""

import json
import logging
from typing import Dict, Any, List, Optional, Literal, Union
from datetime import datetime

from pydantic import BaseModel, Field, ValidationError

from CONSTANT import API_KEY_PATH, VISION_AGENT_MODEL, VISION_NB_STEP_AGENT
from provider.Multimodal.multimodal_agent import ImageAnalysisToolsManager
from provider.get_key import get_anthropic_key
from provider.llm_providers import AnthropicProvider
from provider.Multimodal.multimodal_analyse_v3 import AdaptiveMultiAgentAnalyzer

logger = logging.getLogger(__name__)


# ==================== MODÈLES PYDANTIC POUR LES ARGUMENTS ====================

class AgentAnalyzeImageArgs(BaseModel):
    """Arguments pour analyser une image."""
    image_data: str = Field(..., description="Image encodée en base64 ou chemin vers le fichier")
    force_reanalysis: bool = Field(False, description="Force une nouvelle analyse en ignorant le cache")


class AgentQueryGraphValueArgs(BaseModel):
    """Arguments pour interroger une valeur sur un graphique."""
    image_analysis_id: str = Field(..., description="ID de l'analyse d'image précédente")
    x_value: float = Field(..., description="Valeur X pour laquelle chercher Y")
    curve_index: int = Field(0, description="Index de la courbe à interroger (0 pour la première)")


class AgentRecreateChartArgs(BaseModel):
    """Arguments pour recréer un graphique."""
    image_analysis_id: str = Field(..., description="ID de l'analyse d'image précédente")
    save_path: Optional[str] = Field(None, description="Chemin pour sauvegarder le graphique recréé")
    show_plot: bool = Field(False, description="Afficher le graphique à l'écran")
    modify_code: Optional[Dict] = Field(None, description="Modifications à appliquer au code")


class AgentGetImageDescriptionArgs(BaseModel):
    """Arguments pour obtenir une description d'image."""
    image_analysis_id: str = Field(..., description="ID de l'analyse d'image précédente")
    detail_level: Literal["basic", "detailed", "technical"] = Field(
        "detailed", description="Niveau de détail de la description"
    )


class AgentRequeryImageArgs(BaseModel):
    """Arguments pour poser une question spécifique sur une image."""
    image_analysis_id: str = Field(..., description="ID de l'analyse d'image précédente")
    question: str = Field(..., description="Question spécifique à poser sur l'image")


class AgentManageCacheArgs(BaseModel):
    """Arguments pour gérer le cache."""
    action: Literal["check", "clear_all", "clear_image"] = Field(..., description="Action à effectuer sur le cache")
    image_analysis_id: Optional[str] = Field(None, description="ID de l'analyse pour actions spécifiques")


class AgentGetAnalysisHistoryArgs(BaseModel):
    """Arguments pour récupérer l'historique des analyses."""
    limit: int = Field(10, description="Nombre maximum d'analyses à retourner")
    content_type_filter: Optional[str] = Field(None, description="Filtrer par type de contenu")


class AgentFinalAnswerArgs(BaseModel):
    """Arguments pour la réponse finale."""
    text: str = Field(..., description="La réponse finale complète à la requête de l'utilisateur")


class AgentGetGraphDataArgs(BaseModel):
    """Arguments pour obtenir les données brutes (points x,y) d'un graphique."""
    image_analysis_id: str = Field(..., description="ID de l'analyse d'image précédente")

# ==================== DÉCISION STRUCTURÉE DE L'AGENT ====================


class VisionAgentDecision(BaseModel):
    """Définit la pensée et l'action structurée de l'agent vision."""
    thought: str = Field(..., description="Réflexion détaillée de l'agent sur la situation actuelle")
    tool_name: Literal[
        "analyze_image",
        "get_graph_data",
        "query_graph_value",
        "recreate_chart",
        "get_image_description",
        "requery_image_with_context",
        "manage_cache",
        "get_analysis_history",
        "final_answer"
    ] = Field(..., description="Nom de l'outil à utiliser")

    arguments: Union[
        AgentAnalyzeImageArgs,
        AgentGetGraphDataArgs,
        AgentQueryGraphValueArgs,
        AgentRecreateChartArgs,
        AgentGetImageDescriptionArgs,
        AgentRequeryImageArgs,
        AgentManageCacheArgs,
        AgentGetAnalysisHistoryArgs,
        AgentFinalAnswerArgs
    ] = Field(..., description="Arguments pour l'outil sélectionné")


# ==================== AGENT VISION PRINCIPAL ====================

class VisionAgent:
    """Agent autonome spécialisé dans l'analyse d'images multimodales."""

    def __init__(self,
                 llm_provider: AnthropicProvider,
                 tools_manager: ImageAnalysisToolsManager,
                 max_steps: int = 10):
        """
        Initialise l'agent vision.

        Args:
            llm_provider: Fournisseur LLM pour les décisions
            tools_manager: Gestionnaire des outils d'analyse d'images
            max_steps: Nombre maximum d'étapes pour éviter les boucles infinies
            cache_dir: Répertoire pour stocker le cache persistant
        """
        self.llm = llm_provider
        self.tools_manager = tools_manager
        self.max_steps = max_steps
        self.cache_dir = tools_manager.cache_dir
        self.system_prompt = self._build_system_prompt()

        # Créer le répertoire de cache s'il n'existe pas
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Historique persistant des conversations
        self.conversation_history_file = self.cache_dir / "conversation_history.json"
        self.conversation_history: List[Dict[str, str]] = self._load_conversation_history()

        # Mappage des outils vers les méthodes
        self.tool_mapping = {
            "analyze_image": self.tools_manager.analyze_image,
            "get_graph_data": self.tools_manager.get_graph_data,
            "query_graph_value": self.tools_manager.query_graph_value,
            "recreate_chart": self.tools_manager.recreate_chart,
            "get_image_description": self.tools_manager.get_image_description,
            "requery_image_with_context": self.tools_manager.requery_image_with_context,
            "manage_cache": self.tools_manager.manage_cache,
            "get_analysis_history": self.tools_manager.get_analysis_history
        }

    def _load_conversation_history(self) -> List[Dict[str, str]]:
        """Charge l'historique des conversations."""
        try:
            if self.conversation_history_file.exists():
                with open(self.conversation_history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"❌ Erreur chargement historique conversation: {e}")
            return []

    def _save_conversation_history(self):
        """Sauvegarde l'historique des conversations."""
        try:
            with open(self.conversation_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde historique conversation: {e}")

    """
    - `query_graph_value`: **INTERROGATION DE DONNÉES** - Trouve une valeur Y pour un X donné sur un graphique
      * `image_analysis_id`: ID de l'analyse précédente (requis)
      * `x_value`: Valeur X à rechercher (requis)
      * `curve_index`: Index de la courbe (défaut: 0)
    """

    def _build_system_prompt(self) -> str:
        """Construit le prompt système pour l'agent vision."""
        return """Tu es un AGENT VISION EXPERT autonome, spécialisé dans l'analyse approfondie d'images scientifiques, graphiques, diagrammes et autres contenus visuels.

**MISSION :** Répondre aux requêtes complexes des utilisateurs concernant des images en utilisant des outils d'analyse avancés de manière stratégique et méthodique.

**PRINCIPE FONDAMENTAL :** Tu dois TOUJOURS analyser en profondeur avant de conclure. Pour les requêtes exploratoires, creuse suffisamment pour donner des réponses complètes et précises.

<outils_disponibles>
- `analyze_image`: **OUTIL PRINCIPAL D'ANALYSE** - Analyse complète d'une image (détection de type, extraction de données)
  * `image_data`: Chemin vers l'image ou données base64
  * `force_reanalysis`: Force une nouvelle analyse (défaut: false)

- `get_graph_data`: **EXTRACTION RAPIDE DE DONNÉES** - Pour obtenir les DONNÉES BRUTES (points x, y) d'un graphique déjà analysé. À utiliser quand l'utilisateur demande les valeurs, le max, le min, ou les tendances.
  * `image_analysis_id`: ID de l'analyse précédente (requis)

- `recreate_chart`: **RECRÉATION DE GRAPHIQUES** - Génère le code pour reproduire fidèlement un graphique
  * `image_analysis_id`: ID de l'analyse précédente (requis)
  * `save_path`: Chemin de sauvegarde (optionnel)
  * `show_plot`: Afficher le graphique (défaut: false)
  * `modify_code`: Modifications du code (optionnel)

- `get_image_description`: **DESCRIPTION DÉTAILLÉE** - Obtient une description complète de l'image
  * `image_analysis_id`: ID de l'analyse précédente (requis)
  * `detail_level`: "basic", "detailed", ou "technical" (défaut: "detailed")

- `requery_image_with_context`: **QUESTION CONTEXTUELLE** - Pose une question spécifique sur une image analysée
  * `image_analysis_id`: ID de l'analyse précédente (requis)
  * `question`: Question à poser (requis)

- `manage_cache`: **GESTION DU CACHE** - Vérifie ou nettoie le cache des analyses
  * `action`: "check", "clear_all", ou "clear_image" (requis)
  * `image_analysis_id`: ID spécifique pour clear_image (optionnel)

- `get_analysis_history`: **HISTORIQUE** - Récupère l'historique des analyses effectuées
  * `limit`: Nombre max d'analyses (défaut: 10)
  * `content_type_filter`: Filtre par type (optionnel)

- `final_answer`: **CONCLUSION** - Fournit la réponse finale à l'utilisateur
  * `text`: Réponse complète et détaillée (requis)
</outils_disponibles>

<strategie_generale>
**POUR UNE NOUVELLE IMAGE :**
1. **TOUJOURS commencer par `analyze_image`** - C'est obligatoire pour toute nouvelle image
2. **Vérifier le succès** et le type de contenu détecté
3. **Approfondir selon le besoin** avec les outils spécialisés
4. **Synthétiser** avec `final_answer`

**POUR DES QUESTIONS SUR UNE IMAGE DÉJÀ ANALYSÉE :**
1. **Vérifier si l'ID existe** (utiliser `get_analysis_history` si nécessaire)
2. **Utiliser l'outil approprié** selon la question
3. **Compléter avec d'autres outils** si nécessaire
4. **Répondre de manière complète**

**POUR DES REQUÊTES EXPLORATOIRES :**
- Minimum 3-4 tours d'analyse avant de conclure
- Utiliser plusieurs outils pour avoir une vue complète
- Analyser les détails techniques ET le contexte métier
</strategie_generale>

<exemples_strategiques>
**EXEMPLE 1 - Nouvelle image :**
User: "Analyse cette image de graphique en barres."

Tour 1: {
    "thought": "L'utilisateur me donne une nouvelle image à analyser. Je dois commencer par l'analyser complètement.",
    "tool_name": "analyze_image",
    "arguments": {"image_data": "/path/to/image.jpg"}
}

Tour 2: {
    "thought": "L'analyse montre un bar chart avec 85% de confiance. Je vais obtenir une description détaillée pour mieux comprendre le contenu.",
    "tool_name": "get_image_description", 
    "arguments": {"image_analysis_id": "analysis_xxx", "detail_level": "detailed"}
}

**EXEMPLE 2 - Question sur valeur :**
User: "Quelle est la valeur pour X=2.5 ?"

Tour 1: {
    "thought": "L'utilisateur demande une valeur spécifique. Je dois d'abord vérifier quel graphique a été analysé récemment.",
    "tool_name": "get_analysis_history",
    "arguments": {"limit": 5}
}

Tour 2: {
    "thought": "Je vois qu'il y a une analyse récente d'un line chart. Je vais interroger la valeur demandée.",
    "tool_name": "query_graph_value",
    "arguments": {"image_analysis_id": "analysis_xxx", "x_value": 2.5}
}

**EXEMPLE 3 - Question sur les données :**
User: "Donne-moi les données de la courbe rouge et dis-moi son maximum."

Tour 1: {
    "thought": "L'utilisateur veut les données brutes d'un graphique. L'outil le plus direct pour cela est `get_graph_data`. Je vais d'abord récupérer l'ID de la dernière analyse.",
    "tool_name": "get_analysis_history",
    "arguments": {"limit": 1}
}

Tour 2: {
    "thought": "J'ai l'ID de l'analyse. Je vais maintenant utiliser `get_graph_data` pour extraire les points (x, y).",
    "tool_name": "get_graph_data",
    "arguments": {"image_analysis_id": "analysis_xxx"}
}

Tour 3: {
    "thought": "J'ai reçu les données brutes. Je peux maintenant identifier la série rouge, trouver sa valeur maximale, et formater la réponse finale pour l'utilisateur.",
    "tool_name": "final_answer",
    "arguments": {"text": "Les données de la courbe rouge sont [...]. Sa valeur maximale est Y."}
    
</exemples_strategiques>

**RÈGLES IMPORTANTES :**
1. **Toujours analyser d'abord** une nouvelle image avec `analyze_image`
2. **Vérifier les IDs** d'analyses avant d'utiliser les autres outils
3. **Être précis** dans tes réflexions (`thought`)
4. **Approfondir** avant de conclure avec `final_answer`
5. **Gérer les erreurs** en adaptant ta stratégie

Commence maintenant ton analyse méthodique !"""

    async def _execute_tool(self, tool_name: str, args: BaseModel) -> Any:
        """Exécute un outil avec des arguments validés."""
        logger.info(f"🛠️ Exécution de l'outil '{tool_name}'")

        # Conversion en dictionnaire en excluant les valeurs None
        args_dict = args.model_dump(exclude_none=True)

        try:
            if tool_name == "final_answer":
                # Le final_answer n'est pas un vrai outil, on retourne juste le texte
                return {"success": True, "answer": args_dict["text"]}

            elif tool_name in self.tool_mapping:
                # Appel de l'outil correspondant
                tool_method = self.tool_mapping[tool_name]
                result = await tool_method(**args_dict)
                return result

            else:
                return {"success": False, "error": f"Outil inconnu: {tool_name}"}

        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de l'outil '{tool_name}': {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _format_tool_result_for_llm(self, result: Any) -> str:
        """Formate le résultat d'un outil pour le LLM."""
        try:
            # Conversion en JSON avec gestion de la taille
            if isinstance(result, dict) and result.get("success") and "answer" in result:
                # C'est une réponse finale
                return f"Final Answer Ready: {result['answer']}"

            json_str = json.dumps(result, indent=2, default=str)

            # Limiter la taille pour éviter les tokens excessifs
            if len(json_str) > 12000:
                return f"Tool Result (tronqué): {json_str[:12000]}...\n[Résultat tronqué - utiliser des outils plus spécifiques pour plus de détails]"

            return f"Tool Result: {json_str}"

        except Exception as e:
            logger.error(f"Erreur de formatage du résultat: {e}")
            return f"Tool Result Error: Impossible de formater le résultat - {str(result)[:500]}"

    async def run(self, user_query: str, image_path: Optional[str] = None, use_memory: bool = True) -> str:
        """
        Exécute l'agent pour répondre à une requête utilisateur.

        Args:
            user_query: Question ou demande de l'utilisateur
            image_path: Chemin vers une image (optionnel)
            use_memory: Utiliser la mémoire des conversations précédentes

        Returns:
            Réponse finale de l'agent
        """
        logger.info(f"🚀 Démarrage de l'agent vision - Query: {user_query[:100]}...")

        # Construction du message utilisateur
        user_message = user_query
        if image_path:
            user_message += f"\n[Image fournie: {image_path}]"

        # Gestion de la mémoire
        if use_memory and self.conversation_history:
            self.conversation_history.append({"role": "user", "content": user_message})
            history = self.conversation_history.copy()
        else:
            history = [{"role": "user", "content": user_message}]
            self.conversation_history = history.copy()

        # Boucle principale de raisonnement
        for step in range(self.max_steps):
            logger.info(f"--- Tour de l'Agent Vision {step + 1}/{self.max_steps} ---")

            # Préparation des messages pour le LLM
            messages_for_llm = [
                                   {"role": "system", "content": self.system_prompt}
                               ] + history

            try:
                # Appel au LLM pour obtenir une décision structurée
                decision_dict = await self.llm.generate_response_from_messages(
                    messages=messages_for_llm,
                    pydantic_model=VisionAgentDecision
                )

                if not decision_dict:
                    logger.error("Le LLM n'a pas retourné de décision")
                    break

                # Validation de la décision
                decision = VisionAgentDecision.model_validate(decision_dict)
                logger.info(f"🤔 Décision de l'agent: {decision.tool_name} - {decision.thought[:100]}...")

            except ValidationError as e:
                logger.error(f"Erreur de validation de la décision: {e}")
                return f"Erreur de validation de la décision du LLM: {e}"
            except Exception as e:
                logger.error(f"Erreur lors de l'appel au LLM: {e}")
                return f"Erreur lors de l'appel au LLM: {e}"

            # Ajout de la décision à l'historique
            history.append({
                "role": "assistant",
                "content": decision.model_dump_json(indent=2)
            })

            # Vérification si c'est la réponse finale
            if decision.tool_name == "final_answer":
                logger.info("✅ Réponse finale générée par l'agent")
                if use_memory:
                    self.conversation_history = history.copy()
                    self._save_conversation_history()  # 💾 SAUVEGARDE AUTO
                return decision.arguments.text

            # Exécution de l'outil
            tool_result = await self._execute_tool(decision.tool_name, decision.arguments)

            # Formatage et ajout du résultat à l'historique
            formatted_result = self._format_tool_result_for_llm(tool_result)
            history.append({"role": "user", "content": formatted_result})

            logger.info(f"📊 Résultat outil: {str(tool_result)[:200]}...")

        # Mise à jour de la mémoire même en cas de timeout
        if use_memory:
            self.conversation_history = history.copy()
            self._save_conversation_history()  # 💾 SAUVEGARDE AUTO

        return f"Désolé, je n'ai pas pu aboutir à une réponse finale dans les {self.max_steps} étapes imparties. La dernière action était: {decision.tool_name if 'decision' in locals() else 'inconnue'}"

    def clear_memory(self, clear_cache: bool = False):
        """Efface la mémoire et optionnellement le cache."""
        self.conversation_history = []
        self._save_conversation_history()

        if clear_cache:
            asyncio.run(self.tools_manager.manage_cache("clear_all"))

        logger.info("🧠 Mémoire de l'agent effacée" + (" et cache vidé" if clear_cache else ""))

    def get_memory_summary(self) -> str:
        """Retourne un résumé de la mémoire et du cache."""
        cache_stats = self.tools_manager.get_cache_stats()

        if not self.conversation_history:
            memory_info = "Aucune mémoire conservée"
        else:
            user_messages = [msg["content"] for msg in self.conversation_history if msg["role"] == "user"]
            memory_info = f"Mémoire : {len(self.conversation_history)} messages"

        return f"{memory_info} | Cache : {cache_stats['total_analyses']} analyses ({cache_stats['total_size_mb']} MB)"

    async def analyze_image_workflow(self, image_path: str, follow_up_questions: List[str] = None) -> Dict[str, Any]:
        """
        Workflow complet d'analyse d'image avec questions de suivi optionnelles.

        Args:
            image_path: Chemin vers l'image
            follow_up_questions: Questions de suivi optionnelles

        Returns:
            Dictionnaire avec tous les résultats
        """
        results = {
            "image_path": image_path,
            "timestamp": datetime.now(),
            "steps": []
        }

        # Étape 1: Analyse initiale
        initial_response = await self.run(f"Analyse cette image en détail: {image_path}")
        results["initial_analysis"] = initial_response
        results["steps"].append("initial_analysis")

        # Étapes de suivi si demandées
        if follow_up_questions:
            results["follow_up_responses"] = []
            for question in follow_up_questions:
                response = await self.run(question, use_memory=True)
                results["follow_up_responses"].append({
                    "question": question,
                    "response": response
                })
                results["steps"].append("follow_up")

        return results


# ==================== FONCTION DE TEST ====================

async def test_vision_agent():
    """Test complet de l'agent vision."""

    print("🚀 INITIALISATION DE L'AGENT VISION")
    print("=" * 60)

    # Configuration
    API_KEY = "my_key"

    # Initialisation des composants
    llm_provider = AnthropicProvider(
        model=VISION_AGENT_MODEL,
        api_key=API_KEY,
        system_prompt="Tu es un expert en analyse d'images et raisonnement structuré."
    )

    analyzer = AdaptiveMultiAgentAnalyzer(llm_provider)
    tools_manager = ImageAnalysisToolsManager(analyzer)
    vision_agent = VisionAgent(llm_provider, tools_manager, max_steps=8)

    # Scénarios de test
    test_scenarios = [
        {
            "name": "Analyse complète d'image",
            "query": "Peux-tu analyser cette image en détail et me dire ce qu'elle contient ?",
            "image": "/home/yopla/Documents/llm_models/python/models/multimodal/test_figure/courbe.jpg",
            "use_memory": False
        },
        {
            "name": "Question sur valeur spécifique",
            "query": "Quelle est la valeur Y quand X = 0.4 sur ce graphique ?",
            "image": None,
            "use_memory": True
        },
        {
            "name": "Recréation de graphique",
            "query": "Peux-tu recréer ce graphique et le sauvegarder ?",
            "image": None,
            "use_memory": True
        },
        {
            "name": "Question contextuelle",
            "query": "Quelle ondelette a la meilleure performance sur la métrique NDCG@5 ?",
            "image": None,
            "use_memory": True
        },
        {
            "name": "Gestion du cache",
            "query": "Montre-moi l'état du cache et l'historique des analyses récentes",
            "image": None,
            "use_memory": True
        }
    ]

    # Exécution des tests
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 TEST {i}: {scenario['name']}")
        print("-" * 50)

        try:
            start_time = datetime.now()

            response = await vision_agent.run(
                user_query=scenario['query'],
                image_path=scenario['image'],
                use_memory=scenario['use_memory']
            )

            duration = (datetime.now() - start_time).total_seconds()

            print(f"✅ Réponse (en {duration:.1f}s):")
            print(response)

            # Afficher l'état de la mémoire
            memory_summary = vision_agent.get_memory_summary()
            print(f"\n🧠 {memory_summary}")

        except Exception as e:
            print(f"❌ Erreur: {e}")
            import traceback
            traceback.print_exc()

        print("-" * 50)

    # Test du workflow complet
    print(f"\n🔄 TEST WORKFLOW COMPLET")
    print("-" * 50)

    try:
        workflow_result = await vision_agent.analyze_image_workflow(
            image_path="/home/yopla/Documents/llm_models/python/models/multimodal/test_figure/courbe.jpg",
            follow_up_questions=[
                "Quelle est la valeur maximale visible sur ce graphique ?",
                "Peux-tu recréer ce graphique avec des couleurs différentes ?"
            ]
        )

        print("✅ Workflow terminé:")
        print(f"- Étapes: {workflow_result['steps']}")
        print(f"- Questions de suivi: {len(workflow_result.get('follow_up_responses', []))}")

    except Exception as e:
        print(f"❌ Erreur workflow: {e}")

    print("\n🎉 Tests terminés!")


# ==================== USAGE AVANCÉ ====================

class VisionAgentManager:
    """Gestionnaire avancé pour plusieurs agents vision."""

    def __init__(self, llm_provider: AnthropicProvider, cache_dir: str = "./vision_cache"):
        self.llm_provider = llm_provider
        self.cache_dir = cache_dir
        self.agents = {}
        self.shared_analyzer = AdaptiveMultiAgentAnalyzer(llm_provider)
        self.shared_tools_manager = ImageAnalysisToolsManager(self.shared_analyzer, cache_dir=cache_dir)

    def create_agent(self, agent_id: str, max_steps: int = 10) -> VisionAgent:
        """Crée un nouvel agent vision."""
        agent = VisionAgent(self.llm_provider, self.shared_tools_manager, max_steps)
        self.agents[agent_id] = agent
        return agent

    def get_agent(self, agent_id: str) -> Optional[VisionAgent]:
        """Récupère un agent existant."""
        return self.agents.get(agent_id)

    async def broadcast_query(self, query: str, image_path: str = None) -> Dict[str, str]:
        """Envoie une requête à tous les agents."""
        results = {}
        for agent_id, agent in self.agents.items():
            try:
                response = await agent.run(query, image_path, use_memory=False)
                results[agent_id] = response
            except Exception as e:
                results[agent_id] = f"Erreur: {e}"
        return results


async def my_vision_agent():
    API_KEY = get_anthropic_key(API_KEY_PATH)

    # Initialisation des composants
    llm_provider = AnthropicProvider(
        model=VISION_AGENT_MODEL,
        api_key=API_KEY,
        system_prompt="Tu es un expert en analyse d'images et raisonnement structuré."
    )
    agent_vision = VisionAgentManager(llm_provider, "./vision_cache")
    agent_vision.create_agent("cool", VISION_NB_STEP_AGENT)
    response = await agent_vision.broadcast_query(query="Trace le graphique. Donne la valeur de db5 pour MAP.",
                                 image_path="/home/yopla/Documents/llm_models/python/models/multimodal/test_figure/openai_level5.png")
    print(response)


if __name__ == "__main__":
    import asyncio
    asyncio.run(my_vision_agent())
