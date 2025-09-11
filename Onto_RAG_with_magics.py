import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List
import sys
import nest_asyncio


from agent.Onto_wa_rag.Integration_fortran_RAG import OntoRAG
from agent.Onto_wa_rag.CONSTANT import (
    API_KEY_PATH, CHUNK_SIZE, CHUNK_OVERLAP, ONTOLOGY_PATH_TTL,
    MAX_CONCURRENT, MAX_RESULTS, STORAGE_DIR
)
from agent.Onto_wa_rag.fortran_analysis.providers.consult import FortranEntityExplorer

# --- Imports IPython Magic ---
from IPython.core.magic import Magics, magics_class, line_cell_magic
from IPython.display import display, Markdown

# Appliquer nest_asyncio pour permettre l'utilisation d'asyncio dans un environnement déjà bouclé (comme Jupyter)
nest_asyncio.apply()


# ==============================================================================
# 1. FONCTIONS D'AFFICHAGE (HELPER FUNCTIONS)
# ==============================================================================

async def show_available_commands():
    """Affiche toutes les commandes disponibles pour la magic."""
    display(Markdown("""
### ✨ ONTORAG - Commandes Magiques Disponibles ✨

---

#### 🧠 **Agent Unifié (Fortran + Jupyter)**
- **`/agent <question>`**: Démarre une conversation avec l'agent unifié (analyse Fortran ET Jupyter)
- **`/agent_reply <réponse>`**: Répond à une question de clarification de l'agent
- **`/agent_memory`**: Affiche le résumé de la mémoire actuelle de l'agent
- **`/agent_clear`**: Efface la mémoire de l'agent et termine la conversation
- **`/agent_sources`**: Affiche toutes les sources consultées dans la session courante

---

#### 🔍 **Recherche et Consultation**
- **`<question>`**: (Sans `/`) Lance une requête directe avec l'agent unifié
- **`/search <question>`**: Effectue une recherche sémantique classique
- **`/hierarchical <q>`**: Lance une recherche hiérarchique sur plusieurs niveaux

---

#### 📁 **Gestion des Documents**
- **`/add_docs <var_name>`**: Ajoute des documents depuis une variable Python
- **`/list`**: Liste tous les documents indexés
- **`/stats`**: Affiche les statistiques du RAG

---

#### ❓ **Aide**
- **`/help`**: Affiche ce message d'aide

---
"""))

async def display_query_result(result: Dict[str, Any]):
    """Affiche le résultat d'une query() standard."""
    display(Markdown(f"### 🤖 Réponse\n{result.get('answer', 'Pas de réponse')}"))
    sources = result.get('sources', [])
    if sources:
        md = "#### 📚 Sources\n"
        for source in sources:
            concepts = source.get('detected_concepts', [])
            concept_str = f"**Concepts**: {', '.join(concepts)}" if concepts else ""
            md += f"- **Fichier**: `{source['filename']}` | **Score**: {source['relevance_score']:.2f} | {concept_str}\n"
        display(Markdown(md))


async def display_hierarchical_result(result: Dict[str, Any]):
    """Affiche les résultats de la recherche hiérarchique."""
    display(Markdown(f"### 🤖 Réponse\n{result.get('answer', 'Pas de réponse')}"))
    hierarchical_results = result.get('hierarchical_results', {})
    if hierarchical_results:
        md = "#### 📊 Résultats par niveau conceptuel\n"
        for level, data in hierarchical_results.items():
            md += f"- **{data.get('display_name', level)}** ({len(data.get('results', []))} résultats):\n"
            for i, res in enumerate(data.get('results', [])[:3]):
                md += f"  - `{res['source_info'].get('filename')}` (sim: {res['similarity']:.2f})\n"
        display(Markdown(md))


async def display_document_list(docs: List[Dict[str, Any]]):
    """Affiche la liste des documents."""
    if not docs:
        display(Markdown("📁 Aucun document n'a été indexé."))
        return
    md = f"### 📁 {len(docs)} documents indexés\n"
    for doc in docs:
        md += f"- `{doc.get('filename', 'N/A')}` ({doc.get('total_chunks', 0)} chunks)\n"
    display(Markdown(md))


# ==============================================================================
# 2. CLASSE DE LA MAGIC IPYTHON
# ==============================================================================

@magics_class
class OntoRAGMagic(Magics):
    def __init__(self, shell):
        super(OntoRAGMagic, self).__init__(shell)
        self.rag = None
        self._initialized = False
        print("✨ OntoRAG Magic prête. Initialisation au premier usage...")

    async def _initialize_rag(self):
        """Initialisation asynchrone du moteur RAG."""
        print("🚀 Initialisation du moteur OntoRAG (une seule fois)...")
        self.rag = OntoRAG(storage_dir=STORAGE_DIR, ontology_path=ONTOLOGY_PATH_TTL)
        await self.rag.initialize()
        self._initialized = True
        print("✅ Moteur OntoRAG initialisé et prêt.")

    async def _handle_agent_run(self, user_input: str):
        """Gère un tour de conversation avec l'agent unifié."""
        print("🧠 L'agent réfléchit...")

        # ✅ UTILISER L'AGENT UNIFIÉ avec la version structurée
        agent_response = await self.rag.unified_agent.run(user_input, use_memory=True)

        if agent_response.status == "clarification_needed":
            question_from_agent = agent_response.clarification_question
            display(Markdown(f"""### ❓ L'agent a besoin d'une clarification
    > {question_from_agent}

    **Pour répondre, utilisez la commande :** `%rag /agent_reply <votre_réponse>`"""))

        elif agent_response.status == "success":
            # Affichage enrichi avec les métadonnées
            display(Markdown(f"### ✅ Réponse finale de l'agent\n{agent_response.answer}"))

            # Afficher les sources automatiquement
            if agent_response.sources_consulted:
                sources_md = "\n## 📚 Sources consultées :\n"
                for source in agent_response.sources_consulted:
                    sources_md += f"\n{source.get_citation()}"
                display(Markdown(sources_md))

            # Afficher les métadonnées utiles
            metadata_md = f"""
    ### 📊 Métadonnées de la réponse
    - ⏱️ **Temps d'exécution**: {agent_response.execution_time_total_ms:.0f}ms
    - 🔢 **Étapes utilisées**: {agent_response.steps_taken}/{agent_response.max_steps}
    - 📚 **Sources consultées**: {len(agent_response.sources_consulted)}
    - 🎯 **Niveau de confiance**: {agent_response.confidence_level:.2f}
    """

            # Ajouter les questions de suivi suggérées
            if agent_response.suggested_followup_queries:
                metadata_md += f"\n### 💡 Questions de suivi suggérées :\n"
                for i, suggestion in enumerate(agent_response.suggested_followup_queries[:3], 1):
                    metadata_md += f"{i}. {suggestion}\n"

            display(Markdown(metadata_md))
            print("\n✅ Conversation terminée. Pour une nouvelle question, utilisez à nouveau `/agent`.")

        elif agent_response.status == "timeout":
            display(Markdown(f"""### ⏰ Timeout de l'agent
    L'agent a atteint la limite de temps mais a trouvé des informations partielles :

    {agent_response.answer}"""))

            if agent_response.sources_consulted:
                sources_md = "\n## 📚 Sources consultées malgré le timeout :\n"
                for source in agent_response.sources_consulted:
                    sources_md += f"\n{source.get_citation()}"
                display(Markdown(sources_md))

        elif agent_response.status == "error":
            display(Markdown(f"""### ❌ Erreur de l'agent
    {agent_response.error_details}

    Essayez de reformuler votre question ou utilisez `/help` pour voir les commandes disponibles."""))

        else:
            display(Markdown(f"### ⚠️ Statut inattendu : {agent_response.status}"))

    @line_cell_magic
    def rag(self, line, cell=None):
        """Magic command principale pour interagir avec OntoRAG."""

        async def main():
            if not self._initialized:
                await self._initialize_rag()

            query = cell.strip() if cell else line.strip()
            if not query:
                await show_available_commands()
                return

            parts = query.split(' ', 1)
            command = parts[0]
            args = parts[1].strip() if len(parts) > 1 else ""

            try:
                if command.startswith('/'):
                    # --- COMMANDES DE L'AGENT UNIFIÉ ---
                    if command == '/agent':
                        print("💫 Commande : /agent", args)
                        await self._handle_agent_run(args)

                    elif command == '/agent_reply':
                        print("💬 Réponse utilisateur :", args)
                        await self._handle_agent_run(args)

                    elif command == '/agent_memory':
                        """Affiche le résumé de la mémoire de l'agent."""
                        memory_summary = self.rag.unified_agent.get_memory_summary()
                        display(Markdown(f"### 🧠 Mémoire de l'agent\n{memory_summary}"))

                    elif command == '/agent_clear':
                        """Efface la mémoire de l'agent."""
                        self.rag.unified_agent.clear_memory()
                        display(Markdown("### 🧹 Mémoire de l'agent effacée"))

                    elif command == '/agent_sources':
                        """Affiche toutes les sources utilisées dans la session."""
                        sources = self.rag.unified_agent.get_sources_used()
                        if sources:
                            sources_md = f"### 📚 Sources de la session ({len(sources)} références)\n"
                            for source in sources:
                                sources_md += f"\n{source.get_citation()}"
                            display(Markdown(sources_md))
                        else:
                            display(Markdown("### 📚 Aucune source consultée dans cette session"))

                    elif command == '/add_docs':
                        var_name = args.strip()
                        documents_to_add = self.shell.user_ns.get(var_name)
                        if documents_to_add is None:
                            print(f"❌ Variable '{var_name}' non trouvée.")
                            return
                        print(f"📚 Ajout de {len(documents_to_add)} documents...")
                        results = await self.rag.add_documents_batch(documents_to_add, max_concurrent=MAX_CONCURRENT)
                        print(f"✅ Ajout terminé: {sum(results.values())}/{len(results)} succès.")

                    elif command == '/list':
                        docs = self.rag.list_documents()
                        await display_document_list(docs)

                    elif command == '/stats':
                        stats = self.rag.get_statistics()
                        display(Markdown(f"### 📊 Statistiques OntoRAG\n```json\n{json.dumps(stats, indent=2)}\n```"))

                    elif command == '/search':
                        result = await self.rag.query(args, max_results=MAX_RESULTS)
                        await display_query_result(result)

                    elif command == '/hierarchical':
                        result = await self.rag.hierarchical_query(args)
                        await display_hierarchical_result(result)

                    elif command == '/help':
                        await show_available_commands()

                    else:
                        print(f"❌ Commande inconnue: '{command}'.")
                        await show_available_commands()

                else:  # Requête en langage naturel directe
                    print("🤖 Requête directe via l'agent unifié...")
                    await self._handle_agent_run(query)

            except Exception as e:
                print(f"❌ Une erreur est survenue: {e}")
                import traceback
                traceback.print_exc()

        asyncio.run(main())


# ==============================================================================
# 3. FONCTION DE CHARGEMENT IPYTHON
# ==============================================================================

def load_ipython_extension(ipython):
    """Enregistre la classe de magics lors du chargement de l'extension."""
    ipython.register_magics(OntoRAGMagic)