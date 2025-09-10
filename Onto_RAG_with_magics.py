import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List
import sys
import nest_asyncio


from agent.agent.Onto_wa_rag.Integration_fortran_RAG import OntoRAG
from agent.agent.Onto_wa_rag.CONSTANT import (
    API_KEY_PATH, CHUNK_SIZE, CHUNK_OVERLAP, ONTOLOGY_PATH_TTL,
    MAX_CONCURRENT, MAX_RESULTS, STORAGE_DIR
)
from agent.agent.Onto_wa_rag.fortran_analysis.providers.consult import FortranEntityExplorer

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

#### 🧠 **Agent Conversationnel (Fonctionnalité Principale)**
- **`/agent <question>`**: Démarre une nouvelle conversation avec l'agent. La mémoire est effacée au début.
- **`/agent_reply <réponse>`**: Répond à une question de clarification de l'agent pour continuer la conversation.
- **`/agent_memory`**: Affiche le résumé de la mémoire actuelle de l'agent.
- **`/agent_clear`**: Efface la mémoire de l'agent et termine la conversation en cours.

---

#### 🔍 **Recherche et Consultation**
- **`<question>`**: (Sans `/`) Lance une requête directe en langage naturel.
- **`/search <question>`**: Effectue une recherche sémantique simple.
- **`/hierarchical <q>`**: Lance une recherche hiérarchique sur plusieurs niveaux conceptuels.

---

#### 📁 **Gestion des Documents**
- **`/add_docs <var_name>`**: Ajoute des documents à l'index depuis une variable Python du notebook.
- **`/list`**: Liste tous les documents actuellement indexés.
- **`/stats`**: Affiche les statistiques générales du RAG.

---

#### ❓ **Aide**
- **`/help`**: Affiche ce message d'aide.
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
        """Gère un tour de conversation avec l'agent."""
        print("🧠 L'agent réfléchit...")
        agent_response = await self.rag.agent_fortran.run(user_input, use_memory=True)

        if agent_response.startswith("CLARIFICATION_NEEDED:"):
            question_from_agent = agent_response.replace("CLARIFICATION_NEEDED:", "").strip()
            display(Markdown(f"### ❓ L'agent a besoin d'une clarification\n> {question_from_agent}\n\n"
                             f"**Pour répondre, utilisez la commande :** `%rag /agent_reply <votre_réponse>`"))
        else:
            display(Markdown(f"### ✅ Réponse finale de l'agent\n{agent_response}"))
            print("\nConversation terminée. Pour une nouvelle question, utilisez à nouveau `/agent`.")

    # LA CORRECTION EST ICI : la fonction n'est plus 'async def'
    @line_cell_magic
    def rag(self, line, cell=None):
        """Magic command principale pour interagir avec OntoRAG."""

        # Nous définissons une fonction asynchrone interne pour contenir la logique
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
                    # --- COMMANDES DE L'AGENT ---
                    if command == '/agent':
                        print("🆕 Nouvelle conversation avec l'agent. Mémoire effacée.")
                        self.rag.agent_fortran.clear_memory()
                        await self._handle_agent_run(args)
                    # ... (le reste de la logique des commandes reste identique) ...
                    elif command == '/agent_reply':
                        await self._handle_agent_run(args)
                    elif command == '/add_docs':
                        var_name = args.strip()
                        documents_to_add = self.shell.user_ns.get(var_name)
                        if documents_to_add is None:
                            print(f"❌ Variable '{var_name}' non trouvée.")
                            return
                        print(f"📚 Ajout de {len(documents_to_add)} documents...")
                        results = await self.rag.add_documents_batch(documents_to_add, max_concurrent=MAX_CONCURRENT)
                        print(f"✅ Ajout terminé: {sum(results.values())}/{len(results)} succès.")
                    # (Toutes les autres commandes /help, /list, etc. ici)
                    else:
                        print(f"❌ Commande inconnue: '{command}'.")

                else:  # Requête en langage naturel
                    await self.rag.ask(query)

            except Exception as e:
                print(f"❌ Une erreur est survenue: {e}")
                import traceback
                traceback.print_exc()

        # Et ici, nous exécutons notre fonction asynchrone 'main'
        # Grâce à nest_asyncio, asyncio.run() fonctionne correctement même dans Jupyter.
        asyncio.run(main())


# ==============================================================================
# 3. FONCTION DE CHARGEMENT IPYTHON
# ==============================================================================

def load_ipython_extension(ipython):
    """Enregistre la classe de magics lors du chargement de l'extension."""
    ipython.register_magics(OntoRAGMagic)