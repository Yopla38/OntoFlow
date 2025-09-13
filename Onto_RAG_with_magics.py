import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys
import nest_asyncio


from agent.Onto_wa_rag.Integration_fortran_RAG import OntoRAG
from agent.Onto_wa_rag.CONSTANT import (
    API_KEY_PATH, CHUNK_SIZE, CHUNK_OVERLAP, ONTOLOGY_PATH_TTL,
    MAX_CONCURRENT, MAX_RESULTS, STORAGE_DIR
)
from agent.Onto_wa_rag.fortran_analysis.providers.consult import FortranEntityExplorer

#from chat import Chat
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

#### 🔍 **Recherche (Modes Différents)**
- **`<question>`**: (Sans `/`) **Recherche simple et rapide** avec similarité sémantique
- **`/simple_search <query>`**: Recherche sémantique (5 résultats)
- **`/simple_search_more <query>`**: Recherche sémantique (10 résultats)
- **`/search <question>`**: Recherche classique du RAG avec réponse générée
- **`/hierarchical <q>`**: Recherche hiérarchique sur plusieurs niveaux

---

#### 🧠 **Agent Unifié (Analyse Approfondie)**
- **`/agent <question>`**: **Analyse complète** avec l'agent unifié (Fortran + Jupyter)
- **`/agent_reply <réponse>`**: Répond à une question de clarification de l'agent
- **`/agent_memory`**: Affiche le résumé de la mémoire actuelle de l'agent
- **`/agent_clear`**: Efface la mémoire de l'agent
- **`/agent_sources`**: Affiche toutes les sources consultées dans la session

---

#### 📁 **Gestion des Documents**
- **`/add_docs <var_name>`**: Ajoute des documents depuis une variable Python
- **`/list`**: Liste tous les documents indexés
- **`/stats`**: Affiche les statistiques du RAG

---

#### ❓ **Aide**
- **`/help`**: Affiche ce message d'aide

---

### 🎯 **Quand utiliser quel mode ?**

| Mode | Cas d'usage | Vitesse | Précision |
|------|-------------|---------|-----------|
| **Recherche simple** (`query`) | Recherche rapide de contenu | ⚡⚡⚡ | 🎯🎯 |
| **Recherche classique** (`/search`) | Question avec réponse générée | ⚡⚡ | 🎯🎯🎯 |
| **Agent unifié** (`/agent`) | Analyse complexe, multi-fichiers | ⚡ | 🎯🎯🎯🎯 |
| **Recherche hiérarchique** (`/hierarchical`) | Recherche structurée par niveaux | ⚡ | 🎯🎯🎯 |

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
        retriever = self.rag.unified_agent.semantic_retriever

        # Réindexation à la demande si nécessaire
        if len(retriever.chunks) == 0:
            print("  🔄 Index vide, construction automatique...")
            notebook_count = retriever.build_index_from_existing_chunks(self.rag)

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

    async def _handle_simple_search(self, query: str, max_results: int = 5):
        """Effectue une recherche simple + génération de réponse avec le LLM."""
        print(f"🔍 Recherche simple RAG : '{query}'")

        # Vérifier si l'agent unifié est disponible
        if not hasattr(self.rag, 'unified_agent') or not self.rag.unified_agent:
            display(Markdown("❌ **Agent unifié non disponible**\n\nUtilisez `/search` pour la recherche classique."))
            return

        retriever = self.rag.unified_agent.semantic_retriever

        # Réindexation à la demande si nécessaire
        if len(retriever.chunks) == 0:
            print("  🔄 Index vide, construction automatique...")
            notebook_count = retriever.build_index_from_existing_chunks(self.rag)

            if notebook_count == 0:
                display(Markdown(f"""❌ **Aucun notebook disponible**

    Les documents indexés ne contiennent pas de notebooks Jupyter (.ipynb).

    **Alternatives :**
    - `/search {query}` pour la recherche classique
    - `/list` pour voir les documents disponibles"""))
                return

        # 1. Effectuer la recherche sémantique
        results = retriever.query(query, k=max_results)

        if not results:
            display(Markdown(f"""### 🔍 Recherche : "{query}"

    ❌ **Aucun résultat trouvé** (seuil de similarité : 0.25)

    **Suggestions :**
    - Essayez des termes plus généraux
    - `/agent {query}` pour une analyse approfondie  
    - `/search {query}` pour la recherche classique"""))
            return

        # 2. Générer la réponse avec le LLM
        print(f"  🤖 Génération de la réponse avec {len(results)} chunks de contexte...")

        try:
            answer, sources_info = await self._generate_rag_response(query, results)

            # 3. Afficher la réponse complète
            await self._display_rag_response(query, answer, results, sources_info)

        except Exception as e:
            print(f"❌ Erreur génération réponse: {e}")
            # Fallback : afficher les chunks bruts
            display(Markdown("⚠️ **Erreur de génération LLM**, affichage des chunks bruts :"))
            await self._display_simple_search_results(query, results)

    async def _generate_rag_response(self, query: str, results: List[Dict[str, Any]]) -> Tuple[
        str, List[Dict[str, Any]]]:
        """Génère une réponse avec le LLM à partir des chunks trouvés."""

        # 1. Construire le contexte depuis les chunks
        context_parts = []
        sources_info = []

        for i, result in enumerate(results, 1):
            source_filename = result.get("source_filename", "Unknown")
            content = result.get("content", "")
            similarity_score = result.get("similarity_score", 0.0)

            # Ajouter au contexte
            context_parts.append(f"[Source {i}] De {source_filename} (score: {similarity_score:.3f}):\n{content}")

            # Info pour les sources finales
            sources_info.append({
                "index": i,
                "filename": source_filename,
                "score": similarity_score,
                "source_file": result.get("source_file", ""),
                "tokens": result.get("tokens", "?")
            })

        context = "\n\n".join(context_parts)

        # 2. Construire le prompt pour le LLM
        system_prompt = """Tu es un assistant expert qui répond aux questions en citant TOUJOURS ses sources.

    Tu as accès aux sources suivantes provenant de notebooks Jupyter :
    - Cite OBLIGATOIREMENT tes sources en utilisant [Source N] dans ta réponse
    - Concentre-toi sur les informations les plus pertinentes
    - Structure ta réponse de manière claire et pratique
    - N'invente aucune information qui ne figure pas dans les sources

    Exemple de citation: "D'après [Source 1], pour créer une molécule..."
    """

        user_prompt = f"""Question: {query}

    Contexte disponible:
    {context}

    Réponds à la question en utilisant exclusivement les informations du contexte et en citant tes sources [Source N]."""

        # 3. Appeler le LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        llm_response = await self.rag.rag_engine.llm_provider.generate_response(messages, temperature=0.3)

        return llm_response, sources_info

    async def _display_rag_response(self, query: str, answer: str, results: List[Dict], sources_info: List[Dict]):
        """Affiche la réponse RAG complète avec métadonnées."""

        # En-tête avec statistiques
        total_indexed = len(self.rag.unified_agent.semantic_retriever.chunks)
        indexed_files_count = len(self.rag.unified_agent.semantic_retriever.indexed_files)
        avg_score = sum(r.get("similarity_score", 0) for r in results) / len(results)

        header = f"""### 🤖 Réponse RAG : "{query}"

    📊 **Contexte :** {len(results)} chunks sélectionnés sur {total_indexed} disponibles ({indexed_files_count} notebooks) | **Score moyen :** {avg_score:.3f}

    ---

    """

        # Corps de la réponse
        response_body = f"""### 💡 Réponse

    {answer}

    ---

    """

        # Sources détaillées
        sources_section = "### 📚 Sources utilisées\n\n"
        for source in sources_info:
            score_bar = "🟢" * int(source["score"] * 10) + "⚪" * (10 - int(source["score"] * 10))
            sources_section += f"""**[Source {source['index']}]** `{source['filename']}` | Score: {source['score']:.3f} {score_bar} | Tokens: {source['tokens']}\n\n"""

        # Actions suggérées
        footer = f"""
    ---

    ### 💡 Actions suggérées

    - **Analyse approfondie :** `%rag /agent {query}`
    - **Plus de contexte :** `%rag /simple_search_more {query}`  
    - **Voir les chunks bruts :** `%rag /simple_search_raw {query}`
    """

        # Afficher tout
        display(Markdown(header + response_body + sources_section))

    async def _display_simple_search_results(self, query: str, results: List[Dict[str, Any]]):
        """Affiche les résultats de la recherche simple de manière attractive."""

        # En-tête avec statistiques
        total_indexed = len(self.rag.unified_agent.semantic_retriever.chunks)
        indexed_files = len(self.rag.unified_agent.semantic_retriever.indexed_files)

        header = f"""### 🔍 Résultats de recherche : "{query}"

    📊 **{len(results)} résultat(s) trouvé(s)** sur {total_indexed} chunks indexés ({indexed_files} notebooks)

    ---
    """
        display(Markdown(header))

        # Afficher chaque résultat
        for i, result in enumerate(results, 1):
            score = result["similarity_score"]
            source_file = result["source_filename"]
            tokens = result.get("tokens", "?")
            content = result["content"]

            # Tronquer le contenu si trop long
            if len(content) > 800:
                content_preview = content[:800] + "\n\n*[...contenu tronqué...]*"
            else:
                content_preview = content

            # Barre de score visuelle
            score_bar = "🟢" * int(score * 10) + "⚪" * (10 - int(score * 10))

            result_md = f"""
    #### 📄 Résultat {i}/{len(results)}

    **📁 Source :** `{source_file}` | **🎯 Score :** {score:.3f} {score_bar} | **📝 Tokens :** {tokens}

    ```
    {content_preview}
    ```

    ---
    """
            display(Markdown(result_md))

    async def _handle_execute(self):
        """Récupère la dernière cellule commentaire, extrait le code avec le LLM et l'affiche."""
        try:
            # 1. Récupérer l'historique des cellules
            history = list(self.shell.history_manager.get_range())

            if not history:
                display(Markdown("❌ **Aucune cellule dans l'historique**"))
                return

            # 2. Chercher la dernière cellule qui semble être un commentaire/texte
            last_comment_cell = None
            for session, line_num, cell_content in reversed(history):
                # On cherche une cellule qui contient principalement du texte/commentaires
                # ou qui commence par des commentaires
                cell_content = cell_content.strip()
                if cell_content and (
                        cell_content.startswith('#') or
                        cell_content.startswith('"""') or
                        cell_content.startswith("'''") or
                        len([line for line in cell_content.split('\n') if line.strip().startswith('#')]) > 2
                ):
                    last_comment_cell = cell_content
                    break

            if not last_comment_cell:
                display(Markdown("""❌ **Aucune cellule commentaire trouvée**

    Recherche effectuée pour :
    - Cellules commençant par `#`
    - Cellules commençant par `\"\"\"` ou `'''`
    - Cellules avec plus de 2 lignes de commentaires
                """))
                return

            # 3. Construire le prompt pour le LLM
            system_prompt = """You are a code extraction expert. Your job is to extract executable Python code from comments and text descriptions.

    RULES:
    - Extract ONLY the Python code that can be executed
    - Remove ALL comments, docstrings, and explanatory text
    - Return ONLY the raw, executable Python code
    - Do not add any explanations or markdown formatting
    - If there's no executable code, return "# No executable code found"
    - Preserve the logical structure and indentation of the code"""

            user_prompt = f"""Extract the executable Python code from this text/comment:

    ```
    {last_comment_cell}
    ```

    Return ONLY the raw Python code without any comments or explanations:"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # 4. Appeler le LLM
            print("  🤖 Extraction du code avec le LLM...")
            extracted_code = await self.rag.rag_engine.llm_provider.generate_response(
                messages,
                temperature=0.1  # Température basse pour plus de précision
            )

            # 5. Nettoyer la réponse (enlever les balises markdown si présentes)
            extracted_code = extracted_code.strip()
            if extracted_code.startswith('```python'):
                extracted_code = extracted_code[9:]  # Enlever ```python
            if extracted_code.startswith('```'):
                extracted_code = extracted_code[3:]  # Enlever ```
            if extracted_code.endswith('```'):
                extracted_code = extracted_code[:-3]  # Enlever ``` final

            extracted_code = extracted_code.strip()


            # 6. Afficher le résultat
            if extracted_code and extracted_code != "# No executable code found":
                display(Markdown(f"""### 🚀 Start the HPC agent..."""))
                # TODO regler le pb d'import (deplacer ce fichier un cran au dessus et utiliser venv de agent)
                #chat = Chat(model="gpt-5")
                message = f"""
                Use the remote_run_code tool to run the following python function on 'robin-ubuntu':
                function_source='{extracted_code}, with function_args={{}}'
                """

                #await chat.chat(message)
                #chat.print_history()
            else:
                display(Markdown(f"""### ❌ No code found..."""))

        except Exception as e:
            print(f"❌ Erreur lors de l'extraction du code: {e}")
            import traceback
            traceback.print_exc()

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

                    elif command == '/execute':
                        print("🚀 Extraction et exécution de code de la dernière cellule commentaire...")
                        await self._handle_execute()

                    else:
                        print(f"❌ Commande inconnue: '{command}'.")
                        await show_available_commands()

                else:  # Requête en langage naturel directe
                    print("🤖 Requête directe via SimpleRetriever...")
                    await self._handle_simple_search(query, max_results=5)

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