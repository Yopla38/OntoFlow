import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List

from agent.Onto_wa_rag.Integration_fortran_RAG import OntoRAG
from agent.Onto_wa_rag.CONSTANT import API_KEY_PATH, CHUNK_SIZE, CHUNK_OVERLAP, ONTOLOGY_PATH_TTL, MAX_CONCURRENT, MAX_RESULTS, \
    STORAGE_DIR, FORTRAN_AGENT_NB_STEP
from agent.Onto_wa_rag.fortran_analysis.providers.consult import FortranEntityExplorer


# Imports pour le RAG
from agent.Onto_wa_rag.utils.rag_engine import RAGEngine
from agent.Onto_wa_rag.provider.llm_providers import OpenAIProvider
from agent.Onto_wa_rag.provider.get_key import get_openai_key


async def example_usage():
    """Exemple d'utilisation d'OntoRAG avec tests du système de contexte"""

    rag = OntoRAG(
        storage_dir=STORAGE_DIR,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        ontology_path=ONTOLOGY_PATH_TTL
    )

    await rag.initialize()

    from files_to_index import DOCUMENTS
    import os

    """
    DOCUMENTS = [
        {"filepath": os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_folder/PSbox.f90"),
         "project_name": "BigDFT", "version": "1.9"},
        # Ajoutez ici d'autres documents si vous le souhaitez
    ]
    """
    # Traitement parallèle
    results = await rag.add_documents_batch(DOCUMENTS, max_concurrent=MAX_CONCURRENT)
    print(f"Ajout terminé: {sum(results.values())}/{len(results)} succès")

    # Statistiques
    stats = rag.get_statistics()

    print("\n" + "=" * 100)
    print("🚀 ONTORAG - SYSTÈME DE RECHERCHE DOCUMENTAIRE INTELLIGENT")
    print("=" * 100)

    await show_available_commands()

    while True:
        try:
            query = input('\n💫 Commande : ').strip()

            if query.lower() in ['/quit', '/exit', 'quit', 'exit', 'q']:
                print("👋 Au revoir !")
                break

            elif query in ['/?', '/help', 'help']:
                await show_available_commands()

            # ==================== RECHERCHE ====================
            elif query.startswith('/search '):
                question = query[8:].strip()
                if question:
                    print(f"🔍 Recherche ontologique: {question}")
                    result = await rag.query(question, use_ontology=True)
                    await display_query_result(result)

            elif query.startswith('/hierarchical '):
                # Parser la commande avec options
                parts = query[13:].strip().split()
                if not parts:
                    print("❌ Usage: /hierarchical <question> [--mode=auto|text|fortran|unified]")
                    continue

                # Extraire la question et les options
                question_parts = []
                mode = 'auto'

                for part in parts:
                    if part.startswith('--mode='):
                        mode = part.split('=', 1)[1]
                    else:
                        question_parts.append(part)

                question = ' '.join(question_parts)

                if question:
                    # print(f"🔍 Recherche hiérarchique intelligente: {question} (mode: {mode})")
                    result = await rag.hierarchical_query(question, max_per_level=3, mode=mode)
                    await display_intelligent_hierarchical_result(result)

            elif query.startswith('/find '):
                entity_name = query[6:].strip()
                if entity_name:
                    print(f"🔍 Recherche entité Fortran: {entity_name}")
                    entities = await rag.search_fortran_entities(entity_name)
                    await display_fortran_entities(entities)

            # ==================== GESTION DOCUMENTS ====================

            elif query.startswith('/list'):
                docs = rag.list_documents()
                await display_document_list(docs)

            elif query.startswith('/stats'):
                detail = query[6:].strip()
                if detail == 'fortran':
                    fortran_stats = await rag.get_fortran_stats()
                    print("📊 Statistiques Fortran:")
                    print(json.dumps(fortran_stats, indent=2))
                elif detail == 'entity':
                    entity_stats = await rag.get_entity_manager_stats()
                    print("📊 Statistiques EntityManager:")
                    print(json.dumps(entity_stats, indent=2))
                else:
                    stats = rag.get_statistics()
                    print("📊 Statistiques générales:")
                    print(json.dumps(stats, indent=2))

            # ==================== OUTILS & DIAGNOSTIC ====================
            elif query.startswith('/diagnostic'):
                component = query[11:].strip()
                if component == 'fortran' or not component:
                    diagnosis = await rag.diagnose_fortran_entity_manager()
                    print("🔍 Diagnostic EntityManager:")
                    print(json.dumps(diagnosis, indent=2))

            elif query.startswith('/visualization'):
                print('🎨 Génération visualisation dépendances...')
                html_file = await rag.generate_dependency_visualization(
                    output_file="dependencies.html",
                    max_entities=2000,
                    include_variables=False
                )
                if html_file:
                    print(f"✅ Généré: {html_file}")
                    import webbrowser
                    import os
                    webbrowser.open('file://' + os.path.abspath(html_file))

            elif query.startswith('/consult_entity'):
                entity = query[15:].strip()
                # Si aucun document n'est chargé, l'explorateur sera vide.
                if not rag.custom_processor.fortran_processor.entity_manager.entities:
                    print("\nEntityManager est vide. Ajoutez des documents pour l'utiliser.")
                    return

                explorer = FortranEntityExplorer(rag.custom_processor.fortran_processor.entity_manager,
                                                 rag.ontology_manager)

                report = await explorer.get_full_report(entity)
                # 4. Afficher le rapport de manière lisible
                if "error" in report:
                    print(f"\n--- ERREUR ---")
                    print(report["error"])
                else:
                    print(f"\n--- RAPPORT COMPLET POUR : {report['entity_name']} ---")

                    print("\n[ Résumé ]")
                    for key, value in report['summary'].items():
                        print(f"  - {key.replace('_', ' ').capitalize()}: {value}")

                    print("\n[ Relations Sortantes (ce que cette entité utilise) ]")
                    for key, value in report['outgoing_relations'].items():
                        print(f"  - {key.replace('_', ' ').capitalize()}:")
                        if value:
                            for item in value:
                                print(f"    - {item}")
                        else:
                            print("    - (Aucun)")

                    print("\n[ Relations Entrantes (qui utilise cette entité) ]")
                    if report['incoming_relations']:
                        for caller in report['incoming_relations']:
                            print(f"  - {caller['name']} (type: {caller['type']}, file: {caller.get('file', 'N/A')})")
                    else:
                        print("  - (Appelée par personne)")

                    print("\n[ Contexte Global (où se situe cette entité) ]")
                    parent = report['global_context']['parent_entity']
                    if isinstance(parent, dict):
                        print(f"  - Parent: {parent['name']} (type: {parent['type']})")
                    else:
                        print(f"  - Parent: {parent}")

                    print("\n[ Contexte Local (ce que contient cette entité) ]")
                    children = report['local_context']['children_entities']
                    if children:
                        print("  - Entités enfants:")
                        for child in children:
                            print(f"    - {child['name']} (type: {child['type']})")
                    else:
                        print("  - Pas d'entités enfants.")

                    print("\n[ Concepts associés à cette entité ]")
                    concepts = report['detected_concepts']
                    if concepts:
                        for concept in concepts:
                            print(f"  - {concept['label']} (confiance: {concept['confidence']})")

                    print("\n--- Code Source ---")
                    print(report['local_context']['source_code'])
                    print("--- FIN DU RAPPORT ---")

            elif query.startswith('/refresh'):
                scope = query[8:].strip()
                if scope == 'fortran' or not scope:
                    print("🔄 Réindexation Fortran...")
                    await rag.refresh_fortran_index()
                    print("✅ Réindexation terminée")

            elif query.startswith('/agent '):
                # Récupérer la requête initiale de l'utilisateur
                current_input = query[7:].strip()

                if current_input:
                    # Démarrer une boucle de conversation qui continue tant que l'agent a besoin de clarifications.
                    while True:
                        print("🧠 L'agent réfléchit...")

                        # Appeler l'agent avec l'entrée actuelle.
                        # use_memory=True est CRUCIAL ici pour que l'agent se souvienne du contexte
                        # de sa propre question.
                        agent_response = await rag.agent_fortran.run(current_input, use_memory=True)

                        # Vérifier si la réponse de l'agent est une demande de clarification
                        if agent_response.startswith("CLARIFICATION_NEEDED:"):
                            # 1. Extraire la question de la chaîne de caractères spéciale
                            question_from_agent = agent_response.replace("CLARIFICATION_NEEDED:", "").strip()

                            # 2. Afficher la question à l'utilisateur de manière claire
                            print(f"\n❓ L'agent a besoin d'une clarification pour continuer :")
                            print(f"   '{question_from_agent}'")

                            # 3. Demander une réponse à l'utilisateur via la console
                            user_clarification = input("\nVotre réponse > ")

                            # 4. La réponse de l'utilisateur devient la prochaine entrée pour l'agent
                            current_input = user_clarification

                            # La boucle `while` va maintenant se relancer avec cette nouvelle entrée.

                        else:
                            # Si la réponse n'est PAS une clarification, c'est la réponse finale.
                            print("\n--- RÉPONSE FINALE DE L'AGENT ---")
                            print(agent_response)

                            # Sortir de la boucle de conversation.
                            break

            elif query.startswith("/agent_memory"):
                print("\n--- Mémoire de l'agent ---")
                print("\n" + rag.agent_fortran.get_memory_summary())

            elif query.startswith("/agent_clear"):
                rag.agent_fortran.clear_memory()
                print("🧠 Mémoire de l'agent effacée.")

            # ==================== REQUÊTE NATURELLE ====================
            elif not query.startswith('/'):
                # Requête naturelle - utiliser query() standard
                print(f"💭 Requête naturelle: {query}")
                await rag.ask(query)

            else:
                print("❌ Commande inconnue. Tapez '/help' pour l'aide")

        except KeyboardInterrupt:
            print("\n👋 Au revoir !")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")


async def show_available_commands():
    """Affiche toutes les commandes disponibles utilisant les méthodes existantes"""
    print(f"""
🔍 RECHERCHE
   <question>                   Requête naturelle directe
   /find <entité>               Recherche entités Fortran (méthode: search_fortran_entities)
   /consult_entity              Données brut sur une entité fortran (rapport)
   /agent <question>            Deploit un agent de questionnement (utilise pour les grands résumés et suivre une discussion)
   /agent_clear                 Efface la mémoire de l'agent

📊 RÉSUMÉS & STATISTIQUES
   /stats [detail]              Statistiques (get_statistics/get_fortran_stats)
   /stats entity                Stats EntityManager

📁 GESTION DOCUMENTS
   /list                        Lister documents (list_documents)

🔧 OUTILS & DIAGNOSTIC
   /diagnostic                  Diagnostic système (diagnose_fortran_entity_manager)
   /visualization               Graphique dépendances (generate_dependency_visualization)
   /refresh                     Réindexer (refresh_fortran_index)

❓ AIDE
   /help                        Cette aide
   /quit                        Quitter

""")


async def display_query_result(result: Dict[str, Any]):
    """Affiche le résultat d'une query() standard"""
    print(f"\n🤖 Réponse: {result.get('answer', 'Pas de réponse')}")

    sources = result.get('sources', [])
    if sources:
        print(f"\n📚 Sources ({len(sources)}):")
        for source in sources:
            print(f"\n  📄 {source['filename']} (lignes {source['start_line']}-{source['end_line']})")
            print(f"     Type: {source['entity_type']} - Nom: {source['entity_name']}")
            if source.get('detected_concepts'):
                print(f"     Concepts: {', '.join(source['detected_concepts'])}")
            print(f"     Score: {source['relevance_score']}")


async def display_intelligent_hierarchical_result(result: Dict[str, Any]):
    """Affiche les résultats de la recherche hiérarchique intelligente"""

    print(f"\n🤖 Réponse: {result.get('answer', 'Pas de réponse')}")

    search_mode = result.get('search_mode', 'unknown')
    #print(f"\n🎯 Mode de recherche utilisé: {search_mode}")

    hierarchical_results = result.get('hierarchical_results', {})
    if hierarchical_results:
        print(f"\n📊 Résultats par niveau conceptuel:")

        for conceptual_level, level_data in hierarchical_results.items():
            display_name = level_data.get('display_name', conceptual_level)
            results = level_data.get('results', [])

            print(f"\n📚 {display_name} ({len(results)} résultats):")

            for i, res in enumerate(results[:3]):  # Top 3 par niveau
                if 'entity' in res:
                    # Résultat Fortran
                    entity = res['entity']
                    print(f"  {i + 1}. 🔧 {entity.entity_name} ({entity.entity_type}) "
                          f"in {entity.filename} (sim: {res['similarity']:.2f})")
                else:
                    # Résultat texte
                    source_info = res.get('source_info', {})
                    title = source_info.get('section_title', 'Sans titre')
                    content_type = source_info.get('content_type', 'unknown')
                    icon = "🔧" if content_type == 'fortran' else "📄"
                    print(f"  {i + 1}. {icon} {title} "
                          f"in {source_info.get('filename', 'Unknown')} (sim: {res['similarity']:.2f})")

    total_passages = result.get('total_passages', 0)
    conceptual_levels = result.get('conceptual_levels_found', [])

    print(f"\n📊 Résumé: {total_passages} passages trouvés sur {len(conceptual_levels)} niveaux conceptuels")


async def display_fortran_entities(entities: List[Dict[str, Any]]):
    """Affiche les entités Fortran trouvées"""
    if entities:
        print(f"🔍 {len(entities)} entités Fortran trouvées:")
        for i, entity in enumerate(entities[:10], 1):
            confidence_icon = "🟢" if entity.get('confidence', 0) > 0.8 else "🟡" if entity.get('confidence',
                                                                                              0) > 0.5 else "🔴"
            print(f"  {i}. {confidence_icon} {entity['name']} ({entity['type']}) in {Path(entity['file']).name}")
            print(f"     Confidence: {entity.get('confidence', 0):.2f}, Match: {entity.get('match_type', 'unknown')}")
    else:
        print("❌ Aucune entité Fortran trouvée")


async def display_document_list(docs: List[Dict[str, Any]]):
    """Affiche la liste des documents"""
    if not docs:
        print("📁 Aucun document chargé")
        return

    print(f"📁 {len(docs)} documents chargés:")

    by_project = {}
    for doc in docs:
        project = doc.get('project', 'Unknown')
        if project not in by_project:
            by_project[project] = []
        by_project[project].append(doc)

    for project, project_docs in by_project.items():
        print(f"\n  📂 Projet: {project} ({len(project_docs)} docs)")
        for doc in project_docs:  # Top 5 par projet
            file_type = doc.get('file_type', 'unknown')
            chunks = doc.get('total_chunks', 0)
            print(f"    • {doc.get('filename', 'Unknown')} ({file_type}, {chunks} chunks)")


if __name__ == "__main__":
    asyncio.run(example_usage())