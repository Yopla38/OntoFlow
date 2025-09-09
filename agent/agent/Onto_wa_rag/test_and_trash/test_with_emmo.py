"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# test_full_rag.py
import asyncio
import os

from provider.get_key import get_openai_key
from provider.llm_providers import OpenAIProvider

# Importer notre RAG Engine et les composants ontologiques
from utils.rag_engine import RAGEngine
from utils.wavelet_rag import WaveletRAG
from ontology.ontology_manager import OntologyManager
from ontology.classifier import OntologyClassifier
from utils.enhanced_document_processor import EnhancedDocumentProcessor

# Chemin des clefs API LLM
API_KEY_PATH = "/home/yopla/Documents/keys/"


async def test_full_rag_system():
    """Test complet du système RAG avec ontologie et métadonnées enrichies"""
    print("=== Test complet du système RAG avancé ===\n")

    # 1. Initialiser les fournisseurs LLM
    print("1. Initialisation des fournisseurs LLM...")
    OPENAI_KEY = get_openai_key(api_key_path=API_KEY_PATH)

    llm_provider = OpenAIProvider(
        model="gpt-4o",
        api_key=OPENAI_KEY
    )

    # Même provider pour les embeddings
    embedding_provider = llm_provider

    # 2. Initialiser le RAG avec processeur de documents enrichi
    print("\n2. Initialisation du système RAG avec métadonnées enrichies...")
    processor = EnhancedDocumentProcessor(chunk_size=1000, chunk_overlap=200)

    rag_engine = RAGEngine(
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        storage_dir="storage_enhanced"
    )

    # Remplacer le processeur standard par le processeur enrichi
    rag_engine.processor = processor

    await rag_engine.initialize()
    print("✓ RAG Engine initialisé")

    # 3. Initialiser l'OntologyManager et charger l'ontologie
    print("\n3. Chargement de l'ontologie...")
    ontology_manager = OntologyManager(storage_dir="ontology_data")

    # Utiliser le fichier d'ontologie simple.jsonld de votre projet
    ontology_file = "emmo.jsonld"
    if not os.path.exists(ontology_file):
        print(f"⚠️ Fichier d'ontologie {ontology_file} non trouvé !")
        exit()

    else:
        success = ontology_manager.load_ontology(ontology_file)
        if success:
            print(f"✓ Ontologie chargée: {len(ontology_manager.concepts)} concepts")
        else:
            print("❌ Échec du chargement de l'ontologie")
            return

    # 4. Initialiser le classifieur ontologique
    print("\n4. Initialisation du classifieur ontologique...")
    classifier = OntologyClassifier(
        rag_engine=rag_engine,
        ontology_manager=ontology_manager,
        storage_dir="classifier_data",
        use_hierarchical=True,
        enable_concept_classification=True
    )

    await classifier.initialize()
    print("✓ Classifieur ontologique initialisé")

    # 5. Initialiser le WaveletRAG
    print("\n5. Initialisation du WaveletRAG...")
    wavelet_rag = WaveletRAG(
        rag_engine=rag_engine,
        wavelet="db3",
        levels=3,
        storage_dir="wavelet_storage"
    )

    await wavelet_rag.initialize()
    print("✓ WaveletRAG initialisé")

    # 6. Ajouter les documents PDF
    print("\n6. Ajout des documents PDF...")
    pdf_files = ["documents/1.pdf", "documents/2.pdf", "documents/3.pdf"]
    document_ids = []

    for pdf_path in pdf_files:
        if os.path.exists(pdf_path):
            print(f"Traitement de {pdf_path}...")
            doc_id = await rag_engine.add_document(pdf_path)
            document_ids.append(doc_id)
            print(f"✓ Document ajouté avec ID: {doc_id}")
        else:
            print(f"⚠️ Fichier non trouvé: {pdf_path}")

    if not document_ids:
        print("❌ Aucun document ajouté. Vérifiez que les documents existent dans le dossier 'documents/'.")
        return

    # 7. Classifier les documents avec l'ontologie
    print("\n7. Classification des documents dans l'ontologie...")

    for doc_id in document_ids:
        print(f"\nClassification du document {doc_id}:")

        # Classifier par domaine
        domain_result = await classifier.classify_document(doc_id, force_refresh=True)
        if "domains" in domain_result and domain_result["domains"]:
            print("Domaines détectés:")
            for domain in domain_result["domains"]:
                print(f"- {domain['domain']} (confiance: {domain['confidence']:.2f})")
                for subdomain in domain.get("sub_domains", []):
                    print(f"  └─ {subdomain['domain']} (confiance: {subdomain['confidence']:.2f})")
        else:
            print("Aucun domaine détecté")

        # Classifier par concept
        concept_result = await classifier.classify_document_concepts(doc_id, force_refresh=True)
        if "concepts" in concept_result and concept_result["concepts"]:
            print("\nConcepts détectés:")
            for concept in concept_result["concepts"]:
                print(f"- {concept['label']} (confiance: {concept['confidence']:.2f})")
                for sub in concept.get("sub_concepts", []):
                    print(f"  └─ {sub['label']} (confiance: {sub['confidence']:.2f})")
        else:
            print("Aucun concept détecté")

    # 8. Tester la recherche standard
    print("\n8. Test de recherche standard:")
    query = "Quels sont les concepts clés présentés dans les documents?"  # Remplacez par votre question

    # Recherche avec RAG standard
    result = await rag_engine.chat(query, top_k=3)

    print(f"\nRéponse du RAG standard pour: '{query}'")
    print(f"Réponse: {result['answer']}")
    print(f"Nombre de passages utilisés: {len(result['passages'])}")

    # 9. Tester la recherche avec WaveletRAG
    print("\n9. Test de recherche avec WaveletRAG:")
    query = "Expliquez les concepts techniques mentionnés dans les documents."  # Remplacez par votre question

    # Recherche avec WaveletRAG
    wavelet_result = await wavelet_rag.chat(query, top_k=3)

    print(f"\nRéponse du WaveletRAG pour: '{query}'")
    print(f"Réponse: {wavelet_result['answer']}")
    print(f"Nombre de passages utilisés: {len(wavelet_result['passages'])}")

    # 10. Tester la recherche par concept
    print("\n10. Test de recherche contextuelle par concept:")

    # Trouver un concept qui existe dans l'ontologie
    concept_uri = None
    for uri in ontology_manager.concepts:
        concept = ontology_manager.concepts[uri]
        if concept.label and "apprentissage" in concept.label.lower():
            concept_uri = uri
            break

    if not concept_uri:
        # Fallback si aucun concept sur l'apprentissage n'est trouvé
        concept_uri = list(ontology_manager.concepts.keys())[0]

    concept = ontology_manager.concepts[concept_uri]
    concept_label = concept.label or concept_uri.split('#')[-1]

    query = f"Expliquez comment le concept de {concept_label} est abordé dans les documents."
    print(f"\nRequête contextuelle sur le concept '{concept_label}':")
    print(f"Question: {query}")

    concept_result = await classifier.search_by_concept(
        query=query,
        concept_uri=concept_uri,
        include_subconcepts=True,
        top_k=3
    )

    if "answer" in concept_result:
        print(f"\nRéponse basée sur le concept '{concept_label}':")
        print(f"{concept_result['answer']}")

        if "concepts_included" in concept_result:
            print(f"\nConcepts consultés: {', '.join(concept_result['concepts_included'])}")
    else:
        print(f"❌ Erreur: {concept_result.get('error', 'Aucune réponse générée')}")

    # 11. Tester la détection automatique de concepts
    print("\n11. Test de détection automatique de concepts:")
    query = "Quelles sont les principales théories et méthodes décrites dans ces documents?"
    print(f"Question: {query}")

    auto_concept_result = await classifier.auto_concept_search(
        query=query,
        include_semantic_relations=True
    )

    if "concepts_detected" in auto_concept_result:
        print("\nConcepts détectés automatiquement:")
        for concept in auto_concept_result["concepts_detected"]:
            print(f"- {concept['label']} (confiance: {concept['confidence']:.2f})")

    if "answer" in auto_concept_result:
        print("\nRéponse avec détection automatique de concepts:")
        print(auto_concept_result["answer"])
    else:
        print(f"❌ Erreur: {auto_concept_result.get('error', 'Aucune réponse générée')}")

    print("\n=== Test terminé ===")


if __name__ == "__main__":
    asyncio.run(test_full_rag_system())