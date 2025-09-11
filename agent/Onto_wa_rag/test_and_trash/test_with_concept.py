"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# test_concept_classifier.py
import asyncio
import os

from agent.agent.Onto_wa_rag.CONSTANT import LLM_MODEL
from provider.get_key import get_openai_key

from utils.rag_engine import RAGEngine
from provider.llm_providers import OpenAIProvider
from ontology.ontology_manager import OntologyManager
from ontology.classifier import OntologyClassifier


# Chemin des clefs API LLM
API_KEY_PATH = "/home/yopla/Documents/keys/"


async def test_concept_classification():
    """Test du classifieur de concepts basé sur Hopfield Networks"""
    print("=== Test du Classifieur de Concepts ===")

    # 1. Initialiser le RAG standard
    print("\n1. Initialisation du RAG existant...")
    OPENAI_KEY = get_openai_key(api_key_path=API_KEY_PATH)
    llm_provider = OpenAIProvider(
        model=LLM_MODEL,
        api_key=OPENAI_KEY
    )

    # Même provider pour les embeddings
    embedding_provider = llm_provider

    rag_engine = RAGEngine(
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        storage_dir="storage"
    )

    await rag_engine.initialize()
    print("✓ RAG initialisé")

    # 2. Initialiser l'OntologyManager et charger l'ontologie
    print("\n2. Chargement de l'ontologie scientifique...")
    ontology_manager = OntologyManager(storage_dir="ontology_data")

    success = ontology_manager.load_ontology("simple.jsonld")
    if not success:
        print("❌ Échec du chargement de l'ontologie")
        return

    print(f"✓ Ontologie chargée: {len(ontology_manager.concepts)} concepts")

    # 3. Récupérer/ajouter les documents de test
    print("\n3. Préparation des documents de test...")
    all_documents = await rag_engine.get_all_documents()

    # Si pas assez de documents, en ajouter
    if len(all_documents) < 10:
        print("Ajout de documents d'exemple...")

        # Créer quelques documents simples pour les tests
        with open("documents/ia_deep_learning.md", "w", encoding="utf-8") as f:
            f.write("""# Intelligence Artificielle et Deep Learning

L'intelligence artificielle (IA) est transformée par les techniques modernes de deep learning.
Les réseaux de neurones profonds permettent aujourd'hui de résoudre des problèmes complexes
qui étaient impossibles auparavant.

## Les Réseaux de Neurones

Les réseaux de neurones artificiels, inspirés du cerveau humain, sont la base du deep learning.
Le perceptron multicouche est un modèle fondamental qui a évolué vers des architectures plus complexes.

## Transformers et NLP

Les modèles Transformers comme BERT et GPT ont révolutionné le traitement du langage naturel.
Basés sur des mécanismes d'attention, ils permettent de comprendre le contexte des mots
dans des phrases entières.
""")

        with open("documents/physique_quantique.md", "w", encoding="utf-8") as f:
            f.write("""# Introduction à la Physique Quantique

La mécanique quantique est l'une des théories fondamentales de la physique moderne.
Elle décrit le comportement de la matière à l'échelle atomique et subatomique.

## Dualité Onde-Particule

En physique quantique, les particules comme les électrons présentent des propriétés
à la fois d'ondes et de particules. Cette dualité est un concept fondamental.

## Relation avec la Mécanique Classique

La mécanique quantique étend et remplace les lois de Newton de la mécanique classique
pour expliquer des phénomènes que cette dernière ne peut pas décrire correctement.
""")

        with open("documents/mathematiques_avancees.md", "w", encoding="utf-8") as f:
            f.write("""# Mathématiques Avancées pour les Sciences

Les mathématiques avancées fournissent les outils fondamentaux utilisés dans de nombreux
domaines scientifiques, de la physique à l'informatique.

## Algèbre Linéaire et Applications

L'algèbre linéaire est essentielle pour comprendre les espaces vectoriels et les transformations
linéaires. Elle est largement utilisée en intelligence artificielle pour les réseaux de neurones.

## Analyse Fonctionnelle

L'analyse fonctionnelle est une branche des mathématiques qui étudie les espaces de fonctions.
Elle est particulièrement importante en physique quantique et en mécanique quantique.
""")

        document_paths = [
            "documents/ia_deep_learning.md",
            "documents/physique_quantique.md",
            "documents/mathematiques_avancees.md",
            "documents/informatique_quantique_et_IA.md",
            "documents/Mathemtique_et_reseaux_profonds.md",
            "documents/Maxwell_technologies_quantiques.md",
            "documents/Neuroscience_et_IA.md"
        ]

        document_ids = []
        for path in document_paths:
            if os.path.exists(path):
                doc_id = await rag_engine.add_document(path)
                document_ids.append(doc_id)
                print(f"✓ Document ajouté: {path} (ID: {doc_id})")
            else:
                print(f"❌ Document non trouvé: {path}")
    else:
        # Utiliser les documents existants
        document_ids = list(all_documents.keys())[:3]
        print(f"✓ Utilisation de {len(document_ids)} documents existants")

    if len(document_ids) < 1:
        print("❌ Nombre insuffisant de documents pour le test")
        return

    # 4. Initialiser le classifieur ontologique avec ConceptHopfieldClassifier
    print("\n4. Initialisation du classifieur de concepts...")
    classifier = OntologyClassifier(
        rag_engine=rag_engine,
        ontology_manager=ontology_manager,
        storage_dir="classifier_data",
        use_hierarchical=True,
        enable_concept_classification=True
    )

    await classifier.initialize()
    print("✓ Classifieur de concepts initialisé")


    # 6. Tester la classification de concepts
    print("\n6. Test de classification par concepts...")

    for i, doc_id in enumerate(document_ids[:3]):
        doc_info = await rag_engine.document_store.get_document(doc_id)
        doc_name = os.path.basename(doc_info.get("path", f"Document {i + 1}"))

        print(f"\nClassification de {doc_name} (ID: {doc_id}):")

        # Obtenir la classification par concepts
        result = await classifier.classify_document_concepts(doc_id, force_refresh=True)

        if "concepts" in result and result["concepts"]:
            for concept in result["concepts"]:
                print(f"- {concept['label']} (confiance: {concept['confidence']:.2f})")

                # Afficher les sous-concepts
                for sub in concept.get("sub_concepts", []):
                    print(f"  └─ {sub['label']} (confiance: {sub['confidence']:.2f})")

                    # Afficher les sous-sous-concepts
                    for subsub in sub.get("sub_concepts", []):
                        print(f"     └─ {subsub['label']} (confiance: {subsub['confidence']:.2f})")
        else:
            print("Aucun concept détecté")

    # 7. Tester la recherche contextuelle par concept...
    print("\n7. Test de recherche contextuelle par concept...")

    queries = [
        {
            "query": "Comment fonctionnent les transformers en NLP?",
            "concept": "http://example.org/scientific-ontology#ModelesTransformers",
            "description": "Question sur les modèles transformers"
        },
        {
            "query": "Quelles sont les implications de la dualité onde-particule?",
            "concept": "http://example.org/scientific-ontology#PhysiqueQuantique",
            "description": "Question sur la physique quantique"
        },
        {
            "query": "Comment les réseaux de neurones utilisent-ils l'algèbre linéaire?",
            "concept": "http://example.org/scientific-ontology#ReseauxDeNeurones",
            "description": "Question sur les réseaux de neurones et les mathématiques"
        }
    ]

    for query_info in queries:
        concept_uri = query_info["concept"]

        # Vérifier si le concept existe et obtenir son label
        concept = ontology_manager.concepts.get(concept_uri)
        concept_label = concept.label if concept else concept_uri.split('#')[-1]

        print(f"\n{query_info['description']}:")
        print(f"Question: {query_info['query']}")
        print(f"Contexte: Concept '{concept_label}'")

        # Effectuer la recherche par concept avec un seuil de confiance plus bas
        result = await classifier.search_by_concept(
            query=query_info["query"],
            concept_uri=concept_uri,
            include_subconcepts=True,
            confidence_threshold=0.4  # Seuil plus permissif
        )

        if "answer" in result:
            print(f"\nRéponse en contexte du concept {concept_label}:")
            print(f"{result['answer']}")

            if "concepts_included" in result:
                print(f"\nConcepts consultés: {', '.join(result['concepts_included'])}")

            if "documents_used" in result:
                print(f"Documents utilisés: {len(result['documents_used'])}")

            if "passages" in result and result["passages"]:
                source = result["passages"][0].get("document_name", "source inconnue")
                print(f"Source principale: {source}")
        else:
            print(f"❌ Erreur: {result.get('error', 'Échec de la recherche')}")

    # 8. Tester la recherche automatique par concept
    print("\n8. Test de recherche automatique par concept...")

    test_queries = [
        "Comment fonctionnent les transformers et quelle est leur application au NLP?",
        "Quelles sont les théories fondamentales de la physique quantique?",
        "Comment les réseaux de neurones s'inspirent-ils du cerveau humain?"
    ]

    for test_query in test_queries:
        print(f"\nRequête: {test_query}")

        # Effectuer la recherche automatique
        result = await classifier.auto_concept_search(test_query, include_semantic_relations=True)

        if "concepts_detected" in result:
            print("Concepts détectés automatiquement:")
            for concept in result["concepts_detected"]:
                print(f"- {concept['label']} (confiance: {concept['confidence']:.2f})")

        if "answer" in result:
            print("\nRéponse:")
            print(result["answer"])

            if "passages" in result and result["passages"]:
                source = result["passages"][0].get("document_name", "source inconnue")
                print(f"Source principale: {source}")
        else:
            print(f"❌ Erreur: {result.get('error', 'Échec de la recherche')}")

    # 9. Tester la recherche automatique complexe par concept
    print("\n9. Test de recherche automatique par intersection de concept...")

    test_queries = [
        "Expliquez l'algorithme HHL quantique?",
        "Comment la théorie de l'attention s'applique en neurosciences et en IA?",
        "Quelles sont les différentes applications de l'électrodynamique quantique?"
    ]

    for test_query in test_queries:
        print(f"\nRequête: {test_query}")

        # Effectuer la recherche automatique
        result = await classifier.auto_concept_search(test_query, include_semantic_relations=True)

        if "concepts_detected" in result:
            print("Concepts détectés automatiquement:")
            for concept in result["concepts_detected"]:
                print(f"- {concept['label']} (confiance: {concept['confidence']:.2f})")

        if "answer" in result:
            print("\nRéponse:")
            print(result["answer"])

            if "passages" in result and result["passages"]:
                source = result["passages"][0].get("document_name", "source inconnue")
                print(f"Source principale: {source}")
        else:
            print(f"❌ Erreur: {result.get('error', 'Échec de la recherche')}")

    print("\n=== Test terminé ===")


if __name__ == "__main__":
    asyncio.run(test_concept_classification())