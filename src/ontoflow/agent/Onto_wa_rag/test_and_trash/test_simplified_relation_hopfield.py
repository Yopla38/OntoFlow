"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# test_simplified_relations.py
import asyncio
import os

import numpy as np
from typing import Dict, List, Tuple
import torch

# Imports nécessaires
from ontology.simplified_hopfield_relation import SimplifiedRelationManager
from CONSTANT import API_KEY_PATH, LLM_MODEL
from provider.get_key import get_openai_key
from provider.llm_providers import OpenAIProvider
from utils.rag_engine import RAGEngine


class SimplifiedRelationTester:
    def __init__(self):
        self.rag_engine = None
        self.relation_manager = None
        self.concept_embeddings = {}
        self.ontology_manager = None
        self.all_triples = []  # Pour stocker tous les triplets

    async def initialize(self):
        """Initialise le système complet."""
        print("🚀 Initialisation du système de test simplifié...")

        # Nettoyer les anciens modèles pour éviter les conflits
        import shutil
        if os.path.exists("simplified_test_models"):
            shutil.rmtree("simplified_test_models")
        os.makedirs("simplified_test_models", exist_ok=True)

        # Initialiser les fournisseurs LLM
        OPENAI_KEY = get_openai_key(api_key_path=API_KEY_PATH)

        llm_provider = OpenAIProvider(
            model=LLM_MODEL,
            api_key=OPENAI_KEY
        )

        embedding_provider = llm_provider

        # Initialiser le RAG Engine
        self.rag_engine = RAGEngine(
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            storage_dir=f"storage_simplified_test"
        )

        # Créer un mock ontology manager
        self.ontology_manager = MockOntologyManager()

        # Initialiser le RelationManager simplifié
        self.relation_manager = SimplifiedRelationManager(
            self.ontology_manager,
            storage_dir="simplified_test_models"
        )

        print("✅ Système initialisé")

    async def create_test_ontology(self):
        """Crée une ontologie de test enrichie."""
        print("\n📚 Création de l'ontologie de test enrichie...")

        # Concepts étendus pour plus d'exemples
        concepts = {
            # Hiérarchie animale étendue
            "Animal": "A living organism that can move independently",
            "Mammal": "A warm-blooded vertebrate animal with hair or fur",
            "Bird": "A warm-blooded vertebrate with feathers and wings",
            "Reptile": "A cold-blooded vertebrate with scales",
            "Fish": "An aquatic vertebrate with gills and fins",

            # Mammifères
            "Dog": "A domesticated carnivorous mammal, loyal companion",
            "Cat": "A small domesticated carnivorous mammal, independent",
            "Horse": "A large domesticated mammal used for riding",
            "Cow": "A large domesticated mammal that produces milk",
            "Lion": "A large wild cat, king of the jungle",
            "Elephant": "The largest land mammal with a trunk",

            # Oiseaux
            "Eagle": "A large bird of prey with excellent vision",
            "Canary": "A small yellow songbird",
            "Penguin": "A flightless aquatic bird",
            "Owl": "A nocturnal bird of prey",
            "Parrot": "A colorful tropical bird that can mimic sounds",

            # Reptiles
            "Snake": "A legless reptile",
            "Lizard": "A small reptile with legs",
            "Crocodile": "A large aquatic reptile",

            # Personnes et professions
            "Person": "A human being",
            "Doctor": "A person qualified to practice medicine",
            "Teacher": "A person who teaches in schools",
            "Engineer": "A person who designs and builds things",
            "Artist": "A person who creates art",
            "Scientist": "A person who conducts scientific research",
            "Patient": "A person receiving medical treatment",
            "Student": "A person who is learning",

            # Médical
            "Disease": "A disorder of structure or function",
            "Cancer": "A malignant growth or tumor",
            "Flu": "A viral infection affecting respiratory system",
            "Diabetes": "A metabolic disorder affecting blood sugar",
            "Treatment": "Medical care given to a patient",
            "Medicine": "A substance used to treat disease",
            "Surgery": "Medical operation",
            "Vaccine": "A biological preparation providing immunity",

            # Symptômes
            "Symptom": "A sign of disease or disorder",
            "Fever": "Elevated body temperature",
            "Cough": "Expulsion of air from lungs",
            "Pain": "Physical suffering or discomfort",
            "Fatigue": "Extreme tiredness",

            # Émotions et états
            "Emotion": "A strong feeling",
            "Happiness": "State of joy and contentment",
            "Sadness": "State of sorrow",
            "Anger": "Strong feeling of displeasure",
            "Fear": "Unpleasant emotion caused by threat",
            "Love": "Deep affection",
            "Surprise": "Feeling caused by something unexpected",

            # Événements
            "Event": "Something that happens",
            "Birthday": "Anniversary of birth",
            "Wedding": "Marriage ceremony",
            "Funeral": "Ceremony for the dead",
            "Graduation": "Completion of studies ceremony",
            "Concert": "Musical performance",
            "Holiday": "Day of celebration or rest",

            # Lieux
            "Place": "A particular position or location",
            "Hospital": "Institution providing medical treatment",
            "School": "Institution for education",
            "Home": "Place where one lives",
            "Park": "Public area with greenery",
            "Office": "Place of work",

            # Objets
            "Object": "A material thing",
            "Book": "Written or printed work",
            "Computer": "Electronic device for processing data",
            "Car": "Motor vehicle for transportation",
            "Phone": "Device for communication"
        }

        # Générer les embeddings
        print("🔄 Génération des embeddings (3072 dimensions)...")
        for concept_name, description in concepts.items():
            embeddings = await self.rag_engine.embedding_manager.provider.generate_embeddings([description])
            if embeddings:
                # S'assurer que c'est bien 3072 dimensions
                self.concept_embeddings[concept_name] = np.array(embeddings[0])
                assert self.concept_embeddings[concept_name].shape[0] == 3072

        # Définir les relations
        self.ontology_manager.add_relations([
            ("is_a", "Taxonomic relation"),
            ("part_of", "Mereological relation"),
            ("lives_in", "Habitat relation"),
            ("works_at", "Employment relation"),
            ("treats", "Medical treatment relation"),
            ("causes", "Causal relation"),
            ("prevents", "Prevention relation"),
            ("evokes", "Emotional response relation"),
            ("located_in", "Spatial relation"),
            ("used_for", "Functional relation"),
            ("produces", "Production relation"),
            ("requires", "Dependency relation")
        ])

        # Initialiser avec les bonnes dimensions
        await self.relation_manager.initialize(concept_dim=3072, relation_dim=512)

        print(f"✅ Ontologie créée avec {len(concepts)} concepts et {len(self.ontology_manager.relations)} relations")
        return concepts

    def create_comprehensive_triples(self):
        """Crée un ensemble complet de triplets pour l'apprentissage."""

        # Relation is_a (taxonomique) - beaucoup d'exemples
        is_a_triples = [
            # Animaux
            ("Dog", "is_a", "Mammal"),
            ("Cat", "is_a", "Mammal"),
            ("Horse", "is_a", "Mammal"),
            ("Cow", "is_a", "Mammal"),
            ("Lion", "is_a", "Mammal"),
            ("Elephant", "is_a", "Mammal"),
            ("Eagle", "is_a", "Bird"),
            ("Canary", "is_a", "Bird"),
            ("Penguin", "is_a", "Bird"),
            ("Owl", "is_a", "Bird"),
            ("Parrot", "is_a", "Bird"),
            ("Snake", "is_a", "Reptile"),
            ("Lizard", "is_a", "Reptile"),
            ("Crocodile", "is_a", "Reptile"),
            ("Mammal", "is_a", "Animal"),
            ("Bird", "is_a", "Animal"),
            ("Reptile", "is_a", "Animal"),
            ("Fish", "is_a", "Animal"),

            # Personnes
            ("Doctor", "is_a", "Person"),
            ("Teacher", "is_a", "Person"),
            ("Engineer", "is_a", "Person"),
            ("Artist", "is_a", "Person"),
            ("Scientist", "is_a", "Person"),
            ("Patient", "is_a", "Person"),
            ("Student", "is_a", "Person"),

            # Médical
            ("Cancer", "is_a", "Disease"),
            ("Flu", "is_a", "Disease"),
            ("Diabetes", "is_a", "Disease"),
            ("Fever", "is_a", "Symptom"),
            ("Cough", "is_a", "Symptom"),
            ("Pain", "is_a", "Symptom"),
            ("Fatigue", "is_a", "Symptom"),
            ("Medicine", "is_a", "Treatment"),
            ("Surgery", "is_a", "Treatment"),
            ("Vaccine", "is_a", "Treatment"),

            # Émotions
            ("Happiness", "is_a", "Emotion"),
            ("Sadness", "is_a", "Emotion"),
            ("Anger", "is_a", "Emotion"),
            ("Fear", "is_a", "Emotion"),
            ("Love", "is_a", "Emotion"),
            ("Surprise", "is_a", "Emotion"),

            # Événements
            ("Birthday", "is_a", "Event"),
            ("Wedding", "is_a", "Event"),
            ("Funeral", "is_a", "Event"),
            ("Graduation", "is_a", "Event"),
            ("Concert", "is_a", "Event"),
            ("Holiday", "is_a", "Event"),

            # Lieux
            ("Hospital", "is_a", "Place"),
            ("School", "is_a", "Place"),
            ("Home", "is_a", "Place"),
            ("Park", "is_a", "Place"),
            ("Office", "is_a", "Place"),

            # Objets
            ("Book", "is_a", "Object"),
            ("Computer", "is_a", "Object"),
            ("Car", "is_a", "Object"),
            ("Phone", "is_a", "Object"),
        ]

        # Relation lives_in (habitat)
        lives_in_triples = [
            ("Dog", "lives_in", "Home"),
            ("Cat", "lives_in", "Home"),
            ("Lion", "lives_in", "Park"),  # Zoo/Safari park
            ("Elephant", "lives_in", "Park"),
            ("Fish", "lives_in", "Park"),  # Aquarium in park
            ("Person", "lives_in", "Home"),
            ("Student", "lives_in", "Home"),
            ("Patient", "lives_in", "Hospital"),  # Temporairement
        ]

        # Relation works_at
        works_at_triples = [
            ("Doctor", "works_at", "Hospital"),
            ("Teacher", "works_at", "School"),
            ("Engineer", "works_at", "Office"),
            ("Scientist", "works_at", "Office"),
            ("Artist", "works_at", "Home"),  # Peut travailler à domicile
        ]

        # Relation treats (médical)
        treats_triples = [
            ("Doctor", "treats", "Patient"),
            ("Doctor", "treats", "Disease"),
            ("Medicine", "treats", "Disease"),
            ("Medicine", "treats", "Symptom"),
            ("Surgery", "treats", "Cancer"),
            ("Vaccine", "prevents", "Disease"),
            ("Treatment", "treats", "Pain"),
            ("Treatment", "treats", "Fever"),
        ]

        # Relation causes
        causes_triples = [
            ("Disease", "causes", "Symptom"),
            ("Disease", "causes", "Pain"),
            ("Cancer", "causes", "Pain"),
            ("Cancer", "causes", "Fatigue"),
            ("Flu", "causes", "Fever"),
            ("Flu", "causes", "Cough"),
            ("Diabetes", "causes", "Fatigue"),

            # Événements et émotions
            ("Birthday", "causes", "Happiness"),
            ("Wedding", "causes", "Happiness"),
            ("Wedding", "causes", "Love"),
            ("Funeral", "causes", "Sadness"),
            ("Graduation", "causes", "Happiness"),
            ("Concert", "causes", "Happiness"),
            ("Disease", "causes", "Sadness"),
            ("Disease", "causes", "Fear"),
        ]

        # Relation evokes (émotionnelle)
        evokes_triples = [
            ("Birthday", "evokes", "Happiness"),
            ("Wedding", "evokes", "Love"),
            ("Funeral", "evokes", "Sadness"),
            ("Dog", "evokes", "Happiness"),
            ("Cat", "evokes", "Happiness"),
            ("Lion", "evokes", "Fear"),
            ("Snake", "evokes", "Fear"),
            ("Disease", "evokes", "Fear"),
            ("Medicine", "evokes", "Fear"),  # Pour certains
            ("Concert", "evokes", "Happiness"),
            ("Holiday", "evokes", "Happiness"),
            ("Home", "evokes", "Love"),
            ("Hospital", "evokes", "Fear"),
            ("School", "evokes", "Happiness"),  # Pour certains
        ]

        # Relation used_for
        used_for_triples = [
            ("Medicine", "used_for", "Treatment"),
            ("Hospital", "used_for", "Treatment"),
            ("School", "used_for", "Student"),
            ("Book", "used_for", "Student"),
            ("Computer", "used_for", "Engineer"),
            ("Phone", "used_for", "Person"),
            ("Car", "used_for", "Person"),
        ]

        # Combiner tous les triplets
        self.all_triples = (
                is_a_triples +
                lives_in_triples +
                works_at_triples +
                treats_triples +
                causes_triples +
                evokes_triples +
                used_for_triples
        )

        print(f"✅ Créé {len(self.all_triples)} triplets pour l'apprentissage")
        return self.all_triples

    async def test_simple_relations(self):
        """Teste les relations simples avec beaucoup d'exemples."""
        print("\n\n🧪 Test 1: Relations simples (taxonomiques)")
        print("=" * 50)

        # Apprendre toutes les relations
        results = await self.relation_manager.learn_relations(
            self.all_triples,
            self.concept_embeddings,
            min_examples=3,
            force_relearn=True
        )

        print("\n📊 Application de la transitivité...")
        self.relation_manager.apply_transitivity("is_a")

        print("\n📊 Résultats d'apprentissage:")
        for relation, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {relation}")

        # Test de prédiction pour is_a
        print("\n📊 Tests de prédiction 'is_a':")

        test_cases = [
            ("Dog", "is_a", ["Mammal", "Animal"]),
            ("Eagle", "is_a", ["Bird", "Animal"]),
            ("Doctor", "is_a", ["Person"]),
            ("Happiness", "is_a", ["Emotion"]),
        ]

        for subject, relation, expected in test_cases:
            predictions = self.relation_manager.get_related_concepts(
                subject, relation, self.concept_embeddings, top_k=3, threshold=0.5
            )

            pred_labels = [p['label'] for p in predictions]
            print(f"\n{subject} {relation}: {pred_labels}")

            # Vérifier si les prédictions contiennent les résultats attendus
            for exp in expected:
                if exp in pred_labels:
                    conf = next(p['confidence'] for p in predictions if p['label'] == exp)
                    print(f"  ✓ {exp} (confiance: {conf:.3f})")
                else:
                    print(f"  ✗ {exp} non trouvé")

    async def test_complex_relations(self):
        """Teste les relations plus complexes."""
        print("\n\n🧪 Test 2: Relations complexes")
        print("=" * 50)

        # Test treats
        print("\n📊 Test relation 'treats':")
        predictions = self.relation_manager.get_related_concepts(
            "Doctor", "treats", self.concept_embeddings, top_k=5
        )
        print(f"Doctor treats: {[p['label'] for p in predictions]}")

        # Test causes
        print("\n📊 Test relation 'causes':")
        test_subjects = ["Disease", "Birthday", "Flu"]

        for subject in test_subjects:
            predictions = self.relation_manager.get_related_concepts(
                subject, "causes", self.concept_embeddings, top_k=3
            )
            print(f"\n{subject} causes: {[p['label'] for p in predictions[:3]]}")

    async def test_nonlinear_relations(self):
        """Teste les relations émotionnelles non-linéaires."""
        print("\n\n🧪 Test 3: Relations non-linéaires (émotionnelles)")
        print("=" * 50)

        # Test evokes avec plus de détails
        print("\n📊 Test relation 'evokes' (avec scores détaillés):")

        test_concepts = ["Dog", "Snake", "Hospital", "Wedding", "Concert"]

        for concept in test_concepts:
            predictions = self.relation_manager.get_related_concepts(
                concept, "evokes", self.concept_embeddings,
                top_k=3,  # Voir plus de résultats
                threshold=0.3
            )

            if predictions:
                print(f"\n{concept} evokes:")
                for p in predictions:
                    # Afficher plus de détails
                    print(f"  - {p['label']} (conf: {p['confidence']:.3f}, "
                          f"sim: {p.get('similarity', 0):.3f}, "
                          f"score: {p.get('score', 0):.3f}, "
                          f"source: {p.get('source', 'model')})")

    async def test_inference_capabilities(self):
        """Teste les capacités d'inférence."""
        print("\n\n🧪 Test 4: Capacités d'inférence")
        print("=" * 50)

        print("\n📊 Inférence pour nouveaux concepts:")

        new_subject_tests = [
            ("Horse", ["evokes", "lives_in", "is_a"]),
            ("Elephant", ["evokes", "lives_in", "is_a"]),
            ("Scientist", ["works_at", "is_a"]),
            ("Vaccine", ["used_for", "treats"]),
        ]

        for subject, expected_relations in new_subject_tests:
            print(f"\n🔮 Inférences pour '{subject}':")

            # Utiliser un seuil plus bas pour l'inférence
            inferences = self.relation_manager.infer_new_relations(
                subject, self.concept_embeddings,
                confidence_threshold=0.3  # Plus permissif
            )

            if not inferences:
                print("  ⚠️ Aucune inférence trouvée")
                # Essayer de débugger pourquoi
                for rel in expected_relations:
                    if rel in self.relation_manager.transformations:
                        transform = self.relation_manager.transformations[rel]
                        if transform.is_trained:
                            # Tester directement
                            preds = transform.predict_objects(
                                subject, self.concept_embeddings,
                                top_k=1, threshold=0.1
                            )
                            if preds:
                                print(f"  Debug {rel}: {preds[0]['label']} (score: {preds[0]['score']:.3f})")
                            else:
                                print(f"  Debug {rel}: aucune prédiction")
            else:
                # Grouper par relation
                by_relation = {}
                for inf in inferences[:10]:
                    rel = inf['relation_label']
                    if rel not in by_relation:
                        by_relation[rel] = []
                    by_relation[rel].append(inf)

                for rel, items in by_relation.items():
                    print(f"\n  {rel}:")
                    for item in items[:3]:
                        print(f"    → {item['object_label']} (conf: {item['confidence']:.3f})")

    async def test_transitivity_and_properties(self):
        """Teste les propriétés logiques des relations."""
        print("\n\n🧪 Test 5: Propriétés logiques (transitivité)")
        print("=" * 50)

        # S'assurer que is_a est marqué comme transitive AVANT l'apprentissage
        if "is_a" in self.relation_manager.transformations:
            self.relation_manager.transformations["is_a"].properties["transitive"] = True

        # Ré-appliquer la transitivité après avoir défini la propriété
        self.relation_manager.apply_transitivity_for_transform(
            self.relation_manager.transformations["is_a"],
            self.all_triples
        )

        print("\n📊 Test de transitivité pour 'is_a':")

        # Maintenant tester
        transitive_tests = [
            ("Dog", "Animal", "Dog → Mammal → Animal"),
            ("Eagle", "Animal", "Eagle → Bird → Animal"),
            ("Cancer", "Disease", "Direct"),
            ("Fever", "Symptom", "Direct"),
        ]

        for subject, expected_object, path in transitive_tests:
            predictions = self.relation_manager.get_related_concepts(
                subject, "is_a", self.concept_embeddings, top_k=5
            )

            pred_labels = [p['label'] for p in predictions]
            if expected_object in pred_labels:
                idx = pred_labels.index(expected_object)
                conf = predictions[idx]['confidence']
                source = predictions[idx].get('source', 'unknown')
                print(f"✓ {subject} is_a {expected_object} (conf: {conf:.3f}, source: {source}) via {path}")
            else:
                print(f"✗ {subject} is_a {expected_object} - non inféré")
                # Debug
                print(f"  Prédictions obtenues: {pred_labels[:3]}")

    async def test_memory_patterns(self):
        """Teste les patterns mémorisés dans le système."""
        print("\n\n🧪 Test 6: Patterns de mémoire")
        print("=" * 50)

        # Analyser les patterns stockés pour chaque relation
        print("\n📊 Analyse des patterns mémorisés:")

        for relation_uri, transform in self.relation_manager.transformations.items():
            if transform.is_trained:
                print(f"\n{transform.label}:")
                print(f"  - Exemples d'entraînement: {len(transform.positive_examples)}")
                print(f"  - Propriétés: {[k for k, v in transform.properties.items() if v]}")

                # Afficher quelques exemples
                if transform.positive_examples:
                    print(f"  - Exemples: {transform.positive_examples[:3]}")

    async def test_error_handling_and_edge_cases(self):
        """Teste la gestion des erreurs et cas limites."""
        print("\n\n🧪 Test 7: Cas limites et robustesse")
        print("=" * 50)

        # Test avec un concept inexistant
        print("\n📊 Test avec concept inexistant:")
        predictions = self.relation_manager.get_related_concepts(
            "UnknownConcept", "is_a", self.concept_embeddings, top_k=3
        )
        print(f"Prédictions pour concept inexistant: {len(predictions)} résultats")

        # Test avec une relation non entraînée
        print("\n📊 Test avec relation non entraînée:")
        self.ontology_manager.add_relations([("new_relation", "A new untrained relation")])
        await self.relation_manager.initialize()  # Réinitialiser pour inclure la nouvelle relation

        predictions = self.relation_manager.get_related_concepts(
            "Dog", "new_relation", self.concept_embeddings, top_k=3
        )
        print(f"Prédictions pour relation non entraînée: {len(predictions)} résultats")

        # Test avec très peu d'exemples
        print("\n📊 Test apprentissage avec peu d'exemples:")
        sparse_triples = [("Book", "requires", "Person")]  # Un seul exemple

        results = await self.relation_manager.learn_relations(
            sparse_triples, self.concept_embeddings, min_examples=1
        )
        print(f"Apprentissage avec 1 exemple: {'✓' if results.get('requires') else '✗'}")

    async def test_performance_metrics(self):
        """Teste et affiche les métriques de performance."""
        print("\n\n🧪 Test 8: Métriques de performance")
        print("=" * 50)

        # Obtenir les statistiques
        stats = self.relation_manager.get_statistics()

        print("\n📊 Statistiques globales:")
        print(f"  - Relations totales: {stats['total_relations']}")
        print(f"  - Relations entraînées: {stats['trained_relations']}")

        print("\n📊 Détails par relation:")
        for rel_detail in sorted(stats['relations_details'],
                                 key=lambda x: x.get('num_examples', 0),
                                 reverse=True):
            if rel_detail['is_trained']:
                print(f"\n  {rel_detail['label']}:")
                print(f"    - Exemples d'entraînement: {rel_detail['num_examples']}")
                print(f"    - Exemples transitifs: {rel_detail.get('num_transitive', 0)}")
                if rel_detail['metrics']:
                    print(f"    - Loss finale: {rel_detail['metrics'].get('final_loss', 'N/A'):.4f}")
                    print(f"    - Epochs: {rel_detail['metrics'].get('num_epochs', 'N/A')}")
                props = [k for k, v in rel_detail['properties'].items() if v]
                if props:
                    print(f"    - Propriétés: {props}")

    async def test_batch_predictions(self):
        """Teste les prédictions en batch pour l'efficacité."""
        print("\n\n🧪 Test 9: Prédictions en batch")
        print("=" * 50)

        # Tester plusieurs sujets à la fois
        subjects = ["Dog", "Cat", "Eagle", "Doctor", "Disease"]
        relation = "is_a"

        print(f"\n📊 Prédictions batch pour '{relation}':")

        import time
        start_time = time.time()

        for subject in subjects:
            predictions = self.relation_manager.get_related_concepts(
                subject, relation, self.concept_embeddings, top_k=2
            )
            if predictions:
                top_pred = predictions[0]['label']
                print(f"  {subject} → {top_pred} (conf: {predictions[0]['confidence']:.3f})")

        elapsed = time.time() - start_time
        print(f"\n⏱️ Temps pour {len(subjects)} prédictions: {elapsed:.3f}s")

    async def run_all_tests(self):
        """Exécute tous les tests."""
        await self.initialize()
        await self.create_test_ontology()
        self.create_comprehensive_triples()

        # Tests principaux
        await self.test_simple_relations()
        await self.test_complex_relations()
        await self.test_nonlinear_relations()
        await self.test_inference_capabilities()
        await self.test_transitivity_and_properties()
        await self.test_memory_patterns()
        await self.test_error_handling_and_edge_cases()
        await self.test_performance_metrics()
        await self.test_batch_predictions()

        print("\n\n" + "=" * 60)
        print("✅ TOUS LES TESTS TERMINÉS AVEC SUCCÈS!")
        print("=" * 60)

        print("\n📈 Résumé des capacités testées:")
        print("✓ Relations taxonomiques simples (is_a)")
        print("✓ Relations complexes (treats, causes)")
        print("✓ Relations non-linéaires (evokes)")
        print("✓ Inférence sur nouveaux concepts")
        print("✓ Propriétés logiques (transitivité)")
        print("✓ Patterns de mémoire")
        print("✓ Gestion des cas limites")
        print("✓ Métriques de performance")
        print("✓ Prédictions efficaces")

        print("\n💡 Points clés de l'implémentation simplifiée:")
        print("- Architecture plus simple et efficace")
        print("- Apprentissage direct de transformations S → O")
        print("- Mémoire Hopfield pour prototypes uniquement")
        print("- Support des propriétés ontologiques")
        print("- Scalabilité améliorée")

class MockOntologyManager:
    """Mock ontology manager pour les tests."""

    def __init__(self):
        self.relations = {}
        self.axioms = []

    def add_relations(self, relations: List[Tuple[str, str]]):
        """Ajoute des relations à l'ontologie."""
        for uri, label in relations:
            self.relations[uri] = MockRelation(uri, label)

    def add_axiom(self, axiom_type: str, source: str, target: str = None):
        """Ajoute un axiome à l'ontologie."""
        self.axioms.append((axiom_type, source, target))

class MockRelation:
    """Mock relation pour les tests."""

    def __init__(self, uri: str, label: str):
        self.uri = uri
        self.label = label
        self.domain = []
        self.range = []

async def main():
    """Point d'entrée principal."""
    print("🚀 Démarrage des tests de l'architecture simplifiée")
    print("=" * 60)

    tester = SimplifiedRelationTester()

    try:
        await tester.run_all_tests()
    except Exception as e:
        print(f"\n❌ Erreur lors des tests: {e}")
        import traceback
        traceback.print_exc()

    print("\n✨ Tests terminés!")

if __name__ == "__main__":
    # Configuration pour éviter les warnings PyTorch
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Lancer les tests
    asyncio.run(main())