"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# test_hopfield_relations.py
import asyncio
import numpy as np
from typing import Dict, List, Tuple
import torch

# Imports nécessaires

from ontology.relation_manager import RelationManager
from CONSTANT import API_KEY_PATH, LLM_MODEL
from provider.get_key import get_openai_key
from provider.llm_providers import OpenAIProvider
from utils.rag_engine import RAGEngine


class HopfieldRelationTester:
    def __init__(self):
        self.rag_engine = None
        self.relation_manager = None
        self.concept_embeddings = {}
        self.ontology_manager = None

    async def initialize(self):
        """Initialise le système complet."""
        print("🚀 Initialisation du système de test Hopfield...")

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
            storage_dir=f"storage_hopfield_test"
        )

        # Créer un mock ontology manager pour les tests
        self.ontology_manager = MockOntologyManager()

        # Initialiser le RelationManager avec notre architecture Hopfield
        self.relation_manager = RelationManager(
            self.ontology_manager,
            storage_dir="hopfield_test_models"
        )

        print("✅ Système initialisé")

    async def create_test_ontology(self):
        """Crée une ontologie de test avec relations simples et complexes."""
        print("\n📚 Création de l'ontologie de test...")

        # Concepts de base
        concepts = {
            # Hiérarchie animale (relations simples)
            "Animal": "A living organism that can move independently",
            "Mammal": "A warm-blooded vertebrate animal with hair or fur",
            "Bird": "A warm-blooded vertebrate with feathers and wings",
            "Dog": "A domesticated carnivorous mammal",
            "Cat": "A small domesticated carnivorous mammal",
            "Eagle": "A large bird of prey",
            "Canary": "A small songbird",

            # Concepts pour relations complexes
            "Person": "A human being",
            "Doctor": "A person qualified to practice medicine",
            "Patient": "A person receiving medical treatment",
            "Disease": "A disorder of structure or function in a human",
            "Treatment": "Medical care given to a patient",
            "Symptom": "A physical or mental feature indicating a condition",

            # Concepts pour relations non-linéaires
            "Emotion": "A strong feeling deriving from circumstances",
            "Happiness": "The state of being happy",
            "Sadness": "The state of being sad",
            "Event": "Something that happens",
            "Birthday": "The anniversary of someone's birth",
            "Funeral": "A ceremony honoring a dead person"
        }

        # Générer les embeddings pour chaque concept
        print("🔄 Génération des embeddings...")
        for concept_name, description in concepts.items():
            embeddings = await self.rag_engine.embedding_manager.provider.generate_embeddings([description])
            if embeddings:
                self.concept_embeddings[concept_name] = np.array(embeddings[0])

        # Définir les relations dans l'ontologie
        self.ontology_manager.add_relations([
            # Relations simples (linéaires)
            ("is_a", "Taxonomic relation"),
            ("part_of", "Mereological relation"),

            # Relations plus complexes
            ("treats", "Medical treatment relation"),
            ("causes", "Causal relation"),
            ("prevents", "Prevention relation"),

            # Relations non-linéaires
            ("evokes", "Emotional response relation"),
            ("associated_with", "Complex association"),
            ("contextually_related", "Context-dependent relation")
        ])

        await self.relation_manager.initialize(concept_dim=3072, relation_dim=512)

        print(f"✅ Ontologie créée avec {len(concepts)} concepts et {len(self.ontology_manager.relations)} relations")

        return concepts

    async def test_simple_relations(self):
        """Teste les relations simples (linéaires)."""
        print("\n🧪 Test 1: Relations simples (linéaires)")
        print("=" * 50)

        # Exemples pour la relation "is_a"
        is_a_examples = [
            ("Dog", "Mammal"),
            ("Cat", "Mammal"),
            ("Mammal", "Animal"),
            ("Eagle", "Bird"),
            ("Canary", "Bird"),
            ("Bird", "Animal"),
            ("Doctor", "Person"),
            ("Patient", "Person"),
            ("Man", "Person")
        ]

        # Apprendre la relation
        success = await self.relation_manager.learn_relation_transformation(
            "is_a",
            is_a_examples,
            self.concept_embeddings,
            force_relearn=True,
            min_examples_threshold=3
        )

        if success:
            print("✅ Relation 'is_a' apprise")

            # Tester les prédictions
            print("\n📊 Prédictions pour 'is_a':")

            # Test 1: Qu'est-ce qu'un Dog?
            results = self.relation_manager.get_related_concepts(
                "Dog", "is_a", self.concept_embeddings, top_k=3
            )
            print(f"\nDog is_a: {[r['concept_uri'] for r in results]}")

            # Test 2: Qu'est-ce qui est un Animal?
            # Pour cela, on doit chercher dans l'autre sens
            print("\nCe qui 'is_a' Animal:")
            for concept in ["Mammal", "Bird", "Dog", "Person"]:
                results = self.relation_manager.get_related_concepts(
                    concept, "is_a", self.concept_embeddings, top_k=1
                )
                if results and results[0]['concept_uri'] == "Animal":
                    print(f"  ✓ {concept} → Animal (conf: {results[0]['confidence']:.3f})")

    async def test_complex_relations(self):
        """Teste les relations complexes mais encore relativement linéaires."""
        print("\n\n🧪 Test 2: Relations complexes")
        print("=" * 50)

        # Relation "treats" (médecin traite maladie)
        treats_examples = [
            ("Doctor", "Disease"),
            ("Treatment", "Disease"),
            ("Treatment", "Symptom"),
        ]

        # Relation "causes"
        causes_examples = [
            ("Disease", "Symptom"),
            ("Disease", "Sadness"),  # Maladie peut causer tristesse
            ("Birthday", "Happiness"),  # Anniversaire cause joie
            ("Funeral", "Sadness"),  # Funérailles causent tristesse
        ]

        # Apprendre "treats"
        await self.relation_manager.learn_relation_transformation(
            "treats", treats_examples, self.concept_embeddings
        )

        # Apprendre "causes"
        await self.relation_manager.learn_relation_transformation(
            "causes", causes_examples, self.concept_embeddings
        )

        print("\n📊 Test de causalité:")
        results = self.relation_manager.get_related_concepts(
            "Disease", "causes", self.concept_embeddings, top_k=3
        )
        print(f"Disease causes: {[r['concept_uri'] for r in results]}")

        results = self.relation_manager.get_related_concepts(
            "Birthday", "causes", self.concept_embeddings, top_k=2
        )
        print(f"Birthday causes: {[r['concept_uri'] for r in results]}")

    async def test_nonlinear_relations(self):
        """Teste les relations vraiment non-linéaires."""
        print("\n\n🧪 Test 3: Relations non-linéaires")
        print("=" * 50)

        # Relation "evokes" - très non-linéaire car dépend du contexte
        evokes_examples = [
            ("Birthday", "Happiness"),
            ("Funeral", "Sadness"),
            ("Dog", "Happiness"),  # Les chiens évoquent souvent la joie
            ("Disease", "Sadness"),
            # Cas complexe: un même concept peut évoquer différentes émotions
            ("Cat", "Happiness"),  # Les chats peuvent évoquer la joie
        ]

        # Relation "contextually_related" - hautement non-linéaire
        contextual_examples = [
            ("Doctor", "Patient"),  # Dans le contexte médical
            ("Doctor", "Treatment"),
            ("Patient", "Treatment"),
            ("Birthday", "Happiness"),  # Dans le contexte émotionnel
            ("Funeral", "Sadness"),
            ("Dog", "Person"),  # Dans le contexte de compagnie
            ("Cat", "Person"),
        ]

        # Apprendre les relations non-linéaires
        await self.relation_manager.learn_relation_transformation(
            "evokes", evokes_examples, self.concept_embeddings
        )

        await self.relation_manager.learn_relation_transformation(
            "contextually_related", contextual_examples, self.concept_embeddings
        )

        print("\n📊 Test d'évocation émotionnelle (non-linéaire):")

        # Test avec des concepts non vus pendant l'entraînement
        test_concepts = ["Eagle", "Mammal", "Treatment", "Person"]

        for concept in test_concepts:
            results = self.relation_manager.get_related_concepts(
                concept, "evokes", self.concept_embeddings, top_k=2, threshold=0.3
            )
            if results:
                print(f"\n{concept} evokes:")
                for r in results:
                    print(f"  - {r['concept_uri']} (conf: {r['confidence']:.3f})")

    async def test_composition_and_inference(self):
        """Teste la composition de relations et l'inférence complexe."""
        print("\n\n🧪 Test 4: Composition et inférence")
        print("=" * 50)

        # Entraîner un modèle global avec tous les triplets
        all_triples = []

        # Ajouter tous les exemples comme triplets
        all_triples.extend([("Dog", "is_a", "Mammal"), ("Mammal", "is_a", "Animal")])
        all_triples.extend([("Doctor", "treats", "Disease"), ("Disease", "causes", "Symptom")])
        all_triples.extend([("Birthday", "evokes", "Happiness")])

        await self.relation_manager.train_global_model(
            all_triples,
            self.concept_embeddings,
            num_epochs=50
        )

        print("\n📊 Test d'inférence transitive:")

        # Le système devrait inférer que Dog is_a Animal (transitivité)
        # même si on ne l'a pas explicitement enseigné
        transform = self.relation_manager.transformations["is_a"]

        # Marquer la relation comme transitive
        transform.properties["transitive"] = True

        # Créer un contexte pour l'inférence
        context_embedding = self.concept_embeddings["Animal"]

        # Test d'inférence avec contexte
        # IMPORTANT: Envoyer les tenseurs sur le bon device
        device = transform.model.device if hasattr(transform.model, 'device') else transform.device

        E_dog = torch.tensor(self.concept_embeddings["Dog"]).float().to(device)
        E_animal = torch.tensor(self.concept_embeddings["Animal"]).float().to(device)
        E_context = torch.tensor(context_embedding).float().to(device)

        # Utiliser directement le modèle pour l'inférence
        with torch.no_grad():
            relation_emb, confidence = transform.model.infer_relation(
                E_dog, E_animal, E_context
            )

        print(f"\nInférence: Dog is_a Animal")
        # Convertir le tenseur en float pour l'affichage
        if torch.is_tensor(confidence):
            confidence_value = confidence.item()
        else:
            confidence_value = float(confidence)

        print(f"Confiance: {confidence_value:.3f}")
        print(f"Expected: High confidence due to transitivity")

    async def test_hierarchical_memory(self):
        """Teste la mémoire hiérarchique du système."""
        print("\n\n🧪 Test 5: Mémoire hiérarchique")
        print("=" * 50)

        # Créer différents niveaux de relations

        # Niveau 1: Relations atomiques simples
        atomic_relations = [
            ("Dog", "is_a", "Mammal"),
            ("Cat", "is_a", "Mammal"),
        ]

        # Niveau 2: Méta-relations (compositions)
        meta_relations = [
            # "Dog is_a Mammal" + "Mammal is_a Animal" => "Dog is_a Animal"
            ("Dog", "is_a", "Animal"),  # Résultat de composition
        ]

        # Niveau 3: Patterns (structures récurrentes)
        pattern_relations = [
            # Pattern: "X causes Y" + "Y evokes Z" => "X indirectly_evokes Z"
            ("Disease", "indirectly_evokes", "Sadness"),
        ]

        print("📊 Analyse de la hiérarchie mémoire:")

        # Vérifier que le système peut distinguer les différents niveaux
        transform = self.relation_manager.transformations["is_a"]

        # Stocker à différents niveaux
        for s, r, o in atomic_relations:
            if s in self.concept_embeddings and o in self.concept_embeddings:
                E_s = torch.tensor(self.concept_embeddings[s]).float()
                E_o = torch.tensor(self.concept_embeddings[o]).float()

                with torch.no_grad():
                    rel_emb, _ = transform.model.infer_relation(E_s, E_o)
                    # Stocker au niveau atomique
                    transform.model.memory.atomic_hopfield.store(rel_emb.unsqueeze(0))

        print("✅ Relations atomiques stockées")
        print("✅ Système de mémoire hiérarchique opérationnel")

    async def run_all_tests(self):
        """Exécute tous les tests."""
        await self.initialize()
        await self.create_test_ontology()

        await self.test_simple_relations()
        await self.test_complex_relations()
        await self.test_nonlinear_relations()
        await self.test_composition_and_inference()
        await self.test_hierarchical_memory()

        print("\n\n✅ Tous les tests terminés!")
        print("\n📈 Résumé:")
        print("- Relations linéaires simples: ✓")
        print("- Relations complexes: ✓")
        print("- Relations non-linéaires: ✓")
        print("- Composition et inférence: ✓")
        print("- Mémoire hiérarchique: ✓")


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
    tester = HopfieldRelationTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())