"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# benchmark_2kgbench.py
import os
import pickle
import re
import sys
import json
from typing import List, Dict, Optional, Tuple, Set

import requests
import zipfile
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from datetime import datetime
import shutil

from CONSTANT import LLM_MODEL, MAX_CONCEPT_TO_DETECT, CONFIANCE, CHUNK_SIZE, CHUNK_OVERLAP, USE_SEMANTIC_CHUNKING, BLUE, BOLD, \
    RESET, RED, GREEN, YELLOW
from ontology.concept_hopfield import ConceptHopfieldClassifier
from utils.document_processor import DocumentProcessor

# Importer les modules nécessaires
try:
    from utils.rag_engine import RAGEngine
    from utils.wavelet_rag import WaveletRAG
    from ontology.ontology_manager import OntologyManager
    from ontology.classifier import OntologyClassifier
    from provider.get_key import get_openai_key
    from provider.llm_providers import OpenAIProvider
    from ontology.global_ontology_manager import GlobalOntologyManager
except ImportError:
    print("ERREUR: Impossible d'importer les modules nécessaires.")
    print("Assurez-vous que le chemin vers les modules est correctement configuré.")
    sys.exit(1)



# URLs et chemins pour le benchmark
BENCHMARK_REPO = "https://github.com/cenguix/Text2KGBench/archive/refs/heads/main.zip"
BENCHMARK_DIR = "text2kgbench_data"
RESULTS_DIR = "benchmark_results"
API_KEY_PATH = "/home/yopla/Documents/keys/"  # Adapter si nécessaire


class Text2KGBenchmark:
    """Classe pour évaluer votre système GraphRAG avec le benchmark Text2KGBench"""

    def __init__(self, dataset="wikidata_tekgen", ontology_name=None, max_samples=10):
        """
        Initialise le benchmark avec les paramètres spécifiés.

        Args:
            dataset (str): Dataset à utiliser ("wikidata_tekgen" ou "dbpedia_webnlg")
            ontology_name (str): Nom de l'ontologie à tester (ex: "19_film" ou None pour les lister)
            max_samples (int): Nombre maximum d'échantillons à tester
        """
        self.dataset_name = dataset
        self.ontology_name = ontology_name
        self.max_samples = max_samples
        self.benchmark_dir = BENCHMARK_DIR
        self.results_dir = RESULTS_DIR
        self.ontology_path = None  # Chemin du fichier d'ontologie sélectionné

        # Données pour l'apprentissage et les tests
        self.train_data = []  # Données d'entraînement
        self.test_data = []  # Données de test
        self.ground_truth = {}  # Vérité terrain pour l'évaluation (id -> triplets)

        # Métriques d'évaluation
        self.metrics = {
            "entity_precision": [],
            "entity_recall": [],
            "entity_f1": [],
            "taxonomy_conformance": [],
            "triple_precision": [],
            "triple_recall": [],
            "triple_f1": []
        }
        # Ces attributs seront initialisés plus tard
        self.rag_engine = None
        self.ontology_manager = None
        self.classifier = None
        self.wavelet_rag = None

        # Créer les répertoires nécessaires
        os.makedirs(BENCHMARK_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)

    async def initialize_system(self):
        """Initialise tous les composants du système RAG"""
        print(f"{BLUE}{BOLD}Initialisation du système RAG...{RESET}")

        try:
            # Initialiser les fournisseurs LLM
            OPENAI_KEY = get_openai_key(api_key_path=API_KEY_PATH)

            llm_provider = OpenAIProvider(
                model=LLM_MODEL,
                api_key=OPENAI_KEY
            )

            embedding_provider = llm_provider

            # Initialiser le processeur de documents avec semantic chunking
            processor = DocumentProcessor(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                use_semantic_chunking=USE_SEMANTIC_CHUNKING
            )

            # ancienne version - Initialiser le processeur de documents
            # processor = EnhancedDocumentProcessor(chunk_size=1000, chunk_overlap=200)

            # Initialiser le RAG Engine
            self.rag_engine = RAGEngine(
                llm_provider=llm_provider,
                embedding_provider=embedding_provider,
                storage_dir=f"storage_benchmark_{self.ontology_name}"
            )

            # Remplacer le processeur standard
            self.rag_engine.processor = processor

            await self.rag_engine.initialize()

            # Nettoyer les anciens répertoires du classifieur
            classifier_dir = f"classifier_benchmark_{self.ontology_name}"
            if os.path.exists(classifier_dir):
                shutil.rmtree(classifier_dir)
                print(f"{YELLOW}Répertoire {classifier_dir} nettoyé pour nouvelle initialisation{RESET}")

            # Initialiser l'OntologyManager avec un nom spécifique pour cette ontologie
            onto_storage_name = f"ontology_benchmark_{self.ontology_name}"
            self.ontology_manager = OntologyManager(storage_dir=onto_storage_name)

            # Charger l'ontologie sélectionnée
            if self.ontology_path and os.path.exists(self.ontology_path):
                print(f"{BLUE}Chargement de l'ontologie: {self.ontology_path}{RESET}")
                success = self.ontology_manager.load_ontology(self.ontology_path)
                if not success:
                    print(f"{RED}⚠️ Échec du chargement de l'ontologie{RESET}")
                    return False
            else:
                print(f"{RED}Aucun fichier d'ontologie spécifié ou trouvé{RESET}")
                return False

            # Initialiser le classifieur ontologique APRÈS avoir chargé l'ontologie
            self.classifier = OntologyClassifier(
                rag_engine=self.rag_engine,
                ontology_manager=self.ontology_manager,
                storage_dir=classifier_dir,
                use_hierarchical=False, # TODO verifier si tout les cas fonctionnent
                enable_concept_classification=True,
                enable_relation_learning=True  # Activer l'apprentissage des relations
            )

            # Cette étape est cruciale - elle construit et entraîne les réseaux de Hopfield
            # pour reconnaître les concepts de l'ontologie
            await self.classifier.initialize()

            # Initialiser le WaveletRAG
            self.wavelet_rag = WaveletRAG(
                rag_engine=self.rag_engine,
                wavelet="coif3",
                levels=3,
                storage_dir=f"wavelet_benchmark_{self.ontology_name}"
            )

            await self.wavelet_rag.initialize()

            print(f"{GREEN}{BOLD}✓ Système initialisé avec succès pour l'ontologie {self.ontology_name} !{RESET}")
            return True

        except Exception as e:
            print(f"{RED}Erreur lors de l'initialisation : {str(e)}{RESET}")
            import traceback
            traceback.print_exc()
            return False

    async def initialize_global_ontology(self):
        """Initialise une ontologie globale avec tous les domaines du benchmark"""
        print(f"{BLUE}{BOLD}Initialisation de l'ontologie globale...{RESET}")

        # Créer le répertoire pour l'ontologie globale
        global_onto_dir = f"global_ontology_benchmark_{self.dataset_name}"

        # Nettoyer si demandé
        if os.path.exists(global_onto_dir) and input(f"Réinitialiser l'ontologie globale? (o/n): ").lower() == 'o':
            shutil.rmtree(global_onto_dir)
            print(f"{YELLOW}Répertoire {global_onto_dir} nettoyé{RESET}")

        # Initialiser le gestionnaire d'ontologie global
        self.global_ontology_manager = GlobalOntologyManager(storage_dir=global_onto_dir)

        # Récupérer tous les fichiers TTL
        ttl_ontology_dir = os.path.join(
            self.benchmark_dir,
            "Text2KGBench-main",
            "data",
            self.dataset_name,
            "ontologies",
            "owl"
        )

        if not os.path.exists(ttl_ontology_dir):
            print(f"{RED}Répertoire d'ontologies TTL non trouvé: {ttl_ontology_dir}{RESET}")
            return False

        # Préparer la liste des ontologies à intégrer
        ttl_files = []

        for filename in os.listdir(ttl_ontology_dir):
            if filename.endswith('.ttl'):
                # Extraire le nom du domaine
                if filename.startswith("ont_"):
                    domain_name = filename[4:].split('.')[0]
                else:
                    domain_name = filename.split('.')[0]

                # Description basique
                description = f"Domaine {domain_name} importé depuis {filename}"

                ttl_files.append((
                    os.path.join(ttl_ontology_dir, filename),
                    domain_name,
                    description
                ))

        if not ttl_files:
            print(f"{RED}Aucun fichier TTL trouvé dans {ttl_ontology_dir}{RESET}")
            return False

        print(f"{BLUE}Intégration de {len(ttl_files)} ontologies comme domaines...{RESET}")

        # Intégrer toutes les ontologies
        results = await self.global_ontology_manager.integrate_multiple_ontologies(ttl_files)

        # Vérifier les résultats
        success_count = sum(1 for success in results.values() if success)
        print(f"{GREEN}✓ {success_count}/{len(ttl_files)} ontologies intégrées avec succès{RESET}")

        # Vérifier les ontologies intégrées
        domains_info = self.global_ontology_manager.get_imported_domains_info()
        print(f"{BLUE}Domaines disponibles dans l'ontologie globale:{RESET}")

        for domain_name, info in domains_info.items():
            print(f"  - {domain_name}: {info['concepts_count']} concepts, {info['relations_count']} relations")

        # Initialiser le classifieur avec cette ontologie globale
        classifier_dir = f"classifier_global_benchmark_{self.dataset_name}"
        if os.path.exists(classifier_dir):
            shutil.rmtree(classifier_dir)
            print(f"{YELLOW}Répertoire {classifier_dir} nettoyé pour nouvelle initialisation{RESET}")

        # Initialiser le classifieur ontologique avec l'ontologie globale
        self.global_classifier = OntologyClassifier(
            rag_engine=self.rag_engine,
            ontology_manager=self.global_ontology_manager,
            storage_dir=classifier_dir,
            use_hierarchical=True,
            enable_concept_classification=True,
            enable_relation_learning=True
        )

        # séparer les modèles de concepts par domaine:
        print(f"{BLUE}Création des classifieurs de concepts spécifiques aux domaines...{RESET}")
        domain_concept_classifiers = {}

        for domain_name in self.global_ontology_manager.domains:
            # Créer un sous-répertoire pour ce domaine
            domain_concept_dir = os.path.join(classifier_dir, "domain_concepts", domain_name)
            os.makedirs(domain_concept_dir, exist_ok=True)

            # Créer un classifieur de concepts spécifique à ce domaine
            domain_classifier = ConceptHopfieldClassifier(
                rag_engine=self.rag_engine,
                ontology_manager=self.global_ontology_manager,
                storage_dir=domain_concept_dir,
                domain_specific=True,
                current_domain=domain_name
            )

            # Ajouter à notre dictionnaire de classifieurs par domaine
            domain_concept_classifiers[domain_name] = domain_classifier

            # Initialiser ce classifieur de domaine
            await domain_classifier.initialize(auto_build=True, max_concepts_per_level=300, domain_filter=domain_name)

            print(f"  ✓ Classifieur de concepts créé pour le domaine '{domain_name}'")

        print(f"{BLUE}Génération des embeddings pour les domaines avec noms propres...{RESET}")

        for domain_name in self.global_ontology_manager.domains:
            clean_name = self._clean_domain_name(domain_name)
            print(f"  Génération d'embedding pour {domain_name} (nom propre: {clean_name})...")

            # Forcer la création d'un embedding basé sur le nom propre
            domain_description = f"Domain about {clean_name} knowledge"

            # Enrichir avec concepts principaux si disponibles
            domain = self.global_ontology_manager.domains.get(domain_name)
            if domain and hasattr(domain, 'concepts') and domain.concepts:
                main_concepts = [c.label for c in domain.concepts[:5] if hasattr(c, 'label') and c.label]
                if main_concepts:
                    domain_description += f" covering topics like {', '.join(main_concepts)}"

            # Générer l'embedding
            embeddings = await self.rag_engine.embedding_manager.provider.generate_embeddings([domain_description])
            if embeddings:
                domain_embedding = embeddings[0]

                # Si vous avez accès au classifieur hiérarchique
                if hasattr(self.global_classifier, 'classifier') and hasattr(self.global_classifier.classifier,
                                                                             'level_networks'):
                    if 1 in self.global_classifier.classifier.level_networks:
                        network = self.global_classifier.classifier.level_networks[1]
                        # Stocker l'embedding avec le nom de domaine original (pour correspondance technique)
                        network.store_patterns(
                            np.array([domain_embedding]),
                            [domain_name]  # Garder l'ID technique pour l'identification
                        )
                        network.save()
                        print(f"    ✓ Embedding pour '{clean_name}' ajouté au réseau")

        # Stocker les classifieurs dans l'objet benchmark
        self.domain_concept_classifiers = domain_concept_classifiers
        print(f"{GREEN}{BOLD}✓ Ontologie globale initialisée avec {len(domains_info)} domaines !{RESET}")
        return True

    async def _extract_relation_examples_from_ground_truth(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Extrait les exemples de relations à partir des triplets de vérité terrain.
        Retourne des triplets complets pour PyKEEN.

        Returns:
            Dictionnaire relation_uri -> liste de triplets (sujet, relation, objet)
        """
        relation_examples = {}

        # Collecter toutes les entités mentionnées dans les triplets
        all_entities = set()
        all_relations = set()

        for sample_id, triples in self.ground_truth.items():
            for triple in triples:
                if isinstance(triple, dict) and "rel" in triple and "sub" in triple and "obj" in triple:
                    subject = triple["sub"]
                    relation = triple["rel"]
                    obj = triple["obj"]

                    all_entities.add(subject)
                    all_entities.add(obj)
                    all_relations.add(relation)

        # Générer des embeddings pour toutes ces entités (pour compatibilité future)
        entity_embeddings = await self._generate_entity_embeddings(all_entities)

        # Maintenant, extraire les triplets complets
        for sample_id, triples in self.ground_truth.items():
            for triple in triples:
                if isinstance(triple, dict) and "rel" in triple and "sub" in triple and "obj" in triple:
                    relation = triple["rel"]
                    subject = triple["sub"]
                    obj = triple["obj"]

                    # Convertir en URI
                    rel_uri = self._get_relation_uri(relation)

                    if rel_uri not in relation_examples:
                        relation_examples[rel_uri] = []

                    # Stocker le triplet complet (sujet, relation, objet)
                    relation_examples[rel_uri].append((subject, relation, obj))

        # Mettre les embeddings d'entités à disposition du gestionnaire de relations
        if hasattr(self.classifier, 'relation_manager') and self.classifier.relation_manager:
            entity_embeddings_file = os.path.join(
                self.classifier.relation_manager.storage_dir,
                "entity_embeddings.pkl"
            )
            try:
                with open(entity_embeddings_file, 'wb') as f:
                    pickle.dump(entity_embeddings, f)
                print(f"{GREEN}✓ Embeddings d'entités sauvegardés{RESET}")
            except Exception as e:
                print(f"{YELLOW}Erreur lors de la sauvegarde des embeddings d'entités: {str(e)}{RESET}")

        return relation_examples

    def _get_relation_uri(self, relation_name: str) -> Optional[str]:
        """Convertit un nom de relation en URI"""
        # Chercher dans les relations existantes
        for uri, rel in self.ontology_manager.relations.items():
            if rel.label and rel.label.lower() == relation_name.lower():
                return uri

        # Si non trouvé, construire une URI basique
        return f"http://example.org/ontology#{relation_name}"

    def _get_concept_uri(self, concept_name: str) -> Optional[str]:
        """Convertit un nom de concept en URI"""
        # Chercher dans les concepts existants
        for uri, concept in self.ontology_manager.concepts.items():
            if concept.label and concept.label.lower() == concept_name.lower():
                return uri

        # Si non trouvé, construire une URI basique
        return f"http://example.org/ontology#{concept_name}"

    async def train_global_system(self):
        """Entraîne le système global avec les documents de test chargés."""
        print(f"{BLUE}{BOLD}Entraînement du système global avec {len(self.test_data)} documents...{RESET}")

        # Groupe les documents par domaine pour un traitement efficace
        documents_by_domain = {}

        for i, sample in enumerate(tqdm(self.test_data, desc="Classification des documents")):
            sample_id = sample.get("id", f"test_{i}")
            sample_text = sample.get("sent", "")

            if not sample_text:
                continue

            # Déterminer le domaine du document
            domain_id = self._determine_sample_domain(sample)

            # Grouper par domaine
            if domain_id not in documents_by_domain:
                documents_by_domain[domain_id] = []

            documents_by_domain[domain_id].append((sample_id, sample_text))

        # Traiter les documents par domaine
        total_processed = 0
        for domain_id, documents in documents_by_domain.items():
            print(f"  Traitement de {len(documents)} documents pour le domaine '{domain_id}'...")

            for sample_id, sample_text in documents:
                try:
                    # 1. Créer un fichier temporaire
                    temp_file = os.path.join(self.benchmark_dir, f"temp_train_{sample_id}.txt")
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        f.write(sample_text)

                    # 2. Ajouter le document au système RAG
                    doc_id = await self.rag_engine.add_document_with_id(temp_file, sample_id)

                    # 3. Classifier le document pour entraîner les réseaux
                    await self.global_classifier.classify_document(doc_id, force_refresh=True)
                    await self.global_classifier.classify_document_concepts(doc_id, force_refresh=True)

                    # 4. Ajouter ce document au domaine dans l'ontologie
                    self.global_ontology_manager.associate_document_with_domain(doc_id, domain_id)

                    # 5. Nettoyer le fichier temporaire
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

                    total_processed += 1

                except Exception as e:
                    print(f"{YELLOW}Erreur lors du traitement du document {sample_id}: {str(e)}{RESET}")

        print(f"{GREEN}✓ {total_processed}/{len(self.test_data)} documents traités et classifiés{RESET}")
        return total_processed > 0

    def _calculate_metrics(
            self,
            detected_entity_concepts,
            ground_truth_entity_concepts,
            detected_taxonomy_concepts,
            detected_triples,
            ground_truth_triples
    ):
        """Calcule toutes les métriques d'évaluation pour un échantillon."""
        metrics = {}

        # 1. Précision et rappel pour les entités
        true_positives_entities = len(detected_entity_concepts.intersection(ground_truth_entity_concepts))

        entity_precision = true_positives_entities / len(detected_entity_concepts) if detected_entity_concepts else 0.0
        entity_recall = true_positives_entities / len(
            ground_truth_entity_concepts) if ground_truth_entity_concepts else 0.0

        # F1 score pour les entités
        entity_f1 = 2 * (entity_precision * entity_recall) / (entity_precision + entity_recall) if (
                                                                                                               entity_precision + entity_recall) > 0 else 0.0

        # 2. Conformité ontologique
        ontology_concept_labels = set()
        for _, concept in self.global_ontology_manager.concepts.items():
            if hasattr(concept, 'label') and concept.label:
                ontology_concept_labels.add(concept.label)

        taxonomy_conformance = len(detected_taxonomy_concepts.intersection(ontology_concept_labels)) / len(
            detected_taxonomy_concepts) if detected_taxonomy_concepts else 1.0

        # 3. Précision et rappel pour les triplets
        correct_triples = 0
        matched_gt_indices = set()

        for dt in detected_triples:
            for gt_idx, gt in enumerate(ground_truth_triples):
                if gt_idx not in matched_gt_indices and self._are_triples_equivalent(dt, gt):
                    correct_triples += 1
                    matched_gt_indices.add(gt_idx)
                    break

        triple_precision = correct_triples / len(detected_triples) if detected_triples else 0.0
        triple_recall = correct_triples / len(ground_truth_triples) if ground_truth_triples else 0.0

        # F1 score pour les triplets
        triple_f1 = 2 * (triple_precision * triple_recall) / (triple_precision + triple_recall) if (
                                                                                                               triple_precision + triple_recall) > 0 else 0.0

        # Assembler toutes les métriques
        metrics = {
            "entity_precision": entity_precision,
            "entity_recall": entity_recall,
            "entity_f1": entity_f1,
            "taxonomy_conformance": taxonomy_conformance,
            "triple_precision": triple_precision,
            "triple_recall": triple_recall,
            "triple_f1": triple_f1
        }

        return metrics

    def _save_global_results(self, results):
        """Sauvegarde les résultats du benchmark global avec analyse par domaine."""
        # Créer un nom de fichier avec un timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.results_dir, f"global_benchmark_{self.dataset_name}_{timestamp}.json")

        # Sauvegarder les résultats en JSON
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        print(f"{GREEN}✓ Résultats globaux sauvegardés dans {result_file}{RESET}")

        # Générer un rapport HTML spécifique au benchmark global
        html_file = os.path.join(self.results_dir, f"rapport_global_{self.dataset_name}_{timestamp}.html")
        self._generate_global_html_report(results, html_file)

        # Générer des graphiques par domaine
        plot_file = os.path.join(self.results_dir, f"graphiques_global_{self.dataset_name}_{timestamp}.png")
        self._generate_global_plots(results, plot_file)

    def _generate_global_html_report(self, results, output_file):
        """Génère un rapport HTML spécifique pour le benchmark global."""
        html_content = f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Rapport Benchmark Global Text2KGBench</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .sample {{ margin-bottom: 25px; border: 1px solid #ddd; padding: 10px; }}
            .domain {{ color: #e67e22; font-weight: bold; }}
            .correct {{ color: green; }}
            .incorrect {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>Rapport de Benchmark Global Text2KGBench</h1>
        <p><strong>Dataset:</strong> {results['dataset']}</p>
        <p><strong>Date:</strong> {results['timestamp']}</p>
        <p><strong>Échantillons testés:</strong> {results['test_samples']}</p>

        <h2>Résultats Globaux</h2>
        <table>
            <tr>
                <th>Métrique</th>
                <th>Valeur</th>
            </tr>"""

        # Ajouter les métriques globales
        for metric, value in results['metrics']['overall'].items():
            html_content += f"""
            <tr>
                <td>{metric}</td>
                <td>{value:.4f}</td>
            </tr>"""

        html_content += """
        </table>

        <h2>Résultats par Domaine</h2>
        <table>
            <tr>
                <th>Domaine</th>
                <th>Échantillons</th>
                <th>Précision Entités</th>
                <th>Rappel Entités</th>
                <th>F1 Entités</th>
                <th>Précision Triplets</th>
                <th>Rappel Triplets</th>
                <th>F1 Triplets</th>
            </tr>"""

        # Ajouter les résultats par domaine
        for domain, metrics in results['metrics']['by_domain'].items():
            if metrics.get("samples_count", 0) > 0:
                html_content += f"""
            <tr>
                <td>{domain}</td>
                <td>{metrics.get('samples_count', 0)}</td>
                <td>{metrics.get('entity_precision', 0):.4f}</td>
                <td>{metrics.get('entity_recall', 0):.4f}</td>
                <td>{metrics.get('entity_f1', 0):.4f}</td>
                <td>{metrics.get('triple_precision', 0):.4f}</td>
                <td>{metrics.get('triple_recall', 0):.4f}</td>
                <td>{metrics.get('triple_f1', 0):.4f}</td>
            </tr>"""

        html_content += """
        </table>

        <h2>Exemples par Domaine</h2>"""

        # Regrouper les exemples par domaine
        domains_samples = {}
        for sample in results['sample_results'][:50]:  # Limiter à 50 exemples pour la lisibilité
            domain = sample.get('domain', 'unknown')
            if domain not in domains_samples:
                domains_samples[domain] = []
            domains_samples[domain].append(sample)

        # Ajouter des exemples pour chaque domaine
        for domain, samples in domains_samples.items():
            html_content += f"""
        <h3>Domaine: {domain}</h3>"""

            for i, sample in enumerate(samples[:5]):  # Limiter à 5 exemples par domaine
                html_content += f"""
        <div class="sample">
            <h4>Exemple {i + 1}: {sample['id']}</h4>
            <p><strong>Texte:</strong> {sample['text']}</p>

            <h5>Triplets</h5>
            <table>
                <tr>
                    <th>Type</th>
                    <th>Sujet</th>
                    <th>Relation</th>
                    <th>Objet</th>
                    <th>Statut</th>
                </tr>"""

                # Ajouter les triplets de vérité terrain
                for gt_idx, triple in enumerate(sample.get('ground_truth_triples', [])):
                    html_content += f"""
                <tr>
                    <td>Référence</td>
                    <td>{triple.get('sub', '')}</td>
                    <td>{triple.get('rel', '')}</td>
                    <td>{triple.get('obj', '')}</td>
                    <td></td>
                </tr>"""

                # Ajouter les triplets détectés
                for triple in sample.get('detected_triples', []):
                    # Vérifier si ce triplet est correct
                    is_correct = False
                    for gt_triple in sample.get('ground_truth_triples', []):
                        if self._are_triples_equivalent(triple, gt_triple):
                            is_correct = True
                            break

                    css_class = "correct" if is_correct else "incorrect"
                    status = "✓ Correct" if is_correct else "✗ Incorrect"
                    html_content += f"""
                <tr class="{css_class}">
                    <td>Détecté</td>
                    <td>{triple.get('sub', '')}</td>
                    <td>{triple.get('rel', '')}</td>
                    <td>{triple.get('obj', '')}</td>
                    <td>{status}</td>
                </tr>"""

                html_content += """
            </table>
        </div>"""

        html_content += """
    </body>
    </html>"""

        # Écrire le fichier HTML
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"{GREEN}✓ Rapport HTML global généré : {output_file}{RESET}")

    def _generate_global_plots(self, results, output_file):
        """Génère des graphiques spécifiques pour le benchmark global."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            plt.figure(figsize=(20, 15))

            # 1. Graphique des performances globales
            plt.subplot(2, 2, 1)
            metrics = ['entity_f1', 'triple_f1', 'taxonomy_conformance']
            values = [results['metrics']['overall'][m] for m in metrics]

            bars = plt.bar(metrics, values, color=['#3498db', '#e74c3c', '#2ecc71'])

            plt.title('Performances Globales')
            plt.ylim(0, 1)
            plt.xticks(rotation=15)

            # Ajouter les valeurs sur les barres
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{height:.2f}', ha='center', va='bottom')

            # 2. Graphique des performances par domaine
            plt.subplot(2, 2, 2)

            domains = []
            f1_entities = []
            f1_triples = []

            for domain, metrics in results['metrics']['by_domain'].items():
                if metrics.get('samples_count', 0) > 0:
                    domains.append(domain)
                    f1_entities.append(metrics.get('entity_f1', 0))
                    f1_triples.append(metrics.get('triple_f1', 0))

            x = np.arange(len(domains))
            width = 0.35

            plt.bar(x - width / 2, f1_entities, width, label='F1 Entités')
            plt.bar(x + width / 2, f1_triples, width, label='F1 Triplets')

            plt.title('Performances par Domaine')
            plt.ylim(0, 1)
            plt.xticks(x, domains, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()

            # 3. Graphique du nombre d'échantillons par domaine
            plt.subplot(2, 2, 3)

            sample_counts = [metrics.get('samples_count', 0)
                             for domain, metrics in results['metrics']['by_domain'].items()
                             if metrics.get('samples_count', 0) > 0]

            plt.bar(domains, sample_counts, color='#9b59b6')
            plt.title('Nombre d\'Échantillons par Domaine')
            plt.xticks(rotation=45, ha='right')

            # 4. Graphique radar des performances moyennes
            plt.subplot(2, 2, 4, polar=True)

            metrics = ['entity_precision', 'entity_recall', 'entity_f1',
                       'triple_precision', 'triple_recall', 'triple_f1', 'taxonomy_conformance']
            values = [results['metrics']['overall'][m] for m in metrics]

            # Fermer la figure radar
            values.append(values[0])
            metrics.append(metrics[0])

            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=True)

            plt.polar(angles, values, 'o-', linewidth=2)
            plt.fill(angles, values, alpha=0.25)
            plt.xticks(angles[:-1], metrics[:-1], rotation=45)
            plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=7)
            plt.title('Performances Radar')

            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"{GREEN}✓ Graphiques globaux générés : {output_file}{RESET}")

        except Exception as e:
            print(f"{RED}Erreur lors de la génération des graphiques: {str(e)}{RESET}")
            import traceback
            traceback.print_exc()

    def _extract_triples_from_response(self, response_text: str) -> List[Dict[str, str]]:
        """
        Extrait les triplets de la réponse du LLM.

        Args:
            response_text: Réponse brute du LLM

        Returns:
            Liste de dictionnaires représentant les triplets extraits
        """
        detected_triples = []
        lines = response_text.strip().split('\n')

        for line in lines:
            line = line.strip()

            # Ignorer les lignes vides ou non pertinentes
            if not line or line.startswith('#') or "NO_TRIPLES_FOUND" in line.upper():
                continue

            # Format principal: (subject, relation, object)
            pattern = r'\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,\)]+?)\s*\)'
            match = re.search(pattern, line)

            if match:
                # Nettoyer les entités pour éliminer les préfixes indésirables
                subject = re.sub(r'^(subject|entity|s):\s*', '', match.group(1).strip())
                relation = re.sub(r'^(relation|predicate|rel|r|p):\s*', '', match.group(2).strip())
                obj = re.sub(r'^(object|o):\s*', '', match.group(3).strip())

                detected_triples.append({
                    "sub": subject,
                    "rel": relation,
                    "obj": obj
                })
                continue

            # Format alternatif: sujet, relation, objet (sans parenthèses)
            if ',' in line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 3:
                    # Nettoyer les entités
                    subject = re.sub(r'^(subject|entity|s):\s*', '', parts[0])
                    relation = re.sub(r'^(relation|predicate|rel|r|p):\s*', '', parts[1])
                    obj = re.sub(r'^(object|o):\s*', '', parts[2])

                    detected_triples.append({
                        "sub": subject,
                        "rel": relation,
                        "obj": obj
                    })

        return detected_triples

    def _determine_sample_domain(self, sample):
        """Détermine le domaine d'un échantillon basé sur son ID ou son contenu"""
        sample_id = sample.get("id", "")

        # Méthode 1: Extraire directement du format "ont_18_scientist_test_X"
        match = re.search(r'ont_(\d+_\w+)_test', sample_id)
        if match:
            domain_id = match.group(1)
            if domain_id in self.global_ontology_manager.domains:
                return domain_id

        # Méthode 2: Extraire juste le numéro et le nom
        match = re.search(r'ont_(\d+)_(\w+)', sample_id)
        if match:
            domain_num = match.group(1)
            domain_name = match.group(2)
            domain_id = f"{domain_num}_{domain_name}"

            # Vérifier directement
            if domain_id in self.global_ontology_manager.domains:
                return domain_id

            # Chercher par préfixe
            for domain in self.global_ontology_manager.domains:
                if domain.startswith(domain_num + "_"):
                    return domain

        # Sinon, essayer de détecter le domaine en fonction du contenu
        sample_text = sample.get("sent", "").lower()
        domain_keywords = {
            "film": ["film", "movie", "director", "actor", "actress", "starring"],
            "scientist": ["scientist", "discovery", "theory", "laboratory", "research"],
            "airport": ["airport", "airline", "flight", "runway"],
            "university": ["university", "college", "student", "professor", "campus"],
            "building": ["building", "architecture", "construction", "monument"],
            "country": ["country", "nation", "population", "capital"],
            "music": ["music", "song", "album", "band", "musician", "singer"]
        }

        # Compter les correspondances de mots-clés
        domain_matches = {}
        for domain, keywords in domain_keywords.items():
            domain_matches[domain] = sum(1 for keyword in keywords if keyword in sample_text)

        # Trouver le domaine avec le plus de correspondances
        best_domain = max(domain_matches.items(), key=lambda x: x[1])[0]

        # Vérifier si ce domaine existe dans notre ontologie
        for domain_name in self.global_ontology_manager.domains:
            if best_domain in domain_name:
                return domain_name

        # Si aucun domaine ne correspond, retourner le premier disponible
        return next(iter(self.global_ontology_manager.domains.keys()), "unknown")

    async def _prepare_global_triple_extraction_prompt(self, sample_text, domain_name):
        """
        Prépare un prompt pour l'extraction de triplets en se basant sur la détection
        automatique des concepts pertinents.
        """
        # Nettoyer le nom du domaine pour le prompt
        clean_domain = self._clean_domain_name(domain_name)

        # 1. Utiliser auto_concept_search pour trouver les concepts pertinents
        relevant_concepts = []
        relevant_relations = []

        try:
            # Détecter les concepts pertinents pour ce texte
            concept_results = await self.global_classifier.auto_concept_search(
                query=sample_text,
                include_semantic_relations=True,
                max_concepts=5  # Limiter aux 5 concepts les plus pertinents
            )

            # Extraire les concepts détectés et leurs URIs
            if "concepts_detected" in concept_results:
                relevant_concepts = concept_results["concepts_detected"]
                print(f"  Concepts pertinents détectés: {', '.join([c['label'] for c in relevant_concepts])}")
        except Exception as e:
            print(f"⚠️ Erreur lors de la détection des concepts: {e}")

        # 2. Construire le prompt avec les informations ciblées
        prompt = """Given the following ontology and sentence, extract knowledge graph triples that accurately represent the information in the sentence.

    IMPORTANT FORMAT INSTRUCTIONS:
    - Output ONLY the triples without any explanation
    - Use EXACTLY this format for each triple: (subject, relation, object)
    - Each triple must be on a separate line
    - Use ONLY the relations provided in the ontology below and correctly find the good subject and object, order is important
    - If no valid triples can be extracted, output "NO_TRIPLES_FOUND"

    DOMAIN CONTEXT:
    """
        # Ajouter le contexte du domaine
        prompt += f"This sentence is related to the '{clean_domain}' domain.\n\n"

        # 3. Ajouter les concepts pertinents détectés
        if relevant_concepts:
            prompt += "RELEVANT CONCEPTS DETECTED IN TEXT:\n"
            for concept in relevant_concepts:
                label = concept.get("label", "")
                confidence = concept.get("confidence", 0)

                # Ajouter aussi les informations hiérarchiques si disponibles
                hierarchy = concept.get("hierarchy", "")
                if hierarchy:
                    prompt += f"- {label} (confidence: {confidence:.2f}, hierarchy: {hierarchy})\n"
                else:
                    prompt += f"- {label} (confidence: {confidence:.2f})\n"

        # 4. Ajouter les concepts du domaine (fallback si aucun concept pertinent détecté)
        if not relevant_concepts:
            prompt += "ONTOLOGY CONCEPTS FROM THIS DOMAIN:\n"

            # Obtenir les concepts spécifiques à ce domaine
            domain_concepts = []
            domain = self.global_ontology_manager.domains.get(domain_name)
            if domain:
                for concept in domain.concepts:
                    if concept.label:
                        domain_concepts.append(concept.label)

            # Limiter à 30 concepts
            displayed_concepts = domain_concepts[:30]
            for concept_label in displayed_concepts:
                prompt += f"- {concept_label}\n"

            if len(domain_concepts) > 30:
                prompt += f"... and {len(domain_concepts) - 30} more concepts\n"

        # 5. Trouver les relations pertinentes pour les concepts détectés
        if relevant_concepts:
            # Collecter les relations associées aux concepts pertinents
            concept_uris = []
            for concept in relevant_concepts:
                # Extraire l'URI si disponible, sinon chercher par label
                uri = concept.get("concept_uri")
                if uri:
                    concept_uris.append(uri)

            if concept_uris:
                # Trouver les relations qui ont ces concepts comme domaine ou portée
                for uri, relation in self.global_ontology_manager.relations.items():
                    # Vérifier si la relation est liée à l'un des concepts pertinents
                    if domain_name in uri and relation.label:
                        # Vérifier le domaine et la portée
                        relation_domains = [d.uri for d in relation.domain if hasattr(d, 'uri')]
                        relation_ranges = [r.uri for r in relation.range if hasattr(r, 'uri')]

                        if any(uri in concept_uris for uri in relation_domains) or any(
                                uri in concept_uris for uri in relation_ranges):
                            # Relation pertinente
                            domains = [d.label for d in relation.domain if hasattr(d, 'label') and d.label]
                            ranges = [r.label for r in relation.range if hasattr(r, 'label') and r.label]

                            relation_info = f"- {relation.label}"
                            if domains:
                                relation_info += f" (domain: {', '.join(domains[:3])})"
                            if ranges:
                                relation_info += f" (range: {', '.join(ranges[:3])})"

                            relevant_relations.append(relation_info)

        # 6. Ajouter les relations pertinentes ou toutes les relations du domaine comme fallback
        prompt += "\nRELEVANT RELATIONS FOR EXTRACTION:\n"

        if relevant_relations:
            # Ajouter les relations pertinentes (max 20)
            for relation_info in relevant_relations[:20]:
                prompt += f"{relation_info}\n"

            if len(relevant_relations) > 20:
                prompt += f"... and {len(relevant_relations) - 20} more relevant relations\n"
        else:
            # Fallback: ajouter toutes les relations du domaine
            domain_relations = []
            for uri, relation in self.global_ontology_manager.relations.items():
                if domain_name in uri and relation.label:
                    domains = [d.label for d in relation.domain if hasattr(d, 'label') and d.label]
                    ranges = [r.label for r in relation.range if hasattr(r, 'label') and r.label]

                    relation_info = f"- {relation.label}"
                    if domains:
                        relation_info += f" (domain: {', '.join(domains[:3])})"
                    if ranges:
                        relation_info += f" (range: {', '.join(ranges[:3])})"

                    domain_relations.append(relation_info)

            # Limiter à 30 relations
            for relation_info in domain_relations[:30]:
                prompt += f"{relation_info}\n"

            if len(domain_relations) > 30:
                prompt += f"... and {len(domain_relations) - 30} more relations\n"

        # 7. Ajouter un exemple adapté au domaine
        prompt += "\nEXAMPLE OF CORRECT FORMAT:\n"

        # Sélectionner un exemple adapté au domaine
        if "film" in clean_domain.lower():
            prompt += """Sentence: Super Capers is a 98 minute film with a $2000000 budget, starring Tom Sizemore.
    Triples:
    (Super_Capers, runtime, 98.0)
    (Super_Capers, budget, 2000000.0)
    (Super_Capers, starring, Tom_Sizemore)
    """
        elif "scientist" in clean_domain.lower():
            prompt += """Sentence: Marie Curie discovered radium and polonium, working at the University of Paris.
    Triples:
    (Marie_Curie, discovered, radium)
    (Marie_Curie, discovered, polonium)
    (Marie_Curie, employer, University_of_Paris)
    """
        # [autres exemples selon les domaines...]
        else:
            # Générer un exemple plus générique basé sur le domaine détecté
            prompt += f"""Sentence: Example about {clean_domain}.
    Triples:
    (Entity, relation, Object)
    """

        # 8. Ajouter la phrase à analyser
        prompt += f"\nSENTENCE TO ANALYZE:\n{sample_text}\n"
        prompt += "\nTRIPLES:\n"

        return prompt

    def _clean_domain_name(self, domain_name):
        """Nettoie le nom du domaine en supprimant le préfixe numérique."""
        # Supprimer le préfixe numérique (ex: "18_scientist" -> "scientist")
        match = re.match(r'^\d+_(.+)$', domain_name)
        if match:
            return match.group(1)
        return domain_name

    async def run_global_benchmark(self):
        """Exécute le benchmark en utilisant l'ontologie globale et évalue les performances par domaine."""
        print(f"{BLUE}{BOLD}Exécution du benchmark global sur tous les domaines...{RESET}")

        # Diagnostiquer la distribution des échantillons
        domain_distribution = {}
        for sample in self.test_data:
            domain = self._determine_sample_domain(sample)
            if domain not in domain_distribution:
                domain_distribution[domain] = 0
            domain_distribution[domain] += 1

        print(f"\n{BLUE}Distribution des échantillons à tester:{RESET}")
        for domain, count in sorted(domain_distribution.items()):
            print(f"  {domain}: {count} échantillons")
        print()

        results = {
            "dataset": self.dataset_name,
            "ontology": "global",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_samples": len(self.test_data),
            "metrics": {
                "overall": {
                    "entity_precision": 0.0,
                    "entity_recall": 0.0,
                    "entity_f1": 0.0,
                    "taxonomy_conformance": 0.0,
                    "triple_precision": 0.0,
                    "triple_recall": 0.0,
                    "triple_f1": 0.0
                },
                "by_domain": {}
            },
            "sample_results": []
        }

        # Initialiser les métriques par domaine
        domains_info = self.global_ontology_manager.get_imported_domains_info()
        for domain_name in domains_info:
            results["metrics"]["by_domain"][domain_name] = {
                "entity_precision": [],
                "entity_recall": [],
                "entity_f1": [],
                "taxonomy_conformance": [],
                "triple_precision": [],
                "triple_recall": [],
                "triple_f1": [],
                "samples_count": 0
            }

        # Traiter chaque échantillon de test
        for i, sample in enumerate(tqdm(self.test_data, desc="Test global")):
            sample_id = sample.get("id", f"test_{i}")
            sample_text = sample.get("sent", "")

            if not sample_text:
                continue

            # Déterminer le domaine technique et le nom propre
            domain_id = self._determine_sample_domain(sample)
            clean_domain = self._clean_domain_name(domain_id)

            print(f"\nTraitement de l'échantillon {i + 1}/{len(self.test_data)} (ID: {sample_id})")
            print(f"  Domaine: {domain_id} (nom propre: '{clean_domain}')")

            try:
                # 1. Créer un fichier temporaire pour le texte
                temp_file = os.path.join(self.benchmark_dir, f"temp_test_{sample_id}.txt")
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(sample_text)

                # 2. Ajouter le document temporairement pour obtenir son embedding
                doc_id = await self.rag_engine.add_document(temp_file)

                # 3. Obtenir l'embedding du document
                # Méthode auxiliaire pour extraire l'embedding d'un document complet
                async def get_document_embedding(document_id):
                    """Extrait l'embedding d'un document à partir de ses chunks"""
                    await self.rag_engine.document_store.load_document_chunks(document_id)
                    doc_chunks = await self.rag_engine.document_store.get_document_chunks(document_id)

                    if not doc_chunks:
                        return None

                    # Collecter les embeddings des chunks
                    chunk_embeddings = []
                    for chunk in doc_chunks:
                        chunk_id = chunk["id"]
                        embedding = self.rag_engine.embedding_manager.get_embedding(chunk_id)
                        if embedding is not None:
                            if not isinstance(embedding, np.ndarray):
                                embedding = np.array(embedding)
                            chunk_embeddings.append(embedding)

                    if not chunk_embeddings:
                        return None

                    # Calculer la moyenne des embeddings
                    document_embedding = np.mean(chunk_embeddings, axis=0)

                    # Normaliser
                    norm = np.linalg.norm(document_embedding)
                    if norm > 0:
                        document_embedding = document_embedding / norm

                    return document_embedding

                # Obtenir l'embedding du document
                document_embedding = await get_document_embedding(doc_id)

                if document_embedding is None:
                    print(f"{YELLOW}⚠️ Impossible d'obtenir l'embedding pour l'échantillon {sample_id}{RESET}")
                    continue

                # 4. Utiliser le classifieur spécifique au domaine si disponible
                detected_taxonomy_concepts = set()

                if hasattr(self, 'domain_concept_classifiers') and domain_id in self.domain_concept_classifiers:
                    print(f"  Utilisation du classifieur spécifique au domaine '{domain_id}'")
                    domain_classifier = self.domain_concept_classifiers[domain_id]

                    # Classifier le document avec le classifieur de domaine
                    concept_results = await domain_classifier.classify_document(
                        document_embedding,
                        top_k=3,
                        threshold=0.5
                    )

                    # Extraire les concepts détectés
                    for concept_result in concept_results:
                        if concept_result.get("confidence", 0) > 0.5:
                            label = concept_result.get("label")
                            if label:
                                detected_taxonomy_concepts.add(label)

                        # Ajouter aussi les sous-concepts
                        for sub_concept in concept_result.get("sub_concepts", []):
                            if sub_concept.get("confidence", 0) > 0.5:
                                label = sub_concept.get("label")
                                if label:
                                    detected_taxonomy_concepts.add(label)
                else:
                    # Fallback: utiliser le classifieur global ou une requête standard
                    print(
                        f"  Utilisation de auto_concept_search global (pas de classifieur spécifique pour '{domain_id}')")
                    search_result = await self.global_classifier.auto_concept_search(
                        query=sample_text,
                        include_semantic_relations=True
                    )

                    # Extraire les concepts taxonomiques détectés
                    if "concepts_detected" in search_result and search_result["concepts_detected"]:
                        for concept in search_result["concepts_detected"]:
                            if concept.get("confidence", 0) > 0.5:
                                label = concept.get("label")
                                if label:
                                    detected_taxonomy_concepts.add(label)

                # 5. Construire un prompt qui inclut le domaine de l'échantillon
                triple_prompt = await self._prepare_global_triple_extraction_prompt(
                    sample_text,
                    domain_id
                )

                # 6. Appel au LLM pour l'extraction de triplets
                messages = [
                    {"role": "system", "content": "You are a knowledge graph triple extraction assistant."},
                    {"role": "user", "content": triple_prompt}
                ]

                llm_response = await self.rag_engine.llm_provider.generate_response(messages)

                # 7. Extraire les triplets de la réponse
                detected_triples = self._extract_triples_from_response(llm_response)

                # 8. Extraire les concepts d'entités
                detected_entity_concepts = set()
                for triple in detected_triples:
                    if "sub" in triple and triple["sub"]:
                        detected_entity_concepts.add(triple["sub"])
                    if "obj" in triple and triple["obj"]:
                        detected_entity_concepts.add(triple["obj"])

                # 9. Obtenir les triplets de vérité terrain
                ground_truth_triples = self.ground_truth.get(sample_id, [])

                # 10. Extraire les concepts des triplets de vérité terrain
                ground_truth_entity_concepts = set()
                for triple in ground_truth_triples:
                    if isinstance(triple, dict):
                        if "sub" in triple and triple["sub"]:
                            ground_truth_entity_concepts.add(triple["sub"])
                        if "obj" in triple and triple["obj"]:
                            ground_truth_entity_concepts.add(triple["obj"])

                # 11. Calculer les métriques
                metrics = self._calculate_metrics(
                    detected_entity_concepts,
                    ground_truth_entity_concepts,
                    detected_taxonomy_concepts,
                    detected_triples,
                    ground_truth_triples
                )

                # 12. Stocker les résultats pour cet échantillon
                sample_result = {
                    "id": sample_id,
                    "domain": domain_id,
                    "text": sample_text,
                    "ground_truth_entity_concepts": list(ground_truth_entity_concepts),
                    "detected_entity_concepts": list(detected_entity_concepts),
                    "detected_taxonomy_concepts": list(detected_taxonomy_concepts),
                    "ground_truth_triples": ground_truth_triples,
                    "detected_triples": detected_triples,
                    "llm_response": llm_response,
                    "metrics": metrics
                }

                results["sample_results"].append(sample_result)

                # 13. Ajouter les métriques aux statistiques
                for metric, value in metrics.items():
                    # Métriques globales
                    if isinstance(value, (int, float)):
                        if metric not in self.metrics:
                            self.metrics[metric] = []
                        self.metrics[metric].append(value)

                    # Métriques par domaine
                    if domain_id in results["metrics"]["by_domain"]:
                        results["metrics"]["by_domain"][domain_id][metric].append(value)

                # 14. Incrémenter le compteur d'échantillons pour ce domaine
                if domain_id in results["metrics"]["by_domain"]:
                    results["metrics"]["by_domain"][domain_id]["samples_count"] += 1

                # 15. Nettoyer - supprimer le fichier temporaire
                if os.path.exists(temp_file):
                    os.remove(temp_file)

                # 16. Afficher un résumé de cet échantillon
                print(f"  Domaine: {domain_id}")
                print(f"  Entités détectées: {len(detected_entity_concepts)}/{len(ground_truth_entity_concepts)}")
                print(f"  Classes taxonomiques détectées: {len(detected_taxonomy_concepts)}")
                print(f"  Triplets: {len(detected_triples)}/{len(ground_truth_triples)}")
                print(f"  F1 Entités: {metrics['entity_f1']:.4f}, F1 Triplets: {metrics['triple_f1']:.4f}")

            except Exception as e:
                print(f"{RED}Erreur lors du traitement de l'échantillon {sample_id}: {str(e)}{RESET}")
                import traceback
                traceback.print_exc()

        # Calculer les moyennes globales
        for metric, values in self.metrics.items():
            results["metrics"]["overall"][metric] = np.mean(values) if values else 0.0

        # Calculer les moyennes par domaine
        for domain, metrics in results["metrics"]["by_domain"].items():
            samples_count = metrics.pop("samples_count", 0)
            for metric, values in metrics.items():
                if values:
                    results["metrics"]["by_domain"][domain][metric] = np.mean(values)
                else:
                    results["metrics"]["by_domain"][domain][metric] = 0.0
            results["metrics"]["by_domain"][domain]["samples_count"] = samples_count

        # Afficher les résultats
        print(f"\n{GREEN}{BOLD}Résultats du benchmark global:{RESET}")
        for metric, value in results["metrics"]["overall"].items():
            print(f"  {metric}: {value:.4f}")

        # Afficher les résultats par domaine
        print(f"\n{GREEN}Résultats par domaine:{RESET}")
        for domain, metrics in results["metrics"]["by_domain"].items():
            if metrics["samples_count"] > 0:
                print(f"  {domain} ({metrics['samples_count']} échantillons):")
                for metric, value in metrics.items():
                    if metric != "samples_count":
                        print(f"    {metric}: {value:.4f}")

        # Sauvegarder les résultats
        self._save_global_results(results)

        return results

    async def load_all_test_data(self):
        """Charge les données de test pour tous les domaines"""
        print(f"{BLUE}Chargement des données de test pour tous les domaines...{RESET}")

        # Chemin vers le répertoire des données de test
        test_dir = os.path.join(
            self.benchmark_dir,
            "Text2KGBench-main",
            "data",
            self.dataset_name,
            "test"
        )

        # Chemin vers le répertoire des vérités terrain
        ground_truth_dir = os.path.join(
            self.benchmark_dir,
            "Text2KGBench-main",
            "data",
            self.dataset_name,
            "ground_truth"
        )

        if not os.path.exists(test_dir):
            print(f"{RED}Répertoire de test non trouvé: {test_dir}{RESET}")
            return False

        try:
            # Charger toutes les données de test
            test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                          if f.endswith('.json') or f.endswith('.jsonl')]

            for file_path in test_files:
                print(f"{BLUE}Chargement du fichier de test: {os.path.basename(file_path)}{RESET}")

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            # Tenter de charger comme JSON
                            data = json.load(f)

                            if isinstance(data, list):
                                self.test_data.extend(data)
                            elif isinstance(data, dict):
                                self.test_data.append(data)
                        except json.JSONDecodeError:
                            # Ce n'est pas un JSON valide, essayer comme JSONL
                            f.seek(0)
                            for line in f:
                                if line.strip():
                                    try:
                                        item = json.loads(line)
                                        self.test_data.append(item)
                                    except json.JSONDecodeError:
                                        continue
                except Exception as e:
                    print(f"{YELLOW}Erreur lors du chargement de {file_path}: {str(e)}{RESET}")

            # Charger toutes les vérités terrain
            if os.path.exists(ground_truth_dir):
                print(f"{BLUE}Chargement des vérités terrain...{RESET}")

                gt_files = [os.path.join(ground_truth_dir, f) for f in os.listdir(ground_truth_dir)
                            if f.endswith('.json') or f.endswith('.jsonl')]

                for gt_path in gt_files:
                    # Chargement de la vérité terrain
                    try:
                        with open(gt_path, 'r', encoding='utf-8') as f:
                            try:
                                data = json.load(f)

                                if isinstance(data, list):
                                    for item in data:
                                        if "id" in item and "triples" in item:
                                            self.ground_truth[item["id"]] = item["triples"]
                                elif isinstance(data, dict) and "id" in data and "triples" in data:
                                    self.ground_truth[data["id"]] = data["triples"]
                            except json.JSONDecodeError:
                                # Format JSONL
                                f.seek(0)
                                for line in f.readlines():
                                    if line.strip():
                                        try:
                                            item = json.loads(line)
                                            if "id" in item and "triples" in item:
                                                self.ground_truth[item["id"]] = item["triples"]
                                        except json.JSONDecodeError:
                                            continue
                    except Exception as e:
                        print(f"{YELLOW}Erreur lors du chargement de {gt_path}: {str(e)}{RESET}")

            if self.max_samples > 0 and len(self.test_data) > self.max_samples:
                # Regrouper les échantillons par domaine en se basant sur l'ID
                samples_by_domain = {}

                for sample in self.test_data:
                    # Extraire le nom du domaine directement de l'ID
                    sample_id = sample.get("id", "")
                    domain = None

                    # Format typique: ont_18_scientist_test_X
                    match = re.search(r'ont_(\d+_\w+)_test', sample_id)
                    if match:
                        domain = match.group(1)

                    if not domain:
                        # Autre méthode - extraire juste le numéro
                        match = re.search(r'ont_(\d+)_', sample_id)
                        if match:
                            domain_num = match.group(1)
                            # Chercher le domaine complet
                            for d in self.global_ontology_manager.domains:
                                if d.startswith(domain_num + "_"):
                                    domain = d
                                    break

                    # Si toujours pas de domaine, utiliser une méthode plus générale
                    if not domain:
                        domain = self._determine_sample_domain(sample)

                    if domain not in samples_by_domain:
                        samples_by_domain[domain] = []

                    samples_by_domain[domain].append(sample)

                # Afficher la distribution pour diagnostic
                print(f"\n{BLUE}Distribution originale des échantillons par domaine:{RESET}")
                for domain, samples in samples_by_domain.items():
                    print(f"  {domain}: {len(samples)} échantillons")

                # Sélectionner un nombre équitable d'échantillons par domaine
                balanced_samples = []
                domains_with_samples = len([d for d, s in samples_by_domain.items() if s])
                samples_per_domain = max(1, self.max_samples // domains_with_samples)

                print(f"\n{BLUE}Sélection équilibrée: ~{samples_per_domain} échantillons par domaine{RESET}")

                # Premier passage - prendre un nombre égal d'échantillons par domaine
                for domain, samples in samples_by_domain.items():
                    to_take = min(samples_per_domain, len(samples))
                    balanced_samples.extend(samples[:to_take])
                    print(f"  {domain}: {to_take} échantillons sélectionnés")

                # Si on n'a pas atteint max_samples, prendre d'autres échantillons
                if len(balanced_samples) < self.max_samples:
                    remaining = self.max_samples - len(balanced_samples)
                    for domain, samples in samples_by_domain.items():
                        taken = min(samples_per_domain, len(samples))
                        additional = min(remaining, len(samples) - taken)
                        if additional > 0:
                            balanced_samples.extend(samples[taken:taken + additional])
                            remaining -= additional
                        if remaining <= 0:
                            break

                # Remplacer les données de test
                self.test_data = balanced_samples

                print(
                    f"{GREEN}✓ Sélection équilibrée: {len(self.test_data)} échantillons de {domains_with_samples} domaines{RESET}")

            return len(self.test_data) > 0

        except Exception as e:
            print(f"{RED}Erreur lors du chargement des données: {str(e)}{RESET}")
            return False

    async def initialize_system_base(self):
        """Initialise les composants de base du système RAG sans ontologie spécifique"""
        print(f"{BLUE}{BOLD}Initialisation du système RAG de base...{RESET}")

        try:
            # Initialiser les fournisseurs LLM
            OPENAI_KEY = get_openai_key(api_key_path=API_KEY_PATH)

            llm_provider = OpenAIProvider(
                model=LLM_MODEL,
                api_key=OPENAI_KEY
            )

            embedding_provider = llm_provider

            # Initialiser le processeur de documents
            processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)

            # Initialiser le RAG Engine
            self.rag_engine = RAGEngine(
                llm_provider=llm_provider,
                embedding_provider=embedding_provider,
                storage_dir=f"storage_benchmark_global"
            )

            # Remplacer le processeur standard
            self.rag_engine.processor = processor

            await self.rag_engine.initialize()

            print(f"{GREEN}{BOLD}✓ Système RAG de base initialisé !{RESET}")
            return True

        except Exception as e:
            print(f"{RED}Erreur lors de l'initialisation du système: {str(e)}{RESET}")
            import traceback
            traceback.print_exc()
            return False

    def download_benchmark(self):
        """Télécharge et extrait les données du benchmark Text2KGBench"""
        zip_path = os.path.join(self.benchmark_dir, "text2kgbench.zip")

        # Vérifier si les données sont déjà téléchargées
        if os.path.exists(os.path.join(self.benchmark_dir, "Text2KGBench-main")):
            print(f"{YELLOW}Données du benchmark déjà téléchargées.{RESET}")
            return True

        print(f"{BLUE}Téléchargement des données du benchmark Text2KGBench...{RESET}")

        try:
            # Télécharger le fichier zip
            response = requests.get(BENCHMARK_REPO, stream=True)
            response.raise_for_status()

            # Afficher une barre de progression
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192

            with open(zip_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Téléchargement") as pbar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Extraire le contenu
            print(f"{BLUE}Extraction des données...{RESET}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.benchmark_dir)

            print(f"{GREEN}✓ Données du benchmark téléchargées et extraites avec succès !{RESET}")
            return True

        except Exception as e:
            print(f"{RED}Erreur lors du téléchargement des données: {str(e)}{RESET}")
            return False

    def list_available_ontologies(self):
        """Liste toutes les ontologies disponibles dans le dataset et retourne leurs chemins"""
        # Chemins vers les répertoires d'ontologies (JSON et TTL)
        json_ontology_dir = os.path.join(
            self.benchmark_dir,
            "Text2KGBench-main",
            "data",
            self.dataset_name,
            "ontologies"
        )

        # Le répertoire owl est à l'intérieur du répertoire ontologies
        ttl_ontology_dir = os.path.join(
            self.benchmark_dir,
            "Text2KGBench-main",
            "data",
            self.dataset_name,
            "ontologies",
            "owl"
        )

        ontologies = {}

        # Vérifier les fichiers TTL d'abord (format préféré)
        if os.path.exists(ttl_ontology_dir):
            ttl_files = [f for f in os.listdir(ttl_ontology_dir) if f.endswith(('.ttl', '.owl'))]
            for ttl_file in ttl_files:
                # Extraire le nom de l'ontologie
                if ttl_file.startswith("ont_"):
                    name = ttl_file[4:].split('.')[0]
                else:
                    name = ttl_file.split('.')[0]

                ontologies[name] = os.path.join(ttl_ontology_dir, ttl_file)

        # Si aucun fichier TTL n'est trouvé, essayer les fichiers JSON
        if not ontologies and os.path.exists(json_ontology_dir):
            json_files = [f for f in os.listdir(json_ontology_dir) if f.endswith('.json')]
            for json_file in json_files:
                if "_ontology." in json_file:
                    name = json_file.split('_ontology.')[0]
                else:
                    name = json_file.split('.')[0]

                ontologies[name] = os.path.join(json_ontology_dir, json_file)

        return ontologies

    def load_train_data(self):
        """Charge les données d'entraînement pour l'ontologie sélectionnée"""
        print(f"{BLUE}Chargement des données d'entraînement pour l'ontologie {self.ontology_name}...{RESET}")

        # Chemin vers le répertoire des données d'entraînement
        train_dir = os.path.join(
            self.benchmark_dir,
            "Text2KGBench-main",
            "data",
            self.dataset_name,
            "train"
        )

        if not os.path.exists(train_dir):
            print(
                f"{YELLOW}Répertoire d'entraînement non trouvé: {train_dir}. Utilisation des données de test uniquement.{RESET}")
            return True

        try:
            # Rechercher les fichiers d'entraînement liés à cette ontologie
            train_files = []
            for f in os.listdir(train_dir):
                if f.endswith('.json') or f.endswith('.jsonl'):
                    # Vérifier si le fichier correspond à l'ontologie
                    if self.ontology_name in f or f"ont_{self.ontology_name}" in f:
                        train_files.append(os.path.join(train_dir, f))

            if not train_files:
                print(f"{YELLOW}Aucun fichier d'entraînement trouvé pour l'ontologie {self.ontology_name}{RESET}")
                return True  # Continuer même sans données d'entraînement

            # Charger les données depuis chaque fichier
            for file_path in train_files:
                print(f"{BLUE}Chargement du fichier d'entraînement: {os.path.basename(file_path)}{RESET}")

                # Essayer diverses méthodes de chargement
                try:
                    # Essayer d'abord comme JSON standard
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)

                            # Vérifier si c'est une liste ou un objet unique
                            if isinstance(data, list):
                                for item in data:
                                    if self._is_item_for_ontology(item, self.ontology_name):
                                        self.train_data.append(item)
                            elif isinstance(data, dict):
                                if self._is_item_for_ontology(data, self.ontology_name):
                                    self.train_data.append(data)
                        except json.JSONDecodeError:
                            # Ce n'est pas un JSON valide, essayer comme JSONL
                            f.seek(0)  # Revenir au début du fichier
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue

                                try:
                                    item = json.loads(line)
                                    if self._is_item_for_ontology(item, self.ontology_name):
                                        self.train_data.append(item)
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    print(f"{YELLOW}Erreur lors du chargement de {file_path}: {str(e)}{RESET}")

            print(
                f"{GREEN}✓ {len(self.train_data)} échantillons d'entraînement chargés pour l'ontologie {self.ontology_name}{RESET}")
            return True

        except Exception as e:
            print(f"{RED}Erreur lors du chargement des données d'entraînement: {str(e)}{RESET}")
            import traceback
            traceback.print_exc()
            return False

    def load_test_data(self):
        """Charge les données de test et les vérités terrain pour l'ontologie sélectionnée"""
        print(f"{BLUE}Chargement des données de test pour l'ontologie {self.ontology_name}...{RESET}")

        # Chemin vers le répertoire des données de test
        test_dir = os.path.join(
            self.benchmark_dir,
            "Text2KGBench-main",
            "data",
            self.dataset_name,
            "test"
        )

        # Chemin vers le répertoire des vérités terrain
        ground_truth_dir = os.path.join(
            self.benchmark_dir,
            "Text2KGBench-main",
            "data",
            self.dataset_name,
            "ground_truth"
        )

        if not os.path.exists(test_dir):
            print(f"{RED}Répertoire de test non trouvé: {test_dir}{RESET}")
            return False

        try:
            # 1. Charger les données de test
            test_files = []
            for f in os.listdir(test_dir):
                if f.endswith('.json') or f.endswith('.jsonl'):
                    # Vérifier si le fichier correspond à l'ontologie
                    if self.ontology_name in f or f"ont_{self.ontology_name}" in f:
                        test_files.append(os.path.join(test_dir, f))

            if not test_files:
                # Si aucun fichier spécifique n'est trouvé, chercher dans les fichiers génériques
                for f in os.listdir(test_dir):
                    if f.endswith('.json') or f.endswith('.jsonl'):
                        test_files.append(os.path.join(test_dir, f))

            if not test_files:
                print(f"{RED}Aucun fichier de test trouvé{RESET}")
                return False

            # Charger les données de test
            for file_path in test_files:
                print(f"{BLUE}Examen du fichier de test: {os.path.basename(file_path)}{RESET}")

                # Essayer diverses méthodes de chargement
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            # Tenter de charger comme JSON
                            data = json.load(f)

                            # Vérifier si c'est une liste ou un objet unique
                            if isinstance(data, list):
                                for item in data:
                                    if self._is_item_for_ontology(item, self.ontology_name):
                                        self.test_data.append(item)
                            elif isinstance(data, dict):
                                if self._is_item_for_ontology(data, self.ontology_name):
                                    self.test_data.append(data)
                        except json.JSONDecodeError:
                            # Ce n'est pas un JSON valide, essayer comme JSONL
                            f.seek(0)  # Revenir au début du fichier
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue

                                try:
                                    item = json.loads(line)
                                    if self._is_item_for_ontology(item, self.ontology_name):
                                        self.test_data.append(item)
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    print(f"{YELLOW}Erreur lors du chargement de {file_path}: {str(e)}{RESET}")

            # 2. Charger les fichiers de vérité terrain
            if os.path.exists(ground_truth_dir):
                print(f"{BLUE}Chargement des vérités terrain...{RESET}")

                # Chercher le fichier de vérité terrain spécifique à cette ontologie
                gt_files = []
                for f in os.listdir(ground_truth_dir):
                    if self.ontology_name in f and (f.endswith('.json') or f.endswith('.jsonl')):
                        gt_files.append(os.path.join(ground_truth_dir, f))

                if not gt_files:
                    print(
                        f"{YELLOW}Aucun fichier de vérité terrain trouvé pour l'ontologie {self.ontology_name}{RESET}")
                else:
                    # Charger le fichier de vérité terrain
                    for gt_path in gt_files:
                        print(f"{BLUE}Chargement de la vérité terrain: {os.path.basename(gt_path)}{RESET}")
                        try:
                            with open(gt_path, 'r', encoding='utf-8') as f:
                                try:
                                    # Tenter de charger comme JSON
                                    data = json.load(f)

                                    # Vérifier si c'est une liste ou un objet unique
                                    if isinstance(data, list):
                                        for item in data:
                                            if "id" in item and "triples" in item:
                                                self.ground_truth[item["id"]] = item["triples"]
                                    elif isinstance(data, dict):
                                        if "id" in data and "triples" in data:
                                            self.ground_truth[data["id"]] = data["triples"]
                                except json.JSONDecodeError:
                                    # Ce n'est pas un JSON valide, essayer comme JSONL
                                    f.seek(0)  # Revenir au début du fichier
                                    for line in f:
                                        line = line.strip()
                                        if not line:
                                            continue

                                        try:
                                            item = json.loads(line)
                                            if "id" in item and "triples" in item:
                                                self.ground_truth[item["id"]] = item["triples"]
                                        except json.JSONDecodeError:
                                            continue
                        except Exception as e:
                            print(f"{YELLOW}Erreur lors du chargement de la vérité terrain {gt_path}: {str(e)}{RESET}")
            else:
                print(f"{YELLOW}Répertoire de vérité terrain non trouvé: {ground_truth_dir}{RESET}")

            # Limiter le nombre d'échantillons si nécessaire
            if self.max_samples > 0 and len(self.test_data) > self.max_samples:
                self.test_data = self.test_data[:self.max_samples]

            print(
                f"{GREEN}✓ {len(self.test_data)} échantillons de test et {len(self.ground_truth)} vérités terrain chargés{RESET}")
            return len(self.test_data) > 0

        except Exception as e:
            print(f"{RED}Erreur lors du chargement des données de test: {str(e)}{RESET}")
            import traceback
            traceback.print_exc()
            return False

    def _is_item_for_ontology(self, item, ontology_name):
        """Détermine si un item appartient à l'ontologie spécifiée"""
        # Vérifier dans l'ID
        if "id" in item and (ontology_name in item["id"] or f"ont_{ontology_name}" in item["id"]):
            return True

        # Vérifier dans le texte si des concepts de l'ontologie sont mentionnés
        if "sent" in item and self.ontology_manager and hasattr(self.ontology_manager, 'concepts'):
            text = item["sent"].lower()
            for _, concept in self.ontology_manager.concepts.items():
                if concept.label and concept.label.lower() in text:
                    return True

        return False

    async def train_system(self):
        """Entraîne le système avec les données d'entraînement"""
        if not self.train_data:
            print(f"{YELLOW}Aucune donnée d'entraînement disponible. Passage à la phase de test.{RESET}")
            return True

        print(f"{BLUE}{BOLD}Entraînement du système avec {len(self.train_data)} échantillons...{RESET}")

        # Statistiques de chunking
        chunking_stats = {
            'total_chunks': 0,
            'avg_chunk_size': 0,
            'chunk_types': {},
            'documents_with_structure': 0,
            'documents_without_structure': 0
        }

        try:
            # Pour chaque échantillon d'entraînement
            for i, sample in enumerate(tqdm(self.train_data, desc="Entraînement")):
                sample_id = sample.get("id", f"train_{i}")
                sample_text = sample.get("sent", "")

                if not sample_text:
                    continue

                # Créer un fichier temporaire pour ce texte
                temp_file = os.path.join(self.benchmark_dir, f"temp_train_{sample_id}.txt")
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(sample_text)

                try:
                    additional_metadata = {
                        'sample_id': sample_id,
                        'is_training': True,
                        'dataset': self.dataset_name,
                        'ontology': self.ontology_name,
                        'sample_type': 'train'
                    }

                    # Utiliser la nouvelle méthode avec métadonnées
                    doc_id = await self.rag_engine.add_document_with_metadata(
                        temp_file,
                        sample_id,
                        additional_metadata
                    )

                    # Récupérer les chunks pour les statistiques
                    doc_chunks = await self.rag_engine.document_store.get_document_chunks(doc_id)
                    if doc_chunks:
                        chunking_stats['total_chunks'] += len(doc_chunks)

                        # Analyser les types de chunks
                        for chunk in doc_chunks:
                            metadata = chunk.get('metadata', {})
                            chunk_method = metadata.get('chunk_method', 'unknown')

                            if chunk_method == 'semantic':
                                chunking_stats['documents_with_structure'] += 1
                            elif chunk_method == 'fallback':
                                chunking_stats['documents_without_structure'] += 1

                            # Compter les types de sections
                            section_types = metadata.get('section_types', ['unknown'])
                            for st in section_types:
                                chunking_stats['chunk_types'][st] = chunking_stats['chunk_types'].get(st, 0) + 1


                    # Classifier le document pour entraîner les réseaux de Hopfield
                    # Cette étape est cruciale car elle ajuste les réseaux pour reconnaître les concepts
                    await self.classifier.classify_document(doc_id, force_refresh=True)
                    await self.classifier.classify_document_concepts(doc_id, force_refresh=True)

                    # Supprimer le fichier temporaire
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

                except Exception as e:
                    print(f"{YELLOW}Erreur lors du traitement de l'échantillon {sample_id}: {str(e)}{RESET}")

            # Afficher les statistiques de chunking
            if chunking_stats['total_chunks'] > 0:
                print(f"\n{GREEN}Statistiques de chunking:{RESET}")
                print(f"  - Total de chunks créés: {chunking_stats['total_chunks']}")
                print(
                    f"  - Moyenne de chunks par document: {chunking_stats['total_chunks'] / len(self.train_data):.2f}")

                if USE_SEMANTIC_CHUNKING:
                    print(
                        f"  - Documents avec structure détectée: {chunking_stats['documents_with_structure']}")
                    print(
                        f"  - Documents sans structure (fallback): {chunking_stats['documents_without_structure']}")
                    print(f"  - Distribution des types de contenu:")
                    for chunk_type, count in sorted(chunking_stats['chunk_types'].items()):
                        percentage = (count / chunking_stats['total_chunks']) * 100
                        print(f"    - {chunk_type}: {count} ({percentage:.1f}%)")

                    # Entraînement automatique des relations à partir des documents
                    if hasattr(self.classifier, 'relation_manager') and self.classifier.relation_manager:
                        print(f"{BLUE}Entraînement automatique des relations à partir des documents traités...{RESET}")

                        # Demander à l'utilisateur s'il veut utiliser l'entraînement automatique
                        use_auto_training = input(
                            f"Utiliser l'entraînement automatique des relations à partir des documents? (o/n, défaut: o): "
                        ).lower() != 'n'

                        if use_auto_training:
                            auto_results = await self.classifier.train_relations_from_documents(min_examples=3)
                            auto_success = sum(1 for success in auto_results.values() if success)
                            print(
                                f"{GREEN}✓ {auto_success}/{len(auto_results)} relations apprises automatiquement{RESET}")

            print(f"{GREEN}✓ Entraînement terminé avec {len(self.train_data)} échantillons{RESET}")
            return True

        except Exception as e:
            print(f"{RED}Erreur lors de l'entraînement: {str(e)}{RESET}")
            import traceback
            traceback.print_exc()
            return False

    async def prepare_benchmark(self):
        """Télécharge et prépare tous les éléments nécessaires pour le benchmark"""
        # Télécharger les données du benchmark
        if not self.download_benchmark():
            return False

        # Si aucune ontologie n'est spécifiée, lister les ontologies disponibles
        if not self.ontology_name:
            ontologies = self.list_available_ontologies()
            if not ontologies:
                print(f"{RED}Aucune ontologie trouvée dans le dataset {self.dataset_name}{RESET}")
                return False

            print(f"{GREEN}Ontologies disponibles pour le dataset {self.dataset_name}:{RESET}")
            for i, (name, path) in enumerate(ontologies.items(), 1):
                print(f"  {i}. {name} ({os.path.basename(path)})")

            # Demander à l'utilisateur de choisir une ontologie
            choice = input(f"{BLUE}Choisissez une ontologie (numéro): {RESET}")
            try:
                index = int(choice) - 1
                if 0 <= index < len(ontologies):
                    self.ontology_name = list(ontologies.keys())[index]
                    self.ontology_path = ontologies[self.ontology_name]
                else:
                    print(f"{RED}Choix invalide{RESET}")
                    return False
            except ValueError:
                print(f"{RED}Choix invalide{RESET}")
                return False
        else:
            # Trouver le chemin de l'ontologie spécifiée
            ontologies = self.list_available_ontologies()
            if self.ontology_name not in ontologies:
                print(f"{RED}Ontologie {self.ontology_name} non trouvée{RESET}")
                return False

            self.ontology_path = ontologies[self.ontology_name]

        print(f"{GREEN}Ontologie sélectionnée: {self.ontology_name} ({self.ontology_path}){RESET}")

        # Initialiser le système RAG avec l'ontologie sélectionnée
        if not await self.initialize_system():
            return False

        # Charger les données d'entraînement
        if not self.load_train_data():
            return False

        # Charger les données de test
        if not self.load_test_data():
            return False

        print(f"{GREEN}{BOLD}✓ Préparation du benchmark terminée avec succès !{RESET}")
        return True

    async def _prepare_triple_extraction_prompt(self, sample_text, domain_name, detected_concepts=None):
        """
        Prépare un prompt pour l'extraction de triplets en utilisant les relations détectées

        Args:
            sample_text: Texte de l'échantillon de test
            domain_name: Nom du domaine
            detected_concepts: Liste des concepts pertinents détectés (optionnel)
        """
        # Générer l'embedding du texte pour la détection des relations
        text_embedding = None
        if hasattr(self.classifier, 'relation_manager') and self.classifier.relation_manager:
            text_embedding = await self.classifier._get_text_embedding(sample_text)

        # Extraire les relations possibles si possible
        detected_relations = []
        if text_embedding is not None and detected_concepts:
            # Utiliser uniquement les concepts avec confiance suffisante
            valid_concepts = [c for c in detected_concepts if c.get("confidence", 0) > 0.65]

            if valid_concepts:
                # Préparer les données pour l'extraction de relations
                concept_embeddings = {}
                for concept in valid_concepts:
                    concept_uri = concept.get("concept_uri")
                    if concept_uri in self.classifier.concept_classifier.concept_embeddings:
                        concept_embeddings[concept_uri] = self.classifier.concept_classifier.concept_embeddings[
                            concept_uri]

                # Extraire les relations possibles
                if concept_embeddings and hasattr(self.classifier, 'relation_manager'):
                    detected_relations = self.classifier.relation_manager.extract_relation_triples(
                        sample_text, valid_concepts, concept_embeddings
                    )

        # Construire le prompt de base - code existant avec des adaptations
        prompt = """Given the following ontology and sentence, extract knowledge graph triples that accurately represent the information in the sentence.

    IMPORTANT FORMAT INSTRUCTIONS:
    - Output ONLY the triples without any explanation
    - Use EXACTLY this format for each triple: (subject, relation, object)
    - Each triple must be on a separate line
    - Use ONLY the relations provided in the ontology below
    - If no valid triples can be extracted, output "NO_TRIPLES_FOUND"

    ONTOLOGY:
    """

        # Ajouter les concepts de l'ontologie
        prompt += "Concepts:\n"
        for uri, concept in self.ontology_manager.concepts.items():
            if concept.label:
                prompt += f"- {concept.label}\n"

        # Ajouter les relations de l'ontologie avec plus de détails
        prompt += "\nRelations:\n"
        for uri, relation in self.ontology_manager.relations.items():
            if relation.label:
                # Ajouter le domaine et la portée si disponibles
                domains = [d.label for d in relation.domain if hasattr(d, 'label') and d.label]
                ranges = [r.label for r in relation.range if hasattr(r, 'label') and r.label]

                relation_info = f"- {relation.label}"
                if domains:
                    relation_info += f" (domain: {', '.join(domains)})"
                if ranges:
                    relation_info += f" (range: {', '.join(ranges)})"

                prompt += relation_info + "\n"

        # NOUVELLE SECTION: Ajouter les concepts et relations détectés
        if detected_concepts:
            prompt += "\nCONCEPTS DETECTED IN THE SENTENCE:\n"
            for concept in detected_concepts:
                if concept.get("confidence", 0) > 0.65:
                    prompt += f"- {concept.get('label', '')} (confidence: {concept.get('confidence', 0):.2f})\n"

        # Ajouter les relations détectées si disponibles
        if detected_relations:
            prompt += "\nPOSSIBLE RELATIONS DETECTED:\n"
            for rel in detected_relations[:5]:  # Limiter aux 5 relations les plus probables
                prompt += f"- {rel['subject_label']} → {rel['relation_label']} → {rel['object_label']} (confidence: {rel['confidence']:.2f})\n"

        # Ajouter des exemples qui illustrent clairement la direction des relations - code existant
        prompt += """
    EXAMPLES OF CORRECT FORMAT:
    Sentence: Super Capers is a 98 minute film with a $2000000 budget, starring Tom Sizemore.
    Triples:
    (Super_Capers, runtime, 98.0)
    (Super_Capers, budget, 2000000.0)
    (Super_Capers, starring, Tom_Sizemore)

    Sentence: Lady Anne Vane was married to William Wildman, who was a politician that died in 1784.
    Triples:
    (Lady_Anne_Vane, spouse, William_Wildman)
    (William_Wildman, profession, politician)
    (William_Wildman, deathDate, 1784)
    """

        # Ajouter un exemple réel des données d'entraînement si disponible - code existant
        if self.train_data:
            example = next((item for item in self.train_data if "triples" in item), None)
            if example:
                prompt += f"\nEXAMPLE FROM TRAINING DATA:\nSentence: {example.get('sent', '')}\n"
                prompt += "Triples:\n"
                for triple in example.get("triples", []):
                    if isinstance(triple, dict):
                        prompt += f"({triple.get('sub', '')}, {triple.get('rel', '')}, {triple.get('obj', '')})\n"

        # Ajouter la phrase à analyser
        prompt += f"\nSENTENCE TO ANALYZE:\n{sample_text}\n"
        prompt += "\nTRIPLES:\n"

        return prompt
    def good_old_prepare_triple_extraction_prompt(self, sample_text, ontology_name):
        """
        Prépare un prompt pour l'extraction de triplets par le LLM

        Args:
            sample_text: Texte de l'échantillon de test
            ontology_name: Nom de l'ontologie

        Returns:
            Prompt formaté
        """
        # Construire le prompt pour l'extraction de triplets
        prompt = """Given the following ontology and sentence, extract knowledge graph triples that accurately represent the information in the sentence.

    IMPORTANT FORMAT INSTRUCTIONS:
    - Output ONLY the triples without any explanation
    - Use EXACTLY this format for each triple: (subject, relation, object)
    - Each triple must be on a separate line
    - DO NOT include prefixes like "Subject:", "relation:", "object:" or any other labels
    - Use ONLY the relations provided in the ontology
    - If no valid triples can be extracted, output "NO_TRIPLES_FOUND"

    ONTOLOGY:
    """

        # Ajouter les concepts de l'ontologie
        prompt += "Concepts:\n"
        for uri, concept in self.ontology_manager.concepts.items():
            if concept.label:
                prompt += f"- {concept.label}\n"

        # Ajouter les relations de l'ontologie
        prompt += "\nRelations:\n"
        for uri, relation in self.ontology_manager.relations.items():
            if relation.label:
                # Ajouter le domaine et la portée si disponibles
                domains = [d.label for d in relation.domain if hasattr(d, 'label') and d.label]
                ranges = [r.label for r in relation.range if hasattr(r, 'label') and r.label]

                relation_info = f"- {relation.label}"
                if domains:
                    relation_info += f" (domain: {', '.join(domains)})"
                if ranges:
                    relation_info += f" (range: {', '.join(ranges)})"

                prompt += relation_info + "\n"

        # Ajouter des exemples explicites pour illustrer le format attendu
        prompt += """
    EXAMPLES OF CORRECT FORMAT:
    Sentence: Super Capers is a 98 minute film with a $2000000 budget, starring Tom Sizemore.
    Triples:
    (Super_Capers, runtime, 98.0)
    (Super_Capers, budget, 2000000.0)
    (Super_Capers, starring, Tom_Sizemore)

    Sentence: Cecil Parker had a main role in It's Great to Be Young.
    Triples:
    (It's_Great_to_Be_Young, starring, Cecil_Parker)

    Sentence: A film with no relevant information.
    Triples:
    NO_TRIPLES_FOUND
    """

        # Ajouter un exemple réel des données d'entraînement si disponible
        if self.train_data:
            example = next((item for item in self.train_data if "triples" in item), None)
            if example:
                prompt += f"\nEXAMPLE FROM TRAINING DATA:\nSentence: {example.get('sent', '')}\n"
                prompt += "Triples:\n"
                for triple in example.get("triples", []):
                    if isinstance(triple, dict):
                        prompt += f"({triple.get('sub', '')}, {triple.get('rel', '')}, {triple.get('obj', '')})\n"

        # Ajouter la phrase à analyser
        prompt += f"\nSENTENCE TO ANALYZE:\n{sample_text}\n"
        prompt += "\nTRIPLES:\n"

        return prompt

    def _normalize_string(self, text):
        """Normalise une chaîne pour les comparaisons (minuscules, espaces supprimés, etc.)"""
        if not text:
            return ""
        # Convertir en minuscules
        text = text.lower()
        # Supprimer les parenthèses et leur contenu (comme "(1956_film)")
        text = re.sub(r'\([^)]*\)', '', text)
        # Remplacer les tirets bas et traits d'union par des espaces
        text = text.replace('_', ' ').replace('-', ' ')
        # Supprimer les caractères spéciaux sauf les lettres et chiffres
        text = re.sub(r'[^\w\s]', '', text)
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _are_triples_equivalent(self, detected, reference):
        """
        Vérifie si deux triplets sont équivalents, en utilisant une comparaison plus flexible

        Args:
            detected: Triplet détecté
            reference: Triplet de référence

        Returns:
            bool: True si les triplets sont considérés équivalents
        """
        # Normaliser le sujet, le prédicat et l'objet
        detected_sub = self._normalize_string(detected.get("sub", ""))
        detected_rel = detected.get("rel", "").lower()
        detected_obj = self._normalize_string(detected.get("obj", ""))

        ref_sub = self._normalize_string(reference.get("sub", ""))
        ref_rel = reference.get("rel", "").lower()
        ref_obj = self._normalize_string(reference.get("obj", ""))

        # Vérifier la relation (doit être exacte sauf pour la casse)
        if detected_rel != ref_rel:
            return False

        # Pour le sujet et l'objet, utiliser une comparaison plus flexible
        # Considérer comme équivalent si une chaîne est un sous-ensemble de l'autre
        # ou si la similarité de Levenshtein est élevée

        # Vérifier si l'un contient l'autre
        sub_match = (detected_sub in ref_sub) or (ref_sub in detected_sub) or detected_sub == ref_sub
        obj_match = (detected_obj in ref_obj) or (ref_obj in detected_obj) or detected_obj == ref_obj

        # Si les deux correspondent suffisamment, considérer comme équivalent
        return sub_match and obj_match

    async def run_benchmark(self):
        """
        Exécute le benchmark en entraînant d'abord avec les données d'entraînement,
        puis en testant avec les données de test et en comparant aux vérités terrain
        """
        print(f"{BLUE}{BOLD}Exécution du benchmark Text2KGBench pour l'ontologie {self.ontology_name}...{RESET}")

        results = {
            "dataset": self.dataset_name,
            "ontology": self.ontology_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_samples": len(self.train_data),
            "test_samples": len(self.test_data),
            "chunking_method": "semantic" if USE_SEMANTIC_CHUNKING else "standard",  # NOUVEAU
            "chunking_stats": {},  # NOUVEAU
            "metrics": {
                "entity_precision": 0.0,
                "entity_recall": 0.0,
                "entity_f1": 0.0,
                "taxonomy_conformance": 0.0,
                "triple_precision": 0.0,
                "triple_recall": 0.0,
                "triple_f1": 0.0
            },
            "sample_results": []
        }

        # 1. PHASE D'ENTRAÎNEMENT - Ajouter tous les documents d'entraînement
        if not await self.train_system():
            print(f"{RED}Erreur lors de l'entraînement du système{RESET}")
            return results

        # 2. PHASE DE TEST - Évaluer sur les données de test
        print(f"{BLUE}{BOLD}Évaluation sur {len(self.test_data)} échantillons de test...{RESET}")

        search_quality_stats = {
            'chunks_with_headers_used': 0,
            'chunks_with_structure_used': 0,
            'average_chunks_per_query': 0,
            'retrieval_times': []
        }

        # Traiter chaque échantillon de test
        for i, sample in enumerate(tqdm(self.test_data, desc="Test")):
            sample_id = sample.get("id", f"test_{i}")
            sample_text = sample.get("sent", "")

            if not sample_text:
                continue

            try:
                import time
                start_time = time.time()

                # 1. Utiliser auto_concept_search pour obtenir les concepts taxonomiques
                search_result = await self.classifier.auto_concept_search(
                    query=sample_text,
                    include_semantic_relations=True,
                    max_concepts=MAX_CONCEPT_TO_DETECT,
                    use_structure_boost=True
                )

                retrieval_time = time.time() - start_time
                search_quality_stats['retrieval_times'].append(retrieval_time)

                # Analyser les passages récupérés
                if "passages" in search_result:
                    for passage in search_result["passages"]:
                        metadata = passage.get("metadata", {})
                        if metadata.get("has_headers", False):
                            search_quality_stats['chunks_with_headers_used'] += 1
                        if metadata.get("chunk_method") == "semantic":
                            search_quality_stats['chunks_with_structure_used'] += 1


                # Extraire les concepts taxonomiques détectés par auto_concept_search
                detected_taxonomy_concepts = set()
                if "concepts_detected" in search_result and search_result["concepts_detected"]:
                    for concept in search_result["concepts_detected"]:
                        if concept.get("confidence", 0) > CONFIANCE:  # Seuil de confiance
                            label = concept.get("label")
                            if label:
                                detected_taxonomy_concepts.add(label)

                #Extraire et utiliser les relations détectées
                detected_relations = []
                if hasattr(self.classifier, 'relation_manager') and self.classifier.relation_manager:
                    try:
                        # Obtenir l'embedding du texte
                        text_embedding = await self.classifier._get_text_embedding(sample_text)

                        if text_embedding is not None and "concepts_detected" in search_result:
                            # Récupérer les embeddings des concepts détectés
                            concept_embeddings = {}
                            concept_matches = search_result["concepts_detected"]

                            for concept in concept_matches:
                                concept_uri = concept.get("concept_uri")
                                if concept_uri in self.classifier.concept_classifier.concept_embeddings:
                                    concept_embeddings[concept_uri] = \
                                    self.classifier.concept_classifier.concept_embeddings[concept_uri]

                            # Extraire les relations possibles entre ces concepts
                            if concept_embeddings:
                                detected_relations = self.classifier.relation_manager.extract_relation_triples(
                                    sample_text, concept_matches, concept_embeddings
                                )

                                if detected_relations:
                                    print(f"  Relations détectées: {len(detected_relations)}")
                                    # Afficher un exemple de relation détectée
                                    if detected_relations:
                                        rel = detected_relations[0]
                                        print(
                                            f"  Exemple: {rel['subject_label']} → {rel['relation_label']} → {rel['object_label']} ({rel['confidence']:.2f})")

                                # IMPORTANT: Stocker les relations détectées pour l'évaluation
                                sample_result["detected_relations"] = detected_relations
                    except Exception as e:
                        print(f"{YELLOW}Erreur lors de l'extraction des relations: {str(e)}{RESET}")

                # 2. Utiliser le LLM pour extraire les triplets
                # Préparer le prompt pour l'extraction de triplets
                triple_prompt = await self._prepare_triple_extraction_prompt(
                    sample_text,
                    self.ontology_name,
                    search_result.get("concepts_detected", []))

                # Si nous avons des relations détectées, les ajouter explicitement au prompt
                if detected_relations:
                    triple_prompt += "\nRELATIONS SUGGÉRÉES POUR CETTE PHRASE:\n"
                    for rel in detected_relations[:5]:  # Limiter aux 5 meilleures relations
                        triple_prompt += f"({rel['subject_label']}, {rel['relation_label']}, {rel['object_label']}) - Confiance: {rel['confidence']:.2f}\n"

                    triple_prompt += "\nCes relations sont suggérées sur base de l'analyse. Vérifiez si elles sont correctes et pertinentes pour la phrase.\n"


                # Appel au LLM
                messages = [
                    {"role": "system",
                     "content": "You are a knowledge graph triple extraction assistant. Extract triples in the format (subject, relation, object) from the given text, strictly using only the relations provided in the ontology."},
                    {"role": "user", "content": triple_prompt}
                ]

                llm_response = await self.rag_engine.llm_provider.generate_response(messages)

                # Extraire les triplets de la réponse
                detected_triples = []
                lines = llm_response.strip().split('\n')
                for line in lines:
                    line = line.strip()

                    # Ignorer les lignes vides ou non pertinentes
                    if not line or line.startswith('#') or "NO_TRIPLES_FOUND" in line.upper():
                        continue

                    # Format principal: (subject, relation, object)
                    match = re.search(r'\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,\)]+?)\s*\)', line)
                    if match:
                        # Nettoyer les entités pour éliminer les préfixes indésirables
                        subject = re.sub(r'^(subject|entity|s):\s*', '', match.group(1).strip())
                        relation = re.sub(r'^(relation|predicate|rel|r|p):\s*', '', match.group(2).strip())
                        obj = re.sub(r'^(object|o):\s*', '', match.group(3).strip())

                        detected_triples.append({
                            "sub": subject,
                            "rel": relation,
                            "obj": obj
                        })
                        continue

                    # Format alternatif: sujet, relation, objet (sans parenthèses)
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) == 3:
                        # Nettoyer les entités pour éliminer les préfixes indésirables
                        subject = re.sub(r'^(subject|entity|s):\s*', '', parts[0])
                        relation = re.sub(r'^(relation|predicate|rel|r|p):\s*', '', parts[1])
                        obj = re.sub(r'^(object|o):\s*', '', parts[2])

                        detected_triples.append({
                            "sub": subject,
                            "rel": relation,
                            "obj": obj
                        })

                # Extraire les concepts d'entités (instances) des triplets détectés
                detected_entity_concepts = set()
                for triple in detected_triples:
                    if "sub" in triple and triple["sub"]:
                        detected_entity_concepts.add(triple["sub"])
                    if "obj" in triple and triple["obj"]:
                        detected_entity_concepts.add(triple["obj"])

                # Obtenir les triplets de vérité terrain pour cet échantillon
                ground_truth_triples = self.ground_truth.get(sample_id, [])

                # Extraire les concepts (entités) des triplets de vérité terrain
                ground_truth_entity_concepts = set()
                for triple in ground_truth_triples:
                    if isinstance(triple, dict):
                        if "sub" in triple and triple["sub"]:
                            ground_truth_entity_concepts.add(triple["sub"])
                        if "obj" in triple and triple["obj"]:
                            ground_truth_entity_concepts.add(triple["obj"])

                # Calculer les métriques pour les concepts d'entités
                entity_precision = len(detected_entity_concepts.intersection(ground_truth_entity_concepts)) / len(
                    detected_entity_concepts) if detected_entity_concepts else 0.0
                entity_recall = len(detected_entity_concepts.intersection(ground_truth_entity_concepts)) / len(
                    ground_truth_entity_concepts) if ground_truth_entity_concepts else 0.0
                entity_f1 = 2 * (entity_precision * entity_recall) / (entity_precision + entity_recall) if (
                                                                                                                       entity_precision + entity_recall) > 0 else 0.0

                # Calculer la conformité ontologique pour les concepts taxonomiques
                ontology_concept_labels = {concept.label for _, concept in self.ontology_manager.concepts.items() if
                                           concept.label}
                taxonomy_conformance = len(detected_taxonomy_concepts.intersection(ontology_concept_labels)) / len(
                    detected_taxonomy_concepts) if detected_taxonomy_concepts else 1.0

                # Calculer la précision des triplets avec la comparaison flexible
                correct_triples = 0
                matched_gt_indices = set()  # Pour éviter de compter deux fois le même triplet de référence

                for dt in detected_triples:
                    for gt_idx, gt in enumerate(ground_truth_triples):
                        if gt_idx not in matched_gt_indices and self._are_triples_equivalent(dt, gt):
                            correct_triples += 1
                            matched_gt_indices.add(gt_idx)
                            break

                triple_precision = correct_triples / len(detected_triples) if detected_triples else 0.0
                triple_recall = correct_triples / len(ground_truth_triples) if ground_truth_triples else 0.0
                triple_f1 = 2 * (triple_precision * triple_recall) / (triple_precision + triple_recall) if (
                                                                                                                       triple_precision + triple_recall) > 0 else 0.0

                # Stocker les résultats pour cet échantillon
                sample_result = {
                    "id": sample_id,
                    "text": sample_text,
                    "ground_truth_entity_concepts": list(ground_truth_entity_concepts),
                    "detected_entity_concepts": list(detected_entity_concepts),
                    "detected_taxonomy_concepts": list(detected_taxonomy_concepts),
                    "ground_truth_triples": ground_truth_triples,
                    "detected_triples": detected_triples,
                    "llm_response": llm_response,  # Stocker la réponse brute du LLM pour analyse
                    "metrics": {
                        "entity_precision": entity_precision,
                        "entity_recall": entity_recall,
                        "entity_f1": entity_f1,
                        "taxonomy_conformance": taxonomy_conformance,
                        "triple_precision": triple_precision,
                        "triple_recall": triple_recall,
                        "triple_f1": triple_f1
                    }
                }

                results["sample_results"].append(sample_result)

                # Ajouter les métriques pour cet échantillon
                self.metrics["entity_precision"].append(entity_precision)
                self.metrics["entity_recall"].append(entity_recall)
                self.metrics["entity_f1"].append(entity_f1)
                self.metrics["taxonomy_conformance"].append(taxonomy_conformance)
                self.metrics["triple_precision"].append(triple_precision)
                self.metrics["triple_recall"].append(triple_recall)
                self.metrics["triple_f1"].append(triple_f1)

                # Afficher des informations de progression
                if i % 5 == 0 or i == len(self.test_data) - 1:
                    print(f"\n{BLUE}Échantillon {i + 1}/{len(self.test_data)} ({sample_id}):{RESET}")
                    print(f"  Texte: {sample_text[:100]}..." if len(sample_text) > 100 else f"  Texte: {sample_text}")
                    print(f"  Entités détectées: {', '.join(list(detected_entity_concepts)[:5])}" + (
                        "..." if len(detected_entity_concepts) > 5 else ""))
                    print(f"  Classes taxonomiques: {', '.join(list(detected_taxonomy_concepts)[:5])}" + (
                        "..." if len(detected_taxonomy_concepts) > 5 else ""))
                    print(f"  Triplets détectés: {len(detected_triples)}")
                    print(f"  Triplets corrects: {correct_triples}/{len(ground_truth_triples)}")
                    print(
                        f"  Précision des triplets: {triple_precision:.2f}, Rappel: {triple_recall:.2f}, F1: {triple_f1:.2f}")

            except Exception as e:
                print(f"{RED}Erreur lors du traitement de l'échantillon {sample_id}: {str(e)}{RESET}")
                import traceback
                traceback.print_exc()

        # Calculer les statistiques finales de chunking
        if search_quality_stats['retrieval_times']:
            avg_retrieval_time = np.mean(search_quality_stats['retrieval_times'])
            results["chunking_stats"] = {
                "average_retrieval_time": avg_retrieval_time,
                "chunks_with_headers_percentage":
                    (search_quality_stats['chunks_with_headers_used'] /
                     max(1, search_quality_stats['chunks_with_structure_used'])) * 100,
                "semantic_chunks_used": search_quality_stats['chunks_with_structure_used']
            }

        # Calculer les moyennes globales
        for metric, values in self.metrics.items():
            results["metrics"][metric] = np.mean(values) if values else 0.0

        # Afficher les résultats globaux
        print(f"\n{GREEN}{BOLD}Résultats du benchmark pour l'ontologie {self.ontology_name}:{RESET}")
        for metric, value in results["metrics"].items():
            print(f"  {metric}: {value:.4f}")

        # Afficher les statistiques de chunking
        if USE_SEMANTIC_CHUNKING and results.get("chunking_stats"):
            print(f"\n{GREEN}Performance du chunking sémantique:{RESET}")
            print(f"  - Temps moyen de recherche: {results['chunking_stats']['average_retrieval_time']:.3f}s")
            print(f"  - Chunks avec structure utilisés: {results['chunking_stats']['semantic_chunks_used']}")
            print(f"  - % chunks avec headers: {results['chunking_stats']['chunks_with_headers_percentage']:.1f}%")

        # Sauvegarder les résultats
        self._save_results(results)

        return results

    async def _generate_entity_embeddings(self, entities: Set[str]) -> Dict[str, np.ndarray]:
        """
        Génère des embeddings pour des entités spécifiques mentionnées dans les triplets.

        Args:
            entities: Ensemble d'entités à encoder

        Returns:
            Dictionnaire d'embeddings (entité -> embedding)
        """
        print(f"{BLUE}Génération d'embeddings pour {len(entities)} entités...{RESET}")

        # Grouper par lots pour l'efficacité
        batch_size = 50
        entity_list = list(entities)
        entity_embeddings = {}

        for i in range(0, len(entity_list), batch_size):
            batch = entity_list[i:i + batch_size]

            # Créer des descriptions enrichies pour chaque entité
            descriptions = []
            for entity in batch:
                # Rechercher des triplets où cette entité apparaît pour enrichir la description
                related_info = []

                for _, triples in self.ground_truth.items():
                    for triple in triples:
                        if isinstance(triple, dict):
                            if triple.get("sub") == entity:
                                related_info.append(f"{triple.get('rel')} {triple.get('obj')}")
                            elif triple.get("obj") == entity:
                                related_info.append(f"{triple.get('sub')} {triple.get('rel')}")

                # Limiter à 3 informations pour éviter des descriptions trop longues
                related_str = ", ".join(related_info[:3])

                # Construire une description enrichie
                if related_str:
                    description = f"Entity '{entity}' with attributes: {related_str}"
                else:
                    description = f"Entity '{entity}'"

                descriptions.append(description)

            # Générer les embeddings pour ce lot
            try:
                batch_embeddings = await self.rag_engine.embedding_manager.provider.generate_embeddings(descriptions)

                # Associer chaque embedding à son entité
                for j, entity in enumerate(batch):
                    if j < len(batch_embeddings):
                        # Normaliser l'embedding
                        embedding = batch_embeddings[j]
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm

                        entity_embeddings[entity] = embedding

            except Exception as e:
                print(f"{YELLOW}Erreur lors de la génération d'embeddings pour un lot d'entités: {str(e)}{RESET}")

        print(f"{GREEN}✓ {len(entity_embeddings)}/{len(entities)} embeddings d'entités générés{RESET}")
        return entity_embeddings

    def _save_results(self, results):
        """Sauvegarde les résultats du benchmark"""
        # Créer un nom de fichier avec un timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.results_dir,
                                   f"benchmark_{self.dataset_name}_{self.ontology_name}_{timestamp}.json")

        # Sauvegarder les résultats en JSON
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        print(f"{GREEN}✓ Résultats sauvegardés dans {result_file}{RESET}")

        # Générer un rapport HTML
        html_file = os.path.join(self.results_dir, f"rapport_{self.dataset_name}_{self.ontology_name}_{timestamp}.html")
        self._generate_html_report(results, html_file)

        # Générer des graphiques
        plot_file = os.path.join(self.results_dir,
                                 f"graphiques_{self.dataset_name}_{self.ontology_name}_{timestamp}.png")
        self._generate_plots(results, plot_file)

    def _generate_html_report(self, results, output_file):
        """Génère un rapport HTML détaillé avec diagnostics pour les échecs de détection"""
        # En-tête HTML et styles CSS
        html_content = f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Rapport Benchmark Text2KGBench - {results['ontology']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3, h4, h5 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .sample {{ margin-bottom: 35px; border: 1px solid #ddd; padding: 15px; background-color: #fafafa; }}
            .concepts {{ color: #2980b9; }}
            .relations {{ color: #8e44ad; font-weight: bold; }}
            .triples {{ color: #27ae60; }}
            .highlight {{ background-color: #e1f5fe; }}
            .correct {{ color: green; }}
            .incorrect {{ color: red; }}
            .warning {{ color: orange; }}
            .diagnostic {{ background-color: #fff8e1; padding: 10px; border-left: 5px solid #ffc107; margin: 10px 0; }}
            .missed {{ background-color: #ffebee; padding: 5px; }}
            .detected {{ background-color: #e8f5e9; padding: 5px; }}
            .partial {{ background-color: #fff3e0; padding: 5px; }}
            .success-rate {{ font-size: 24px; font-weight: bold; text-align: center; margin: 20px 0; }}
            .metrics-card {{ background-color: #f5f5f5; border-radius: 5px; padding: 10px; margin-bottom: 10px; }}
            .entity-match {{ display: flex; justify-content: space-between; border-bottom: 1px solid #eee; padding: 5px 0; }}
            .suggestion {{ background-color: #e3f2fd; padding: 10px; border-left: 5px solid #2196f3; margin: 10px 0; }}
            .chunking-info {{ 
                background-color: #e8f5e9; 
                padding: 10px; 
                border-radius: 5px; 
                margin: 10px 0; 
            }}
        </style>
    </head>
    <body>
        <h1>Rapport de Benchmark Text2KGBench</h1>
        <p><strong>Dataset:</strong> {results['dataset']}</p>
        <p><strong>Ontologie:</strong> {results['ontology']}</p>
        <p><strong>Date:</strong> {results['timestamp']}</p>
        <p><strong>Méthode de chunking:</strong> {results.get('chunking_method', 'standard')}</p>
        <p><strong>Échantillons d'entraînement:</strong> {results['train_samples']}</p>
        <p><strong>Échantillons de test:</strong> {results['test_samples']}</p>

        
        <div class="metrics-card">
            <h2>Résultats Globaux</h2>
            <table>
                <tr>
                    <th>Métrique</th>
                    <th>Valeur</th>
                    <th>Interprétation</th>
                </tr>
    """

        # Tableau d'interprétation des métriques
        interpretations = {
            "entity_precision": "Proportion des entités détectées qui sont correctes",
            "entity_recall": "Proportion des entités de référence qui ont été détectées",
            "entity_f1": "Équilibre entre précision et rappel des entités",
            "taxonomy_conformance": "Proportion des concepts détectés qui existent dans l'ontologie",
            "triple_precision": "Proportion des triplets détectés qui sont corrects",
            "triple_recall": "Proportion des triplets de référence qui ont été détectés",
            "triple_f1": "Équilibre entre précision et rappel des triplets"
        }

        # Ajouter les métriques globales avec formatage coloré selon la valeur
        for metric, value in results['metrics'].items():
            css_class = ""
            if value < 0.4:
                css_class = "incorrect"
            elif value < 0.7:
                css_class = "warning"
            else:
                css_class = "correct"

            html_content += f"""
            <tr>
                <td>{metric}</td>
                <td class="{css_class}">{value:.4f}</td>
                <td>{interpretations.get(metric, "")}</td>
            </tr>"""

        html_content += """
            </table>
        </div>

        <div class="metrics-card">
            <h2>Statistiques des Relations</h2>
    """

        # Calculer les statistiques des relations détectées
        relation_stats = {}
        total_relations_detected = 0

        for sample in results['sample_results']:
            for rel in sample.get("detected_relations", []):
                rel_label = rel["relation_label"]
                if rel_label not in relation_stats:
                    relation_stats[rel_label] = {"count": 0, "correct": 0}

                relation_stats[rel_label]["count"] += 1
                total_relations_detected += 1

                # Vérifier si la relation est correcte (existe dans un triplet de référence)
                is_correct = False
                for gt_triple in sample.get("ground_truth_triples", []):
                    if isinstance(gt_triple, dict) and gt_triple.get("rel") == rel_label:
                        is_correct = True
                        break

                if is_correct:
                    relation_stats[rel_label]["correct"] += 1

        # Afficher le tableau des relations si des relations ont été détectées
        if relation_stats:
            html_content += f"""
            <p>Total des relations détectées: <strong>{total_relations_detected}</strong></p>
            <table>
                <tr>
                    <th>Relation</th>
                    <th>Occurrences</th>
                    <th>Précision</th>
                    <th>Statut</th>
                </tr>
        """

            # Ajouter chaque relation avec son taux de précision
            for rel, stats in sorted(relation_stats.items(), key=lambda x: x[1]["count"], reverse=True):
                precision = stats["correct"] / stats["count"] if stats["count"] > 0 else 0

                # Déterminer la classe CSS et le statut en fonction de la précision
                css_class = ""
                status = ""
                if precision < 0.4:
                    css_class = "incorrect"
                    status = "Peu fiable"
                elif precision < 0.7:
                    css_class = "warning"
                    status = "Modérément fiable"
                else:
                    css_class = "correct"
                    status = "Fiable"

                html_content += f"""
                <tr>
                    <td>{rel}</td>
                    <td>{stats["count"]}</td>
                    <td class="{css_class}">{precision:.2f}</td>
                    <td>{status}</td>
                </tr>"""

            html_content += """
            </table>
        """
        else:
            html_content += "<p>Aucune relation détectée.</p>"

        html_content += """
        </div>

        <h2>Résultats Détaillés</h2>
    """

        # Fonction auxiliaire pour analyser les entités manquantes/supplémentaires
        def analyze_missing_entities(detected, reference):
            missing = [ent for ent in reference if ent not in detected]
            extra = [ent for ent in detected if ent not in reference]
            common = [ent for ent in detected if ent in reference]
            return missing, extra, common

        # Limiter à 20 échantillons pour la lisibilité
        max_samples = min(20, len(results['sample_results']))
        for i, sample in enumerate(results['sample_results'][:max_samples]):
            sample_id = sample['id']
            text = sample['text']

            # Récupérer les données pour cet échantillon
            ground_truth_entities = set(sample.get('ground_truth_entity_concepts', []))
            detected_entities = set(sample.get('detected_entity_concepts', []))
            taxonomy_concepts = set(sample.get('detected_taxonomy_concepts', []))
            ground_truth_triples = sample.get('ground_truth_triples', [])
            detected_triples = sample.get('detected_triples', [])
            detected_relations = sample.get('detected_relations', [])

            # Analyser les différences entre entités détectées et référence
            missing_entities, extra_entities, common_entities = analyze_missing_entities(
                detected_entities, ground_truth_entities
            )

            # Calculer les métriques spécifiques à cet échantillon
            entity_recall = len(common_entities) / len(ground_truth_entities) if ground_truth_entities else 1.0
            entity_precision = len(common_entities) / len(detected_entities) if detected_entities else 0.0

            # Début de la section pour cet échantillon
            html_content += f"""
            <div class="sample">
                <h3>Échantillon {i + 1}: {sample_id}</h3>
                <p><strong>Texte:</strong> {text}</p>

                <div class="success-rate">
                    <div>Entités: <span class="{'correct' if entity_recall >= 0.7 else 'warning' if entity_recall >= 0.4 else 'incorrect'}">{entity_recall:.2f}</span> (rappel)</div>
                    <div>Triplets: <span class="{'correct' if sample['metrics']['triple_recall'] >= 0.7 else 'warning' if sample['metrics']['triple_recall'] >= 0.4 else 'incorrect'}">{sample['metrics']['triple_recall']:.2f}</span> (rappel)</div>
                </div>

                <h4>Analyse des Entités</h4>
                <div class="entity-mapping">
    """

            # Liste des entités détectées correctement
            if common_entities:
                html_content += "<h5>✓ Entités correctement détectées:</h5><ul>"
                for entity in common_entities:
                    html_content += f"<li class='detected'>{entity}</li>"
                html_content += "</ul>"

            # Liste des entités de référence non détectées avec diagnostic
            if missing_entities:
                html_content += "<h5>❌ Entités non détectées:</h5><ul>"
                for entity in missing_entities:
                    # Vérifier si l'entité apparaît dans les relations détectées
                    found_in_relations = False
                    related_entities = []

                    for rel in detected_relations:
                        if rel['subject_label'] == entity or rel['object_label'] == entity:
                            found_in_relations = True
                            if rel['subject_label'] == entity:
                                related_entities.append(rel['object_label'])
                            else:
                                related_entities.append(rel['subject_label'])

                    # Générer un diagnostic approprié
                    if found_in_relations:
                        diagnostic = f"Détectée comme relation mais pas comme entité indépendante. Reliée à: {', '.join(related_entities)}"
                    else:
                        # Normaliser l'entité et le texte pour une recherche plus robuste
                        entity_cleaned = entity.replace('_', ' ').lower()
                        text_lower = text.lower()
                        entity_in_text = entity_cleaned in text_lower

                        if entity_in_text:
                            diagnostic = "Présente dans le texte mais non reconnue. Possiblement un problème de seuil de confiance ou d'absence dans l'ontologie."
                        else:
                            # Recherche de forme similaire (sans accents, etc.)
                            import unicodedata
                            text_normalized = unicodedata.normalize('NFKD', text_lower)
                            text_normalized = ''.join([c for c in text_normalized if not unicodedata.combining(c)])
                            entity_normalized = unicodedata.normalize('NFKD', entity_cleaned)
                            entity_normalized = ''.join([c for c in entity_normalized if not unicodedata.combining(c)])

                            if entity_normalized in text_normalized:
                                diagnostic = "Présente dans le texte sous une forme différente (accents, casse). Améliorer la normalisation."
                            else:
                                diagnostic = "Non présente explicitement dans le texte. Possiblement une information implicite ou complexe."

                    html_content += f"<li class='missed'>{entity} <span class='diagnostic'>Diagnostic: {diagnostic}</span></li>"
                html_content += "</ul>"

            # Liste des entités détectées en trop
            if extra_entities:
                html_content += "<h5>⚠️ Entités détectées en trop:</h5><ul>"
                for entity in extra_entities:
                    # Chercher si une entité similaire existe dans les références
                    similar_entity = None
                    best_similarity = 0

                    for ref_entity in ground_truth_entities:
                        # Normaliser pour la comparaison
                        entity_norm = entity.lower().replace('_', '')
                        ref_norm = ref_entity.lower().replace('_', '')

                        # Mesure de similarité simple
                        if entity_norm in ref_norm or ref_norm in entity_norm:
                            similarity = min(len(entity_norm), len(ref_norm)) / max(len(entity_norm), len(ref_norm))
                            if similarity > 0.7 and similarity > best_similarity:
                                best_similarity = similarity
                                similar_entity = ref_entity

                    if similar_entity:
                        html_content += f"<li class='partial'>{entity} (pourrait être une variante de {similar_entity})</li>"
                    else:
                        html_content += f"<li class='partial'>{entity}</li>"
                html_content += "</ul>"

            # Section pour les concepts taxonomiques
            html_content += f"""
                </div>

                <h4>Classes Taxonomiques Détectées</h4>
                <div class="taxonomy-concepts">
                    <p class="concepts">{', '.join(taxonomy_concepts)}</p>
    """

            # Diagnostic amélioré pour les concepts taxonomiques
            # Analyser si les concepts taxonomiques couvrent les entités
            missing_taxonomy = False

            if taxonomy_concepts:
                # Calculer la proportion des entités couvertes par les concepts taxonomiques
                entity_words = set()
                for entity in ground_truth_entities:
                    entity_clean = entity.replace('_', ' ').lower()
                    words = [word for word in entity_clean.split() if len(word) > 3]  # Ignorer les mots courts
                    entity_words.update(words)

                # Calculer le chevauchement entre mots d'entités et concepts taxonomiques
                concept_words = set(c.lower() for c in taxonomy_concepts)
                word_overlap = any(word in concept_words for word in entity_words)

                # Ne diagnostiquer un problème que s'il n'y a vraiment aucun chevauchement
                missing_taxonomy = not word_overlap and len(entity_words) > 1
            else:
                missing_taxonomy = len(ground_truth_entities) > 0

            # Afficher le diagnostic taxonomique approprié
            if not taxonomy_concepts and ground_truth_entities:
                html_content += """
                    <div class="diagnostic">
                        <p><strong>Diagnostic:</strong> Aucun concept taxonomique détecté alors que des entités sont attendues.</p>
                        <p>Causes possibles: seuil de confiance trop élevé ou ontologie ne contenant pas les classes adéquates.</p>
                    </div>
                """
            elif missing_taxonomy:
                # Identifier les entités sans correspondance taxonomique
                uncovered_entities = []
                for entity in ground_truth_entities:
                    entity_words = set(word.lower() for word in entity.replace('_', ' ').split() if len(word) > 3)
                    concept_overlap = False
                    for concept in taxonomy_concepts:
                        concept_lower = concept.lower()
                        if any(word in concept_lower for word in entity_words):
                            concept_overlap = True
                            break

                    if not concept_overlap:
                        uncovered_entities.append(entity)

                if uncovered_entities:
                    html_content += f"""
                        <div class="diagnostic">
                            <p><strong>Diagnostic:</strong> Concepts taxonomiques partiels ou insuffisants.</p>
                            <p>Entités sans classification taxonomique adaptée: {', '.join(uncovered_entities[:3])}</p>
                        </div>
                    """

            # Section pour les relations détectées
            html_content += f"""
                </div>

                <h4>Relations Détectées</h4>
    """

            if detected_relations:
                html_content += "<ul>"
                for rel in detected_relations:
                    # Vérifier si la relation correspond à un triplet de référence
                    is_correct = False
                    for gt_triple in ground_truth_triples:
                        if (isinstance(gt_triple, dict) and
                                gt_triple.get("rel") == rel["relation_label"] and
                                (gt_triple.get("sub") == rel["subject_label"] or
                                 gt_triple.get("obj") == rel["object_label"])):
                            is_correct = True
                            break

                    css_class = "correct" if is_correct else "incorrect"
                    html_content += f"""
                    <li class="{css_class}">
                        {rel['subject_label']} → <strong>{rel['relation_label']}</strong> → {rel['object_label']} 
                        (confiance: {rel['confidence']:.2f})
                        {" ✓ Correct" if is_correct else " ❌ Incorrect"}
                    </li>"""
                html_content += "</ul>"
            else:
                # Diagnostic sur l'absence de relations
                if ground_truth_triples:
                    html_content += """
                    <div class="diagnostic">
                        <p><strong>Diagnostic:</strong> Aucune relation détectée malgré la présence de triplets de référence.</p>
                        <p>Causes possibles:</p>
                        <ul>
                            <li>Seuil de confiance trop élevé</li>
                            <li>Relations trop complexes ou implicites dans le texte</li>
                            <li>Manque d'exemples d'apprentissage pour ces relations</li>
                        </ul>
                    </div>
                    """
                else:
                    html_content += "<p>Aucune relation détectée, ce qui est correct car il n'y a pas de triplets de référence.</p>"

            # Tableau comparatif des triplets
            html_content += f"""
                <h4>Triplets</h4>
                <table>
                    <tr>
                        <th>Type</th>
                        <th>Sujet</th>
                        <th>Relation</th>
                        <th>Objet</th>
                        <th>Statut</th>
                        <th>Diagnostic</th>
                    </tr>
            """

            # Ajouter les triplets de vérité terrain
            for gt_idx, triple in enumerate(ground_truth_triples):
                # Vérifier si ce triplet a été détecté
                is_detected = False
                for dt in detected_triples:
                    if self._are_triples_equivalent(dt, triple):
                        is_detected = True
                        break

                status = "✓ Détecté" if is_detected else "❌ Non détecté"
                css_class = "correct" if is_detected else "incorrect"

                # Diagnostic en cas de non-détection
                diagnostic = ""
                if not is_detected:
                    # Vérifier les causes possibles
                    subject_detected = any(triple.get('sub', '') == dt.get('sub', '') for dt in detected_triples)
                    object_detected = any(triple.get('obj', '') == dt.get('obj', '') for dt in detected_triples)
                    relation_detected = any(triple.get('rel', '') == dt.get('rel', '') for dt in detected_triples)

                    if subject_detected and object_detected:
                        diagnostic = "Entités détectées mais relation manquante"
                    elif subject_detected:
                        diagnostic = "Sujet détecté mais objet manquant"
                    elif object_detected:
                        diagnostic = "Objet détecté mais sujet manquant"
                    else:
                        diagnostic = "Aucun élément du triplet détecté"

                    # Vérifier aussi dans les relations
                    rel_found = False
                    for rel in detected_relations:
                        if (rel['subject_label'] == triple.get('sub', '') and
                                rel['object_label'] == triple.get('obj', '') and
                                rel['relation_label'] == triple.get('rel', '')):
                            rel_found = True
                            break

                    if rel_found:
                        diagnostic += ". Relation détectée mais non convertie en triplet."

                html_content += f"""
                        <tr class="{css_class}">
                            <td>Référence</td>
                            <td>{triple.get('sub', '')}</td>
                            <td>{triple.get('rel', '')}</td>
                            <td>{triple.get('obj', '')}</td>
                            <td>{status}</td>
                            <td>{diagnostic}</td>
                        </tr>"""

            # Ajouter les triplets détectés
            for triple in detected_triples:
                # Vérifier si ce triplet est correct
                is_correct = False
                for gt_triple in ground_truth_triples:
                    if self._are_triples_equivalent(triple, gt_triple):
                        is_correct = True
                        break

                css_class = "correct" if is_correct else "incorrect"
                status = "✓ Correct" if is_correct else "✗ Incorrect"

                # Diagnostic pour les triplets incorrects
                diagnostic = ""
                if not is_correct:
                    # Vérifier si des parties du triplet sont correctes
                    subject_correct = any(triple.get('sub', '') == gt.get('sub', '') for gt in ground_truth_triples)
                    object_correct = any(triple.get('obj', '') == gt.get('obj', '') for gt in ground_truth_triples)
                    relation_correct = any(triple.get('rel', '') == gt.get('rel', '') for gt in ground_truth_triples)

                    if subject_correct and relation_correct:
                        diagnostic = "Sujet et relation corrects, mais mauvais objet"
                    elif subject_correct and object_correct:
                        diagnostic = "Entités correctes mais mauvaise relation"
                    elif relation_correct and object_correct:
                        diagnostic = "Relation et objet corrects, mais mauvais sujet"
                    elif subject_correct:
                        diagnostic = "Seul le sujet est correct"
                    elif object_correct:
                        diagnostic = "Seul l'objet est correct"
                    elif relation_correct:
                        diagnostic = "Seule la relation est correcte"
                    else:
                        diagnostic = "Triplet complètement erroné"

                html_content += f"""
                        <tr class="{css_class}">
                            <td>Détecté</td>
                            <td>{triple.get('sub', '')}</td>
                            <td>{triple.get('rel', '')}</td>
                            <td>{triple.get('obj', '')}</td>
                            <td>{status}</td>
                            <td>{diagnostic}</td>
                        </tr>"""

            # Suggestions d'amélioration personnalisées pour cet échantillon
            html_content += """
                </table>

                <h4>Suggestions d'amélioration</h4>
                <div class="suggestion">
            """

            # Générer des suggestions adaptées aux problèmes de cet échantillon
            if missing_entities and not detected_entities:
                html_content += """
                    <p><strong>Problème d'extraction d'entités:</strong> Aucune entité détectée.</p>
                    <ul>
                        <li>Vérifier si les entités référencées existent dans l'ontologie</li>
                        <li>Abaisser le seuil de confiance pour la détection d'entités</li>
                        <li>Ajouter des exemples d'apprentissage pour ces types d'entités</li>
                    </ul>
                """
            elif missing_entities:
                html_content += f"""
                    <p><strong>Détection partielle des entités:</strong> {len(common_entities)}/{len(ground_truth_entities)} entités correctement détectées.</p>
                    <ul>
                        <li>Entités manquantes: {', '.join(missing_entities)}</li>
                        <li>Vérifier si ces entités sont présentes dans l'ontologie</li>
                        <li>Améliorer la normalisation des entités pour gérer les variations orthographiques</li>
                    </ul>
                """

            if ground_truth_triples and not detected_triples:
                html_content += """
                    <p><strong>Problème d'extraction de triplets:</strong> Aucun triplet détecté.</p>
                    <ul>
                        <li>Améliorer le prompt d'extraction</li>
                        <li>Ajouter des exemples spécifiques à ce type de relation</li>
                        <li>Vérifier si les relations utilisées sont bien définies dans l'ontologie</li>
                        <li>Augmenter le paramètre beta du réseau de Hopfield (actuellement 20.0)</li>
                    </ul>
                """

            html_content += """
                </div>
            </div>
    """

        # Section finale avec recommandations générales
        html_content += """
        <h2>Interprétation des Résultats</h2>
        <ul>
            <li><strong>Précision des entités:</strong> Proportion de concepts détectés qui sont corrects.</li>
            <li><strong>Rappel des entités:</strong> Proportion de concepts attendus qui ont été détectés.</li>
            <li><strong>F1 des entités:</strong> Moyenne harmonique de la précision et du rappel pour les entités.</li>
            <li><strong>Conformité Ontologique:</strong> Proportion de concepts détectés qui existent dans l'ontologie.</li>
            <li><strong>Précision des triplets:</strong> Proportion de triplets détectés qui sont corrects.</li>
            <li><strong>Rappel des triplets:</strong> Proportion de triplets attendus qui ont été détectés.</li>
            <li><strong>F1 des triplets:</strong> Moyenne harmonique de la précision et du rappel pour les triplets.</li>
        </ul>

        <h2>Recommandations Générales</h2>
        <div class="suggestion">
            <p><strong>Pour améliorer la détection d'entités:</strong></p>
            <ul>
                <li>Augmenter MAX_CONCEPT_TO_DETECT (actuellement {MAX_CONCEPT_TO_DETECT})</li>
                <li>Réduire CONFIANCE (actuellement {CONFIANCE})</li>
                <li>Améliorer la normalisation des entités pour gérer les variations orthographiques</li>
            </ul>

            <p><strong>Pour améliorer l'extraction de relations:</strong></p>
            <ul>
                <li>Augmenter la valeur de beta dans le réseau de Hopfield</li>
                <li>Ajouter plus d'exemples d'apprentissage pour chaque relation</li>
                <li>Réduire le seuil de confiance pour la détection de relations (0.45 recommandé)</li>
            </ul>

            <p><strong>Pour améliorer l'extraction de triplets:</strong></p>
            <ul>
                <li>Enrichir les prompts avec les relations détectées</li>
                <li>Utiliser la normalisation des caractères spéciaux dans les comparaisons</li>
                <li>Augmenter le nombre d'exemples dans le prompt</li>
            </ul>
        </div>
    </body>
    </html>
    """

        # Écrire le fichier HTML
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"{GREEN}✓ Rapport HTML détaillé généré : {output_file}{RESET}")

    def _generate_plots(self, results, output_file):
        """Génère des graphiques à partir des résultats"""
        try:
            # Extraire les données pour les graphiques
            sample_ids = [sample["id"] for sample in results["sample_results"]]
            metrics = ["precision", "recall", "f1", "ontology_conformance", "triple_accuracy"]

            # Limiter le nombre d'échantillons à afficher si trop nombreux
            max_display = 10
            if len(sample_ids) > max_display:
                step = len(sample_ids) // max_display
                sample_indices = list(range(0, len(sample_ids), step))
                if sample_indices[-1] != len(sample_ids) - 1:
                    sample_indices.append(len(sample_ids) - 1)

                sample_ids = [sample_ids[i] for i in sample_indices]
                sample_results = [results["sample_results"][i] for i in sample_indices]
            else:
                sample_results = results["sample_results"]

            # Créer une figure avec plusieurs sous-graphiques
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Résultats du Benchmark Text2KGBench - {results["ontology"]}', fontsize=16)

            # Graphique des métriques par échantillon
            ax1 = axes[0, 0]

            # Préparer les données pour chaque métrique
            for metric in metrics:
                values = [sample["metrics"][metric] for sample in sample_results]
                ax1.plot(range(len(values)), values, label=metric, marker='o')

            ax1.set_xlabel('Échantillons')
            ax1.set_ylabel('Score')
            ax1.set_title('Métriques par échantillon')
            ax1.set_xticks(range(len(sample_ids)))
            ax1.set_xticklabels([s[-4:] for s in sample_ids], rotation=45)  # Afficher seulement la fin de l'ID
            ax1.legend()
            ax1.set_ylim(0, 1)
            ax1.grid(True)

            # Graphique de la distribution des métriques
            ax2 = axes[0, 1]

            for i, metric in enumerate(metrics[:3]):  # Uniquement precision, recall, f1
                all_values = [sample["metrics"][metric] for sample in results["sample_results"]]
                ax2.hist(all_values, bins=10, alpha=0.7, label=metric)

            ax2.set_xlabel('Valeur')
            ax2.set_ylabel('Fréquence')
            ax2.set_title('Distribution des métriques')
            ax2.legend()
            ax2.grid(True)

            # Graphique du nombre de concepts et triplets
            ax3 = axes[1, 0]

            concept_data = []
            triple_data = []

            for sample in sample_results:
                gt_concepts = len(sample["ground_truth_concepts"])
                det_concepts = len(sample["detected_concepts"])
                concept_data.append((gt_concepts, det_concepts))

                gt_triples = len(sample.get("ground_truth_triples", []))
                det_triples = len(sample.get("detected_triples", []))
                triple_data.append((gt_triples, det_triples))

            # Tracer les barres pour les concepts
            x = np.arange(len(sample_results))
            width = 0.2

            ax3.bar(x - width * 1.5, [c[0] for c in concept_data], width, label='Concepts attendus')
            ax3.bar(x - width / 2, [c[1] for c in concept_data], width, label='Concepts détectés')
            ax3.bar(x + width / 2, [t[0] for t in triple_data], width, label='Triplets attendus')
            ax3.bar(x + width * 1.5, [t[1] for t in triple_data], width, label='Triplets détectés')

            ax3.set_xlabel('Échantillons')
            ax3.set_ylabel('Nombre')
            ax3.set_title('Concepts et Triplets par échantillon')
            ax3.set_xticks(x)
            ax3.set_xticklabels([s[-4:] for s in sample_ids], rotation=45)
            ax3.legend()
            ax3.grid(True)

            # Graphique des moyennes globales
            ax4 = axes[1, 1]

            metric_values = [results["metrics"][metric] for metric in metrics]
            bars = ax4.bar(metrics, metric_values, color='purple')

            # Ajouter les valeurs au-dessus des barres
            for bar in bars:
                height = bar.get_height()
                ax4.annotate(f'{height:.2f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom')

            ax4.set_xlabel('Métriques')
            ax4.set_ylabel('Score moyen')
            ax4.set_title('Moyennes globales')
            ax4.set_ylim(0, 1)
            ax4.grid(True)

            # Ajuster la mise en page
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            # Sauvegarder la figure
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"{GREEN}✓ Graphiques générés : {output_file}{RESET}")

        except Exception as e:
            print(f"{RED}Erreur lors de la génération des graphiques: {str(e)}{RESET}")
            import traceback
            traceback.print_exc()


async def main():
    """Fonction principale pour exécuter le benchmark"""
    print(f"{BLUE}{BOLD}Benchmark Text2KGBench pour évaluation de GraphRAG{RESET}")

    # Demander à l'utilisateur de choisir le dataset
    print(f"\n{YELLOW}Choix du dataset:{RESET}")
    print("1. wikidata_tekgen (13,474 phrases, 10 ontologies)")
    print("2. dbpedia_webnlg (4,860 phrases, 19 ontologies)")
    dataset_choice = input(f"Votre choix (1/2, défaut: 2): ")
    dataset = "wikidata_tekgen" if dataset_choice == "1" else "dbpedia_webnlg"

    # Demander le nombre maximum d'échantillons à tester
    max_samples_input = input(f"Nombre maximum d'échantillons à tester (défaut: 10): ")
    max_samples = int(max_samples_input) if max_samples_input.isdigit() else 10

    # Demander le mode d'ontologie
    print(f"\n{YELLOW}Mode d'utilisation des ontologies:{RESET}")
    print("1. Mode standard (une ontologie à la fois)")
    print("2. Mode ontologie globale (fusion de toutes les ontologies)")
    ontology_mode = input(f"Votre choix (1/2, défaut: 1): ")
    use_global_ontology = ontology_mode == "2"

    # Initialiser le benchmark sans spécifier d'ontologie (pour la liste)
    benchmark = Text2KGBenchmark(dataset=dataset, ontology_name=None, max_samples=max_samples)

    # Télécharger les données du benchmark si nécessaire
    if not benchmark.download_benchmark():
        print(f"{RED}Échec du téléchargement du benchmark. Arrêt.{RESET}")
        return

    if use_global_ontology:
        # Initialiser le système avec l'ontologie globale
        if not await benchmark.initialize_system_base():
            print(f"{RED}Échec de l'initialisation du système RAG. Arrêt.{RESET}")
            return

        # Initialiser l'ontologie globale
        if not await benchmark.initialize_global_ontology():
            print(f"{RED}Échec de l'initialisation de l'ontologie globale. Arrêt.{RESET}")
            return

        # Charger les données de test sur tous les domaines
        if not await benchmark.load_all_test_data():
            print(f"{RED}Échec du chargement des données de test. Arrêt.{RESET}")
            return

        # AJOUTER CETTE LIGNE: Entraîner le système global avec les documents chargés
        if not await benchmark.train_global_system():
            print(f"{RED}Échec de l'entraînement du système global. Arrêt.{RESET}")
            return

        # Exécuter le benchmark global
        results = await benchmark.run_global_benchmark()
    else:
        # Lister les ontologies disponibles et demander un choix
        ontologies = benchmark.list_available_ontologies()
        if not ontologies:
            print(f"{RED}Aucune ontologie trouvée dans le dataset {dataset}{RESET}")
            return

        print(f"\n{GREEN}Ontologies disponibles pour le dataset {dataset}:{RESET}")
        for i, (name, path) in enumerate(ontologies.items(), 1):
            print(f"  {i}. {name} ({os.path.basename(path)})")

        # Demander à l'utilisateur de choisir une ontologie
        choice = input(f"\n{BLUE}Choisissez une ontologie (numéro): {RESET}")
        try:
            index = int(choice) - 1
            if 0 <= index < len(ontologies):
                ontology_name = list(ontologies.keys())[index]
            else:
                print(f"{RED}Choix invalide{RESET}")
                return
        except ValueError:
            print(f"{RED}Choix invalide{RESET}")
            return

        # Créer un nouveau benchmark avec l'ontologie sélectionnée
        benchmark = Text2KGBenchmark(dataset=dataset, ontology_name=ontology_name, max_samples=max_samples)
        benchmark.ontology_path = ontologies[ontology_name]

        # Nettoyer les répertoires précédents si nécessaire
        cleanup = input(f"Nettoyer les répertoires précédents? (o/n, défaut: n): ").lower() == 'o'
        if cleanup:
            dirs_to_clean = [f"storage_benchmark_{ontology_name}", f"ontology_benchmark_{ontology_name}",
                             f"classifier_benchmark_{ontology_name}", f"wavelet_benchmark_{ontology_name}"]
            for dir_path in dirs_to_clean:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    print(f"{YELLOW}Répertoire {dir_path} nettoyé{RESET}")

        # Préparer le benchmark
        if not await benchmark.prepare_benchmark():
            print(f"{RED}Échec de la préparation du benchmark. Arrêt.{RESET}")
            return

        # Exécuter le benchmark
        results = await benchmark.run_benchmark()

        print(f"\n{GREEN}{BOLD}✓ Benchmark terminé !{RESET}")
        print(f"Consultez les rapports générés dans le dossier: {RESULTS_DIR}")


async def compare_chunking_methods():
    """Compare les performances entre chunking standard et sémantique"""

    print(f"{BLUE}{BOLD}Comparaison des méthodes de chunking{RESET}")

    # Test avec chunking standard
    global USE_SEMANTIC_CHUNKING
    USE_SEMANTIC_CHUNKING = False

    benchmark_standard = Text2KGBenchmark(
        dataset="dbpedia_webnlg",
        ontology_name="film",
        max_samples=20
    )

    await benchmark_standard.prepare_benchmark()
    results_standard = await benchmark_standard.run_benchmark()

    # Test avec chunking sémantique
    USE_SEMANTIC_CHUNKING = True

    benchmark_semantic = Text2KGBenchmark(
        dataset="dbpedia_webnlg",
        ontology_name="film",
        max_samples=20
    )

    await benchmark_semantic.prepare_benchmark()
    results_semantic = await benchmark_semantic.run_benchmark()

    # Comparer les résultats
    print(f"\n{GREEN}{BOLD}Résultats de la comparaison:{RESET}")
    print(f"\nChunking Standard:")
    for metric, value in results_standard["metrics"].items():
        print(f"  {metric}: {value:.4f}")

    print(f"\nChunking Sémantique:")
    for metric, value in results_semantic["metrics"].items():
        print(f"  {metric}: {value:.4f}")

    # Calculer les améliorations
    print(f"\n{BLUE}Améliorations avec chunking sémantique:{RESET}")
    for metric in results_standard["metrics"]:
        improvement = ((results_semantic["metrics"][metric] -
                        results_standard["metrics"][metric]) /
                       max(0.001, results_standard["metrics"][metric])) * 100

        color = GREEN if improvement > 0 else RED
        print(f"  {metric}: {color}{improvement:+.1f}%{RESET}")


if __name__ == "__main__":
    try:
        # Ajouter une option pour la comparaison
        if len(sys.argv) > 1 and sys.argv[1] == "--compare":
            asyncio.run(compare_chunking_methods())
        else:
            asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Benchmark interrompu par l'utilisateur.{RESET}")
    except Exception as e:
        print(f"\n{RED}Erreur: {str(e)}{RESET}")
        import traceback

        traceback.print_exc()