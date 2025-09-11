"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# ontology/pykeenrelation_manager.py
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from scipy.spatial.distance import cosine
import torch
from datetime import datetime
import hashlib

# PyKEEN imports
from pykeen import predict
from pykeen.datasets import Dataset
from pykeen.models import TransE, TransR, ComplEx, RotatE, DistMult
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
from pykeen.optimizers import Adam
from pykeen.losses import MarginRankingLoss
from pykeen.trackers import ConsoleResultTracker

from CONSTANT import RELATION_CONFIDENCE


"""
Modèles recommandés selon le cas d'usage :

TransE : Relations simples, hiérarchiques
TransR : Relations complexes nécessitant des espaces différents
ComplEx : Relations asymétriques, compositions
RotatE : Relations avec des patterns de rotation
DistMult : Relations symétriques
"""


class PykeenRelationTransformation:
    """Transformation de relation utilisant PyKEEN pour l'apprentissage avancé."""

    def __init__(self, relation_uri: str, label: str = None, model_type: str = "TransE"):
        """
        Initialise une transformation PyKEEN.

        Args:
            relation_uri: URI de la relation
            label: Label humain de la relation
            model_type: Type de modèle PyKEEN ('TransE', 'TransR', 'ComplEx', 'RotatE', 'DistMult')
        """
        self.uri = relation_uri
        self.label = label or self._extract_label_from_uri(relation_uri)
        self.model_type = model_type
        self.model = None
        self.triples_factory = None
        self.entity_to_id = {}
        self.relation_to_id = {}
        self.id_to_entity = {}
        self.id_to_relation = {}

        # Propriétés de l'ontologie (conservées)
        self.domain_concept_uris = []
        self.range_concept_uris = []
        self.properties = {
            "transitive": False,
            "symmetric": False,
            "asymmetric": False,
            "reflexive": False,
            "irreflexive": False,
            "functional": False,
            "inverse_functional": False
        }

        # Métadonnées d'entraînement
        self.inverse_relation_uri = None
        self.examples_count = 0
        self.examples_hash = None
        self.last_trained = None
        self.training_loss = None
        self.is_trained = False

        # Configuration d'entraînement
        self.embedding_dim = 128
        self.num_epochs = 100
        self.learning_rate = 0.001
        self.batch_size = 256

    def _extract_label_from_uri(self, uri: str) -> str:
        """Extrait un label lisible à partir de l'URI."""
        if '#' in uri:
            return uri.split('#')[-1]
        return uri.split('/')[-1]

    def _get_model_class(self):
        """Retourne la classe de modèle PyKEEN appropriée."""
        model_classes = {
            "TransE": TransE,
            "TransR": TransR,
            "ComplEx": ComplEx,
            "RotatE": RotatE,
            "DistMult": DistMult
        }
        return model_classes.get(self.model_type, TransE)

    def learn_from_triples(self, triples: List[Tuple[str, str, str]],
                           negative_sampling_ratio: float = 1.0,
                           validation_triples: Optional[List[Tuple[str, str, str]]] = None) -> bool:
        """
        Apprend à partir de triplets (sujet, relation, objet) avec PyKEEN.

        Args:
            triples: Liste de triplets (head, relation, tail)
            negative_sampling_ratio: Ratio d'échantillonnage négatif
            validation_triples: Triplets de validation optionnels

        Returns:
            True si l'apprentissage a réussi
        """
        try:
            # Préparer les données pour PyKEEN
            mapped_triples = []
            entities = set()
            relations = set()

            for head, relation, tail in triples:
                mapped_triples.append([head, relation, tail])
                entities.add(head)
                entities.add(tail)
                relations.add(relation)

            if len(mapped_triples) < 3:
                print(f"⚠️ Pas assez de triplets pour {self.label}: {len(mapped_triples)}")
                return False

            # Créer le TriplesFactory
            triples_array = np.array(mapped_triples)
            self.triples_factory = TriplesFactory.from_labeled_triples(
                triples=triples_array,
                create_inverse_triples=self.properties.get("symmetric", False)
            )

            # Stocker les mappings
            self.entity_to_id = self.triples_factory.entity_to_id
            self.relation_to_id = self.triples_factory.relation_to_id
            self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
            self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}

            # Initialiser le modèle
            model_class = self._get_model_class()
            # Passer la loss function au modèle selon le type
            if self.model_type in ["TransE", "TransR"]:
                # Pour TransE et TransR, utiliser MarginRankingLoss
                self.model = model_class(
                    triples_factory=self.triples_factory,
                    embedding_dim=self.embedding_dim,
                    loss=MarginRankingLoss(margin=1.0),
                    random_seed=42
                )
            else:
                # Pour les autres modèles, utiliser les paramètres par défaut
                self.model = model_class(
                    triples_factory=self.triples_factory,
                    embedding_dim=self.embedding_dim,
                    random_seed=42
                )

            # Configuration de l'entraînement
            optimizer = Adam(params=self.model.parameters(), lr=self.learning_rate)
            loss_function = MarginRankingLoss(margin=1.0)

            # Créer le training loop sans loss_function
            training_loop = SLCWATrainingLoop(
                model=self.model,
                triples_factory=self.triples_factory,
                optimizer=optimizer,
                # Supprimé: loss_function=loss_function,
                negative_sampler_kwargs={"num_negs_per_pos": int(negative_sampling_ratio)}
            )

            # Entraîner le modèle avec des paramètres adaptés
            print(f"🔄 Entraînement {self.model_type} pour {self.label} ({len(mapped_triples)} triplets)...")

            # Ajuster les paramètres selon le nombre de triplets
            if len(mapped_triples) < 20:
                # Pour peu d'exemples, réduire les epochs et batch size
                num_epochs = min(50, self.num_epochs)
                batch_size = min(32, self.batch_size)
            else:
                num_epochs = self.num_epochs
                batch_size = self.batch_size

            losses = training_loop.train(
                triples_factory=self.triples_factory,
                num_epochs=num_epochs,
                batch_size=batch_size,
                sub_batch_size=None
            )

            # Sauvegarder les résultats
            self.training_loss = float(losses[-1]) if losses else None
            self.last_trained = datetime.now().isoformat()
            self.examples_count = len(triples)
            self.is_trained = True

            print(f"✓ Modèle {self.model_type} entraîné pour {self.label} (loss: {self.training_loss:.4f})")
            return True

        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement pour {self.label}: {e}")
            return False

    def predict_objects(self, subject: str, top_k: int = 10, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Prédit les objets les plus probables pour un sujet donné via cette relation.

        Args:
            subject: Entité sujet
            top_k: Nombre maximum de prédictions
            threshold: Seuil de confiance minimum

        Returns:
            Liste des objets prédits avec leurs scores
        """
        if not self.is_trained or self.model is None:
            return []

        if subject not in self.entity_to_id:
            return []

        try:
            # Créer les triplets de prédiction (subject, relation, ?)
            prediction_triples = []
            for entity in self.entity_to_id.keys():
                if entity != subject:  # Éviter l'auto-prédiction sauf si réflexive
                    if entity != subject or self.properties.get("reflexive", False):
                        prediction_triples.append([subject, self.uri, entity])

            if not prediction_triples:
                return []

            # Convertir en tensor
            prediction_array = np.array(prediction_triples)

            # Utiliser PyKEEN pour scorer les triplets
            scores = self.model.score_hrt(
                torch.tensor([self.entity_to_id[subject]] * len(prediction_triples)),
                torch.tensor([self.relation_to_id.get(self.uri, 0)] * len(prediction_triples)),
                torch.tensor([self.entity_to_id[triple[2]] for triple in prediction_triples])
            )

            # Préparer les résultats
            results = []
            for i, score in enumerate(scores.detach().numpy()):
                if score >= threshold:
                    object_entity = prediction_triples[i][2]
                    results.append({
                        "concept_uri": object_entity,
                        "label": object_entity.split('/')[-1],  # Simplified label
                        "similarity": float(score),
                        "confidence": float(score)
                    })

            # Trier par score décroissant
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]

        except Exception as e:
            print(f"⚠️ Erreur lors de la prédiction pour {self.label}: {e}")
            return []

    def get_entity_embedding(self, entity: str) -> Optional[np.ndarray]:
        """
        Récupère l'embedding d'une entité apprise par le modèle.

        Args:
            entity: URI de l'entité

        Returns:
            Embedding de l'entité ou None si non trouvée
        """
        if not self.is_trained or entity not in self.entity_to_id:
            return None

        try:
            entity_id = self.entity_to_id[entity]
            embedding = self.model.entity_representations[0](torch.tensor([entity_id]))
            return embedding.detach().numpy().flatten()
        except Exception:
            return None

    def get_relation_embedding(self) -> Optional[np.ndarray]:
        """
        Récupère l'embedding de la relation.

        Returns:
            Embedding de la relation ou None si non disponible
        """
        if not self.is_trained or self.uri not in self.relation_to_id:
            return None

        try:
            relation_id = self.relation_to_id[self.uri]
            embedding = self.model.relation_representations[0](torch.tensor([relation_id]))
            return embedding.detach().numpy().flatten()
        except Exception:
            return None

    def evaluate_model(self, test_triples: List[Tuple[str, str, str]]) -> Dict[str, float]:
        """
        Évalue le modèle sur des triplets de test.

        Args:
            test_triples: Triplets de test

        Returns:
            Métriques d'évaluation
        """
        if not self.is_trained or not test_triples:
            return {}

        try:
            # Préparer les triplets de test
            test_array = np.array([[t[0], t[1], t[2]] for t in test_triples
                                   if t[0] in self.entity_to_id and t[2] in self.entity_to_id])

            if len(test_array) == 0:
                return {}

            test_factory = TriplesFactory.from_labeled_triples(
                triples=test_array,
                entity_to_id=self.entity_to_id,
                relation_to_id=self.relation_to_id
            )

            # Évaluer
            evaluator = RankBasedEvaluator()
            results = evaluator.evaluate(
                model=self.model,
                mapped_triples=test_factory.mapped_triples
            )

            return {
                "hits_at_1": float(results.get_metric("hits_at_1")),
                "hits_at_3": float(results.get_metric("hits_at_3")),
                "hits_at_10": float(results.get_metric("hits_at_10")),
                "mean_rank": float(results.get_metric("mean_rank")),
                "mean_reciprocal_rank": float(results.get_metric("mean_reciprocal_rank"))
            }

        except Exception as e:
            print(f"⚠️ Erreur lors de l'évaluation pour {self.label}: {e}")
            return {}


class PykeenRelationManager:
    """Gestionnaire de relations utilisant PyKEEN pour l'apprentissage avancé."""

    def __init__(self, ontology_manager, storage_dir: str = "pykeen_models"):
        """
        Initialise le gestionnaire PyKEEN.

        Args:
            ontology_manager: Gestionnaire d'ontologie
            storage_dir: Répertoire de stockage pour les modèles
        """
        self.ontology_manager = ontology_manager
        self.storage_dir = storage_dir
        self.transformations = {}  # URI -> PyKeenRelationTransformation
        self.global_model = None  # Modèle global pour toutes les relations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Créer le répertoire de stockage
        os.makedirs(storage_dir, exist_ok=True)
        print(f"🚀 PyKEEN utilisant le device: {self.device}")

    async def initialize(self, model_type: str = "TransE"):
        """
        Initialise le gestionnaire en créant les transformations.

        Args:
            model_type: Type de modèle à utiliser par défaut
        """
        print("Initialisation du gestionnaire PyKEEN...")

        # Charger ou créer les transformations
        if not await self._load_transformations():
            for uri, relation in self.ontology_manager.relations.items():
                transform = PykeenRelationTransformation(uri, relation.label, model_type)

                # Ajouter les concepts de domaine et de portée
                for domain_concept in relation.domain:
                    if hasattr(domain_concept, 'uri'):
                        transform.domain_concept_uris.append(domain_concept.uri)

                for range_concept in relation.range:
                    if hasattr(range_concept, 'uri'):
                        transform.range_concept_uris.append(range_concept.uri)

                self.transformations[uri] = transform

            print(f"✓ {len(self.transformations)} transformations PyKEEN créées")

        # Analyser les propriétés des relations
        self._analyze_relation_properties()

    async def train_global_model(self, all_triples: List[Tuple[str, str, str]],
                                 model_type: str = "TransE",
                                 embedding_dim: int = 128,
                                 num_epochs: int = 200) -> bool:
        """
        Entraîne un modèle global sur tous les triplets.

        Args:
            all_triples: Tous les triplets disponibles
            model_type: Type de modèle PyKEEN
            embedding_dim: Dimension des embeddings
            num_epochs: Nombre d'époques d'entraînement

        Returns:
            True si l'entraînement a réussi
        """
        if not all_triples:
            print("⚠️ Aucun triplet fourni pour l'entraînement global")
            return False

        try:
            print(f"🔄 Entraînement du modèle global {model_type} avec {len(all_triples)} triplets...")

            # Préparer les données
            triples_array = np.array([[t[0], t[1], t[2]] for t in all_triples])
            triples_factory = TriplesFactory.from_labeled_triples(triples=triples_array)

            # Sélectionner le modèle
            model_classes = {
                "TransE": TransE,
                "TransR": TransR,
                "ComplEx": ComplEx,
                "RotatE": RotatE,
                "DistMult": DistMult
            }
            model_class = model_classes.get(model_type, TransE)

            # Créer le modèle
            self.global_model = model_class(
                triples_factory=triples_factory,
                embedding_dim=embedding_dim,
                random_seed=42
            ).to(self.device)

            # Configuration d'entraînement
            optimizer = Adam(params=self.global_model.parameters(), lr=0.001)
            loss_function = MarginRankingLoss(margin=1.0)

            training_loop = SLCWATrainingLoop(
                model=self.global_model,
                triples_factory=triples_factory,
                optimizer=optimizer,
                loss_function=loss_function
            )

            # Entraîner
            losses = training_loop.train(
                triples_factory=triples_factory,
                num_epochs=num_epochs,
                batch_size=512
            )

            print(f"✓ Modèle global {model_type} entraîné (loss finale: {losses[-1]:.4f})")

            # Sauvegarder
            await self._save_global_model()
            return True

        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement global: {e}")
            return False

    async def learn_relation_transformation(self, relation_uri: str,
                                            examples: List[Tuple[str, str]],
                                            concept_embeddings: Dict[str, np.ndarray] = None,
                                            force_relearn: bool = False,
                                            min_examples_threshold: int = 5) -> bool:
        """
        Apprend une transformation de relation spécifique.

        Args:
            relation_uri: URI de la relation
            examples: Exemples (sujet, objet)
            concept_embeddings: Embeddings de concepts (non utilisé avec PyKEEN)
            force_relearn: Force le ré-apprentissage
            min_examples_threshold: Seuil minimum d'exemples

        Returns:
            True si l'apprentissage a réussi
        """
        if relation_uri not in self.transformations:
            print(f"⚠️ Relation {relation_uri} non trouvée")
            return False

        transform = self.transformations[relation_uri]

        # Vérifier si déjà appris
        if not force_relearn and transform.is_trained and transform.examples_count >= min_examples_threshold:
            print(f"✓ Transformation pour {transform.label} déjà apprise")
            return True

        if len(examples) < min_examples_threshold:
            print(f"⚠️ Pas assez d'exemples pour {transform.label}: {len(examples)}")
            return False

        # Convertir les exemples en triplets
        triples = [(subj, relation_uri, obj) for subj, obj in examples]

        # Filtrer les triplets pour éviter les doublons
        unique_triples = list(set(triples))
        if len(unique_triples) < len(triples):
            print(f"📝 {len(triples) - len(unique_triples)} triplets dupliqués supprimés pour {transform.label}")

        if len(unique_triples) < min_examples_threshold:
            print(
                f"⚠️ Pas assez de triplets uniques pour {transform.label}: {len(unique_triples)} < {min_examples_threshold}")
            return False

        # Apprendre avec PyKEEN
        success = transform.learn_from_triples(unique_triples)

        if success:
            transform.examples_hash = self._compute_examples_hash(examples)
            await self._save_transformations()

        return success

    def get_related_concepts(self, concept_uri: str, relation_uri: str,
                             concept_embeddings: Dict[str, np.ndarray] = None,
                             top_k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Trouve les concepts liés via PyKEEN.

        Args:
            concept_uri: URI du concept source
            relation_uri: URI de la relation
            concept_embeddings: Non utilisé avec PyKEEN
            top_k: Nombre maximum de résultats
            threshold: Seuil minimal de similarité

        Returns:
            Liste des concepts liés
        """
        if relation_uri not in self.transformations:
            return []

        transform = self.transformations[relation_uri]
        return transform.predict_objects(concept_uri, top_k, threshold)

    # Méthodes utilitaires conservées de l'implémentation originale
    def _analyze_relation_properties(self):
        """Analyse les propriétés logiques des relations dans l'ontologie."""
        properties_found = 0
        for uri, transform in self.transformations.items():
            for axiom_type, source, target in self.ontology_manager.axioms:
                if source == uri:
                    if axiom_type == "transitive_property":
                        transform.properties["transitive"] = True
                        properties_found += 1
                    elif axiom_type == "symmetric_property":
                        transform.properties["symmetric"] = True
                        properties_found += 1
                    elif axiom_type == "asymmetric_property":
                        transform.properties["asymmetric"] = True
                        properties_found += 1
                    elif axiom_type == "reflexive_property":
                        transform.properties["reflexive"] = True
                        properties_found += 1
                    elif axiom_type == "irreflexive_property":
                        transform.properties["irreflexive"] = True
                        properties_found += 1
                    elif axiom_type == "functional_property":
                        transform.properties["functional"] = True
                        properties_found += 1
                    elif axiom_type == "inverse_functional_property":
                        transform.properties["inverse_functional"] = True
                        properties_found += 1
                    elif axiom_type == "inverse_of" and target in self.transformations:
                        transform.inverse_relation_uri = target
                        self.transformations[target].inverse_relation_uri = uri
                        properties_found += 1

        print(f"✓ {properties_found} propriétés logiques de relations identifiées")

    def _compute_examples_hash(self, examples: List[Tuple[str, str]]) -> str:
        """Calcule un hash des exemples."""
        examples_str = str(sorted(examples))
        return hashlib.md5(examples_str.encode()).hexdigest()

    async def _save_transformations(self):
        """Sauvegarde les transformations."""
        transform_path = os.path.join(self.storage_dir, "pykeen_transformations.pkl")
        try:
            # Sauvegarder uniquement les métadonnées, pas les modèles PyTorch
            serializable_data = {}
            for uri, transform in self.transformations.items():
                serializable_data[uri] = {
                    'uri': transform.uri,
                    'label': transform.label,
                    'model_type': transform.model_type,
                    'domain_concept_uris': transform.domain_concept_uris,
                    'range_concept_uris': transform.range_concept_uris,
                    'properties': transform.properties,
                    'examples_count': transform.examples_count,
                    'examples_hash': transform.examples_hash,
                    'last_trained': transform.last_trained,
                    'is_trained': transform.is_trained
                }

            with open(transform_path, 'wb') as f:
                pickle.dump(serializable_data, f)
        except Exception as e:
            print(f"⚠️ Erreur lors de la sauvegarde: {e}")

    async def _load_transformations(self) -> bool:
        """Charge les transformations sauvegardées."""
        transform_path = os.path.join(self.storage_dir, "pykeen_transformations.pkl")
        if os.path.exists(transform_path):
            try:
                with open(transform_path, 'rb') as f:
                    data = pickle.load(f)

                for uri, transform_data in data.items():
                    transform = PykeenRelationTransformation(
                        transform_data['uri'],
                        transform_data['label'],
                        transform_data['model_type']
                    )
                    transform.domain_concept_uris = transform_data['domain_concept_uris']
                    transform.range_concept_uris = transform_data['range_concept_uris']
                    transform.properties = transform_data['properties']
                    transform.examples_count = transform_data['examples_count']
                    transform.examples_hash = transform_data.get('examples_hash')
                    transform.last_trained = transform_data.get('last_trained')
                    transform.is_trained = transform_data.get('is_trained', False)

                    self.transformations[uri] = transform

                print(f"✓ {len(self.transformations)} transformations PyKEEN chargées")
                return True
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement: {e}")
        return False

    async def _save_global_model(self):
        """Sauvegarde le modèle global."""
        if self.global_model:
            model_path = os.path.join(self.storage_dir, "global_model.pkl")
            try:
                torch.save(self.global_model.state_dict(), model_path)
            except Exception as e:
                print(f"⚠️ Erreur lors de la sauvegarde du modèle global: {e}")