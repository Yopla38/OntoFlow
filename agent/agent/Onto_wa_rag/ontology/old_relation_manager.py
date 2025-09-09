"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# ontology/relation_manager.py
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from scipy.spatial.distance import cosine

from CONSTANT import RELATION_CONFIDENCE


class RelationTransformation:
    """Représente une transformation entre concepts via une relation ontologique."""

    def __init__(self, relation_uri: str, label: str = None):
        """
        Initialise une transformation de relation.

        Args:
            relation_uri: URI de la relation
            label: Label humain de la relation
        """
        self.uri = relation_uri
        self.label = label or self._extract_label_from_uri(relation_uri)
        self.domain_concept_uris = []  # Concepts qui peuvent être sujets
        self.range_concept_uris = []  # Concepts qui peuvent être objets
        self.transformation_matrix = None  # Matrice de transformation
        self.properties = {  # Propriétés logiques
            "transitive": False,
            "symmetric": False,
            "asymmetric": False,
            "reflexive": False,
            "irreflexive": False,
            "functional": False,
            "inverse_functional": False
        }
        self.inverse_relation_uri = None  # URI de la relation inverse si elle existe
        self.examples_count = 0  # Nombre d'exemples utilisés pour l'apprentissage
        self.examples_hash = None  # Hash des exemples utilisés
        self.last_trained = None  # Timestamp du dernier entraînement
        self.training_error = None  # Erreur d'entraînement pour monitoring

    def _extract_label_from_uri(self, uri: str) -> str:
        """Extrait un label lisible à partir de l'URI."""
        if '#' in uri:
            return uri.split('#')[-1]
        return uri.split('/')[-1]

    def apply(self, subject_embedding: np.ndarray) -> np.ndarray:
        """
        Applique la transformation à l'embedding d'un sujet.

        Args:
            subject_embedding: Embedding du concept sujet

        Returns:
            Embedding transformé représentant l'objet attendu
        """
        if self.transformation_matrix is not None:
            result = np.matmul(subject_embedding, self.transformation_matrix)
            # Normaliser le résultat
            norm = np.linalg.norm(result)
            if norm > 0:
                result = result / norm
            return result
        return subject_embedding  # Fallback si pas de matrice

    def learn_from_examples(self, subject_embeddings: List[np.ndarray],
                            object_embeddings: List[np.ndarray],
                            learning_rate: float = 0.01,
                            max_iterations: int = 150) -> bool:

        from datetime import datetime
        """Version améliorée avec régularisation et early stopping"""
        if len(subject_embeddings) != len(object_embeddings) or len(subject_embeddings) == 0:
            return False

        # Convertir en tableaux numpy
        subjects = np.vstack(subject_embeddings)
        objects = np.vstack(object_embeddings)

        # Obtenir la dimension des embeddings
        emb_dim = subjects.shape[1]

        # Initialiser la matrice de transformation si nécessaire
        if self.transformation_matrix is None:
            self.transformation_matrix = np.eye(emb_dim) * 0.1  # Initialisation plus stable

        # Suivi pour early stopping
        best_error = float('inf')
        best_matrix = None
        patience = 10
        patience_counter = 0

        # Descente de gradient avec early stopping
        for i in range(max_iterations):
            # Prédictions actuelles
            predictions = np.matmul(subjects, self.transformation_matrix)

            # Calculer l'erreur
            error = np.mean(np.square(predictions - objects))

            # Early stopping
            if error < best_error:
                best_error = error
                best_matrix = self.transformation_matrix.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.transformation_matrix = best_matrix
                    break

            # Mettre à jour la matrice de transformation
            gradient = np.matmul(subjects.T, predictions - objects) / len(subjects)
            self.transformation_matrix -= learning_rate * gradient

            # Réduire le learning rate progressivement
            if i % 30 == 0 and i > 0:
                learning_rate *= 0.9

        # Sauvegarder les métadonnées d'entraînement
        self.last_trained = datetime.now().isoformat()
        self.training_error = best_error if 'best_error' in locals() else None
        self.examples_count += len(subject_embeddings)

        return True


class RelationManager:
    """Gère les transformations de relations dans une ontologie."""

    def __init__(self, ontology_manager, storage_dir: str = "relation_models"):
        """
        Initialise le gestionnaire de relations.

        Args:
            ontology_manager: Gestionnaire d'ontologie
            storage_dir: Répertoire de stockage pour les modèles de relations
        """
        self.ontology_manager = ontology_manager
        self.storage_dir = storage_dir
        self.transformations = {}  # URI -> RelationTransformation

        # Créer le répertoire de stockage
        os.makedirs(storage_dir, exist_ok=True)

    async def initialize(self):
        """Initialise le gestionnaire en chargeant les relations de l'ontologie."""
        print("Initialisation du gestionnaire de relations ontologiques...")

        # Charger les transformations existantes ou en créer de nouvelles
        if not await self._load_transformations():
            # Créer les transformations pour toutes les relations de l'ontologie
            for uri, relation in self.ontology_manager.relations.items():
                self.transformations[uri] = RelationTransformation(uri, relation.label)

                # Ajouter les concepts de domaine et de portée
                for domain_concept in relation.domain:
                    if hasattr(domain_concept, 'uri'):
                        self.transformations[uri].domain_concept_uris.append(domain_concept.uri)

                for range_concept in relation.range:
                    if hasattr(range_concept, 'uri'):
                        self.transformations[uri].range_concept_uris.append(range_concept.uri)

            print(f"✓ {len(self.transformations)} transformations de relations créées")

        # Analyser les propriétés des relations
        self._analyze_relation_properties()

    async def _load_transformations(self) -> bool:
        """
        Charge les transformations sauvegardées.

        Returns:
            True si le chargement a réussi, False sinon
        """
        transform_path = os.path.join(self.storage_dir, "relation_transformations.pkl")

        if os.path.exists(transform_path):
            try:
                with open(transform_path, 'rb') as f:
                    self.transformations = pickle.load(f)
                print(f"✓ {len(self.transformations)} transformations de relations chargées")
                return True
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement des transformations: {e}")

        return False

    async def _save_transformations(self):
        """Sauvegarde les transformations de relations."""
        transform_path = os.path.join(self.storage_dir, "relation_transformations.pkl")

        try:
            with open(transform_path, 'wb') as f:
                pickle.dump(self.transformations, f)
        except Exception as e:
            print(f"⚠️ Erreur lors de la sauvegarde des transformations: {e}")

    def _analyze_relation_properties(self):
        """Analyse les propriétés logiques des relations dans l'ontologie."""
        # Compteur pour le débogage
        properties_found = 0

        for uri, transform in self.transformations.items():
            # Analyser les axiomes dans l'ontologie pour déterminer les propriétés
            for axiom_type, source, target in self.ontology_manager.axioms:
                if source == uri:
                    property_updated = False

                    # Propriétés OWL standard
                    if axiom_type == "transitive_property":
                        transform.properties["transitive"] = True
                        property_updated = True
                    elif axiom_type == "symmetric_property":
                        transform.properties["symmetric"] = True
                        property_updated = True
                    elif axiom_type == "asymmetric_property":
                        transform.properties["asymmetric"] = True
                        property_updated = True
                    elif axiom_type == "reflexive_property":
                        transform.properties["reflexive"] = True
                        property_updated = True
                    elif axiom_type == "irreflexive_property":
                        transform.properties["irreflexive"] = True
                        property_updated = True
                    elif axiom_type == "functional_property":
                        transform.properties["functional"] = True
                        property_updated = True
                    elif axiom_type == "inverse_functional_property":
                        transform.properties["inverse_functional"] = True
                        property_updated = True
                    elif axiom_type == "inverse_of" and target in self.transformations:
                        transform.inverse_relation_uri = target
                        self.transformations[target].inverse_relation_uri = uri
                        property_updated = True

                    if property_updated:
                        properties_found += 1

        print(f"✓ {properties_found} propriétés logiques de relations identifiées")

    async def learn_relation_transformation(self, relation_uri: str, examples: List[Tuple[str, str]],
                                            concept_embeddings: Dict[str, np.ndarray],
                                            force_relearn: bool = False,
                                            min_examples_threshold: int = 5) -> bool:
        """
        Apprend la transformation pour une relation à partir d'exemples.

        Args:
            relation_uri: URI de la relation
            examples: Liste de tuples (sujet_uri, objet_uri) ou (sujet_label, objet_label)
            concept_embeddings: Dictionnaire d'embeddings (URI ou label -> embedding)
            force_relearn: Si True, force le re-apprentissage même si déjà appris
            min_examples_threshold: Nombre minimum d'exemples pour considérer l'apprentissage suffisant

        Returns:
            True si l'apprentissage a réussi, False sinon
        """
        if relation_uri not in self.transformations:
            print(f"⚠️ Relation {relation_uri} non trouvée")
            return False

        transform = self.transformations[relation_uri]

        # Vérifier si l'apprentissage est nécessaire
        if not force_relearn and self._is_already_learned(transform, examples, min_examples_threshold):
            print(f"✓ Transformation pour {transform.label} déjà apprise avec {transform.examples_count} exemples")
            return True

        # Collecter les embeddings pour l'apprentissage
        subject_embeddings = []
        object_embeddings = []

        for subject_id, object_id in examples:
            if subject_id in concept_embeddings and object_id in concept_embeddings:
                subject_embeddings.append(concept_embeddings[subject_id])
                object_embeddings.append(concept_embeddings[object_id])

        if not subject_embeddings:
            print(f"⚠️ Aucun exemple valide pour la relation {relation_uri}")
            return False

        # Apprendre la transformation
        success = transform.learn_from_examples(subject_embeddings, object_embeddings)

        if success:
            # Stocker un hash des exemples pour éviter le re-calcul
            examples_hash = self._compute_examples_hash(examples)
            transform.examples_hash = examples_hash

            # Sauvegarder les transformations
            await self._save_transformations()
            print(
                f"✓ Transformation pour {transform.label} ({relation_uri}) apprise avec {len(subject_embeddings)} exemples")

        return success

    def _is_already_learned(self, transform: RelationTransformation,
                            examples: List[Tuple[str, str]],
                            min_examples_threshold: int) -> bool:
        """
        Vérifie si une transformation a déjà été apprise suffisamment.

        Args:
            transform: Transformation à vérifier
            examples: Nouveaux exemples
            min_examples_threshold: Seuil minimum d'exemples

        Returns:
            True si déjà suffisamment apprise
        """
        # Vérifier si la matrice existe
        if transform.transformation_matrix is None:
            return False

        # Vérifier si on a suffisamment d'exemples
        if transform.examples_count < min_examples_threshold:
            return False

        # Vérifier si les exemples ont changé (optionnel)
        current_hash = self._compute_examples_hash(examples)
        if hasattr(transform, 'examples_hash') and transform.examples_hash == current_hash:
            return True

        # Si on a beaucoup plus d'exemples que le minimum, considérer comme appris
        return transform.examples_count >= min_examples_threshold * 2

    def _compute_examples_hash(self, examples: List[Tuple[str, str]]) -> str:
        """Calcule un hash des exemples pour détecter les changements."""
        import hashlib
        examples_str = str(sorted(examples))
        return hashlib.md5(examples_str.encode()).hexdigest()

    def get_related_concepts(self, concept_uri: str, relation_uri: str,
                             concept_embeddings: Dict[str, np.ndarray],
                             top_k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Trouve les concepts liés à un concept via une relation.

        Args:
            concept_uri: URI du concept source
            relation_uri: URI de la relation
            concept_embeddings: Dictionnaire d'embeddings de concepts
            top_k: Nombre maximum de résultats
            threshold: Seuil minimal de similarité

        Returns:
            Liste des concepts les plus probablement liés via la relation
        """
        if relation_uri not in self.transformations or concept_uri not in concept_embeddings:
            return []

        transform = self.transformations[relation_uri]

        # Appliquer la transformation
        source_embedding = concept_embeddings[concept_uri]
        transformed_embedding = transform.apply(source_embedding)

        # Chercher les concepts les plus similaires au résultat
        results = []

        for target_uri, target_embedding in concept_embeddings.items():
            # Éviter l'auto-référence (sauf si la relation est réflexive)
            if target_uri == concept_uri and not transform.properties["reflexive"]:
                continue

            # Vérifier que le concept cible est dans la portée (range) de la relation
            # Si la relation a une portée définie
            if transform.range_concept_uris and target_uri not in transform.range_concept_uris:
                continue

            # Calculer la similarité
            similarity = 1.0 - cosine(transformed_embedding, target_embedding)

            if similarity >= threshold:
                # Récupérer le concept pour son label
                concept = self.ontology_manager.concepts.get(target_uri)
                label = concept.label if concept and hasattr(concept, 'label') else target_uri.split('/')[-1]

                results.append({
                    "concept_uri": target_uri,
                    "label": label,
                    "similarity": float(similarity)
                })

        # Trier par similarité décroissante
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def extract_relation_triples(self, text: str, concept_matches: List[Dict[str, Any]],
                                 concept_embeddings: Dict[str, np.ndarray],
                                 top_k_relations: int = 3) -> List[Dict[str, Any]]:
        """
        Extrait les triplets relation possibles à partir d'un texte et des concepts détectés.

        Args:
            text: Texte à analyser
            concept_matches: Concepts détectés dans le texte
            concept_embeddings: Dictionnaire d'embeddings de concepts
            top_k_relations: Nombre maximum de relations à considérer par paire de concepts

        Returns:
            Liste des triplets (sujet, relation, objet) les plus probables
        """
        # Aucun triplet possible avec moins de 2 concepts
        if len(concept_matches) < 2:
            return []

        triples = []

        # Pour chaque paire de concepts, chercher les relations possibles
        for i, concept1 in enumerate(concept_matches):
            for j in range(i + 1, len(concept_matches)):
                concept2 = concept_matches[j]

                c1_uri = concept1["concept_uri"]
                c2_uri = concept2["concept_uri"]

                if c1_uri not in concept_embeddings or c2_uri not in concept_embeddings:
                    continue

                # Chercher les relations possibles dans les deux directions
                possible_relations_c1_to_c2 = self._find_possible_relations(c1_uri, c2_uri, concept_embeddings)
                possible_relations_c2_to_c1 = self._find_possible_relations(c2_uri, c1_uri, concept_embeddings)

                # Ajouter les triplets les plus probables
                for rel in possible_relations_c1_to_c2[:top_k_relations]:
                    triple = {
                        "subject_uri": c1_uri,
                        "subject_label": concept1["label"],
                        "relation_uri": rel["relation_uri"],
                        "relation_label": rel["relation_label"],
                        "object_uri": c2_uri,
                        "object_label": concept2["label"],
                        "confidence": rel["confidence"]
                    }
                    triples.append(triple)

                for rel in possible_relations_c2_to_c1[:top_k_relations]:
                    triple = {
                        "subject_uri": c2_uri,
                        "subject_label": concept2["label"],
                        "relation_uri": rel["relation_uri"],
                        "relation_label": rel["relation_label"],
                        "object_uri": c1_uri,
                        "object_label": concept1["label"],
                        "confidence": rel["confidence"]
                    }
                    triples.append(triple)

        # Trier par confiance et limiter le nombre de triplets
        triples.sort(key=lambda x: x["confidence"], reverse=True)
        return triples

    def _find_possible_relations(self, subject_uri: str, object_uri: str,
                                 concept_embeddings: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Trouve les relations possibles entre deux concepts.

        Args:
            subject_uri: URI du concept sujet
            object_uri: URI du concept objet
            concept_embeddings: Dictionnaire d'embeddings de concepts

        Returns:
            Liste des relations possibles avec leur score de confiance
        """
        if subject_uri not in concept_embeddings or object_uri not in concept_embeddings:
            return []

        subject_embedding = concept_embeddings[subject_uri]
        object_embedding = concept_embeddings[object_uri]

        results = []

        for rel_uri, transform in self.transformations.items():
            # Vérifier que le sujet est dans le domaine et l'objet dans la portée
            domain_ok = not transform.domain_concept_uris or subject_uri in transform.domain_concept_uris
            range_ok = not transform.range_concept_uris or object_uri in transform.range_concept_uris

            if not domain_ok or not range_ok:
                continue

            # Si la transformation n'a pas de matrice ou peu d'exemples,
            # créer une approximation basée sur la distance directe
            if transform.transformation_matrix is None or transform.examples_count < 3:
                # Distance euclidienne entre les concepts comme heuristique
                baseline_similarity = 1.0 - cosine(subject_embedding, object_embedding)
                confidence = baseline_similarity * 0.6  # Ajustement pour éviter les fausses relations
            else:
                # Appliquer la transformation et calculer la similarité
                transformed = transform.apply(subject_embedding)
                confidence = 1.0 - cosine(transformed, object_embedding)

            # Ne garder que les relations avec une confiance suffisante
            if confidence > RELATION_CONFIDENCE:
                results.append({
                    "relation_uri": rel_uri,
                    "relation_label": transform.label,
                    "confidence": float(confidence)
                })

        # Trier par confiance
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results


