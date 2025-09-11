"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

import os
import re
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import asyncio

import numpy as np


@dataclass
class DocumentHierarchy:
    """Hiérarchie conceptuelle du document"""
    document_concepts: List[Dict[str, Any]] = field(default_factory=list)
    section_concepts: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    paragraph_concepts: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


@dataclass
class SemanticChunk:
    """Chunk sémantique avec ses concepts"""
    chunk_id: str
    text: str
    start_pos: int
    end_pos: int
    level: str  # 'document', 'section', 'paragraph'
    concepts: List[Dict[str, Any]] = field(default_factory=list)
    inherited_concepts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GenericContextResolver:
    """Résolveur de contexte générique basé sur l'ontologie"""

    def __init__(self, ontology_manager, concept_classifier):
        self.ontology_manager = ontology_manager
        self.concept_classifier = concept_classifier

        # Cache des compatibilités calculées
        self.compatibility_cache = {}
        self.concept_embeddings_cache = {}

    def _extract_document_context(self, document_concepts: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extrait le contexte de façon générique via clustering de concepts"""

        if not document_concepts:
            return {}

        # 1. Récupérer les embeddings des concepts dominants
        concept_embeddings = []
        concept_info = []

        for concept in document_concepts[:10]:  # Top 10
            uri = concept.get('concept_uri', '')
            embedding = self._get_concept_embedding(uri)
            if embedding is not None:
                concept_embeddings.append(embedding)
                concept_info.append(concept)

        if not concept_embeddings:
            return {}

        # 2. Clustering automatique pour identifier les groupes sémantiques
        context_clusters = self._cluster_concepts(concept_embeddings, concept_info)

        # 3. Calculer les poids des clusters
        context_weights = {}
        total_confidence = sum(c['confidence'] for c in document_concepts[:10])

        for cluster_id, cluster_info in context_clusters.items():
            cluster_weight = sum(c['confidence'] for c in cluster_info['concepts']) / total_confidence
            context_weights[cluster_id] = cluster_weight

        print(f"   🧠 Contexte générique: {len(context_clusters)} clusters détectés")
        for cluster_id, weight in context_weights.items():
            concepts_labels = [c['label'] for c in context_clusters[cluster_id]['concepts'][:3]]
            print(f"      - {cluster_id}: {weight:.2f} ({', '.join(concepts_labels)})")

        return context_weights, context_clusters

    def _cluster_concepts(self, embeddings: List[np.ndarray], concept_info: List[Dict]) -> Dict[str, Dict]:
        """Clustering automatique des concepts par similarité sémantique"""

        if len(embeddings) < 2:
            return {"cluster_0": {"concepts": concept_info, "centroid": embeddings[0] if embeddings else None}}

        # Clustering simple par seuil de similarité
        clusters = {}
        cluster_id = 0

        for i, (embedding, concept) in enumerate(zip(embeddings, concept_info)):
            assigned = False

            # Tenter d'assigner à un cluster existant
            for cid, cluster_data in clusters.items():
                centroid = cluster_data['centroid']
                similarity = np.dot(embedding, centroid)

                if similarity > 0.7:  # Seuil de similarité
                    cluster_data['concepts'].append(concept)
                    # Recalculer le centroïde
                    all_embeddings = [self._get_concept_embedding(c['concept_uri']) for c in cluster_data['concepts']]
                    cluster_data['centroid'] = np.mean([e for e in all_embeddings if e is not None], axis=0)
                    assigned = True
                    break

            # Créer un nouveau cluster si pas assigné
            if not assigned:
                cluster_name = f"cluster_{cluster_id}"
                clusters[cluster_name] = {
                    "concepts": [concept],
                    "centroid": embedding
                }
                cluster_id += 1

        return clusters

    def _get_concept_context_score(self, concept: Dict[str, Any],
                                   context_clusters: Dict[str, Dict]) -> float:
        """Calcule le score de contexte de façon générique"""

        concept_uri = concept.get('concept_uri', '')
        concept_embedding = self._get_concept_embedding(concept_uri)

        if concept_embedding is None:
            return 0.0

        max_compatibility = 0.0
        best_cluster = None

        # Calculer la compatibilité avec chaque cluster de contexte
        for cluster_id, cluster_data in context_clusters.items():
            cluster_centroid = cluster_data['centroid']

            # Similarité sémantique directe
            semantic_similarity = np.dot(concept_embedding, cluster_centroid)

            # Compatibilité ontologique
            ontology_compatibility = self._calculate_ontology_compatibility(
                concept_uri, [c['concept_uri'] for c in cluster_data['concepts']]
            )

            # Score combiné
            total_compatibility = (semantic_similarity * 0.6) + (ontology_compatibility * 0.4)

            if total_compatibility > max_compatibility:
                max_compatibility = total_compatibility
                best_cluster = cluster_id

        print(f"     🎯 {concept['label']} → {best_cluster} (compat: {max_compatibility:.2f})")
        return max_compatibility

    def _calculate_ontology_compatibility(self, concept_uri: str, cluster_concept_uris: List[str]) -> float:
        """Calcule la compatibilité ontologique générique"""

        if not cluster_concept_uris:
            return 0.0

        # Cache key
        cache_key = f"{concept_uri}::{':'.join(sorted(cluster_concept_uris))}"
        if cache_key in self.compatibility_cache:
            return self.compatibility_cache[cache_key]

        max_compatibility = 0.0

        for cluster_uri in cluster_concept_uris:
            compatibility = self._calculate_pairwise_compatibility(concept_uri, cluster_uri)
            max_compatibility = max(max_compatibility, compatibility)

        # Cache du résultat
        self.compatibility_cache[cache_key] = max_compatibility
        return max_compatibility

    def _calculate_pairwise_compatibility(self, uri1: str, uri2: str) -> float:
        """Calcule la compatibilité entre deux concepts via l'ontologie"""

        if uri1 == uri2:
            return 1.0

        # 1. Relations hiérarchiques directes
        if self._are_hierarchically_related(uri1, uri2):
            return 0.8

        # 2. Relations sémantiques dans l'ontologie
        semantic_relation = self._get_semantic_relation(uri1, uri2)
        if semantic_relation:
            return self._relation_to_score(semantic_relation)

        # 3. Même domaine ontologique
        if self._same_ontology_domain(uri1, uri2):
            return 0.4

        # 4. Co-occurrence apprise (si disponible)
        cooccurrence = self._get_learned_cooccurrence(uri1, uri2)
        if cooccurrence > 0:
            return cooccurrence * 0.6

        return 0.0

    def _are_hierarchically_related(self, uri1: str, uri2: str) -> bool:
        """Vérifie les relations hiérarchiques"""

        concept1 = self.ontology_manager.concepts.get(uri1)
        concept2 = self.ontology_manager.concepts.get(uri2)

        if not concept1 or not concept2:
            return False

        # Parent-enfant direct
        if hasattr(concept1, 'parents'):
            for parent in concept1.parents:
                if hasattr(parent, 'uri') and parent.uri == uri2:
                    return True

        if hasattr(concept2, 'parents'):
            for parent in concept2.parents:
                if hasattr(parent, 'uri') and parent.uri == uri1:
                    return True

        return False

    def _get_semantic_relation(self, uri1: str, uri2: str) -> Optional[str]:
        """Récupère la relation sémantique entre deux concepts"""

        for axiom_type, source, target in self.ontology_manager.axioms:
            if (source == uri1 and target == uri2) or (source == uri2 and target == uri1):
                return axiom_type

        return None

    def _relation_to_score(self, relation_type: str) -> float:
        """Convertit un type de relation en score de compatibilité"""

        relation_scores = {
            'semantic_equivalent': 0.9,
            'semantic_similar': 0.7,
            'semantic_related': 0.5,
            'semantic_opposite': -0.8,  # Incompatible !
            'semantic_exclusive': -0.9  # Très incompatible !
        }

        return relation_scores.get(relation_type, 0.3)

    def _same_ontology_domain(self, uri1: str, uri2: str) -> bool:
        """Vérifie si deux concepts sont du même domaine ontologique"""

        if '#' not in uri1 or '#' not in uri2:
            return False

        domain1 = uri1.rsplit('#', 1)[0]
        domain2 = uri2.rsplit('#', 1)[0]

        return domain1 == domain2

    def _get_learned_cooccurrence(self, uri1: str, uri2: str) -> float:
        """Score de co-occurrence appris (à implémenter si apprentissage disponible)"""

        # TODO: Implémenter avec les données d'apprentissage
        # Analyser les documents où les deux concepts apparaissent ensemble
        return 0.0

    def _get_concept_embedding(self, concept_uri: str) -> Optional[np.ndarray]:
        """Récupère l'embedding d'un concept avec cache"""

        if concept_uri in self.concept_embeddings_cache:
            return self.concept_embeddings_cache[concept_uri]

        embedding = None
        if (hasattr(self.concept_classifier, 'concept_embeddings') and
                concept_uri in self.concept_classifier.concept_embeddings):
            embedding = self.concept_classifier.concept_embeddings[concept_uri]

        self.concept_embeddings_cache[concept_uri] = embedding
        return embedding

class SelectiveConceptDetector:
    """Détection sélective de concepts selon le niveau hiérarchique"""

    def __init__(self, concept_classifier, ontology_manager):
        self.concept_classifier = concept_classifier
        self.ontology_manager = ontology_manager

    async def detect_document_level_concepts(self, text: str, max_concepts: int = 5) -> List[Dict[str, Any]]:
        """Détection SÉLECTIVE au niveau document - seulement les concepts dominants"""

        print(f"🧠 Détection sélective au niveau document (max {max_concepts} concepts)")

        # Détecter tous les concepts
        all_concepts = await self.concept_classifier.smart_concept_detection(text)

        # NOUVEAU : Filtrage par dominance contextuelle
        dominant_concepts = self._filter_dominant_concepts(all_concepts, text, max_concepts)

        print(f"   📋 {len(dominant_concepts)} concepts dominants sélectionnés:")
        for concept in dominant_concepts:
            print(
                f"      - {concept['label']} (conf: {concept['confidence']:.2f}, dominance: {concept.get('dominance_score', 0):.2f})")

        return dominant_concepts

    def _filter_dominant_concepts(self, concepts: List[Dict[str, Any]], text: str, max_concepts: int) -> List[
        Dict[str, Any]]:
        """Filtre les concepts dominants selon le contexte"""

        # 1. Calculer la dominance contextuelle
        for concept in concepts:
            concept['dominance_score'] = self._calculate_dominance_score(concept, text)

        # 2. Résoudre les ambiguïtés AVANT la sélection
        resolved_concepts = self._resolve_ambiguities_early(concepts)

        # 3. Sélectionner les plus dominants
        resolved_concepts.sort(key=lambda x: x['dominance_score'], reverse=True)

        return resolved_concepts[:max_concepts]

    def _calculate_dominance_score(self, concept: Dict[str, Any], text: str) -> float:
        """Calcule le score de dominance d'un concept dans le texte"""
        label = concept.get('label', '').lower()
        uri = concept.get('concept_uri', '').lower()
        base_confidence = concept.get('confidence', 0)

        # Compter les occurrences dans le texte
        text_lower = text.lower()
        label_occurrences = text_lower.count(label)

        # Bonus pour les concepts qui apparaissent plusieurs fois
        frequency_bonus = min(0.3, label_occurrences * 0.1)

        # Bonus pour les concepts en début de document
        position_bonus = 0.1 if label in text_lower[:200] else 0

        # Malus pour les concepts très génériques
        generic_penalty = 0.1 if label in ['domaine physique', 'physics'] else 0

        dominance = base_confidence + frequency_bonus + position_bonus - generic_penalty

        return dominance

    def _resolve_ambiguities_early(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Résout les ambiguïtés AVANT la sélection des concepts dominants"""

        # Grouper par label pour identifier les ambiguïtés
        grouped = {}
        for concept in concepts:
            label = concept.get('label', '').lower()
            if label not in grouped:
                grouped[label] = []
            grouped[label].append(concept)

        resolved = []

        for label, concept_group in grouped.items():
            if len(concept_group) == 1:
                resolved.append(concept_group[0])
            else:
                # Ambiguïté : prendre le plus dominant
                print(f"   ⚠️  Ambiguïté précoce sur '{label}' - {len(concept_group)} variantes")

                best_concept = max(concept_group, key=lambda x: x['dominance_score'])
                best_concept['early_ambiguity_resolved'] = True
                resolved.append(best_concept)

                chosen = best_concept.get('concept_uri', '').split('#')[-1]
                print(f"   ✅ Choix précoce: {chosen} (dominance: {best_concept['dominance_score']:.2f})")

        return resolved

    async def detect_section_level_concepts(self, text: str, document_concepts: List[Dict[str, Any]],
                                            max_concepts: int = 8) -> List[Dict[str, Any]]:
        """Détection au niveau section - dans le voisinage des concepts document"""

        print(f"🔍 Détection au niveau section (max {max_concepts} concepts)")

        # Construire l'espace de recherche depuis les concepts document
        search_space = self._build_section_search_space(document_concepts)

        # Détection globale
        all_detected = await self.concept_classifier.smart_concept_detection(text)

        # Filtrer par l'espace de recherche
        section_concepts = []
        for concept in all_detected:
            concept_uri = concept.get('concept_uri', '')
            if concept_uri in search_space:
                # Calculer la vraie distance depuis les concepts document
                distance = self._calculate_real_distance(concept_uri, document_concepts)
                concept['path_distance'] = distance
                concept['in_section_path'] = True
                section_concepts.append(concept)

        # Limiter et trier
        section_concepts.sort(key=lambda x: (x['path_distance'], -x['confidence']))
        return section_concepts[:max_concepts]

    def _build_section_search_space(self, document_concepts: List[Dict[str, Any]]) -> Set[str]:
        """Construit l'espace de recherche pour les sections"""
        search_space = set()

        for concept in document_concepts:
            concept_uri = concept.get('concept_uri', '')
            if concept_uri:
                # Concept lui-même
                search_space.add(concept_uri)

                # Voisins directs seulement (distance 1)
                neighbors = self._get_direct_neighbors(concept_uri)
                search_space.update(neighbors)

        return search_space

    def _get_direct_neighbors(self, concept_uri: str) -> Set[str]:
        """Récupère seulement les voisins directs (distance 1)"""
        neighbors = set()

        # Relations hiérarchiques directes
        concept = self.ontology_manager.concepts.get(concept_uri)
        if concept:
            if hasattr(concept, 'parents') and concept.parents:
                for parent in concept.parents:
                    if hasattr(parent, 'uri'):
                        neighbors.add(parent.uri)

            if hasattr(concept, 'children') and concept.children:
                for child in concept.children:
                    if hasattr(child, 'uri'):
                        neighbors.add(child.uri)

        # Relations sémantiques directes
        for axiom_type, source, target in self.ontology_manager.axioms:
            if source == concept_uri:
                neighbors.add(target)
            elif target == concept_uri:
                neighbors.add(source)

        return neighbors

    def _calculate_real_distance(self, concept_uri: str, document_concepts: List[Dict[str, Any]]) -> int:
        """Calcule la vraie distance dans l'ontologie"""
        min_distance = 999

        for doc_concept in document_concepts:
            doc_uri = doc_concept.get('concept_uri', '')
            if doc_uri == concept_uri:
                return 0

            # Vérifier si c'est un voisin direct
            doc_neighbors = self._get_direct_neighbors(doc_uri)
            if concept_uri in doc_neighbors:
                min_distance = min(min_distance, 1)

        return min_distance if min_distance < 999 else 2


class LevelBasedSearchEngine:
    """Moteur de recherche par niveau hiérarchique"""

    def __init__(self, document_store, embedding_provider):
        self.document_store = document_store
        self.embedding_provider = embedding_provider

    async def search_by_level(self, query: str, level: str = 'document',
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """Recherche par niveau spécifique"""

        print(f"🔍 Recherche niveau '{level}' pour: {query}")

        # Générer l'embedding de la requête
        query_embedding = await self.embedding_provider.generate_embeddings([query])
        query_embedding = query_embedding[0]

        # Collecter les chunks du niveau demandé
        level_chunks = []

        for doc_id, chunks in self.document_store.document_chunks.items():
            for chunk in chunks:
                chunk_level = chunk.get('metadata', {}).get('chunk_level', 'unknown')
                if chunk_level == level:
                    level_chunks.append(chunk)

        if not level_chunks:
            print(f"   ⚠️  Aucun chunk de niveau '{level}' trouvé")
            return []

        # Calculer les similarités
        similarities = []
        for chunk in level_chunks:
            chunk_id = chunk['id']
            chunk_embedding = self.document_store.embedding_manager.get_embedding(chunk_id)

            if chunk_embedding is not None:
                similarity = float(np.dot(query_embedding, chunk_embedding))

                # CORRECTION : Enrichir avec les métadonnées de source
                metadata = chunk.get('metadata', {})
                result = {
                    'chunk': chunk,
                    'similarity': similarity,
                    'source_info': {
                        'filename': metadata.get('filename', 'Fichier inconnu'),
                        'filepath': metadata.get('filepath', ''),
                        'start_line': metadata.get('start_pos', 0),
                        'end_line': metadata.get('end_pos', 0),
                        'section_title': metadata.get('section_title', 'Section sans titre'),
                        'chunk_level': metadata.get('chunk_level', level),
                        'document_id': chunk.get('document_id', 'unknown')
                    }
                }
                similarities.append(result)

        # Trier et limiter
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = similarities[:top_k]

        print(f"   ✅ {len(top_results)} résultats trouvés au niveau '{level}'")

        return top_results

    async def hierarchical_search(self, query: str, max_per_level: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """Recherche hiérarchique sur tous les niveaux"""

        print(f"🔍 Recherche hiérarchique pour: {query}")

        results = {}

        # Rechercher à chaque niveau
        for level in ['document', 'section', 'paragraph']:
            level_results = await self.search_by_level(query, level, max_per_level)
            if level_results:
                results[level] = level_results

                print(f"   📋 Niveau {level}: {len(level_results)} résultats")
                for result in level_results[:2]:  # Top 2
                    chunk = result['chunk']
                    title = chunk.get('metadata', {}).get('section_title', 'Sans titre')
                    print(f"      - {title} (sim: {result['similarity']:.2f})")

        return results


class OntologyPathNavigator:
    """Navigue dans l'ontologie en suivant un chemin conceptuel"""

    def __init__(self, ontology_manager, concept_classifier):
        self.ontology_manager = ontology_manager
        self.concept_classifier = concept_classifier

    def build_search_space(self, inherited_concepts: List[Dict[str, Any]]) -> Set[str]:
        """Construit l'espace de recherche basé sur les concepts hérités - VERSION CORRIGÉE"""
        search_space = set()

        print(f"   🛤️  Construction de l'espace de recherche depuis {len(inherited_concepts)} concepts hérités")

        for concept in inherited_concepts:
            concept_uri = concept.get('concept_uri', '')
            concept_label = concept.get('label', '')

            if not concept_uri:
                continue

            # Ajouter le concept lui-même
            search_space.add(concept_uri)

            # Ajouter ses voisins AVEC contrôle de distance
            neighbors = self.get_concept_neighbors(concept_uri, max_distance=2)

            # CORRECTION : Filtrer les voisins par pertinence
            relevant_neighbors = set()
            for neighbor_uri in neighbors:
                distance = self._calculate_ontology_distance(concept_uri, neighbor_uri)
                if distance <= 2:  # Seulement distance 1 et 2
                    relevant_neighbors.add(neighbor_uri)

            search_space.update(relevant_neighbors)

            print(f"     - {concept_label} → {len(relevant_neighbors)} voisins pertinents ajoutés")

        print(f"   🔍 Espace de recherche final: {len(search_space)} concepts possibles")
        return search_space

    async def detect_concepts_in_path(self, text: str, inherited_concepts: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        """Détecte les concepts en restant dans le chemin ontologique"""

        if not inherited_concepts:
            # Pas de concepts hérités = détection libre (niveau document)
            print(f"   🆓 Détection libre (pas de concepts hérités)")
            return await self.concept_classifier.smart_concept_detection(text)

        # Construire l'espace de recherche
        search_space = self.build_search_space(inherited_concepts)

        # Détection globale
        all_detected = await self.concept_classifier.smart_concept_detection(text)

        # Filtrer par l'espace de recherche
        filtered_concepts = []
        for concept in all_detected:
            concept_uri = concept.get('concept_uri', '')

            if concept_uri in search_space:
                concept['in_ontology_path'] = True
                concept['path_distance'] = self._calculate_path_distance(concept_uri, inherited_concepts)
                filtered_concepts.append(concept)
                print(f"     ✅ Gardé: {concept['label']} (dans le chemin, distance: {concept['path_distance']})")
            else:
                print(f"     ❌ Éliminé: {concept['label']} (hors chemin ontologique)")

        print(f"   ✅ {len(filtered_concepts)} concepts retenus dans le chemin")
        return filtered_concepts

    def _calculate_path_distance(self, concept_uri: str, inherited_concepts: List[Dict[str, Any]]) -> int:
        """Calcule la distance minimale dans le chemin ontologique - VERSION CORRIGÉE"""
        min_distance = 999

        print(f"     🔍 Calcul distance pour {concept_uri.split('#')[-1] if '#' in concept_uri else concept_uri}")

        for inherited in inherited_concepts:
            inherited_uri = inherited.get('concept_uri', '')
            inherited_label = inherited.get('label', '')

            if inherited_uri == concept_uri:
                print(f"       → Distance 0 (même concept que {inherited_label})")
                return 0  # Même concept

            # CORRECTION 1 : Vérifier les relations directes dans l'ontologie
            distance = self._calculate_ontology_distance(concept_uri, inherited_uri)
            if distance < min_distance:
                min_distance = distance
                print(f"       → Distance {distance} via {inherited_label}")

        final_distance = min_distance if min_distance < 999 else 3
        print(f"       ✅ Distance finale: {final_distance}")
        return final_distance

    def _calculate_ontology_distance(self, concept_uri: str, inherited_uri: str) -> int:
        """Calcule la distance réelle dans l'ontologie entre deux concepts"""

        # Distance 1 : Relations directes
        if self._are_directly_related(concept_uri, inherited_uri):
            return 1

        # Distance 2 : Relations via un intermédiaire
        if self._are_related_via_intermediate(concept_uri, inherited_uri):
            return 2

        # Distance 3 : Même domaine
        if self._are_same_domain(concept_uri, inherited_uri):
            return 3

        return 999  # Non lié

    def _are_directly_related(self, concept_uri1: str, concept_uri2: str) -> bool:
        """Vérifie si deux concepts sont directement liés"""

        # 1. Relations hiérarchiques (parent/enfant)
        concept1 = self.ontology_manager.concepts.get(concept_uri1)
        concept2 = self.ontology_manager.concepts.get(concept_uri2)

        if concept1 and concept2:
            # Vérifier si concept1 est parent de concept2
            if hasattr(concept1, 'children'):
                for child in concept1.children:
                    if hasattr(child, 'uri') and child.uri == concept_uri2:
                        return True

            # Vérifier si concept2 est parent de concept1
            if hasattr(concept2, 'children'):
                for child in concept2.children:
                    if hasattr(child, 'uri') and child.uri == concept_uri1:
                        return True

            # Vérifier si concept1 est enfant de concept2
            if hasattr(concept1, 'parents'):
                for parent in concept1.parents:
                    if hasattr(parent, 'uri') and parent.uri == concept_uri2:
                        return True

            # Vérifier si concept2 est enfant de concept1
            if hasattr(concept2, 'parents'):
                for parent in concept2.parents:
                    if hasattr(parent, 'uri') and parent.uri == concept_uri1:
                        return True

        # 2. Relations sémantiques définies dans l'ontologie
        for axiom_type, source, target in self.ontology_manager.axioms:
            if (source == concept_uri1 and target == concept_uri2) or \
                    (source == concept_uri2 and target == concept_uri1):
                return True

        return False

    def _are_related_via_intermediate(self, concept_uri1: str, concept_uri2: str) -> bool:
        """Vérifie si deux concepts sont liés via un intermédiaire"""

        # Récupérer les voisins directs de concept1
        neighbors1 = self.get_concept_neighbors(concept_uri1, max_distance=1)

        # Récupérer les voisins directs de concept2
        neighbors2 = self.get_concept_neighbors(concept_uri2, max_distance=1)

        # Vérifier s'ils ont des voisins communs
        common_neighbors = neighbors1.intersection(neighbors2)

        return len(common_neighbors) > 0

    def _are_same_domain(self, concept_uri1: str, concept_uri2: str) -> bool:
        """Vérifie si deux concepts appartiennent au même domaine"""

        if '#' not in concept_uri1 or '#' not in concept_uri2:
            return False

        domain1 = concept_uri1.rsplit('#', 1)[0]
        domain2 = concept_uri2.rsplit('#', 1)[0]

        return domain1 == domain2

    def get_concept_neighbors(self, concept_uri: str, max_distance: int = 2) -> Set[str]:
        """Récupère les concepts voisins - VERSION AMÉLIORÉE"""
        neighbors = set()

        # 1. Relations hiérarchiques directes
        concept = self.ontology_manager.concepts.get(concept_uri)
        if concept:
            # Parents
            if hasattr(concept, 'parents') and concept.parents:
                for parent in concept.parents:
                    if hasattr(parent, 'uri'):
                        neighbors.add(parent.uri)

            # Enfants
            if hasattr(concept, 'children') and concept.children:
                for child in concept.children:
                    if hasattr(child, 'uri'):
                        neighbors.add(child.uri)

        # 2. Relations sémantiques
        for axiom_type, source, target in self.ontology_manager.axioms:
            if source == concept_uri:
                neighbors.add(target)
            elif target == concept_uri:
                neighbors.add(source)

        # 3. CORRECTION : Concepts du même domaine avec filtrage
        domain = concept_uri.rsplit('#', 1)[0] if '#' in concept_uri else concept_uri
        for uri in self.ontology_manager.concepts.keys():
            if uri.startswith(domain) and uri != concept_uri:
                # NOUVEAU : Limiter aux concepts sémantiquement proches
                if self._are_semantically_close(concept_uri, uri):
                    neighbors.add(uri)

        return neighbors

    def _are_semantically_close(self, concept_uri1: str, concept_uri2: str) -> bool:
        """Vérifie si deux concepts sont sémantiquement proches"""

        # Récupérer les labels
        concept1 = self.ontology_manager.concepts.get(concept_uri1)
        concept2 = self.ontology_manager.concepts.get(concept_uri2)

        if not concept1 or not concept2:
            return False

        label1 = concept1.label.lower() if hasattr(concept1, 'label') else ''
        label2 = concept2.label.lower() if hasattr(concept2, 'label') else ''

        # Règles de proximité sémantique
        proximity_rules = [
            # Distance et longueur d'onde ne sont pas proches
            (['distance', 'meter'], ['wavelength', 'longueur']),
            # Mécanique et optique ne sont pas proches
            (['mechanical', 'mécanique'], ['optical', 'optique']),
            # Mais les unités du même type sont proches
            (['unit', 'unité'], ['unit', 'unité']),
        ]

        for group1, group2 in proximity_rules:
            in_group1 = any(keyword in label1 for keyword in group1)
            in_group2 = any(keyword in label2 for keyword in group2)

            if in_group1 and in_group2:
                return False  # Concepts opposés

        # Par défaut, les concepts du même domaine sont proches
        return True


class ConceptualPathResolver:
    """Résolveur d'ambiguïtés générique basé sur la structure ontologique"""

    def __init__(self, document_concepts: List[Dict[str, Any]], ontology_manager):
        self.document_concepts = document_concepts
        self.ontology_manager = ontology_manager

        # Analyser automatiquement le contexte depuis l'ontologie
        self.document_context = self._extract_ontological_context()
        self.concept_domains = self._build_concept_domain_map()
        self.compatibility_matrix = self._build_compatibility_matrix()

    def _extract_ontological_context(self) -> Dict[str, float]:
        """Extrait le contexte automatiquement depuis la structure ontologique"""

        context_weights = {}

        print(f"🧠 Analyse ontologique du contexte document ({len(self.document_concepts)} concepts)")

        for concept in self.document_concepts:
            concept_uri = concept.get('concept_uri', '')
            confidence = concept.get('confidence', 0)

            if not concept_uri:
                continue

            # 1. Extraire le domaine ontologique
            domain = self._extract_concept_domain(concept_uri)
            if domain:
                context_weights[domain] = context_weights.get(domain, 0) + confidence

            # 2. Analyser les parents pour comprendre la hiérarchie
            parent_domains = self._get_parent_domains(concept_uri)
            for parent_domain in parent_domains:
                # Poids réduit pour les parents
                context_weights[parent_domain] = context_weights.get(parent_domain, 0) + (confidence * 0.5)

            # 3. Analyser les axiomes pour les relations sémantiques
            semantic_domains = self._get_semantic_domains(concept_uri)
            for sem_domain in semantic_domains:
                context_weights[sem_domain] = context_weights.get(sem_domain, 0) + (confidence * 0.3)

        # Normaliser
        total_weight = sum(context_weights.values())
        if total_weight > 0:
            context_weights = {k: v / total_weight for k, v in context_weights.items()}

        print(f"✅ Contexte ontologique extrait: {context_weights}")
        return context_weights

    def _extract_concept_domain(self, concept_uri: str) -> Optional[str]:
        """Extrait le domaine d'un concept depuis l'ontologie"""

        # 1. Domaine depuis l'URI (namespace)
        if '#' in concept_uri:
            base_uri = concept_uri.rsplit('#', 1)[0]
            domain_name = base_uri.split('/')[-1]
            return domain_name

        # 2. Domaine depuis les métadonnées RDF
        concept = self.ontology_manager.concepts.get(concept_uri)
        if concept and hasattr(concept, 'domain'):
            return concept.domain

        # 3. Domaine depuis les parents
        parent_domains = self._get_parent_domains(concept_uri)
        if parent_domains:
            return parent_domains[0]  # Le plus proche

        return None

    def _get_parent_domains(self, concept_uri: str) -> List[str]:
        """Récupère les domaines des concepts parents"""
        domains = []

        concept = self.ontology_manager.concepts.get(concept_uri)
        if concept and hasattr(concept, 'parents'):
            for parent in concept.parents:
                if hasattr(parent, 'uri'):
                    parent_domain = self._extract_concept_domain(parent.uri)
                    if parent_domain and parent_domain not in domains:
                        domains.append(parent_domain)

        return domains

    def _get_semantic_domains(self, concept_uri: str) -> List[str]:
        """Récupère les domaines via les relations sémantiques"""
        domains = []

        for axiom_type, source, target in self.ontology_manager.axioms:
            related_uri = None

            if source == concept_uri:
                related_uri = target
            elif target == concept_uri:
                related_uri = source

            if related_uri:
                related_domain = self._extract_concept_domain(related_uri)
                if related_domain and related_domain not in domains:
                    domains.append(related_domain)

        return domains

    def _build_concept_domain_map(self) -> Dict[str, str]:
        """Construit une map concept_uri -> domaine principal"""
        concept_domain_map = {}

        for concept_uri in self.ontology_manager.concepts.keys():
            domain = self._extract_concept_domain(concept_uri)
            if domain:
                concept_domain_map[concept_uri] = domain

        return concept_domain_map

    def _build_compatibility_matrix(self) -> Dict[Tuple[str, str], float]:
        """Construit une matrice de compatibilité entre domaines"""
        compatibility = {}

        all_domains = set(self.concept_domains.values())

        for domain1 in all_domains:
            for domain2 in all_domains:
                if domain1 == domain2:
                    compatibility[(domain1, domain2)] = 1.0  # Compatible avec soi-même
                else:
                    # Calculer la compatibilité basée sur l'ontologie
                    comp_score = self._calculate_domain_compatibility(domain1, domain2)
                    compatibility[(domain1, domain2)] = comp_score

        print(f"🔗 Matrice de compatibilité construite: {len(compatibility)} relations")
        return compatibility

    def _calculate_domain_compatibility(self, domain1: str, domain2: str) -> float:
        """Calcule la compatibilité entre deux domaines"""

        # 1. Compter les concepts partagés
        domain1_concepts = {uri for uri, domain in self.concept_domains.items() if domain == domain1}
        domain2_concepts = {uri for uri, domain in self.concept_domains.items() if domain == domain2}

        # 2. Relations hiérarchiques entre domaines
        hierarchical_score = self._calculate_hierarchical_compatibility(domain1_concepts, domain2_concepts)

        # 3. Relations sémantiques entre domaines
        semantic_score = self._calculate_semantic_compatibility(domain1_concepts, domain2_concepts)

        # 4. Score final
        compatibility = (hierarchical_score * 0.6) + (semantic_score * 0.4)

        # 5. Bonus/malus selon patterns connus (optionnel, peut être supprimé)
        pattern_bonus = self._calculate_pattern_compatibility(domain1, domain2)
        compatibility += pattern_bonus

        return max(0.0, min(1.0, compatibility))  # Borner entre 0 et 1

    def _calculate_hierarchical_compatibility(self, concepts1: Set[str], concepts2: Set[str]) -> float:
        """Compatibilité basée sur les relations hiérarchiques"""
        if not concepts1 or not concepts2:
            return 0.0

        relation_count = 0
        total_checks = 0

        for concept1 in concepts1:
            for concept2 in concepts2:
                total_checks += 1

                if self._are_hierarchically_related(concept1, concept2):
                    relation_count += 1

        return relation_count / total_checks if total_checks > 0 else 0.0

    def _calculate_semantic_compatibility(self, concepts1: Set[str], concepts2: Set[str]) -> float:
        """Compatibilité basée sur les relations sémantiques"""
        if not concepts1 or not concepts2:
            return 0.0

        relation_count = 0
        total_checks = 0

        for concept1 in concepts1:
            for concept2 in concepts2:
                total_checks += 1

                if self._are_semantically_related(concept1, concept2):
                    relation_count += 1

        return relation_count / total_checks if total_checks > 0 else 0.0

    def _calculate_pattern_compatibility(self, domain1: str, domain2: str) -> float:
        """Bonus/malus basé sur des patterns optionnels (peut être supprimé)"""

        # Cette section est optionnelle et peut être supprimée pour une approche 100% générique
        # Je la garde juste pour montrer comment on pourrait ajouter des règles spécifiques si nécessaire

        domain1_lower = domain1.lower()
        domain2_lower = domain2.lower()

        # Règles génériques basées sur des patterns linguistiques
        opposite_patterns = [
            (['mechanical', 'physique'], ['optical', 'optique']),
            (['temporal', 'time'], ['spatial', 'space']),
            (['macro', 'large'], ['micro', 'small']),
        ]

        for pattern1, pattern2 in opposite_patterns:
            in_pattern1 = any(p in domain1_lower for p in pattern1)
            in_pattern2 = any(p in domain2_lower for p in pattern2)

            if in_pattern1 and in_pattern2:
                return -0.2  # Légère incompatibilité

        return 0.0  # Neutre

    def _are_hierarchically_related(self, concept1_uri: str, concept2_uri: str) -> bool:
        """Vérifie si deux concepts sont hiérarchiquement liés"""
        concept1 = self.ontology_manager.concepts.get(concept1_uri)
        concept2 = self.ontology_manager.concepts.get(concept2_uri)

        if not concept1 or not concept2:
            return False

        # Parent-enfant
        if hasattr(concept1, 'parents'):
            for parent in concept1.parents:
                if hasattr(parent, 'uri') and parent.uri == concept2_uri:
                    return True

        if hasattr(concept2, 'parents'):
            for parent in concept2.parents:
                if hasattr(parent, 'uri') and parent.uri == concept1_uri:
                    return True

        return False

    def _are_semantically_related(self, concept1_uri: str, concept2_uri: str) -> bool:
        """Vérifie si deux concepts sont sémantiquement liés"""
        for axiom_type, source, target in self.ontology_manager.axioms:
            if (source == concept1_uri and target == concept2_uri) or \
                    (source == concept2_uri and target == concept1_uri):
                return True
        return False

    def resolve_ambiguities(self, detected_concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Résout les ambiguïtés de manière générique"""

        if not detected_concepts:
            return []

        # Grouper par label
        grouped = {}
        for concept in detected_concepts:
            label = concept.get('label', '').lower().strip()
            if label not in grouped:
                grouped[label] = []
            grouped[label].append(concept)

        resolved = []

        for label, concepts_group in grouped.items():
            if len(concepts_group) == 1:
                # Pas d'ambiguïté - juste calculer le score contextuel
                concept = concepts_group[0].copy()
                context_score = self._calculate_generic_context_score(concept)
                concept['context_score'] = context_score
                concept['final_confidence'] = concept['confidence'] + (context_score * 0.3)
                resolved.append(concept)

            else:
                # AMBIGUÏTÉ - résolution générique
                print(f"⚠️ AMBIGUÏTÉ sur '{label}' - {len(concepts_group)} variantes")

                best_concept = None
                best_score = -999

                for concept in concepts_group:
                    context_score = self._calculate_generic_context_score(concept)
                    final_score = concept['confidence'] + (context_score * 0.7)

                    print(f"   - {concept.get('concept_uri', '').split('#')[-1]}: "
                          f"base={concept['confidence']:.2f}, "
                          f"context={context_score:.2f}, "
                          f"final={final_score:.2f}")

                    if final_score > best_score:
                        best_score = final_score
                        best_concept = concept.copy()
                        best_concept['context_score'] = context_score
                        best_concept['final_confidence'] = final_score
                        best_concept['ambiguity_resolved'] = True
                        best_concept['resolution_method'] = 'generic_ontological'

                if best_concept:
                    resolved.append(best_concept)
                    chosen = best_concept.get('concept_uri', '').split('#')[-1]
                    print(f"   ✅ CHOIX: {chosen} (score: {best_score:.2f})")

        # Trier par score final
        resolved.sort(key=lambda x: x.get('final_confidence', x['confidence']), reverse=True)
        return resolved

    def _calculate_generic_context_score(self, concept: Dict[str, Any]) -> float:
        """Calcule le score contextuel de manière générique"""

        concept_uri = concept.get('concept_uri', '')
        if not concept_uri:
            return 0.0

        concept_domain = self.concept_domains.get(concept_uri)
        if not concept_domain:
            return 0.0

        print(f"   🎯 Score générique pour {concept.get('label', 'unknown')} (domaine: {concept_domain})")

        total_score = 0.0

        # Pour chaque domaine du contexte document
        for doc_domain, doc_weight in self.document_context.items():
            # Récupérer la compatibilité entre les domaines
            compatibility = self.compatibility_matrix.get((concept_domain, doc_domain), 0.0)

            contribution = doc_weight * compatibility
            total_score += contribution

            print(f"     - vs {doc_domain}: compat={compatibility:.2f}, "
                  f"poids={doc_weight:.2f}, contrib={contribution:.2f}")

        print(f"   ✅ Score final: {total_score:.2f}")
        return total_score


class old_ConceptualPathResolver:
    """Résout les ambiguïtés conceptuelles"""

    def __init__(self, document_concepts: List[Dict[str, Any]]):
        self.document_concepts = document_concepts
        self.document_context = self._extract_document_context()

    def _extract_document_context(self) -> Dict[str, float]:
        """Extrait le contexte sémantique du document"""
        context_weights = {}

        print(f"   🧠 Analyse du contexte document depuis {len(self.document_concepts)} concepts")

        for concept in self.document_concepts:
            label = concept.get('label', '').lower()
            confidence = concept.get('confidence', 0)

            # Patterns de classification
            if any(keyword in label for keyword in
                   ['mécanique', 'mechanical', 'precision', 'dimension', 'tolerance', 'distance']):
                context_weights['mechanical'] = context_weights.get('mechanical', 0) + confidence
                print(f"     - {label} → mechanical (+{confidence:.2f})")

            if any(keyword in label for keyword in
                   ['optique', 'optical', 'longueur d\'onde', 'wavelength', 'spectro', 'lumière', 'light']):
                context_weights['optical'] = context_weights.get('optical', 0) + confidence
                print(f"     - {label} → optical (+{confidence:.2f})")

        # Normaliser
        total_weight = sum(context_weights.values())
        if total_weight > 0:
            context_weights = {k: v / total_weight for k, v in context_weights.items()}

        print(f"   ✅ Contexte document: {context_weights}")
        return context_weights

    def resolve_ambiguities(self, detected_concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Résout les ambiguïtés selon le contexte du document"""

        if not detected_concepts:
            return []

        # Grouper par label pour identifier les ambiguïtés
        grouped = {}
        for concept in detected_concepts:
            label = concept.get('label', '').lower()
            if label not in grouped:
                grouped[label] = []
            grouped[label].append(concept)

        resolved = []

        for label, concepts_group in grouped.items():
            if len(concepts_group) == 1:
                # Pas d'ambiguïté
                resolved.append(concepts_group[0])
            else:
                # Ambiguïté détectée
                print(f"   ⚠️  AMBIGUÏTÉ sur '{label}' - {len(concepts_group)} variantes")

                best_concept = None
                best_score = -999

                for concept in concepts_group:
                    score = self._calculate_context_score(concept)
                    print(f"     - {concept.get('concept_uri', '').split('#')[-1]}: score {score:.2f}")

                    if score > best_score:
                        best_score = score
                        best_concept = concept.copy()
                        best_concept['ambiguity_resolved'] = True
                        best_concept['context_score'] = score

                if best_concept:
                    chosen_variant = best_concept.get('concept_uri', '').split('#')[-1]
                    print(f"   ✅ Choix: {chosen_variant} (score: {best_score:.2f})")
                    resolved.append(best_concept)

        return resolved

    def _calculate_context_score(self, concept: Dict[str, Any]) -> float:
        """Calcule le score contextuel d'un concept"""
        label = concept.get('label', '').lower()
        uri = concept.get('concept_uri', '').lower()
        base_confidence = concept.get('confidence', 0)

        context_bonus = 0.0

        for context_type, weight in self.document_context.items():
            if context_type == 'mechanical':
                if any(keyword in label or keyword in uri for keyword in ['distance', 'mechanical', 'precision']):
                    context_bonus += weight * 0.5
                elif any(keyword in label or keyword in uri for keyword in ['wavelength', 'optical', 'light']):
                    context_bonus -= weight * 0.5

            elif context_type == 'optical':
                if any(keyword in label or keyword in uri for keyword in ['wavelength', 'optical', 'light']):
                    context_bonus += weight * 0.5
                elif any(keyword in label or keyword in uri for keyword in ['distance', 'mechanical', 'precision']):
                    context_bonus -= weight * 0.5

        return base_confidence + context_bonus


class HierarchicalSemanticChunker:
    """Chunker sémantique avec navigation ontologique par chemin"""

    def __init__(self, min_chunk_size: int = 200, max_chunk_size: int = 1000,
                 overlap_sentences: int = 2, ontology_manager=None, concept_classifier=None):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences
        self.ontology_manager = ontology_manager
        self.concept_classifier = concept_classifier

        # Navigateur ontologique
        if ontology_manager and concept_classifier:
            self.path_navigator = OntologyPathNavigator(ontology_manager, concept_classifier)
        else:
            self.path_navigator = None


        self.selective_detector = None

        # Résolveurs de chemin par document
        self.document_path_resolvers = {}  # document_id -> ConceptualPathResolver
        self.document_resolvers = {}

    async def _process_section_selective(self, section: Dict[str, Any],
                                         document_concepts: List[Dict[str, Any]],
                                         document_id: str, filepath: str,
                                         metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Traite une section avec détection sélective"""

        section_title = section.get('title', 'Sans titre')
        section_text = section.get('text', '')

        print(f"🔍 Section: {section_title}")

        if not section_text.strip():
            return []

        # Détection ciblée au niveau section
        section_concepts = await self.selective_detector.detect_section_level_concepts(
            section_text, document_concepts, max_concepts=8
        )

        # Résoudre les ambiguïtés avec le contexte document
        resolver = self.document_resolvers.get(document_id)
        if resolver:
            section_concepts = resolver.resolve_ambiguities(section_concepts)

        if section_concepts:
            print(f"   📋 {len(section_concepts)} concepts retenus:")
            for concept in section_concepts[:3]:
                distance = concept.get('path_distance', 0)
                resolved = " (résolu)" if concept.get('ambiguity_resolved') else ""
                print(f"      - {concept['label']} (dist: {distance}, conf: {concept['confidence']:.2f}){resolved}")

        # Créer le chunk
        chunk = SemanticChunk(
            chunk_id=f"{document_id}-section-{section.get('section_id', 'unknown')}",
            text=section_text,
            start_pos=section.get('start_line', 0),
            end_pos=section.get('end_line', 0),
            level='section',
            concepts=section_concepts,
            inherited_concepts=document_concepts,
            metadata={
                'section_title': section_title,
                'section_level': section.get('level', 0),
                'selective_detection': True,
                'concepts_count': len(section_concepts),
                'inherited_count': len(document_concepts)
            }
        )

        chunk_dict = await self._chunk_to_dict(chunk, document_id, filepath, metadata)
        return [chunk_dict]

    async def create_semantic_chunks(self, text: str, document_id: str, filepath: str,
                                     metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chunking avec détection sélective"""

        filename = Path(filepath).name
        print(f"📄 Chunking sélectif: {filename}")

        if not self.selective_detector:
            print(f"   ⚠️  Pas de détecteur sélectif - fallback")
            return await self._fallback_chunking(text, document_id, filepath, metadata)

        try:
            # ÉTAPE 1: Détection SÉLECTIVE au niveau document
            document_concepts = await self.selective_detector.detect_document_level_concepts(text, max_concepts=5)

            if not document_concepts:
                print(f"   ⚠️  Aucun concept dominant détecté")
                return await self._fallback_chunking(text, document_id, filepath, metadata)

            # ÉTAPE 2: Créer le résolveur contextuel
            resolver = ConceptualPathResolver(document_concepts, self.ontology_manager)
            self.document_resolvers[document_id] = resolver

            # ÉTAPE 3: Traiter les sections avec détection ciblée
            sections = self._extract_sections(text)
            all_chunks = []

            for section in sections:
                section_chunks = await self._process_section_selective(
                    section, document_concepts, document_id, filepath, metadata
                )
                all_chunks.extend(section_chunks)

            # ÉTAPE 4: Optimiser
            optimized_chunks = self._optimize_chunks(all_chunks)

            print(f"✅ Chunking sélectif terminé: {len(optimized_chunks)} chunks")
            return optimized_chunks

        except Exception as e:
            print(f"❌ Erreur chunking sélectif: {e}")
            return await self._fallback_chunking(text, document_id, filepath, metadata)

    def set_ontology_components(self, ontology_manager, concept_classifier):
        """Configure les composants ontologiques après l'initialisation"""
        self.ontology_manager = ontology_manager
        self.concept_classifier = concept_classifier

        # Recréer le navigateur ontologique
        if ontology_manager and concept_classifier:
            self.path_navigator = OntologyPathNavigator(ontology_manager, concept_classifier)
            self.selective_detector = SelectiveConceptDetector(concept_classifier, ontology_manager)
            print("✅ Navigateur ontologique configuré")
        else:
            self.path_navigator = None
            print("⚠️ Composants ontologiques manquants")

    async def old_create_semantic_chunks(self, text: str, document_id: str, filepath: str,
                                     metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Crée des chunks avec navigation ontologique"""

        filename = Path(filepath).name
        print(f"📄 Chunking avec navigation ontologique: {filename}")

        try:
            # ÉTAPE 1: Détection des concepts au niveau document (point de départ)
            document_concepts = await self._detect_document_concepts(text, document_id, filename)

            if not document_concepts:
                print(f"   ⚠️  Aucun concept détecté - fallback")
                return await self._fallback_chunking(text, document_id, filepath, metadata)

            # ÉTAPE 2: Créer le résolveur de chemin pour ce document
            path_resolver = ConceptualPathResolver(document_concepts)
            self.document_path_resolvers[document_id] = path_resolver

            # ÉTAPE 3: Diviser en sections
            sections = self._extract_sections(text)

            # ÉTAPE 4: Traiter chaque section avec navigation ontologique
            all_chunks = []

            for section in sections:
                section_chunks = await self._process_section_with_path(
                    section, document_concepts, document_id, filepath, metadata
                )
                all_chunks.extend(section_chunks)

            # ÉTAPE 5: Optimiser et finaliser
            optimized_chunks = self._optimize_chunks(all_chunks)

            print(f"✅ Chunking terminé: {len(optimized_chunks)} chunks créés")
            return optimized_chunks

        except Exception as e:
            print(f"❌ Erreur chunking: {e}")
            import traceback
            traceback.print_exc()
            return await self._fallback_chunking(text, document_id, filepath, metadata)

    async def _detect_document_concepts(self, text: str, document_id: str, filename: str) -> List[Dict[str, Any]]:
        """Détecte les concepts au niveau document (point de départ)"""

        print(f"🧠 Détection concepts document: {filename}")

        if not self.path_navigator:
            print(f"   ⚠️  Pas de navigateur ontologique")
            return []

        # Détection libre au niveau document
        concepts = await self.path_navigator.detect_concepts_in_path(text, [])

        if concepts:
            print(f"   📋 {len(concepts)} concepts détectés (point de départ)")
            for concept in concepts[:5]:
                print(f"      - {concept['label']} (conf: {concept['confidence']:.2f})")
        else:
            print(f"   ⚠️  Aucun concept détecté au niveau document")

        return concepts

    async def _process_section_with_path(self, section: Dict[str, Any],
                                         document_concepts: List[Dict[str, Any]],
                                         document_id: str, filepath: str,
                                         metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Traite une section avec navigation ontologique"""

        section_title = section.get('title', 'Sans titre')
        section_text = section.get('text', '')

        print(f"🔍 Section: {section_title}")

        if not section_text.strip():
            return []

        # NAVIGATION ONTOLOGIQUE: Détecter les concepts dans le chemin
        section_concepts = await self.path_navigator.detect_concepts_in_path(
            section_text, document_concepts  # ← Héritage depuis le document
        )

        # Résoudre les ambiguïtés avec le contexte du document
        resolver = self.document_path_resolvers.get(document_id)
        if resolver:
            section_concepts = resolver.resolve_ambiguities(section_concepts)

        # Créer le chunk de section
        if section_concepts:
            print(f"   📋 {len(section_concepts)} concepts retenus:")
            for concept in section_concepts[:3]:
                in_path = " (chemin)" if concept.get('in_ontology_path') else ""
                resolved = " (résolu)" if concept.get('ambiguity_resolved') else ""
                print(f"      - {concept['label']} (conf: {concept['confidence']:.2f}){in_path}{resolved}")

        # Créer le chunk sémantique
        chunk = SemanticChunk(
            chunk_id=f"{document_id}-section-{section.get('section_id', 'unknown')}",
            text=section_text,
            start_pos=section.get('start_line', 0),
            end_pos=section.get('end_line', 0),
            level='section',
            concepts=section_concepts,
            inherited_concepts=document_concepts,
            metadata={
                'section_title': section_title,
                'section_level': section.get('level', 0),
                'ontology_path_applied': True,
                'concepts_count': len(section_concepts),
                'inherited_count': len(document_concepts)
            }
        )

        # Convertir en format standard
        chunk_dict = await self._chunk_to_dict(chunk, document_id, filepath, metadata)
        return [chunk_dict]

    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extrait les sections du texte"""
        sections = []
        lines = text.split('\n')
        current_section = {'lines': [], 'title': '', 'level': 0, 'start_line': 0}
        line_number = 0

        for line in lines:
            line_number += 1

            # Détecter les headers
            header_info = self._detect_header(line)

            if header_info:
                # Finaliser section précédente
                if current_section['lines']:
                    current_section['end_line'] = line_number - 1
                    current_section['text'] = '\n'.join(current_section['lines'])
                    current_section['section_id'] = f"section_{len(sections)}"
                    sections.append(current_section)

                # Nouvelle section
                level, title, header_type = header_info
                current_section = {
                    'lines': [line],
                    'title': title,
                    'level': level,
                    'start_line': line_number,
                    'header_type': header_type
                }
            else:
                current_section['lines'].append(line)

        # Dernière section
        if current_section['lines']:
            current_section['end_line'] = line_number
            current_section['text'] = '\n'.join(current_section['lines'])
            current_section['section_id'] = f"section_{len(sections)}"
            sections.append(current_section)

        return sections

    def _detect_header(self, line: str) -> Optional[Tuple[int, str, str]]:
        """Détecte les headers"""
        line = line.strip()

        # Markdown headers
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            title = line.lstrip('#').strip()
            if title:
                return (level, title, 'markdown')

        # Headers numérotés
        match = re.match(r'^((?:\d+\.)+)\s*(.+)$', line)
        if match:
            level = match.group(1).count('.')
            title = match.group(2).strip()
            return (level, title, 'numbered')

        # Headers majuscules
        if line.isupper() and len(line) > 3 and not line.endswith('.'):
            return (1, line, 'capital')

        return None

    async def _chunk_to_dict(self, chunk: SemanticChunk, document_id: str,
                             filepath: str, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convertit un SemanticChunk en dictionnaire"""

        content_hash = hashlib.md5(chunk.text.encode()).hexdigest()

        metadata = {
            **base_metadata,
            **chunk.metadata,
            'detected_concepts': chunk.concepts,
            'inherited_concepts': chunk.inherited_concepts,

            # CRITIQUE : chunk_level doit être dans les métadonnées persistantes
            'chunk_level': chunk.level,  # ← IMPORTANT !

            'content_hash': content_hash,
            'chunk_method': 'ontology_path_navigation',
            'created_at': datetime.now().isoformat(),
            'filepath': filepath,
            'filename': os.path.basename(filepath),
            'start_pos': chunk.start_pos,
            'end_pos': chunk.end_pos,
            'section_title': chunk.metadata.get('section_title', 'Section sans titre'),
            'section_level': chunk.metadata.get('section_level', 0)
        }

        return {
            "id": chunk.chunk_id,
            "document_id": document_id,
            "text": chunk.text,
            "start_pos": chunk.start_pos,
            "end_pos": chunk.end_pos,
            "metadata": metadata
        }

    def _optimize_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimise la collection de chunks"""
        if not chunks:
            return chunks

        # Éliminer doublons
        seen_hashes = set()
        unique_chunks = []

        for chunk in chunks:
            content_hash = chunk['metadata']['content_hash']
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_chunks.append(chunk)

        # Trier par position
        unique_chunks.sort(key=lambda x: x.get('start_pos', 0))

        # Renuméroter
        for i, chunk in enumerate(unique_chunks):
            chunk['id'] = f"{chunk['document_id']}-chunk-{i}"
            chunk['metadata']['chunk_index'] = i
            chunk['metadata']['total_chunks'] = len(unique_chunks)

        return unique_chunks

    async def _fallback_chunking(self, text: str, document_id: str, filepath: str,
                                 metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunking de fallback"""
        print(f"🔄 Chunking de fallback")

        sentences = self._split_into_sentences(text)
        chunks = []

        current_sentences = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size > self.max_chunk_size and current_sentences:
                chunk_text = ' '.join(current_sentences)
                chunk = {
                    "id": f"{document_id}-fallback-{len(chunks)}",
                    "document_id": document_id,
                    "text": chunk_text,
                    "start_pos": text.find(current_sentences[0]),
                    "end_pos": text.find(current_sentences[-1]) + len(current_sentences[-1]),
                    "metadata": {
                        "filepath": filepath,
                        "filename": os.path.basename(filepath),
                        "chunk_method": "fallback",
                        "detected_concepts": [],
                        **(metadata or {})
                    }
                }
                chunks.append(chunk)
                current_sentences = []
                current_size = 0

            current_sentences.append(sentence)
            current_size += sentence_size + 1

        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            chunk = {
                "id": f"{document_id}-fallback-{len(chunks)}",
                "document_id": document_id,
                "text": chunk_text,
                "start_pos": text.find(current_sentences[0]),
                "end_pos": text.find(current_sentences[-1]) + len(current_sentences[-1]),
                "metadata": {
                    "filepath": filepath,
                    "filename": os.path.basename(filepath),
                    "chunk_method": "fallback",
                    "detected_concepts": [],
                    **(metadata or {})
                }
            }
            chunks.append(chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Divise en phrases"""
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]


# Wrapper pour compatibilité
class SemanticChunker:
    """Wrapper pour compatibilité"""

    def __init__(self, min_chunk_size: int = 200, max_chunk_size: int = 1000, overlap_sentences: int = 2):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences
        self.hierarchical_chunker = None

    def set_ontology_manager(self, ontology_manager, concept_classifier):
        """Configure les composants ontologiques"""
        self.hierarchical_chunker = HierarchicalSemanticChunker(
            min_chunk_size=self.min_chunk_size,
            max_chunk_size=self.max_chunk_size,
            overlap_sentences=self.overlap_sentences,
            ontology_manager=ontology_manager,
            concept_classifier=concept_classifier
        )

    def create_semantic_chunks(self, text: str, document_id: str, filepath: str,
                               metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Version synchrone"""
        if self.hierarchical_chunker:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.hierarchical_chunker.create_semantic_chunks(text, document_id, filepath, metadata)
                )
            finally:
                loop.close()
        else:
            return self._simple_chunking(text, document_id, filepath, metadata)

    def _simple_chunking(self, text: str, document_id: str, filepath: str,
                         metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunking simple"""
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            if current_size + len(sentence) > self.max_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "id": f"{document_id}-simple-{len(chunks)}",
                    "document_id": document_id,
                    "text": chunk_text,
                    "start_pos": text.find(current_chunk[0]),
                    "end_pos": text.find(current_chunk[-1]) + len(current_chunk[-1]),
                    "metadata": {
                        "filepath": filepath,
                        "filename": os.path.basename(filepath),
                        "chunk_method": "simple",
                        "detected_concepts": [],
                        **(metadata or {})
                    }
                })
                current_chunk = []
                current_size = 0

            current_chunk.append(sentence)
            current_size += len(sentence) + 1

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "id": f"{document_id}-simple-{len(chunks)}",
                "document_id": document_id,
                "text": chunk_text,
                "start_pos": text.find(current_chunk[0]),
                "end_pos": text.find(current_chunk[-1]) + len(current_chunk[-1]),
                "metadata": {
                    "filepath": filepath,
                    "filename": os.path.basename(filepath),
                    "chunk_method": "simple",
                    "detected_concepts": [],
                    **(metadata or {})
                }
            })

        return chunks


class ConceptualHierarchicalEngine:
    """Moteur de recherche hiérarchique unifié pour texte ET Fortran"""

    def __init__(self, document_store, embedding_provider, entity_manager=None):
        self.document_store = document_store
        self.embedding_provider = embedding_provider
        self.entity_manager = entity_manager

        # Moteur texte existant
        self.text_engine = LevelBasedSearchEngine(document_store, embedding_provider)

    async def intelligent_hierarchical_search(
            self,
            query: str,
            max_per_level: int = 3,
            mode: str = 'auto'  # 'auto', 'text', 'fortran', 'unified'
    ) -> Dict[str, Any]:
        """Recherche hiérarchique intelligente avec auto-détection"""

        print(f"🔍 Recherche hiérarchique intelligente: {query} (mode: {mode})")

        # 1. Analyser le contenu disponible
        content_analysis = await self._analyze_available_content()

        # 2. Déterminer la stratégie de recherche
        if mode == 'auto':
            search_mode = await self._detect_optimal_mode(query, content_analysis)
        else:
            search_mode = mode

        print(f"   🎯 Mode détecté/choisi: {search_mode}")

        # 3. Recherche selon le mode
        if search_mode == 'fortran':
            return await self._fortran_hierarchical_search(query, max_per_level)
        elif search_mode == 'text':
            return await self._text_hierarchical_search(query, max_per_level)
        elif search_mode == 'unified':
            return await self._unified_hierarchical_search(query, max_per_level)
        else:
            # Fallback
            return await self._unified_hierarchical_search(query, max_per_level)

    async def _analyze_available_content(self) -> Dict[str, Any]:
        """Analyse le contenu disponible pour déterminer les types"""

        analysis = {
            'total_chunks': 0,
            'fortran_chunks': 0,
            'text_chunks': 0,
            'conceptual_chunks': 0,
            'content_types': set()
        }

        # Parcourir tous les documents
        for doc_id, chunks in self.document_store.document_chunks.items():
            for chunk in chunks:
                analysis['total_chunks'] += 1

                metadata = chunk.get('metadata', {})
                content_type = metadata.get('content_type', 'unknown')
                analysis['content_types'].add(content_type)

                if content_type == 'fortran':
                    analysis['fortran_chunks'] += 1
                elif metadata.get('chunk_level') in ['document', 'section', 'paragraph']:
                    analysis['text_chunks'] += 1

                if metadata.get('conceptual_level'):
                    analysis['conceptual_chunks'] += 1

        print(f"   📊 Analyse contenu: {analysis['fortran_chunks']} Fortran, "
              f"{analysis['text_chunks']} texte, {analysis['conceptual_chunks']} conceptuel")

        return analysis

    async def _detect_optimal_mode(self, query: str, content_analysis: Dict[str, Any]) -> str:
        """Détecte le mode optimal selon la requête et le contenu"""

        query_lower = query.lower()

        # Patterns Fortran
        fortran_indicators = [
            'function', 'subroutine', 'module', 'program',
            'call', 'use', 'interface', 'type',
            '.f90', '.f95', 'fortran'
        ]

        # Patterns texte
        text_indicators = [
            'section', 'paragraphe', 'chapitre', 'document',
            'résumé', 'introduction', 'conclusion'
        ]

        # Score de la requête
        fortran_score = sum(1 for indicator in fortran_indicators if indicator in query_lower)
        text_score = sum(1 for indicator in text_indicators if indicator in query_lower)

        # Score du contenu disponible
        fortran_ratio = content_analysis['fortran_chunks'] / max(1, content_analysis['total_chunks'])
        text_ratio = content_analysis['text_chunks'] / max(1, content_analysis['total_chunks'])

        # Décision
        if fortran_score > text_score and fortran_ratio > 0.3:
            return 'fortran'
        elif text_score > fortran_score and text_ratio > 0.3:
            return 'text'
        elif content_analysis['conceptual_chunks'] > content_analysis['total_chunks'] * 0.5:
            return 'unified'  # La majorité du contenu est conceptuellement enrichi
        else:
            return 'unified'  # Mode par défaut

    async def _fortran_hierarchical_search(self, query: str, max_per_level: int) -> Dict[str, Any]:
        """Recherche hiérarchique spécialisée Fortran"""

        print(f"   🔧 Recherche Fortran hiérarchique")

        # Niveaux Fortran
        fortran_levels = {
            'container': ['module', 'program'],
            'component': ['function', 'subroutine', 'type_definition', 'interface'],
            'detail': ['internal_function', 'variable_declaration', 'parameter']
        }

        results = {}

        for conceptual_level, native_types in fortran_levels.items():
            level_results = await self._search_fortran_level(query, native_types, max_per_level)
            if level_results:
                results[conceptual_level] = {
                    'results': level_results,
                    'native_types': native_types,
                    'display_name': self._get_fortran_level_display_name(conceptual_level)
                }

        return {
            'search_mode': 'fortran',
            'hierarchical_results': results,
            'total_levels': len(results)
        }

    async def _text_hierarchical_search(self, query: str, max_per_level: int) -> Dict[str, Any]:
        """Recherche hiérarchique texte (existante)"""

        print(f"   📄 Recherche texte hiérarchique")

        # Utiliser le moteur existant
        text_results = await self.text_engine.hierarchical_search(query, max_per_level)

        # Reformater pour uniformité
        unified_results = {}
        level_mapping = {
            'document': 'container',
            'section': 'component',
            'paragraph': 'detail'
        }

        for text_level, results in text_results.items():
            conceptual_level = level_mapping.get(text_level, text_level)
            unified_results[conceptual_level] = {
                'results': results,
                'native_types': [text_level],
                'display_name': self._get_text_level_display_name(text_level)
            }

        return {
            'search_mode': 'text',
            'hierarchical_results': unified_results,
            'total_levels': len(unified_results)
        }

    async def _unified_hierarchical_search(self, query: str, max_per_level: int) -> Dict[str, Any]:
        """Recherche hiérarchique unifiée (texte + Fortran)"""

        print(f"   🌐 Recherche unifiée hiérarchique")

        conceptual_levels = ['container', 'component', 'detail']
        results = {}

        for conceptual_level in conceptual_levels:
            level_results = await self._search_unified_conceptual_level(
                query, conceptual_level, max_per_level
            )
            if level_results:
                results[conceptual_level] = {
                    'results': level_results,
                    'display_name': self._get_unified_level_display_name(conceptual_level)
                }

        return {
            'search_mode': 'unified',
            'hierarchical_results': results,
            'total_levels': len(results)
        }

    async def _search_fortran_level(self, query: str, native_types: List[str], max_results: int) -> List[
        Dict[str, Any]]:
        """Recherche dans un niveau Fortran spécifique"""

        if not self.entity_manager:
            return []

        all_results = []

        # Rechercher par type d'entité
        for entity_type in native_types:
            entities = await self.entity_manager.get_entities_by_type(entity_type)

            for entity in entities:
                # Score de pertinence simple
                score = self._calculate_entity_relevance(query, entity)
                if score > 0.3:  # Seuil de pertinence
                    all_results.append({
                        'entity': entity,
                        'similarity': score,
                        'source_info': {
                            'filename': entity.filename,
                            'start_line': entity.start_line,
                            'end_line': entity.end_line,
                            'entity_type': entity.entity_type,
                            'entity_name': entity.entity_name
                        }
                    })

        # Trier par pertinence
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        return all_results[:max_results]

    async def _search_unified_conceptual_level(self, query: str, conceptual_level: str, max_results: int) -> List[
        Dict[str, Any]]:
        """Recherche dans un niveau conceptuel unifié"""

        # Générer l'embedding de la requête
        query_embedding = await self.embedding_provider.generate_embeddings([query])
        query_embedding = query_embedding[0]

        level_chunks = []

        # Collecter les chunks du niveau conceptuel
        for doc_id, chunks in self.document_store.document_chunks.items():
            for chunk in chunks:
                metadata = chunk.get('metadata', {})
                chunk_conceptual_level = metadata.get('conceptual_level')

                # Fallback vers le mapping texte traditionnel
                if not chunk_conceptual_level:
                    text_level = metadata.get('chunk_level')
                    text_to_conceptual = {
                        'document': 'container',
                        'section': 'component',
                        'paragraph': 'detail'
                    }
                    chunk_conceptual_level = text_to_conceptual.get(text_level)

                if chunk_conceptual_level == conceptual_level:
                    level_chunks.append(chunk)

        if not level_chunks:
            return []

        # Calculer les similarités
        similarities = []
        for chunk in level_chunks:
            chunk_id = chunk['id']
            chunk_embedding = self.document_store.embedding_manager.get_embedding(chunk_id)

            if chunk_embedding is not None:
                similarity = float(np.dot(query_embedding, chunk_embedding))

                metadata = chunk.get('metadata', {})
                similarities.append({
                    'chunk': chunk,
                    'similarity': similarity,
                    'source_info': {
                        'filename': metadata.get('filename', 'Unknown'),
                        'start_line': metadata.get('start_pos', 0),
                        'end_line': metadata.get('end_pos', 0),
                        'section_title': metadata.get('section_title') or metadata.get('entity_name', 'Sans titre'),
                        'content_type': metadata.get('content_type', 'unknown'),
                        'native_type': metadata.get('native_type') or metadata.get('chunk_level', 'unknown')
                    }
                })

        # Trier et limiter
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:max_results]

    def _calculate_entity_relevance(self, query: str, entity) -> float:
        """Calcule la pertinence d'une entité Fortran pour une requête"""
        query_lower = query.lower()
        entity_name_lower = entity.entity_name.lower()

        # Score de base sur le nom
        if query_lower == entity_name_lower:
            return 1.0
        elif query_lower in entity_name_lower:
            return 0.8
        elif entity_name_lower in query_lower:
            return 0.7

        # Score sur les concepts
        concept_score = 0.0
        for concept_label in entity.concepts:
            if query_lower in concept_label.lower():
                concept_score = max(concept_score, 0.6)

        # Score sur les dépendances
        dep_score = 0.0
        for dep in entity.dependencies:
            if query_lower in dep.lower():
                dep_score = max(dep_score, 0.5)

        return max(concept_score, dep_score)

    def _get_fortran_level_display_name(self, conceptual_level: str) -> str:
        """Nom d'affichage pour les niveaux Fortran"""
        names = {
            'container': 'Modules/Programmes',
            'component': 'Fonctions/Subroutines',
            'detail': 'Détails/Variables'
        }
        return names.get(conceptual_level, conceptual_level)

    def _get_text_level_display_name(self, text_level: str) -> str:
        """Nom d'affichage pour les niveaux texte"""
        names = {
            'document': 'Documents',
            'section': 'Sections',
            'paragraph': 'Paragraphes'
        }
        return names.get(text_level, text_level)

    def _get_unified_level_display_name(self, conceptual_level: str) -> str:
        """Nom d'affichage pour les niveaux unifiés"""
        names = {
            'container': 'Conteneurs (Modules/Documents)',
            'component': 'Composants (Fonctions/Sections)',
            'detail': 'Détails (Variables/Paragraphes)'
        }
        return names.get(conceptual_level, conceptual_level)