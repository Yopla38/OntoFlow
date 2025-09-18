"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# core/concept_detector.py
"""
Détecteur de concepts sémantiques centralisé.
Extrait et consolide la logique de détection de concepts du système.
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import numpy as np

from ..utils.caching import global_cache

logger = logging.getLogger(__name__)


@dataclass
class DetectedConcept:
    """Concept détecté avec métadonnées"""
    label: str
    confidence: float
    category: str
    concept_uri: Optional[str] = None
    keywords: List[str] = None
    detection_method: str = "rule_based"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'label': self.label,
            'confidence': self.confidence,
            'category': self.category,
            'concept_uri': self.concept_uri,
            'keywords': self.keywords or [],
            'detection_method': self.detection_method
        }


class ConceptDetector:
    """
    Détecteur de concepts sémantiques pour code Fortran.
    Centralise et améliore la logique dispersée dans le système.
    """

    def __init__(self, ontology_manager=None):
        self.ontology_manager = ontology_manager

        # Dictionnaire de concepts par domaine
        self.concept_patterns = {
            # Physique computationnelle
            'molecular_dynamics': {
                'keywords': ['molecular', 'dynamics', 'md', 'particle', 'lennard', 'jones', 'verlet'],
                'category': 'physics',
                'confidence_base': 0.8
            },
            'quantum_mechanics': {
                'keywords': ['quantum', 'wavefunction', 'hamiltonian', 'eigenvalue', 'orbital'],
                'category': 'physics',
                'confidence_base': 0.9
            },
            'thermodynamics': {
                'keywords': ['temperature', 'thermal', 'entropy', 'enthalpy', 'heat'],
                'category': 'physics',
                'confidence_base': 0.7
            },
            'fluid_dynamics': {
                'keywords': ['fluid', 'flow', 'viscosity', 'reynolds', 'navier', 'stokes'],
                'category': 'physics',
                'confidence_base': 0.8
            },
            'electromagnetism': {
                'keywords': ['electric', 'magnetic', 'electromagnetic', 'maxwell', 'field'],
                'category': 'physics',
                'confidence_base': 0.8
            },

            # Mathématiques numériques
            'linear_algebra': {
                'keywords': ['matrix', 'vector', 'eigenvalue', 'decomposition', 'solve'],
                'category': 'mathematics',
                'confidence_base': 0.7
            },
            'numerical_analysis': {
                'keywords': ['numerical', 'discretization', 'approximation', 'error', 'convergence'],
                'category': 'mathematics',
                'confidence_base': 0.8
            },
            'optimization': {
                'keywords': ['optimize', 'minimize', 'maximize', 'gradient', 'constraint'],
                'category': 'mathematics',
                'confidence_base': 0.8
            },
            'statistics': {
                'keywords': ['statistical', 'probability', 'distribution', 'random', 'variance'],
                'category': 'mathematics',
                'confidence_base': 0.7
            },
            'fourier_analysis': {
                'keywords': ['fourier', 'fft', 'transform', 'frequency', 'spectral'],
                'category': 'mathematics',
                'confidence_base': 0.9
            },

            # Algorithmique
            'iterative_methods': {
                'keywords': ['iterate', 'iteration', 'convergence', 'tolerance', 'residual'],
                'category': 'algorithm',
                'confidence_base': 0.7
            },
            'monte_carlo': {
                'keywords': ['monte', 'carlo', 'random', 'sampling', 'simulation'],
                'category': 'algorithm',
                'confidence_base': 0.9
            },
            'finite_elements': {
                'keywords': ['finite', 'element', 'fem', 'mesh', 'discretization'],
                'category': 'algorithm',
                'confidence_base': 0.9
            },
            'parallel_computing': {
                'keywords': ['parallel', 'mpi', 'openmp', 'thread', 'distributed'],
                'category': 'algorithm',
                'confidence_base': 0.8
            },

            # Calcul scientifique
            'density_functional': {
                'keywords': ['density', 'functional', 'dft', 'exchange', 'correlation'],
                'category': 'computational',
                'confidence_base': 0.9
            },
            'molecular_orbital': {
                'keywords': ['molecular', 'orbital', 'lcao', 'basis', 'gaussian'],
                'category': 'computational',
                'confidence_base': 0.8
            },
            'force_field': {
                'keywords': ['force', 'field', 'potential', 'interaction', 'energy'],
                'category': 'computational',
                'confidence_base': 0.7
            }
        }

        # Cache pour éviter la re-détection
        self._detection_cache: Dict[str, List[DetectedConcept]] = {}

    async def detect_concepts(self,
                              text: str,
                              entity_name: str = "",
                              embedding: Optional[np.ndarray] = None) -> List[DetectedConcept]:
        """
        Détecte les concepts dans un texte avec méthodes multiples.

        Args:
            text: Texte à analyser
            entity_name: Nom de l'entité (pour le cache)
            embedding: Embedding du texte (optionnel)

        Returns:
            Liste des concepts détectés
        """
        # Vérifier le cache
        cache_key = f"concepts_{hash(text)}_{entity_name}"
        if cache_key in self._detection_cache:
            return self._detection_cache[cache_key]

        detected_concepts = []

        # 1. Détection par patterns textuels
        pattern_concepts = await self._detect_by_patterns(text, entity_name)
        detected_concepts.extend(pattern_concepts)

        # 2. Détection ontologique si disponible
        if self.ontology_manager and embedding is not None:
            onto_concepts = await self._detect_by_ontology(text, embedding)
            detected_concepts.extend(onto_concepts)

        # 3. Détection par nom d'entité
        name_concepts = await self._detect_by_entity_name(entity_name)
        detected_concepts.extend(name_concepts)

        # 4. Fusionner et dédupliquer
        final_concepts = self._merge_and_deduplicate(detected_concepts)

        # Mettre en cache
        self._detection_cache[cache_key] = final_concepts

        return final_concepts

    async def _detect_by_patterns(self, text: str, entity_name: str) -> List[DetectedConcept]:
        """Détection par patterns textuels"""
        text_lower = text.lower()
        entity_lower = entity_name.lower()
        detected = []

        for concept_name, concept_info in self.concept_patterns.items():
            score = 0
            matched_keywords = []

            # Chercher les mots-clés dans le texte
            for keyword in concept_info['keywords']:
                if keyword in text_lower:
                    score += 1
                    matched_keywords.append(keyword)

                # Bonus si dans le nom de l'entité
                if keyword in entity_lower:
                    score += 2
                    if keyword not in matched_keywords:
                        matched_keywords.append(keyword)

            # Calculer la confiance
            max_score = len(concept_info['keywords']) + 2  # Bonus possible pour nom
            if score > 0:
                confidence = min(1.0,
                                 (score / max_score) * concept_info['confidence_base'])

                if confidence >= 0.3:  # Seuil minimum
                    detected.append(DetectedConcept(
                        label=concept_name,
                        confidence=confidence,
                        category=concept_info['category'],
                        keywords=matched_keywords,
                        detection_method='pattern_matching'
                    ))

        return detected

    async def _detect_by_ontology(self, text: str, embedding: np.ndarray) -> List[DetectedConcept]:
        """Détection par ontologie (si disponible)"""
        if not self.ontology_manager:
            return []

        try:
            classifier = getattr(self.ontology_manager, 'classifier', None)
            if not classifier:
                return []

            concept_classifier = getattr(classifier, 'concept_classifier', None)
            if not concept_classifier:
                return []

            # Utiliser la classification directe par embedding
            if hasattr(concept_classifier, 'classify_embedding_direct'):
                ontology_concepts = await concept_classifier.classify_embedding_direct(
                    embedding=embedding,
                    min_confidence=0.3
                )

                detected = []
                for concept in ontology_concepts:
                    detected.append(DetectedConcept(
                        label=concept.get('label', ''),
                        confidence=concept.get('confidence', 0),
                        category=concept.get('category', 'ontology'),
                        concept_uri=concept.get('concept_uri', ''),
                        detection_method='ontology_classifier'
                    ))

                return detected

        except Exception as e:
            logger.debug(f"Erreur détection ontologique: {e}")

        return []

    async def _detect_by_entity_name(self, entity_name: str) -> List[DetectedConcept]:
        """Détection par analyse du nom d'entité"""
        if not entity_name:
            return []

        name_lower = entity_name.lower()
        detected = []

        # Patterns spécifiques dans les noms
        name_patterns = {
            'initialization': ['init', 'initialize', 'setup'],
            'calculation': ['calc', 'compute', 'evaluate'],
            'solver': ['solve', 'solver', 'solution'],
            'output': ['output', 'write', 'print', 'save'],
            'input': ['input', 'read', 'load'],
            'utility': ['util', 'helper', 'tool'],
            'test': ['test', 'check', 'verify'],
            'main': ['main', 'driver', 'control']
        }

        for concept, patterns in name_patterns.items():
            if any(pattern in name_lower for pattern in patterns):
                detected.append(DetectedConcept(
                    label=concept,
                    confidence=0.8,
                    category='functional',
                    keywords=[p for p in patterns if p in name_lower],
                    detection_method='name_analysis'
                ))

        return detected

    def _merge_and_deduplicate(self, concepts: List[DetectedConcept]) -> List[DetectedConcept]:
        """Fusionne et déduplique les concepts"""
        # Grouper par label
        concept_groups = {}
        for concept in concepts:
            label = concept.label
            if label not in concept_groups:
                concept_groups[label] = []
            concept_groups[label].append(concept)

        # Fusionner chaque groupe
        merged = []
        for label, group in concept_groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Prendre la meilleure confiance et fusionner les métadonnées
                best_concept = max(group, key=lambda c: c.confidence)

                # Fusionner les mots-clés
                all_keywords = set()
                for concept in group:
                    if concept.keywords:
                        all_keywords.update(concept.keywords)

                # Prendre la meilleure méthode de détection
                best_method = "hybrid" if len(
                    set(c.detection_method for c in group)) > 1 else best_concept.detection_method

                merged_concept = DetectedConcept(
                    label=label,
                    confidence=best_concept.confidence,
                    category=best_concept.category,
                    concept_uri=best_concept.concept_uri,
                    keywords=list(all_keywords),
                    detection_method=best_method
                )
                merged.append(merged_concept)

        # Trier par confiance
        merged.sort(key=lambda c: c.confidence, reverse=True)

        return merged[:10]  # Top 10 concepts

    async def detect_concepts_for_entity(self,
                                         entity_code: str,
                                         entity_name: str,
                                         entity_type: str) -> List[DetectedConcept]:
        """
        Détection de concepts spécialisée pour une entité Fortran.
        Ajoute des heuristiques spécifiques au type d'entité.
        """
        # Détection de base
        concepts = await self.detect_concepts(entity_code, entity_name)

        # Ajouter des concepts spécifiques au type
        type_concepts = await self._detect_by_entity_type(entity_type, entity_name, entity_code)
        concepts.extend(type_concepts)

        # Re-fusionner après ajout
        return self._merge_and_deduplicate(concepts)

    async def _detect_by_entity_type(self, entity_type: str, entity_name: str, code: str) -> List[DetectedConcept]:
        """Détection spécifique au type d'entité"""
        detected = []

        if entity_type == 'module':
            # Les modules définissent souvent un domaine
            if 'constants' in entity_name.lower():
                detected.append(DetectedConcept(
                    label='physical_constants',
                    confidence=0.9,
                    category='data',
                    detection_method='type_heuristic'
                ))
            elif 'utils' in entity_name.lower() or 'utilities' in entity_name.lower():
                detected.append(DetectedConcept(
                    label='utility_functions',
                    confidence=0.8,
                    category='functional',
                    detection_method='type_heuristic'
                ))

        elif entity_type in ['subroutine', 'function']:
            # Analyser la signature et le code
            if 'force' in entity_name.lower():
                detected.append(DetectedConcept(
                    label='force_calculation',
                    confidence=0.8,
                    category='physics',
                    detection_method='type_heuristic'
                ))
            elif 'energy' in entity_name.lower():
                detected.append(DetectedConcept(
                    label='energy_calculation',
                    confidence=0.8,
                    category='physics',
                    detection_method='type_heuristic'
                ))

        return detected

    async def get_concept_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de détection de concepts"""
        cache_stats = await global_cache.semantic_contexts.get("concept_stats")
        if cache_stats:
            return cache_stats

        # Calculer les statistiques
        total_patterns = len(self.concept_patterns)
        categories = set(info['category'] for info in self.concept_patterns.values())

        stats = {
            'total_concept_patterns': total_patterns,
            'categories': list(categories),
            'patterns_by_category': {},
            'cache_size': len(self._detection_cache),
            'ontology_available': self.ontology_manager is not None
        }

        # Compter par catégorie
        for category in categories:
            count = sum(1 for info in self.concept_patterns.values()
                        if info['category'] == category)
            stats['patterns_by_category'][category] = count

        # Mettre en cache
        await global_cache.semantic_contexts.set("concept_stats", stats, ttl=3600)

        return stats

    def clear_cache(self):
        """Vide le cache de détection"""
        self._detection_cache.clear()

    async def add_concept_pattern(self,
                                  name: str,
                                  keywords: List[str],
                                  category: str,
                                  confidence_base: float = 0.7):
        """Ajoute un nouveau pattern de concept"""
        self.concept_patterns[name] = {
            'keywords': keywords,
            'category': category,
            'confidence_base': confidence_base
        }

        # Invalider le cache des statistiques
        await global_cache.semantic_contexts.delete("concept_stats")

        logger.info(f"Nouveau pattern de concept ajouté: {name}")


# Instance globale
_global_detector = None


def get_concept_detector(ontology_manager=None) -> ConceptDetector:
    """Factory pour obtenir le détecteur de concepts global"""
    global _global_detector
    if _global_detector is None:
        _global_detector = ConceptDetector(ontology_manager)
    return _global_detector