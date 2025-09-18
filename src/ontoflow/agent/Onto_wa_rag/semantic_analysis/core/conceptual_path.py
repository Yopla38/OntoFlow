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

from semantic_analysis.core.semantic_chunker import GenericContextResolver


@dataclass
class DocumentHierarchy:
    """Représente la hiérarchie conceptuelle d'un document"""
    document_concepts: List[Dict[str, Any]] = field(default_factory=list)
    section_concepts: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    paragraph_concepts: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    sentence_concepts: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def get_inherited_concepts(self, level: str, section_id: str = None, paragraph_id: str = None) -> List[
        Dict[str, Any]]:
        """Récupère les concepts hérités selon la hiérarchie"""
        inherited = []

        # Toujours hériter des concepts du document
        inherited.extend(self.document_concepts)

        if level in ['paragraph', 'sentence'] and section_id and section_id in self.section_concepts:
            inherited.extend(self.section_concepts[section_id])

        if level == 'sentence' and paragraph_id and paragraph_id in self.paragraph_concepts:
            inherited.extend(self.paragraph_concepts[paragraph_id])

        return inherited


@dataclass
class SemanticChunk:
    """Représente un chunk sémantique avec ses métadonnées"""
    chunk_id: str
    text: str
    start_pos: int
    end_pos: int
    level: str  # 'document', 'section', 'paragraph', 'sentence'
    parent_id: Optional[str] = None
    section_id: Optional[str] = None
    paragraph_id: Optional[str] = None

    # Concepts détectés
    direct_concepts: List[Dict[str, Any]] = field(default_factory=list)
    inherited_concepts: List[Dict[str, Any]] = field(default_factory=list)
    all_concepts: List[Dict[str, Any]] = field(default_factory=list)

    # Métadonnées
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Combine les concepts directs et hérités"""
        concept_uris = set()
        self.all_concepts = []

        # Ajouter les concepts directs (priorité)
        for concept in self.direct_concepts:
            uri = concept.get('concept_uri', '')
            if uri and uri not in concept_uris:
                concept_uris.add(uri)
                self.all_concepts.append(concept)

        # Ajouter les concepts hérités (avec poids réduit)
        for concept in self.inherited_concepts:
            uri = concept.get('concept_uri', '')
            if uri and uri not in concept_uris:
                concept_copy = concept.copy()
                concept_copy['confidence'] *= 0.7  # Réduire la confiance des concepts hérités
                concept_copy['inherited'] = True
                concept_uris.add(uri)
                self.all_concepts.append(concept_copy)


class ConceptualPath:
    """Représente un chemin conceptuel dans la hiérarchie"""

    def __init__(self, document_concepts: List[Dict[str, Any]], ontology_manager, concept_classifier):
        self.document_concepts = document_concepts

        # NOUVEAU : Résolveur générique
        self.context_resolver = GenericContextResolver(ontology_manager, concept_classifier)

        # Extraire le contexte de façon générique
        self.context_weights, self.context_clusters = self.context_resolver._extract_document_context(document_concepts)

        print(f"🔧 ConceptualPath générique créé avec {len(document_concepts)} concepts")
        print(f"🔧 {len(self.context_clusters)} clusters contextuels détectés")

    def resolve_ambiguous_concepts(self, detected_concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Résolution d'ambiguïté générique"""

        if not detected_concepts:
            return []

        resolved = []
        grouped_by_label = {}

        # Grouper par label
        for concept in detected_concepts:
            label = concept.get('label', '').lower().strip()
            if label not in grouped_by_label:
                grouped_by_label[label] = []
            grouped_by_label[label].append(concept)

        print(f"   🎯 RÉSOLUTION GÉNÉRIQUE - {len(grouped_by_label)} labels uniques")

        # Traiter chaque groupe
        for label, concepts_group in grouped_by_label.items():
            if len(concepts_group) == 1:
                # Pas d'ambiguïté - calculer le score contextuel
                concept = concepts_group[0].copy()
                context_score = self.context_resolver._get_concept_context_score(
                    concept, self.context_clusters
                )
                concept['context_score'] = context_score
                concept['final_confidence'] = concept['confidence'] + (context_score * 0.3)
                resolved.append(concept)

            else:
                # Ambiguïté - résolution générique
                print(f"       ⚠️ AMBIGUÏTÉ sur '{label}' ({len(concepts_group)} variantes)")

                best_concept = None
                best_score = -999

                for concept in concepts_group:
                    context_score = self.context_resolver._get_concept_context_score(
                        concept, self.context_clusters
                    )

                    # Score final avec fort poids contextuel
                    final_score = concept['confidence'] + (context_score * 0.7)

                    if final_score > best_score:
                        best_score = final_score
                        best_concept = concept.copy()
                        best_concept['context_score'] = context_score
                        best_concept['final_confidence'] = final_score
                        best_concept['ambiguity_resolved'] = True
                        best_concept['resolution_method'] = 'generic_ontology_context'

                if best_concept:
                    resolved.append(best_concept)
                    chosen_uri = best_concept.get('concept_uri', '').split('#')[-1]
                    print(f"       ✅ CHOIX GÉNÉRIQUE: {chosen_uri} (score: {best_score:.2f})")

        # Trier par score final
        resolved.sort(key=lambda x: x.get('final_confidence', x['confidence']), reverse=True)
        return resolved[:10]

    def old__init__(self, document_concepts: List[Dict[str, Any]]):
        self.document_concepts = document_concepts
        self.dominant_domains = self._extract_dominant_domains(document_concepts)
        self.concept_preferences = self._build_concept_preferences(document_concepts)
        self.document_context = self._extract_document_context(document_concepts)

        # NOUVEAU : Debug pour tracer les problèmes
        print(f"🔧 ConceptualPath créé avec {len(document_concepts)} concepts")
        print(f"🔧 Contexte calculé: {self.document_context}")

    def _extract_document_context(self, concepts: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extrait le contexte sémantique du document - VERSION CORRIGÉE"""
        context_weights = {}

        # CORRECTION 1: Analyser d'abord les concepts principaux
        print(f"   🔍 Analyse des concepts pour contexte:")
        for concept in concepts[:10]:  # Top 10 concepts
            label = concept.get('label', '').lower()
            confidence = concept.get('confidence', 0)

            print(f"     - {label} (conf: {confidence:.2f})")

            # CORRECTION 2: Patterns plus précis
            if any(keyword in label for keyword in ['mécanique', 'mechanical', 'precision', 'dimension', 'tolerance']):
                context_weights['mechanical'] = context_weights.get('mechanical', 0) + confidence
                print(f"       → Contribue à 'mechanical': +{confidence:.2f}")

            if any(keyword in label for keyword in
                   ['optique', 'optical', 'longueur d\'onde', 'wavelength', 'spectro', 'lumière', 'light']):
                context_weights['optical'] = context_weights.get('optical', 0) + confidence
                print(f"       → Contribue à 'optical': +{confidence:.2f}")

            # CORRECTION 3: Patterns pour "distance" vs "wavelength"
            if 'distance' in label and 'wavelength' not in label:
                context_weights['mechanical'] = context_weights.get('mechanical', 0) + confidence * 0.8
                print(f"       → Contribue à 'mechanical' (distance): +{confidence * 0.8:.2f}")

        # CORRECTION 4: Normaliser APRÈS debug
        total_weight = sum(context_weights.values())
        if total_weight > 0:
            context_weights = {k: v / total_weight for k, v in context_weights.items()}

        print(f"   ✅ Contexte final: {context_weights}")
        return context_weights

    def _get_concept_context_score(self, concept: Dict[str, Any]) -> float:
        """Calcule le score de contexte pour un concept - VERSION CORRIGÉE"""
        label = concept.get('label', '').lower()
        uri = concept.get('concept_uri', '').lower()

        print(f"   🎯 Calcul score contexte pour '{label}' (URI: {uri})")

        score = 0.0

        # CORRECTION 5: Logique simplifiée et claire
        for context_type, weight in self.document_context.items():
            print(f"     Contexte {context_type}: poids {weight:.2f}")

            if context_type == 'mechanical':
                # CORRECTION 6: Privilégier "distance" en contexte mécanique
                if any(keyword in label or keyword in uri for keyword in
                       ['distance', 'dimension', 'precision', 'tolerance', 'mechanical']):
                    bonus = weight * 1.0
                    score += bonus
                    print(f"       → Bonus mécanique: +{bonus:.2f}")
                # CORRECTION 7: Pénaliser "wavelength" en contexte mécanique
                elif any(keyword in label or keyword in uri for keyword in
                         ['wavelength', 'longueur', 'optical', 'light', 'spectro']):
                    malus = weight * -0.8  # Forte pénalité
                    score += malus
                    print(f"       → Malus mécanique: {malus:.2f}")

            elif context_type == 'optical':
                # CORRECTION 8: Privilégier "wavelength" en contexte optique
                if any(keyword in label or keyword in uri for keyword in
                       ['wavelength', 'longueur', 'optical', 'light', 'spectro']):
                    bonus = weight * 1.0
                    score += bonus
                    print(f"       → Bonus optique: +{bonus:.2f}")
                # CORRECTION 9: Pénaliser "distance" en contexte optique
                elif any(keyword in label or keyword in uri for keyword in
                         ['distance', 'dimension', 'precision', 'tolerance', 'mechanical']):
                    malus = weight * -0.8  # Forte pénalité
                    score += malus
                    print(f"       → Malus optique: {malus:.2f}")

        print(f"   ✅ Score final pour '{label}': {score:.2f}")
        return score

    def resolve_ambiguous_concepts(self, detected_concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Résout l'ambiguïté - VERSION TOTALEMENT CORRIGÉE"""
        print(f"   🎯 RÉSOLUTION D'AMBIGUÏTÉ - {len(detected_concepts)} concepts détectés")
        print(f"   🎯 Contexte document: {self.document_context}")

        if not detected_concepts:
            return []

        resolved = []
        grouped_by_label = {}

        # CORRECTION 10: Grouper par label exact
        for concept in detected_concepts:
            label = concept.get('label', '').lower().strip()
            if label not in grouped_by_label:
                grouped_by_label[label] = []
            grouped_by_label[label].append(concept)

        print(f"   📊 {len(grouped_by_label)} labels uniques trouvés")

        # CORRECTION 11: Traiter chaque groupe d'ambiguïtés
        for label, concepts_group in grouped_by_label.items():
            print(f"     🔍 Traitement du label '{label}' ({len(concepts_group)} variantes)")

            if len(concepts_group) == 1:
                # Pas d'ambiguïté - calculer juste le score contextuel
                concept = concepts_group[0].copy()
                context_score = self._get_concept_context_score(concept)
                concept['context_score'] = context_score
                concept['final_confidence'] = concept['confidence'] + (context_score * 0.3)
                resolved.append(concept)
                print(f"       ✅ Pas d'ambiguïté - Score final: {concept['final_confidence']:.2f}")
            else:
                # CORRECTION 12: Résolution d'ambiguïté stricte
                print(f"       ⚠️ AMBIGUÏTÉ DÉTECTÉE pour '{label}':")

                best_concept = None
                best_score = -999

                for i, concept in enumerate(concepts_group):
                    uri = concept.get('concept_uri', '')
                    base_confidence = concept['confidence']
                    context_score = self._get_concept_context_score(concept)

                    # CORRECTION 13: Score final avec fort poids sur le contexte
                    final_score = base_confidence + (context_score * 0.7)  # 70% de poids au contexte !

                    print(f"         Variante {i + 1}: {uri.split('#')[-1] if '#' in uri else uri}")
                    print(f"           Confiance base: {base_confidence:.2f}")
                    print(f"           Score contexte: {context_score:.2f}")
                    print(f"           Score final: {final_score:.2f}")

                    if final_score > best_score:
                        best_score = final_score
                        best_concept = concept.copy()
                        best_concept['context_score'] = context_score
                        best_concept['final_confidence'] = final_score
                        best_concept['resolution_reason'] = 'context_disambiguation'
                        best_concept['ambiguity_resolved'] = True
                        best_concept['beat_alternatives'] = len(concepts_group) - 1

                if best_concept:
                    resolved.append(best_concept)
                    chosen_uri = best_concept.get('concept_uri', '').split('#')[-1] if '#' in best_concept.get(
                        'concept_uri', '') else 'unknown'
                    print(f"       ✅ CHOIX: {chosen_uri} avec score {best_score:.2f}")
                else:
                    print(f"       ❌ Aucun concept choisi pour '{label}'")

        # CORRECTION 14: Trier par score final
        resolved.sort(key=lambda x: x.get('final_confidence', x['confidence']), reverse=True)

        print(f"   ✅ Résolution terminée: {len(resolved)} concepts")
        return resolved[:10]

    def _extract_dominant_domains(self, concepts: List[Dict[str, Any]]) -> Set[str]:
        """Extrait les domaines dominants du document"""
        domains = set()
        for concept in concepts[:5]:  # Top 5 concepts du document
            uri = concept.get('concept_uri', '')
            if '#' in uri:
                domain = uri.rsplit('#', 1)[0]
                domains.add(domain)
        return domains

    def _build_concept_preferences(self, concepts: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Construit les préférences conceptuelles basées sur les concepts du document"""
        preferences = {}

        for concept in concepts:
            label = concept.get('label', '').lower()
            uri = concept.get('concept_uri', '')

            if label not in preferences or concept['confidence'] > preferences[label]['confidence']:
                preferences[label] = {
                    'preferred_uri': uri,
                    'preferred_domain': uri.rsplit('#', 1)[0] if '#' in uri else '',
                    'confidence': concept['confidence'],
                    'concept_data': concept
                }

        return preferences

