"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# providers/semantic_context.py
"""
SemanticContextProvider refactorisé utilisant ConceptDetector et EntityManager.
Entités conceptuellement similaires avec analyse sémantique avancée.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

from .base_provider import BaseContextProvider

logger = logging.getLogger(__name__)


class SemanticContextProvider(BaseContextProvider):
    """
    Fournit le contexte sémantique : entités conceptuellement similaires.
    Version refactorisée utilisant ConceptDetector et analyse unifiée.
    """

    async def get_context(self, entity_name: str, max_tokens: int = 2000) -> Dict[str, Any]:
        """Interface principale compatible"""
        return await self.get_semantic_context(entity_name, max_tokens)

    async def get_semantic_context(self, entity_name: str, max_tokens: int = 2000) -> Dict[str, Any]:
        """
        Récupère le contexte sémantique d'une entité.
        Simplifié grâce au ConceptDetector unifié.
        """
        await self._ensure_initialized()

        # Résoudre l'entité
        resolved_entity = await self.resolve_entity(entity_name)
        if not resolved_entity:
            return await self.create_error_context(
                entity_name,
                f"Entity '{entity_name}' not found"
            )

        entity = resolved_entity['entity_object']

        context = {
            'entity': entity_name,
            'type': 'semantic',
            'main_concepts': [],
            'similar_entities': [],
            'concept_clusters': {},
            'algorithmic_patterns': [],
            'semantic_neighbors': [],
            'cross_file_relations': [],
            'concept_analysis': {},
            'tokens_used': 0
        }

        tokens_budget = max_tokens

        # 1. Concepts principaux (utilise ConceptDetector)
        if tokens_budget > 200:
            context['main_concepts'] = await self._get_main_concepts_enriched(entity)
            context['concept_analysis'] = await self._analyze_entity_concepts(entity)
            tokens_budget -= self.calculate_tokens_used(context['main_concepts'])

        # 2. Entités similaires (optimisé avec cache)
        if tokens_budget > 400:
            context['similar_entities'] = await self._find_similar_entities_smart(
                entity, int(tokens_budget * 0.4)
            )
            tokens_budget -= self.calculate_tokens_used(context['similar_entities'])

        # 3. Clusters de concepts (analyse avancée)
        if entity.concepts and tokens_budget > 300:
            context['concept_clusters'] = await self._build_concept_clusters_advanced(
                entity, int(tokens_budget * 0.25)
            )
            tokens_budget -= self.calculate_tokens_used(context['concept_clusters'])

        # 4. Patterns algorithmiques (utilise FortranAnalyzer)
        if tokens_budget > 200:
            context['algorithmic_patterns'] = await self._get_algorithmic_patterns_unified(entity)
            tokens_budget -= self.calculate_tokens_used(context['algorithmic_patterns'])

        # 5. Voisins sémantiques (basé sur concepts)
        if tokens_budget > 150:
            context['semantic_neighbors'] = await self._find_semantic_neighbors_optimized(
                entity, int(tokens_budget * 0.2)
            )
            tokens_budget -= self.calculate_tokens_used(context['semantic_neighbors'])

        # 6. Relations cross-file
        if tokens_budget > 100:
            context['cross_file_relations'] = await self._find_cross_file_relations_smart(
                entity, int(tokens_budget * 0.15)
            )

        context['tokens_used'] = max_tokens - tokens_budget

        return context

    async def _get_main_concepts_enriched(self, entity) -> List[Dict[str, Any]]:
        """
        Concepts principaux enrichis avec ConceptDetector.
        Combine concepts détectés + analyse en temps réel.
        """
        # Concepts existants depuis les métadonnées
        existing_concepts = []
        for concept_data in entity.detected_concepts:
            if isinstance(concept_data, dict):
                existing_concepts.append(concept_data)
            else:
                existing_concepts.append({
                    'label': str(concept_data),
                    'confidence': 0.7,
                    'category': 'metadata',
                    'detection_method': 'metadata'
                })

        # Analyse en temps réel avec ConceptDetector
        entity_code = await self._get_entity_complete_code(entity)
        if entity_code:
            realtime_concepts = await self.concept_detector.detect_concepts_for_entity(
                entity_code, entity.entity_name, entity.entity_type
            )

            # Convertir en format dict
            for concept in realtime_concepts:
                existing_concepts.append(concept.to_dict())

        # Fusionner et dédupliquer
        concept_map = {}
        for concept in existing_concepts:
            label = concept.get('label', '')
            if label:
                if label not in concept_map or concept.get('confidence', 0) > concept_map[label].get('confidence', 0):
                    concept_map[label] = concept

        return list(concept_map.values())[:5]  # Top 5

    async def _analyze_entity_concepts(self, entity) -> Dict[str, Any]:
        """Analyse approfondie des concepts d'une entité"""
        analysis = {
            'total_concepts': len(entity.detected_concepts),
            'concept_categories': defaultdict(int),
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'detection_methods': defaultdict(int),
            'concept_keywords': set()
        }

        for concept_data in entity.detected_concepts:
            if isinstance(concept_data, dict):
                # Catégories
                category = concept_data.get('category', 'unknown')
                analysis['concept_categories'][category] += 1

                # Distribution de confiance
                confidence = concept_data.get('confidence', 0)
                if confidence >= 0.8:
                    analysis['confidence_distribution']['high'] += 1
                elif confidence >= 0.5:
                    analysis['confidence_distribution']['medium'] += 1
                else:
                    analysis['confidence_distribution']['low'] += 1

                # Méthodes de détection
                method = concept_data.get('detection_method', 'unknown')
                analysis['detection_methods'][method] += 1

                # Mots-clés
                keywords = concept_data.get('keywords', [])
                if keywords:
                    analysis['concept_keywords'].update(keywords)

        # Convertir en format sérialisable
        analysis['concept_categories'] = dict(analysis['concept_categories'])
        analysis['detection_methods'] = dict(analysis['detection_methods'])
        analysis['concept_keywords'] = list(analysis['concept_keywords'])

        return analysis

    async def _find_similar_entities_smart(self, entity, max_tokens: int) -> List[Dict[str, Any]]:
        """
        Entités similaires avec analyse intelligente.
        Combine similarité conceptuelle + embeddings si disponibles.
        """
        similar_entities = []

        # 1. Similarité par concepts
        concept_similar = await self._find_similar_by_concepts(entity)
        similar_entities.extend(concept_similar)

        # 2. Similarité par type et nom
        type_similar = await self._find_similar_by_type_and_name(entity)
        similar_entities.extend(type_similar)

        # 3. Similarité par fichier et contexte
        context_similar = await self._find_similar_by_context(entity)
        similar_entities.extend(context_similar)

        # 4. Utiliser RAG si disponible pour embeddings
        if hasattr(self.rag_engine, 'find_similar'):
            embedding_similar = await self._find_similar_by_embeddings(entity, int(max_tokens * 0.3))
            similar_entities.extend(embedding_similar)

        # Fusionner, scorer et dédupliquer
        return await self._merge_and_rank_similar_entities(similar_entities, entity)

    async def _find_similar_by_concepts(self, entity) -> List[Dict[str, Any]]:
        """Similarité basée sur les concepts partagés"""
        similar = []
        entity_concepts = entity.concepts

        if not entity_concepts:
            return similar

        # Parcourir toutes les entités
        for other_entity in self.entity_manager.entities.values():
            if other_entity.entity_id == entity.entity_id:
                continue

            shared_concepts = entity_concepts.intersection(other_entity.concepts)

            if shared_concepts:
                similarity_score = len(shared_concepts) / max(len(entity_concepts), len(other_entity.concepts))

                if similarity_score > 0.2:  # Seuil
                    similar.append({
                        'name': other_entity.entity_name,
                        'type': other_entity.entity_type,
                        'similarity': similarity_score,
                        'similarity_reasons': [f"Shared concepts: {', '.join(list(shared_concepts)[:3])}"],
                        'file': other_entity.filepath,
                        'method': 'concept_similarity'
                    })

        return similar

    async def _find_similar_by_type_and_name(self, entity) -> List[Dict[str, Any]]:
        """Similarité basée sur le type et les patterns de noms"""
        similar = []

        # Entités du même type
        same_type_entities = await self.entity_manager.get_entities_by_type(entity.entity_type)

        for other_entity in same_type_entities:
            if other_entity.entity_id == entity.entity_id:
                continue

            similarity_score = 0
            reasons = []

            # Même type
            similarity_score += 0.3
            reasons.append(f"Same type ({entity.entity_type})")

            # Similarité de nom
            name_similarity = self._calculate_name_similarity(entity.entity_name, other_entity.entity_name)
            similarity_score += name_similarity * 0.4

            if name_similarity > 0.3:
                reasons.append(f"Similar name (score: {name_similarity:.2f})")

            # Même parent
            if entity.parent_entity and entity.parent_entity == other_entity.parent_entity:
                similarity_score += 0.2
                reasons.append(f"Same parent: {entity.parent_entity}")

            if similarity_score > 0.4:  # Seuil
                similar.append({
                    'name': other_entity.entity_name,
                    'type': other_entity.entity_type,
                    'similarity': similarity_score,
                    'similarity_reasons': reasons,
                    'file': other_entity.filepath,
                    'method': 'type_name_similarity'
                })

        return similar

    async def _find_similar_by_context(self, entity) -> List[Dict[str, Any]]:
        """Similarité basée sur le contexte (même fichier, dépendances similaires)"""
        similar = []

        # Entités du même fichier
        if entity.filepath:
            file_entities = await self.entity_manager.get_entities_in_file(entity.filepath)

            for other_entity in file_entities:
                if other_entity.entity_id == entity.entity_id:
                    continue

                similarity_score = 0.4  # Bonus pour même fichier
                reasons = [f"Same file: {entity.filepath.split('/')[-1]}"]

                # Dépendances similaires
                shared_deps = entity.dependencies.intersection(other_entity.dependencies)
                if shared_deps:
                    dep_similarity = len(shared_deps) / max(len(entity.dependencies), len(other_entity.dependencies), 1)
                    similarity_score += dep_similarity * 0.3
                    reasons.append(f"Shared dependencies: {', '.join(list(shared_deps)[:2])}")

                if similarity_score > 0.4:
                    similar.append({
                        'name': other_entity.entity_name,
                        'type': other_entity.entity_type,
                        'similarity': similarity_score,
                        'similarity_reasons': reasons,
                        'file': other_entity.filepath,
                        'method': 'context_similarity'
                    })

        return similar

    async def _find_similar_by_embeddings(self, entity, max_tokens: int) -> List[Dict[str, Any]]:
        """Similarité basée sur les embeddings (si RAG disponible)"""
        similar = []

        try:
            # Récupérer le code de l'entité
            entity_code = await self._get_entity_complete_code(entity)
            if not entity_code or len(entity_code) < 50:
                return similar

            # Utiliser le RAG pour trouver des chunks similaires
            similar_chunks = await self.rag_engine.find_similar(
                entity_code,
                max_results=15,
                min_similarity=0.6
            )

            for chunk_id, similarity_score in similar_chunks:
                # Récupérer l'entité correspondante
                chunk_entity_info = await self.chunk_access.get_entity_info_from_chunk(chunk_id)

                if (chunk_entity_info and
                        chunk_entity_info.get('name') != entity.entity_name):
                    similar.append({
                        'name': chunk_entity_info['name'],
                        'type': chunk_entity_info['type'],
                        'similarity': similarity_score,
                        'similarity_reasons': ['Embedding similarity'],
                        'file': chunk_entity_info.get('filepath', ''),
                        'method': 'embedding_similarity'
                    })

        except Exception as e:
            logger.debug(f"Embedding similarity search failed: {e}")

        return similar

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calcule la similarité entre deux noms"""
        name1_lower = name1.lower()
        name2_lower = name2.lower()

        # Similarité exacte
        if name1_lower == name2_lower:
            return 1.0

        # L'un contient l'autre
        if name1_lower in name2_lower or name2_lower in name1_lower:
            return 0.7

        # Préfixes/suffixes communs
        common_parts = 0
        parts1 = name1_lower.split('_')
        parts2 = name2_lower.split('_')

        for part in parts1:
            if part in parts2:
                common_parts += 1

        if common_parts > 0:
            return min(0.6, common_parts / max(len(parts1), len(parts2)))

        return 0.0

    async def _merge_and_rank_similar_entities(self, similar_entities: List[Dict[str, Any]], entity) -> List[
        Dict[str, Any]]:
        """Fusionne et classe les entités similaires"""
        # Grouper par nom d'entité
        entity_groups = defaultdict(list)
        for sim_entity in similar_entities:
            entity_groups[sim_entity['name']].append(sim_entity)

        # Fusionner chaque groupe
        merged = []
        for entity_name, group in entity_groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Fusionner les scores et raisons
                best_similarity = max(item['similarity'] for item in group)
                all_reasons = []
                all_methods = set()

                for item in group:
                    all_reasons.extend(item['similarity_reasons'])
                    all_methods.add(item['method'])

                merged_entity = {
                    'name': entity_name,
                    'type': group[0]['type'],
                    'similarity': best_similarity,
                    'similarity_reasons': list(set(all_reasons))[:4],  # Top 4 raisons uniques
                    'file': group[0]['file'],
                    'method': 'hybrid' if len(all_methods) > 1 else group[0]['method'],
                    'confidence': best_similarity
                }
                merged.append(merged_entity)

        # Trier par similarité
        merged.sort(key=lambda x: x['similarity'], reverse=True)

        return merged[:8]  # Top 8

    async def _build_concept_clusters_advanced(self, entity, max_tokens: int) -> Dict[str, Any]:
        """
        Clusters de concepts avancés utilisant ConceptDetector.
        """
        if not entity.concepts:
            return {}

        clusters = {
            'primary_concepts': [],
            'concept_network': {},
            'related_entities': {},
            'concept_statistics': {}
        }

        # Analyser les concepts principaux
        main_concepts = await self._get_main_concepts_enriched(entity)
        clusters['primary_concepts'] = main_concepts[:5]

        # Pour chaque concept, trouver des entités liées
        for concept_data in main_concepts:
            concept_label = concept_data.get('label', '')
            if not concept_label:
                continue

            related_entities = await self._find_entities_with_concept_smart(concept_label)
            clusters['related_entities'][concept_label] = related_entities[:5]

        # Statistiques de concepts
        clusters['concept_statistics'] = await self.concept_detector.get_concept_statistics()

        return clusters

    async def _find_entities_with_concept_smart(self, concept_label: str) -> List[Dict[str, Any]]:
        """Trouve les entités partageant un concept (optimisé)"""
        entities_with_concept = []

        for other_entity in self.entity_manager.entities.values():
            if concept_label.lower() in [c.lower() for c in other_entity.concepts]:
                # Calculer la confiance du concept
                concept_confidence = 0.5  # Défaut

                for concept_data in other_entity.detected_concepts:
                    if isinstance(concept_data, dict):
                        if concept_data.get('label', '').lower() == concept_label.lower():
                            concept_confidence = concept_data.get('confidence', 0.5)
                            break

                entities_with_concept.append({
                    'name': other_entity.entity_name,
                    'type': other_entity.entity_type,
                    'confidence': concept_confidence,
                    'file': other_entity.filepath,
                    'is_grouped': other_entity.is_grouped
                })

        # Trier par confiance
        entities_with_concept.sort(key=lambda x: x['confidence'], reverse=True)
        return entities_with_concept

    async def _get_algorithmic_patterns_unified(self, entity) -> List[Dict[str, Any]]:
        """
        Patterns algorithmiques utilisant FortranAnalyzer unifié.
        """
        await self._ensure_initialized()

        patterns_analysis = await self.analyzer.detect_algorithmic_patterns(entity.entity_name)

        if 'error' in patterns_analysis:
            return []

        return patterns_analysis.get('detected_patterns', [])

    async def _find_semantic_neighbors_optimized(self, entity, max_tokens: int) -> List[Dict[str, Any]]:
        """
        Voisins sémantiques optimisés avec scoring intelligent.
        """
        neighbors = []

        # Utiliser les concepts pour un scoring plus précis
        if not entity.concepts:
            return neighbors

        entity_concepts = entity.concepts

        # Parcourir toutes les entités et scorer la proximité sémantique
        for other_entity in self.entity_manager.entities.values():
            if other_entity.entity_id == entity.entity_id:
                continue

            semantic_score = 0
            shared_concepts = []

            # Score basé sur les concepts partagés
            other_concepts = other_entity.concepts
            common_concepts = entity_concepts.intersection(other_concepts)

            if common_concepts:
                concept_similarity = len(common_concepts) / max(len(entity_concepts), len(other_concepts))
                semantic_score += concept_similarity * 0.6
                shared_concepts = list(common_concepts)

            # Bonus pour même type
            if entity.entity_type == other_entity.entity_type:
                semantic_score += 0.2

            # Bonus pour même contexte (fichier ou parent)
            if (entity.filepath == other_entity.filepath or
                    (entity.parent_entity and entity.parent_entity == other_entity.parent_entity)):
                semantic_score += 0.1

            # Bonus pour dépendances similaires
            shared_deps = entity.dependencies.intersection(other_entity.dependencies)
            if shared_deps:
                semantic_score += 0.1

            if semantic_score > 0.3:  # Seuil de proximité sémantique
                neighbors.append({
                    'name': other_entity.entity_name,
                    'type': other_entity.entity_type,
                    'semantic_score': round(semantic_score, 3),
                    'shared_concepts': shared_concepts[:3],
                    'file': other_entity.filepath,
                    'reasoning': self._build_semantic_reasoning(semantic_score, shared_concepts, entity, other_entity)
                })

        # Trier par score sémantique
        neighbors.sort(key=lambda x: x['semantic_score'], reverse=True)
        return neighbors[:8]

    def _build_semantic_reasoning(self, score: float, shared_concepts: List[str], entity, other_entity) -> List[str]:
        """Construit les raisons de la proximité sémantique"""
        reasons = []

        if shared_concepts:
            reasons.append(f"Shared concepts: {', '.join(shared_concepts[:2])}")

        if entity.entity_type == other_entity.entity_type:
            reasons.append(f"Same type: {entity.entity_type}")

        if entity.filepath == other_entity.filepath:
            reasons.append("Same file")
        elif entity.parent_entity == other_entity.parent_entity:
            reasons.append(f"Same parent: {entity.parent_entity}")

        return reasons

    async def _find_cross_file_relations_smart(self, entity, max_tokens: int) -> List[Dict[str, Any]]:
        """
        Relations cross-file optimisées avec EntityManager.
        """
        if not entity.filepath:
            return []

        cross_file_relations = []
        entity_concepts = entity.concepts

        # Chercher dans les autres fichiers
        for other_entity in self.entity_manager.entities.values():
            other_filepath = other_entity.filepath

            # Skip même fichier ou fichier vide
            if not other_filepath or other_filepath == entity.filepath:
                continue

            # Calculer la force de relation
            relation_strength = 0
            relation_types = []

            # Concepts partagés
            shared_concepts = entity_concepts.intersection(other_entity.concepts)
            if shared_concepts:
                concept_strength = len(shared_concepts) / max(len(entity_concepts), len(other_entity.concepts))
                relation_strength += concept_strength * 0.6
                relation_types.append('concept_similarity')

            # Dépendances
            if entity.entity_name in other_entity.dependencies:
                relation_strength += 0.4
                relation_types.append('dependency')
            elif other_entity.entity_name in entity.dependencies:
                relation_strength += 0.4
                relation_types.append('reverse_dependency')

            # Appels de fonctions
            if entity.entity_name in other_entity.called_functions:
                relation_strength += 0.3
                relation_types.append('function_call')
            elif other_entity.entity_name in entity.called_functions:
                relation_strength += 0.3
                relation_types.append('calls_function')

            if relation_strength > 0.2:  # Seuil de relation
                cross_file_relations.append({
                    'entity': other_entity.entity_name,
                    'type': other_entity.entity_type,
                    'file': other_filepath,
                    'relation_strength': round(relation_strength, 3),
                    'relation_types': relation_types,
                    'shared_concepts': list(shared_concepts)[:3] if shared_concepts else []
                })

        # Trier par force de relation
        cross_file_relations.sort(key=lambda x: x['relation_strength'], reverse=True)
        return cross_file_relations[:6]

    # === API simplifiée pour usage externe ===

    async def get_entity_concepts(self, entity_name: str) -> List[Dict[str, Any]]:
        """Récupère uniquement les concepts d'une entité"""
        entity = await self.entity_manager.find_entity(entity_name)
        if entity:
            return await self._get_main_concepts_enriched(entity)
        return []

    async def find_entities_by_concept(self, concept_label: str) -> List[str]:
        """Trouve les entités contenant un concept donné"""
        return [e['name'] for e in await self._find_entities_with_concept_smart(concept_label)]

    async def get_concept_network(self, entity_name: str) -> Dict[str, List[str]]:
        """Réseau de concepts autour d'une entité"""
        entity = await self.entity_manager.find_entity(entity_name)
        if not entity or not entity.concepts:
            return {}

        network = {}
        for concept in entity.concepts:
            related_entities = await self._find_entities_with_concept_smart(concept)
            network[concept] = [e['name'] for e in related_entities[:5]]

        return network