"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# core/fortran_analyzer.py
"""
Analyseur Fortran unifié pour patterns, appels et dépendances.
Centralise toute la logique d'analyse qui était dispersée.
"""

import asyncio
import re
import logging
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict, deque

from .fortran_parser import UnifiedFortranParser, get_unified_parser
from .entity_manager import EntityManager, UnifiedEntity
from ..utils.caching import global_cache
from ..utils.chunk_access import ChunkAccessManager
from ..utils.fortran_patterns import FortranTextProcessor

logger = logging.getLogger(__name__)


class FortranAnalyzer:
    """
    Analyseur Fortran unifié qui combine parsing, analyse des dépendances
    et détection des patterns. Remplace la logique dispersée dans le système.
    """

    def __init__(self, document_store, entity_manager: EntityManager):
        self.document_store = document_store
        self.entity_manager = entity_manager
        self.chunk_access = ChunkAccessManager(document_store)
        self.parser = get_unified_parser("hybrid")
        self.text_processor = FortranTextProcessor(use_hybrid=True)

        # Cache pour analyses coûteuses
        self._call_cache_built = False

        # Patterns algorithmiques pour l'analyse sémantique
        self.algorithmic_patterns = {
            "iterative_solver": ["iteration", "convergence", "tolerance", "residual"],
            "linear_algebra": ["matrix", "vector", "eigenvalue", "decomposition"],
            "numerical_integration": ["quadrature", "integration", "simpson", "gauss"],
            "optimization": ["minimize", "maximize", "gradient", "objective"],
            "differential_equations": ["ode", "pde", "derivative", "boundary"],
            "fft": ["transform", "fourier", "frequency", "spectral"],
            "monte_carlo": ["random", "sampling", "probability", "statistical"],
            "mesh": ["grid", "mesh", "nodes", "elements"],
            "io_operations": ["read", "write", "file", "output"],
            "parallel": ["mpi", "openmp", "parallel", "thread"]
        }

    async def analyze_function_calls(self, entity_name: str) -> Dict[str, Any]:
        """
        Analyse complète des appels de fonctions pour une entité.
        Remplace les méthodes dispersées du système existant.
        """
        entity = await self.entity_manager.find_entity(entity_name)
        if not entity:
            return {'error': f"Entity '{entity_name}' not found"}

        # Construire le cache si nécessaire
        await self._ensure_call_cache_built()

        # Analyser les appels sortants (que cette entité appelle)
        outgoing_calls = await self._analyze_outgoing_calls(entity)

        # Analyser les appels entrants (qui appelle cette entité)
        incoming_calls = await self.entity_manager.find_entity_callers(entity_name)

        # Analyser les patterns dans les appels
        call_patterns = await self._analyze_call_patterns(entity, outgoing_calls)

        return {
            'entity_name': entity_name,
            'outgoing_calls': outgoing_calls,
            'incoming_calls': incoming_calls,
            'call_patterns': call_patterns,
            'call_statistics': {
                'total_outgoing': len(outgoing_calls),
                'total_incoming': len(incoming_calls),
                'unique_targets': len(set(call['name'] for call in outgoing_calls)),
                'call_complexity': self._calculate_call_complexity(outgoing_calls)
            }
        }

    async def _ensure_call_cache_built(self):
        """S'assure que le cache des appels est construit"""
        if self._call_cache_built:
            return

        logger.info("🔗 Construction du cache des appels de fonctions...")

        # Construire l'index des entités disponibles
        all_entities = {}
        for entity in self.entity_manager.entities.values():
            clean_name = re.sub(r'_part_\d+$', '', entity.entity_name)
            all_entities[clean_name.lower()] = (clean_name, entity.entity_type, entity.filepath)
            all_entities[entity.entity_name.lower()] = (entity.entity_name, entity.entity_type, entity.filepath)

        # Analyser chaque chunk pour détecter les appels
        total_calls_found = 0

        for entity in self.entity_manager.entities.values():
            for chunk_info in entity.chunks:
                chunk_id = chunk_info['chunk_id']

                # Récupérer le texte du chunk
                chunk_text = await self.chunk_access.get_chunk_text(chunk_id)
                if not chunk_text:
                    continue

                # Détecter les appels avec le parser hybride
                detected_calls = await self._detect_calls_in_chunk(chunk_text, all_entities)

                if detected_calls:
                    # Mettre à jour l'entité
                    entity.called_functions.update(detected_calls)

                    # Cacher les appels pour ce chunk
                    await global_cache.function_calls.set(chunk_id, list(detected_calls))
                    total_calls_found += len(detected_calls)

        self._call_cache_built = True
        logger.info(f"✅ Cache des appels construit: {total_calls_found} appels détectés")

    async def _detect_calls_in_chunk(self, chunk_text: str, entities_index: Dict[str, Tuple[str, str, str]]) -> Set[
        str]:
        """Détecte les appels de fonctions dans un chunk avec le parser hybride"""
        # Utiliser le parser hybride pour une détection robuste
        calls = self.text_processor.extract_function_calls(chunk_text)

        # Filtrer et valider les appels
        validated_calls = set()
        for call in calls:
            call_lower = call.lower()

            # Vérifier si c'est une entité connue
            if call_lower in entities_index:
                original_name, entity_type, filepath = entities_index[call_lower]
                validated_calls.add(original_name)

        return validated_calls

    async def _analyze_outgoing_calls(self, entity: UnifiedEntity) -> List[Dict[str, Any]]:
        """Analyse les appels sortants d'une entité"""
        outgoing_calls = []

        for call_name in entity.called_functions:
            call_entity = await self.entity_manager.find_entity(call_name)

            call_info = {
                'name': call_name,
                'resolved': call_entity is not None,
                'target_type': call_entity.entity_type if call_entity else 'unknown',
                'target_file': call_entity.filepath if call_entity else 'unknown',
                'is_internal': call_entity.parent_entity == entity.entity_name if call_entity else False,
                'is_external': call_entity.filepath != entity.filepath if call_entity else True
            }

            # Ajouter des métadonnées sur la signature si disponible
            if call_entity and call_entity.signature:
                call_info['signature'] = call_entity.signature

            outgoing_calls.append(call_info)

        return outgoing_calls

    async def _analyze_call_patterns(self, entity: UnifiedEntity, outgoing_calls: List[Dict[str, Any]]) -> Dict[
        str, Any]:
        """Analyse les patterns dans les appels"""
        patterns = {
            'internal_calls': 0,
            'external_calls': 0,
            'cross_file_calls': 0,
            'unresolved_calls': 0,
            'call_types': defaultdict(int),
            'target_files': set()
        }

        for call in outgoing_calls:
            if call['resolved']:
                patterns['call_types'][call['target_type']] += 1
                patterns['target_files'].add(call['target_file'])

                if call['is_internal']:
                    patterns['internal_calls'] += 1
                elif call['is_external']:
                    patterns['external_calls'] += 1
                    if call['target_file'] != entity.filepath:
                        patterns['cross_file_calls'] += 1
            else:
                patterns['unresolved_calls'] += 1

        patterns['target_files'] = list(patterns['target_files'])
        patterns['call_types'] = dict(patterns['call_types'])

        return patterns

    def _calculate_call_complexity(self, outgoing_calls: List[Dict[str, Any]]) -> str:
        """Calcule la complexité basée sur les appels"""
        total_calls = len(outgoing_calls)

        if total_calls == 0:
            return "none"
        elif total_calls <= 3:
            return "low"
        elif total_calls <= 8:
            return "medium"
        else:
            return "high"

    async def analyze_dependencies(self, entity_name: str) -> Dict[str, Any]:
        """
        Analyse complète des dépendances (USE statements + appels + hiérarchie)
        """
        entity = await self.entity_manager.find_entity(entity_name)
        if not entity:
            return {'error': f"Entity '{entity_name}' not found"}

        # Dépendances USE
        use_dependencies = await self._analyze_use_dependencies(entity)

        # Dépendances d'appels
        call_dependencies = await self._analyze_call_dependencies(entity)

        # Dépendances hiérarchiques
        hierarchical_deps = await self._analyze_hierarchical_dependencies(entity)

        # Impact analysis
        impact_analysis = await self._analyze_impact(entity)

        return {
            'entity_name': entity_name,
            'use_dependencies': use_dependencies,
            'call_dependencies': call_dependencies,
            'hierarchical_dependencies': hierarchical_deps,
            'impact_analysis': impact_analysis,
            'dependency_summary': {
                'total_use_deps': len(use_dependencies),
                'total_call_deps': len(call_dependencies),
                'total_children': len(hierarchical_deps.get('children', [])),
                'has_parent': hierarchical_deps.get('parent') is not None,
                'risk_level': impact_analysis.get('risk_level', 'unknown')
            }
        }

    async def _analyze_use_dependencies(self, entity: UnifiedEntity) -> List[Dict[str, Any]]:
        """Analyse les dépendances USE"""
        use_deps = []

        for dep_name in entity.dependencies:
            dep_entity = await self.entity_manager.find_entity(dep_name)

            dep_info = {
                'name': dep_name,
                'resolved': dep_entity is not None,
                'type': dep_entity.entity_type if dep_entity else 'unknown',
                'file': dep_entity.filepath if dep_entity else 'unknown'
            }

            if dep_entity:
                dep_info['public_interface'] = await self._get_public_interface(dep_entity)

            use_deps.append(dep_info)

        return use_deps

    async def _get_public_interface(self, entity: UnifiedEntity) -> List[str]:
        """Récupère l'interface publique d'un module"""
        if entity.entity_type != 'module':
            return []

        # Récupérer le code du module
        if not entity.chunks:
            return []

        # Combiner le texte de tous les chunks
        module_text = ""
        for chunk_info in entity.chunks:
            chunk_text = await self.chunk_access.get_chunk_text(chunk_info['chunk_id'])
            if chunk_text:
                module_text += chunk_text + "\n"

        # Extraire l'interface publique
        return self.text_processor.extract_public_interface(module_text)

    async def _analyze_call_dependencies(self, entity: UnifiedEntity) -> List[Dict[str, Any]]:
        """Analyse les dépendances d'appels"""
        call_deps = []

        for call_name in entity.called_functions:
            call_entity = await self.entity_manager.find_entity(call_name)

            dep_info = {
                'name': call_name,
                'resolved': call_entity is not None,
                'type': call_entity.entity_type if call_entity else 'unknown',
                'file': call_entity.filepath if call_entity else 'unknown',
                'relationship': 'calls'
            }

            if call_entity:
                # Analyser la nature de la dépendance
                if call_entity.filepath == entity.filepath:
                    dep_info['scope'] = 'same_file'
                else:
                    dep_info['scope'] = 'cross_file'

                if call_entity.parent_entity == entity.entity_name:
                    dep_info['relationship'] = 'calls_internal'

            call_deps.append(dep_info)

        return call_deps

    async def _analyze_hierarchical_dependencies(self, entity: UnifiedEntity) -> Dict[str, Any]:
        """Analyse les dépendances hiérarchiques"""
        hierarchical = {
            'parent': None,
            'children': [],
            'siblings': []
        }

        # Parent
        if entity.parent_entity:
            parent_entity = await self.entity_manager.find_entity(entity.parent_entity)
            if parent_entity:
                hierarchical['parent'] = {
                    'name': parent_entity.entity_name,
                    'type': parent_entity.entity_type,
                    'file': parent_entity.filepath
                }

        # Enfants
        children = await self.entity_manager.get_children(entity.entity_id)
        for child in children:
            hierarchical['children'].append({
                'name': child.entity_name,
                'type': child.entity_type,
                'file': child.filepath
            })

        # Frères et sœurs (même parent)
        if entity.parent_entity:
            parent_entity = await self.entity_manager.find_entity(entity.parent_entity)
            if parent_entity:
                all_children = await self.entity_manager.get_children(parent_entity.entity_id)
                for sibling in all_children:
                    if sibling.entity_id != entity.entity_id:
                        hierarchical['siblings'].append({
                            'name': sibling.entity_name,
                            'type': sibling.entity_type
                        })

        return hierarchical

    async def _analyze_impact(self, entity: UnifiedEntity) -> Dict[str, Any]:
        """Analyse l'impact potentiel de modifications"""
        # Trouver qui dépend de cette entité
        callers = await self.entity_manager.find_entity_callers(entity.entity_name)

        # Calculer l'impact indirect (niveau 2)
        indirect_impact = set()
        for caller in callers[:10]:  # Limiter pour éviter l'explosion
            indirect_callers = await self.entity_manager.find_entity_callers(caller['name'])
            indirect_impact.update(c['name'] for c in indirect_callers)

        # Déterminer le niveau de risque
        total_impact = len(callers) + len(indirect_impact)
        if total_impact == 0:
            risk_level = "low"
        elif total_impact <= 5:
            risk_level = "medium"
        else:
            risk_level = "high"

        # Modules affectés
        affected_modules = set()
        for caller in callers:
            caller_entity = await self.entity_manager.find_entity(caller['name'])
            if caller_entity and caller_entity.filepath:
                module_name = self._extract_module_name(caller_entity.filepath)
                if module_name:
                    affected_modules.add(module_name)

        return {
            'risk_level': risk_level,
            'direct_dependents': [c['name'] for c in callers],
            'indirect_dependents': list(indirect_impact),
            'affected_modules': list(affected_modules),
            'total_impact_entities': total_impact,
            'recommendations': self._generate_impact_recommendations(entity, risk_level, total_impact)
        }

    def _extract_module_name(self, filepath: str) -> Optional[str]:
        """Extrait le nom du module depuis le chemin"""
        if not filepath:
            return None

        filename = filepath.split('/')[-1]
        if '.' in filename:
            return '.'.join(filename.split('.')[:-1])
        return filename

    def _generate_impact_recommendations(self, entity: UnifiedEntity,
                                         risk_level: str, total_impact: int) -> List[str]:
        """Génère des recommandations basées sur l'analyse d'impact"""
        recommendations = []

        if risk_level == "low":
            recommendations.append("Modification à faible risque - tests unitaires suffisants")
        elif risk_level == "medium":
            recommendations.append("Risque modéré - tester les dépendants directs")
        else:
            recommendations.append("Risque élevé - tests complets et validation avec l'équipe")

        if entity.entity_type == 'module':
            recommendations.append("Module critique - vérifier l'interface publique")
        elif entity.entity_type in ['function', 'subroutine']:
            recommendations.append("Vérifier la signature et les contrats de la fonction")

        if entity.is_grouped:
            recommendations.append("Entité complexe (multi-chunks) - attention aux effets de bord")

        if total_impact > 10:
            recommendations.append("Impact important - coordonner avec les autres développeurs")

        return recommendations

    async def detect_algorithmic_patterns(self, entity_name: str) -> Dict[str, Any]:
        """
        Détecte les patterns algorithmiques dans une entité
        """
        entity = await self.entity_manager.find_entity(entity_name)
        if not entity:
            return {'error': f"Entity '{entity_name}' not found"}

        # Récupérer le code de l'entité
        entity_code = await self._get_entity_full_code(entity)

        # Détecter les patterns
        detected_patterns = []
        code_lower = entity_code.lower()

        for pattern_name, keywords in self.algorithmic_patterns.items():
            score = 0
            matched_keywords = []

            # Chercher les mots-clés dans le code
            for keyword in keywords:
                if keyword in code_lower:
                    score += 2
                    matched_keywords.append(keyword)

            # Chercher dans les concepts détectés
            concept_labels = [c.lower() for c in entity.concepts]
            for keyword in keywords:
                if any(keyword in concept for concept in concept_labels):
                    score += 3
                    if keyword not in matched_keywords:
                        matched_keywords.append(keyword)

            # Si suffisamment de correspondances
            if score >= 3:
                detected_patterns.append({
                    'pattern': pattern_name,
                    'score': score,
                    'matched_keywords': matched_keywords,
                    'description': self._get_pattern_description(pattern_name),
                    'confidence': min(1.0, score / 10)
                })

        # Trier par score
        detected_patterns.sort(key=lambda x: x['score'], reverse=True)

        return {
            'entity_name': entity_name,
            'detected_patterns': detected_patterns[:5],  # Top 5
            'pattern_summary': {
                'total_patterns': len(detected_patterns),
                'highest_confidence': detected_patterns[0]['confidence'] if detected_patterns else 0,
                'primary_pattern': detected_patterns[0]['pattern'] if detected_patterns else None
            }
        }

    async def _get_entity_full_code(self, entity: UnifiedEntity) -> str:
        """Récupère le code complet d'une entité"""
        code_parts = []

        for chunk_info in sorted(entity.chunks, key=lambda x: x.get('part_index', 0)):
            chunk_text = await self.chunk_access.get_chunk_text(chunk_info['chunk_id'])
            if chunk_text:
                code_parts.append(chunk_text)

        return '\n'.join(code_parts)

    def _get_pattern_description(self, pattern_name: str) -> str:
        """Retourne une description du pattern algorithmique"""
        descriptions = {
            "iterative_solver": "Algorithme itératif avec convergence",
            "linear_algebra": "Opérations d'algèbre linéaire",
            "numerical_integration": "Intégration numérique",
            "optimization": "Algorithme d'optimisation",
            "differential_equations": "Résolution d'équations différentielles",
            "fft": "Transformée de Fourier rapide",
            "monte_carlo": "Méthode Monte Carlo",
            "mesh": "Opérations sur maillage",
            "io_operations": "Opérations d'entrée/sortie",
            "parallel": "Calcul parallèle"
        }
        return descriptions.get(pattern_name, f"Pattern {pattern_name}")

    async def build_dependency_graph(self, entity_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Construit un graphe de dépendances complet pour une entité.
        Utilise EntityManager mais ajoute l'analyse détaillée.
        """
        # Utiliser EntityManager pour la structure de base
        base_graph = await self.entity_manager.get_dependency_graph(entity_name, max_depth)

        if not base_graph:
            return {}

        # Enrichir avec l'analyse détaillée
        enriched_graph = base_graph.copy()
        enriched_graph['analysis'] = {}

        # Analyser chaque nœud
        for node_name in base_graph.get('nodes', {}):
            node_analysis = await self.analyze_function_calls(node_name)
            if 'error' not in node_analysis:
                enriched_graph['analysis'][node_name] = {
                    'call_complexity': node_analysis['call_statistics']['call_complexity'],
                    'outgoing_calls_count': node_analysis['call_statistics']['total_outgoing'],
                    'incoming_calls_count': node_analysis['call_statistics']['total_incoming']
                }

        # Analyser les relations inter-fichiers
        enriched_graph['cross_file_relations'] = await self._analyze_cross_file_relations(base_graph)

        return enriched_graph

    async def _analyze_cross_file_relations(self, graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyse les relations entre fichiers"""
        file_relations = defaultdict(lambda: defaultdict(int))

        for edge in graph.get('edges', []):
            from_entity = await self.entity_manager.find_entity(edge['from'])
            to_entity = await self.entity_manager.find_entity(edge['to'])

            if (from_entity and to_entity and
                    from_entity.filepath != to_entity.filepath):

                from_file = self._extract_module_name(from_entity.filepath)
                to_file = self._extract_module_name(to_entity.filepath)

                if from_file and to_file:
                    file_relations[from_file][to_file] += 1

        # Convertir en liste
        relations = []
        for from_file, targets in file_relations.items():
            for to_file, count in targets.items():
                relations.append({
                    'from_file': from_file,
                    'to_file': to_file,
                    'relation_count': count,
                    'relation_strength': min(1.0, count / 5)  # Normaliser
                })

        return sorted(relations, key=lambda x: x['relation_count'], reverse=True)

    def get_analysis_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'analyseur"""
        cache_stats = global_cache.get_all_stats()

        return {
            'cache_built': self._call_cache_built,
            'parser_stats': self.parser.get_parsing_stats(),
            'entity_stats': self.entity_manager.get_stats(),
            'cache_stats': {
                'function_calls': cache_stats['function_calls'].__dict__,
                'dependency_graphs': cache_stats['dependency_graphs'].__dict__
            }
        }


# Instance globale
async def get_fortran_analyzer(document_store, entity_manager: EntityManager = None) -> FortranAnalyzer:
    """Factory pour obtenir l'analyseur Fortran"""
    if entity_manager is None:
        from .entity_manager import get_entity_manager
        entity_manager = await get_entity_manager(document_store)

    return FortranAnalyzer(document_store, entity_manager)