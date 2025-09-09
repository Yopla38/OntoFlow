"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# providers/global_context.py
"""
GlobalContextProvider refactorisé utilisant FortranAnalyzer et EntityManager.
Vue d'ensemble du projet et graphe de dépendances optimisés.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .base_provider import BaseContextProvider

logger = logging.getLogger(__name__)


class GlobalContextProvider(BaseContextProvider):
    """
    Fournit le contexte global : vue d'ensemble du projet et graphe de dépendances.
    Version refactorisée utilisant les composants unifiés.
    """

    async def get_context(self, entity_name: str, max_tokens: int = 3000) -> Dict[str, Any]:
        """Interface principale compatible"""
        return await self.get_global_context(entity_name, max_tokens)

    async def get_global_context(self, entity_name: str, max_tokens: int = 3000) -> Dict[str, Any]:
        """
        Récupère le contexte global d'une entité.
        Simplifié grâce à EntityManager et FortranAnalyzer.
        """
        await self._ensure_initialized()

        # Vérifier que l'entité existe
        resolved_entity = await self.resolve_entity(entity_name)
        if not resolved_entity:
            return await self.create_error_context(
                entity_name,
                f"Entity '{entity_name}' not found"
            )

        entity = resolved_entity['entity_object']

        context = {
            'entity': entity_name,
            'type': 'global',
            'project_overview': {},
            'module_hierarchy': {},
            'dependency_graph': {},
            'architectural_patterns': [],
            'related_modules': [],
            'impact_analysis': {},
            'tokens_used': 0
        }

        tokens_budget = max_tokens

        # 1. Vue d'ensemble du projet (utilise EntityManager)
        context['project_overview'] = await self._get_project_overview_optimized(int(tokens_budget * 0.15))
        tokens_budget -= self.calculate_tokens_used(context['project_overview'])

        # 2. Hiérarchie des modules (cache intelligent)
        context['module_hierarchy'] = await self._get_module_hierarchy_cached(int(tokens_budget * 0.2))
        tokens_budget -= self.calculate_tokens_used(context['module_hierarchy'])

        # 3. Graphe de dépendances (utilise FortranAnalyzer)
        if tokens_budget > 500:
            context['dependency_graph'] = await self.analyzer.build_dependency_graph(
                entity_name, max_depth=2
            )
            tokens_budget -= self.calculate_tokens_used(context['dependency_graph'])

        # 4. Analyse d'impact (utilise FortranAnalyzer)
        if tokens_budget > 300:
            impact_analysis = await self.analyzer.analyze_dependencies(entity_name)
            if 'error' not in impact_analysis:
                context['impact_analysis'] = impact_analysis['impact_analysis']
                tokens_budget -= self.calculate_tokens_used(context['impact_analysis'])

        # 5. Modules liés sémantiquement
        if tokens_budget > 200:
            context['related_modules'] = await self._find_semantically_related_modules_optimized(
                entity, int(tokens_budget * 0.15)
            )

        # 6. Patterns architecturaux
        if tokens_budget > 100:
            context['architectural_patterns'] = await self._identify_architectural_patterns(entity)

        context['tokens_used'] = max_tokens - tokens_budget

        return context

    async def _get_project_overview_optimized(self, max_tokens: int) -> Dict[str, Any]:
        """Vue d'ensemble du projet optimisée - VERSION CORRIGÉE"""
        cache_key = "project_overview_stats"

        # Utiliser le cache pour éviter les recalculs
        try:
            cached_overview = await self.cache.entities.get(cache_key)
            if cached_overview:
                return cached_overview
        except Exception as e:
            logger.debug(f"Erreur cache overview: {e}")

        # Récupérer les statistiques depuis EntityManager
        try:
            entity_stats = self.entity_manager.get_stats()
        except Exception as e:
            logger.debug(f"Erreur stats EntityManager: {e}")
            entity_stats = {'total_entities': 0, 'total_chunks': 0}

        # Analyser les modules principaux - CORRECTION
        try:
            modules = await self._get_entities_by_type_safe('module')
            main_modules = await self._identify_main_modules_smart(modules)
        except Exception as e:
            logger.debug(f"Erreur analyse modules principaux: {e}")
            main_modules = []
            modules = []

        # Analyser la structure des fichiers
        try:
            file_analysis = await self._analyze_file_structure_optimized()
        except Exception as e:
            logger.debug(f"Erreur analyse fichiers: {e}")
            file_analysis = {'total_files': 0, 'files': []}

        overview = {
            'statistics': entity_stats,
            'main_modules': main_modules,
            'file_structure': file_analysis,
            'architectural_style': await self._identify_architectural_style_smart(modules),
            'quality_metrics': {
                'grouped_entities_ratio': entity_stats.get('grouped_entities', 0) / max(1, entity_stats.get(
                    'total_entities', 1)),
                'incomplete_entities_ratio': entity_stats.get('incomplete_entities', 0) / max(1, entity_stats.get(
                    'total_entities', 1)),
                'compression_ratio': entity_stats.get('compression_ratio', 1.0)
            }
        }

        # Mettre en cache pour 1h
        try:
            await self.cache.entities.set(cache_key, overview, ttl=3600)
        except Exception as e:
            logger.debug(f"Erreur mise en cache overview: {e}")

        return overview

    async def _get_module_hierarchy_cached(self, max_tokens: int) -> Dict[str, Any]:
        """Hiérarchie des modules avec cache intelligent - VERSION CORRIGÉE"""
        cache_key = "module_hierarchy_complete"

        cached_hierarchy = await self.cache.dependency_graphs.get(cache_key)
        if cached_hierarchy:
            return cached_hierarchy

        modules = await self.entity_manager.get_entities_by_type('module')

        hierarchy = {
            'modules': {},
            'dependency_tree': {},
            'circular_dependencies': [],
            'independent_modules': [],
            'module_statistics': {}
        }

        # Construire les informations de modules - CORRECTION ICI
        for module in modules:
            try:
                # CORRECTION: Utiliser await et convertir en liste
                children = await self.entity_manager.get_children(module.entity_id)
                children_names = [child.entity_name for child in children]  # Convertir en liste de noms

                hierarchy['modules'][module.entity_name] = {
                    'dependencies': list(getattr(module, 'dependencies', set())),
                    'children': children_names,  # Liste de noms, pas d'objets
                    'filepath': getattr(module, 'filepath', ''),
                    'concepts': list(getattr(module, 'concepts', set()))[:3],
                    'is_grouped': getattr(module, 'is_grouped', False),
                    'confidence': getattr(module, 'confidence', 1.0)
                }
            except Exception as e:
                logger.debug(f"Erreur traitement module {getattr(module, 'entity_name', 'unknown')}: {e}")
                # Module avec erreur - données minimales
                hierarchy['modules'][getattr(module, 'entity_name', 'unknown')] = {
                    'dependencies': [],
                    'children': [],
                    'filepath': '',
                    'concepts': [],
                    'is_grouped': False,
                    'confidence': 0.5
                }

        # Détecter les dépendances circulaires (optimisé)
        try:
            hierarchy['circular_dependencies'] = await self._detect_circular_dependencies_smart(modules)
        except Exception as e:
            logger.debug(f"Erreur détection dépendances circulaires: {e}")
            hierarchy['circular_dependencies'] = []

        # Modules indépendants
        try:
            hierarchy['independent_modules'] = [
                getattr(module, 'entity_name', 'unknown') for module in modules
                if not getattr(module, 'dependencies', set())
            ]
        except Exception as e:
            logger.debug(f"Erreur calcul modules indépendants: {e}")
            hierarchy['independent_modules'] = []

        # Statistiques par module - CORRECTION: Calcul sécurisé
        try:
            total_modules = len(modules)
            if total_modules > 0:
                avg_deps = sum(len(getattr(m, 'dependencies', set())) for m in modules) / total_modules

                # Calcul sécurisé des enfants moyens
                children_counts = []
                for module in modules:
                    try:
                        children = await self.entity_manager.get_children(module.entity_id)
                        children_counts.append(len(children))
                    except Exception:
                        children_counts.append(0)

                avg_children = sum(children_counts) / len(children_counts) if children_counts else 0
            else:
                avg_deps = 0
                avg_children = 0

            hierarchy['module_statistics'] = {
                'total_modules': total_modules,
                'avg_dependencies': avg_deps,
                'avg_children': avg_children
            }
        except Exception as e:
            logger.debug(f"Erreur calcul statistiques: {e}")
            hierarchy['module_statistics'] = {
                'total_modules': len(modules),
                'avg_dependencies': 0,
                'avg_children': 0
            }

        # Cache pour 30 minutes
        try:
            await self.cache.dependency_graphs.set(cache_key, hierarchy, ttl=1800)
        except Exception as e:
            logger.debug(f"Erreur mise en cache: {e}")

        return hierarchy

    async def _identify_main_modules_smart(self, modules: List) -> List[Dict[str, Any]]:
        """Identifie les modules principaux - VERSION FINALE CORRIGÉE"""
        main_modules = []

        for module in modules:
            score = 0

            try:
                # CORRECTION: Await et gestion d'erreurs robuste
                try:
                    children = await self.entity_manager.get_children(module.entity_id)
                    children_count = len(children) if children else 0
                except Exception as e:
                    logger.debug(f"Erreur get_children pour {getattr(module, 'entity_name', 'unknown')}: {e}")
                    children_count = 0

                deps_count = len(getattr(module, 'dependencies', set()))

                # Score basé sur la complexité
                score += children_count * 2  # Bonus pour entités contenues
                score += max(0, 5 - deps_count)  # Bonus pour peu de dépendances

                # Bonus pour noms suggestifs
                entity_name = getattr(module, 'entity_name', '')
                name_lower = entity_name.lower()
                if any(keyword in name_lower for keyword in ['main', 'core', 'base', 'util', 'driver']):
                    score += 5

                # Bonus pour concepts importants
                concepts = getattr(module, 'concepts', set())
                important_concepts = {'initialization', 'main', 'driver', 'control'}
                if concepts.intersection(important_concepts):
                    score += 3

                # Bonus pour regroupement (complexité)
                is_grouped = getattr(module, 'is_grouped', False)
                if is_grouped:
                    score += 2

                main_modules.append({
                    'name': entity_name,
                    'score': score,
                    'children_count': children_count,
                    'dependencies_count': deps_count,
                    'filepath': getattr(module, 'filepath', ''),
                    'concepts': list(concepts)[:3],
                    'is_grouped': is_grouped,
                    'confidence': getattr(module, 'confidence', 1.0)
                })

            except Exception as e:
                logger.debug(f"Erreur analyse module {getattr(module, 'entity_name', 'unknown')}: {e}")
                # Module avec erreur, score minimal
                main_modules.append({
                    'name': getattr(module, 'entity_name', 'unknown'),
                    'score': 0,
                    'children_count': 0,
                    'dependencies_count': 0,
                    'filepath': getattr(module, 'filepath', ''),
                    'concepts': [],
                    'is_grouped': False,
                    'confidence': 0.5,
                    'error': str(e)
                })

        # Trier par score et retourner le top 5
        main_modules.sort(key=lambda x: x['score'], reverse=True)
        return main_modules[:5]

    async def _analyze_file_structure_optimized(self) -> Dict[str, Any]:
        """Analyse de structure de fichiers optimisée - VERSION CORRIGÉE"""
        file_to_entities = self.entity_manager.file_to_entities

        file_analysis = {
            'total_files': len(file_to_entities),
            'files': list(file_to_entities.keys()),
            'file_distribution': {},
            'largest_files': [],
            'file_types': defaultdict(int)
        }

        # Analyser la distribution
        for filepath, entity_ids in file_to_entities.items():
            entity_count = len(entity_ids)
            file_analysis['file_distribution'][filepath] = entity_count

            # Type de fichier
            ext = filepath.split('.')[-1].lower() if '.' in filepath else 'no_extension'
            file_analysis['file_types'][ext] += 1

        # Fichiers les plus volumineux
        largest = sorted(
            file_analysis['file_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        file_analysis['largest_files'] = [
            {'filepath': fp, 'entity_count': count}
            for fp, count in largest
        ]

        file_analysis['file_types'] = dict(file_analysis['file_types'])

        return file_analysis

    async def _identify_architectural_style_smart(self, modules: List) -> str:
        """Identifie le style architectural avec analyse enrichie - VERSION CORRIGÉE"""
        total_modules = len(modules)

        if total_modules <= 1:
            return "monolithic"

        # Analyser les patterns avec EntityManager
        total_dependencies = sum(len(getattr(module, 'dependencies', set())) for module in modules)
        avg_dependencies = total_dependencies / total_modules if total_modules > 0 else 0

        # Analyser la distribution des enfants - CORRECTION ICI
        children_distribution = []

        try:
            for module in modules:
                # CORRECTION: Utiliser await et gérer les erreurs
                try:
                    children = await self.entity_manager.get_children(module.entity_id)
                    children_distribution.append(len(children))
                except Exception as e:
                    logger.debug(f"Erreur get_children pour {getattr(module, 'entity_name', 'unknown')}: {e}")
                    children_distribution.append(0)  # Fallback

            avg_children = sum(children_distribution) / len(children_distribution) if children_distribution else 0

            # Modules indépendants
            independent_count = sum(1 for module in modules if not getattr(module, 'dependencies', set()))
            independence_ratio = independent_count / total_modules

            # Heuristiques enrichies
            if avg_dependencies < 1 and independence_ratio > 0.5:
                return "loosely_coupled"
            elif avg_dependencies > 3:
                return "highly_coupled"
            elif independence_ratio > 0.3:
                return "layered_with_utilities"
            elif avg_children > 5:
                return "hierarchical_modular"
            else:
                return "modular"

        except Exception as e:
            logger.error(f"Erreur analyse style architectural: {e}")
            return "unknown"

    async def _detect_circular_dependencies_smart(self, modules: List) -> List[List[str]]:
        """
        Détection de dépendances circulaires optimisée.
        Utilise EntityManager au lieu de re-parser.
        """
        # Construire le graphe depuis EntityManager
        graph = {}
        for module in modules:
            graph[module.entity_name] = list(module.dependencies)

        # DFS pour détecter les cycles (algorithme classique)
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node, path):
            if node in rec_stack:
                # Cycle détecté
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor in graph:  # Le module existe
                    dfs(neighbor, path[:])  # Copie du chemin

            rec_stack.remove(node)
            path.pop()

        for module_name in graph:
            if module_name not in visited:
                dfs(module_name, [])

        return cycles

    async def _find_semantically_related_modules_optimized(self, entity, max_tokens: int) -> List[Dict[str, Any]]:
        """
        Modules sémantiquement liés via concepts et embeddings.
        Version optimisée utilisant ConceptDetector.
        """
        related_modules = []

        try:
            # Utiliser les concepts de l'entité pour trouver des modules similaires
            entity_concepts = entity.concepts

            if not entity_concepts:
                return []

            # Chercher des modules avec des concepts similaires
            all_modules = await self.entity_manager.get_entities_by_type('module')

            for module in all_modules:
                if module.entity_id == entity.entity_id:
                    continue

                # Calculer la similarité conceptuelle
                common_concepts = entity_concepts.intersection(module.concepts)

                if common_concepts:
                    similarity_score = len(common_concepts) / max(len(entity_concepts), len(module.concepts))

                    if similarity_score > 0.2:  # Seuil de similarité
                        related_modules.append({
                            'module': module.entity_name,
                            'similarity': similarity_score,
                            'shared_concepts': list(common_concepts),
                            'file': module.filepath,
                            'confidence': module.confidence
                        })

            # Trier par similarité
            related_modules.sort(key=lambda x: x['similarity'], reverse=True)
            return related_modules[:8]

        except Exception as e:
            logger.debug(f"Semantic module search failed: {e}")
            return []

    async def _identify_architectural_patterns(self, entity) -> List[Dict[str, Any]]:
        """Identifie les patterns architecturaux - VERSION CORRIGÉE"""
        patterns = []

        try:
            await self._ensure_initialized()

            if self.analyzer:
                # Analyser les patterns algorithmiques
                try:
                    algo_patterns = await self.analyzer.detect_algorithmic_patterns(entity.entity_name)

                    if 'error' not in algo_patterns:
                        for pattern in algo_patterns.get('detected_patterns', []):
                            patterns.append({
                                'type': 'algorithmic',
                                'pattern': pattern.get('pattern', ''),
                                'confidence': pattern.get('confidence', 0),
                                'description': pattern.get('description', '')
                            })
                except Exception as e:
                    logger.debug(f"Erreur patterns algorithmiques: {e}")

            # Analyser les patterns structurels
            try:
                if getattr(entity, 'entity_type', '') == 'module':
                    children = await self.entity_manager.get_children(entity.entity_id)
                    if len(children) > 5:
                        patterns.append({
                            'type': 'structural',
                            'pattern': 'container_module',
                            'confidence': 0.8,
                            'description': f"Module conteneur avec {len(children)} sous-entités"
                        })
            except Exception as e:
                logger.debug(f"Erreur patterns structurels: {e}")

        except Exception as e:
            logger.debug(f"Erreur identification patterns: {e}")

        return patterns
    # === Méthodes d'API simplifiée ===

    async def get_project_stats(self) -> Dict[str, Any]:
        """Statistiques globales du projet"""
        await self._ensure_initialized()
        return self.entity_manager.get_stats()

    async def get_module_dependencies(self, module_name: str) -> List[str]:
        """Dépendances d'un module spécifique"""
        module = await self.entity_manager.find_entity(module_name)
        return list(module.dependencies) if module else []

    async def find_circular_dependencies(self) -> List[List[str]]:
        """Trouve toutes les dépendances circulaires"""
        modules = await self.entity_manager.get_entities_by_type('module')
        return await self._detect_circular_dependencies_smart(modules)

    async def get_impact_summary(self, entity_name: str) -> Dict[str, Any]:
        """Résumé d'impact pour modification"""
        await self._ensure_initialized()

        deps_analysis = await self.analyzer.analyze_dependencies(entity_name)
        if 'error' in deps_analysis:
            return {'error': deps_analysis['error']}

        impact = deps_analysis['impact_analysis']

        return {
            'risk_level': impact['risk_level'],
            'total_affected': impact['total_impact_entities'],
            'direct_dependents': len(impact['direct_dependents']),
            'affected_modules': len(impact['affected_modules']),
            'recommendations': impact['recommendations'][:3]  # Top 3
        }