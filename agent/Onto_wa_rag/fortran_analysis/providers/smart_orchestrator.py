"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# providers/smart_orchestrator.py
"""
Orchestrateur intelligent refactorisé remplaçant SmartContextProvider.
Utilise tous les composants des phases 1-2 pour une orchestration optimisée.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .base_provider import BaseContextProvider, provider_registry
from .local_context import LocalContextProvider
from .global_context import GlobalContextProvider
from .semantic_context import SemanticContextProvider
from ..core.entity_manager import EntityManager, get_entity_manager
from ..core.fortran_analyzer import get_fortran_analyzer
from ..utils.caching import global_cache

logger = logging.getLogger(__name__)


@dataclass
class ContextStrategy:
    """Stratégie de génération de contexte"""
    local_weight: float
    global_weight: float
    semantic_weight: float
    max_tokens: int

    def __post_init__(self):
        # Normaliser les poids
        total = self.local_weight + self.global_weight + self.semantic_weight
        if total > 0:
            self.local_weight /= total
            self.global_weight /= total
            self.semantic_weight /= total


class SmartContextOrchestrator:
    """
    Orchestrateur intelligent qui combine tous les types de contextes.
    Version refactorisée utilisant les composants unifiés des phases 1-2.
    """

    def __init__(self, document_store, rag_engine):
        self.document_store = document_store
        self.rag_engine = rag_engine

        # Composants centraux (initialisés à la demande)
        self.entity_manager: Optional[EntityManager] = None
        self.analyzer = None

        # Providers spécialisés
        self.local_provider: Optional[LocalContextProvider] = None
        self.global_provider: Optional[GlobalContextProvider] = None
        self.semantic_provider: Optional[SemanticContextProvider] = None

        # Configuration et cache
        self.cache = global_cache
        self._initialized = False

        # Stratégies prédéfinies pour différents types d'agents
        self.strategies = self._define_context_strategies()

    async def initialize(self):
        """Initialise l'orchestrateur et tous ses composants"""
        if self._initialized:
            return

        logger.info("🚀 Initialisation du SmartContextOrchestrator...")

        # 1. Initialiser EntityManager
        self.entity_manager = await get_entity_manager(self.document_store)

        # 2. Initialiser FortranAnalyzer
        self.analyzer = await get_fortran_analyzer(self.document_store, self.entity_manager)

        # 3. Créer les providers avec les composants unifiés
        self.local_provider = LocalContextProvider(self.document_store, self.rag_engine, self.entity_manager)
        self.global_provider = GlobalContextProvider(self.document_store, self.rag_engine, self.entity_manager)
        self.semantic_provider = SemanticContextProvider(self.document_store, self.rag_engine, self.entity_manager)

        # 4. Enregistrer dans le registre
        provider_registry.register('local', self.local_provider)
        provider_registry.register('global', self.global_provider)
        provider_registry.register('semantic', self.semantic_provider)

        self._initialized = True
        logger.info("✅ SmartContextOrchestrator initialisé")

    def _define_context_strategies(self) -> Dict[Tuple[str, str], ContextStrategy]:
        """Définit les stratégies de contexte pour différents types d'agents et tâches"""
        return {
            # === Développeur ===
            ("developer", "code_understanding"): ContextStrategy(0.6, 0.3, 0.1, 4000),
            ("developer", "debugging"): ContextStrategy(0.7, 0.2, 0.1, 3000),
            ("developer", "refactoring"): ContextStrategy(0.4, 0.4, 0.2, 5000),
            ("developer", "optimization"): ContextStrategy(0.3, 0.2, 0.5, 4000),

            # === Reviewer ===
            ("reviewer", "code_review"): ContextStrategy(0.3, 0.4, 0.3, 4500),
            ("reviewer", "architecture_review"): ContextStrategy(0.2, 0.6, 0.2, 5000),
            ("reviewer", "performance_review"): ContextStrategy(0.4, 0.2, 0.4, 4000),

            # === Analyzer ===
            ("analyzer", "dependency_analysis"): ContextStrategy(0.3, 0.7, 0.0, 3500),
            ("analyzer", "impact_analysis"): ContextStrategy(0.2, 0.8, 0.0, 4000),
            ("analyzer", "bug_detection"): ContextStrategy(0.7, 0.2, 0.1, 3000),

            # === Documentation ===
            ("documenter", "api_documentation"): ContextStrategy(0.5, 0.3, 0.2, 4000),
            ("documenter", "tutorial_writing"): ContextStrategy(0.3, 0.2, 0.5, 4500),

            # === Maintainer ===
            ("maintainer", "legacy_understanding"): ContextStrategy(0.4, 0.4, 0.2, 5000),
            ("maintainer", "migration_planning"): ContextStrategy(0.2, 0.6, 0.2, 5000),
        }

    async def get_context_for_agent(self,
                                    entity_name: str,
                                    agent_type: str = "developer",
                                    task_context: str = "code_understanding",
                                    max_tokens: int = 4000) -> Dict[str, Any]:
        """
        Interface principale pour les agents IA.
        Version optimisée avec cache et parallélisation.
        """
        await self.initialize()

        # 1. Déterminer la stratégie
        strategy = self._get_strategy(agent_type, task_context, max_tokens)

        logger.info(f"🎯 Génération contexte pour '{entity_name}' - {agent_type}/{task_context}")
        logger.info(
            f"   Stratégie: L:{strategy.local_weight:.1f} G:{strategy.global_weight:.1f} S:{strategy.semantic_weight:.1f}")

        # 2. Vérifier le cache global
        cache_key = f"context_{entity_name}_{agent_type}_{task_context}_{max_tokens}"
        cached_context = await self.cache.semantic_contexts.get(cache_key)
        if cached_context:
            logger.info("🎯 Contexte récupéré du cache")
            return cached_context

        # 3. Générer les contextes en parallèle
        contexts = await self._generate_contexts_parallel(entity_name, strategy)

        # 4. Fusionner et optimiser
        final_context = await self._merge_and_optimize_contexts(contexts, strategy, entity_name, agent_type,
                                                                task_context)

        # 5. Mettre en cache
        await self.cache.semantic_contexts.set(cache_key, final_context, ttl=1800)

        logger.info(f"✅ Contexte généré: {final_context.get('total_tokens', 0)} tokens")

        return final_context

    def _get_strategy(self, agent_type: str, task_context: str, max_tokens: int) -> ContextStrategy:
        """Récupère ou crée une stratégie de contexte"""
        strategy_key = (agent_type, task_context)

        if strategy_key in self.strategies:
            strategy = self.strategies[strategy_key]
            # Ajuster max_tokens si nécessaire
            if max_tokens != strategy.max_tokens:
                return ContextStrategy(
                    strategy.local_weight,
                    strategy.global_weight,
                    strategy.semantic_weight,
                    max_tokens
                )
            return strategy
        else:
            # Stratégie par défaut
            logger.warning(f"Stratégie non définie pour {agent_type}/{task_context}, utilisation par défaut")
            return ContextStrategy(0.5, 0.3, 0.2, max_tokens)

    async def _generate_contexts_parallel(self, entity_name: str, strategy: ContextStrategy) -> Dict[str, Any]:
        """Génère les contextes en parallèle selon la stratégie"""
        tasks = []

        # Créer les tâches selon les poids
        if strategy.local_weight > 0:
            local_tokens = int(strategy.max_tokens * strategy.local_weight)
            tasks.append(('local', self.local_provider.get_local_context(entity_name, local_tokens)))

        if strategy.global_weight > 0:
            global_tokens = int(strategy.max_tokens * strategy.global_weight)
            tasks.append(('global', self.global_provider.get_global_context(entity_name, global_tokens)))

        if strategy.semantic_weight > 0:
            semantic_tokens = int(strategy.max_tokens * strategy.semantic_weight)
            tasks.append(('semantic', self.semantic_provider.get_semantic_context(entity_name, semantic_tokens)))

        # Exécuter en parallèle
        if tasks:
            results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)

            contexts = {}
            for i, (context_type, _) in enumerate(tasks):
                result = results[i]
                if isinstance(result, Exception):
                    logger.error(f"Erreur génération contexte {context_type}: {result}")
                    contexts[context_type] = {"error": str(result)}
                else:
                    contexts[context_type] = result

            return contexts
        else:
            return {}

    async def _merge_and_optimize_contexts(self,
                                           contexts: Dict[str, Dict[str, Any]],
                                           strategy: ContextStrategy,
                                           entity_name: str,
                                           agent_type: str,
                                           task_context: str) -> Dict[str, Any]:
        """
        Fusionne et optimise les contextes selon la stratégie.
        Version améliorée avec insights intelligents.
        """
        merged_context = {
            'entity': entity_name,
            'contexts': contexts,
            'summary': {},
            'key_insights': [],
            'recommendations': [],
            'agent_specific_insights': [],
            'total_tokens': 0,
            'generation_info': {}
        }

        # Créer le résumé consolidé
        merged_context['summary'] = await self._create_intelligent_summary(contexts, strategy)

        # Extraire les insights clés
        merged_context['key_insights'] = await self._extract_intelligent_insights(contexts, strategy)

        # Générer des recommandations
        merged_context['recommendations'] = await self._generate_intelligent_recommendations(
            contexts, strategy, agent_type, task_context
        )

        # Insights spécifiques à l'agent
        merged_context['agent_specific_insights'] = await self._generate_agent_specific_insights(
            contexts, agent_type, task_context, entity_name
        )

        # Calculer les tokens
        total_tokens = sum(
            ctx.get('tokens_used', 0) for ctx in contexts.values()
            if isinstance(ctx, dict) and 'error' not in ctx
        )
        merged_context['total_tokens'] = total_tokens

        # Informations de génération enrichies
        merged_context['generation_info'] = {
            'agent_type': agent_type,
            'task_context': task_context,
            'strategy_used': {
                'local_weight': strategy.local_weight,
                'global_weight': strategy.global_weight,
                'semantic_weight': strategy.semantic_weight
            },
            'contexts_generated': list(contexts.keys()),
            'entity_resolution': await self._get_entity_resolution_info(entity_name),
            'generation_timestamp': asyncio.get_event_loop().time()
        }

        # Optimiser si nécessaire
        if total_tokens > strategy.max_tokens:
            merged_context = await self._optimize_context_size(merged_context, strategy.max_tokens)

        return merged_context

    async def _create_intelligent_summary(self, contexts: Dict[str, Dict[str, Any]], strategy: ContextStrategy) -> Dict[
        str, Any]:
        """Crée un résumé intelligent consolidé"""
        summary = {
            'entity_overview': {},
            'complexity_analysis': {},
            'architectural_role': '',
            'key_relationships': {},
            'quality_indicators': {}
        }

        # Vue d'ensemble de l'entité depuis le contexte local
        if 'local' in contexts and 'error' not in contexts['local']:
            local_ctx = contexts['local']
            main_def = local_ctx.get('main_definition', {})

            summary['entity_overview'] = {
                'name': main_def.get('name', ''),
                'type': main_def.get('type', ''),
                'signature': main_def.get('signature', ''),
                'file': main_def.get('location', {}).get('file', ''),
                'concepts': main_def.get('concepts', [])[:3],
                'is_grouped': main_def.get('metadata', {}).get('is_grouped', False)
            }

            # Analyse de complexité
            deps_count = len(local_ctx.get('immediate_dependencies', []))
            calls_count = len(local_ctx.get('called_functions', []))
            children_count = len(local_ctx.get('children_context', []))

            complexity_level = "low"
            if deps_count > 5 or calls_count > 8 or children_count > 5:
                complexity_level = "high"
            elif deps_count > 2 or calls_count > 4 or children_count > 2:
                complexity_level = "medium"

            summary['complexity_analysis'] = {
                'level': complexity_level,
                'dependencies_count': deps_count,
                'function_calls_count': calls_count,
                'children_count': children_count,
                'complexity_factors': []
            }

            if deps_count > 3:
                summary['complexity_analysis']['complexity_factors'].append(f"Many dependencies ({deps_count})")
            if calls_count > 5:
                summary['complexity_analysis']['complexity_factors'].append(f"Many function calls ({calls_count})")
            if children_count > 3:
                summary['complexity_analysis']['complexity_factors'].append(
                    f"Contains many sub-entities ({children_count})")

        # Rôle architectural depuis le contexte global
        if 'global' in contexts and 'error' not in contexts['global']:
            global_ctx = contexts['global']
            impact = global_ctx.get('impact_analysis', {})

            if impact:
                risk_level = impact.get('risk_level', 'unknown')
                affected_count = len(impact.get('direct_dependents', []))

                summary['architectural_role'] = f"Impact: {risk_level}, affects {affected_count} entities"

                # Relations clés
                summary['key_relationships'] = {
                    'dependents': impact.get('direct_dependents', [])[:5],
                    'affected_modules': impact.get('affected_modules', [])[:3]
                }

        # Indicateurs de qualité depuis les concepts sémantiques
        if 'semantic' in contexts and 'error' not in contexts['semantic']:
            semantic_ctx = contexts['semantic']

            concepts_count = len(semantic_ctx.get('main_concepts', []))
            similar_count = len(semantic_ctx.get('similar_entities', []))
            patterns_count = len(semantic_ctx.get('algorithmic_patterns', []))

            summary['quality_indicators'] = {
                'conceptual_richness': concepts_count,
                'pattern_detection': patterns_count,
                'similarity_connections': similar_count,
                'semantic_clarity': 'high' if concepts_count > 3 else 'medium' if concepts_count > 1 else 'low'
            }

        return summary

    async def _extract_intelligent_insights(self, contexts: Dict[str, Dict[str, Any]], strategy: ContextStrategy) -> \
    List[str]:
        """Extrait des insights intelligents multi-contextes"""
        insights = []

        # Insights du contexte local
        if 'local' in contexts and 'error' not in contexts['local']:
            local_insights = await self._extract_local_insights(contexts['local'])
            insights.extend(local_insights)

        # Insights du contexte global
        if 'global' in contexts and 'error' not in contexts['global']:
            global_insights = await self._extract_global_insights(contexts['global'])
            insights.extend(global_insights)

        # Insights du contexte sémantique
        if 'semantic' in contexts and 'error' not in contexts['semantic']:
            semantic_insights = await self._extract_semantic_insights(contexts['semantic'])
            insights.extend(semantic_insights)

        # Insights cross-contexte
        cross_insights = await self._extract_cross_context_insights(contexts)
        insights.extend(cross_insights)

        return insights[:10]  # Top 10 insights

    async def _extract_local_insights(self, local_ctx: Dict[str, Any]) -> List[str]:
        """Insights spécifiques au contexte local"""
        insights = []

        main_def = local_ctx.get('main_definition', {})
        entity_type = main_def.get('type', '')
        entity_name = main_def.get('name', '')

        # Analyse par type
        if entity_type == 'subroutine':
            calls_count = len(local_ctx.get('called_functions', []))
            if calls_count > 6:
                insights.append(f"Complex subroutine: {calls_count} function calls suggest high algorithmic complexity")
            elif calls_count == 0:
                insights.append("Simple subroutine: no function calls, likely computational or data manipulation")

        elif entity_type == 'module':
            deps_count = len(local_ctx.get('immediate_dependencies', []))
            children_count = len(local_ctx.get('children_context', []))

            if children_count > 5:
                insights.append(f"Container module: {children_count} procedures suggest well-organized functionality")
            if deps_count > 4:
                insights.append(f"Highly coupled module: {deps_count} dependencies indicate complex interactions")
            elif deps_count == 0:
                insights.append("Self-contained module: no external dependencies, good for reusability")

        # Analyse des patterns de noms
        if entity_name:
            name_lower = entity_name.lower()
            if 'compute' in name_lower or 'calc' in name_lower:
                insights.append("Computational routine: likely performance-critical, consider optimization")
            elif 'init' in name_lower:
                insights.append("Initialization routine: critical for system setup and state management")
            elif 'part_' in name_lower:
                insights.append("Split entity: part of larger function, review chunking strategy if needed")

        return insights

    async def _extract_global_insights(self, global_ctx: Dict[str, Any]) -> List[str]:
        """Insights spécifiques au contexte global"""
        insights = []

        # Analyse d'impact
        impact = global_ctx.get('impact_analysis', {})
        if impact:
            risk_level = impact.get('risk_level', '')
            if risk_level == 'high':
                insights.append("High impact entity: changes will affect many components, coordinate with team")
            elif risk_level == 'low':
                insights.append("Low impact entity: safe to modify with minimal testing")

            recommendations = impact.get('recommendations', [])
            for rec in recommendations[:2]:
                insights.append(f"Impact recommendation: {rec}")

        # Analyse de la hiérarchie des modules
        hierarchy = global_ctx.get('module_hierarchy', {})
        if hierarchy:
            circular_deps = hierarchy.get('circular_dependencies', [])
            if circular_deps:
                insights.append(f"Circular dependencies detected: {len(circular_deps)} cycles need resolution")

        # Vue d'ensemble du projet
        overview = global_ctx.get('project_overview', {})
        if overview:
            arch_style = overview.get('architectural_style', '')
            if arch_style:
                insights.append(f"Project architecture: {arch_style} style")

        return insights

    async def _extract_semantic_insights(self, semantic_ctx: Dict[str, Any]) -> List[str]:
        """Insights spécifiques au contexte sémantique"""
        insights = []

        # Patterns algorithmiques
        patterns = semantic_ctx.get('algorithmic_patterns', [])
        if patterns:
            top_pattern = patterns[0]
            pattern_name = top_pattern.get('pattern', '')
            confidence = top_pattern.get('confidence', 0)

            if confidence > 0.8:
                insights.append(f"Strong algorithmic pattern: {pattern_name} (confidence: {confidence:.2f})")
            else:
                insights.append(f"Potential algorithmic pattern: {pattern_name}")

        # Entités similaires
        similar_entities = semantic_ctx.get('similar_entities', [])
        if len(similar_entities) > 3:
            insights.append(f"Common implementation pattern: {len(similar_entities)} similar entities found")

        # Concepts principaux
        main_concepts = semantic_ctx.get('main_concepts', [])
        high_confidence_concepts = [c for c in main_concepts if c.get('confidence', 0) > 0.8]

        if high_confidence_concepts:
            concept_labels = [c.get('label', '') for c in high_confidence_concepts[:2]]
            insights.append(f"Strong conceptual identity: {', '.join(concept_labels)}")

        return insights

    async def _extract_cross_context_insights(self, contexts: Dict[str, Dict[str, Any]]) -> List[str]:
        """Insights basés sur la corrélation entre contextes"""
        insights = []

        # Corrélation complexité locale vs impact global
        if 'local' in contexts and 'global' in contexts:
            local_complexity = self._assess_local_complexity(contexts['local'])
            global_impact = self._assess_global_impact(contexts['global'])

            if local_complexity == 'high' and global_impact == 'high':
                insights.append("Critical complexity: high local complexity with high global impact")
            elif local_complexity == 'low' and global_impact == 'high':
                insights.append("Hidden importance: simple implementation but high architectural importance")

        # Corrélation patterns sémantiques vs structure
        if 'semantic' in contexts and 'local' in contexts:
            patterns_count = len(contexts['semantic'].get('algorithmic_patterns', []))
            calls_count = len(contexts['local'].get('called_functions', []))

            if patterns_count > 0 and calls_count > 5:
                insights.append("Complex algorithmic implementation: multiple patterns with many function calls")

        return insights

    def _assess_local_complexity(self, local_ctx: Dict[str, Any]) -> str:
        """Évalue la complexité locale"""
        deps = len(local_ctx.get('immediate_dependencies', []))
        calls = len(local_ctx.get('called_functions', []))
        children = len(local_ctx.get('children_context', []))

        total_complexity = deps + calls + children

        if total_complexity > 10:
            return 'high'
        elif total_complexity > 5:
            return 'medium'
        else:
            return 'low'

    def _assess_global_impact(self, global_ctx: Dict[str, Any]) -> str:
        """Évalue l'impact global"""
        impact = global_ctx.get('impact_analysis', {})
        return impact.get('risk_level', 'unknown')

    async def _generate_intelligent_recommendations(self,
                                                    contexts: Dict[str, Dict[str, Any]],
                                                    strategy: ContextStrategy,
                                                    agent_type: str,
                                                    task_context: str) -> List[str]:
        """Génère des recommandations intelligentes selon l'agent et la tâche"""
        recommendations = []

        # Recommandations basées sur l'agent
        if agent_type == "developer":
            recommendations.extend(await self._get_developer_recommendations(contexts, task_context))
        elif agent_type == "reviewer":
            recommendations.extend(await self._get_reviewer_recommendations(contexts, task_context))
        elif agent_type == "analyzer":
            recommendations.extend(await self._get_analyzer_recommendations(contexts, task_context))
        elif agent_type == "documenter":
            recommendations.extend(await self._get_documenter_recommendations(contexts, task_context))

        # Recommandations générales
        general_recs = await self._get_general_recommendations(contexts)
        recommendations.extend(general_recs)

        return recommendations[:8]  # Top 8 recommandations

    async def _get_developer_recommendations(self, contexts: Dict[str, Dict[str, Any]], task: str) -> List[str]:
        """Recommandations spécifiques aux développeurs"""
        recs = []

        if task == "debugging" and 'local' in contexts:
            calls_count = len(contexts['local'].get('called_functions', []))
            if calls_count > 5:
                recs.append("Debug strategy: Focus on function call chain, use step-through debugging")

        elif task == "optimization" and 'semantic' in contexts:
            patterns = contexts['semantic'].get('algorithmic_patterns', [])
            if patterns:
                pattern_name = patterns[0].get('pattern', '')
                recs.append(f"Optimization target: Review {pattern_name} implementation for performance gains")

        elif task == "refactoring" and 'global' in contexts:
            impact = contexts['global'].get('impact_analysis', {})
            if impact.get('risk_level') == 'high':
                recs.append("Refactoring caution: High impact entity, implement comprehensive tests first")

        return recs

    async def _get_reviewer_recommendations(self, contexts: Dict[str, Dict[str, Any]], task: str) -> List[str]:
        """Recommandations spécifiques aux reviewers"""
        recs = []

        if task == "code_review" and 'local' in contexts:
            main_def = contexts['local'].get('main_definition', {})
            if main_def.get('metadata', {}).get('is_grouped'):
                recs.append("Review focus: Grouped entity, verify chunk boundaries and completeness")

        elif task == "architecture_review" and 'global' in contexts:
            circular_deps = contexts['global'].get('module_hierarchy', {}).get('circular_dependencies', [])
            if circular_deps:
                recs.append(f"Architecture issue: {len(circular_deps)} circular dependencies need resolution")

        return recs

    async def _get_analyzer_recommendations(self, contexts: Dict[str, Dict[str, Any]], task: str) -> List[str]:
        """Recommandations spécifiques aux analyzers"""
        recs = []

        if task == "dependency_analysis" and 'global' in contexts:
            overview = contexts['global'].get('project_overview', {})
            arch_style = overview.get('architectural_style', '')
            if arch_style == 'highly_coupled':
                recs.append("Dependency concern: High coupling detected, consider modularization")

        return recs

    async def _get_documenter_recommendations(self, contexts: Dict[str, Dict[str, Any]], task: str) -> List[str]:
        """Recommandations spécifiques aux documenters"""
        recs = []

        if task == "api_documentation" and 'local' in contexts:
            signature = contexts['local'].get('main_definition', {}).get('signature', '')
            if signature == "Signature not found":
                recs.append("Documentation gap: Signature extraction failed, review code structure")

        return recs

    async def _get_general_recommendations(self, contexts: Dict[str, Dict[str, Any]]) -> List[str]:
        """Recommandations générales basées sur l'analyse"""
        recs = []

        # Recommandations de qualité
        if 'semantic' in contexts:
            concepts = contexts['semantic'].get('main_concepts', [])
            low_confidence_concepts = [c for c in concepts if c.get('confidence', 0) < 0.5]

            if len(low_confidence_concepts) > len(concepts) / 2:
                recs.append("Code clarity: Low concept confidence, consider better naming and documentation")

        return recs

    async def _generate_agent_specific_insights(self,
                                                contexts: Dict[str, Dict[str, Any]],
                                                agent_type: str,
                                                task_context: str,
                                                entity_name: str) -> List[str]:
        """Génère des insights spécifiques à l'agent et à la tâche"""
        insights = []

        # Insights basés sur le type d'agent
        if agent_type == "developer":
            if task_context == "debugging":
                insights.append("🐛 Focus on call stack and variable state changes")
                if 'local' in contexts:
                    calls = contexts['local'].get('called_functions', [])
                    if calls:
                        insights.append(
                            f"🔍 Key functions to examine: {', '.join([c.get('name', '') for c in calls[:3]])}")

        elif agent_type == "reviewer":
            insights.append("👀 Review checklist: signature, dependencies, error handling")
            if 'global' in contexts:
                impact = contexts['global'].get('impact_analysis', {})
                if impact.get('risk_level') == 'high':
                    insights.append("⚠️ High-risk change: Request additional reviewers")

        elif agent_type == "analyzer":
            insights.append("📊 Analysis focus: patterns, dependencies, complexity metrics")
            if 'semantic' in contexts:
                patterns = contexts['semantic'].get('algorithmic_patterns', [])
                if patterns:
                    insights.append(f"🔬 Detected patterns: {', '.join([p.get('pattern', '') for p in patterns[:2]])}")

        return insights

    async def _get_entity_resolution_info(self, entity_name: str) -> Dict[str, Any]:
        """Informations sur la résolution de l'entité"""
        entity = await self.entity_manager.find_entity(entity_name)

        if entity:
            return {
                'resolved': True,
                'entity_id': entity.entity_id,
                'resolution_method': 'entity_manager',
                'confidence': entity.confidence,
                'is_grouped': entity.is_grouped,
                'source_method': entity.source_method
            }
        else:
            return {
                'resolved': False,
                'resolution_method': 'failed',
                'suggestions': await self._find_similar_entity_names(entity_name)
            }

    async def _find_similar_entity_names(self, entity_name: str) -> List[str]:
        """Trouve des noms d'entités similaires"""
        all_entities = list(self.entity_manager.entities.values())
        similar = []

        name_lower = entity_name.lower()
        for entity in all_entities:
            entity_name_lower = entity.entity_name.lower()
            if name_lower in entity_name_lower or entity_name_lower in name_lower:
                similar.append(entity.entity_name)

        return similar[:5]

    async def _optimize_context_size(self, context: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        """Optimise la taille du contexte si nécessaire"""
        # Stratégie d'optimisation intelligente
        contexts = context.get('contexts', {})

        # 1. Réduire les listes longues en gardant les plus pertinents
        for context_type, context_data in contexts.items():
            if not isinstance(context_data, dict):
                continue

            # Optimiser les listes selon leur importance
            optimizations = {
                'called_functions': 8,
                'similar_entities': 6,
                'semantic_neighbors': 6,
                'immediate_dependencies': 8,
                'cross_file_relations': 5
            }

            for key, max_items in optimizations.items():
                if key in context_data and isinstance(context_data[key], list):
                    if len(context_data[key]) > max_items:
                        context_data[key] = context_data[key][:max_items]

        # 2. Réduire la verbosité des recommandations et insights
        if 'key_insights' in context and len(context['key_insights']) > 8:
            context['key_insights'] = context['key_insights'][:8]

        if 'recommendations' in context and len(context['recommendations']) > 6:
            context['recommendations'] = context['recommendations'][:6]

        # 3. Marquer comme optimisé
        context['optimized'] = True
        context['optimization_applied'] = True

        return context

    # === Interface de compatibilité avec l'ancien SmartContextProvider ===

    async def get_local_context(self, entity_name: str, max_tokens: int = 2000) -> Dict[str, Any]:
        """Interface de compatibilité"""
        await self.initialize()
        return await self.local_provider.get_local_context(entity_name, max_tokens)

    async def get_global_context(self, entity_name: str, max_tokens: int = 3000) -> Dict[str, Any]:
        """Interface de compatibilité"""
        await self.initialize()
        return await self.global_provider.get_global_context(entity_name, max_tokens)

    async def get_semantic_context(self, entity_name: str, max_tokens: int = 2000) -> Dict[str, Any]:
        """Interface de compatibilité"""
        await self.initialize()
        return await self.semantic_provider.get_semantic_context(entity_name, max_tokens)

    # === Méthodes utilitaires et statistiques ===

    async def search_entities(self, query: str) -> List[Dict[str, Any]]:
        """Interface de recherche d'entités"""
        await self.initialize()

        # Recherche exacte
        entity = await self.entity_manager.find_entity(query)
        if entity:
            return [{
                'name': entity.entity_name,
                'type': entity.entity_type,
                'file': entity.filepath,
                'match_type': 'exact',
                'confidence': entity.confidence
            }]

        # Recherche fuzzy
        similar_names = await self._find_similar_entity_names(query)
        results = []

        for name in similar_names:
            entity = await self.entity_manager.find_entity(name)
            if entity:
                results.append({
                    'name': entity.entity_name,
                    'type': entity.entity_type,
                    'file': entity.filepath,
                    'match_type': 'fuzzy',
                    'confidence': entity.confidence
                })

        return results

    def get_index_stats(self) -> Dict[str, Any]:
        """Statistiques de l'index"""
        if self.entity_manager:
            return self.entity_manager.get_stats()
        return {}

    def get_cache_stats(self) -> Dict[str, Any]:
        """Statistiques des caches"""
        return self.cache.get_all_stats()

    async def clear_caches(self):
        """Vide tous les caches"""
        await self.cache.cleanup_all_expired()
        if self.entity_manager:
            self.entity_manager.clear_caches()


# Instance globale
_global_orchestrator = None


async def get_smart_orchestrator(document_store, rag_engine) -> SmartContextOrchestrator:
    """Factory pour obtenir l'orchestrateur global"""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = SmartContextOrchestrator(document_store, rag_engine)
        await _global_orchestrator.initialize()
    return _global_orchestrator
