# output/text_generator.py
"""
Générateur de contexte textuel pour LLMs refactorisé.
Utilise SmartContextOrchestrator et les composants unifiés des phases 1-3.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from ..providers.smart_orchestrator import SmartContextOrchestrator, get_smart_orchestrator
from ..core.entity_manager import EntityManager, get_entity_manager
from ..core.fortran_analyzer import get_fortran_analyzer
from ..core.concept_detector import get_concept_detector
from ..utils.caching import global_cache

logger = logging.getLogger(__name__)


class ContextualTextGenerator:
    """
    Générateur de contexte textuel pour LLMs - VERSION REFACTORISÉE.
    Utilise SmartContextOrchestrator pour une génération intelligente et optimisée.
    """

    def __init__(self, document_store, rag_engine):
        self.document_store = document_store
        self.rag_engine = rag_engine

        # Composants centraux des phases 1-3
        self.orchestrator: Optional[SmartContextOrchestrator] = None
        self.entity_manager: Optional[EntityManager] = None
        self.analyzer = None
        self.concept_detector = None

        # Configuration
        self.cache = global_cache
        self._initialized = False

    async def initialize(self):
        """Initialise tous les composants"""
        if self._initialized:
            return

        logger.info("🔧 Initialisation du ContextualTextGenerator...")

        # Composants centraux
        self.orchestrator = await get_smart_orchestrator(self.document_store, self.rag_engine)
        self.entity_manager = await get_entity_manager(self.document_store)
        self.analyzer = await get_fortran_analyzer(self.document_store, self.entity_manager)
        self.concept_detector = get_concept_detector(getattr(self.rag_engine, 'classifier', None))

        self._initialized = True
        logger.info("✅ ContextualTextGenerator initialisé")

    async def get_contextual_text(self,
                                  element_name: str,
                                  context_type: str = "complete",
                                  agent_perspective: str = "developer",
                                  task_context: str = "code_understanding",
                                  max_tokens: int = 4000,
                                  format_style: str = "detailed") -> str:
        """
        Génère un contexte textuel complet pour un élément donné.
        Version refactorisée utilisant SmartContextOrchestrator.
        """
        await self.initialize()

        logger.info(f"📝 Génération contexte textuel: {element_name} ({format_style})")

        # 1. Résoudre l'élément (amélioration avec EntityManager)
        resolved_entity = await self._resolve_element_enhanced(element_name)

        if not resolved_entity:
            return await self._generate_not_found_text(element_name)

        # 2. Récupérer le contexte selon le type demandé
        context_data = await self._get_context_by_type(
            resolved_entity, context_type, agent_perspective, task_context, max_tokens
        )

        # 3. Formater selon le style demandé
        if format_style == "detailed":
            return await self._format_detailed_context_v2(context_data, resolved_entity)
        elif format_style == "summary":
            return await self._format_summary_context_v2(context_data, resolved_entity)
        elif format_style == "bullet_points":
            return await self._format_bullet_context_v2(context_data, resolved_entity)
        else:
            return await self._format_detailed_context_v2(context_data, resolved_entity)

    async def _resolve_element_enhanced(self, element_name: str) -> Optional[Dict[str, Any]]:
        """
        Résolution d'élément améliorée utilisant EntityManager.
        Remplace la logique manuelle par une recherche intelligente.
        """
        # 1. Recherche directe par entité
        entity = await self.entity_manager.find_entity(element_name)
        if entity:
            return {
                'name': entity.entity_name,
                'type': entity.entity_type,
                'resolution_method': 'entity_manager_direct',
                'entity_object': entity,
                'confidence': entity.confidence
            }

        # 2. Recherche par concept si pas trouvé
        concept_results = await self._search_by_concept_enhanced(element_name)
        if concept_results:
            return concept_results

        # 3. Recherche par fichier
        if '.' in element_name or '/' in element_name:
            file_results = await self._search_by_file_enhanced(element_name)
            if file_results:
                return file_results

        # 4. Recherche fuzzy (fallback)
        search_results = await self.orchestrator.search_entities(element_name)
        if search_results:
            best_match = search_results[0]
            entity = await self.entity_manager.find_entity(best_match['name'])
            if entity:
                return {
                    'name': entity.entity_name,
                    'type': entity.entity_type,
                    'resolution_method': 'fuzzy_search',
                    'entity_object': entity,
                    'alternatives': [r['name'] for r in search_results[1:5]]
                }

        return None

    async def _search_by_concept_enhanced(self, concept_name: str) -> Optional[Dict[str, Any]]:
        """
        Recherche par concept améliorée utilisant ConceptDetector.
        """
        # Utiliser ConceptDetector pour trouver des entités liées
        try:
            # Chercher dans toutes les entités celles qui ont ce concept
            matching_entities = []

            for entity in self.entity_manager.entities.values():
                entity_concepts = entity.concepts

                # Recherche flexible du concept
                concept_lower = concept_name.lower()
                for concept in entity_concepts:
                    if (concept_lower in concept.lower() or
                            concept.lower() in concept_lower):
                        # Calculer la confiance basée sur la correspondance
                        confidence = 0.8 if concept_lower == concept.lower() else 0.6

                        matching_entities.append({
                            'entity': entity,
                            'matched_concept': concept,
                            'confidence': confidence
                        })
                        break

            if matching_entities:
                # Trier par confiance et prendre le meilleur
                best_match = max(matching_entities, key=lambda x: x['confidence'])

                return {
                    'name': f"concept_{concept_name}",
                    'type': 'concept_group',
                    'resolution_method': 'concept_search_enhanced',
                    'concept_name': concept_name,
                    'best_entity': best_match['entity'],
                    'all_matches': matching_entities[:10],
                    'total_matches': len(matching_entities)
                }

        except Exception as e:
            logger.debug(f"Erreur recherche concept: {e}")

        return None

    async def _search_by_file_enhanced(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Recherche par fichier améliorée utilisant EntityManager"""
        entities_in_file = await self.entity_manager.get_entities_in_file(filepath)

        if entities_in_file:
            # Trouver l'entité principale (module > program > première fonction)
            main_entity = self._find_main_entity_in_file_smart(entities_in_file)

            if main_entity:
                return {
                    'name': main_entity.entity_name,
                    'type': main_entity.entity_type,
                    'resolution_method': 'file_main_entity',
                    'entity_object': main_entity,
                    'file_path': filepath,
                    'total_entities_in_file': len(entities_in_file)
                }

        return None

    def _find_main_entity_in_file_smart(self, entities: List) -> Optional[Any]:
        """Trouve l'entité principale d'un fichier avec logique intelligente"""
        if not entities:
            return None

        # Priorités : module > program > type > subroutine > function
        priority_order = ['module', 'program', 'type_definition', 'subroutine', 'function']

        for entity_type in priority_order:
            for entity in entities:
                if entity.entity_type == entity_type:
                    return entity

        # Si aucune priorité trouvée, prendre la première
        return entities[0]

    async def _get_context_by_type(self,
                                   resolved_entity: Dict[str, Any],
                                   context_type: str,
                                   agent_perspective: str,
                                   task_context: str,
                                   max_tokens: int) -> Dict[str, Any]:
        """Récupère le contexte selon le type avec SmartContextOrchestrator"""

        entity_name = resolved_entity['name']

        if context_type == "complete":
            return await self.orchestrator.get_context_for_agent(
                entity_name, agent_perspective, task_context, max_tokens
            )
        elif context_type == "local":
            return await self.orchestrator.get_local_context(entity_name, max_tokens)
        elif context_type == "global":
            return await self.orchestrator.get_global_context(entity_name, max_tokens)
        elif context_type == "semantic":
            return await self.orchestrator.get_semantic_context(entity_name, max_tokens)
        else:
            # Type non supporté, utiliser "complete" par défaut
            return await self.orchestrator.get_context_for_agent(
                entity_name, agent_perspective, task_context, max_tokens
            )

    async def _format_detailed_context_v2(self, context_data: Dict[str, Any],
                                          resolved_entity: Dict[str, Any]) -> str:
        """
        Formatage détaillé amélioré utilisant les nouvelles structures.
        Version v2 optimisée et enrichie.
        """
        lines = []
        entity_name = resolved_entity['name']
        entity_type = resolved_entity.get('type', 'unknown')
        resolution_method = resolved_entity.get('resolution_method', 'unknown')

        # === EN-TÊTE ENRICHI ===
        lines.append("=" * 80)

        if entity_type == 'concept_group':
            return await self._format_concept_group_detailed(context_data, resolved_entity)

        lines.append(f"📋 CONTEXTE FORTRAN DÉTAILLÉ - v2.0")
        lines.append("=" * 80)
        lines.append(f"Entité analysée: {entity_name} ({entity_type})")
        lines.append(f"Méthode de résolution: {resolution_method}")

        if 'alternatives' in resolved_entity:
            lines.append(f"Alternatives trouvées: {', '.join(resolved_entity['alternatives'])}")

        confidence = resolved_entity.get('confidence', 1.0)
        lines.append(f"Confiance de résolution: {confidence:.3f}")
        lines.append("")

        # === RÉSUMÉ EXÉCUTIF (NOUVEAU) ===
        summary = context_data.get('summary', {})
        if summary:
            lines.append("🎯 RÉSUMÉ EXÉCUTIF")
            lines.append("-" * 40)

            entity_overview = summary.get('entity_overview', {})
            if entity_overview:
                lines.append(f"Type: {entity_overview.get('type', 'N/A')}")
                lines.append(f"Signature: {entity_overview.get('signature', 'N/A')}")
                lines.append(f"Fichier: {entity_overview.get('file', 'N/A')}")

                if entity_overview.get('concepts'):
                    lines.append(f"Concepts clés: {', '.join(entity_overview['concepts'])}")

            complexity = summary.get('complexity_analysis', {})
            if complexity:
                level = complexity.get('level', 'unknown')
                factors = complexity.get('complexity_factors', [])
                lines.append(f"Complexité: {level}")
                if factors:
                    lines.append(f"Facteurs: {'; '.join(factors)}")

            lines.append("")

        # === INSIGHTS INTELLIGENTS (NOUVEAU) ===
        key_insights = context_data.get('key_insights', [])
        if key_insights:
            lines.append("💡 INSIGHTS CLÉS")
            lines.append("-" * 40)
            for insight in key_insights:
                lines.append(f"• {insight}")
            lines.append("")

        # === DÉFINITION PRINCIPALE (AMÉLIORÉE) ===
        await self._add_main_definition_section_v2(lines, context_data)

        # === CONTEXTE LOCAL ENRICHI ===
        await self._add_local_context_section_v2(lines, context_data)

        # === CONTEXTE GLOBAL AVEC ANALYSE D'IMPACT ===
        await self._add_global_context_section_v2(lines, context_data)

        # === CONTEXTE SÉMANTIQUE AVANCÉ ===
        await self._add_semantic_context_section_v2(lines, context_data)

        # === RECOMMANDATIONS INTELLIGENTES (NOUVEAU) ===
        recommendations = context_data.get('recommendations', [])
        if recommendations:
            lines.append("🎯 RECOMMANDATIONS")
            lines.append("-" * 40)
            for rec in recommendations:
                lines.append(f"• {rec}")
            lines.append("")

        # === INSIGHTS SPÉCIFIQUES À L'AGENT (NOUVEAU) ===
        agent_insights = context_data.get('agent_specific_insights', [])
        if agent_insights:
            lines.append("🤖 INSIGHTS SPÉCIFIQUES À L'AGENT")
            lines.append("-" * 40)
            for insight in agent_insights:
                lines.append(f"• {insight}")
            lines.append("")

        # === INFORMATIONS DE GÉNÉRATION (ENRICHIES) ===
        generation_info = context_data.get('generation_info', {})
        if generation_info:
            lines.append("📊 MÉTADONNÉES DE GÉNÉRATION")
            lines.append("-" * 40)

            agent_type = generation_info.get('agent_type', 'unknown')
            task_context = generation_info.get('task_context', 'unknown')
            lines.append(f"Agent: {agent_type} | Tâche: {task_context}")

            strategy = generation_info.get('strategy_used', {})
            if strategy:
                lines.append(f"Stratégie: L:{strategy.get('local_weight', 0):.1f} "
                             f"G:{strategy.get('global_weight', 0):.1f} "
                             f"S:{strategy.get('semantic_weight', 0):.1f}")

            contexts_gen = generation_info.get('contexts_generated', [])
            total_tokens = context_data.get('total_tokens', 0)
            lines.append(f"Contextes générés: {', '.join(contexts_gen)}")
            lines.append(f"Tokens utilisés: {total_tokens}")

            # Informations de résolution d'entité
            entity_resolution = generation_info.get('entity_resolution', {})
            if entity_resolution:
                lines.append(f"Résolution d'entité: {entity_resolution.get('resolved', False)}")
                if entity_resolution.get('is_grouped'):
                    lines.append("📦 Entité regroupée (multi-chunks)")

        lines.append("")
        lines.append("=" * 80)

        return '\n'.join(lines)

    async def _add_main_definition_section_v2(self, lines: List[str], context_data: Dict[str, Any]):
        """Ajoute la section de définition principale améliorée"""
        contexts = context_data.get('contexts', {})

        if 'local' in contexts and 'error' not in contexts['local']:
            local_ctx = contexts['local']
            main_def = local_ctx.get('main_definition', {})

            if main_def:
                lines.append("🔍 DÉFINITION PRINCIPALE")
                lines.append("-" * 40)
                lines.append(f"Nom: {main_def.get('name', 'N/A')}")
                lines.append(f"Type: {main_def.get('type', 'N/A')}")

                location = main_def.get('location', {})
                if location:
                    lines.append(f"Fichier: {location.get('file', 'N/A')}")
                    lines.append(f"Lignes: {location.get('lines', 'N/A')}")

                signature = main_def.get('signature', '')
                if signature and signature != "Signature not found":
                    lines.append(f"Signature: {signature}")

                # Métadonnées enrichies
                metadata = main_def.get('metadata', {})
                if metadata:
                    if metadata.get('is_grouped'):
                        lines.append(f"📦 Entité regroupée: {metadata.get('chunk_count', 0)} chunks")

                    confidence = metadata.get('confidence', 0)
                    if confidence:
                        lines.append(f"Confiance: {confidence:.3f}")

                # Concepts détectés
                concepts = main_def.get('concepts', [])
                if concepts:
                    concept_labels = []
                    for concept in concepts[:3]:
                        if isinstance(concept, dict):
                            label = concept.get('label', '')
                            confidence = concept.get('confidence', 0)
                            concept_labels.append(f"{label} ({confidence:.2f})")
                        else:
                            concept_labels.append(str(concept))

                    lines.append(f"Concepts détectés: {', '.join(concept_labels)}")

                lines.append("")

    async def _add_local_context_section_v2(self, lines: List[str], context_data: Dict[str, Any]):
        """Ajoute la section de contexte local améliorée"""
        contexts = context_data.get('contexts', {})

        if 'local' in contexts and 'error' not in contexts['local']:
            local_ctx = contexts['local']

            # Dépendances USE
            immediate_deps = local_ctx.get('immediate_dependencies', [])
            if immediate_deps:
                lines.append("📦 DÉPENDANCES IMMÉDIATES")
                lines.append("-" * 40)
                for dep in immediate_deps:
                    dep_name = dep.get('name', 'N/A')
                    dep_type = dep.get('type', 'N/A')
                    lines.append(f"• {dep_name} ({dep_type})")

                    concepts = dep.get('concepts', [])
                    if concepts:
                        lines.append(f"  └─ Concepts: {', '.join(concepts[:3])}")

                    if dep.get('is_grouped'):
                        lines.append("  └─ 📦 Entité regroupée")
                lines.append("")

            # Fonctions appelées avec analyse enrichie
            called_functions = local_ctx.get('called_functions', [])
            if called_functions:
                lines.append("🔗 FONCTIONS APPELÉES")
                lines.append("-" * 40)
                for func in called_functions:
                    func_name = func.get('name', 'N/A')
                    resolved = func.get('resolved', False)
                    target_type = func.get('target_type', 'unknown')

                    status = "✅" if resolved else "❌"
                    lines.append(f"{status} {func_name} ({target_type})")

                    if func.get('signature'):
                        lines.append(f"  └─ {func['signature']}")

                    if func.get('summary'):
                        summary = func['summary'][:80] + "..." if len(func['summary']) > 80 else func['summary']
                        lines.append(f"  └─ {summary}")

                    if func.get('is_internal'):
                        lines.append("  └─ 🏠 Fonction interne")
                lines.append("")

            # Contexte parent
            parent_context = local_ctx.get('parent_context')
            if parent_context:
                lines.append("👨‍👩‍👧‍👦 CONTEXTE PARENT")
                lines.append("-" * 40)
                lines.append(f"Parent: {parent_context.get('name', 'N/A')} ({parent_context.get('type', 'N/A')})")

                if parent_context.get('concepts'):
                    lines.append(f"Concepts parent: {', '.join(parent_context['concepts'])}")

                if parent_context.get('is_grouped'):
                    lines.append("📦 Parent regroupé")
                lines.append("")

            # Contexte des enfants
            children_context = local_ctx.get('children_context', [])
            if children_context:
                lines.append("👶 ENTITÉS ENFANTS")
                lines.append("-" * 40)
                for child in children_context[:5]:  # Top 5
                    child_name = child.get('name', 'N/A')
                    child_type = child.get('type', 'N/A')
                    calls_count = child.get('called_functions_count', 0)

                    lines.append(f"• {child_name} ({child_type})")

                    if calls_count > 0:
                        lines.append(f"  └─ Appelle {calls_count} fonctions")

                    if child.get('is_internal'):
                        lines.append("  └─ 🏠 Interne")
                lines.append("")

    async def _add_global_context_section_v2(self, lines: List[str], context_data: Dict[str, Any]):
        """Ajoute la section de contexte global améliorée"""
        contexts = context_data.get('contexts', {})

        if 'global' in contexts and 'error' not in contexts['global']:
            global_ctx = contexts['global']

            # Analyse d'impact enrichie
            impact_analysis = global_ctx.get('impact_analysis', {})
            if impact_analysis:
                lines.append("💥 ANALYSE D'IMPACT")
                lines.append("-" * 40)

                risk_level = impact_analysis.get('risk_level', 'unknown')
                total_impact = impact_analysis.get('total_impact_entities', 0)

                risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk_level, "⚪")
                lines.append(f"Niveau de risque: {risk_emoji} {risk_level.upper()}")
                lines.append(f"Entités impactées: {total_impact}")

                direct_deps = impact_analysis.get('direct_dependents', [])
                if direct_deps:
                    lines.append(f"Dépendants directs: {', '.join(direct_deps[:5])}")
                    if len(direct_deps) > 5:
                        lines.append(f"  ... et {len(direct_deps) - 5} autres")

                affected_modules = impact_analysis.get('affected_modules', [])
                if affected_modules:
                    lines.append(f"Modules affectés: {', '.join(affected_modules)}")

                recommendations = impact_analysis.get('recommendations', [])
                if recommendations:
                    lines.append("Recommandations d'impact:")
                    for rec in recommendations[:3]:
                        lines.append(f"  • {rec}")

                lines.append("")

            # Vue d'ensemble du projet
            project_overview = global_ctx.get('project_overview', {})
            if project_overview:
                lines.append("🏗️ VUE D'ENSEMBLE DU PROJET")
                lines.append("-" * 40)

                stats = project_overview.get('statistics', {})
                if stats:
                    lines.append(f"Entités totales: {stats.get('total_entities', 0)}")
                    lines.append(f"Modules: {stats.get('modules', 0)} | "
                                 f"Fonctions: {stats.get('functions', 0)} | "
                                 f"Subroutines: {stats.get('subroutines', 0)}")

                arch_style = project_overview.get('architectural_style', '')
                if arch_style:
                    lines.append(f"Style architectural: {arch_style}")

                quality_metrics = project_overview.get('quality_metrics', {})
                if quality_metrics:
                    grouped_ratio = quality_metrics.get('grouped_entities_ratio', 0)
                    compression_ratio = quality_metrics.get('compression_ratio', 1)
                    lines.append(f"Entités regroupées: {grouped_ratio:.1%}")
                    lines.append(f"Ratio de compression: {compression_ratio:.1f}x")

                lines.append("")

    async def _add_semantic_context_section_v2(self, lines: List[str], context_data: Dict[str, Any]):
        """Ajoute la section de contexte sémantique améliorée"""
        contexts = context_data.get('contexts', {})

        if 'semantic' in contexts and 'error' not in contexts['semantic']:
            semantic_ctx = contexts['semantic']

            # Concepts principaux
            main_concepts = semantic_ctx.get('main_concepts', [])
            if main_concepts:
                lines.append("🧠 CONCEPTS PRINCIPAUX")
                lines.append("-" * 40)
                for concept in main_concepts:
                    if isinstance(concept, dict):
                        label = concept.get('label', '')
                        confidence = concept.get('confidence', 0)
                        category = concept.get('category', '')
                        method = concept.get('detection_method', '')

                        confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                        lines.append(f"{confidence_emoji} {label} ({confidence:.3f})")
                        lines.append(f"  └─ Catégorie: {category} | Méthode: {method}")
                    else:
                        lines.append(f"• {concept}")
                lines.append("")

            # Entités similaires
            similar_entities = semantic_ctx.get('similar_entities', [])
            if similar_entities:
                lines.append("🔄 ENTITÉS SIMILAIRES")
                lines.append("-" * 40)
                for similar in similar_entities[:5]:
                    name = similar.get('name', 'N/A')
                    similarity = similar.get('similarity', 0)
                    method = similar.get('method', 'unknown')

                    similarity_emoji = "🟢" if similarity > 0.8 else "🟡" if similarity > 0.6 else "🔴"
                    lines.append(f"{similarity_emoji} {name} (similarité: {similarity:.3f})")

                    reasons = similar.get('similarity_reasons', [])
                    if reasons:
                        lines.append(f"  └─ Raisons: {', '.join(reasons[:2])}")

                    lines.append(f"  └─ Méthode: {method}")
                lines.append("")

            # Patterns algorithmiques
            patterns = semantic_ctx.get('algorithmic_patterns', [])
            if patterns:
                lines.append("🔬 PATTERNS ALGORITHMIQUES")
                lines.append("-" * 40)
                for pattern in patterns[:3]:
                    pattern_name = pattern.get('pattern', 'N/A')
                    confidence = pattern.get('confidence', 0)
                    description = pattern.get('description', '')

                    confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                    lines.append(f"{confidence_emoji} {pattern_name} ({confidence:.3f})")

                    if description:
                        lines.append(f"  └─ {description}")

                    keywords = pattern.get('matched_keywords', [])
                    if keywords:
                        lines.append(f"  └─ Mots-clés: {', '.join(keywords[:4])}")
                lines.append("")

    async def _format_concept_group_detailed(self, context_data: Dict[str, Any],
                                             resolved_entity: Dict[str, Any]) -> str:
        """Formatage spécialisé pour les groupes de concepts"""
        lines = []
        concept_name = resolved_entity.get('concept_name', 'Unknown')

        lines.append("=" * 80)
        lines.append(f"📋 ANALYSE DE CONCEPT : {concept_name}")
        lines.append("=" * 80)
        lines.append(f"Concept recherché: {concept_name}")
        lines.append(f"Méthode de résolution: {resolved_entity.get('resolution_method', 'unknown')}")

        all_matches = resolved_entity.get('all_matches', [])
        total_matches = resolved_entity.get('total_matches', len(all_matches))

        lines.append(f"Entités contenant ce concept: {len(all_matches)} (total: {total_matches})")
        lines.append("")

        # Analyse des entités avec ce concept
        lines.append(f"🔗 ENTITÉS CONTENANT LE CONCEPT '{concept_name}'")
        lines.append("-" * 60)

        for i, match in enumerate(all_matches[:10], 1):  # Top 10
            entity = match['entity']
            confidence = match['confidence']
            matched_concept = match['matched_concept']

            lines.append(f"{i}. {entity.entity_name} ({entity.entity_type})")
            lines.append(f"   └─ Fichier: {entity.filename}")
            lines.append(f"   └─ Concept détecté: {matched_concept}")
            lines.append(f"   └─ Confiance: {confidence:.3f}")

            if entity.is_grouped:
                lines.append("   └─ 📦 Entité regroupée")

            lines.append("")

        # Statistiques du concept
        lines.append("📊 STATISTIQUES DU CONCEPT")
        lines.append("-" * 40)

        # Répartition par type
        type_distribution = {}
        for match in all_matches:
            entity_type = match['entity'].entity_type
            type_distribution[entity_type] = type_distribution.get(entity_type, 0) + 1

        lines.append("Répartition par type:")
        for entity_type, count in sorted(type_distribution.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  • {entity_type}: {count}")

        # Répartition par fichier
        file_distribution = {}
        for match in all_matches:
            filename = match['entity'].filename or 'Unknown'
            file_distribution[filename] = file_distribution.get(filename, 0) + 1

        lines.append("\nRépartition par fichier:")
        for filename, count in sorted(file_distribution.items(), key=lambda x: x[1], reverse=True)[:5]:
            lines.append(f"  • {filename}: {count}")

        lines.append("")
        lines.append("=" * 80)

        return '\n'.join(lines)

    async def _format_summary_context_v2(self, context_data: Dict[str, Any],
                                         resolved_entity: Dict[str, Any]) -> str:
        """Formatage résumé amélioré pour LLM"""
        lines = []
        entity_name = resolved_entity['name']
        entity_type = resolved_entity.get('type', 'unknown')

        lines.append(f"📋 RÉSUMÉ: {entity_name} ({entity_type})")
        lines.append("=" * 60)

        # Résumé exécutif
        summary = context_data.get('summary', {})
        if summary:
            entity_overview = summary.get('entity_overview', {})
            if entity_overview:
                signature = entity_overview.get('signature', '')
                if signature and signature != "Signature not found":
                    lines.append(f"Signature: {signature}")

                file_info = entity_overview.get('file', '')
                if file_info:
                    lines.append(f"Fichier: {file_info}")

                concepts = entity_overview.get('concepts', [])
                if concepts:
                    lines.append(f"Concepts: {', '.join(concepts[:3])}")

            complexity = summary.get('complexity_analysis', {})
            if complexity:
                level = complexity.get('level', 'unknown')
                deps_count = complexity.get('dependencies_count', 0)
                calls_count = complexity.get('function_calls_count', 0)
                lines.append(f"Complexité: {level} ({deps_count} deps, {calls_count} appels)")

        # Insights clés (résumé)
        key_insights = context_data.get('key_insights', [])
        if key_insights:
            lines.append("Insights clés:")
            for insight in key_insights[:3]:
                lines.append(f"• {insight}")

        # Recommandations (résumé)
        recommendations = context_data.get('recommendations', [])
        if recommendations:
            lines.append("Recommandations:")
            for rec in recommendations[:2]:
                lines.append(f"• {rec}")

        return '\n'.join(lines)

    async def _format_bullet_context_v2(self, context_data: Dict[str, Any],
                                        resolved_entity: Dict[str, Any]) -> str:
        """Formatage en bullet points amélioré"""
        lines = []
        entity_name = resolved_entity['name']
        entity_type = resolved_entity.get('type', 'unknown')

        lines.append(f"• ENTITÉ: {entity_name} ({entity_type})")

        # Points du résumé
        summary = context_data.get('summary', {})
        if summary:
            complexity = summary.get('complexity_analysis', {})
            if complexity:
                level = complexity.get('level', 'unknown')
                lines.append(f"• COMPLEXITÉ: {level}")

        # Points du contexte local
        contexts = context_data.get('contexts', {})
        if 'local' in contexts and 'error' not in contexts['local']:
            local_ctx = contexts['local']

            deps = local_ctx.get('immediate_dependencies', [])
            if deps:
                dep_names = [d.get('name', 'N/A') for d in deps[:3]]
                lines.append(f"• DÉPEND DE: {', '.join(dep_names)}")

            calls = local_ctx.get('called_functions', [])
            if calls:
                call_names = [c.get('name', 'N/A') for c in calls[:5]]
                lines.append(f"• APPELLE: {', '.join(call_names)}")

        # Points du contexte global
        if 'global' in contexts and 'error' not in contexts['global']:
            global_ctx = contexts['global']

            impact = global_ctx.get('impact_analysis', {})
            if impact:
                risk_level = impact.get('risk_level', 'unknown')
                total_impact = impact.get('total_impact_entities', 0)
                lines.append(f"• IMPACT: {risk_level} ({total_impact} entités)")

        # Points sémantiques
        if 'semantic' in contexts and 'error' not in contexts['semantic']:
            semantic_ctx = contexts['semantic']

            concepts = semantic_ctx.get('main_concepts', [])
            if concepts:
                concept_labels = [c.get('label', 'N/A') for c in concepts[:3]]
                lines.append(f"• CONCEPTS: {', '.join(concept_labels)}")

            patterns = semantic_ctx.get('algorithmic_patterns', [])
            if patterns:
                pattern_names = [p.get('pattern', 'N/A') for p in patterns[:2]]
                lines.append(f"• PATTERNS: {', '.join(pattern_names)}")

        # Insights et recommandations
        insights = context_data.get('key_insights', [])
        for insight in insights[:3]:
            lines.append(f"• INSIGHT: {insight}")

        recommendations = context_data.get('recommendations', [])
        for rec in recommendations[:2]:
            lines.append(f"• RECOMMANDATION: {rec}")

        return '\n'.join(lines)

    async def _generate_not_found_text(self, element_name: str) -> str:
        """Génère un message d'erreur informatif avec suggestions intelligentes"""
        lines = [
            "❌ ÉLÉMENT NON TROUVÉ",
            "=" * 50,
            f"Élément recherché: '{element_name}'",
            "",
            "💡 Suggestions:"
        ]

        # Utiliser EntityManager pour des suggestions intelligentes
        search_results = await self.orchestrator.search_entities(element_name)
        if search_results:
            lines.append("Entités similaires trouvées:")
            for result in search_results[:5]:
                confidence = result.get('confidence', 0)
                confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                lines.append(
                    f"  {confidence_emoji} {result['name']} ({result['type']}) - {result['file'].split('/')[-1]}")
        else:
            lines.append("  • Vérifiez l'orthographe")
            lines.append("  • Essayez avec un nom partiel")
            lines.append("  • Utilisez le nom du fichier")

        # Statistiques disponibles
        stats = self.entity_manager.get_stats()
        lines.extend([
            "",
            f"📊 Base de données disponible:",
            f"  • {stats['total_entities']} entités totales",
            f"  • {stats.get('entity_types', {}).get('module', 0)} modules",
            f"  • {stats.get('entity_types', {}).get('function', 0)} fonctions",
            f"  • {stats.get('entity_types', {}).get('subroutine', 0)} subroutines"
        ])

        return '\n'.join(lines)

    # === Méthodes de convenance pour usage externe ===

    async def get_quick_context(self, element_name: str) -> str:
        """Contexte rapide en format bullet points"""
        return await self.get_contextual_text(
            element_name,
            context_type="local",
            format_style="bullet_points",
            max_tokens=1000
        )

    async def get_full_context(self, element_name: str) -> str:
        """Contexte complet détaillé"""
        return await self.get_contextual_text(
            element_name,
            context_type="complete",
            format_style="detailed",
            max_tokens=6000
        )

    async def get_dependency_context(self, element_name: str) -> str:
        """Contexte focalisé sur les dépendances"""
        return await self.get_contextual_text(
            element_name,
            context_type="global",
            agent_perspective="analyzer",
            task_context="dependency_analysis",
            format_style="detailed",
            max_tokens=3000
        )

    async def get_semantic_context_text(self, element_name: str) -> str:
        """Contexte sémantique en texte"""
        return await self.get_contextual_text(
            element_name,
            context_type="semantic",
            format_style="detailed",
            max_tokens=2500
        )

    # === Nouvelles méthodes spécialisées ===

    async def get_concept_analysis(self, concept_name: str) -> str:
        """Analyse complète d'un concept spécifique"""
        return await self.get_contextual_text(
            concept_name,
            context_type="semantic",
            agent_perspective="analyzer",
            task_context="concept_analysis",
            format_style="detailed",
            max_tokens=4000
        )

    async def get_developer_context(self, element_name: str, task: str = "code_understanding") -> str:
        """Contexte optimisé pour développeurs"""
        return await self.get_contextual_text(
            element_name,
            context_type="complete",
            agent_perspective="developer",
            task_context=task,
            format_style="detailed",
            max_tokens=4000
        )

    async def get_reviewer_context(self, element_name: str, review_type: str = "code_review") -> str:
        """Contexte optimisé pour review de code"""
        return await self.get_contextual_text(
            element_name,
            context_type="complete",
            agent_perspective="reviewer",
            task_context=review_type,
            format_style="detailed",
            max_tokens=4500
        )

    def get_stats(self) -> Dict[str, Any]:
        """Statistiques du générateur de texte"""
        cache_stats = self.cache.get_all_stats()

        return {
            'initialized': self._initialized,
            'entity_manager_stats': self.entity_manager.get_stats() if self.entity_manager else {},
            'orchestrator_cache_stats': self.orchestrator.get_cache_stats() if self.orchestrator else {},
            'semantic_cache_stats': cache_stats.get('semantic_contexts', {})
        }