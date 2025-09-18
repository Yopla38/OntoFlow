"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# context_providers/smart_context_provider.py
import asyncio
import re
from typing import Dict, List, Any, Optional
import logging

from .entity_index import EntityIndex
from .local_context_provider import LocalContextProvider
from .global_context_provider import GlobalContextProvider
from .semantic_context_provider import SemanticContextProvider

logger = logging.getLogger(__name__)


class SmartContextProvider:
    """Orchestrateur intelligent qui combine tous les types de contextes"""

    def __init__(self, document_store, rag_engine):
        self.document_store = document_store
        self.rag_engine = rag_engine

        # Index partagé
        self.entity_index = EntityIndex(document_store)

        # Providers spécialisés
        self.local_provider = LocalContextProvider(document_store, rag_engine, self.entity_index)
        self.global_provider = GlobalContextProvider(document_store, rag_engine, self.entity_index)
        self.semantic_provider = SemanticContextProvider(document_store, rag_engine, self.entity_index)

        self._initialized = False

    async def initialize(self):
        """Initialisation une seule fois"""
        if self._initialized:
            return

        logger.info("🚀 Initialisation du SmartContextProvider...")
        await self.entity_index.build_index()

        logger.info("🔗 Construction du cache des appels de fonctions...")
        await self._ensure_call_patterns_cache()

        await self._ensure_entity_groups()

        stats = self.entity_index.get_stats()
        logger.info(f"✅ SmartContextProvider initialisé avec {stats['total_entities']} entités")

        self._initialized = True

    async def _build_comprehensive_function_call_cache_exact_copy(self):
        """
        COPIE EXACTE de la méthode du visualiseur qui MARCHE.
        """
        print("🔗 Construction du cache des appels de fonctions (copie exacte visualiseur)...")

        entity_index = self.entity_index

        # Patterns Fortran améliorés (EXACTEMENT comme le visualiseur)
        call_patterns = [
            # Appels de subroutines
            re.compile(r'\bcall\s+(\w+)', re.IGNORECASE),

            # Appels de fonctions (plus restrictifs)
            re.compile(r'=\s*(\w+)\s*\(', re.IGNORECASE),  # var = function(
            re.compile(r'\+\s*(\w+)\s*\(', re.IGNORECASE),  # ... + function(
            re.compile(r'-\s*(\w+)\s*\(', re.IGNORECASE),  # ... - function(
            re.compile(r'\*\s*(\w+)\s*\(', re.IGNORECASE),  # ... * function(
            re.compile(r'/\s*(\w+)\s*\(', re.IGNORECASE),  # ... / function(
            re.compile(r'\(\s*(\w+)\s*\(', re.IGNORECASE),  # (function(
            re.compile(r',\s*(\w+)\s*\(', re.IGNORECASE),  # , function(

            # Fonctions dans expressions
            re.compile(r'sqrt\s*\(\s*.*?(\w+)\s*\(', re.IGNORECASE),  # sqrt(...function(...)
            re.compile(r'if\s*\(\s*(\w+)\s*\(', re.IGNORECASE),  # if (function(

            # USE statements
            re.compile(r'use\s+(\w+)', re.IGNORECASE),
        ]

        # Mots-clés Fortran à ignorer (EXACTEMENT comme le visualiseur)
        fortran_keywords = {
            'if', 'then', 'else', 'endif', 'elseif', 'do', 'while', 'enddo', 'select',
            'case', 'where', 'forall', 'real', 'integer', 'logical', 'character', 'complex',
            'allocate', 'deallocate', 'nullify', 'write', 'read', 'print', 'open', 'close',
            'sqrt', 'sin', 'cos', 'exp', 'log', 'abs', 'max', 'min', 'sum', 'size', 'len',
            'trim', 'adjustl', 'adjustr', 'present', 'associated', 'allocated',
            'huge', 'tiny', 'epsilon', 'precision', 'range', 'digits',
            'modulo', 'mod', 'int', 'nint', 'floor', 'ceiling', 'aint', 'anint'
        }

        # Construire un index de toutes les entités disponibles (EXACTEMENT comme le visualiseur)
        all_entity_names = {}  # nom_lower -> (nom_original, type, fichier)

        for chunk_id, entity_info in entity_index.chunk_to_entity.items():
            name = entity_info.get('name', '')
            base_name = entity_info.get('base_name', '')
            entity_type = entity_info.get('type', '')
            filepath = entity_info.get('filepath', '')

            # Supprimer les suffixes _part_X pour la recherche
            clean_name = re.sub(r'_part_\d+$', '', name) if name else ''
            clean_base_name = re.sub(r'_part_\d+$', '', base_name) if base_name else ''

            if clean_name:
                all_entity_names[clean_name.lower()] = (clean_name, entity_type, filepath)
            if clean_base_name and clean_base_name != clean_name:
                all_entity_names[clean_base_name.lower()] = (clean_base_name, entity_type, filepath)

        print(f"📋 {len(all_entity_names)} entités disponibles pour la résolution des appels")

        # Analyser chaque chunk pour les appels (EXACTEMENT comme le visualiseur)
        total_calls_found = 0

        for chunk_id, entity_info in entity_index.chunk_to_entity.items():
            # Récupérer le texte du chunk
            try:
                chunk_text = await self._get_chunk_text_exact_copy(chunk_id)
                if not chunk_text:
                    continue
            except Exception as e:
                continue

            # Nettoyer le texte (EXACTEMENT comme le visualiseur)
            cleaned_text = self._remove_fortran_comments_exact_copy(chunk_text)

            # Détecter les appels
            detected_calls = set()

            for pattern in call_patterns:
                matches = pattern.findall(cleaned_text)
                for match in matches:
                    match_lower = match.lower()

                    # Filtrer les mots-clés Fortran
                    if match_lower in fortran_keywords:
                        continue

                    # Vérifier si c'est un nom d'entité connu
                    if match_lower in all_entity_names:
                        original_name, entity_type, filepath = all_entity_names[match_lower]
                        detected_calls.add(original_name)

                        # Debug pour les appels importants (EXACTEMENT comme le visualiseur)
                        if match_lower in ['random_gaussian', 'distance', 'lennard_jones_force',
                                           'apply_periodic_boundary']:
                            entity_name = entity_info.get('name', 'unknown')
                            print(f"✅ Appel détecté: {entity_name} → {original_name}")

            # Mettre à jour le cache
            if detected_calls:
                entity_index.call_patterns_cache[chunk_id] = list(detected_calls)
                total_calls_found += len(detected_calls)
            else:
                entity_index.call_patterns_cache[chunk_id] = []

        print(
            f"✅ Cache des appels construit: {total_calls_found} appels détectés dans {len(entity_index.call_patterns_cache)} chunks")

    async def _get_chunk_text_exact_copy(self, chunk_id: str) -> Optional[str]:
        """Copie exacte de la méthode du visualiseur"""
        try:
            # Parser le chunk_id pour extraire le document_id
            parts = chunk_id.split('-chunk-')
            if len(parts) != 2:
                return None

            document_id = parts[0]

            # Charger les chunks du document si nécessaire
            document_store = self.document_store
            await document_store.load_document_chunks(document_id)
            chunks = await document_store.get_document_chunks(document_id)

            if chunks:
                for chunk in chunks:
                    if chunk['id'] == chunk_id:
                        return chunk['text']

            return None

        except Exception as e:
            return None

    def _remove_fortran_comments_exact_copy(self, text: str) -> str:
        """Copie exacte de la méthode du visualiseur"""
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Trouver le premier '!' qui n'est pas dans une chaîne
            in_string = False
            quote_char = None
            comment_pos = -1

            i = 0
            while i < len(line):
                char = line[i]

                if not in_string:
                    if char in ['"', "'"]:
                        in_string = True
                        quote_char = char
                    elif char == '!':
                        comment_pos = i
                        break
                else:
                    if char == quote_char:
                        # Vérifier si ce n'est pas échappé
                        if i == 0 or line[i - 1] != '\\':
                            in_string = False
                            quote_char = None

                i += 1

            if comment_pos >= 0:
                line = line[:comment_pos]

            cleaned_lines.append(line.rstrip())

        return '\n'.join(cleaned_lines)

    async def _ensure_call_patterns_cache(self):
        """S'assurer que le cache est construit EXACTEMENT comme le visualiseur."""

        # Si le cache est vide, le construire avec la méthode exacte
        if not hasattr(self.entity_index, 'call_patterns_cache') or not self.entity_index.call_patterns_cache:
            # Utiliser notre copie exacte au lieu du visualiseur
            await self._build_comprehensive_function_call_cache_exact_copy()

            logger.info(f"✅ Cache des appels construit: {len(self.entity_index.call_patterns_cache)} chunks analysés")


    async def get_context_for_agent(self,
                                    entity_name: str,
                                    agent_type: str = "developer",
                                    task_context: str = "code_understanding",
                                    max_tokens: int = 4000) -> Dict[str, Any]:
        """Interface principale pour les agents IA"""

        if not self._initialized:
            await self.initialize()

        # 1. Déterminer la stratégie de contexte
        context_strategy = self._determine_context_strategy(agent_type, task_context)

        logger.info(f"🎯 Génération de contexte pour '{entity_name}' - Agent: {agent_type}, Tâche: {task_context}")
        logger.info(f"   Stratégie: Local({context_strategy['local_weight']:.1f}) "
                    f"Global({context_strategy['global_weight']:.1f}) "
                    f"Semantic({context_strategy['semantic_weight']:.1f})")

        # 2. Récupérer les contextes en parallèle
        contexts = {}
        tasks = []

        if context_strategy.get('local_weight', 0) > 0:
            local_tokens = int(max_tokens * context_strategy['local_weight'])
            tasks.append(('local', self.local_provider.get_local_context(entity_name, local_tokens)))

        if context_strategy.get('global_weight', 0) > 0:
            global_tokens = int(max_tokens * context_strategy['global_weight'])
            tasks.append(('global', self.global_provider.get_global_context(entity_name, global_tokens)))

        if context_strategy.get('semantic_weight', 0) > 0:
            semantic_tokens = int(max_tokens * context_strategy['semantic_weight'])
            tasks.append(('semantic', self.semantic_provider.get_semantic_context(entity_name, semantic_tokens)))

        # Exécuter les tâches en parallèle
        if tasks:
            results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)

            for i, (context_type, _) in enumerate(tasks):
                result = results[i]
                if isinstance(result, Exception):
                    logger.error(f"Erreur lors de la génération du contexte {context_type}: {result}")
                    contexts[context_type] = {"error": str(result)}
                else:
                    contexts[context_type] = result

        # 3. Fusionner et optimiser le contexte final
        final_context = await self._merge_contexts(contexts, context_strategy, max_tokens)

        # 4. Ajouter les métadonnées de génération
        final_context.update({
            "generation_info": {
                "agent_type": agent_type,
                "task_context": task_context,
                "strategy_used": context_strategy,
                "total_tokens": final_context.get("total_tokens", 0),
                "contexts_generated": list(contexts.keys()),
                "entity_index_stats": self.entity_index.get_stats()
            }
        })

        logger.info(f"✅ Contexte généré: {final_context.get('total_tokens', 0)} tokens, "
                    f"types: {list(contexts.keys())}")

        return final_context

    def _determine_context_strategy(self, agent_type: str, task_context: str) -> Dict[str, float]:
        """Détermine la stratégie de contexte selon l'agent et la tâche"""

        strategies = {
            # === Développeur ===
            ("developer", "code_understanding"): {
                "local_weight": 0.6,  # Focus sur les dépendances immédiates
                "global_weight": 0.3,  # Vue d'ensemble modérée
                "semantic_weight": 0.1  # Peu de contexte sémantique
            },

            ("developer", "debugging"): {
                "local_weight": 0.7,  # Focus fort sur le local
                "global_weight": 0.2,  # Vue d'ensemble limitée
                "semantic_weight": 0.1  # Patterns d'erreurs
            },

            ("developer", "refactoring"): {
                "local_weight": 0.4,  # Dépendances importantes
                "global_weight": 0.4,  # Impact global crucial
                "semantic_weight": 0.2  # Patterns similaires utiles
            },

            ("developer", "optimization"): {
                "local_weight": 0.3,  # Code spécifique
                "global_weight": 0.2,  # Architecture
                "semantic_weight": 0.5  # Patterns d'optimisation
            },

            # === Reviewer ===
            ("reviewer", "code_review"): {
                "local_weight": 0.3,  # Code spécifique
                "global_weight": 0.4,  # Impact architectural
                "semantic_weight": 0.3  # Patterns et bonnes pratiques
            },

            ("reviewer", "architecture_review"): {
                "local_weight": 0.2,  # Moins de détails locaux
                "global_weight": 0.6,  # Focus sur l'architecture
                "semantic_weight": 0.2  # Patterns architecturaux
            },

            ("reviewer", "performance_review"): {
                "local_weight": 0.4,  # Algorithmes spécifiques
                "global_weight": 0.2,  # Impact global
                "semantic_weight": 0.4  # Patterns de performance
            },

            # === Analyzer ===
            ("analyzer", "bug_detection"): {
                "local_weight": 0.7,  # Focus sur les appels directs
                "global_weight": 0.2,  # Dépendances
                "semantic_weight": 0.1  # Patterns de bugs
            },

            ("analyzer", "dependency_analysis"): {
                "local_weight": 0.3,  # Code local
                "global_weight": 0.7,  # Graphe de dépendances
                "semantic_weight": 0.0  # Pas besoin de sémantique
            },

            ("analyzer", "impact_analysis"): {
                "local_weight": 0.2,  # Détails limités
                "global_weight": 0.8,  # Impact global crucial
                "semantic_weight": 0.0  # Analyse factuelle
            },

            # === Documentation ===
            ("documenter", "api_documentation"): {
                "local_weight": 0.5,  # Interface et signature
                "global_weight": 0.3,  # Contexte d'utilisation
                "semantic_weight": 0.2  # Exemples similaires
            },

            ("documenter", "tutorial_writing"): {
                "local_weight": 0.3,  # Code d'exemple
                "global_weight": 0.2,  # Vue d'ensemble
                "semantic_weight": 0.5  # Patterns et exemples
            },

            # === Maintainer ===
            ("maintainer", "legacy_understanding"): {
                "local_weight": 0.4,  # Code spécifique
                "global_weight": 0.4,  # Architecture historique
                "semantic_weight": 0.2  # Patterns de l'époque
            },

            ("maintainer", "migration_planning"): {
                "local_weight": 0.2,  # Moins de détails
                "global_weight": 0.6,  # Vue d'ensemble cruciale
                "semantic_weight": 0.2  # Patterns de migration
            }
        }

        # Stratégie par défaut
        default_strategy = {
            "local_weight": 0.5,
            "global_weight": 0.3,
            "semantic_weight": 0.2
        }

        strategy = strategies.get((agent_type, task_context), default_strategy)

        # Normaliser les poids pour s'assurer qu'ils totalisent 1.0
        total_weight = sum(strategy.values())
        if total_weight > 0:
            strategy = {k: v / total_weight for k, v in strategy.items()}

        return strategy

    async def _merge_contexts(self, contexts: Dict[str, Dict[str, Any]],
                              strategy: Dict[str, float], max_tokens: int) -> Dict[str, Any]:
        """Fusionne et optimise les contextes selon la stratégie"""

        merged_context = {
            "entity": None,
            "contexts": contexts,
            "summary": {},
            "key_insights": [],
            "recommendations": [],
            "total_tokens": 0
        }

        # Extraire l'entité principale
        for context_type, context_data in contexts.items():
            if not isinstance(context_data, dict) or 'error' in context_data:
                continue

            if context_data.get('entity') and not merged_context["entity"]:
                merged_context["entity"] = context_data['entity']
                break

        # Créer le résumé consolidé
        merged_context["summary"] = await self._create_consolidated_summary(contexts, strategy)

        # Extraire les insights clés
        merged_context["key_insights"] = await self._extract_key_insights(contexts, strategy)

        # Générer des recommandations
        merged_context["recommendations"] = await self._generate_recommendations(contexts, strategy)

        # Calculer les tokens utilisés
        total_tokens = 0
        for context_data in contexts.values():
            if isinstance(context_data, dict) and 'tokens_used' in context_data:
                total_tokens += context_data['tokens_used']

        merged_context["total_tokens"] = total_tokens

        # Optimiser si on dépasse la limite
        if total_tokens > max_tokens:
            merged_context = await self._optimize_context_size(merged_context, max_tokens)

        return merged_context

    async def _extract_key_insights(self, contexts: Dict[str, Dict[str, Any]],
                                    strategy: Dict[str, float]) -> List[str]:
        """Extrait les insights clés de tous les contextes"""

        insights = []

        # Insights du contexte local
        if 'local' in contexts and 'error' not in contexts['local']:
            local_ctx = contexts['local']

            # Analyser la définition principale
            main_def = local_ctx.get('main_definition', {})
            entity_type = main_def.get('type', '')
            entity_name = main_def.get('name', '')

            # Insights spécifiques par type
            if entity_type == 'subroutine':
                calls_count = len(local_ctx.get('called_functions', []))
                if calls_count > 5:
                    insights.append(f"Complex subroutine: calls {calls_count} functions")
                elif calls_count == 0:
                    insights.append("Simple subroutine: no function calls detected")
                else:
                    insights.append(f"Moderate complexity: {calls_count} function calls")

            elif entity_type == 'module':
                children = len(local_ctx.get('children_context', []))
                deps = len(local_ctx.get('immediate_dependencies', []))
                if children > 5:
                    insights.append(f"Large module: contains {children} procedures")
                if deps > 3:
                    insights.append(f"High coupling: depends on {deps} modules")
                elif deps == 0:
                    insights.append("Self-contained module: no external dependencies")

            elif entity_type in ['type', 'type_definition']:
                insights.append("Data structure definition: check for proper encapsulation")

            # Analyser les patterns dans le nom
            if entity_name:
                name_lower = entity_name.lower()
                if 'compute' in name_lower:
                    insights.append("Computational routine: potential optimization target")
                elif 'init' in name_lower:
                    insights.append("Initialization routine: critical for system setup")
                elif 'solve' in name_lower:
                    insights.append("Solver routine: likely performance-critical")
                elif 'part_' in name_lower:
                    insights.append("Split entity: part of a larger function")

        # Insights du contexte global
        if 'global' in contexts and 'error' not in contexts['global']:
            global_ctx = contexts['global']

            # Analyse d'impact
            if 'impact_analysis' in global_ctx:
                impact = global_ctx['impact_analysis']
                risk_level = impact.get('risk_level', '')

                if risk_level == 'high':
                    insights.append("High impact entity: changes affect many components")
                elif risk_level == 'low':
                    insights.append("Low impact entity: safe to modify")

            # Vue d'ensemble du projet
            if 'project_overview' in global_ctx:
                overview = global_ctx['project_overview']
                arch_style = overview.get('architectural_style', '')
                if arch_style:
                    insights.append(f"Project architecture: {arch_style}")

        # Insights du contexte sémantique
        if 'semantic' in contexts and 'error' not in contexts['semantic']:
            semantic_ctx = contexts['semantic']

            patterns = semantic_ctx.get('algorithmic_patterns', [])
            if patterns:
                top_pattern = patterns[0]
                pattern_name = top_pattern.get('pattern', '')
                if pattern_name:
                    insights.append(f"Algorithmic pattern: {pattern_name}")

            similar_count = len(semantic_ctx.get('similar_entities', []))
            if similar_count > 3:
                insights.append(f"Common pattern: {similar_count} similar entities found")

        return insights[:8]  # Limiter à 8 insights

    async def _generate_recommendations(self, contexts: Dict[str, Dict[str, Any]],
                                        strategy: Dict[str, float]) -> List[str]:
        """Génère des recommandations basées sur l'analyse des contextes"""

        recommendations = []

        # Recommandations du contexte local
        if 'local' in contexts and 'error' not in contexts['local']:
            local_ctx = contexts['local']
            main_def = local_ctx.get('main_definition', {})
            entity_type = main_def.get('type', '')
            entity_name = main_def.get('name', '')

            # Recommandations spécifiques par type
            if entity_type == 'subroutine':
                calls_count = len(local_ctx.get('called_functions', []))
                if calls_count > 6:
                    recommendations.append("Consider breaking down into smaller subroutines")

                signature = main_def.get('signature', '').lower()
                if 'intent(inout)' in signature:
                    recommendations.append("Review intent(inout) parameters for side effects")

            elif entity_type == 'module':
                deps_count = len(local_ctx.get('immediate_dependencies', []))
                if deps_count > 4:
                    recommendations.append("High dependency count - consider refactoring")
                elif deps_count == 0:
                    recommendations.append("Self-contained module - good for reusability")

            # Recommandations basées sur les noms
            if entity_name:
                name_lower = entity_name.lower()
                if 'temp' in name_lower or 'tmp' in name_lower:
                    recommendations.append("Temporary naming - consider more descriptive names")

                if 'part_' in name_lower:
                    recommendations.append("Split entity - consider reviewing chunking strategy")

                if name_lower.startswith('compute_'):
                    recommendations.append("Computational routine - profile for performance")

        # Recommandations du contexte global
        if 'global' in contexts and 'error' not in contexts['global']:
            global_ctx = contexts['global']

            if 'impact_analysis' in global_ctx:
                impact = global_ctx['impact_analysis']

                for rec in impact.get('recommendations', []):
                    recommendations.append(rec)

        # Recommandations du contexte sémantique
        if 'semantic' in contexts and 'error' not in contexts['semantic']:
            semantic_ctx = contexts['semantic']

            similar_entities = semantic_ctx.get('similar_entities', [])
            if len(similar_entities) > 2:
                recommendations.append("Review similar entities for consistency")

            patterns = semantic_ctx.get('algorithmic_patterns', [])
            if patterns:
                pattern_name = patterns[0].get('pattern', '')
                if pattern_name:
                    recommendations.append(f"Consider optimizations for {pattern_name} pattern")

        return recommendations[:6]  # Limiter à 6 recommandations

    async def _create_consolidated_summary(self, contexts: Dict[str, Dict[str, Any]],
                                           strategy: Dict[str, float]) -> Dict[str, Any]:
        """Crée un résumé consolidé des différents contextes"""

        summary = {
            "entity_overview": {},
            "key_dependencies": [],
            "architectural_role": "",
            "main_concepts": [],
            "complexity_indicators": {}
        }

        # Résumé de l'entité depuis le contexte local
        if 'local' in contexts and 'error' not in contexts['local']:
            definition = contexts['local'].get('main_definition', {})
            summary["entity_overview"] = {
                "name": definition.get('name', ''),
                "type": definition.get('type', ''),
                "location": definition.get('location', {}),
                "signature": definition.get('signature', ''),
                "concepts": definition.get('concepts', [])
            }

            # Indicateurs de complexité
            calls_count = len(contexts['local'].get('called_functions', []))
            deps_count = len(contexts['local'].get('immediate_dependencies', []))

            summary["complexity_indicators"] = {
                "function_calls": calls_count,
                "dependencies": deps_count,
                "complexity_level": "high" if calls_count > 5 else "medium" if calls_count > 2 else "low"
            }

        # Dépendances clés depuis le contexte global
        if 'global' in contexts and 'error' not in contexts['global']:
            dep_graph = contexts['global'].get('dependency_graph', {})
            summary["key_dependencies"] = dep_graph.get('summary', {})

            # Rôle architectural
            impact = contexts['global'].get('impact_analysis', {})
            if impact:
                risk_level = impact.get('risk_level', 'unknown')
                affected_modules = len(impact.get('affected_modules', []))
                summary["architectural_role"] = f"Risk: {risk_level}, affects {affected_modules} modules"

        # Concepts principaux depuis le contexte sémantique
        if 'semantic' in contexts and 'error' not in contexts['semantic']:
            concepts = contexts['semantic'].get('main_concepts', [])
            summary["main_concepts"] = concepts[:5]  # Top 5 concepts

        return summary

    async def _optimize_context_size(self, context: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        """Optimise la taille du contexte si nécessaire"""

        # Stratégie simple : réduire les listes longues
        contexts = context.get('contexts', {})

        for context_type, context_data in contexts.items():
            if not isinstance(context_data, dict):
                continue

            # Réduire les listes dans chaque contexte
            for key, value in context_data.items():
                if isinstance(value, list) and len(value) > 10:
                    context_data[key] = value[:10]  # Garder seulement les 10 premiers

        # Recalculer les tokens (approximation)
        total_size = len(str(context))
        estimated_tokens = total_size // 4  # Approximation

        context["total_tokens"] = estimated_tokens
        context["optimized"] = estimated_tokens != context.get("original_tokens", estimated_tokens)

        return context

    # === Méthodes d'accès direct aux providers ===

    async def get_local_context(self, entity_name: str, max_tokens: int = 2000) -> Dict[str, Any]:
        """Accès direct au contexte local"""
        if not self._initialized:
            await self.initialize()
        return await self.local_provider.get_local_context(entity_name, max_tokens)

    async def get_global_context(self, entity_name: str, max_tokens: int = 3000) -> Dict[str, Any]:
        """Accès direct au contexte global"""
        if not self._initialized:
            await self.initialize()
        return await self.global_provider.get_global_context(entity_name, max_tokens)

    async def get_semantic_context(self, entity_name: str, max_tokens: int = 2000) -> Dict[str, Any]:
        """Accès direct au contexte sémantique"""
        if not self._initialized:
            await self.initialize()
        return await self.semantic_provider.get_semantic_context(entity_name, max_tokens)

    # === Utilitaires ===

    async def search_entities(self, query: str) -> List[Dict[str, Any]]:
        """Recherche d'entités par nom ou pattern"""
        if not self._initialized:
            await self.initialize()

        results = []

        # Recherche exacte
        chunks = await self.entity_index.find_entity(query)
        if chunks:
            for chunk_id in chunks:
                entity_info = await self.entity_index.get_entity_info(chunk_id)
                if entity_info:
                    results.append({
                        "name": entity_info['name'],
                        "type": entity_info['type'],
                        "file": entity_info.get('filepath', ''),
                        "match_type": "exact"
                    })

        # Recherche fuzzy si pas de résultats exacts
        if not results:
            all_names = list(self.entity_index.name_to_chunks.keys())
            fuzzy_matches = [name for name in all_names
                             if query.lower() in name.lower()]

            for name in fuzzy_matches[:10]:
                chunks = await self.entity_index.find_entity(name)
                if chunks:
                    entity_info = await self.entity_index.get_entity_info(chunks[0])
                    if entity_info:
                        results.append({
                            "name": entity_info['name'],
                            "type": entity_info['type'],
                            "file": entity_info.get('filepath', ''),
                            "match_type": "fuzzy"
                        })

        return results

    def get_index_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'index"""
        return self.entity_index.get_stats()

    def debug_entity(self, entity_name: str) -> Dict[str, Any]:
        """Informations de debug pour une entité"""
        return self.entity_index.debug_entity(entity_name)

    async def _ensure_entity_groups(self):
        """S'assure que les entity_groups sont créés (même logique que le visualiseur)."""

        if hasattr(self, 'entity_groups') and self.entity_groups:
            return

        logger.info("📦 Création des groupes d'entités (logique visualiseur)...")

        # Utiliser exactement la même logique que le visualiseur
        await self._group_split_chunks()

        logger.info(f"✅ {len(self.entity_groups)} groupes d'entités créés")

    async def _group_split_chunks(self):
        """
        COPIE EXACTE de la méthode du visualiseur.
        Regroupe les chunks splittés en entités complètes.
        """
        logger.info("📦 Regroupement des chunks splittés...")

        # Initialiser
        self.chunk_to_entity_mapping = {}
        self.entity_groups = {}

        # Parcourir tous les chunks (même logique que le visualiseur)
        for chunk_id, entity_info in self.entity_index.chunk_to_entity.items():

            # Déterminer l'ID de l'entité complète (même logique)
            if entity_info.get('is_partial', False):
                entity_key = entity_info.get('parent_entity_id')
                if not entity_key:
                    entity_key = entity_info.get('base_entity_name') or entity_info.get('entity_name', 'unknown')
            else:
                entity_key = entity_info.get('entity_id')
                if not entity_key:
                    base_name = entity_info.get('base_entity_name') or entity_info.get('entity_name', 'unknown')
                    filepath = entity_info.get('filepath', '')
                    start_line = entity_info.get('start_line', 0)
                    entity_key = f"{filepath}#{base_name}#{start_line}"

            # Stocker le mapping chunk -> entité
            self.chunk_to_entity_mapping[chunk_id] = entity_key

            # Initialiser le groupe d'entité s'il n'existe pas (même logique)
            if entity_key not in self.entity_groups:
                entity_bounds = entity_info.get('entity_bounds', {})

                self.entity_groups[entity_key] = {
                    'entity_id': entity_key,
                    'entity_name': entity_info.get('base_name') or entity_info.get('name', 'unknown'),
                    'entity_type': entity_info.get('type', 'code'),
                    'filepath': entity_info.get('filepath', ''),
                    'filename': entity_info.get('filename', 'Unknown'),
                    'chunks': [],
                    'chunk_ids': set(),
                    'all_dependencies': set(),
                    'all_concepts': set(),
                    'all_matched_concepts': set(),
                    'best_score': 0,
                    'total_score': 0,
                    'entity_start': entity_bounds.get('start_line') or entity_info.get('start_line'),
                    'entity_end': entity_bounds.get('end_line') or entity_info.get('end_line'),
                    'expected_parts': entity_info.get('total_parts', 1),
                    'is_internal': entity_info.get('is_internal_function', False),
                    'parent_entity': entity_info.get('parent_entity_name', ''),
                    'parent_entity_type': entity_info.get('parent_entity_type', ''),
                    'qualified_name': entity_info.get('full_qualified_name', ''),
                }

            # Ajouter le chunk au groupe (même logique)
            group = self.entity_groups[entity_key]
            group['chunks'].append({
                'chunk_id': chunk_id,
                'entity_info': entity_info,
                'part_index': entity_info.get('part_index', 0),
                'part_sequence': entity_info.get('part_sequence', 0)
            })
            group['chunk_ids'].add(chunk_id)

            # Agréger les dépendances (même logique)
            dependencies = entity_info.get('dependencies', [])
            if isinstance(dependencies, list):
                group['all_dependencies'].update(dependencies)

            # Agréger les concepts (même logique)
            concepts = entity_info.get('concepts', [])
            if isinstance(concepts, list):
                for concept in concepts:
                    if isinstance(concept, dict):
                        group['all_concepts'].add(concept.get('label', str(concept)))
                    else:
                        group['all_concepts'].add(str(concept))

            # Mettre à jour les bounds
            if entity_info.get('start_line'):
                if not group['entity_start'] or entity_info['start_line'] < group['entity_start']:
                    group['entity_start'] = entity_info['start_line']

            if entity_info.get('end_line'):
                if not group['entity_end'] or entity_info['end_line'] > group['entity_end']:
                    group['entity_end'] = entity_info['end_line']

        # Post-traitement : trier les chunks
        for entity_key, group in self.entity_groups.items():
            group['chunks'].sort(
                key=lambda x: (
                    x['entity_info'].get('part_sequence', 0),
                    x['entity_info'].get('start_line', 0)
                )
            )
            group['is_complete'] = len(group['chunks']) == group['expected_parts'] or group['expected_parts'] == 1

        logger.info(f"✅ Regroupement terminé: {len(self.entity_groups)} entités complètes créées")

    def find_entity_callers(self, entity_name: str) -> List[Dict[str, Any]]:
        """
        Trouve qui appelle une entité donnée.
        UTILISE LA MÊME LOGIQUE QUE LE VISUALISEUR.
        """

        if not hasattr(self, 'entity_groups'):
            # Les groupes ne sont pas encore créés
            return []

        callers = []

        # Créer un mapping nom -> clé (même logique que le visualiseur)
        name_to_key_map = {}
        for key, group in self.entity_groups.items():
            group_name = group['entity_name'].lower()
            name_to_key_map[group_name] = key

        # Trouver la clé de l'entité cible
        target_key = name_to_key_map.get(entity_name.lower())

        # Parcourir tous les groupes pour trouver qui appelle l'entité cible
        for source_key, source_group in self.entity_groups.items():
            if source_key == target_key:  # Skip l'entité elle-même
                continue

            source_name = source_group['entity_name']

            # Collecter tous les appels depuis les chunks de cette entité (même logique)
            all_called_functions = set()

            for chunk_info in source_group.get('chunks', []):
                chunk_id = chunk_info.get('chunk_id', '')
                if chunk_id and hasattr(self.entity_index, 'call_patterns_cache'):
                    calls = self.entity_index.call_patterns_cache.get(chunk_id, [])
                    all_called_functions.update(calls)

            # Vérifier si l'entité cible est appelée
            if entity_name.lower() in [call.lower() for call in all_called_functions]:
                callers.append({
                    'name': source_name,
                    'type': source_group['entity_type'],
                    'file': source_group.get('filepath', ''),
                    'entity_key': source_key
                })

        return callers

    def find_entity_calls(self, entity_name: str) -> List[str]:
        """
        Trouve ce qu'appelle une entité donnée.
        UTILISE LA MÊME LOGIQUE QUE LE VISUALISEUR.
        """

        if not hasattr(self, 'entity_groups'):
            return []

        # Trouver le groupe de l'entité
        target_group = None
        for key, group in self.entity_groups.items():
            if group['entity_name'].lower() == entity_name.lower():
                target_group = group
                break

        if not target_group:
            return []

        # Collecter tous les appels (même logique que le visualiseur)
        all_called_functions = set()

        for chunk_info in target_group.get('chunks', []):
            chunk_id = chunk_info.get('chunk_id', '')
            if chunk_id and hasattr(self.entity_index, 'call_patterns_cache'):
                calls = self.entity_index.call_patterns_cache.get(chunk_id, [])
                all_called_functions.update(calls)

        return list(all_called_functions)
