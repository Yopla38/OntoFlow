"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# providers/local_context.py
"""
LocalContextProvider refactorisé utilisant les composants unifiés.
Simplifié et optimisé grâce aux phases 1-2.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from .base_provider import BaseContextProvider

logger = logging.getLogger(__name__)


class LocalContextProvider(BaseContextProvider):
    """
    Fournit le contexte local : dépendances immédiates d'une entité.
    Version refactorisée utilisant EntityManager + FortranAnalyzer.
    """

    async def get_context(self, entity_name: str, max_tokens: int = 2000) -> Dict[str, Any]:
        """Interface principale compatible avec l'ancien code"""
        return await self.get_local_context(entity_name, max_tokens)

    async def get_local_context(self, entity_name: str, max_tokens: int = 2000) -> Dict[str, Any]:
        """
        Récupère le contexte local d'une entité.
        Version simplifiée grâce aux composants unifiés.
        """
        await self._ensure_initialized()

        # 1. Résoudre l'entité
        resolved_entity = await self.resolve_entity(entity_name)
        if not resolved_entity:
            return await self.create_error_context(
                entity_name,
                f"Entity '{entity_name}' not found"
            )

        entity = resolved_entity['entity_object']

        # Préparer le contexte de base
        context = {
            'entity': entity_name,
            'type': 'local',
            'main_definition': {},
            'immediate_dependencies': [],
            'called_functions': [],
            'parent_context': None,
            'children_context': [],
            'file_context': {},
            'tokens_used': 0
        }

        tokens_budget = max_tokens

        # 2. Définition principale (utilise BaseProvider)
        main_definition = await self.get_entity_definition(entity_name)
        if main_definition:
            context['main_definition'] = main_definition
            tokens_budget -= self.calculate_tokens_used(main_definition)

        # 3. Dépendances immédiates (optimisé)
        if entity.dependencies and tokens_budget > 200:
            context['immediate_dependencies'] = await self.get_dependency_contexts(
                list(entity.dependencies), int(tokens_budget * 0.3)
            )
            tokens_budget -= self.calculate_tokens_used(context['immediate_dependencies'])

        # 4. Fonctions appelées (utilise FortranAnalyzer)
        if tokens_budget > 300:
            context['called_functions'] = await self.find_function_calls_with_analysis(entity_name)
            tokens_budget -= self.calculate_tokens_used(context['called_functions'])

        # 5. Contexte parent si fonction interne
        if entity.parent_entity and tokens_budget > 200:
            context['parent_context'] = await self._get_parent_context_optimized(
                entity.parent_entity, int(tokens_budget * 0.2)
            )
            tokens_budget -= self.calculate_tokens_used(context['parent_context'])

        # 6. Contexte des enfants
        if tokens_budget > 150:
            children = await self.entity_manager.get_children(entity.entity_id)
            if children:
                context['children_context'] = await self._get_children_context_optimized(
                    children, int(tokens_budget * 0.2)
                )
                tokens_budget -= self.calculate_tokens_used(context['children_context'])

        # 7. Contexte du fichier
        if entity.filepath and tokens_budget > 100:
            context['file_context'] = await self.get_file_context(
                entity.filepath, entity_name, int(tokens_budget * 0.1)
            )

        context['tokens_used'] = max_tokens - tokens_budget

        return context

    async def _create_entity_summary(self, entity, max_tokens: int) -> str:
        """
        Crée un résumé d'entité pour le contexte local.
        Version simplifiée qui délègue à BaseProvider mais avec logique locale.
        """
        try:
            # Utiliser la méthode sécurisée du BaseProvider
            return await self._create_entity_summary_safe(entity, max_tokens)

        except Exception as e:
            logger.debug(f"Erreur création résumé entité: {e}")
            # Fallback : créer un résumé minimal
            entity_name = getattr(entity, 'entity_name', 'unknown')
            entity_type = getattr(entity, 'entity_type', 'unknown')
            return f"{entity_type.title()} {entity_name}"

    async def _create_entity_summary_local(self, entity, max_tokens: int) -> str:
        """
        Crée un résumé spécialisé pour le contexte local avec plus de détails.
        """
        try:
            entity_name = getattr(entity, 'entity_name', 'unknown')
            entity_type = getattr(entity, 'entity_type', 'unknown')

            summary_parts = [f"{entity_type.title()} {entity_name}"]

            # Ajouter la signature si disponible
            signature = getattr(entity, 'signature', '')
            if signature and signature != "Signature not found":
                # Tronquer la signature si trop longue
                if len(signature) > 100:
                    signature = signature[:100] + "..."
                summary_parts.append(f"Signature: {signature}")

            # Ajouter les dépendances principales
            dependencies = getattr(entity, 'dependencies', set())
            if dependencies:
                deps_list = list(dependencies)[:3]  # Top 3
                summary_parts.append(f"Uses: {', '.join(deps_list)}")

            # Ajouter les appels de fonctions principaux
            called_functions = getattr(entity, 'called_functions', set())
            if called_functions:
                calls_list = list(called_functions)[:3]  # Top 3
                summary_parts.append(f"Calls: {', '.join(calls_list)}")

            # Ajouter des informations sur la complexité
            if entity_type in ['subroutine', 'function']:
                complexity_info = []

                if len(called_functions) > 5:
                    complexity_info.append(f"{len(called_functions)} function calls")

                if len(dependencies) > 2:
                    complexity_info.append(f"{len(dependencies)} dependencies")

                if complexity_info:
                    summary_parts.append(f"Complexity: {', '.join(complexity_info)}")

            # Ajouter des concepts si disponibles
            concepts = getattr(entity, 'concepts', set())
            if concepts:
                concepts_list = list(concepts)[:2]  # Top 2
                summary_parts.append(f"Concepts: {', '.join(concepts_list)}")

            # Joindre et vérifier la longueur
            full_summary = '; '.join(summary_parts)

            # Approximation : 4 caractères par token
            if len(full_summary) > max_tokens * 4:
                # Tronquer intelligemment
                truncated = full_summary[:max_tokens * 4]
                last_semicolon = truncated.rfind(';')
                if last_semicolon > len(truncated) * 0.7:  # Si le point-virgule est vers la fin
                    full_summary = truncated[:last_semicolon]
                else:
                    full_summary = truncated + "..."

            return full_summary

        except Exception as e:
            logger.debug(f"Erreur création résumé local: {e}")
            # Fallback minimal
            entity_name = getattr(entity, 'entity_name', 'unknown')
            entity_type = getattr(entity, 'entity_type', 'unknown')
            return f"{entity_type.title()} {entity_name}"

    # OPTIONNEL: Méthode encore plus détaillée pour les contextes parents importants
    async def _create_detailed_entity_summary(self, entity, max_tokens: int) -> str:
        """
        Crée un résumé très détaillé pour les entités importantes (comme les parents).
        """
        try:
            entity_name = getattr(entity, 'entity_name', 'unknown')
            entity_type = getattr(entity, 'entity_type', 'unknown')
            filepath = getattr(entity, 'filepath', '')

            summary_parts = []

            # En-tête avec localisation
            if filepath:
                filename = Path(filepath).name
                summary_parts.append(f"{entity_type.title()} {entity_name} in {filename}")
            else:
                summary_parts.append(f"{entity_type.title()} {entity_name}")

            # Signature complète
            signature = getattr(entity, 'signature', '')
            if signature and signature != "Signature not found":
                summary_parts.append(f"Signature: {signature}")

            # Analyse des enfants pour les modules/conteneurs
            if hasattr(self, 'entity_manager'):
                try:
                    children = await self.entity_manager.get_children(entity.entity_id)
                    if children:
                        child_types = {}
                        for child in children:
                            child_type = child.entity_type
                            child_types[child_type] = child_types.get(child_type, 0) + 1

                        child_summary = []
                        for child_type, count in child_types.items():
                            if count == 1:
                                child_summary.append(f"1 {child_type}")
                            else:
                                child_summary.append(f"{count} {child_type}s")

                        summary_parts.append(f"Contains: {', '.join(child_summary)}")
                except Exception:
                    pass  # Skip if entity_manager not available

            # Dépendances avec plus de détails
            dependencies = getattr(entity, 'dependencies', set())
            if dependencies:
                deps_count = len(dependencies)
                if deps_count <= 3:
                    summary_parts.append(f"Uses: {', '.join(list(dependencies))}")
                else:
                    main_deps = list(dependencies)[:3]
                    summary_parts.append(f"Uses: {', '.join(main_deps)} and {deps_count - 3} others")

            # Appels de fonctions avec analyse
            called_functions = getattr(entity, 'called_functions', set())
            if called_functions:
                calls_count = len(called_functions)
                if calls_count <= 3:
                    summary_parts.append(f"Calls: {', '.join(list(called_functions))}")
                else:
                    main_calls = list(called_functions)[:3]
                    summary_parts.append(f"Calls: {', '.join(main_calls)} and {calls_count - 3} others")

            # Informations sur la complexité et qualité
            quality_indicators = []

            # Groupement
            is_grouped = getattr(entity, 'is_grouped', False)
            if is_grouped:
                quality_indicators.append("multi-part entity")

            # Confiance
            confidence = getattr(entity, 'confidence', 1.0)
            if confidence < 0.8:
                quality_indicators.append(f"confidence: {confidence:.2f}")

            # Source de parsing
            source_method = getattr(entity, 'source_method', '')
            if source_method and source_method != 'hybrid':
                quality_indicators.append(f"parsed via {source_method}")

            if quality_indicators:
                summary_parts.append(f"Quality: {', '.join(quality_indicators)}")

            # Concepts détectés
            concepts = getattr(entity, 'concepts', set())
            if concepts:
                concepts_list = list(concepts)[:2]
                summary_parts.append(f"Concepts: {', '.join(concepts_list)}")

            # Assembler le résumé
            full_summary = '. '.join(summary_parts)

            # Gestion de la longueur
            max_chars = max_tokens * 4  # Approximation
            if len(full_summary) > max_chars:
                # Tronquer intelligemment au dernier point
                truncated = full_summary[:max_chars]
                last_period = truncated.rfind('.')
                if last_period > len(truncated) * 0.8:
                    full_summary = truncated[:last_period + 1]
                else:
                    full_summary = truncated + "..."

            return full_summary

        except Exception as e:
            logger.debug(f"Erreur création résumé détaillé: {e}")
            return await self._create_entity_summary(entity, max_tokens)  # Fallback

    async def _get_parent_context_optimized(self, parent_name: str, max_tokens: int) -> Optional[Dict[str, Any]]:
        """
        Contexte parent optimisé utilisant EntityManager.
        Remplace la logique de recherche manuelle.
        """
        parent_entity = await self.entity_manager.find_entity(parent_name)
        if not parent_entity:
            return None

        return {
            'name': parent_name,
            'type': parent_entity.entity_type,
            'summary': await self._create_entity_summary_local(parent_entity, max_tokens),
            'signature': parent_entity.signature or "Signature not available",
            'children': [child.entity_name for child in
                         await self.entity_manager.get_children(parent_entity.entity_id)],
            'concepts': list(parent_entity.concepts)[:3],
            'is_grouped': parent_entity.is_grouped
        }

    async def _get_children_context_optimized(self, children_entities: List, max_tokens: int) -> List[Dict[str, Any]]:
        """
        Contexte des enfants optimisé.
        Évite les recherches redondantes grâce à EntityManager.
        """
        contexts = []
        tokens_per_child = max_tokens / max(len(children_entities), 1)

        for child_entity in children_entities[:8]:  # Limiter à 8
            child_context = {
                'name': child_entity.entity_name,
                'type': child_entity.entity_type,
                'signature': child_entity.signature or f"{child_entity.entity_type} {child_entity.entity_name}(...)",
                'summary': await self._create_entity_summary_local(child_entity, int(tokens_per_child)),
                'is_internal': bool(child_entity.parent_entity),
                'concepts': list(child_entity.concepts)[:2],
                'called_functions_count': len(child_entity.called_functions)
            }
            contexts.append(child_context)

        return contexts

    # === Méthodes d'optimisation avec cache ===

    async def get_local_context_cached(self, entity_name: str, max_tokens: int = 2000) -> Dict[str, Any]:
        """Version avec cache pour les appels fréquents"""
        cache_key = f"local_context_{entity_name.lower()}_{max_tokens}"

        return await self.get_cached_or_compute(
            cache_key,
            lambda: self.get_local_context(entity_name, max_tokens),
            cache_type='semantic_contexts',
            ttl=900  # 15 minutes
        )

    # === Méthodes utilitaires spécialisées ===

    async def get_entity_immediate_calls(self, entity_name: str) -> List[str]:
        """
        Récupère uniquement les appels immédiats (API simple).
        Utilise le cache de FortranAnalyzer.
        """
        await self._ensure_initialized()

        entity = await self.entity_manager.find_entity(entity_name)
        if entity:
            return list(entity.called_functions)

        return []

    async def get_entity_dependencies_only(self, entity_name: str) -> List[str]:
        """
        Récupère uniquement les dépendances USE (API simple).
        """
        entity = await self.entity_manager.find_entity(entity_name)
        if entity:
            return list(entity.dependencies)

        return []

    async def is_entity_internal(self, entity_name: str) -> bool:
        """Vérifie si une entité est interne à une autre"""
        entity = await self.entity_manager.find_entity(entity_name)
        return bool(entity and entity.parent_entity)

    async def get_entity_complexity(self, entity_name: str) -> Dict[str, Any]:
        """
        Analyse la complexité locale d'une entité.
        Utilise FortranAnalyzer pour une évaluation précise.
        """
        await self._ensure_initialized()

        call_analysis = await self.analyzer.analyze_function_calls(entity_name)
        if 'error' in call_analysis:
            return {'complexity': 'unknown', 'error': call_analysis['error']}

        stats = call_analysis['call_statistics']

        # Calculer la complexité basée sur plusieurs facteurs
        complexity_score = 0

        # Appels sortants
        complexity_score += min(stats['total_outgoing'], 10)

        # Appels entrants
        complexity_score += min(stats['total_incoming'], 5)

        # Diversité des cibles
        complexity_score += min(stats['unique_targets'], 8)

        # Déterminer le niveau
        if complexity_score <= 5:
            level = "low"
        elif complexity_score <= 15:
            level = "medium"
        else:
            level = "high"

        return {
            'complexity': level,
            'score': complexity_score,
            'factors': {
                'outgoing_calls': stats['total_outgoing'],
                'incoming_calls': stats['total_incoming'],
                'unique_targets': stats['unique_targets'],
                'call_complexity': stats['call_complexity']
            }
        }