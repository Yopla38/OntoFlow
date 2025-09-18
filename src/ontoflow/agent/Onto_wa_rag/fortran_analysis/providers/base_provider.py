"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# providers/base_provider.py (Version corrigée)
"""
Provider de base avec utilitaires communs - VERSION CORRIGÉE.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

from ..core.entity_manager import EntityManager
from ..utils.chunk_access import ChunkAccessManager
from ..utils.caching import global_cache

logger = logging.getLogger(__name__)


class BaseContextProvider(ABC):
    """Provider de base avec correction des erreurs de types"""

    def __init__(self, document_store, rag_engine, entity_manager: EntityManager):
        self.document_store = document_store
        self.rag_engine = rag_engine
        self.entity_manager = entity_manager

        # Composants unifiés des phases 1-2
        self.chunk_access = ChunkAccessManager(document_store)
        self.analyzer = None
        self.concept_detector = None

        # Cache unifié
        self.cache = global_cache

        # Initialisation différée
        self._initialized = False

    async def _ensure_initialized(self):
        """S'assure que tous les composants sont initialisés"""
        if self._initialized:
            return

        try:
            # Initialiser l'analyseur avec les dépendances
            from ..core.fortran_analyzer import get_fortran_analyzer
            self.analyzer = await get_fortran_analyzer(self.document_store, self.entity_manager)

            # Initialiser le détecteur de concepts
            from ..core.concept_detector import get_concept_detector
            ontology_manager = getattr(self.rag_engine, 'classifier', None)
            self.concept_detector = get_concept_detector(ontology_manager)

            self._initialized = True
        except Exception as e:
            logger.error(f"Erreur initialisation BaseProvider: {e}")
            # Continuer avec composants partiels
            self._initialized = True

    async def resolve_entity(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Résout une entité par nom avec gestion d'erreurs robuste"""
        await self._ensure_initialized()

        try:
            # Recherche dans EntityManager
            entity = await self.entity_manager.find_entity(entity_name)
            if entity:
                return {
                    'name': entity.entity_name,
                    'type': entity.entity_type,
                    'resolution_method': 'entity_manager_direct',
                    'entity_id': entity.entity_id,
                    'entity_object': entity
                }
        except Exception as e:
            logger.debug(f"Erreur résolution entité {entity_name}: {e}")

        # Recherche fuzzy en fallback
        try:
            similar_entities = await self._find_similar_entity_names(entity_name)
            if similar_entities:
                best_match = similar_entities[0]
                best_entity = await self.entity_manager.find_entity(best_match)
                if best_entity:
                    return {
                        'name': best_entity.entity_name,
                        'type': best_entity.entity_type,
                        'resolution_method': 'fuzzy_match',
                        'entity_id': best_entity.entity_id,
                        'entity_object': best_entity,
                        'alternatives': similar_entities[1:5]
                    }
        except Exception as e:
            logger.debug(f"Erreur recherche fuzzy {entity_name}: {e}")

        return None

    async def get_entity_definition(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """SIMPLIFIÉE - UnifiedEntity a tout ce qu'il faut"""
        resolved = await self.resolve_entity(entity_name)
        if not resolved:
            return None

        entity = resolved['entity_object']  # UnifiedEntity

        # Plus besoin de _get_entity_complete_code complexe
        # UnifiedEntity.chunks contient déjà le code

        return {
            'name': entity.entity_name,
            'type': entity.entity_type,
            'signature': entity.signature or f"{entity.entity_type} {entity.entity_name}(...)",
            'location': {
                'file': entity.filename,
                'filepath': entity.filepath,
                'lines': f"{entity.start_line}-{entity.end_line}"
            },
            'metadata': {
                'is_grouped': entity.is_grouped,
                'is_complete': entity.is_complete,
                'chunk_count': len(entity.chunks),
                'confidence': entity.confidence,
                'parent': entity.parent_entity
            },
            'concepts': entity.detected_concepts,
            'dependencies': list(entity.dependencies),
            'called_functions': list(entity.called_functions)
        }

    def _normalize_detected_concepts(self, detected_concepts: List[Any]) -> List[Dict[str, Any]]:
        """SIMPLIFIÉE - detected_concepts est déjà normalisé dans UnifiedEntity"""
        normalized = []

        for concept in detected_concepts:
            if isinstance(concept, dict):
                # Déjà bon format
                normalized.append(concept)
            else:
                # Convertir string/autre en dict
                normalized.append({
                    'label': str(concept),
                    'confidence': 0.5,
                    'category': 'unknown',
                    'detection_method': 'legacy_conversion'
                })

        return normalized[:10]

    async def _extract_entity_signature_safe(self, entity, entity_code: str) -> str:
        """Extraction de signature avec gestion d'erreurs"""
        try:
            # Utiliser signature existante si disponible
            if hasattr(entity, 'signature') and entity.signature and entity.signature != "Signature not found":
                return entity.signature

            # Utiliser l'analyseur unifié si disponible
            if self.analyzer:
                from ..core.fortran_parser import get_unified_parser
                parser = get_unified_parser()
                signature = await parser.extract_signature_only(entity_code, getattr(entity, 'filepath', 'unknown.f90'))
                return signature

        except Exception as e:
            logger.debug(f"Erreur extraction signature: {e}")

        # Fallback : créer une signature basique
        entity_type = getattr(entity, 'entity_type', 'unknown')
        entity_name = getattr(entity, 'entity_name', 'unknown')

        if entity_type in ['subroutine', 'function']:
            return f"{entity_type} {entity_name}(...)"
        else:
            return f"{entity_type} {entity_name}"

    async def _get_entity_complete_code(self, entity) -> str:
        """Récupère le code complet d'une entité avec gestion d'erreurs"""
        try:
            if not hasattr(entity, 'chunks') or not entity.chunks:
                return ""

            # Trier les chunks par ordre (important pour les entités splittées)
            sorted_chunks = sorted(entity.chunks, key=lambda x: x.get('part_index', 0))

            code_parts = []
            for chunk_info in sorted_chunks:
                chunk_id = chunk_info.get('chunk_id', '')
                if chunk_id:
                    chunk_text = await self.chunk_access.get_chunk_text(chunk_id)
                    if chunk_text:
                        code_parts.append(chunk_text)

            return '\n'.join(code_parts)

        except Exception as e:
            logger.debug(f"Erreur récupération code entité: {e}")
            return ""

    async def get_dependency_contexts(self, dependencies: List[str], max_tokens: int) -> List[Dict[str, Any]]:
        """Récupère le contexte des dépendances avec gestion d'erreurs robuste"""
        await self._ensure_initialized()

        contexts = []
        tokens_per_dep = max_tokens / max(len(dependencies), 1)

        for dep_name in dependencies[:10]:  # Limiter
            try:
                dep_entity = await self.entity_manager.find_entity(dep_name)

                if dep_entity:
                    # Contexte enrichi depuis EntityManager
                    dep_context = {
                        'name': dep_name,
                        'type': getattr(dep_entity, 'entity_type', 'unknown'),
                        'file': getattr(dep_entity, 'filepath', ''),
                        'summary': await self._create_entity_summary_safe(dep_entity, int(tokens_per_dep)),
                        'public_interface': await self._get_public_interface_safe(dep_entity),
                        'concepts': list(getattr(dep_entity, 'concepts', set()))[:3],
                        'is_grouped': getattr(dep_entity, 'is_grouped', False),
                        'confidence': getattr(dep_entity, 'confidence', 1.0)
                    }
                    contexts.append(dep_context)
                else:
                    # Contexte minimal pour dépendances non résolues
                    contexts.append({
                        'name': dep_name,
                        'type': 'unresolved',
                        'file': 'unknown',
                        'summary': f"Dépendance non résolue: {dep_name}",
                        'public_interface': [],
                        'concepts': [],
                        'resolution_status': 'failed'
                    })

            except Exception as e:
                logger.debug(f"Erreur traitement dépendance {dep_name}: {e}")
                # Contexte d'erreur
                contexts.append({
                    'name': dep_name,
                    'type': 'error',
                    'file': 'unknown',
                    'summary': f"Erreur traitement: {dep_name}",
                    'public_interface': [],
                    'concepts': [],
                    'error': str(e)
                })

        return contexts

    async def _create_entity_summary_safe(self, entity, max_tokens: int) -> str:
        """Crée un résumé d'entité avec gestion d'erreurs"""
        try:
            entity_code = await self._get_entity_complete_code(entity)

            if not entity_code:
                entity_type = getattr(entity, 'entity_type', 'unknown')
                entity_name = getattr(entity, 'entity_name', 'unknown')
                return f"{entity_type.title()} {entity_name}"

            # Résumé intelligent basé sur le type
            entity_type = getattr(entity, 'entity_type', 'unknown')

            if entity_type == 'module':
                return self._summarize_module_safe(entity, entity_code, max_tokens)
            elif entity_type in ['function', 'subroutine']:
                return self._summarize_procedure_safe(entity, entity_code, max_tokens)
            else:
                return self._summarize_generic_safe(entity_code, max_tokens)

        except Exception as e:
            logger.debug(f"Erreur création résumé: {e}")
            entity_name = getattr(entity, 'entity_name', 'unknown')
            entity_type = getattr(entity, 'entity_type', 'unknown')
            return f"{entity_type.title()} {entity_name} (summary generation failed)"

    def _summarize_module_safe(self, entity, code: str, max_tokens: int) -> str:
        """Résumé sécurisé pour modules"""
        try:
            entity_name = getattr(entity, 'entity_name', 'unknown')
            summary_parts = [f"Module {entity_name}"]

            # Ajouter les USE statements
            dependencies = getattr(entity, 'dependencies', set())
            if dependencies:
                deps = list(dependencies)[:3]
                summary_parts.append(f"Uses: {', '.join(deps)}")

            # Ajouter les concepts principaux
            concepts = getattr(entity, 'concepts', set())
            if concepts:
                concepts_list = list(concepts)[:2]
                summary_parts.append(f"Concepts: {', '.join(concepts_list)}")

            return '; '.join(summary_parts)

        except Exception as e:
            logger.debug(f"Erreur résumé module: {e}")
            return f"Module {getattr(entity, 'entity_name', 'unknown')}"

    def _summarize_procedure_safe(self, entity, code: str, max_tokens: int) -> str:
        """Résumé sécurisé pour procédures"""
        try:
            entity_name = getattr(entity, 'entity_name', 'unknown')
            entity_type = getattr(entity, 'entity_type', 'unknown')

            summary_parts = [f"{entity_type.title()} {entity_name}"]

            signature = getattr(entity, 'signature', '')
            if signature and signature != "Signature not found":
                summary_parts.append(f"Signature: {signature}")

            called_functions = getattr(entity, 'called_functions', set())
            if called_functions:
                calls = list(called_functions)[:3]
                summary_parts.append(f"Calls: {', '.join(calls)}")

            return '; '.join(summary_parts)

        except Exception as e:
            logger.debug(f"Erreur résumé procédure: {e}")
            return f"{getattr(entity, 'entity_type', 'unknown').title()} {getattr(entity, 'entity_name', 'unknown')}"

    def _summarize_generic_safe(self, code: str, max_tokens: int) -> str:
        """Résumé générique sécurisé"""
        try:
            from ..utils.chunk_access import create_chunk_summary
            return create_chunk_summary(code, max_tokens)
        except Exception as e:
            logger.debug(f"Erreur résumé générique: {e}")
            return "Summary generation failed"

    async def _get_public_interface_safe(self, entity) -> List[str]:
        """Récupère l'interface publique avec gestion d'erreurs"""
        try:
            if getattr(entity, 'entity_type', '') != 'module':
                return []

            entity_code = await self._get_entity_complete_code(entity)
            if not entity_code:
                return []

            # Utiliser l'extracteur de patterns unifié
            from ..utils.fortran_patterns import FortranTextProcessor
            processor = FortranTextProcessor()
            return processor.extract_public_interface(entity_code)

        except Exception as e:
            logger.debug(f"Erreur interface publique: {e}")
            return []

    async def find_function_calls_with_analysis(self, entity_name: str) -> List[Dict[str, Any]]:
        """Trouve les appels de fonctions avec gestion d'erreurs"""
        await self._ensure_initialized()

        try:
            if not self.analyzer:
                # Fallback sans analyseur
                entity = await self.entity_manager.find_entity(entity_name)
                if entity and hasattr(entity, 'called_functions'):
                    return [
                        {
                            'name': call_name,
                            'resolved': False,
                            'target_type': 'unknown',
                            'target_file': 'unknown',
                            'is_internal': False,
                            'source': 'entity_manager_fallback'
                        }
                        for call_name in list(entity.called_functions)[:10]
                    ]
                return []

            # Utiliser FortranAnalyzer pour analyse complète
            call_analysis = await self.analyzer.analyze_function_calls(entity_name)

            if 'error' in call_analysis:
                return []

            # Enrichir les appels avec résolution
            enriched_calls = []
            for call in call_analysis.get('outgoing_calls', []):
                call_info = {
                    'name': call.get('name', ''),
                    'resolved': call.get('resolved', False),
                    'target_type': call.get('target_type', 'unknown'),
                    'target_file': call.get('target_file', 'unknown'),
                    'is_internal': call.get('is_internal', False),
                    'source': 'fortran_analyzer'
                }

                # Ajouter signature si disponible
                if call.get('resolved'):
                    target_entity = await self.entity_manager.find_entity(call['name'])
                    if target_entity:
                        if hasattr(target_entity, 'signature') and target_entity.signature:
                            call_info['signature'] = target_entity.signature
                        call_info['summary'] = await self._create_entity_summary_safe(target_entity, 100)

                enriched_calls.append(call_info)

            return enriched_calls

        except Exception as e:
            logger.debug(f"Erreur analyse appels pour {entity_name}: {e}")
            return []

    async def _find_similar_entity_names(self, entity_name: str, limit: int = 5) -> List[str]:
        """Trouve des noms d'entités similaires avec gestion d'erreurs"""
        try:
            all_entities = list(self.entity_manager.entities.values())

            # Recherche par sous-chaîne
            similar = []
            name_lower = entity_name.lower()

            for entity in all_entities:
                entity_name_attr = getattr(entity, 'entity_name', '')
                entity_name_lower = entity_name_attr.lower()
                if (name_lower in entity_name_lower or
                        entity_name_lower in name_lower):
                    similar.append(entity_name_attr)

            return similar[:limit]

        except Exception as e:
            logger.debug(f"Erreur recherche entités similaires: {e}")
            return []

    # === Méthodes utilitaires ===

    async def get_cached_or_compute(self, cache_key: str, compute_func, cache_type: str = 'entities', ttl: int = 1800):
        """Utilitaire générique pour cache-or-compute avec gestion d'erreurs"""
        try:
            cache_manager = getattr(self.cache, cache_type)

            # Vérifier le cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Calculer et cacher
            result = await compute_func()
            await cache_manager.set(cache_key, result, ttl=ttl)

            return result

        except Exception as e:
            logger.debug(f"Erreur cache-or-compute: {e}")
            # Exécuter sans cache en cas d'erreur
            try:
                return await compute_func()
            except Exception as e2:
                logger.error(f"Erreur exécution fonction: {e2}")
                return None

    # === Méthodes abstraites ===

    @abstractmethod
    async def get_context(self, entity_name: str, max_tokens: int = 2000) -> Dict[str, Any]:
        """Méthode principale à implémenter par chaque provider"""
        pass

    # === Méthodes utilitaires communes ===

    async def create_error_context(self, entity_name: str, error_message: str) -> Dict[str, Any]:
        """Crée un contexte d'erreur standardisé"""
        suggestions = await self._find_similar_entity_names(entity_name)

        return {
            'entity': entity_name,
            'error': error_message,
            'suggestions': suggestions,
            'timestamp': asyncio.get_event_loop().time()
        }

    def calculate_tokens_used(self, context_data: Dict[str, Any]) -> int:
        """Calcule approximativement les tokens utilisés"""
        try:
            text_content = str(context_data)
            return len(text_content) // 4  # Approximation
        except Exception:
            return 0

    async def get_file_context(self, filepath: str, current_entity: str, max_tokens: int) -> Dict[str, Any]:
        """Récupère le contexte des autres entités du même fichier"""
        try:
            entities_in_file = await self.entity_manager.get_entities_in_file(filepath)

            other_entities = []
            for entity in entities_in_file:
                entity_name = getattr(entity, 'entity_name', '')
                if entity_name != current_entity:
                    concepts = getattr(entity, 'concepts', set())
                    other_entities.append({
                        'name': entity_name,
                        'type': getattr(entity, 'entity_type', 'unknown'),
                        'lines': f"{getattr(entity, 'start_line', '')}-{getattr(entity, 'end_line', '')}",
                        'is_grouped': getattr(entity, 'is_grouped', False),
                        'concepts': list(concepts)[:2]  # Top 2 concepts
                    })

            return {
                'filepath': filepath,
                'other_entities': other_entities[:15],  # Limiter
                'total_entities': len(entities_in_file),
                'file_summary': f"{len(entities_in_file)} entités dans {filepath.split('/')[-1]}"
            }

        except Exception as e:
            logger.debug(f"Erreur contexte fichier: {e}")
            return {
                'filepath': filepath,
                'other_entities': [],
                'total_entities': 0,
                'file_summary': f"Erreur analyse fichier {filepath}"
            }


# Reste du code (ProviderRegistry, etc.) inchangé...
class ProviderRegistry:
    """Registre des providers pour injection de dépendances"""

    def __init__(self):
        self._providers: Dict[str, BaseContextProvider] = {}

    def register(self, name: str, provider: BaseContextProvider):
        """Enregistre un provider"""
        self._providers[name] = provider

    def get(self, name: str) -> Optional[BaseContextProvider]:
        """Récupère un provider"""
        return self._providers.get(name)

    def get_all(self) -> Dict[str, BaseContextProvider]:
        """Récupère tous les providers"""
        return self._providers.copy()


# Instance globale du registre
provider_registry = ProviderRegistry()
