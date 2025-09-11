"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# utils/caching.py
"""
Système de cache centralisé pour remplacer les caches éparpillés
dans EntityIndex, GlobalContextProvider, SemanticContextProvider.
"""

import asyncio
import time
import weakref
from typing import Dict, Any, Optional, Callable, TypeVar, Generic, Union, List
from dataclasses import dataclass, field
from collections import OrderedDict
from abc import ABC, abstractmethod
import logging
import json
import hashlib

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Entrée de cache avec métadonnées"""
    value: T
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None

    def is_expired(self) -> bool:
        """Vérifie si l'entrée a expiré"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def touch(self):
        """Met à jour le temps d'accès"""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Statistiques de cache"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    entries: int = 0
    memory_usage_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheStrategy(ABC):
    """Interface pour les stratégies de cache"""

    @abstractmethod
    def should_evict(self, cache: Dict[str, CacheEntry], max_size: int) -> Optional[str]:
        """Retourne la clé à évincer ou None"""
        pass


class LRUStrategy(CacheStrategy):
    """Stratégie Least Recently Used"""

    def should_evict(self, cache: Dict[str, CacheEntry], max_size: int) -> Optional[str]:
        if len(cache) <= max_size:
            return None

        # Trouver l'entrée la moins récemment utilisée
        oldest_key = min(cache.keys(), key=lambda k: cache[k].last_accessed)
        return oldest_key


class LFUStrategy(CacheStrategy):
    """Stratégie Least Frequently Used"""

    def should_evict(self, cache: Dict[str, CacheEntry], max_size: int) -> Optional[str]:
        if len(cache) <= max_size:
            return None

        # Trouver l'entrée la moins fréquemment utilisée
        least_used_key = min(cache.keys(), key=lambda k: cache[k].access_count)
        return least_used_key


class SmartCache(Generic[T]):
    """
    Cache intelligent avec TTL, stratégies d'éviction et statistiques.
    Remplace les caches simples utilisés dans le système existant.
    """

    def __init__(self,
                 max_size: int = 1000,
                 default_ttl: Optional[float] = None,
                 strategy: CacheStrategy = None,
                 enable_stats: bool = True):

        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy or LRUStrategy()
        self.enable_stats = enable_stats

        self._cache: Dict[str, CacheEntry[T]] = {}
        self._stats = CacheStats()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[T]:
        """Récupère une valeur du cache"""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired():
                # Entrée expirée, la supprimer
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                return None

            # Entrée valide
            entry.touch()
            self._stats.hits += 1
            return entry.value

    async def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Stocke une valeur dans le cache"""
        async with self._lock:
            # Utiliser TTL spécifique ou par défaut
            effective_ttl = ttl if ttl is not None else self.default_ttl

            # Créer l'entrée
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=effective_ttl
            )

            # Vérifier si éviction nécessaire
            if key not in self._cache and len(self._cache) >= self.max_size:
                evict_key = self.strategy.should_evict(self._cache, self.max_size - 1)
                if evict_key:
                    del self._cache[evict_key]
                    self._stats.evictions += 1

            # Stocker l'entrée
            self._cache[key] = entry
            self._stats.entries = len(self._cache)

    async def delete(self, key: str) -> bool:
        """Supprime une entrée du cache"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.entries = len(self._cache)
                return True
            return False

    async def clear(self) -> None:
        """Vide le cache"""
        async with self._lock:
            self._cache.clear()
            self._stats = CacheStats()

    def get_stats(self) -> CacheStats:
        """Retourne les statistiques du cache"""
        return self._stats

    async def cleanup_expired(self) -> int:
        """Nettoie les entrées expirées, retourne le nombre supprimé"""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                del self._cache[key]

            self._stats.entries = len(self._cache)
            self._stats.evictions += len(expired_keys)

            return len(expired_keys)


class MultiLevelCache:
    """
    Cache multi-niveaux pour optimiser différents types de données.
    Remplace les caches spécialisés existants.
    """

    def __init__(self):
        # Cache pour les chunks (accès très fréquent, TTL court)
        self.chunks = SmartCache[Dict[str, Any]](
            max_size=500,
            default_ttl=300,  # 5 minutes
            strategy=LRUStrategy()
        )

        # Cache pour les appels de fonctions (accès fréquent, TTL long)
        self.function_calls = SmartCache[List[str]](
            max_size=1000,
            default_ttl=1800,  # 30 minutes
            strategy=LFUStrategy()
        )

        # Cache pour les graphes de dépendances (calcul coûteux, TTL long)
        self.dependency_graphs = SmartCache[Dict[str, Any]](
            max_size=50,
            default_ttl=3600,  # 1 heure
            strategy=LFUStrategy()
        )

        # Cache pour les contextes sémantiques (accès moyen, TTL moyen)
        self.semantic_contexts = SmartCache[Dict[str, Any]](
            max_size=200,
            default_ttl=900,  # 15 minutes
            strategy=LRUStrategy()
        )

        # Cache pour les entités (accès fréquent, pas d'expiration)
        self.entities = SmartCache[Dict[str, Any]](
            max_size=2000,
            default_ttl=None,  # Pas d'expiration
            strategy=LFUStrategy()
        )

    async def cleanup_all_expired(self) -> Dict[str, int]:
        """Nettoie tous les caches expirés"""
        results = {}

        for cache_name in ['chunks', 'function_calls', 'dependency_graphs',
                           'semantic_contexts', 'entities']:
            cache = getattr(self, cache_name)
            expired_count = await cache.cleanup_expired()
            results[cache_name] = expired_count

        return results

    def get_all_stats(self) -> Dict[str, CacheStats]:
        """Retourne les statistiques de tous les caches"""
        return {
            'chunks': self.chunks.get_stats(),
            'function_calls': self.function_calls.get_stats(),
            'dependency_graphs': self.dependency_graphs.get_stats(),
            'semantic_contexts': self.semantic_contexts.get_stats(),
            'entities': self.entities.get_stats()
        }


class LegacyCacheAdapter:
    """
    Adaptateur pour maintenir la compatibilité avec les caches existants.
    Permet la migration progressive sans casser le code existant.
    """

    def __init__(self, multi_cache: MultiLevelCache):
        self.multi_cache = multi_cache

        # Mapping pour EntityIndex.call_patterns_cache
        self._call_patterns_cache = {}

        # Mapping pour GlobalContextProvider._dependency_graph_cache
        self._dependency_graph_cache = {}

        # Mapping pour SemanticContextProvider._semantic_cache
        self._semantic_cache = {}

    # === Interface compatible avec EntityIndex ===

    @property
    def call_patterns_cache(self) -> Dict[str, List[str]]:
        """Compatible avec EntityIndex.call_patterns_cache"""
        return self._call_patterns_cache

    async def get_cached_call_patterns(self, chunk_id: str) -> Optional[List[str]]:
        """Compatible avec EntityIndex.get_cached_call_patterns"""
        # Essayer le nouveau cache d'abord
        result = await self.multi_cache.function_calls.get(chunk_id)
        if result is not None:
            return result

        # Fallback vers l'ancien cache
        return self._call_patterns_cache.get(chunk_id)

    async def cache_call_patterns(self, chunk_id: str, calls: List[str]) -> None:
        """Compatible avec EntityIndex.cache_call_patterns"""
        # Stocker dans les deux systèmes pendant la transition
        await self.multi_cache.function_calls.set(chunk_id, calls)
        self._call_patterns_cache[chunk_id] = calls

    # === Interface compatible avec GlobalContextProvider ===

    async def get_dependency_graph(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Compatible avec GlobalContextProvider._dependency_graph_cache"""
        # Essayer le nouveau cache
        result = await self.multi_cache.dependency_graphs.get(cache_key)
        if result is not None:
            return result

        # Fallback vers l'ancien cache
        return self._dependency_graph_cache.get(cache_key)

    async def cache_dependency_graph(self, cache_key: str, graph: Dict[str, Any]) -> None:
        """Cache un graphe de dépendances"""
        await self.multi_cache.dependency_graphs.set(cache_key, graph)
        self._dependency_graph_cache[cache_key] = graph

    # === Interface compatible avec SemanticContextProvider ===

    async def get_semantic_context(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Compatible avec SemanticContextProvider._semantic_cache"""
        result = await self.multi_cache.semantic_contexts.get(entity_name)
        if result is not None:
            return result

        return self._semantic_cache.get(entity_name)

    async def cache_semantic_context(self, entity_name: str, context: Dict[str, Any]) -> None:
        """Cache un contexte sémantique"""
        await self.multi_cache.semantic_contexts.set(entity_name, context)
        self._semantic_cache[entity_name] = context


# Instance globale pour utilisation dans le système
global_cache = MultiLevelCache()
legacy_adapter = LegacyCacheAdapter(global_cache)


# Tâche de nettoyage périodique
async def periodic_cleanup(interval: int = 600):  # 10 minutes
    """Tâche de nettoyage périodique des caches expirés"""
    while True:
        try:
            await asyncio.sleep(interval)
            results = await global_cache.cleanup_all_expired()

            total_cleaned = sum(results.values())
            if total_cleaned > 0:
                logger.info(f"Cache cleanup: {total_cleaned} expired entries removed")

        except Exception as e:
            logger.error(f"Error in cache cleanup: {e}")


def start_cache_cleanup_task():
    """Démarre la tâche de nettoyage périodique"""
    task = asyncio.create_task(periodic_cleanup())
    return task


# Fonctions utilitaires pour migration
def create_cache_key(*args) -> str:
    """Crée une clé de cache stable à partir d'arguments"""
    # Convertir tous les arguments en string et hasher
    key_string = '|'.join(str(arg) for arg in args)
    return hashlib.md5(key_string.encode()).hexdigest()


async def migrate_existing_cache(old_cache: Dict, new_cache: SmartCache,
                                 key_transform: Callable[[str], str] = None):
    """Migre un ancien cache vers le nouveau système"""
    for key, value in old_cache.items():
        cache_key = key_transform(key) if key_transform else key
        await new_cache.set(cache_key, value)

    logger.info(f"Migrated {len(old_cache)} entries to new cache system")