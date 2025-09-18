"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# utils/chunk_access.py
"""
Utilitaire centralisé pour l'accès aux chunks.
Remplace toutes les méthodes _get_chunk_by_id() dupliquées dans le système.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ChunkAccessStats:
    """Statistiques d'accès aux chunks"""
    cache_hits: int = 0
    cache_misses: int = 0
    total_requests: int = 0
    failed_requests: int = 0


class ChunkAccessManager:
    """
    Gestionnaire centralisé pour l'accès aux chunks.
    Utilisé par tous les providers pour éviter la duplication de code.
    """

    def __init__(self, document_store):
        self.document_store = document_store
        self._chunk_cache: Dict[str, Dict[str, Any]] = {}
        self._loaded_documents: set = set()
        self.stats = ChunkAccessStats()

    async def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère un chunk par son ID (méthode unifiée).

        Basée sur la logique exacte de LocalContextProvider._get_chunk_by_id
        avec mise en cache pour les performances.
        """
        self.stats.total_requests += 1

        # Vérifier le cache d'abord
        if chunk_id in self._chunk_cache:
            self.stats.cache_hits += 1
            return self._chunk_cache[chunk_id]

        self.stats.cache_misses += 1

        try:
            # Parser le chunk_id pour extraire le document_id
            # Format: document_id-chunk-N (logique exacte du code existant)
            parts = chunk_id.split('-chunk-')
            if len(parts) != 2:
                logger.debug(f"Invalid chunk_id format: {chunk_id}")
                self.stats.failed_requests += 1
                return None

            document_id = parts[0]

            # Charger les chunks du document si nécessaire
            await self._ensure_document_loaded(document_id)
            chunks = await self.document_store.get_document_chunks(document_id)

            if chunks:
                for chunk in chunks:
                    if chunk['id'] == chunk_id:
                        # Mettre en cache pour les prochaines requêtes
                        self._chunk_cache[chunk_id] = chunk
                        return chunk

            self.stats.failed_requests += 1
            return None

        except Exception as e:
            logger.debug(f"Error retrieving chunk {chunk_id}: {e}")
            self.stats.failed_requests += 1
            return None

    async def get_chunk_text(self, chunk_id: str) -> Optional[str]:
        """
        Récupère le texte d'un chunk (basé sur la logique du visualiseur).

        Méthode équivalente à SmartContextProvider._get_chunk_text_exact_copy
        """
        chunk = await self.get_chunk_by_id(chunk_id)
        return chunk.get('text') if chunk else None

    async def get_chunks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Récupère tous les chunks d'un document"""
        await self._ensure_document_loaded(document_id)
        return await self.document_store.get_document_chunks(document_id) or []

    async def get_entity_info_from_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Extrait les informations d'entité depuis les métadonnées du chunk"""
        chunk = await self.get_chunk_by_id(chunk_id)
        if not chunk:
            return None

        metadata = chunk.get('metadata', {})
        return {
            'name': metadata.get('entity_name', ''),
            'type': metadata.get('entity_type', ''),
            'filepath': metadata.get('filepath', ''),
            'filename': metadata.get('filename', ''),
            'start_line': metadata.get('start_pos'),
            'end_line': metadata.get('end_pos'),
            'dependencies': metadata.get('dependencies', []),
            'concepts': metadata.get('detected_concepts', []),
            'is_internal': metadata.get('is_internal_function', False),
            'parent': metadata.get('parent_entity_name', ''),
            'qualified_name': metadata.get('full_qualified_name', ''),
            'chunk_size': len(chunk.get('text', '')),
            'metadata': metadata
        }

    async def find_chunks_by_entity_name(self, entity_name: str,
                                         case_sensitive: bool = False) -> List[str]:
        """Trouve les chunk IDs contenant une entité donnée"""
        # Cette méthode sera utilisée par EntityIndex
        all_docs = await self.document_store.get_all_documents()
        matching_chunks = []

        for doc_id in all_docs:
            chunks = await self.get_chunks_by_document(doc_id)
            for chunk in chunks:
                metadata = chunk.get('metadata', {})
                chunk_entity_name = metadata.get('entity_name', '')

                if case_sensitive:
                    match = chunk_entity_name == entity_name
                else:
                    match = chunk_entity_name.lower() == entity_name.lower()

                if match:
                    matching_chunks.append(chunk['id'])

        return matching_chunks

    async def _ensure_document_loaded(self, document_id: str):
        """S'assure qu'un document est chargé (évite les rechargements)"""
        if document_id not in self._loaded_documents:
            await self.document_store.load_document_chunks(document_id)
            self._loaded_documents.add(document_id)

    def clear_cache(self):
        """Vide le cache (utile pour les tests)"""
        self._chunk_cache.clear()
        self._loaded_documents.clear()
        self.stats = ChunkAccessStats()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de cache"""
        hit_rate = (self.stats.cache_hits / max(1, self.stats.total_requests)) * 100

        return {
            'cache_hits': self.stats.cache_hits,
            'cache_misses': self.stats.cache_misses,
            'total_requests': self.stats.total_requests,
            'failed_requests': self.stats.failed_requests,
            'hit_rate_percent': round(hit_rate, 2),
            'cached_chunks': len(self._chunk_cache),
            'loaded_documents': len(self._loaded_documents)
        }


# Fonctions utilitaires pour compatibilité avec le code existant
async def get_chunk_by_id(document_store, chunk_id: str) -> Optional[Dict[str, Any]]:
    """
    Fonction utilitaire pour compatibilité avec le code existant.
    Utilise un gestionnaire temporaire pour maintenir l'interface.
    """
    manager = ChunkAccessManager(document_store)
    return await manager.get_chunk_by_id(chunk_id)


def create_chunk_summary(chunk_text: str, max_tokens: int) -> str:
    """
    Crée un résumé de chunk (basé sur LocalContextProvider._create_summary).
    Méthode utilitaire commune.
    """
    words = chunk_text.split()
    max_words = max_tokens * 3  # Approximation : 1 token ≈ 3/4 mots

    if len(words) <= max_words:
        return chunk_text

    # Prendre le début et essayer de finir à une phrase complète
    truncated = ' '.join(words[:max_words])

    # Chercher le dernier point pour finir proprement
    last_period = truncated.rfind('.')
    if last_period > len(truncated) * 0.8:  # Si le point est vers la fin
        truncated = truncated[:last_period + 1]

    return truncated + "..."
