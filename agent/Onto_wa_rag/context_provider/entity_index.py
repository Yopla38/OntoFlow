"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# context_providers/entity_index.py
import asyncio
import re
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EntityIndex:
    """Index intelligent basé sur les métadonnées existantes des chunks"""

    def __init__(self, document_store):
        self.document_store = document_store

        # Index principal : nom -> [chunk_ids]
        self.name_to_chunks: Dict[str, List[str]] = defaultdict(list)

        # Métadonnées des chunks : chunk_id -> entity_info
        self.chunk_to_entity: Dict[str, Dict[str, Any]] = {}

        # Hiérarchie des modules : module_name -> info
        self.module_hierarchy: Dict[str, Dict[str, Any]] = {}

        # Relations parent-enfant : parent_name -> [child_names]
        self.parent_to_children: Dict[str, List[str]] = defaultdict(list)

        # Relations inverses : child_name -> parent_name
        self.child_to_parent: Dict[str, str] = {}

        # Cache des patterns d'appels détectés
        self.call_patterns_cache: Dict[str, List[str]] = {}

        # Statistiques
        self.stats = {
            'total_entities': 0,
            'modules': 0,
            'functions': 0,
            'subroutines': 0,
            'internal_functions': 0,
            'use_dependencies': 0
        }

        self._initialized = False

    async def build_index(self) -> None:
        """Construction de l'index à partir des métadonnées existantes"""
        if self._initialized:
            logger.info("Index déjà initialisé")
            return

        logger.info("🔍 Construction de l'index des entités...")

        all_docs = await self.document_store.get_all_documents()
        total_chunks = 0

        for doc_id in all_docs:
            await self.document_store.load_document_chunks(doc_id)
            chunks = await self.document_store.get_document_chunks(doc_id)

            if chunks:
                for chunk in chunks:
                    await self._index_chunk(chunk)
                    total_chunks += 1

        # Post-traitement : construire les relations hiérarchiques
        self._build_hierarchy_relations()

        self.stats['total_entities'] = len(self.chunk_to_entity)

        logger.info(f"✅ Index construit: {total_chunks} chunks, {self.stats['total_entities']} entités")
        logger.info(f"   Modules: {self.stats['modules']}, Functions: {self.stats['functions']}")
        logger.info(f"   Subroutines: {self.stats['subroutines']}, Internal: {self.stats['internal_functions']}")

        self._initialized = True

    async def _index_chunk(self, chunk: Dict[str, Any]) -> None:
        """Index un chunk individuel en exploitant ses métadonnées"""
        metadata = chunk['metadata']
        chunk_id = chunk['id']

        # Extraire les informations de l'entité
        entity_name = metadata.get('entity_name', '')
        base_name = metadata.get('base_entity_name', entity_name)
        entity_type = metadata.get('entity_type', '')

        if not entity_name:
            return  # Skip les chunks sans entité identifiée

        # 1. Indexer les noms d'entités
        self._add_to_index(entity_name, chunk_id)
        if base_name and base_name != entity_name:
            self._add_to_index(base_name, chunk_id)

        dependencies = metadata.get('dependencies', [])
        if not dependencies:
            # Analyser le texte du chunk pour trouver les USE statements
            text = chunk.get('text', '')
            use_pattern = re.compile(r'^\s*use\s+(\w+)', re.IGNORECASE | re.MULTILINE)
            matches = use_pattern.findall(text)
            dependencies.extend(matches)

            # Mettre à jour les métadonnées pour les stats
            if matches:
                self.stats['use_dependencies'] += len(matches)

        # 2. Gérer les fonctions internes avec noms qualifiés
        if metadata.get('is_internal_function'):
            qualified_name = metadata.get('full_qualified_name')
            parent_entity = metadata.get('parent_entity')

            if qualified_name:
                self._add_to_index(qualified_name, chunk_id)
            if parent_entity:
                self._add_parent_child_relation(parent_entity, entity_name)

            self.stats['internal_functions'] += 1

        # 3. Gérer les modules et leurs dépendances
        if entity_type == 'module':
            self.module_hierarchy[entity_name] = {
                'chunk_id': chunk_id,
                'dependencies': metadata.get('dependencies', []),
                'children': [],
                'filepath': metadata.get('filepath', ''),
                'concepts': metadata.get('detected_concepts', [])
            }
            self.stats['modules'] += 1
            self.stats['use_dependencies'] += len(metadata.get('dependencies', []))

        # 4. Compter les types d'entités
        if entity_type == 'function':
            self.stats['functions'] += 1
        elif entity_type == 'subroutine':
            self.stats['subroutines'] += 1

        # 5. Relations parent-enfant génériques
        parent_name = metadata.get('parent_entity_name')
        if parent_name:
            self._add_parent_child_relation(parent_name, entity_name)

        # 6. Stocker les métadonnées complètes de l'entité
        self.chunk_to_entity[chunk_id] = {
            'name': entity_name,
            'base_name': base_name,
            'type': entity_type,
            'parent': parent_name,
            'dependencies': dependencies,
            'concepts': metadata.get('detected_concepts', []),
            'filepath': metadata.get('filepath', ''),
            'filename': metadata.get('filename', ''),
            'start_line': metadata.get('start_pos'),
            'end_line': metadata.get('end_pos'),
            'is_internal': metadata.get('is_internal_function', False),
            'qualified_name': metadata.get('full_qualified_name'),
            'chunk_size': len(chunk.get('text', '')),
            'metadata': metadata  # Référence complète si besoin
        }

    def _add_to_index(self, name: str, chunk_id: str) -> None:
        """Ajoute un nom à l'index avec ses variantes"""
        if name and chunk_id not in self.name_to_chunks[name]:
            self.name_to_chunks[name].append(chunk_id)

        # Ajouter les variantes du nom (cas, underscores)
        name_lower = name.lower()
        if name_lower != name and chunk_id not in self.name_to_chunks[name_lower]:
            self.name_to_chunks[name_lower].append(chunk_id)

    def _add_parent_child_relation(self, parent_name: str, child_name: str) -> None:
        """Ajoute une relation parent-enfant"""
        if child_name not in self.parent_to_children[parent_name]:
            self.parent_to_children[parent_name].append(child_name)

        self.child_to_parent[child_name] = parent_name

    def _build_hierarchy_relations(self) -> None:
        """Construit les relations hiérarchiques complètes"""
        # Ajouter les enfants aux modules
        for module_name, module_info in self.module_hierarchy.items():
            children = self.parent_to_children.get(module_name, [])
            module_info['children'] = children

    # === Méthodes de recherche ===

    async def find_entity(self, name: str) -> Optional[List[str]]:
        """Trouve les chunk IDs pour une entité donnée"""
        if not self._initialized:
            await self.build_index()

        # Recherche exacte
        chunks = self.name_to_chunks.get(name, [])
        if chunks:
            return chunks

        # Recherche insensible à la casse
        chunks = self.name_to_chunks.get(name.lower(), [])
        if chunks:
            return chunks

        # Recherche fuzzy (noms similaires)
        similar_names = [n for n in self.name_to_chunks.keys()
                         if name.lower() in n.lower() or n.lower() in name.lower()]

        if similar_names:
            # Retourner les chunks du nom le plus similaire
            best_match = min(similar_names, key=lambda x: abs(len(x) - len(name)))
            return self.name_to_chunks[best_match]

        return None

    async def get_entity_info(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Récupère les informations d'une entité par son chunk ID"""
        if not self._initialized:
            await self.build_index()

        return self.chunk_to_entity.get(chunk_id)

    async def get_children(self, entity_name: str) -> List[str]:
        """Récupère les entités enfants d'une entité"""
        if not self._initialized:
            await self.build_index()

        return self.parent_to_children.get(entity_name, [])

    async def get_parent(self, entity_name: str) -> Optional[str]:
        """Récupère l'entité parent d'une entité"""
        if not self._initialized:
            await self.build_index()

        return self.child_to_parent.get(entity_name)

    async def get_module_dependencies(self, module_name: str) -> List[str]:
        """Récupère les dépendances USE d'un module"""
        if not self._initialized:
            await self.build_index()

        module_info = self.module_hierarchy.get(module_name)
        return module_info['dependencies'] if module_info else []

    async def get_all_modules(self) -> Dict[str, Dict[str, Any]]:
        """Récupère tous les modules indexés"""
        if not self._initialized:
            await self.build_index()

        return self.module_hierarchy

    async def find_entities_by_type(self, entity_type: str) -> List[str]:
        """Trouve toutes les entités d'un type donné"""
        if not self._initialized:
            await self.build_index()

        chunk_ids = []
        for chunk_id, entity_info in self.chunk_to_entity.items():
            if entity_info['type'] == entity_type:
                chunk_ids.append(chunk_id)

        return chunk_ids

    async def find_entities_in_file(self, filepath: str) -> List[str]:
        """Trouve toutes les entités dans un fichier donné"""
        if not self._initialized:
            await self.build_index()

        chunk_ids = []
        for chunk_id, entity_info in self.chunk_to_entity.items():
            if entity_info['filepath'] == filepath:
                chunk_ids.append(chunk_id)

        return chunk_ids

    # === Cache des patterns d'appels ===

    def cache_call_patterns(self, chunk_id: str, calls: List[str]) -> None:
        """Met en cache les patterns d'appels détectés pour un chunk"""
        self.call_patterns_cache[chunk_id] = calls

    def get_cached_call_patterns(self, chunk_id: str) -> Optional[List[str]]:
        """Récupère les patterns d'appels mis en cache"""
        return self.call_patterns_cache.get(chunk_id)

    # === Statistiques et debug ===

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'index"""
        return {
            **self.stats,
            'indexed_names': len(self.name_to_chunks),
            'indexed_chunks': len(self.chunk_to_entity),
            'hierarchy_relations': len(self.parent_to_children),
            'cached_call_patterns': len(self.call_patterns_cache)
        }

    def debug_entity(self, entity_name: str) -> Dict[str, Any]:
        """Informations de debug pour une entité"""
        chunks = self.name_to_chunks.get(entity_name, [])

        debug_info = {
            'entity_name': entity_name,
            'found_chunks': len(chunks),
            'chunk_ids': chunks,
            'children': self.parent_to_children.get(entity_name, []),
            'parent': self.child_to_parent.get(entity_name),
            'similar_names': [name for name in self.name_to_chunks.keys()
                              if entity_name.lower() in name.lower()][:10]
        }

        if chunks:
            entity_info = self.chunk_to_entity.get(chunks[0])
            if entity_info:
                debug_info['entity_info'] = entity_info

        return debug_info
