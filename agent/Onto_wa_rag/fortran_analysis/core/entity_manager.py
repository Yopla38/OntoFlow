"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# core/entity_manager.py
"""
Gestionnaire unifié des entités Fortran.
Remplace et unifie la logique de regroupement du visualiseur et de l'EntityIndex.
"""

import asyncio
import os
import pickle
import re
import logging
from typing import Dict, List, Set, Optional, Any, Tuple, Literal
from dataclasses import dataclass, field
from collections import defaultdict

from ..utils.chunk_access import ChunkAccessManager
from ..utils.caching import global_cache

logger = logging.getLogger(__name__)


@dataclass
class UnifiedEntity:
    """Entité Fortran unique - remplace FortranEntity et UnifiedEntity"""
    # === DONNÉES DE BASE (ex-FortranEntity) ===
    entity_name: str
    entity_type: str
    start_line: int
    end_line: int

    # === MÉTADONNÉES ESSENTIELLES ===
    entity_id: str = ""  # Généré automatiquement si vide
    filepath: str = ""
    filename: str = ""
    parent_entity: Optional[str] = None
    signature: str = ""
    source_method: str = "hybrid"
    confidence: float = 1.0
    access_level: Optional[str] = None  # Public or private
    arguments: Optional = None
    return_type: Optional = None

    # === RELATIONS (Sets pour compatibilité) ===
    dependencies: List[Dict[str, Any]] = field(default_factory=list)# Set[str] = field(default_factory=set)
    called_functions: List[Dict[str, Any]] = field(default_factory=list)
    concepts: Set[str] = field(default_factory=set)

    qualified_name: Optional[str] = ""  # <- AJOUTER
    chunk_ids: Set[str] = field(default_factory=set)  # <- AJOUTER
    all_chunk_ids: List[str] = field(default_factory=list)  # <- AJOUTER

    # === REGROUPEMENT (optionnel, défauts simples) ===
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    detected_concepts: List[Dict[str, Any]] = field(default_factory=list)
    is_complete: bool = True
    is_grouped: bool = False
    source_code: str = ""
    entity_role: Literal['summary', 'documentation', 'code', 'header', 'default'] = 'default'
    notebook_summary: str = ""

    # === PROPRIÉTÉS CALCULÉES ===
    @property
    def base_name(self) -> str:
        """Nom sans suffixes _part_X"""
        return re.sub(r'_part_\d+$', '', self.entity_name)

    @property
    def name(self) -> str:
        """Alias pour compatibilité avec FortranEntity"""
        return self.entity_name

    @property
    def parent(self) -> Optional[str]:
        """Alias pour compatibilité avec FortranEntity"""
        return self.parent_entity

    def __post_init__(self):
        """Génère entity_id si manquant"""
        if not self.entity_id:
            self.entity_id = f"{self.filepath}#{self.entity_type}#{self.entity_name}#{self.start_line}"

    # === FACTORY METHODS ===
    @classmethod
    def from_parser_result(cls, name: str, entity_type: str, start_line: int, end_line: int,
                           filepath: str = "", **kwargs) -> 'UnifiedEntity':
        """Crée une entité depuis le parser (remplace FortranEntity)"""
        return cls(
            entity_name=name,
            entity_type=entity_type,
            start_line=start_line,
            end_line=end_line,
            filepath=filepath,
            filename=filepath.split('/')[-1] if filepath else "",
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Format dict compatible avec l'existant"""
        return {
            'name': self.entity_name,
            'type': self.entity_type,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'parent': self.parent_entity,
            'dependencies': list(self.dependencies),
            'called_functions': self.called_functions,
            'signature': self.signature,
            'source_method': self.source_method,
            'confidence': self.confidence,
            'filepath': self.filepath,
            'entity_id': self.entity_id,
            'is_grouped': self.is_grouped,
            'chunk_count': len(self.chunks),
            'access_level': self.access_level,
            'entity_role': self.entity_role,
            'notebook_summary': self.notebook_summary,
            'source_code_preview': self.source_code
        }


class EntityManager:
    """
    Gestionnaire unifié des entités Fortran.
    Combine et améliore la logique de l'EntityIndex et du visualiseur.
    """

    def __init__(self, document_store):
        self.document_store = document_store
        self.chunk_access = ChunkAccessManager(document_store)

        # Index principal : nom -> entity_id
        self.name_to_entity: Dict[str, str] = {}

        # Entités unifiées : entity_id -> UnifiedEntity
        self.entities: Dict[str, UnifiedEntity] = {}

        # Relations hiérarchiques
        self.parent_to_children: Dict[str, List[str]] = defaultdict(list)
        self.child_to_parent: Dict[str, str] = {}

        # Index par fichier : filepath -> [entity_ids]
        self.file_to_entities: Dict[str, List[str]] = defaultdict(list)

        # Index par type : entity_type -> [entity_ids]
        self.type_to_entities: Dict[str, List[str]] = defaultdict(list)

        # Cache pour les recherches fréquentes
        self._search_cache: Dict[str, List[str]] = {}

        self.persistence_path = None
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return

        # Stocke le chemin de persistance pour une utilisation future (save/load)
        self.persistence_path = os.path.join(self.document_store.storage_dir, 'entity_manager.pkl')

        # Essayer de charger l'état sauvegardé
        if os.path.exists(self.persistence_path):
            try:
                self.load_state()
                self._initialized = True
                logger.info(f"✅ EntityManager restauré depuis {self.persistence_path}: {len(self.entities)} entités")
                return
            except Exception as e:
                logger.error(f"Échec de la restauration de l'état, reconstruction complète... Erreur: {e}")

        # Si le chargement échoue, reconstruire comme avant (fallback)
        logger.info("🔧 Aucun état sauvegardé trouvé. Initialisation complète de l'EntityManager...")
        await self._build_unified_index()
        await self._build_relationships()
        await self._detect_and_group_split_entities()

        # Sauvegarder le nouvel état pour la prochaine fois
        self.save_state()

    def save_state(self):
        """
        Sauvegarde l'état complet de l'EntityManager dans un fichier.
        Utilise pickle pour une sérialisation rapide et efficace.
        """
        logger.info(f"💾 Sauvegarde de l'état de l'EntityManager dans {self.persistence_path}...")
        try:
            # Créer le répertoire parent s'il n'existe pas
            parent_dir = os.path.dirname(self.persistence_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            # Regrouper tous les dictionnaires d'index dans un seul objet d'état
            state = {
                'entities': self.entities,
                'name_to_entity': self.name_to_entity,
                'parent_to_children': self.parent_to_children,
                'child_to_parent': self.child_to_parent,
                'file_to_entities': self.file_to_entities,
                'type_to_entities': self.type_to_entities,
            }

            # Écrire l'objet d'état dans le fichier en mode binaire
            with open(self.persistence_path, 'wb') as f:
                pickle.dump(state, f)

            logger.info(f"✅ État de l'EntityManager sauvegardé avec succès ({len(self.entities)} entités).")

        except (IOError, pickle.PicklingError) as e:
            logger.error(f"❌ Échec de la sauvegarde de l'état de l'EntityManager : {e}", exc_info=True)

    def load_state(self):
        """
        Charge l'état complet de l'EntityManager depuis un fichier.
        Lève une exception si le fichier n'existe pas ou si le chargement échoue.
        """
        if not os.path.exists(self.persistence_path):
            raise FileNotFoundError(f"Le fichier d'état {self.persistence_path} n'a pas été trouvé.")

        logger.info(f"🔄 Chargement de l'état de l'EntityManager depuis {self.persistence_path}...")
        try:
            # Lire le fichier en mode binaire
            with open(self.persistence_path, 'rb') as f:
                state = pickle.load(f)

            # Restaurer tous les index depuis l'objet d'état chargé
            self.entities = state.get('entities', {})
            self.name_to_entity = state.get('name_to_entity', {})
            self.parent_to_children = state.get('parent_to_children', defaultdict(list))
            self.child_to_parent = state.get('child_to_parent', {})
            self.file_to_entities = state.get('file_to_entities', defaultdict(list))
            self.type_to_entities = state.get('type_to_entities', defaultdict(list))

            for entity in self.entities.values():
                if not hasattr(entity, 'entity_role'):
                    entity.entity_role = 'default'
                if not hasattr(entity, 'notebook_summary'):
                    entity.notebook_summary = ''
                if not hasattr(entity, 'source_code'):
                    entity.source_code = ''

            logger.info(f"✅ État de l'EntityManager restauré avec succès ({len(self.entities)} entités).")

        except (IOError, pickle.UnpicklingError, KeyError) as e:
            logger.error(
                f"❌ Échec du chargement de l'état de l'EntityManager. Le fichier est peut-être corrompu. Erreur : {e}",
                exc_info=True)
            # Propage l'erreur pour que la méthode appelante sache que le chargement a échoué
            raise

    async def add_document_entities(self, document_id: str):
        """Ajoute les entités d'un document à l'index"""
        try:
            await self._index_document_entities(document_id)
            logger.info(f"Entités du document {document_id} indexées")
        except Exception as e:
            logger.error(f"Erreur indexation document {document_id}: {e}")

    async def remove_document_entities(self, document_id: str):
        """Supprime les entités d'un document de l'index"""
        try:
            # Trouver les entités de ce document
            entities_to_remove = []
            for entity_id, entity in self.entities.items():
                # Vérifier si l'entité appartient à ce document
                if any(chunk_info.get('chunk_id', '').startswith(f"{document_id}-")
                       for chunk_info in entity.chunks):
                    entities_to_remove.append(entity_id)

            # Supprimer les entités
            for entity_id in entities_to_remove:
                if entity_id in self.entities:
                    entity = self.entities[entity_id]

                    # Nettoyer les index
                    self.name_to_entity.pop(entity.entity_name.lower(), None)

                    if entity.filepath in self.file_to_entities:
                        if entity_id in self.file_to_entities[entity.filepath]:
                            self.file_to_entities[entity.filepath].remove(entity_id)

                    if entity.entity_type in self.type_to_entities:
                        if entity_id in self.type_to_entities[entity.entity_type]:
                            self.type_to_entities[entity.entity_type].remove(entity_id)

                    # Supprimer l'entité
                    del self.entities[entity_id]

            logger.info(f"Supprimé {len(entities_to_remove)} entités du document {document_id}")

        except Exception as e:
            logger.error(f"Erreur suppression entités document {document_id}: {e}")

    async def list_entities(self, limit: int = 10) -> List[UnifiedEntity]:
        """Liste les entités (nouvelle méthode)"""
        all_entities = list(self.entities.values())
        return all_entities[:limit]

    def get_all_entities(self) -> List[UnifiedEntity]:
        """
        Retourne une liste de toutes les entités UnifiedEntity gérées.
        C'est la méthode à utiliser pour obtenir une collection complète
        à des fins de filtrage ou d'itération.
        """
        # self.entities est un dictionnaire {entity_id: UnifiedEntity}
        # .values() nous donne une collection de tous les objets UnifiedEntity
        return list(self.entities.values())

    async def _build_unified_index(self):
        """Construit l'index unifié depuis tous les documents"""
        all_docs = await self.document_store.get_all_documents()

        for doc_id in all_docs:
            await self._index_document_entities(doc_id)

    async def _index_document_entities(self, document_id: str):
        """Indexe les entités d'un document"""
        chunks = await self.chunk_access.get_chunks_by_document(document_id)

        for chunk in chunks:
            entity_info = await self.chunk_access.get_entity_info_from_chunk(chunk['id'])
            if not entity_info or not entity_info.get('name'):
                continue

            await self._add_entity_from_chunk(chunk, entity_info)

    async def _add_entity_from_chunk(self, chunk: Dict[str, Any], entity_info: Dict[str, Any]):
        """Ajoute une entité depuis un chunk"""
        chunk_id = chunk['id']
        entity_name = entity_info['name']
        base_name = re.sub(r'_part_\d+$', '', entity_name)

        # Déterminer l'ID unique de l'entité
        entity_id = self._generate_entity_id(entity_info, base_name)

        # Créer ou récupérer l'entité unifiée
        if entity_id not in self.entities:
            self.entities[entity_id] = UnifiedEntity(
                entity_id=entity_id,
                entity_name=base_name,
                entity_type=entity_info.get('type', 'unknown'),
                filepath=entity_info.get('filepath', ''),
                filename=entity_info.get('filename', ''),
                start_line=entity_info.get('start_pos'),
                end_line=entity_info.get('end_pos'),
                parent_entity=entity_info.get('parent', ''),
                qualified_name=entity_info.get('qualified_name', ''),
                signature=entity_info.get('signature', ''),
                source_method=entity_info.get('source_method', 'chunk_metadata'),
                access_level=entity_info.get('access_level'),
                called_functions=entity_info.get('called_functions'),
                entity_role=entity_info.get('entity_role', 'default'),
                notebook_summary=entity_info.get('notebook_summary', ''),
                source_code=entity_info.get('source_code', '')

            )

        entity = self.entities[entity_id]

        # Ajouter le chunk à l'entité
        chunk_info = {
            'chunk_id': chunk_id,
            'entity_info': entity_info,
            'part_index': self._extract_part_index(entity_name),
            'start_line': entity_info.get('start_line'),
            'end_line': entity_info.get('end_line')
        }

        entity.chunks.append(chunk_info)
        entity.chunk_ids.add(chunk_id)
        entity.all_chunk_ids.append(chunk_id)

        # Agréger les métadonnées
        entity.dependencies.update(entity_info.get('dependencies', []))
        entity.concepts.update(self._extract_concept_labels(entity_info.get('concepts', [])))
        entity.detected_concepts.extend(entity_info.get('detected_concepts', []))

        entity.called_functions.update(entity_info.get('called_functions', []))

        # Mettre à jour les bounds
        if entity_info.get('start_pos'):
            if not entity.start_line or entity_info['start_pos'] < entity.start_line:
                entity.start_line = entity_info['start_pos']
        if entity_info.get('end_pos'):
            if not entity.end_line or entity_info['end_pos'] > entity.end_line:
                entity.end_line = entity_info['end_pos']

        # Indexer par nom
        self.name_to_entity[base_name.lower()] = entity_id
        self.name_to_entity[entity_name.lower()] = entity_id

        # Indexer par fichier et type
        if entity.filepath:
            if entity_id not in self.file_to_entities[entity.filepath]:
                self.file_to_entities[entity.filepath].append(entity_id)
        if entity_id not in self.type_to_entities[entity.entity_type]:
            self.type_to_entities[entity.entity_type].append(entity_id)

    def update_entity_concepts(self, entity_id: str, new_concepts: List[Dict[str, Any]]):
        """Met à jour (ajoute) les concepts détectés pour une entité spécifique."""
        if entity_id in self.entities:
            entity = self.entities[entity_id]
            existing_uris = {c.get('concept_uri') for c in entity.detected_concepts if c.get('concept_uri')}
            for concept in new_concepts:
                if concept.get('concept_uri') not in existing_uris:
                    entity.detected_concepts.append(concept)
                    existing_uris.add(concept.get('concept_uri'))

    def _generate_entity_id(self, entity_info: Dict[str, Any], base_name: str) -> str:
        """Génère un ID unique pour une entité"""
        filepath = entity_info.get('filepath', '')
        entity_type = entity_info.get('type', 'unknown')
        start_line = entity_info.get('start_line', 0)

        # Utiliser parent_entity_id s'il existe (pour les entités splittées)
        parent_id = entity_info.get('parent_entity_id')
        if parent_id:
            return parent_id

        # Sinon créer un ID basé sur le contexte
        return f"{filepath}#{entity_type}#{base_name}#{start_line}"

    def _extract_part_index(self, entity_name: str) -> int:
        """Extrait l'index de partie depuis le nom"""
        match = re.search(r'_part_(\d+)$', entity_name)
        return int(match.group(1)) if match else 0

    def _extract_concept_labels(self, concepts: List[Any]) -> Set[str]:
        """Extrait les labels des concepts"""
        labels = set()
        for concept in concepts:
            if isinstance(concept, dict):
                label = concept.get('label', '')
                if label:
                    labels.add(label)
            else:
                labels.add(str(concept))
        return labels

    async def _build_relationships(self):
        """Construit les relations hiérarchiques"""
        for entity in self.entities.values():
            if entity.parent_entity:
                parent_entity_id = self.name_to_entity.get(entity.parent_entity.lower())
                if parent_entity_id:
                    self.parent_to_children[parent_entity_id].append(entity.entity_id)
                    self.child_to_parent[entity.entity_id] = parent_entity_id

    async def _detect_and_group_split_entities(self):
        """Détecte et regroupe les entités splittées"""
        for entity in self.entities.values():
            if len(entity.chunks) > 1:
                entity.is_grouped = True
                entity.expected_parts = len(entity.chunks)

                # Trier les chunks par partie
                entity.chunks.sort(key=lambda x: x.get('part_index', 0))

                # Vérifier la complétude
                part_indices = [chunk.get('part_index', 0) for chunk in entity.chunks]
                expected_indices = list(range(1, len(entity.chunks) + 1))
                entity.is_complete = sorted(part_indices) == expected_indices

                logger.debug(f"Entité regroupée: {entity.entity_name} "
                             f"({len(entity.chunks)} parties, "
                             f"{'complète' if entity.is_complete else 'incomplète'})")

    # === Méthodes de recherche et accès ===

    async def find_entity(self, name: str) -> Optional[UnifiedEntity]:
        """Trouve une entité par nom (exact ou fuzzy)"""
        entity_id = self.name_to_entity.get(name.lower())
        if entity_id:
            return self.entities.get(entity_id)

        # Recherche fuzzy
        return await self._fuzzy_search_entity(name)

    async def _fuzzy_search_entity(self, name: str) -> Optional[UnifiedEntity]:
        """Recherche fuzzy d'entité"""
        name_lower = name.lower()

        # Chercher des noms similaires
        similar_names = [
            entity_name for entity_name in self.name_to_entity.keys()
            if name_lower in entity_name or entity_name in name_lower
        ]

        if similar_names:
            # Prendre le plus similaire (par longueur)
            best_match = min(similar_names, key=lambda x: abs(len(x) - len(name_lower)))
            entity_id = self.name_to_entity[best_match]
            return self.entities.get(entity_id)

        return None

    async def get_entities_by_type(self, entity_type: str) -> List[UnifiedEntity]:
        """Récupère les entités par type - VERSION CORRIGÉE POUR ÉVITER GENERATORS"""
        if not self._initialized:
            await self.initialize()

        entity_ids = self.type_to_entities.get(entity_type, [])

        # CORRECTION: S'assurer de retourner une liste, pas un generator
        entities = []
        for eid in entity_ids:
            if eid in self.entities:
                entities.append(self.entities[eid])

        return entities  # Liste explicite

    async def get_entities_in_file(self, filepath: str) -> List[UnifiedEntity]:
        """Récupère les entités dans un fichier - VERSION CORRIGÉE"""
        if not self._initialized:
            await self.initialize()

        entity_ids = self.file_to_entities.get(filepath, [])

        # CORRECTION: Liste explicite
        entities = []
        for eid in entity_ids:
            if eid in self.entities:
                entities.append(self.entities[eid])

        return entities  # Liste explicite

    async def get_children(self, entity_id: str) -> List[UnifiedEntity]:
        """Récupère les entités enfants - VERSION CORRIGÉE"""
        if not self._initialized:
            await self.initialize()

        child_ids = self.parent_to_children.get(entity_id, [])
        children = []

        for child_id in child_ids:
            if child_id in self.entities:
                children.append(self.entities[child_id])

        return children  # Retourner une liste, pas un generator

    async def get_parent(self, entity_id: str) -> Optional[UnifiedEntity]:
        """Récupère l'entité parent"""
        parent_id = self.child_to_parent.get(entity_id)
        return self.entities.get(parent_id) if parent_id else None

    # === Méthodes d'analyse des dépendances ===

    async def find_entity_callers(self, entity_name: str) -> List[Dict[str, Any]]:
        """Trouve qui appelle une entité, avec le numéro de ligne EXACT de l'appel."""
        cache_key = f"callers_{entity_name.lower()}"
        cached_result = await global_cache.function_calls.get(cache_key)
        if cached_result:
            return cached_result

        callers = []
        target_name_lower = entity_name.lower()

        for entity in self.entities.values():
            if entity.entity_name.lower() == target_name_lower:
                continue

            if not isinstance(entity.called_functions, list):
                continue

            # ▼▼▼ NOUVELLE LOGIQUE D'ITÉRATION PRÉCISE ▼▼▼
            # On parcourt chaque appel DANS l'entité pour trouver une correspondance.
            for call_info in entity.called_functions:
                # Sécurité : s'assurer que call_info est bien un dictionnaire avec un nom
                if isinstance(call_info, dict) and 'name' in call_info:
                    call_name_lower = call_info['name'].lower()

                    # Si le nom de l'appel correspond à notre cible...
                    if call_name_lower == target_name_lower:
                        # On a trouvé un appel ! On récupère la ligne de CET appel.
                        line_of_call = call_info.get('line', 0)  # Ligne exacte de l'appel

                        # On ajoute les infos de l'entité APPELANTE avec la LIGNE EXACTE.
                        callers.append({
                            'name': entity.entity_name,  # Qui appelle
                            'type': entity.entity_type,
                            'file': entity.filepath,
                            'entity_id': entity.entity_id,
                            'call_line': line_of_call  # <--- INFORMATION PRÉCISE
                        })

        await global_cache.function_calls.set(cache_key, callers, ttl=1800)
        return callers

    async def get_dependency_graph(self, entity_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """Construit un graphe de dépendances centré sur une entité"""
        cache_key = f"depgraph_{entity_name.lower()}_{max_depth}"

        # Vérifier le cache
        cached_graph = await global_cache.dependency_graphs.get(cache_key)
        if cached_graph:
            return cached_graph

        graph = await self._build_dependency_subgraph(entity_name, max_depth)

        # Mettre en cache
        await global_cache.dependency_graphs.set(cache_key, graph, ttl=3600)  # 1h

        return graph

    async def _build_dependency_subgraph(self, entity_name: str, max_depth: int) -> Dict[str, Any]:
        """Construit un sous-graphe de dépendances"""
        entity = await self.find_entity(entity_name)
        if not entity:
            return {}

        visited = set()
        graph = {
            'root_entity': entity_name,
            'nodes': {},
            'edges': [],
            'levels': defaultdict(list)
        }

        # BFS pour explorer les dépendances
        queue = [(entity.entity_id, 0)]

        while queue and len(graph['nodes']) < 50:  # Limiter la taille
            entity_id, level = queue.pop(0)

            if entity_id in visited or level > max_depth:
                continue

            visited.add(entity_id)
            current_entity = self.entities.get(entity_id)
            if not current_entity:
                continue

            # Ajouter le nœud
            graph['nodes'][current_entity.entity_name] = {
                'type': current_entity.entity_type,
                'file': current_entity.filepath,
                'level': level,
                'entity_id': entity_id,
                'is_grouped': current_entity.is_grouped
            }

            graph['levels'][level].append(current_entity.entity_name)

            # Explorer les dépendances
            for dep_name in current_entity.dependencies:
                dep_entity = await self.find_entity(dep_name)
                if dep_entity and dep_entity.entity_id not in visited:
                    queue.append((dep_entity.entity_id, level + 1))
                    graph['edges'].append({
                        'from': current_entity.entity_name,
                        'to': dep_name,
                        'type': 'uses'
                    })

            # Explorer les appels de fonctions
            for call_name in current_entity.called_functions:
                call_entity = await self.find_entity(call_name)
                if call_entity and call_entity.entity_id not in visited:
                    queue.append((call_entity.entity_id, level + 1))
                    graph['edges'].append({
                        'from': current_entity.entity_name,
                        'to': call_name,
                        'type': 'calls'
                    })

        return graph

    # === Méthodes utilitaires et statistiques ===

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du gestionnaire"""
        type_counts = defaultdict(int)
        grouped_count = 0
        incomplete_count = 0
        total_chunks = 0

        for entity in self.entities.values():
            type_counts[entity.entity_type] += 1
            if entity.is_grouped:
                grouped_count += 1
            if not entity.is_complete:
                incomplete_count += 1
            total_chunks += len(entity.chunks)

        return {
            'total_entities': len(self.entities),
            'total_chunks': total_chunks,
            'grouped_entities': grouped_count,
            'incomplete_entities': incomplete_count,
            'entity_types': dict(type_counts),
            'files_indexed': len(self.file_to_entities),
            'compression_ratio': total_chunks / max(1, len(self.entities))
        }

    async def get_stats_async(self) -> Dict[str, Any]:
        """Version async des statistiques"""
        return self.get_stats()  # Déléguer à la version sync

    async def update_function_calls(self, entity_name: str, called_functions: List[str]):
        """Met à jour les appels de fonctions d'une entité"""
        entity = await self.find_entity(entity_name)
        if entity:
            entity.called_functions = set(called_functions)

            # Invalider les caches liés
            cache_keys_to_clear = [
                f"callers_{entity_name.lower()}",
                f"depgraph_{entity_name.lower()}_2"
            ]

            for cache_key in cache_keys_to_clear:
                await global_cache.function_calls.delete(cache_key)
                await global_cache.dependency_graphs.delete(cache_key)

    def clear_caches(self):
        """Vide les caches (utile pour tests)"""
        self._search_cache.clear()

    async def rebuild_index(self):
        """Reconstruit l'index complet"""
        logger.info("🔄 Reconstruction de l'index des entités...")

        # Réinitialiser les structures
        self.entities.clear()
        self.name_to_entity.clear()
        self.file_to_entities.clear()
        self.type_to_entities.clear()
        self.parent_to_children.clear()
        self.child_to_parent.clear()

        # Reconstruire
        await self._build_unified_index()
        await self._build_relationships()
        await self._detect_and_group_split_entities()

        logger.info(f"✅ Index reconstruit: {len(self.entities)} entités")


# Instance globale pour utilisation dans le système
_global_entity_manager = None


async def get_entity_manager(document_store) -> EntityManager:
    """Factory pour obtenir le gestionnaire d'entités global"""
    global _global_entity_manager
    if _global_entity_manager is None:
        _global_entity_manager = EntityManager(document_store)
        await _global_entity_manager.initialize()
    return _global_entity_manager
