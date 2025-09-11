"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue - Jupyter Processor
    ------------------------------------------
"""

# jupyter_analysis/integration/jupyter_document_processor.py
"""
Point d'entrée pour l'intégration avec OntoDocumentProcessor.
Gère le workflow complet pour Jupyter : parsing → entities → chunks → indexation
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import numpy as np

from ..CONSTANT import RED, TOP_K_CONCEPT
from .jupyter_notebook_parser import get_jupyter_analyzer, chunk_notebook_entities
from ..fortran_analysis.core.entity_manager import EntityManager, get_entity_manager
from ..fortran_analysis.providers.smart_orchestrator import SmartContextOrchestrator

logger = logging.getLogger(__name__)


class JupyterDocumentProcessor:
    """
    Processeur Jupyter intégré pour OntoDocumentProcessor.
    Point d'entrée principal du module jupyter_analysis.
    """

    def __init__(self, document_store=None, ontology_manager=None):
        self.document_store = document_store
        self.ontology_manager = ontology_manager

        # Composants principaux (initialisés à la demande)
        self.entity_manager: Optional[EntityManager] = None
        self.jupyter_analyzer = None
        self.orchestrator: Optional[SmartContextOrchestrator] = None

        self._initialized = False

    async def initialize_with_document_store(self, document_store, rag_engine=None):
        """
        Initialise avec document_store une fois qu'il est disponible.
        Appelé par OntoDocumentProcessor après sa propre initialisation.
        """
        if self._initialized:
            return

        logger.info("🔧 Initialisation du JupyterDocumentProcessor...")

        self.document_store = document_store

        # Utiliser l'instance globale de l'EntityManager
        self.entity_manager = await get_entity_manager(document_store)

        # Initialiser l'analyseur Jupyter
        self.jupyter_analyzer = get_jupyter_analyzer()

        # Initialiser l'orchestrateur si RAG disponible
        if rag_engine:
            self.orchestrator = SmartContextOrchestrator(document_store, rag_engine)
            await self.orchestrator.initialize()

        self._initialized = True
        logger.info("✅ JupyterDocumentProcessor initialisé")

    def _enrich_chunk_with_conceptual_level(self, chunk: Dict[str, Any], entity_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enrichit un chunk Jupyter avec le niveau conceptuel unifié"""

        jupyter_type = entity_info.get('entity_type', 'unknown')

        # Mapping Jupyter → niveau conceptuel
        level_mapping = {
            # Niveau conteneur (équivalent 'document')
            'notebook': 'container',
            'markdown_cell': 'container',

            # Niveau composant (équivalent 'section')
            'code_cell': 'component',
            'class': 'component',
            'function': 'component',
            'async function': 'component',

            # Niveau détail (équivalent 'paragraph')
            'variable': 'detail',
            'import': 'detail',
            'parameter': 'detail'
        }

        conceptual_level = level_mapping.get(jupyter_type, 'component')

        # ENRICHIR les métadonnées
        chunk['metadata']['conceptual_level'] = conceptual_level
        chunk['metadata']['native_type'] = jupyter_type
        chunk['metadata']['content_type'] = 'jupyter'
        chunk['metadata']['hierarchy_compatible'] = True

        return chunk

    async def process_jupyter_document(self,
                                       filepath: str,
                                       document_id: str,
                                       text_content: str,
                                       metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Traite un document Jupyter notebook (.ipynb).
        """
        if not self._initialized:
            raise RuntimeError("JupyterDocumentProcessor non initialisé")

        logger.info(f"📓 Traitement du notebook Jupyter: {Path(filepath).name}")

        try:
            # 1. Analyser le notebook
            entities, raw_content = self.jupyter_analyzer.analyze_file(filepath)

            if not entities:
                logger.warning(f"Aucune entité trouvée dans {filepath}. Création d'un chunk de fallback.")
                return self._create_fallback_chunk(document_id, filepath, text_content, metadata)

            logger.info(f"📊 {len(entities)} entités détectées dans le notebook.")

            # 2. Indexer les entités
            indexed_entities = await self._index_entities_for_document(document_id, entities)

            # 3. Créer les chunks sémantiques
            chunks = await self._create_semantic_chunks_from_entities(
                entities,
                indexed_entities,
                document_id,
                filepath,
                raw_content,
                metadata
            )

            # 4. Enrichissement des concepts au niveau du chunk
            if self.ontology_manager and hasattr(self.ontology_manager, 'classifier'):
                chunks = await self._enrich_chunks_with_concepts_batch(chunks)

            # 5. Enrichir les chunks avec niveau conceptuel
            final_chunks = []
            for chunk in chunks:
                entity_info = chunk.get('metadata', {})

                # Enrichir le chunk avec le niveau conceptuel
                chunk = self._enrich_chunk_with_conceptual_level(chunk, entity_info)
                final_chunks.append(chunk)

                # Mettre à jour l'entité correspondante avec les concepts du chunk
                entity_id = entity_info.get('entity_id')
                chunk_concepts = entity_info.get('detected_concepts')
                if entity_id and chunk_concepts:
                    self.entity_manager.update_entity_concepts(entity_id, chunk_concepts)

            # ✅ CORRECTION : Nettoyer les chunks pour la sérialisation
            final_chunks = self._sanitize_chunks_for_serialization(final_chunks)


            self.entity_manager.save_state()
            logger.info(f"✅ {len(chunks)} chunks créés et {len(indexed_entities)} entités indexées.")
            return final_chunks

        except Exception as e:
            logger.error(f"❌ Erreur traitement Jupyter {filepath}: {e}", exc_info=True)
            return self._create_fallback_chunk(document_id, filepath, text_content, metadata)

    async def _index_entities_for_document(self, document_id: str, entities: List) -> List:
        """Indexation et retour des entités indexées pour Jupyter"""
        try:
            logger.info(f"🔧 Indexation de {len(entities)} entités Jupyter pour document {document_id}")

            indexed_entities = []

            for entity in entities:
                try:
                    # Générer un entity_id unique si pas déjà défini
                    if not entity.entity_id:
                        entity.entity_id = f"{document_id}#{entity.entity_type}#{entity.entity_name}#{entity.start_line}"

                    # Assurer que l'EntityManager est disponible
                    if not self.entity_manager:
                        logger.error("EntityManager non disponible")
                        continue

                    # INDEXATION DIRECTE dans EntityManager
                    entity_name_lower = entity.entity_name.lower()

                    # 1. Index par nom (clé principale)
                    self.entity_manager.name_to_entity[entity_name_lower] = entity.entity_id

                    # 2. Stocker l'entité dans la base
                    self.entity_manager.entities[entity.entity_id] = entity

                    # 3. Index par fichier
                    if entity.filepath:
                        if entity.entity_id not in self.entity_manager.file_to_entities[entity.filepath]:
                            self.entity_manager.file_to_entities[entity.filepath].append(entity.entity_id)

                    # 4. Index par type
                    if entity.entity_id not in self.entity_manager.type_to_entities[entity.entity_type]:
                        self.entity_manager.type_to_entities[entity.entity_type].append(entity.entity_id)

                    indexed_entities.append(entity)
                    logger.debug(f"✅ Entité indexée: {entity.entity_name} ({entity.entity_type}) -> {entity.entity_id}")

                except Exception as e:
                    logger.warning(f"❌ Erreur indexation entité {entity.entity_name}: {e}")
                    continue

            # APRÈS indexation, construire les relations
            await self._build_entity_relationships()

            # Marquer comme initialisé
            self.entity_manager._initialized = True

            logger.info(f"✅ {len(indexed_entities)} entités Jupyter indexées avec succès pour document {document_id}")
            return indexed_entities

        except Exception as e:
            logger.error(f"❌ Erreur indexation entités Jupyter pour {document_id}: {e}")
            return entities

    async def _create_semantic_chunks_from_entities(
            self,
            entities: List,
            indexed_entities: List,
            document_id: str,
            filepath: str,
            raw_content: Dict[str, Any],
            metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Crée des chunks sémantiques pour Jupyter en utilisant VOTRE chunker intelligent.
        """
        chunks = []

        # ✅ UTILISATION DE VOTRE CHUNKER FOURNI
        notebook_chunks = chunk_notebook_entities(
            entities,
            target_size=400,
            max_size=800,
            chars_per_token=4  # Vous pouvez ajuster selon vos besoins (4 pour anglais, 5 pour français)
        )

        logger.info(f"📓 Chunker Jupyter a créé {len(notebook_chunks)} chunks optimisés")

        for chunk_idx, notebook_chunk in enumerate(notebook_chunks):
            chunk_id = f"{document_id}-chunk-{chunk_idx}"
            chunk_text = notebook_chunk["content"]  # Le texte combiné des entités
            estimated_tokens = notebook_chunk["tokens"]  # L'estimation de tokens de votre chunker

            # Trouver les entités correspondantes pour cette partie du texte
            relevant_entities = self._find_entities_for_chunk_text(chunk_text, indexed_entities)

            # Prendre la première entité comme entité principale (ou créer une par défaut)
            if relevant_entities:
                main_entity = relevant_entities[0]
                entity_metadata = {
                    'entity_name': main_entity.entity_name,
                    'entity_type': main_entity.entity_type,
                    'entity_id': main_entity.entity_id,
                    'start_pos': main_entity.start_line,
                    'end_pos': main_entity.end_line,
                    'filepath': main_entity.filepath,
                    'filename': main_entity.filename,
                    'dependencies': list(main_entity.dependencies),
                    'called_functions': list(main_entity.called_functions),
                    'parent_entity_name': main_entity.parent_entity,
                    'signature': main_entity.signature,
                    'source_method': main_entity.source_method,
                    'confidence': main_entity.confidence,
                    'detected_concepts': main_entity.detected_concepts,
                    'concepts': list(main_entity.concepts),
                    'notebook_summary': getattr(main_entity, 'notebook_summary', ''),
                    'entity_role': getattr(main_entity, 'entity_role', 'default'),
                }
            else:
                # Entité par défaut
                entity_metadata = {
                    'entity_name': f'jupyter_chunk_{chunk_idx}',
                    'entity_type': 'jupyter_fragment',
                    'entity_id': f"{document_id}#fragment#{chunk_idx}",
                    'start_pos': 1,
                    'end_pos': 1,
                    'filepath': filepath,
                    'filename': os.path.basename(filepath),
                }

            # Construire les métadonnées complètes du chunk
            chunk_metadata = {
                **metadata,
                **entity_metadata,
                'chunk_index': chunk_idx,
                'estimated_tokens': estimated_tokens,  # ✅ Utilise l'estimation de VOTRE chunker
                'is_notebook_chunk': True,
                'chunk_strategy': 'jupyter_intelligent_chunker',  # ✅ Indique l'utilisation de votre chunker
                'parser_version': 'jupyter_ast_v1.0',
                'relevant_entities_count': len(relevant_entities),
            }

            chunk = {
                'id': chunk_id,
                'text': chunk_text,
                'metadata': chunk_metadata
            }

            chunks.append(chunk)
            logger.debug(f"✅ Chunk {chunk_idx}: {estimated_tokens} tokens, {len(relevant_entities)} entités")

        return chunks

    def _find_entities_for_chunk_text(self, chunk_text: str, entities: List) -> List:
        """
        Trouve les entités qui correspondent le mieux à un texte de chunk donné.
        Utilise une correspondance basée sur le contenu.
        """
        relevant_entities = []

        for entity in entities:
            if hasattr(entity, 'source_code') and entity.source_code:
                # Si le code source de l'entité est inclus dans le chunk
                if entity.source_code.strip() in chunk_text:
                    relevant_entities.append(entity)
                # Ou si le chunk contient le nom de l'entité
                elif entity.entity_name in chunk_text:
                    relevant_entities.append(entity)

        return relevant_entities

    def _sanitize_chunks_for_serialization(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Nettoie les chunks pour s'assurer qu'ils sont sérialisables avec pickle.
        Supprime toute coroutine, fonction, ou objet non sérialisable.
        """
        import types
        import inspect

        def is_serializable(obj):
            """Vérifie si un objet est sérialisable avec pickle"""
            if obj is None:
                return True
            if isinstance(obj, (str, int, float, bool, list, tuple, dict)):
                return True
            if inspect.iscoroutine(obj) or inspect.isfunction(obj) or inspect.ismethod(obj):
                return False
            if isinstance(obj, types.GeneratorType):
                return False
            return True

        def clean_dict(d):
            """Nettoie récursivement un dictionnaire"""
            if not isinstance(d, dict):
                if isinstance(d, list):
                    return [clean_dict(item) for item in d if is_serializable(item)]
                elif isinstance(d, tuple):
                    return tuple(clean_dict(item) for item in d if is_serializable(item))
                elif is_serializable(d):
                    return d
                else:
                    logger.warning(f"Objet non sérialisable supprimé: {type(d)}")
                    return None

            cleaned = {}
            for key, value in d.items():
                if not is_serializable(key):
                    logger.warning(f"Clé non sérialisable ignorée: {type(key)}")
                    continue

                if isinstance(value, dict):
                    cleaned_value = clean_dict(value)
                    if cleaned_value is not None:
                        cleaned[key] = cleaned_value
                elif isinstance(value, (list, tuple)):
                    cleaned_items = []
                    for item in value:
                        cleaned_item = clean_dict(item)
                        if cleaned_item is not None:
                            cleaned_items.append(cleaned_item)
                    if isinstance(value, tuple):
                        cleaned[key] = tuple(cleaned_items)
                    else:
                        cleaned[key] = cleaned_items
                elif is_serializable(value):
                    cleaned[key] = value
                else:
                    logger.warning(f"Valeur non sérialisable supprimée pour clé '{key}': {type(value)}")

            return cleaned

        logger.info(f"🧹 Nettoyage de {len(chunks)} chunks pour la sérialisation...")

        sanitized_chunks = []
        for i, chunk in enumerate(chunks):
            try:
                # Nettoyer le chunk complet
                clean_chunk = clean_dict(chunk)

                # Vérification supplémentaire des métadonnées critiques
                if 'metadata' in clean_chunk:
                    metadata = clean_chunk['metadata']

                    # S'assurer que les listes sont bien des listes simples
                    if 'dependencies' in metadata:
                        deps = metadata['dependencies']
                        if isinstance(deps, list):
                            clean_deps = []
                            for dep in deps:
                                if isinstance(dep, dict) and 'name' in dep:
                                    clean_deps.append({
                                        'name': str(dep['name']),
                                        'line': int(dep.get('line', 0))
                                    })
                                elif isinstance(dep, str):
                                    clean_deps.append({'name': dep, 'line': 0})
                            metadata['dependencies'] = clean_deps

                    if 'called_functions' in metadata:
                        calls = metadata['called_functions']
                        if isinstance(calls, list):
                            clean_calls = []
                            for call in calls:
                                if isinstance(call, dict) and 'name' in call:
                                    clean_calls.append({
                                        'name': str(call['name']),
                                        'line': int(call.get('line', 0))
                                    })
                                elif isinstance(call, str):
                                    clean_calls.append({'name': call, 'line': 0})
                            metadata['called_functions'] = clean_calls

                    if 'detected_concepts' in metadata:
                        concepts = metadata['detected_concepts']
                        if isinstance(concepts, list):
                            clean_concepts = []
                            for concept in concepts:
                                if isinstance(concept, dict):
                                    clean_concept = {}
                                    for k, v in concept.items():
                                        if is_serializable(v):
                                            clean_concept[str(k)] = v
                                    if clean_concept:
                                        clean_concepts.append(clean_concept)
                            metadata['detected_concepts'] = clean_concepts

                sanitized_chunks.append(clean_chunk)

            except Exception as e:
                logger.error(f"Erreur lors du nettoyage du chunk {i}: {e}")
                # Créer un chunk de fallback
                fallback_chunk = {
                    'id': chunk.get('id', f'fallback-{i}'),
                    'text': str(chunk.get('text', '')),
                    'metadata': {
                        'entity_name': f'sanitized_chunk_{i}',
                        'entity_type': 'fallback',
                        'chunk_index': i,
                        'sanitization_error': True
                    }
                }
                sanitized_chunks.append(fallback_chunk)

        logger.info(f"✅ {len(sanitized_chunks)} chunks nettoyés et prêts pour la sérialisation")
        return sanitized_chunks

    async def _enrich_chunks_with_concepts_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Version optimisée qui enrichit les chunks Jupyter avec des concepts en traitant
        tous les chunks en parallèle (batch processing).
        """
        logger.info(f"🚀🔬 Lancement de l'enrichissement par lots de {len(chunks)} chunks Jupyter...")

        # Vérification initiale
        classifier = getattr(self.ontology_manager, 'classifier', None)
        if not classifier:
            logger.warning("Classifier non disponible ou invalide. Skip de l'enrichissement par lots.")
            return chunks

        # Collecter les textes des chunks pertinents
        chunks_to_process: List[Tuple[Dict[str, Any], str]] = []
        for chunk in chunks:
            chunk_text = chunk.get('text', '')
            if chunk_text:
                chunks_to_process.append((chunk, chunk_text))

        if not chunks_to_process:
            logger.info("Aucun chunk Jupyter avec du contenu textuel à traiter.")
            return chunks

        texts_to_embed = [text for chunk, text in chunks_to_process]
        logger.info(f"Préparation de l'embedding pour {len(texts_to_embed)} textes Jupyter.")

        try:
            # BATCH EMBEDDING
            all_embeddings = await classifier.rag_engine.embedding_manager.provider.generate_embeddings(texts_to_embed)

            if not all_embeddings or len(all_embeddings) != len(texts_to_embed):
                logger.error("L'appel batch aux embeddings a échoué pour Jupyter.")
                return chunks

            # Normalisation des embeddings
            normalized_embeddings = []
            for emb in all_embeddings:
                norm = np.linalg.norm(emb)
                normalized_embeddings.append(emb / norm if norm > 0 else emb)

            # CLASSIFICATION EN PARALLÈLE
            classification_tasks = []
            for embedding in normalized_embeddings:
                task = classifier.concept_classifier.auto_detect_concepts(
                    query_embedding=embedding,
                    min_confidence=0.3,
                    max_concepts=TOP_K_CONCEPT
                )
                classification_tasks.append(task)

            logger.info(f"Lancement de {len(classification_tasks)} classifications Jupyter en parallèle...")
            all_results = await asyncio.gather(*classification_tasks, return_exceptions=True)
            logger.info("Toutes les classifications Jupyter sont terminées.")

            # RÉ-ASSOCIER les résultats aux chunks
            for i, result in enumerate(all_results):
                original_chunk = chunks_to_process[i][0]
                metadata = original_chunk.setdefault('metadata', {})

                if isinstance(result, Exception):
                    entity_name = metadata.get('entity_name', 'N/A')
                    logger.warning(f"⚠️ Erreur de classification pour un chunk Jupyter de '{entity_name}': {result}")
                elif result:
                    metadata['detected_concepts'] = result[:TOP_K_CONCEPT]

        except Exception as e:
            logger.error(f"Erreur majeure durant le traitement par lots des concepts Jupyter: {e}", exc_info=True)

        return chunks

    async def _build_entity_relationships(self):
        """Construit les relations parent-enfant entre entités Jupyter"""
        try:
            # Nettoyer les relations existantes
            self.entity_manager.parent_to_children.clear()
            self.entity_manager.child_to_parent.clear()

            for entity in self.entity_manager.entities.values():
                if entity.parent_entity:
                    # Trouver l'entity_id du parent
                    parent_entity_id = self.entity_manager.name_to_entity.get(entity.parent_entity.lower())
                    if parent_entity_id:
                        # Ajouter la relation parent -> enfant
                        if parent_entity_id not in self.entity_manager.parent_to_children:
                            self.entity_manager.parent_to_children[parent_entity_id] = []
                        self.entity_manager.parent_to_children[parent_entity_id].append(entity.entity_id)

                        # Ajouter la relation enfant -> parent
                        self.entity_manager.child_to_parent[entity.entity_id] = parent_entity_id

                        logger.debug(f"🔗 Relation Jupyter: {entity.parent_entity} -> {entity.entity_name}")

            logger.info(f"✅ Relations Jupyter construites: {len(self.entity_manager.parent_to_children)} parents")

        except Exception as e:
            logger.error(f"❌ Erreur construction relations Jupyter: {e}")

    def _create_fallback_chunk(self, document_id: str, filepath: str,
                               text_content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Crée un chunk de fallback en cas d'erreur de parsing"""
        chunk_id = f"{document_id}-chunk-0"

        fallback_metadata = {
            **metadata,
            'entity_name': Path(filepath).stem,
            'entity_type': 'fallback_notebook',
            'chunk_index': 0,
            'is_fallback': True,
            'parser_error': True,
            'chunk_strategy': 'fallback_full_notebook'
        }

        return [{
            'id': chunk_id,
            'text': text_content,
            'metadata': fallback_metadata
        }]

    # === Méthodes utilitaires pour l'intégration ===

    async def get_entity_context(self, entity_name: str, context_type: str = "local") -> Dict[str, Any]:
        """
        Récupère le contexte d'une entité Jupyter (utilisé par le RAG).
        """
        if not self._initialized or not self.orchestrator:
            return {"error": "JupyterDocumentProcessor non initialisé"}

        if context_type == "smart":
            return await self.orchestrator.get_context_for_agent(
                entity_name, "data_scientist", "notebook_understanding"
            )
        elif context_type == "local":
            return await self.orchestrator.get_local_context(entity_name)
        elif context_type == "global":
            return await self.orchestrator.get_global_context(entity_name)
        elif context_type == "semantic":
            return await self.orchestrator.get_semantic_context(entity_name)
        else:
            return {"error": f"Type de contexte non supporté: {context_type}"}

    async def search_entities(self, query: str) -> List[Dict[str, Any]]:
        """Recherche d'entités Jupyter (interface pour le RAG)"""
        if not self.orchestrator:
            return []
        return await self.orchestrator.search_entities(query)

    async def sync_orchestrator_with_entities(self):
        """Synchronise l'orchestrateur avec les entités de l'EntityManager"""
        try:
            if not self.orchestrator or not self.entity_manager:
                return

            logger.info("🔄 Synchronisation de l'orchestrateur avec les nouvelles entités Jupyter...")
            await self.orchestrator.initialize()
            logger.info("✅ Synchronisation Jupyter terminée")

        except Exception as e:
            logger.error(f"Erreur synchronisation orchestrateur Jupyter: {e}")

    async def diagnose_entity_manager(self) -> Dict[str, Any]:
        """Diagnostic de l'EntityManager pour Jupyter"""
        if not hasattr(self, 'entity_manager') or self.entity_manager is None:
            return {
                "error": "EntityManager non initialisé pour Jupyter",
                "initialized": self._initialized,
                "has_entity_manager_attr": hasattr(self, 'entity_manager'),
                "entity_manager_is_none": self.entity_manager is None if hasattr(self, 'entity_manager') else True
            }

        diagnosis = {
            "initialized": self._initialized,
            "entity_manager_initialized": self.entity_manager._initialized if hasattr(self.entity_manager,
                                                                                      '_initialized') else False,
            "total_entities": len(self.entity_manager.entities),
            "entities_by_type": {},
            "entities_by_file": {},
            "sample_entities": [],
            "index_health": {
                "name_to_entity_count": len(self.entity_manager.name_to_entity),
                "file_to_entities_count": len(self.entity_manager.file_to_entities),
                "type_to_entities_count": len(self.entity_manager.type_to_entities)
            }
        }

        # Compter par type
        for entity in self.entity_manager.entities.values():
            entity_type = entity.entity_type
            diagnosis["entities_by_type"][entity_type] = diagnosis["entities_by_type"].get(entity_type, 0) + 1

        # Compter par fichier
        for filepath, entity_ids in self.entity_manager.file_to_entities.items():
            filename = Path(filepath).name
            diagnosis["entities_by_file"][filename] = len(entity_ids)

        # Échantillon d'entités
        sample_entities = list(self.entity_manager.entities.values())[:10]
        for entity in sample_entities:
            diagnosis["sample_entities"].append({
                "name": entity.entity_name,
                "type": entity.entity_type,
                "file": Path(entity.filepath).name if entity.filepath else "unknown",
                "entity_id": entity.entity_id
            })

        return diagnosis

    def get_stats(self) -> Dict[str, Any]:
        """Statistiques du module Jupyter"""
        stats = {
            'initialized': self._initialized,
            'entity_manager_stats': {},
            'cache_stats': {}
        }

        if self.entity_manager:
            stats['entity_manager_stats'] = self.entity_manager.get_stats()

        if self.orchestrator:
            stats['cache_stats'] = self.orchestrator.get_cache_stats()

        return stats


# Instance globale pour usage dans OntoDocumentProcessor
_global_jupyter_processor = JupyterDocumentProcessor()


async def get_jupyter_processor(document_store=None, rag_engine=None,
                                ontology_manager=None) -> JupyterDocumentProcessor:
    """Factory pour obtenir le processeur Jupyter global"""
    global _global_jupyter_processor

    if document_store and not _global_jupyter_processor._initialized:
        _global_jupyter_processor.ontology_manager = ontology_manager
        await _global_jupyter_processor.initialize_with_document_store(document_store, rag_engine)

    return _global_jupyter_processor