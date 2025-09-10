"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# fortran_analysis/integration/document_processor_integration.py
"""
Point d'entrée pour l'intégration avec OntoDocumentProcessor.
Gère le workflow complet : parsing → entities → chunks → indexation
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import numpy as np

from ...CONSTANT import RED, TOP_K_CONCEPT
from ..core.hybrid_fortran_parser import get_fortran_analyzer
from ..core.entity_manager import EntityManager
from ..providers.smart_orchestrator import SmartContextOrchestrator
from ..utils.chunk_access import ChunkAccessManager

logger = logging.getLogger(__name__)

class FortranDocumentProcessor:
    """
    Processeur Fortran intégré pour OntoDocumentProcessor.
    Point d'entrée principal du module fortran_analysis.
    """

    def __init__(self, document_store=None, ontology_manager=None):
        self.document_store = document_store  # Sera fourni plus tard
        self.ontology_manager = ontology_manager

        # Composants principaux (initialisés à la demande)
        self.entity_manager: Optional[EntityManager] = None
        self.fortran_analyzer = None
        self.orchestrator: Optional[SmartContextOrchestrator] = None

        self._initialized = False

    async def initialize_with_document_store(self, document_store, rag_engine=None):
        """
        Initialise avec document_store une fois qu'il est disponible.
        Appelé par OntoDocumentProcessor après sa propre initialisation.
        """
        if self._initialized:
            return

        logger.info("🔧 Initialisation du FortranDocumentProcessor...")

        self.document_store = document_store

        # CORRECTION: Utiliser l'instance globale au lieu de créer une nouvelle
        from ..core.entity_manager import get_entity_manager
        self.entity_manager = await get_entity_manager(document_store)

        # Initialiser EntityManager
        # self.entity_manager = EntityManager(document_store)
        # await self.entity_manager.initialize()

        # Initialiser l'analyseur
        self.fortran_analyzer = get_fortran_analyzer("hybrid")

        # Initialiser l'orchestrateur si RAG disponible
        if rag_engine:
            self.orchestrator = SmartContextOrchestrator(document_store, rag_engine)
            await self.orchestrator.initialize()

        self._initialized = True
        logger.info("✅ FortranDocumentProcessor initialisé")

    def _enrich_chunk_with_conceptual_level(self, chunk: Dict[str, Any], entity_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enrichit un chunk Fortran avec le niveau conceptuel unifié"""

        fortran_type = entity_info.get('entity_type', 'unknown')

        # Mapping Fortran → niveau conceptuel
        level_mapping = {
            # Niveau conteneur (équivalent 'document')
            'module': 'container',
            'program': 'container',

            # Niveau composant (équivalent 'section')
            'function': 'component',
            'subroutine': 'component',
            'type_definition': 'component',
            'interface': 'component',

            # Niveau détail (équivalent 'paragraph')
            'internal_function': 'detail',
            'variable_declaration': 'detail',
            'parameter': 'detail'
        }

        conceptual_level = level_mapping.get(fortran_type, 'component')

        # ENRICHIR les métadonnées
        chunk['metadata']['conceptual_level'] = conceptual_level
        chunk['metadata']['native_type'] = fortran_type  # Garder l'info originale
        chunk['metadata']['content_type'] = 'fortran'  # Marquer comme Fortran
        chunk['metadata']['hierarchy_compatible'] = True  # Compatible avec recherche hiérarchique

        return chunk


    async def diagnose_entity_manager(self) -> Dict[str, Any]:
        """CORRECTION: Vérifier d'abord que entity_manager existe"""

        # VÉRIFICATION: entity_manager existe-t-il ?
        if not hasattr(self, 'entity_manager') or self.entity_manager is None:
            return {
                "error": "EntityManager non initialisé",
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

    async def process_fortran_document(self,
                                       filepath: str,
                                       document_id: str,
                                       text_content: str,
                                       metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """VERSION FINALE qui gère la correction des numéros de ligne."""
        if not self._initialized:
            raise RuntimeError("FortranDocumentProcessor non initialisé")

        logger.info(f"🔬 Traitement du fichier Fortran avec gestion des 'include': {Path(filepath).name}")

        try:
            # 1. Obtenir les entités (avec lignes aplaties), le code aplati, ET la carte des sources
            entities, full_processed_code, source_map = self.fortran_analyzer.analyze_file(
                filepath, self.ontology_manager
            )

            if not entities:
                logger.warning(f"Aucune entité trouvée dans {filepath}. Création d'un chunk de fallback.")
                return self._create_fallback_chunk(document_id, filepath, text_content, metadata)

            logger.info(f"📊 {len(entities)} entités brutes détectées.")

            # =================================================================
            # 2. CORRIGER LES ENTITÉS AVANT DE LES MANIPULER
            #    Cette étape met à jour les entités en mémoire avec les bonnes informations.
            # =================================================================
            corrected_entities = self._correct_entities_with_sourcemap(entities, source_map)

            # 3. Indexer les entités maintenant corrigées
            indexed_entities = await self._index_entities_for_document(document_id, corrected_entities)

            # 4. Créer les chunks sémantiques.
            #    IMPORTANT : on utilise le code APLATI pour le découpage, car les
            #    numéros de ligne des entités *originales* (non corrigées) correspondent à ce code.
            chunks = await self._create_semantic_chunks_from_entities(
                entities,  # <-- Utiliser les entités originales (lignes aplaties) pour le chunking
                indexed_entities,  # <-- Passer les entités corrigées pour les métadonnées riches
                document_id,
                filepath,
                full_processed_code,  # <-- Le code aplati
                metadata
            )

            # =========================================================================
            # ÉTAPE 5 : NOUVEAU - ENRICHISSEMENT DES CONCEPTS AU NIVEAU DU CHUNK
            # C'est le bon endroit pour le faire.
            # =========================================================================
            if self.ontology_manager and hasattr(self.ontology_manager, 'classifier'):
                chunks = await self._enrich_chunks_with_concepts_batch(chunks)

            # 5. Enrichir les chunks
            final_chunks = []
            for chunk in chunks:
                entity_info = chunk.get('metadata', {})  # Contient maintenant 'detected_concepts'

                # Enrichir le chunk avec le niveau conceptuel
                chunk = self._enrich_chunk_with_conceptual_level(chunk, entity_info)
                final_chunks.append(chunk)

                # On met à jour l'entité correspondante avec les concepts du chunk.
                entity_id = entity_info.get('entity_id')
                chunk_concepts = entity_info.get('detected_concepts')
                if entity_id and chunk_concepts:
                    self.entity_manager.update_entity_concepts(entity_id, chunk_concepts)

            self.entity_manager.save_state()
            logger.info(
                f"✅ {len(chunks)} chunks créés et {len(indexed_entities)} entités indexées avec des informations de ligne correctes.")
            return chunks

        except Exception as e:
            logger.error(f"❌ Erreur traitement Fortran {filepath}: {e}", exc_info=True)
            return self._create_fallback_chunk(document_id, filepath, text_content, metadata)

    async def _enrich_chunks_with_concepts_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Version optimisée qui enrichit les chunks avec des concepts en traitant
        tous les chunks en parallèle (batch processing).
        """
        logger.info(f"🚀🔬 Lancement de l'enrichissement par lots de {len(chunks)} chunks...")

        # Vérification initiale
        classifier = getattr(self.ontology_manager, 'classifier', None)
        if not classifier:
            logger.warning("Classifier non disponible ou invalide. Skip de l'enrichissement par lots.")
            return chunks

        # --- ÉTAPE 1 : COLLECTER les textes des chunks pertinents ---
        # Nous créons une liste de tuples pour garder une référence au chunk original
        chunks_to_process: List[Tuple[Dict[str, Any], str]] = []
        for chunk in chunks:
            chunk_text = chunk.get('text', '')
            if chunk_text:  # On ne traite que les chunks qui ont du texte
                chunks_to_process.append((chunk, chunk_text))

        if not chunks_to_process:
            logger.info("Aucun chunk avec du contenu textuel à traiter.")
            return chunks

        texts_to_embed = [text for chunk, text in chunks_to_process]
        logger.info(f"Préparation de l'embedding pour {len(texts_to_embed)} textes.")

        try:
            # --- ÉTAPE 2 : BATCH EMBEDDING (Un seul appel réseau) ---
            classifier = getattr(self.ontology_manager, 'classifier', None)
            if not classifier:
                logger.warning(
                    "Classifier non disponible. Skip de l'enrichissement.")
                return chunks

            all_embeddings = await classifier.rag_engine.embedding_manager.provider.generate_embeddings(texts_to_embed)

            if not all_embeddings or len(all_embeddings) != len(texts_to_embed):
                logger.error("L'appel batch aux embeddings a échoué ou retourné un nombre incorrect de résultats.")
                return chunks

            # Normalisation des embeddings, comme dans _get_text_embedding
            normalized_embeddings = []
            for emb in all_embeddings:
                norm = np.linalg.norm(emb)
                normalized_embeddings.append(emb / norm if norm > 0 else emb)

            # --- ÉTAPE 3 : CLASSIFIER EN PARALLÈLE ---
            # On prépare une liste de toutes les tâches de classification à exécuter
            classification_tasks = []
            for embedding in normalized_embeddings:
                task = classifier.concept_classifier.auto_detect_concepts(
                    query_embedding=embedding,
                    min_confidence=0.3,
                    max_concepts= TOP_K_CONCEPT
                )
                classification_tasks.append(task)

            # On exécute toutes les tâches en parallèle et on récupère les résultats
            # return_exceptions=True est CRUCIAL pour éviter qu'une seule erreur ne fasse tout planter.
            logger.info(f"Lancement de {len(classification_tasks)} classifications en parallèle...")
            all_results = await asyncio.gather(*classification_tasks, return_exceptions=True)
            logger.info("Toutes les classifications sont terminées.")

            # --- ÉTAPE 4 : RÉ-ASSOCIER les résultats aux chunks ---
            for i, result in enumerate(all_results):
                # On récupère le chunk original correspondant à ce résultat
                original_chunk = chunks_to_process[i][0]
                metadata = original_chunk.setdefault('metadata', {})  # Assure que metadata existe

                if isinstance(result, Exception):
                    entity_name = metadata.get('entity_name', 'N/A')
                    logger.warning(f"⚠️ Erreur de classification pour un chunk de '{entity_name}': {result}")
                elif result:  # Si la classification a retourné des concepts
                    metadata['detected_concepts'] = result[:TOP_K_CONCEPT]  # Top 5
                    # logger.debug(f"✅ Concepts pour '{metadata.get('entity_name')}': {[c.get('label') for c in result]}")

        except Exception as e:
            logger.error(f"Une erreur majeure est survenue durant le traitement par lots des concepts: {e}",
                         exc_info=True)

        return chunks

    async def _enrich_chunks_with_concepts(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parcourt les chunks, et pour ceux qui sont pertinents, utilise le classifier
        pour détecter et attacher des concepts ontologiques.
        """
        logger.info(f"🔬 Enrichissement de {len(chunks)} chunks avec des concepts ontologiques...")

        # Le classifier est nécessaire
        classifier = getattr(self.ontology_manager, 'classifier', None)
        if not classifier or not hasattr(classifier, 'classify_text_direct'):
            logger.warning(
                "Classifier non disponible ou ne supporte pas 'classify_text_direct'. Skip de l'enrichissement.")
            return chunks

        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            entity_type = metadata.get('entity_type')
            chunk_text = chunk.get('text', '')

            try:
                # Ici, on classifie le TEXTE FINAL du chunk.
                # La méthode `classify_text_direct` devrait gérer en interne l'appel à l'embedding.
                detected_concepts = await classifier.classify_text_direct(
                    text=chunk_text,
                    min_confidence=0.3
                )

                if detected_concepts:
                    # On stocke les concepts dans les métadonnées du CHUNK
                    metadata['detected_concepts'] = detected_concepts[:5]  # Top 5
                    logger.debug(
                        f"✅ Concept(s) pour chunk de '{metadata.get('entity_name')}': {[c.get('label') for c in detected_concepts]}")

            except Exception as e:
                logger.warning(f"⚠️ Erreur de classification pour le chunk de '{metadata.get('entity_name')}': {e}")

        return chunks

    def _correct_entities_with_sourcemap(self, entities: List, source_map: 'SourceMap') -> List:
        """
        Met à jour une liste d'entités avec les informations de fichier et de ligne
        correctes en utilisant une carte des sources.
        """
        if not source_map:  # Si la carte est vide, ne rien faire
            return entities

        for entity in entities:
            try:
                # Les numéros de ligne du parser sont 1-based, la carte est 0-indexed
                start_index = entity.start_line - 1
                end_index = entity.end_line - 1

                # Vérifier les limites pour éviter les erreurs
                if 0 <= start_index < len(source_map) and 0 <= end_index < len(source_map):
                    original_start_filepath, original_start_line = source_map[start_index]
                    _, original_end_line = source_map[end_index]  # Le fichier de fin est le même

                    # Mettre à jour l'entité
                    entity.filepath = original_start_filepath
                    entity.filename = os.path.basename(original_start_filepath)
                    entity.start_line = original_start_line
                    entity.end_line = original_end_line
                else:
                    logger.error(
                        f"Numéro de ligne ({entity.start_line}) hors des limites de la source map pour l'entité {entity.name}.")

            except IndexError:
                logger.error(f"Erreur d'index dans la source map pour l'entité {entity.name}.")
                continue

        return entities

    async def _index_entities_for_document(self, document_id: str, entities: List) -> List:
        """CORRECTION: Indexation et retour des entités indexées"""
        try:
            logger.info(f"🔧 Indexation de {len(entities)} entités pour document {document_id}")

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

            # Grouper les entités splittées
            await self._detect_grouped_entities()

            # Marquer comme initialisé
            self.entity_manager._initialized = True

            logger.info(f"✅ {len(indexed_entities)} entités indexées avec succès pour document {document_id}")
            return indexed_entities

        except Exception as e:
            logger.error(f"❌ Erreur indexation entités pour {document_id}: {e}")
            return entities  # Retourner les entités originales en cas d'erreur

    async def _create_semantic_chunks_from_entities(
            self,
            raw_entities: List,  # Entités avec numéros de ligne RELATIFS au code aplati.
            corrected_entities: List,  # Entités avec les VRAIS numéros de ligne et fichiers.
            document_id: str,
            original_filepath: str,  # Le chemin du fichier de base (utile pour le fallback).
            flattened_code: str,  # Le code source COMPLET avec les 'include' résolus.
            metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Crée des chunks sémantiques en utilisant une dissociation contrôlée des informations :
        - Utilise les 'raw_entities' pour DÉCOUPER le 'flattened_code'.
        - Utilise les 'corrected_entities' pour PEUPLER les métadonnées du chunk.
        """
        chunks = []
        # Le découpage se fait sur le code aplati, car les numéros de ligne des 'raw_entities'
        # correspondent à ce contenu.
        lines = flattened_code.splitlines()

        # Pour une performance optimale, créez une table de hachage (dictionnaire) pour
        # retrouver l'entité corrigée à partir de son ID, au lieu de la rechercher dans une boucle.
        corrected_entity_map = {ent.entity_id: ent for ent in corrected_entities}

        # ÉTAPE 1 : Itérer sur les entités brutes pour extraire le texte.
        for raw_entity in raw_entities:
            try:
                # Extraire le code de l'entité en utilisant les coordonnées de l'entité BRUTE,
                # car elles correspondent au `flattened_code`.
                start_line_raw = max(0, raw_entity.start_line - 1)  # Convert to 0-indexed
                end_line_raw = min(len(lines), raw_entity.end_line)
                entity_code = '\n'.join(lines[start_line_raw:end_line_raw])

                if not entity_code.strip():
                    continue

                # ÉTAPE 2 : Retrouver l'entité CORRIGÉE correspondante.
                # C'est elle qui contient les informations finales correctes (fichier, ligne, etc.).
                corrected_entity = corrected_entity_map.get(raw_entity.entity_id)
                if not corrected_entity:
                    logger.warning(
                        f"Impossible de trouver l'entité corrigée pour {raw_entity.entity_id}. Le chunk ne sera pas créé.")
                    continue

                # La logique de découpage d'une grosse entité reste la même.
                chunk_parts = self._split_entity_if_needed(entity_code, raw_entity)

                for part_idx, chunk_text in enumerate(chunk_parts):
                    chunk_id = f"{document_id}-chunk-{len(chunks)}"

                    # ÉTAPE 3 : Construire les métadonnées en utilisant exclusivement
                    # les informations de l'entité CORRIGÉE.
                    chunk_metadata = {
                        **metadata,
                        # --- Informations sur l'entité (provenant de l'entité CORRIGÉE) ---
                        'entity_name': corrected_entity.entity_name,
                        'entity_type': corrected_entity.entity_type,
                        'entity_id': corrected_entity.entity_id,
                        'start_pos': corrected_entity.start_line,  # VRAI numéro de ligne de début
                        'end_pos': corrected_entity.end_line,  # VRAI numéro de ligne de fin
                        'filepath': corrected_entity.filepath,  # VRAI chemin de fichier
                        'filename': corrected_entity.filename,  # VRAI nom de fichier

                        # --- Informations sur le chunking (calculées ici) ---
                        'chunk_index': len(chunks),
                        'is_split_entity': len(chunk_parts) > 1,
                        'part_index': part_idx + 1 if len(chunk_parts) > 1 else 0,
                        'total_parts': len(chunk_parts),
                        'parent_entity_id': corrected_entity.entity_id if len(chunk_parts) > 1 else None,

                        # --- Relations Fortran (provenant de l'entité CORRIGÉE) ---
                        'dependencies': list(corrected_entity.dependencies),
                        'called_functions': list(corrected_entity.called_functions),
                        'parent_entity_name': corrected_entity.parent_entity,
                        'signature': corrected_entity.signature,
                        'source_method': corrected_entity.source_method,
                        'confidence': corrected_entity.confidence,

                        # --- Concepts détectés (provenant de l'entité CORRIGÉE) ---
                        'detected_concepts': corrected_entity.detected_concepts,
                        'concepts': list(corrected_entity.concepts),

                        # --- Informations techniques (provenant de l'entité CORRIGÉE) ---
                        'is_grouped': corrected_entity.is_grouped,
                        'is_complete': corrected_entity.is_complete,
                        'qualified_name': corrected_entity.qualified_name or corrected_entity.entity_name,
                        'is_internal_function': bool(corrected_entity.parent_entity),
                        'access_level': corrected_entity.access_level,

                        # --- Métadonnées du parser ---
                        'parser_version': 'hybrid_v2.0_sourcemap',
                        'chunk_strategy': 'entity_based_semantic'
                    }

                    chunk = {
                        'id': chunk_id,
                        'text': chunk_text,  # Le texte vient du découpage du code aplati
                        'metadata': chunk_metadata  # Les métadonnées viennent de l'entité corrigée
                    }

                    chunks.append(chunk)

            except Exception as e:
                logger.warning(f"Erreur lors de la création du chunk pour l'entité {raw_entity.entity_name}: {e}",
                               exc_info=True)
                continue

        # Si, après tout cela, aucun chunk n'a pu être créé, on utilise le fallback.
        if not chunks:
            # Note : Le fallback utilise le `flattened_code` pour le contenu, car c'est une
            # représentation plus complète que le fichier original si des 'include' existent.
            logger.warning("Aucun chunk sémantique n'a pu être créé, utilisation du fallback.")
            return self._create_fallback_chunk(document_id, original_filepath, flattened_code, metadata)

        return chunks

    def _split_entity_if_needed(self, entity_code: str, entity, max_chunk_size: int = 2000) -> List[str]:
        """Splitte une entité si elle est trop grande"""
        if len(entity_code) <= max_chunk_size:
            return [entity_code]

        # Stratégie de split intelligente pour Fortran
        lines = entity_code.splitlines()
        chunks = []
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > max_chunk_size and current_chunk:
                # Sauvegarder le chunk actuel
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        # Ajouter le dernier chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def _create_fallback_chunk(self, document_id: str, filepath: str,
                               text_content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Crée un chunk de fallback en cas d'erreur de parsing"""
        chunk_id = f"{document_id}-chunk-0"

        fallback_metadata = {
            **metadata,
            'entity_name': Path(filepath).stem,
            'entity_type': 'fallback_document',
            'chunk_index': 0,
            'is_fallback': True,
            'parser_error': True,
            'chunk_strategy': 'fallback_full_document'
        }

        return [{
            'id': chunk_id,
            'text': text_content,
            'metadata': fallback_metadata
        }]

    async def _build_entity_relationships(self):
        """Construit les relations parent-enfant entre entités"""
        try:
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

                        logger.debug(f"🔗 Relation: {entity.parent_entity} -> {entity.entity_name}")

            logger.info(f"✅ Relations construites: {len(self.entity_manager.parent_to_children)} parents")

        except Exception as e:
            logger.error(f"❌ Erreur construction relations: {e}")

    async def _detect_grouped_entities(self):
        """Détecte et groupe les entités splittées (xxx_part_1, xxx_part_2, etc.)"""
        try:
            grouped_count = 0

            # Grouper par nom de base
            base_names = {}
            for entity in self.entity_manager.entities.values():
                base_name = entity.base_name  # Propriété qui supprime _part_X
                if base_name != entity.entity_name:  # C'est une partie
                    if base_name not in base_names:
                        base_names[base_name] = []
                    base_names[base_name].append(entity)

            # Marquer les entités groupées
            for base_name, parts in base_names.items():
                if len(parts) > 1:
                    for entity in parts:
                        entity.is_grouped = True
                        entity.is_complete = True  # Assume complete for now
                    grouped_count += len(parts)

                    logger.debug(f"📦 Entité groupée: {base_name} ({len(parts)} parties)")

            logger.info(f"✅ {grouped_count} entités groupées détectées")

        except Exception as e:
            logger.error(f"❌ Erreur détection groupement: {e}")

    # === Méthodes utilitaires pour l'intégration ===

    async def get_entity_context(self, entity_name: str, context_type: str = "local") -> Dict[str, Any]:
        """
        Récupère le contexte d'une entité (utilisé par le RAG).
        Interface principale pour les requêtes du système RAG.
        """
        if not self._initialized or not self.orchestrator:
            return {"error": "FortranDocumentProcessor non initialisé"}

        if context_type == "smart":
            return await self.orchestrator.get_context_for_agent(
                entity_name, "developer", "code_understanding"
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
        """Recherche d'entités (interface pour le RAG)"""
        if not self.orchestrator:
            return []
        return await self.orchestrator.search_entities(query)

    async def sync_orchestrator_with_entities(self):
        """Synchronise l'orchestrateur avec les entités de l'EntityManager"""
        try:
            if not self.orchestrator or not self.entity_manager:
                return

            logger.info("🔄 Synchronisation de l'orchestrateur avec les nouvelles entités...")

            # Réinitialiser l'orchestrateur avec les nouvelles entités
            await self.orchestrator.initialize()

            logger.info("✅ Synchronisation terminée")

        except Exception as e:
            logger.error(f"Erreur synchronisation orchestrateur: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Statistiques du module"""
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
_global_fortran_processor = FortranDocumentProcessor()


async def get_fortran_processor(document_store=None, rag_engine=None,
                                ontology_manager=None) -> FortranDocumentProcessor:
    """Factory pour obtenir le processeur Fortran global"""
    global _global_fortran_processor

    if document_store and not _global_fortran_processor._initialized:
        _global_fortran_processor.ontology_manager = ontology_manager
        await _global_fortran_processor.initialize_with_document_store(document_store, rag_engine)

    return _global_fortran_processor