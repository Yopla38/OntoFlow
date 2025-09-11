"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# embedding_manager.py
import asyncio
import json
import os
import pickle
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from ..provider.llm_providers import LLMProvider


class EmbeddingManager:
    """Gestionnaire d'embeddings pour les documents"""

    def __init__(self, embedding_provider: LLMProvider, storage_dir: str = "embeddings"):
        """
        Initialise le gestionnaire d'embeddings

        Args:
            embedding_provider: Provider LLM pour générer les embeddings
            storage_dir: Répertoire pour stocker les embeddings persistants
        """
        self.provider = embedding_provider
        self.storage_dir = storage_dir
        self.embeddings: Dict[str, List[float]] = {}

        # Créer le répertoire s'il n'existe pas
        os.makedirs(storage_dir, exist_ok=True)

    async def create_embeddings(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """
        Crée des embeddings pour une liste de chunks en évitant les recalculs inutiles

        Args:
            chunks: Liste de chunks à transformer en embeddings

        Returns:
            Dictionnaire d'embeddings (chunk_id -> embedding)
        """
        # Filtrer les chunks dont les embeddings n'existent pas déjà
        chunks_to_process = []
        chunk_ids_to_process = []

        for chunk in chunks:
            chunk_id = chunk["id"]
            if chunk_id not in self.embeddings:
                chunks_to_process.append(chunk)
                chunk_ids_to_process.append(chunk_id)

        # Si tous les embeddings existent déjà
        if not chunks_to_process:
            print(f"✓ Tous les embeddings pour ces {len(chunks)} chunks existent déjà.")
            return {chunk["id"]: self.embeddings[chunk["id"]]
                    for chunk in chunks if chunk["id"] in self.embeddings}

        print(f"Génération de {len(chunks_to_process)}/{len(chunks)} nouveaux embeddings...")

        # Extraire le texte des chunks à traiter
        texts = [chunk["text"] for chunk in chunks_to_process]

        # Générer les embeddings
        new_embeddings = await self.provider.generate_embeddings(texts)

        # Associer les embeddings avec les IDs de chunks
        embeddings_dict = {chunk_id: embedding
                           for chunk_id, embedding in zip(chunk_ids_to_process, new_embeddings)}

        # Mettre à jour le dictionnaire d'embeddings
        self.embeddings.update(embeddings_dict)

        return embeddings_dict

    async def save_embeddings(self, document_id: str) -> str:
        """
        Sauvegarde les embeddings d'un document sur disque

        Args:
            document_id: ID du document

        Returns:
            Chemin du fichier de sauvegarde
        """
        # Filtrer les embeddings pour ce document
        document_embeddings = {
            chunk_id: embedding
            for chunk_id, embedding in self.embeddings.items()
            if chunk_id.startswith(f"{document_id}-chunk-")
        }

        # Créer le nom de fichier
        embedding_path = os.path.join(self.storage_dir, f"{document_id}.pkl")

        # Sauvegarder de façon asynchrone
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._save_embeddings_sync(embedding_path, document_embeddings)
        )

        return embedding_path

    def _save_embeddings_sync(self, path: str, embeddings: Dict[str, List[float]]) -> None:
        """Sauvegarde synchrone des embeddings"""
        with open(path, 'wb') as f:
            pickle.dump(embeddings, f)

    async def load_embeddings(self, document_id: str) -> bool:
        """
        Charge les embeddings d'un document depuis le disque

        Args:
            document_id: ID du document

        Returns:
            True si le chargement a réussi, False sinon
        """
        embedding_path = os.path.join(self.storage_dir, f"{document_id}.pkl")

        if not os.path.exists(embedding_path):
            print(f"Fichier d'embeddings non trouvé: {embedding_path}")
            return False

        try:
            # Charger de façon asynchrone
            loop = asyncio.get_event_loop()
            loaded_embeddings = await loop.run_in_executor(
                None,
                lambda: self._load_embeddings_sync(embedding_path)
            )

            if loaded_embeddings:
                # Compter combien d'embeddings sont chargés pour ce document
                #count = sum(1 for k in loaded_embeddings if k.startswith(f"{document_id}-chunk-"))
                #print(f"Chargement de {count} embeddings pour le document {document_id}.")

                # Mettre à jour le dictionnaire d'embeddings
                self.embeddings.update(loaded_embeddings)
                return True

            return False
        except Exception as e:
            print(f"Erreur lors du chargement des embeddings: {str(e)}")
            return False

    def _load_embeddings_sync(self, path: str) -> Optional[Dict[str, List[float]]]:
        """Chargement synchrone des embeddings"""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Erreur lors du chargement des embeddings: {str(e)}")
            return None

    def get_embedding(self, chunk_id: str) -> Optional[List[float]]:
        """Récupère l'embedding d'un chunk par son ID"""
        return self.embeddings.get(chunk_id)

    def get_all_embeddings(self) -> Dict[str, List[float]]:
        """Récupère tous les embeddings"""
        return self.embeddings

    def get_document_embeddings(self, document_id: str) -> Dict[str, List[float]]:
        """Récupère tous les embeddings d'un document spécifique"""
        return {
            chunk_id: embedding
            for chunk_id, embedding in self.embeddings.items()
            if chunk_id.startswith(f"{document_id}-chunk-")
        }


class EmbeddingManagerWithBatch:
    """Gestionnaire d'embeddings pour les documents avec support de Batch API"""

    def __init__(
            self,
            embedding_provider: LLMProvider,
            storage_dir: str = "embeddings",
            use_batch: bool = False,
            batch_dir: str = "batch_files",
            batch_min_size: int = 20,  # Taille minimale pour utiliser le batch
            embedding_model: str = "text-embedding-3-large"
    ):
        """
        Initialise le gestionnaire d'embeddings

        Args:
            embedding_provider: Provider LLM pour générer les embeddings
            storage_dir: Répertoire pour stocker les embeddings persistants
            use_batch: Utiliser l'API Batch d'OpenAI pour les embeddings (plus économique)
            batch_dir: Répertoire pour stocker les fichiers de batch
            batch_min_size: Nombre minimum de requêtes pour utiliser l'API Batch
            embedding_model: Modèle d'embedding à utiliser
        """
        self.provider = embedding_provider
        self.storage_dir = storage_dir
        self.use_batch = use_batch
        self.batch_dir = batch_dir
        self.batch_min_size = batch_min_size
        self.embedding_model = embedding_model
        self.embeddings: Dict[str, List[float]] = {}
        self.active_batches: Dict[str, Dict[str, Any]] = {}  # ID du batch -> infos

        # Créer les répertoires s'ils n'existent pas
        os.makedirs(storage_dir, exist_ok=True)
        if use_batch:
            os.makedirs(batch_dir, exist_ok=True)
            self._load_active_batches()

    def _load_active_batches(self):
        """Charge les batchs actifs depuis le stockage persistant"""
        batch_status_path = os.path.join(self.batch_dir, "batch_status.json")
        if os.path.exists(batch_status_path):
            try:
                with open(batch_status_path, 'r') as f:
                    self.active_batches = json.load(f)
                print(f"✓ Chargement de {len(self.active_batches)} batchs actifs")
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement des batchs actifs: {str(e)}")
                self.active_batches = {}

    def _save_active_batches(self):
        """Sauvegarde les batchs actifs dans un stockage persistant"""
        batch_status_path = os.path.join(self.batch_dir, "batch_status.json")
        try:
            with open(batch_status_path, 'w') as f:
                json.dump(self.active_batches, f, indent=2)
        except Exception as e:
            print(f"⚠️ Erreur lors de la sauvegarde des batchs actifs: {str(e)}")

    async def create_embeddings(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """
        Crée des embeddings pour une liste de chunks en évitant les recalculs inutiles

        Args:
            chunks: Liste de chunks à transformer en embeddings

        Returns:
            Dictionnaire d'embeddings (chunk_id -> embedding)
        """
        # Filtrer les chunks dont les embeddings n'existent pas déjà
        chunks_to_process = []
        chunk_ids_to_process = []

        for chunk in chunks:
            chunk_id = chunk["id"]
            if chunk_id not in self.embeddings:
                chunks_to_process.append(chunk)
                chunk_ids_to_process.append(chunk_id)

        # Si tous les embeddings existent déjà
        if not chunks_to_process:
            print(f"✓ Tous les embeddings pour ces {len(chunks)} chunks existent déjà.")
            return {chunk["id"]: self.embeddings[chunk["id"]]
                    for chunk in chunks if chunk["id"] in self.embeddings}

        print(f"Génération de {len(chunks_to_process)}/{len(chunks)} nouveaux embeddings...")

        # Extraire le texte des chunks à traiter
        texts = [chunk["text"] for chunk in chunks_to_process]

        # Vérifier s'il y a des batchs actifs à traiter d'abord
        if self.use_batch and self.active_batches:
            await self._check_and_process_active_batches()

        # Selon le mode, utiliser l'API batch ou l'API standard
        if self.use_batch and len(chunks_to_process) >= self.batch_min_size:
            # Traiter avec l'API Batch
            batch_results = await self._process_with_batch(texts, chunk_ids_to_process)

            # Mettre à jour le dictionnaire d'embeddings
            self.embeddings.update(batch_results)
            return {chunk["id"]: self.embeddings.get(chunk["id"])
                    for chunk in chunks if chunk["id"] in self.embeddings}
        else:
            # Générer les embeddings via l'API standard
            new_embeddings = await self.provider.generate_embeddings(texts)

            # Associer les embeddings avec les IDs de chunks
            embeddings_dict = {chunk_id: embedding
                               for chunk_id, embedding in zip(chunk_ids_to_process, new_embeddings)}

            # Mettre à jour le dictionnaire d'embeddings
            self.embeddings.update(embeddings_dict)

            return {chunk["id"]: self.embeddings.get(chunk["id"])
                    for chunk in chunks if chunk["id"] in self.embeddings}

    async def _check_and_process_active_batches(self):
        """Vérifie et traite les batchs actifs s'il y en a"""
        if not self.active_batches:
            return

        print(f"Vérification de {len(self.active_batches)} batchs actifs...")

        # Liste des batchs terminés à retirer
        completed_batches = []

        for batch_id, batch_info in self.active_batches.items():
            if batch_info.get("status") in ["completed", "failed", "expired", "cancelled"]:
                # Batch déjà terminé, on le retire
                completed_batches.append(batch_id)
                continue

            # Vérifier le statut actuel
            client = self.provider.client
            try:
                batch = await client.batches.retrieve(batch_id)
                current_status = batch.status

                # Mise à jour du statut
                batch_info["status"] = current_status

                if current_status == "completed":
                    # Récupérer les résultats
                    output_file_id = batch.output_file_id
                    if output_file_id:
                        try:
                            # Télécharger les résultats
                            response = await client.files.content(output_file_id)
                            content = response.text

                            # Traiter les résultats
                            for line in content.strip().split('\n'):
                                result = json.loads(line)
                                custom_id = result.get('custom_id')

                                if custom_id and result.get('response', {}).get('status_code') == 200:
                                    # Extraire l'embedding
                                    embedding_data = result['response']['body']['data'][0]['embedding']
                                    self.embeddings[custom_id] = embedding_data

                            print(
                                f"✓ Batch {batch_id} complété, {len(batch_info.get('chunk_ids', []))} embeddings récupérés")
                            completed_batches.append(batch_id)
                        except Exception as e:
                            print(f"❌ Erreur lors de la récupération des résultats du batch {batch_id}: {str(e)}")
                            batch_info["error"] = str(e)
                            completed_batches.append(batch_id)

                elif current_status in ["failed", "expired", "cancelled"]:
                    print(f"❌ Batch {batch_id} terminé avec statut {current_status}")
                    completed_batches.append(batch_id)

                else:
                    print(f"Batch {batch_id}: {current_status} - "
                          f"{batch.request_counts.completed}/{batch.request_counts.total} complétés")

            except Exception as e:
                print(f"⚠️ Erreur lors de la vérification du batch {batch_id}: {str(e)}")
                batch_info["error"] = str(e)

        # Supprimer les batchs terminés
        for batch_id in completed_batches:
            if batch_id in self.active_batches:
                del self.active_batches[batch_id]

        # Sauvegarder les modifications
        self._save_active_batches()

    async def _process_with_batch(self, texts: List[str], chunk_ids: List[str]) -> Dict[str, List[float]]:
        """
        Traite une liste de textes avec l'API Batch d'OpenAI

        Args:
            texts: Liste des textes à traiter
            chunk_ids: Liste des IDs de chunks correspondants

        Returns:
            Dictionnaire des embeddings (chunk_id -> embedding)
        """
        # Limites de l'API Batch
        MAX_REQUESTS_PER_BATCH = 50000
        MAX_BATCH_FILE_SIZE_MB = 200

        # Diviser en plusieurs batchs si nécessaire
        batches_data = []
        current_batch = []
        current_size_bytes = 0
        current_ids = []
        current_texts = []

        for i, (text, chunk_id) in enumerate(zip(texts, chunk_ids)):
            # Créer une requête d'embedding
            request = {
                "custom_id": chunk_id,
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": self.embedding_model,
                    "input": text
                }
            }

            # Estimer la taille
            request_size = len(json.dumps(request).encode('utf-8'))

            # Si ajouter cette requête dépasse les limites, créer un nouveau batch
            if (len(current_batch) + 1 > MAX_REQUESTS_PER_BATCH or
                    (current_size_bytes + request_size) / (1024 * 1024) > MAX_BATCH_FILE_SIZE_MB):
                batches_data.append((current_batch, current_ids, current_texts))
                current_batch = [request]
                current_size_bytes = request_size
                current_ids = [chunk_id]
                current_texts = [text]
            else:
                current_batch.append(request)
                current_size_bytes += request_size
                current_ids.append(chunk_id)
                current_texts.append(text)

        # Ajouter le dernier batch s'il n'est pas vide
        if current_batch:
            batches_data.append((current_batch, current_ids, current_texts))

        # Créer et soumettre chaque batch
        final_results = {}
        batch_tasks = []

        for batch_index, (batch_requests, batch_chunk_ids, batch_texts) in enumerate(batches_data):
            batch_task = asyncio.create_task(
                self._submit_batch(batch_requests, batch_chunk_ids, batch_texts, batch_index)
            )
            batch_tasks.append(batch_task)

        # Attendre tous les résultats
        all_batch_results = await asyncio.gather(*batch_tasks)

        # Combiner les résultats
        for batch_result in all_batch_results:
            final_results.update(batch_result)

        return final_results

    async def _submit_batch(
            self,
            batch_requests: List[Dict[str, Any]],
            batch_chunk_ids: List[str],
            batch_texts: List[str],
            batch_index: int
    ) -> Dict[str, List[float]]:
        """
        Soumet un batch à l'API et attend sa complétion ou le place en attente

        Args:
            batch_requests: Liste des requêtes à soumettre
            batch_chunk_ids: Liste des IDs de chunks correspondants
            batch_texts: Textes originaux (pour retry si nécessaire)
            batch_index: Index du batch

        Returns:
            Dictionnaire des embeddings disponibles immédiatement
        """
        # Créer un ID unique pour ce batch
        batch_id = f"emb_batch_{uuid.uuid4().hex[:8]}"

        # Préparer le fichier JSONL
        batch_file_path = os.path.join(self.batch_dir, f"{batch_id}.jsonl")
        with open(batch_file_path, 'w', encoding='utf-8') as f:
            for request in batch_requests:
                f.write(json.dumps(request) + '\n')

        # Uploader le fichier
        try:
            client = self.provider.client

            # Upload du fichier
            with open(batch_file_path, 'rb') as f:
                file_obj = await client.files.create(
                    file=f,
                    purpose="batch"
                )

            file_id = file_obj.id
            print(f"✓ Fichier batch {batch_index} uploadé: {file_id}")

            # Création du batch
            batch = await client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/embeddings",
                completion_window="24h",
                metadata={
                    "description": f"Embeddings batch {batch_index}"
                }
            )

            batch_id = batch.id
            print(f"✓ Batch {batch_index} créé: {batch_id}")

            # Suivre ce batch
            self.active_batches[batch_id] = {
                "status": batch.status,
                "created_at": batch.created_at,
                "chunk_ids": batch_chunk_ids,
                "file_id": file_id,
                "batch_index": batch_index,
                "texts": batch_texts  # Stocker les textes pour retry
            }

            # Sauvegarder l'état des batchs
            self._save_active_batches()

            # Attendre max 5 minutes pour une complétion rapide
            MAX_WAIT = 300  # 5 minutes
            start_time = time.time()

            while time.time() - start_time < MAX_WAIT:
                # Vérifier toutes les 10 secondes
                await asyncio.sleep(10)

                batch = await client.batches.retrieve(batch_id)
                if batch.status == "completed":
                    # Récupérer les résultats
                    output_file_id = batch.output_file_id

                    if output_file_id:
                        response = await client.files.content(output_file_id)
                        content = response.text

                        results = {}
                        for line in content.strip().split('\n'):
                            result = json.loads(line)
                            custom_id = result.get('custom_id')

                            if custom_id and result.get('response', {}).get('status_code') == 200:
                                embedding_data = result['response']['body']['data'][0]['embedding']
                                results[custom_id] = embedding_data

                        print(f"✓ Batch {batch_id} complété rapidement ({int(time.time() - start_time)}s)")

                        # Nettoyer
                        del self.active_batches[batch_id]
                        self._save_active_batches()

                        return results
                elif batch.status in ["failed", "expired", "cancelled"]:
                    print(f"❌ Batch {batch_id} a échoué avec statut: {batch.status}")
                    # Fallback vers l'API standard
                    embeddings = await self.provider.generate_embeddings(batch_texts)
                    return {chunk_id: embedding for chunk_id, embedding in zip(batch_chunk_ids, embeddings)}

            # Si on arrive ici, le batch est toujours en cours après le temps d'attente
            print(f"⏳ Batch {batch_id} toujours en cours après {MAX_WAIT}s, passage en mode asynchrone")
            return {}  # Retourner un dict vide, les embeddings seront récupérés plus tard

        except Exception as e:
            print(f"❌ Erreur lors de la création du batch {batch_index}: {str(e)}")
            # Fallback vers l'API standard
            try:
                embeddings = await self.provider.generate_embeddings(batch_texts)
                return {chunk_id: embedding for chunk_id, embedding in zip(batch_chunk_ids, embeddings)}
            except Exception as e2:
                print(f"❌ Échec du fallback: {str(e2)}")
                return {}

    async def retrieve_all_batch_results(self) -> int:
        """
        Récupère tous les résultats des batchs en attente

        Returns:
            Nombre d'embeddings récupérés
        """
        if not self.active_batches:
            return 0

        await self._check_and_process_active_batches()

        # Compter combien d'embeddings ont été ajoutés
        count = sum(1 for batch in self.active_batches.values()
                    for chunk_id in batch.get("chunk_ids", [])
                    if chunk_id in self.embeddings)

        return count

    async def cancel_batch(self, batch_id: str) -> bool:
        """
        Annule un batch en cours

        Args:
            batch_id: ID du batch à annuler

        Returns:
            True si l'annulation a réussi, False sinon
        """
        if batch_id not in self.active_batches:
            print(f"⚠️ Batch {batch_id} non trouvé")
            return False

        try:
            client = self.provider.client
            await client.batches.cancel(batch_id)

            # Mettre à jour le statut
            self.active_batches[batch_id]["status"] = "cancelling"
            self._save_active_batches()

            print(f"✓ Demande d'annulation envoyée pour le batch {batch_id}")
            return True
        except Exception as e:
            print(f"❌ Erreur lors de l'annulation du batch {batch_id}: {str(e)}")
            return False

    async def cancel_all_batches(self) -> Dict[str, Any]:
        """
        Annule tous les batchs actifs

        Returns:
            Résultats des annulations
        """
        results = {"cancelled": 0, "failed": 0}

        for batch_id in list(self.active_batches.keys()):
            success = await self.cancel_batch(batch_id)
            if success:
                results["cancelled"] += 1
            else:
                results["failed"] += 1

        return results

    async def get_batch_status(self) -> Dict[str, Any]:
        """
        Obtient le statut de tous les batchs actifs

        Returns:
            Informations sur les batchs actifs
        """
        if not self.use_batch:
            return {"enabled": False}

        stats = {
            "enabled": True,
            "active_batches": len(self.active_batches),
            "batches": {}
        }

        for batch_id, info in self.active_batches.items():
            stats["batches"][batch_id] = {
                "status": info.get("status", "unknown"),
                "created_at": info.get("created_at"),
                "count": len(info.get("chunk_ids", [])),
                "batch_index": info.get("batch_index")
            }

        return stats

    async def save_embeddings(self, document_id: str) -> str:
        """
        Sauvegarde les embeddings d'un document sur disque

        Args:
            document_id: ID du document

        Returns:
            Chemin du fichier de sauvegarde
        """
        # Vérifier les batchs complétés pour avoir les derniers embeddings
        if self.use_batch and self.active_batches:
            await self._check_and_process_active_batches()

        # Filtrer les embeddings pour ce document
        document_embeddings = {
            chunk_id: embedding
            for chunk_id, embedding in self.embeddings.items()
            if chunk_id.startswith(f"{document_id}-chunk-")
        }

        # Créer le nom de fichier
        embedding_path = os.path.join(self.storage_dir, f"{document_id}.pkl")

        # Sauvegarder de façon asynchrone
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._save_embeddings_sync(embedding_path, document_embeddings)
        )

        return embedding_path

    def _save_embeddings_sync(self, path: str, embeddings: Dict[str, List[float]]) -> None:
        """Sauvegarde synchrone des embeddings"""
        with open(path, 'wb') as f:
            pickle.dump(embeddings, f)

    async def load_embeddings(self, document_id: str) -> bool:
        """
        Charge les embeddings d'un document depuis le disque

        Args:
            document_id: ID du document

        Returns:
            True si le chargement a réussi, False sinon
        """
        embedding_path = os.path.join(self.storage_dir, f"{document_id}.pkl")

        if not os.path.exists(embedding_path):
            print(f"Fichier d'embeddings non trouvé: {embedding_path}")
            return False

        try:
            # Charger de façon asynchrone
            loop = asyncio.get_event_loop()
            loaded_embeddings = await loop.run_in_executor(
                None,
                lambda: self._load_embeddings_sync(embedding_path)
            )

            if loaded_embeddings:
                # Mettre à jour le dictionnaire d'embeddings
                self.embeddings.update(loaded_embeddings)
                return True

            return False
        except Exception as e:
            print(f"Erreur lors du chargement des embeddings: {str(e)}")
            return False

    def _load_embeddings_sync(self, path: str) -> Optional[Dict[str, List[float]]]:
        """Chargement synchrone des embeddings"""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Erreur lors du chargement des embeddings: {str(e)}")
            return None

    def get_embedding(self, chunk_id: str) -> Optional[List[float]]:
        """Récupère l'embedding d'un chunk par son ID"""
        return self.embeddings.get(chunk_id)

    def get_all_embeddings(self) -> Dict[str, List[float]]:
        """Récupère tous les embeddings"""
        return self.embeddings

    def get_document_embeddings(self, document_id: str) -> Dict[str, List[float]]:
        """Récupère tous les embeddings d'un document spécifique"""
        return {
            chunk_id: embedding
            for chunk_id, embedding in self.embeddings.items()
            if chunk_id.startswith(f"{document_id}-chunk-")
        }