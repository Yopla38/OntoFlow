"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# document_store.py
import asyncio
import json
import os
import shutil
from typing import List, Dict, Any, Optional
import pickle


class DocumentStore:
    """Stockage des documents et gestion de leurs métadonnées"""

    def __init__(
            self,
            document_processor,
            embedding_manager,
            storage_dir: str = "storage"
    ):
        """
        Initialise le stockage de documents

        Args:
            document_processor: Processeur pour traiter les documents
            embedding_manager: Gestionnaire d'embeddings
            storage_dir: Répertoire pour stocker les documents et métadonnées
        """
        self.processor = document_processor
        self.embedding_manager = embedding_manager
        self.storage_dir = storage_dir
        self.documents_dir = os.path.join(storage_dir, "documents")
        self.metadata_path = os.path.join(storage_dir, "metadata.json")

        # Dictionnaire pour stocker les métadonnées des documents
        self.documents: Dict[str, Dict[str, Any]] = {}

        # Dictionnaire pour stocker les chunks de documents
        self.document_chunks: Dict[str, List[Dict[str, Any]]] = {}

        # Créer les répertoires nécessaires
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.documents_dir, exist_ok=True)

        # Charger les métadonnées existantes
        self._load_metadata()

    async def add_document_with_id(self,
                                   filepath: str,
                                   document_id: str,
                                   additional_metadata: Optional[Dict[str, Any]] = None
                                   ) -> str:
        """
        Ajoute un document avec un ID spécifique en assurant la cohérence avec metadata.json
        """
        # Vérifier si le document existe déjà dans metadata.json
        if document_id in self.documents:
            doc_info = self.documents[document_id]
            doc_path = doc_info.get("path", "")

            if os.path.exists(doc_path):
                print(f"Document {document_id} déjà présent avec fichier existant.")

                # Vérifier si chunks et embeddings existent
                chunks_path = os.path.join(self.storage_dir, "chunks", f"{document_id}.pkl")
                embedding_path = os.path.join(self.embedding_manager.storage_dir, f"{document_id}.pkl")

                # Si tout existe, pas besoin de réindexer
                if os.path.exists(chunks_path) and os.path.exists(embedding_path):
                    return document_id

                # Sinon, continuer pour recréer les fichiers manquants
                print(f"Fichiers associés manquants pour document {document_id}, réindexation...")

        # Traiter le document en utilisant l'ID fourni
        _, chunks = await self.processor.process_document(filepath, document_id, additional_metadata)

        # Copier le document dans le répertoire avec nom standardisé
        filename = os.path.basename(filepath)
        document_path = os.path.join(self.documents_dir, f"{document_id}_beir_{document_id}.txt")

        # Copier de façon asynchrone
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: shutil.copy2(filepath, document_path))

        # Stocker les métadonnées au format standard observé
        self.documents[document_id] = {
            "id": document_id,
            "path": document_path,
            "original_path": filepath,
            "original_filename": filename,
            "chunks_count": len(chunks),
            "beir_id": document_id,  # Stocker l'ID BEIR explicitement
            "additional_metadata": additional_metadata
        }

        # Stocker les chunks
        self.document_chunks[document_id] = chunks

        # Sauvegarder les chunks
        chunks_dir = os.path.join(self.storage_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        chunks_path = os.path.join(chunks_dir, f"{document_id}.pkl")
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._save_chunks_sync(chunks_path, chunks)
        )

        # Créer ou charger les embeddings
        embedding_path = os.path.join(self.embedding_manager.storage_dir, f"{document_id}.pkl")
        if os.path.exists(embedding_path):
            print(f"Embeddings existants trouvés pour {document_id}, chargement...")
            await self.embedding_manager.load_embeddings(document_id)
        else:
            # Créer les embeddings
            print(f"Création des embeddings pour {document_id}...")
            await self.embedding_manager.create_embeddings(chunks)
            # Sauvegarder les embeddings
            await self.embedding_manager.save_embeddings(document_id)

        # Sauvegarder les métadonnées mises à jour
        await self._save_metadata()

        return document_id

    def _load_metadata(self) -> None:
        """Charge les métadonnées des documents depuis le fichier et vérifie leur cohérence"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)

                # Vérifier la cohérence des entrées dans metadata.json
                inconsistent_docs = []
                for doc_id, doc_info in self.documents.items():
                    # Vérifier si le fichier du document existe
                    doc_path = doc_info.get("path", "")
                    if not os.path.exists(doc_path):
                        print(f"⚠️ Fichier manquant pour document {doc_id}: {doc_path}")
                        inconsistent_docs.append(doc_id)

                # Option: supprimer les entrées incohérentes
                for doc_id in inconsistent_docs:
                    del self.documents[doc_id]

                print(f"Métadonnées chargées: {len(self.documents)} documents")
                if inconsistent_docs:
                    print(f"Attention: {len(inconsistent_docs)} documents ont des incohérences")

            except Exception as e:
                print(f"Erreur lors du chargement des métadonnées: {str(e)}")
                self.documents = {}
        else:
            print(f"Fichier de métadonnées non trouvé: {self.metadata_path}")
            self.documents = {}

    async def old_save_metadata(self):
        """Sauvegarde les métadonnées des documents"""
        try:
            metadata = {
                "documents": self.documents,
                "document_chunks": {}
            }

            # CORRECTION : Créer une copie de la liste des clés pour éviter l'erreur
            document_ids = list(self.document_chunks.keys())
            for document_id in document_ids:
                if document_id in self.document_chunks:  # Vérifier que la clé existe encore
                    metadata["document_chunks"][document_id] = len(self.document_chunks[document_id])

            metadata_path = self.storage_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            print(f"Erreur lors de la sauvegarde des métadonnées: {e}")

    async def _save_metadata(self) -> None:
        """Sauvegarde les métadonnées des documents dans le fichier"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._save_metadata_sync()
        )

        # Sauvegarder également les chunks pour chaque document
        for document_id in self.document_chunks:
            await self.save_chunks(document_id)

    def _save_metadata_sync(self) -> None:
        """Sauvegarde synchrone des métadonnées"""
        try:
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des métadonnées: {str(e)}")

    async def save_chunks(self, document_id: str) -> str:
        """
        Sauvegarde les chunks d'un document sur disque

        Args:
            document_id: ID du document

        Returns:
            Chemin du fichier de sauvegarde
        """
        if document_id not in self.document_chunks:
            return ""

        chunks = self.document_chunks[document_id]
        chunks_dir = os.path.join(self.storage_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        chunks_path = os.path.join(chunks_dir, f"{document_id}.pkl")

        # Sauvegarder de façon asynchrone
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._save_chunks_sync(chunks_path, chunks)
        )

        return chunks_path

    def _save_chunks_sync(self, path: str, chunks: List[Dict[str, Any]]) -> None:
        """Sauvegarde synchrone des chunks"""
        with open(path, 'wb') as f:
            pickle.dump(chunks, f)

    async def load_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Charge les chunks d'un document depuis le disque

        Args:
            document_id: ID du document

        Returns:
            Liste des chunks ou liste vide si échec
        """
        chunks_path = os.path.join(self.storage_dir, "chunks", f"{document_id}.pkl")

        if not os.path.exists(chunks_path):
            print(f"Fichier de chunks non trouvé: {chunks_path}")
            return []

        # Charger de façon asynchrone
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            None,
            lambda: self._load_chunks_sync(chunks_path)
        )

        return chunks or []

    def _load_chunks_sync(self, path: str) -> Optional[List[Dict[str, Any]]]:
        """Chargement synchrone des chunks"""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Erreur lors du chargement des chunks: {str(e)}")
            return None

    async def add_document(self, filepath: str) -> str:
        """
        Ajoute un document au stockage

        Args:
            filepath: Chemin vers le document

        Returns:
            ID du document ajouté
        """
        # Traiter le document
        document_id, chunks = await self.processor.process_document(filepath)

        # Copier le document dans le répertoire de stockage
        filename = os.path.basename(filepath)
        document_path = os.path.join(self.documents_dir, f"{document_id}_{filename}")

        # Copier de façon asynchrone
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: shutil.copy2(filepath, document_path))

        # Stocker les métadonnées
        self.documents[document_id] = {
            "id": document_id,
            "path": document_path,
            "original_path": filepath,
            "original_filename": filename,
            "chunks_count": len(chunks)
        }

        # Stocker les chunks
        self.document_chunks[document_id] = chunks

        # Créer les embeddings
        await self.embedding_manager.create_embeddings(chunks)

        # Sauvegarder les embeddings
        await self.embedding_manager.save_embeddings(document_id)

        # Sauvegarder les métadonnées
        await self._save_metadata()

        return document_id

    async def add_documents(self, filepaths: List[str]) -> List[str]:
        """
        Ajoute plusieurs documents au stockage

        Args:
            filepaths: Liste des chemins vers les documents

        Returns:
            Liste des IDs des documents ajoutés
        """
        document_ids = []
        for filepath in filepaths:
            document_id = await self.add_document(filepath)
            document_ids.append(document_id)
        return document_ids

    async def remove_document(self, document_id: str) -> bool:
        """
        Supprime un document du stockage

        Args:
            document_id: ID du document à supprimer

        Returns:
            True si la suppression a réussi, False sinon
        """
        if document_id not in self.documents:
            return False

        # Récupérer le chemin du document
        document_path = self.documents[document_id]["path"]

        # Supprimer le fichier
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: os.remove(document_path) if os.path.exists(document_path) else None)

        # Supprimer les embeddings
        embedding_path = os.path.join(self.embedding_manager.storage_dir, f"{document_id}.pkl")
        await loop.run_in_executor(None, lambda: os.remove(embedding_path) if os.path.exists(embedding_path) else None)

        # Supprimer les métadonnées et chunks
        del self.documents[document_id]
        if document_id in self.document_chunks:
            del self.document_chunks[document_id]

        # Sauvegarder les métadonnées
        await self._save_metadata()

        return True

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Récupère les métadonnées d'un document par son ID"""
        return self.documents.get(document_id)

    async def get_all_documents(self) -> Dict[str, Dict[str, Any]]:
        """Récupère tous les documents"""
        return self.documents

    async def get_document_chunks(self, document_id: str) -> Optional[List[Dict[str, Any]]]:
        """Récupère les chunks d'un document par son ID"""
        return self.document_chunks.get(document_id)

    async def old_load_document_chunks(self, document_id: str) -> bool:
        """
        Charge les chunks d'un document depuis son fichier

        Args:
            document_id: ID du document

        Returns:
            True si le chargement a réussi, False sinon
        """
        if document_id not in self.documents:
            # print(f"Document {document_id} non trouvé dans les métadonnées.")
            return False

        # Vérifier si les chunks sont déjà chargés
        if document_id in self.document_chunks and self.document_chunks[document_id]:
            # print(f"Chunks du document {document_id} déjà en mémoire.")

            # Vérifier quand même si les embeddings sont chargés
            embedding_loaded = await self.embedding_manager.load_embeddings(document_id)

            return True

        # Essayer de charger les chunks depuis le fichier
        saved_chunks = await self.load_chunks(document_id)
        if saved_chunks:
            print(f"Chargement de {len(saved_chunks)} chunks sauvegardés pour le document {document_id}.")
            self.document_chunks[document_id] = saved_chunks

            # Charger les embeddings
            embedding_loaded = await self.embedding_manager.load_embeddings(document_id)
            if not embedding_loaded:
                print(f"Embeddings non trouvés. Création des embeddings...")
                await self.embedding_manager.create_embeddings(saved_chunks)
                await self.embedding_manager.save_embeddings(document_id)

            return True

        # Si aucun chunk sauvegardé, recréer les chunks

    async def load_document_chunks(self, document_id: str) -> bool:
        """
        Charge les chunks d'un document depuis son fichier avec vérification et réparation
        """
        if document_id not in self.documents:
            print(f"Document {document_id} non trouvé dans metadata.json")
            return False

        # Déjà en mémoire? Vérifier les embeddings aussi
        if document_id in self.document_chunks and self.document_chunks[document_id]:
            #print(f"Chunks du document {document_id} déjà en mémoire.")

            # Vérifier si les embeddings sont chargés
            embedding_path = os.path.join(self.embedding_manager.storage_dir, f"{document_id}.pkl")
            if os.path.exists(embedding_path):
                if not any(chunk_id.startswith(f"{document_id}-chunk-")
                           for chunk_id in self.embedding_manager.embeddings):
                    print(f"Chargement des embeddings pour {document_id}...")
                    await self.embedding_manager.load_embeddings(document_id)
            else:
                print(f"⚠️ Embeddings manquants pour {document_id}, création...")
                await self.embedding_manager.create_embeddings(self.document_chunks[document_id])
                await self.embedding_manager.save_embeddings(document_id)

            return True

        # Chercher les chunks dans le fichier
        chunks_path = os.path.join(self.storage_dir, "chunks", f"{document_id}.pkl")
        if os.path.exists(chunks_path):
            saved_chunks = await self.load_chunks(document_id)
            if saved_chunks:
                print(f"Chargement de {len(saved_chunks)} chunks pour document {document_id}.")
                self.document_chunks[document_id] = saved_chunks

                # Vérifier et charger/créer les embeddings
                embedding_path = os.path.join(self.embedding_manager.storage_dir, f"{document_id}.pkl")
                if os.path.exists(embedding_path):
                    await self.embedding_manager.load_embeddings(document_id)
                else:
                    print(f"Embeddings non trouvés pour {document_id}, création...")
                    await self.embedding_manager.create_embeddings(saved_chunks)
                    await self.embedding_manager.save_embeddings(document_id)

                return True

        # Recréer les chunks si nécessaire
        print(f"Chunks non trouvés pour {document_id}, recréation...")

        document_path = self.documents[document_id]["path"]

        if not os.path.exists(document_path):
            print(f"Fichier du document {document_id} non trouvé: {document_path}")
            return False

        print(f"Recréation des chunks pour le document {document_id}...")
        try:
            _, chunks = await self.processor.process_document(document_path)

            # Stocker les chunks
            self.document_chunks[document_id] = chunks
            print(f"{len(chunks)} chunks créés pour le document {document_id}.")

            # Sauvegarder les chunks
            await self.save_chunks(document_id)

            # Charger ou créer les embeddings
            embedding_loaded = await self.embedding_manager.load_embeddings(document_id)
            if not embedding_loaded:
                print(f"Création des embeddings pour le document {document_id}...")
                await self.embedding_manager.create_embeddings(chunks)
                await self.embedding_manager.save_embeddings(document_id)

            return True
        except Exception as e:
            print(f"Erreur lors du traitement du document {document_id}: {str(e)}")
            return False

    async def initialize(self) -> None:
        """Initialise le stockage en chargeant tous les documents et embeddings"""
        print(f"Initialisation du stockage de documents...")
        print(f"Documents trouvés dans les métadonnées: {len(self.documents)}")

        # Charger les chunks et embeddings pour chaque document
        for document_id in self.documents:
            print(f"Chargement du document {document_id}...")
            success = await self.load_document_chunks(document_id)
            if success:
                print(f"Document {document_id} chargé avec succès.")
            else:
                print(f"Erreur lors du chargement du document {document_id}.")

