"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# retriever.py
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy.spatial.distance import cosine


class Retriever:
    """Récupération des passages pertinents"""

    def __init__(
            self,
            embedding_manager,
            document_store
    ):
        """
        Initialise le récupérateur

        Args:
            embedding_manager: Gestionnaire d'embeddings
            document_store: Stockage de documents
        """
        self.embedding_manager = embedding_manager
        self.document_store = document_store

    async def retrieve_with_filters(
            self,
            query: str,
            filters: Dict[str, Any] = None,
            document_id: Optional[str] = None,
            top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Récupère les passages les plus pertinents avec filtres sur les métadonnées

        Args:
            query: Requête de recherche
            filters: Filtres sur les métadonnées (ex: {"section_level": 1, "language": "fr"})
            document_id: ID du document (si None, recherche dans tous les documents)
            top_k: Nombre de passages à retourner

        Returns:
            Liste des passages les plus pertinents avec leur score
        """
        # Générer l'embedding de la requête
        query_embedding = (await self.embedding_manager.provider.generate_embeddings([query]))[0]

        # Récupérer tous les passages pertinents
        passages = await self.retrieve_with_embedding(query_embedding, document_id, top_k * 3)

        # Appliquer les filtres si spécifiés
        if filters:
            filtered_passages = []
            for passage in passages:
                metadata = passage.get("metadata", {})
                match = True

                # Vérifier chaque critère de filtre
                for key, value in filters.items():
                    # Gérer les filtres spéciaux
                    if key == "min_info_density":
                        if metadata.get("info_density", 0) < value:
                            match = False
                            break
                    elif key == "has_entities":
                        if not metadata.get("entities", []):
                            match = False
                            break
                    elif key == "keyword_match":
                        if not any(kw in metadata.get("chunk_keywords", []) for kw in value):
                            match = False
                            break
                    elif key == "section_path_contains":
                        if not any(value in title for title in metadata.get("section_path", [])):
                            match = False
                            break
                    # Filtre standard d'égalité
                    elif metadata.get(key) != value:
                        match = False
                        break

                if match:
                    filtered_passages.append(passage)

            passages = filtered_passages

        # Trier par pertinence et prendre les top_k
        passages.sort(key=lambda x: x["similarity"], reverse=True)
        return passages[:top_k]

    async def retrieve_with_embedding(
            self,
            query_embedding: np.ndarray,  # Modifié pour accepter directement un numpy array
            document_id: Optional[str] = None,
            top_k: int = 5,
            skip_loading: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Récupère les passages les plus pertinents pour une requête (optimisé avec SIMD)

        Args:
            query_embedding: Embedding de la requête sous forme de numpy array
            document_id: ID du document (si None, recherche dans tous les documents)
            top_k: Nombre de passages à retourner
            skip_loading: Si True, suppose que les documents sont déjà chargés

        Returns:
            Liste des passages les plus pertinents avec leur score
        """
        # 1. Accéder à tous les embeddings disponibles en mémoire
        all_embeddings = self.embedding_manager.get_all_embeddings()

        # 2. Filtrer par document_id si spécifié
        if document_id:
            filtered_embeddings = {
                chunk_id: embedding
                for chunk_id, embedding in all_embeddings.items()
                if chunk_id.startswith(f"{document_id}-chunk-") and embedding is not None
            }
        else:
            filtered_embeddings = {k: v for k, v in all_embeddings.items() if v is not None}

        if not filtered_embeddings:
            return []

        # 3. Convertir les embeddings en une seule matrice NumPy pour calcul vectorisé
        chunk_ids = list(filtered_embeddings.keys())
        embeddings_matrix = np.vstack([filtered_embeddings[chunk_id] for chunk_id in chunk_ids])

        # 4. Calculer toutes les similarités en une seule opération vectorisée
        # Si les embeddings sont déjà normalisés, le produit scalaire = similarité cosinus
        similarities = np.dot(embeddings_matrix, query_embedding)

        # 5. Obtenir les indices des top_k éléments
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # 6. Créer la liste des paires (chunk_id, similarity) pour les meilleurs résultats
        top_chunks = [(chunk_ids[idx], similarities[idx]) for idx in top_indices]

        # 7. Charger et formater les documents pour les résultats finaux
        passages = []
        loaded_documents = set()

        for chunk_id, similarity in top_chunks:
            # Extraire l'ID du document à partir de l'ID du chunk
            parts = chunk_id.split("-chunk-")
            if len(parts) != 2:
                continue

            document_id = parts[0]

            # Charger les chunks du document si nécessaire (une seule fois par document)
            if not skip_loading and document_id not in loaded_documents:
                await self.document_store.load_document_chunks(document_id)
                loaded_documents.add(document_id)

            # Récupérer les informations sur le chunk
            doc_chunks = await self.document_store.get_document_chunks(document_id)

            if not doc_chunks:
                continue

            # Trouver le chunk correspondant
            chunk = None
            for c in doc_chunks:
                if c["id"] == chunk_id:
                    chunk = c
                    break

            if not chunk:
                continue

            # Récupérer les informations sur le document
            document = await self.document_store.get_document(document_id)

            if not document:
                continue

            # Ajouter le passage aux résultats
            passages.append({
                "document_id": document_id,
                "chunk_id": chunk_id,
                "text": chunk["text"],
                "similarity": float(similarity),  # Convertir le np.float en Python float
                "start_pos": chunk["start_pos"],
                "end_pos": chunk["end_pos"],
                "metadata": chunk.get("metadata", {}),
                "document_name": document.get("original_filename", ""),
                "document_path": document.get("path", "")
            })

        return passages


    async def retrieve(
            self,
            query: str,
            document_id: Optional[str] = None,
            top_k: int = 5,
            skip_loading: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Récupère les passages les plus pertinents pour une requête

        Args:
            query: Requête de recherche
            document_id: ID du document (si None, recherche dans tous les documents)
            top_k: Nombre de passages à retourner
            skip_loading: Si True, suppose que les documents sont déjà chargés

        Returns:
            Liste des passages les plus pertinents avec leur score
        """
        # Générer l'embedding de la requête
        query_embedding = (await self.embedding_manager.provider.generate_embeddings([query]))[0]

        passages_with_scores = []

        # Si un document spécifique est demandé
        if document_id:
            if not skip_loading:  # Ne charge que si nécessaire
                await self.document_store.load_document_chunks(document_id)
            doc_chunks = await self.document_store.get_document_chunks(document_id)

            if not doc_chunks:
                return []

            for chunk in doc_chunks:
                chunk_id = chunk["id"]
                embedding = self.embedding_manager.get_embedding(chunk_id)

                if embedding:
                    # Calculer la similarité avec la requête
                    similarity = 1 - cosine(query_embedding, embedding)
                    passages_with_scores.append({
                        "document_id": document_id,
                        "chunk_id": chunk_id,
                        "text": chunk["text"],
                        "similarity": similarity,
                        "start_pos": chunk["start_pos"],
                        "end_pos": chunk["end_pos"],
                        "metadata": chunk.get("metadata", {})
                    })
        else:
            # Pour tous les documents
            all_documents = await self.document_store.get_all_documents()

            for doc_id in all_documents:
                if not skip_loading:
                    await self.document_store.load_document_chunks(doc_id)

                doc_chunks = await self.document_store.get_document_chunks(doc_id)

                if not doc_chunks:
                    continue

                for chunk in doc_chunks:
                    chunk_id = chunk["id"]
                    embedding = self.embedding_manager.get_embedding(chunk_id)

                    if embedding:
                        # Calculer la similarité avec la requête
                        similarity = 1 - cosine(query_embedding, embedding)
                        passages_with_scores.append({
                            "document_id": doc_id,
                            "chunk_id": chunk_id,
                            "text": chunk["text"],
                            "similarity": similarity,
                            "start_pos": chunk["start_pos"],
                            "end_pos": chunk["end_pos"],
                            "metadata": chunk.get("metadata", {})
                        })
                        
        # Trier par similarité et prendre les top_k
        passages_with_scores.sort(key=lambda x: x["similarity"], reverse=True)
        top_passages = passages_with_scores[:top_k]

        # Ajouter des informations sur les documents
        for passage in top_passages:
            doc_id = passage["document_id"]
            doc_info = await self.document_store.get_document(doc_id)

            if doc_info:
                passage["document_name"] = doc_info.get("original_filename", "")
                passage["document_path"] = doc_info.get("path", "")

        return top_passages

    async def retrieve_with_structure_boost(
            self,
            query: str,
            document_id: Optional[str] = None,
            top_k: int = 5,
            boost_headers: float = 1.2,  # Boost pour les chunks avec headers
            boost_first_paragraph: float = 1.1,  # Boost pour les premiers paragraphes
            skip_loading: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Récupère les passages en tenant compte de la structure du document

        Args:
            query: Requête de recherche
            document_id: ID du document (si None, recherche dans tous)
            top_k: Nombre de passages à retourner
            boost_headers: Multiplicateur pour chunks contenant des headers
            boost_first_paragraph: Multiplicateur pour les premiers paragraphes
            skip_loading: Si True, suppose que les documents sont déjà chargés

        Returns:
            Liste des passages les plus pertinents avec leur score ajusté
        """
        # Récupérer les résultats de base
        passages = await self.retrieve(query, document_id, top_k * 2, skip_loading)

        # Ajuster les scores selon la structure
        for passage in passages:
            metadata = passage.get("metadata", {})
            original_score = passage["similarity"]
            adjusted_score = original_score

            # Boost si le chunk contient des headers
            if metadata.get("has_headers", False):
                adjusted_score *= boost_headers

            # Boost si c'est un premier paragraphe d'une section
            section_hierarchy = metadata.get("section_hierarchy", [])
            if section_hierarchy and metadata.get("section_types", [])[0] == "paragraph":
                adjusted_score *= boost_first_paragraph

            # Pénaliser les chunks partiels
            if metadata.get("is_partial_section", False):
                adjusted_score *= 0.9

            # Stocker les deux scores
            passage["original_similarity"] = original_score
            passage["similarity"] = adjusted_score
            passage["score_adjustments"] = {
                "has_headers": metadata.get("has_headers", False),
                "is_first_paragraph": bool(section_hierarchy),
                "is_partial": metadata.get("is_partial_section", False)
            }

        # Re-trier selon les scores ajustés
        passages.sort(key=lambda x: x["similarity"], reverse=True)

        return passages[:top_k]