"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# rag_engine.py
import os
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple

from ..context_provider import SmartContextProvider
from ..context_provider.entity_index import logger
from ..provider.llm_providers import LLMProvider
from ..utils.document_processor import DocumentProcessor
from ..utils.embedding_manager import EmbeddingManager
from ..utils.document_store import DocumentStore
from ..utils.retriever import Retriever
from ..utils.highlighter import Highlighter


class RAGEngine:
    """Moteur RAG orchestrant les différentes composantes"""

    def __init__(
            self,
            llm_provider: LLMProvider,
            embedding_provider: LLMProvider,
            storage_dir: str = "storage",
            chunk_size: int = 1000,
            chunk_overlap: int = 200
    ):
        """
        Initialise le moteur RAG

        Args:
            llm_provider: Provider LLM pour la génération de réponses
            embedding_provider: Provider LLM pour la génération d'embeddings
            storage_dir: Répertoire de stockage
            chunk_size: Taille des chunks
            chunk_overlap: Chevauchement entre chunks
        """
        self.llm_provider = llm_provider

        # Initialiser les composants
        self.processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.embedding_manager = EmbeddingManager(embedding_provider,
                                                  os.path.join(storage_dir, "embeddings"))
        self.document_store = DocumentStore(self.processor, self.embedding_manager, storage_dir)
        self.retriever = Retriever(self.embedding_manager, self.document_store)
        self.highlighter = Highlighter(self.document_store,
                                       os.path.join(storage_dir, "highlighted"))
        self.context_provier = None

    async def initialize(self):
        """Initialise le moteur en chargeant les documents et embeddings"""
        await self.document_store.initialize()

    async def initialize_context_provider(self):
        """Initialise le fournisseur de contexte"""
        logger.info("🔧 Initialisation du Context Provider...")

        self.context_provider = SmartContextProvider(
            self.document_store,
            self  # self = RAGEngine
        )
        await self.context_provider.initialize()

        logger.info("✅ Context Provider initialisé")

    async def get_agent_context(self,
                                entity_name: str,
                                agent_type: str = "developer",
                                task_context: str = "code_understanding",
                                max_tokens: int = 4000) -> Dict[str, Any]:
        """Interface principale pour obtenir le contexte d'un agent"""
        if not self.context_provider:
            await self.initialize_context_provider()

        return await self.context_provider.get_context_for_agent(
            entity_name, agent_type, task_context, max_tokens
        )

    async def search_entities(self, query: str) -> List[Dict[str, Any]]:
        """Recherche d'entités dans le projet"""
        if not self.context_provider:
            await self.initialize_context_provider()

        return await self.context_provider.search_entities(query)

    async def get_local_context(self, entity_name: str, max_tokens: int = 2000) -> Dict[str, Any]:
        """Contexte local d'une entité"""
        if not self.context_provider:
            await self.initialize_context_provider()

        return await self.context_provider.get_local_context(entity_name, max_tokens)

    async def get_global_context(self, entity_name: str, max_tokens: int = 3000) -> Dict[str, Any]:
        """Contexte global d'une entité"""
        if not self.context_provider:
            await self.initialize_context_provider()

        return await self.context_provider.get_global_context(entity_name, max_tokens)

    async def get_semantic_context(self, entity_name: str, max_tokens: int = 2000) -> Dict[str, Any]:
        """Contexte sémantique d'une entité"""
        if not self.context_provider:
            await self.initialize_context_provider()

        return await self.context_provider.get_semantic_context(entity_name, max_tokens)

    def get_context_stats(self) -> Dict[str, Any]:
        """Statistiques du système de contexte"""
        if not self.context_provider:
            return {"error": "Context provider not initialized"}

        return self.context_provider.get_index_stats()

    async def add_document_with_id(self, filepath: str, document_id: str,
                                   additional_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Ajoute un document avec un ID spécifique

        Args:
            filepath: Chemin vers le document
            document_id: ID à utiliser
            additional_metadata: Métadonnées additionnelles

        Returns:
            ID du document ajouté
        """
        return await self.document_store.add_document_with_id(filepath, document_id, additional_metadata)

    async def add_document(self, filepath: str) -> str:
        """
        Ajoute un document au système

        Args:
            filepath: Chemin vers le document

        Returns:
            ID du document ajouté
        """
        return await self.document_store.add_document(filepath)

    async def add_documents(self, filepaths: List[str]) -> List[str]:
        """
        Ajoute plusieurs documents au système

        Args:
            filepaths: Liste des chemins vers les documents

        Returns:
            Liste des IDs des documents ajoutés
        """
        return await self.document_store.add_documents(filepaths)

    async def remove_document(self, document_id: str) -> bool:
        """
        Supprime un document du système

        Args:
            document_id: ID du document à supprimer

        Returns:
            True si la suppression a réussi, False sinon
        """
        return await self.document_store.remove_document(document_id)

    async def get_all_documents(self) -> Dict[str, Dict[str, Any]]:
        """Récupère tous les documents"""
        return await self.document_store.get_all_documents()

    async def search(
            self,
            query: str,
            document_id: Optional[str] = None,
            top_k: int = 5,
            skip_loading: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Recherche les passages les plus pertinents

        Args:
            query: Requête de recherche
            document_id: ID du document (si None, recherche dans tous les documents)
            top_k: Nombre de passages à retourner
            skip_loading: Si True, suppose que les documents sont déjà chargés

        Returns:
            Liste des passages les plus pertinents avec leur score
        """
        return await self.retriever.retrieve(query, document_id, top_k, skip_loading=skip_loading)

    async def search_with_embedding(
            self,
            query: str,
            document_id: Optional[str] = None,
            top_k: int = 5,
            skip_loading: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Recherche les passages les plus pertinents

        Args:
            query: Requête de recherche
            document_id: ID du document (si None, recherche dans tous les documents)
            top_k: Nombre de passages à retourner
            skip_loading: Si True, suppose que les documents sont déjà chargés

        Returns:
            Liste des passages les plus pertinents avec leur score
        """
        return await self.retriever.retrieve_with_embedding(query, document_id, top_k, skip_loading=skip_loading)

    async def highlight_passages(
            self,
            document_id: str,
            passages: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Surligne des passages dans un PDF

        Args:
            document_id: ID du document
            passages: Liste des passages à surligner

        Returns:
            Chemin du PDF surligné ou None si échec
        """
        return await self.highlighter.highlight_passages(document_id, passages)

    async def generate_answer(
            self,
            query: str,
            passages: List[Dict[str, Any]],
            system_prompt: Optional[str] = None
    ) -> str:
        """
        Génère une réponse basée sur les passages récupérés

        Args:
            query: Requête de l'utilisateur
            passages: Passages récupérés
            system_prompt: Prompt système optionnel

        Returns:
            Réponse générée
        """
        # Construire le contexte à partir des passages
        context = self._build_context(passages)

        # Construire le prompt pour le LLM
        prompt = self._build_prompt(query, context, system_prompt)

        # Générer la réponse
        response = await self.llm_provider.generate_response(prompt)

        return response

    def _build_context(self, passages: List[Dict[str, Any]]) -> str:
        """Construit le contexte à partir des passages"""
        context = "Contexte :\n\n"

        for i, passage in enumerate(passages, 1):
            # CORRECTION : extraire le nom du document des métadonnées
            doc_name = self._extract_document_name(passage)
            context += f"[Passage {i} de {doc_name}]\n{passage['text']}\n\n"

        return context

    def _extract_document_name(self, passage: Dict[str, Any]) -> str:
        """Extrait le nom du document à partir d'un passage"""

        # Méthode 1: Via source_info (pour les nouveaux passages du système ontologique)
        if 'source_info' in passage:
            source_info = passage['source_info']
            if 'file' in source_info and source_info['file'] != 'Unknown':
                return source_info['file']

        # Méthode 2: Via metadata.filename
        if 'metadata' in passage:
            metadata = passage['metadata']
            if 'filename' in metadata and metadata['filename']:
                return metadata['filename']

        # Méthode 3: Via document_id et document_store
        if hasattr(self, 'document_store') and 'document_id' in passage:
            doc_id = passage['document_id']
            if doc_id in self.document_store.documents:
                doc_info = self.document_store.documents[doc_id]
                filename = doc_info.get('original_filename', '')
                if filename:
                    return filename

        # Méthode 4: Extraire depuis filepath
        if 'metadata' in passage and 'filepath' in passage['metadata']:
            filepath = passage['metadata']['filepath']
            if filepath:
                from pathlib import Path
                return Path(filepath).name

        # Fallback
        return "Document inconnu"

    def _build_prompt(
            self,
            query: str,
            context: str,
            system_prompt: Optional[str] = None
    ) -> Union[str, List[Dict[str, str]]]:
        """
        Construit le prompt pour le LLM

        Args:
            query: Requête de l'utilisateur
            context: Contexte extrait des documents
            system_prompt: Prompt système optionnel

        Returns:
            Prompt formaté pour le LLM
        """
        default_system = """Tu es un assistant IA qui aide à répondre aux questions en utilisant uniquement le contexte fourni. 
Si la réponse ne se trouve pas dans le contexte, indique-le clairement. 
Ne fabrique pas d'informations. Cite les sources mentionnées dans le contexte.
Base ta réponse exclusivement sur les informations contenues dans les passages fournis."""

        # Utiliser le provider LLM pour formater le prompt
        provider_name = type(self.llm_provider).__name__

        if "OpenAI" in provider_name:
            # Format pour OpenAI (chat messages)
            messages = [
                {"role": "system", "content": system_prompt or default_system},
                {"role": "user", "content": f"{context}\n\nQuestion: {query}"}
            ]
            return messages
        else:
            # Format texte standard pour les autres providers
            prompt = f"{system_prompt or default_system}\n\n{context}\n\nQuestion: {query}\n\nRéponse:"
            return prompt

    async def chat(
            self,
            query: str,
            top_k: int = 5,
            document_id: Optional[str] = None,
            system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Effectue une recherche et génère une réponse pour le chat

        Args:
            query: Requête de l'utilisateur
            top_k: Nombre de passages à récupérer
            document_id: ID du document (si None, recherche dans tous les documents)
            system_prompt: Prompt système optionnel

        Returns:
            Dictionnaire contenant la réponse et les passages
        """
        # Récupérer les passages pertinents
        passages = await self.search(query, document_id, top_k)

        # Générer une réponse
        answer = await self.generate_answer(query, passages, system_prompt)

        # Vérifier si on peut surligner des passages dans un PDF
        highlighted_pdf = None
        if document_id and all(p["document_id"] == document_id for p in passages):
            document = await self.document_store.get_document(document_id)
            if document and document["path"].lower().endswith('.pdf'):
                highlighted_pdf = await self.highlight_passages(document_id, passages)

        return {
            "answer": answer,
            "passages": passages,
            "highlighted_pdf": highlighted_pdf
        }

    # ------------- RELATIONS -------------------
    async def extract_relations(self, query: str, confidence_threshold: float = 0.65) -> List[Dict[str, Any]]:
        """
        Extrait les relations possibles entre concepts dans un texte.

        Args:
            query: Texte à analyser
            confidence_threshold: Seuil minimal de confiance pour les concepts

        Returns:
            Liste des triplets (sujet, relation, objet) possibles
        """
        # Utiliser la référence au classifieur si disponible
        if hasattr(self, 'classifier') and self.classifier:
            return await self.classifier.extract_possible_relations(query, confidence_threshold)

        return []

    async def add_document_with_metadata(
            self,
            filepath: str,
            document_id: str = None,
            additional_metadata: Dict[str, Any] = None
    ) -> str:
        """
        Ajoute un document avec des métadonnées supplémentaires

        Args:
            filepath: Chemin vers le document
            document_id: ID à utiliser (si None, génère un UUID)
            additional_metadata: Métadonnées supplémentaires

        Returns:
            ID du document ajouté
        """
        # Si pas d'ID fourni, en générer un
        if document_id is None:
            document_id = str(uuid.uuid4())

        # Utiliser la méthode du document_store
        return await self.document_store.add_document_with_id(
            filepath,
            document_id,
            additional_metadata
        )

    async def get_related_concepts(self, concept_uri: str, relation_uri: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Récupère les concepts liés à un concept via une relation.

        Args:
            concept_uri: URI du concept source
            relation_uri: URI de la relation
            top_k: Nombre maximum de résultats

        Returns:
            Liste des concepts liés
        """
        # Utiliser la référence au classifieur si disponible
        if hasattr(self, 'classifier') and self.classifier:
            return await self.classifier.get_related_concepts(concept_uri, relation_uri, top_k)

        return []

    async def learn_ontology_relations(self, examples: Dict[str, List[Tuple[str, str]]]) -> Dict[str, bool]:
        """
        Entraîne le système à reconnaître les relations ontologiques à partir d'exemples.

        Args:
            examples: Dictionnaire {relation_uri: [(sujet_uri1, objet_uri1), (sujet_uri2, objet_uri2), ...]}

        Returns:
            Dictionnaire des résultats d'apprentissage par relation
        """
        if hasattr(self, 'classifier') and self.classifier:
            return await self.classifier.learn_relations_from_examples(examples)

        return {}

    async def query_with_relations(self, query: str, use_relations: bool = True) -> Dict[str, Any]:
        """
        Répond à une requête en utilisant les relations ontologiques.

        Args:
            query: Question ou requête de l'utilisateur
            use_relations: Si True, enrichit la réponse avec des informations sur les relations détectées

        Returns:
            Dictionnaire contenant la réponse et les informations associées
        """
        # Effectuer une recherche standard
        search_result = await self.chat(query)

        # Si l'utilisation des relations est désactivée, retourner juste la recherche standard
        if not use_relations or not hasattr(self, 'classifier') or not self.classifier:
            return search_result

        # Extraire les relations possibles dans la requête
        relations = await self.extract_relations(query)

        # Si aucune relation n'est trouvée, retourner juste la recherche standard
        if not relations:
            return search_result

        # Enrichir la réponse avec les informations sur les relations
        search_result["detected_relations"] = relations

        # Créer un prompt enrichi qui inclut les relations détectées
        if relations and search_result.get("answer"):
            enriched_answer = search_result["answer"]

            # Ajouter une section sur les relations détectées
            enriched_answer += "\n\nRelations détectées dans votre question:\n"
            for i, rel in enumerate(relations[:3], 1):  # Limiter à 3 relations pour la lisibilité
                enriched_answer += f"{i}. {rel['subject_label']} → {rel['relation_label']} → {rel['object_label']} (confiance: {rel['confidence']:.2f})\n"

            search_result["enriched_answer"] = enriched_answer

        return search_result

