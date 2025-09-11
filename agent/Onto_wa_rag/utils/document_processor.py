"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# document_processor.py
import asyncio
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ..utils.mega_converter import MarkdownConverter
from ..semantic_analysis.core.semantic_chunker import SemanticChunker


class DocumentProcessor:
    """Classe pour traiter les documents, extraire le texte et créer des chunks"""

    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 use_semantic_chunking: bool = True
                 ):
        """
        Initialise le processeur de documents

        Args:
            chunk_size: Taille des chunks en caractères
            chunk_overlap: Chevauchement entre chunks consécutifs
            use_semantic_chunking: Utiliser le chunking sémantique
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic_chunking = use_semantic_chunking
        self.converter = MarkdownConverter()

        # Initialiser le chunker sémantique si nécessaire
        if use_semantic_chunking:
            self.semantic_chunker = SemanticChunker(
                min_chunk_size=200,
                max_chunk_size=chunk_size,
                overlap_sentences=2,  # 2 phrases d'overlap
            )

    async def process_document(self,
                               filepath: str,
                               document_id: Optional[str] = None,
                               additional_metadata: Optional[Dict[str, Any]] = None
                               ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Traite un document avec possibilité de spécifier un ID

        Args:
            filepath: Chemin vers le document
            document_id: ID prédéfini à utiliser (si None, génère un UUID)
            additional_metadata: Métadonnées supplémentaires à ajouter
        """
        # Utiliser l'ID fourni ou générer un UUID
        doc_id = document_id if document_id is not None else str(uuid.uuid4())

        # Déterminer le type de fichier
        extension = Path(filepath).suffix.lower()
        file_type = extension.lstrip(".")

        # === Notebook handler ===
        if file_type == "ipynb":
            print(f"✂️  Utilisation du retriever spécialisé pour notebook: {Path(filepath).name}")
            from retriever_adapter import SimpleRetriever
            retriever = SimpleRetriever()
            retriever.build_index_from_notebook(str(filepath))

            chunks = []
            for i, c in enumerate(retriever.chunks):
                chunks.append({
                    "id": f"{doc_id}_chunk_{i}",
                    "document_id": doc_id,
                    "content": c["content"],
                    "metadata": {
                        "filepath": str(filepath),
                        "filename": Path(filepath).name,
                        "file_type": "notebook",
                        "tokens": c.get("tokens", None),
                        **(additional_metadata or {})
                    }
                })
            return doc_id, chunks

        # Sinon: traitement standard
        text_content, doc_metadata = await self._extract_text_with_metadata(filepath)
        metadata = {**doc_metadata, **(additional_metadata or {})}

        if self.use_semantic_chunking:
            chunks = self.semantic_chunker.create_semantic_chunks(
                text_content, doc_id, filepath, metadata
            )
        else:
            chunks = self._create_chunks(text_content, doc_id, filepath)

        return doc_id, chunks

    async def _extract_text_with_metadata(self, filepath: str) -> Tuple[str, Dict[str, Any]]:
        """Extrait le texte et fusionne métadonnées système + convertisseur"""
        loop = asyncio.get_event_loop()
        conv_result = await loop.run_in_executor(None, self.converter.convert_local, filepath)
        text = conv_result.text_content

        meta_sys = {
            'file_size': os.path.getsize(filepath),
            'file_type': os.path.splitext(filepath)[1].lstrip('.').lower(),
            'modification_date': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat(),
            'extraction_date': datetime.now().isoformat(),
            'filename': os.path.basename(filepath),
            'filepath': str(filepath)
        }
        meta_conv = getattr(conv_result, 'metadata', {}) or {}
        return text, {**meta_sys, **meta_conv}

    async def _extract_text(self, filepath: str) -> str:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.converter.convert_local, filepath)
        return result.text_content

    def _create_chunks(self, text: str, document_id: str, filepath: str) -> List[Dict[str, Any]]:
        """Divise un texte en chunks en respectant les frontières des phrases"""
        import re
        chunks = []
        if len(text) <= self.chunk_size:
            chunks.append({
                "id": f"{document_id}-chunk-0",
                "document_id": document_id,
                "text": text,
                "start_pos": 0,
                "end_pos": len(text),
                "metadata": {"filepath": filepath, "filename": os.path.basename(filepath)}
            })
            return chunks

        sentence_pattern = r'(?<=[.!?])\s+|\n\s*'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks_sentences = []
        current_chunk, current_size = [], 0

        for sentence in sentences:
            sentence_size = len(sentence) + (1 if current_chunk else 0)

            if len(sentence) > self.chunk_size and not current_chunk:
                # Handle very long sentence case...
                start = 0
                while start < len(sentence):
                    end = min(start + self.chunk_size, len(sentence))
                    chunk_text = sentence[start:end].strip()
                    chunks.append({
                        "id": f"{document_id}-chunk-{len(chunks)}",
                        "document_id": document_id,
                        "text": chunk_text,
                        "start_pos": start,
                        "end_pos": end,
                        "metadata": {"filepath": filepath, "filename": os.path.basename(filepath)}
                    })
                    start = end - self.chunk_overlap
                continue

            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks_sentences.append(current_chunk)
                overlap_sentences, overlap_size = [], 0
                for s in reversed(current_chunk):
                    if overlap_size + len(s) + 1 <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s) + 1
                current_chunk, current_size = overlap_sentences.copy(), sum(len(s) for s in overlap_sentences)

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunks_sentences.append(current_chunk)

        for i, sentence_list in enumerate(chunks_sentences):
            chunk_text = ' '.join(sentence_list)
            start_pos = text.find(sentence_list[0]) if sentence_list else 0
            end_pos = start_pos + len(chunk_text)
            chunks.append({
                "id": f"{document_id}-chunk-{i}",
                "document_id": document_id,
                "text": chunk_text,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "metadata": {"filepath": filepath, "filename": os.path.basename(filepath)}
            })
        return chunks
