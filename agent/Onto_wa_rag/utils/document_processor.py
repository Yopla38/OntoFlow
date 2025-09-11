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
                #respect_boundaries=True
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

        # Extraire le texte et les métadonnées
        text_content, doc_metadata = await self._extract_text_with_metadata(filepath)

        # Fusionner les métadonnées
        metadata = {**doc_metadata, **(additional_metadata or {})}
        #metadata = {**doc_metadata, **result.metadata, **(additional_metadata or {})}

        # Créer les chunks selon la méthode choisie
        if self.use_semantic_chunking:
            chunks = self.semantic_chunker.create_semantic_chunks(
                text_content, doc_id, filepath, metadata
            )
        else:
            chunks = self._create_chunks(text_content, doc_id, filepath)

        return doc_id, chunks

    async def _extract_text_with_metadata(self, filepath: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extrait le texte (markdown) et fusionne les métadonnées :
          – métadonnées système (taille fichier, dates…)
          – métadonnées fournies par le convertisseur (titre, doi, auteurs…)
        """
        loop = asyncio.get_event_loop()

        # conversion asynchrone
        conv_result = await loop.run_in_executor(None, self.converter.convert_local, filepath)
        # conv_result doit exposer .text_content et (optionnellement) .metadata
        text = conv_result.text_content

        # --- métadonnées système (toujours présentes) ---
        meta_sys = {
            'file_size': os.path.getsize(filepath),
            'file_type': os.path.splitext(filepath)[1].lstrip('.').lower(),
            'modification_date': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat(),
            'extraction_date': datetime.now().isoformat(),
            'filename': os.path.basename(filepath),
            'filepath': str(filepath)
        }

        # --- métadonnées issues du convertisseur ---
        meta_conv = getattr(conv_result, 'metadata', {}) or {}

        # fusion
        metadata = {**meta_sys, **meta_conv}

        return text, metadata

    async def _extract_text(self, filepath: str) -> str:
        """Extrait le texte d'un document en utilisant le convertisseur"""
        # Utiliser run_in_executor pour rendre cette opération asynchrone
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.converter.convert_local, filepath)
        return result.text_content

    def _create_chunks(self, text: str, document_id: str, filepath: str) -> List[Dict[str, Any]]:
        """
        Divise un texte en chunks en respectant les frontières des phrases

        Args:
            text: Texte à diviser
            document_id: ID du document parent
            filepath: Chemin du document

        Returns:
            Liste de dictionnaires représentant chaque chunk
        """
        import re

        chunks = []

        # Si le texte est plus court que la taille de chunk, pas besoin de diviser
        if len(text) <= self.chunk_size:
            chunks.append({
                "id": f"{document_id}-chunk-0",
                "document_id": document_id,
                "text": text,
                "start_pos": 0,
                "end_pos": len(text),
                "metadata": {
                    "filepath": filepath,
                    "filename": os.path.basename(filepath)
                }
            })
            return chunks

        # Diviser le texte en phrases
        # Cette regex capture les fins de phrases (points, points d'exclamation, points d'interrogation)
        # et les sauts de ligne comme séparateurs de phrases
        sentence_pattern = r'(?<=[.!?])\s+|\n\s*'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Construire les chunks
        chunks_sentences = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            # Calculer la taille de cette phrase (+1 pour l'espace si ce n'est pas la première phrase)
            sentence_size = len(sentence) + (1 if current_chunk else 0)

            # Cas spécial: une seule phrase est déjà plus grande que le chunk_size
            if len(sentence) > self.chunk_size and not current_chunk:
                # Diviser la phrase en segments plus petits (en essayant de couper aux espaces)
                start = 0
                while start < len(sentence):
                    end = start + self.chunk_size

                    # Si on n'a pas atteint la fin, essayer de couper à un espace
                    if end < len(sentence):
                        # Chercher le dernier espace avant la limite
                        last_space = sentence.rfind(' ', start, end)
                        if last_space > start:
                            end = last_space
                    else:
                        end = len(sentence)

                    chunk_text = sentence[start:end].strip()
                    if len(chunk_text) >= 50:  # Ne pas créer de chunks trop petits
                        chunks.append({
                            "id": f"{document_id}-chunk-{len(chunks)}",
                            "document_id": document_id,
                            "text": chunk_text,
                            "start_pos": start,
                            "end_pos": end,
                            "metadata": {
                                "filepath": filepath,
                                "filename": os.path.basename(filepath)
                            }
                        })

                    # Avancer avec chevauchement
                    if end < len(sentence):
                        # Trouver un bon point de départ pour le chevauchement
                        overlap_start = max(start, end - self.chunk_overlap)
                        if overlap_start > start:
                            # Chercher le premier espace après le début du chevauchement
                            next_space = sentence.find(' ', overlap_start)
                            if next_space > overlap_start and next_space < end:
                                overlap_start = next_space + 1

                        start = overlap_start
                    else:
                        break

                # Continuer avec la phrase suivante
                continue

            # Si ajouter cette phrase dépasse la taille max et qu'on a déjà des phrases
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Créer un chunk avec les phrases actuelles
                chunks_sentences.append(current_chunk)

                # Calculer combien de phrases on garde pour le chevauchement
                overlap_size = 0
                overlap_sentences = []

                for s in reversed(current_chunk):
                    if overlap_size + len(s) + 1 <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s) + 1
                    else:
                        break

                # Commencer un nouveau chunk avec les phrases qui se chevauchent
                current_chunk = overlap_sentences.copy()
                current_size = sum(len(s) for s in current_chunk) + max(0, len(current_chunk) - 1)

            # Ajouter la phrase au chunk actuel
            current_chunk.append(sentence)
            current_size += sentence_size

        # Ajouter le dernier chunk s'il existe
        if current_chunk:
            chunks_sentences.append(current_chunk)

        # Convertir les listes de phrases en chunks finaux
        for i, sentence_list in enumerate(chunks_sentences):
            # Joindre les phrases
            chunk_text = ' '.join(sentence_list)

            # Calculer les positions dans le texte original (approximatif)
            start_pos = text.find(sentence_list[0])
            if start_pos == -1:
                start_pos = 0  # Fallback si phrase non trouvée

            end_pos = start_pos + len(chunk_text)

            chunks.append({
                "id": f"{document_id}-chunk-{i}",
                "document_id": document_id,
                "text": chunk_text,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "metadata": {
                    "filepath": filepath,
                    "filename": os.path.basename(filepath)
                }
            })

        return chunks

