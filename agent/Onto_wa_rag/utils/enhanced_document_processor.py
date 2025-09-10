"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

import asyncio
import os
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from utils.mega_converter import MarkdownConverter



class EnhancedDocumentProcessor:
    """Processeur de documents enrichi avec extraction avancée de métadonnées et chunking intelligent"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialise le processeur de documents enrichi

        Args:
            chunk_size: Taille des chunks en caractères
            chunk_overlap: Chevauchement entre chunks consécutifs
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.converter = MarkdownConverter()

        # Initialiser les outils NLP si besoin
        self.nlp = None
        model_name = "fr_core_news_sm"
        import spacy
        try:
            self.nlp = spacy.load(model_name)
            print("✓ Spacy chargé avec succès pour l'extraction d'entités")
        except OSError:
            print(f"Le modèle '{model_name}' est introuvable. Téléchargement en cours...")
            from spacy.cli import download
            download(model_name)
            self.nlp = spacy.load(model_name)
            print("✓ Spacy chargé avec succès pour l'extraction d'entités")

    async def process_document(self, filepath: str, document_id: Optional[str] = None) -> Tuple[
        str, List[Dict[str, Any]]]:
        """
        Traite un document avec extraction avancée de métadonnées

        Args:
            filepath: Chemin vers le document
            document_id: ID prédéfini à utiliser (si None, génère un UUID)
        """
        # Utiliser l'ID fourni ou générer un UUID
        doc_id = document_id if document_id is not None else str(uuid.uuid4())

        # Extraire le texte du document
        text_content = await self._extract_text(filepath)

        # Extraire les métadonnées du document
        doc_metadata = await self._extract_document_metadata(filepath, text_content)

        # Analyser la structure du document
        document_structure = self._analyze_document_structure(text_content)

        # Créer des chunks intelligents qui respectent la structure
        chunks = await self._create_intelligent_chunks(text_content, doc_id, filepath, document_structure, doc_metadata)

        # Enrichir les chunks avec des informations sémantiques
        enhanced_chunks = await self._enhance_chunks_with_semantics(chunks)

        return doc_id, enhanced_chunks

    async def _extract_text(self, filepath: str) -> str:
        """Extrait le texte d'un document en utilisant le convertisseur"""
        # Utiliser run_in_executor pour rendre cette opération asynchrone
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.converter.convert_local, filepath)
        return result.text_content

    async def _extract_document_metadata(self, filepath: str, text_content: str) -> Dict[str, Any]:
        """
        Extrait des métadonnées avancées du document
        """
        # Métadonnées de base du fichier
        filename = os.path.basename(filepath)
        file_ext = os.path.splitext(filename)[1].lower()

        try:
            creation_date = datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()
        except:
            creation_date = None

        try:
            modification_date = datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
        except:
            modification_date = None

        # Extraire le titre du document (première ligne ou premier titre markdown)
        title = None
        lines = text_content.split('\n')
        if lines:
            # Rechercher le premier titre markdown
            for line in lines:
                if line.startswith('# '):
                    title = line.replace('# ', '').strip()
                    break

            # Si aucun titre n'est trouvé, utiliser la première ligne non vide
            if not title:
                for line in lines:
                    if line.strip():
                        title = line.strip()
                        break

        # Détecter la langue
        language = self._detect_language(text_content)

        # Extraire les mots-clés
        keywords = self._extract_keywords(text_content)

        return {
            "filename": filename,
            "file_extension": file_ext,
            "title": title,
            "creation_date": creation_date,
            "modification_date": modification_date,
            "language": language,
            "document_length": len(text_content),
            "keywords": keywords,
            "source_path": filepath
        }

    def _analyze_document_structure(self, text_content: str) -> List[Dict[str, Any]]:
        """
        Analyse la structure hiérarchique du document (titres, sections)
        """
        lines = text_content.split('\n')
        sections = []
        current_section = None
        section_stack = []

        line_index = 0
        char_position = 0

        # Créer une correspondance entre positions de caractères et numéros de ligne
        line_positions = [0]  # Position de début de chaque ligne
        current_pos = 0
        for line in lines:
            current_pos += len(line) + 1  # +1 pour le caractère '\n'
            line_positions.append(current_pos)

        for line in lines:
            # Détecter les titres markdown (# Titre, ## Sous-titre, etc.)
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match:
                header_level = len(header_match.group(1))
                header_text = header_match.group(2).strip()

                # Terminer la section courante si elle existe
                if current_section:
                    current_section["end_pos"] = char_position
                    current_section["end_line"] = line_index
                    current_section["content_length"] = current_section["end_pos"] - current_section["start_pos"]

                # Dépiler les sections de niveau supérieur ou égal
                while section_stack and section_stack[-1]["level"] >= header_level:
                    section_stack.pop()

                # Créer une nouvelle section
                new_section = {
                    "title": header_text,
                    "level": header_level,
                    "start_pos": char_position,
                    "start_line": line_index,
                    "end_pos": None,  # Sera rempli plus tard
                    "end_line": None,
                    "content_length": None,
                    "subsections": [],
                    "parent": section_stack[-1]["id"] if section_stack else None,
                    "id": len(sections)
                }

                # Mettre à jour la hiérarchie
                if section_stack:
                    section_stack[-1]["subsections"].append(new_section["id"])

                sections.append(new_section)
                section_stack.append(new_section)
                current_section = new_section

            line_index += 1
            char_position += len(line) + 1  # +1 pour le saut de ligne

        # Terminer la dernière section
        if current_section and current_section["end_pos"] is None:
            current_section["end_pos"] = char_position
            current_section["end_line"] = line_index
            current_section["content_length"] = current_section["end_pos"] - current_section["start_pos"]

        # Si aucune section n'a été trouvée, créer une section pour le document entier
        if not sections:
            sections.append({
                "title": "Document entier",
                "level": 0,
                "start_pos": 0,
                "start_line": 0,
                "end_pos": char_position,
                "end_line": line_index,
                "content_length": char_position,
                "subsections": [],
                "parent": None,
                "id": 0
            })

        # Calculer les chemins complets pour chaque section
        for section in sections:
            section["path"] = self._calculate_section_path(section, sections)

        # Stocker les positions des lignes pour la conversion caractère -> ligne
        self.line_positions = line_positions

        return sections

    def _get_line_number(self, char_position: int) -> int:
        """
        Convertit une position de caractère en numéro de ligne

        Args:
            char_position: Position du caractère dans le document

        Returns:
            Numéro de ligne (indexé à partir de 1)
        """
        if not hasattr(self, 'line_positions'):
            return 1  # Fallback si la conversion n'est pas disponible

        # Trouver la ligne correspondant à cette position
        for i, pos in enumerate(self.line_positions):
            if pos > char_position:
                return i  # La position est dans la ligne précédente

        return len(self.line_positions)  # Dernière ligne

    def _calculate_section_path(self, section: Dict[str, Any], all_sections: List[Dict[str, Any]]) -> List[str]:
        """Calcule le chemin hiérarchique complet d'une section"""
        path = [section["title"]]
        parent_id = section["parent"]

        while parent_id is not None:
            parent = all_sections[parent_id]
            path.insert(0, parent["title"])
            parent_id = parent["parent"]

        return path

    async def _create_intelligent_chunks(
            self,
            text: str,
            document_id: str,
            filepath: str,
            structure: List[Dict[str, Any]],
            doc_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Crée des chunks qui respectent la structure du document
        """
        chunks = []

        # Si le document est très court
        if len(text) <= self.chunk_size:
            start_line = 1
            end_line = text.count('\n') + 1

            chunk = {
                "id": f"{document_id}-chunk-0",
                "document_id": document_id,
                "text": text,
                "start_pos": 0,
                "end_pos": len(text),
                "metadata": {
                    "filepath": filepath,
                    "filename": os.path.basename(filepath),
                    "document_title": doc_metadata.get("title", ""),
                    "language": doc_metadata.get("language", ""),
                    "content_type": "full_document",
                    "document_keywords": doc_metadata.get("keywords", []),
                    "creation_date": doc_metadata.get("creation_date"),
                    "section_path": [],
                    "is_complete_document": True,
                    "start_line": start_line,
                    "end_line": end_line
                }
            }
            chunks.append(chunk)
            return chunks

        # Trier les sections par position de début
        sorted_sections = sorted(structure, key=lambda s: s["start_pos"])

        # Traiter chaque section
        for section in sorted_sections:
            section_text = text[section["start_pos"]:section["end_pos"]]
            section_path = section["path"]

            # Convertir les positions en numéros de ligne
            start_line = section.get("start_line", 0) + 1  # +1 pour l'indexation à partir de 1
            end_line = section.get("end_line", 0) + 1

            # Si la section est suffisamment petite, la garder entière
            if len(section_text) <= self.chunk_size:
                chunk = {
                    "id": f"{document_id}-chunk-{len(chunks)}",
                    "document_id": document_id,
                    "text": section_text,
                    "start_pos": section["start_pos"],
                    "end_pos": section["end_pos"],
                    "metadata": {
                        "filepath": filepath,
                        "filename": os.path.basename(filepath),
                        "document_title": doc_metadata.get("title", ""),
                        "section_title": section["title"],
                        "section_level": section["level"],
                        "section_path": section_path,
                        "content_type": "section",
                        "language": doc_metadata.get("language", ""),
                        "document_keywords": doc_metadata.get("keywords", []),
                        "creation_date": doc_metadata.get("creation_date"),
                        "is_complete_section": True,
                        "start_line": start_line,
                        "end_line": end_line
                    }
                }
                chunks.append(chunk)
            else:
                # Diviser la section en paragraphes
                paragraphs = self._split_into_paragraphs(section_text)

                # Créer des chunks qui respectent les frontières des paragraphes
                current_chunk_text = ""
                current_paragraphs = []
                current_start = section["start_pos"]

                for paragraph in paragraphs:
                    # Si ajouter ce paragraphe dépasse la taille maximale et qu'on a déjà du contenu
                    if len(current_chunk_text) + len(paragraph) > self.chunk_size and current_chunk_text:
                        # Créer un chunk avec les paragraphes actuels
                        chunk_end = current_start + len(current_chunk_text)
                        chunk = {
                            "id": f"{document_id}-chunk-{len(chunks)}",
                            "document_id": document_id,
                            "text": current_chunk_text,
                            "start_pos": current_start,
                            "end_pos": chunk_end,
                            "metadata": {
                                "filepath": filepath,
                                "filename": os.path.basename(filepath),
                                "document_title": doc_metadata.get("title", ""),
                                "section_title": section["title"],
                                "section_level": section["level"],
                                "section_path": section_path,
                                "content_type": "partial_section",
                                "language": doc_metadata.get("language", ""),
                                "document_keywords": doc_metadata.get("keywords", []),
                                "creation_date": doc_metadata.get("creation_date"),
                                "paragraph_count": len(current_paragraphs),
                                "is_complete_section": False
                            }
                        }
                        chunks.append(chunk)

                        # Commencer un nouveau chunk avec chevauchement
                        overlap_text = ""
                        overlap_paragraphs = []
                        current_size = 0

                        # Ajouter des paragraphes au chevauchement jusqu'à atteindre overlap
                        for p in reversed(current_paragraphs):
                            if current_size + len(p) <= self.chunk_overlap:
                                overlap_paragraphs.insert(0, p)
                                current_size += len(p)
                            else:
                                break

                        overlap_text = "".join(overlap_paragraphs)
                        current_start = chunk_end - len(overlap_text)
                        current_chunk_text = overlap_text
                        current_paragraphs = overlap_paragraphs.copy()

                    # Ajouter le paragraphe au chunk actuel
                    current_chunk_text += paragraph
                    current_paragraphs.append(paragraph)

                # Ajouter le dernier chunk s'il existe
                if current_chunk_text:
                    chunk = {
                        "id": f"{document_id}-chunk-{len(chunks)}",
                        "document_id": document_id,
                        "text": current_chunk_text,
                        "start_pos": current_start,
                        "end_pos": current_start + len(current_chunk_text),
                        "metadata": {
                            "filepath": filepath,
                            "filename": os.path.basename(filepath),
                            "document_title": doc_metadata.get("title", ""),
                            "section_title": section["title"],
                            "section_level": section["level"],
                            "section_path": section_path,
                            "content_type": "partial_section",
                            "language": doc_metadata.get("language", ""),
                            "document_keywords": doc_metadata.get("keywords", []),
                            "creation_date": doc_metadata.get("creation_date"),
                            "paragraph_count": len(current_paragraphs),
                            "is_complete_section": (current_start == section["start_pos"] and
                                                    current_start + len(current_chunk_text) == section["end_pos"])
                        }
                    }
                    chunks.append(chunk)

        return chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Divise un texte en paragraphes"""
        # Normaliser les sauts de ligne
        normalized_text = re.sub(r'\r\n', '\n', text)

        # Diviser par lignes vides (paragraphes)
        paragraphs = re.split(r'\n\s*\n', normalized_text)

        # Ajouter le saut de paragraphe à la fin de chaque paragraphe
        result = []
        for p in paragraphs:
            if p.strip():  # Ignorer les paragraphes vides
                result.append(p.strip() + "\n\n")

        return result

    async def _enhance_chunks_with_semantics(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrichit les chunks avec des métadonnées sémantiques"""
        enhanced_chunks = []

        for chunk in chunks:
            # Copier le chunk original
            enhanced_chunk = chunk.copy()
            chunk_text = chunk["text"]

            # Extraire les entités nommées si spaCy est disponible
            entities = []
            if self.nlp:
                doc = self.nlp(chunk_text[:min(len(chunk_text), 10000)])  # Limite pour performance
                for ent in doc.ents:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })

            # Calculer la densité informationnelle du chunk
            info_density = self._calculate_info_density(chunk_text)

            # Extraire les références et citations
            references = self._extract_references(chunk_text)

            # Mettre à jour les métadonnées
            enhanced_chunk["metadata"].update({
                "entities": entities,
                "info_density": info_density,
                "references": references,
                "chunk_keywords": self._extract_keywords(chunk_text, max_keywords=5),
            })

            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks

    def _detect_language(self, text: str) -> str:
        """Détecte la langue du texte"""
        # Méthode simple basée sur les mots fréquents
        french_stopwords = {'le', 'la', 'les', 'de', 'du', 'des', 'et', 'en', 'un', 'une', 'est', 'pour', 'dans', 'par',
                            'sur', 'au', 'que', 'qui', 'ce', 'cette'}
        english_stopwords = {'the', 'of', 'and', 'to', 'in', 'a', 'is', 'for', 'on', 'with', 'by', 'that', 'this', 'as',
                             'it', 'from', 'are', 'be', 'at'}

        # Extraire les mots
        words = re.findall(r'\b\w+\b', text.lower())

        # Compter les occurrences
        french_count = sum(1 for word in words if word in french_stopwords)
        english_count = sum(1 for word in words if word in english_stopwords)

        if french_count > english_count:
            return 'fr'
        elif english_count > french_count:
            return 'en'
        else:
            return 'fr'  # Par défaut

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extrait les mots-clés importants du texte"""
        # Stopwords pour le français
        stopwords = {'le', 'la', 'les', 'de', 'du', 'des', 'et', 'en', 'un', 'une', 'est', 'pour', 'dans', 'par', 'sur',
                     'au', 'que', 'qui', 'ce', 'cette', 'ces', 'il', 'elle', 'ils', 'elles', 'on', 'nous', 'vous', 'je',
                     'tu'}

        # Extraire les mots
        words = re.findall(r'\b\w{3,}\b', text.lower())

        # Compter les occurrences
        word_count = {}
        for word in words:
            if word not in stopwords:
                word_count[word] = word_count.get(word, 0) + 1

        # Trier par fréquence
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

        # Retourner les mots-clés les plus fréquents
        return [word for word, count in sorted_words[:max_keywords]]

    def _calculate_info_density(self, text: str) -> float:
        """Calcule un score de densité informationnelle pour le texte"""
        # Longueur du texte
        text_length = len(text)
        if text_length == 0:
            return 0.0

        # Nombre de termes significatifs
        significant_terms = len(self._extract_keywords(text, max_keywords=100))

        # Présence d'entités nommées
        entities_count = 0
        if self.nlp:
            doc = self.nlp(text[:min(len(text), 5000)])  # Limiter pour performance
            entities_count = len(doc.ents)

        # Longueur moyenne des mots (indice de complexité)
        words = re.findall(r'\b\w+\b', text)
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)

        # Calculer un score composite
        density = (
                (significant_terms / max(len(words), 1) * 0.5) +  # Proportion de termes significatifs
                (min(entities_count / 20, 1.0) * 0.3) +  # Présence d'entités (max 20)
                (min((avg_word_length - 3) / 3, 1.0) * 0.2)  # Complexité des mots
        )

        return min(max(density, 0.0), 1.0)  # Normaliser entre 0 et 1

    def _extract_references(self, text: str) -> List[Dict[str, Any]]:
        """Extrait les références et citations du texte"""
        references = []

        # Rechercher les citations entre guillemets
        quote_pattern = r'[«"]([^«"]+)[»"]'
        for match in re.finditer(quote_pattern, text):
            references.append({
                "type": "citation",
                "text": match.group(1),
                "start": match.start(),
                "end": match.end()
            })

        # Rechercher les références bibliographiques
        biblio_pattern = r'\[([^\]]+)\]'
        for match in re.finditer(biblio_pattern, text):
            references.append({
                "type": "reference",
                "text": match.group(1),
                "start": match.start(),
                "end": match.end()
            })

        # Rechercher les liens
        url_pattern = r'https?://[^\s)>]+'
        for match in re.finditer(url_pattern, text):
            references.append({
                "type": "url",
                "text": match.group(0),
                "start": match.start(),
                "end": match.end()
            })

        return references