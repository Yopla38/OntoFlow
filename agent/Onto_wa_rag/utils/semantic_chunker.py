"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# utils/semantic_chunker.py
import os
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import hashlib
from datetime import datetime


@dataclass
class DocumentSection:
    """Représente une section structurée du document"""
    level: int  # Niveau hiérarchique (1 = titre principal, 2 = sous-titre, etc.)
    title: str
    content: str
    start_pos: int
    end_pos: int
    section_type: str  # 'header', 'paragraph', 'list', 'table', 'code'
    metadata: Dict[str, Any]


class SemanticChunker:
    """Chunker intelligent qui préserve la structure sémantique des documents"""

    def __init__(
            self,
            min_chunk_size: int = 200,
            max_chunk_size: int = 1000,
            overlap_sentences: int = 2,  # Overlap en nombre de phrases
            respect_boundaries: bool = True  # Respecter les limites de sections
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences
        self.respect_boundaries = respect_boundaries

        # Patterns pour détecter la structure
        self.header_patterns = [
            # Markdown headers
            (r'^#{1,6}\s+(.+)$', 'markdown'),
            # Numbered sections
            (r'^(\d+\.)+\s*(.+)$', 'numbered'),
            # Capital letters sections
            (r'^([A-Z][A-Z\s]+):?\s*$', 'capital'),
            # Roman numerals
            (r'^[IVXLCDM]+\.\s+(.+)$', 'roman'),
        ]

    def extract_document_structure(self, text: str) -> List[DocumentSection]:
        """Extrait la structure hiérarchique du document"""
        sections = []
        lines = text.split('\n')
        current_pos = 0

        i = 0
        while i < len(lines):
            line = lines[i]
            line_start = current_pos

            # Détecter les headers
            header_info = self._detect_header(line)
            if header_info:
                level, title, header_type = header_info

                # Collecter le contenu de la section
                content_lines = []
                j = i + 1

                # Continuer jusqu'au prochain header de niveau égal ou supérieur
                while j < len(lines):
                    next_header = self._detect_header(lines[j])
                    if next_header and next_header[0] <= level:
                        break
                    content_lines.append(lines[j])
                    j += 1

                content = '\n'.join(content_lines).strip()
                section = DocumentSection(
                    level=level,
                    title=title,
                    content=content,
                    start_pos=line_start,
                    end_pos=current_pos + len(line) + len(content),
                    section_type='header',
                    metadata={'header_type': header_type}
                )
                sections.append(section)

                i = j - 1  # On reprendra au prochain header

            else:
                # Détecter les autres types de contenu
                if self._is_list_item(line):
                    # Collecter tous les items de la liste
                    list_items = [line]
                    j = i + 1
                    while j < len(lines) and self._is_list_item(lines[j]):
                        list_items.append(lines[j])
                        j += 1

                    section = DocumentSection(
                        level=99,  # Pas de niveau hiérarchique pour les listes
                        title="List",
                        content='\n'.join(list_items),
                        start_pos=line_start,
                        end_pos=current_pos + sum(len(item) + 1 for item in list_items),
                        section_type='list',
                        metadata={'item_count': len(list_items)}
                    )
                    sections.append(section)
                    i = j - 1

                elif self._is_code_block(line):
                    # Gérer les blocs de code
                    code_lines = [line]
                    j = i + 1
                    in_code = True
                    while j < len(lines) and in_code:
                        code_lines.append(lines[j])
                        if lines[j].strip() == '```':
                            in_code = False
                        j += 1

                    section = DocumentSection(
                        level=99,
                        title="Code Block",
                        content='\n'.join(code_lines),
                        start_pos=line_start,
                        end_pos=current_pos + sum(len(l) + 1 for l in code_lines),
                        section_type='code',
                        metadata={'language': self._detect_code_language(code_lines[0])}
                    )
                    sections.append(section)
                    i = j - 1

                elif line.strip():  # Paragraphe normal
                    # Collecter le paragraphe complet
                    para_lines = [line]
                    j = i + 1
                    while j < len(lines) and lines[j].strip() and not self._is_special_line(lines[j]):
                        para_lines.append(lines[j])
                        j += 1

                    section = DocumentSection(
                        level=99,
                        title="Paragraph",
                        content=' '.join(para_lines),
                        start_pos=line_start,
                        end_pos=current_pos + sum(len(l) + 1 for l in para_lines),
                        section_type='paragraph',
                        metadata={}
                    )
                    sections.append(section)
                    i = j - 1

            current_pos += len(line) + 1  # +1 pour le \n
            i += 1

        return sections

    def create_semantic_chunks(
            self,
            text: str,
            document_id: str,
            filepath: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Crée des chunks en respectant la structure sémantique du document"""

        # Extraire la structure
        sections = self.extract_document_structure(text)

        if not sections:
            # Fallback sur l'ancien système si pas de structure détectée
            return self._create_fallback_chunks(text, document_id, filepath, metadata)

        chunks = []
        chunk_index = 0

        # Grouper les sections en chunks cohérents
        current_chunk_sections = []
        current_chunk_size = 0

        for i, section in enumerate(sections):
            section_size = len(section.content)

            # Décider si on doit créer un nouveau chunk
            should_split = False

            if self.respect_boundaries and section.section_type == 'header' and section.level <= 2:
                # Toujours commencer un nouveau chunk pour les headers importants
                should_split = True
            elif current_chunk_size + section_size > self.max_chunk_size and current_chunk_sections:
                # Taille maximale atteinte
                should_split = True

            if should_split and current_chunk_sections:
                # Créer le chunk avec les sections accumulées
                chunk = self._create_chunk_from_sections(
                    current_chunk_sections,
                    chunk_index,
                    document_id,
                    filepath,
                    metadata
                )
                chunks.append(chunk)
                chunk_index += 1

                # Gérer l'overlap si nécessaire
                if self.overlap_sentences > 0:
                    current_chunk_sections = self._get_overlap_sections(current_chunk_sections)
                    current_chunk_size = sum(len(s.content) for s in current_chunk_sections)
                else:
                    current_chunk_sections = []
                    current_chunk_size = 0

            # Ajouter la section courante
            current_chunk_sections.append(section)
            current_chunk_size += section_size

            # Si la section seule est trop grande, la découper
            if section_size > self.max_chunk_size:
                # Découper la section en sous-chunks
                sub_chunks = self._split_large_section(
                    section, chunk_index, document_id, filepath, metadata
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
                current_chunk_sections = []
                current_chunk_size = 0

        # Créer le dernier chunk
        if current_chunk_sections:
            chunk = self._create_chunk_from_sections(
                current_chunk_sections,
                chunk_index,
                document_id,
                filepath,
                metadata
            )
            chunks.append(chunk)

        return chunks


    def _create_chunk_from_sections(
            self,
            sections: List[DocumentSection],
            chunk_index: int,
            document_id: str,
            filepath: str,
            base_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Crée un chunk à partir d'une liste de sections"""

        # Construire le texte du chunk avec structure préservée
        chunk_parts = []
        chunk_metadata = base_metadata or {}

        # Hiérarchie des sections dans le chunk
        section_hierarchy = []

        for section in sections:
            if section.section_type == 'header':
                # Préserver la hiérarchie
                hierarchy_entry = {
                    'level': section.level,
                    'title': section.title,
                    'type': section.section_type
                }
                section_hierarchy.append(hierarchy_entry)

                # Ajouter le header au texte
                if section.level == 1:
                    chunk_parts.append(f"# {section.title}")
                elif section.level == 2:
                    chunk_parts.append(f"## {section.title}")
                else:
                    chunk_parts.append(f"### {section.title}")

            # Ajouter le contenu
            if section.content:
                chunk_parts.append(section.content)

        chunk_text = '\n\n'.join(chunk_parts)

        # Calculer un hash du contenu pour la déduplication
        content_hash = hashlib.md5(chunk_text.encode()).hexdigest()

        # Enrichir les métadonnées
        chunk_metadata.update({
            'section_hierarchy': section_hierarchy,
            'section_types': list(set(s.section_type for s in sections)),
            'section_count': len(sections),
            'has_headers': any(s.section_type == 'header' for s in sections),
            'has_lists': any(s.section_type == 'list' for s in sections),
            'has_code': any(s.section_type == 'code' for s in sections),
            'content_hash': content_hash,
            'chunk_method': 'semantic',
            'created_at': datetime.now().isoformat()
        })

        # Position dans le document original
        start_pos = sections[0].start_pos if sections else 0
        end_pos = sections[-1].end_pos if sections else len(chunk_text)

        return {
            "id": f"{document_id}-chunk-{chunk_index}",
            "document_id": document_id,
            "text": chunk_text,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "metadata": {
                "filepath": filepath,
                "filename": os.path.basename(filepath),
                **chunk_metadata
            }
        }

    def _detect_header(self, line: str) -> Optional[Tuple[int, str, str]]:
        """Détecte si une ligne est un header et retourne (niveau, titre, type)"""
        line = line.strip()

        # Headers Markdown
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            title = line.lstrip('#').strip()
            if title:
                return (level, title, 'markdown')

        # Headers numérotés (1. 1.1. etc.)
        match = re.match(r'^((?:\d+\.)+)\s*(.+)$', line)
        if match:
            level = match.group(1).count('.')
            title = match.group(2).strip()
            return (level, title, 'numbered')

        # Headers en majuscules
        if line.isupper() and len(line) > 3 and not line.endswith('.'):
            return (1, line, 'capital')

        return None

    def _is_list_item(self, line: str) -> bool:
        """Détecte si une ligne est un item de liste"""
        patterns = [
            r'^\s*[-*+]\s+',  # Listes non ordonnées
            r'^\s*\d+\.\s+',  # Listes ordonnées
            r'^\s*[a-z]\)\s+',  # Listes alphabétiques
        ]
        return any(re.match(pattern, line) for pattern in patterns)

    def _is_code_block(self, line: str) -> bool:
        """Détecte le début d'un bloc de code"""
        return line.strip().startswith('```')

    def _is_special_line(self, line: str) -> bool:
        """Détecte si une ligne est spéciale (header, liste, etc.)"""
        return (self._detect_header(line) is not None or
                self._is_list_item(line) or
                self._is_code_block(line))

    def _get_overlap_sections(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        """Extrait les sections pour l'overlap"""
        # Prendre les dernières phrases des dernières sections
        overlap_sections = []
        sentences_count = 0

        for section in reversed(sections):
            if section.section_type == 'paragraph':
                sentences = self._split_into_sentences(section.content)
                if sentences:
                    # Prendre les dernières phrases
                    needed = self.overlap_sentences - sentences_count
                    if needed > 0:
                        overlap_content = ' '.join(sentences[-needed:])
                        overlap_section = DocumentSection(
                            level=section.level,
                            title=f"[Overlap] {section.title}",
                            content=overlap_content,
                            start_pos=section.start_pos,
                            end_pos=section.end_pos,
                            section_type='overlap',
                            metadata={'original_type': section.section_type}
                        )
                        overlap_sections.insert(0, overlap_section)
                        sentences_count += min(needed, len(sentences))

                        if sentences_count >= self.overlap_sentences:
                            break

        return overlap_sections

    def _split_into_sentences(self, text: str) -> List[str]:
        """Divise un texte en phrases"""
        # Pattern amélioré pour la détection de phrases
        sentence_endings = r'[.!?]'
        pattern = rf'(?<={sentence_endings})\s+(?=[A-Z])'

        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_large_section(
            self,
            section: DocumentSection,
            start_chunk_index: int,
            document_id: str,
            filepath: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Découpe une section trop grande en plusieurs chunks"""
        chunks = []

        if section.section_type == 'paragraph':
            # Découper par phrases
            sentences = self._split_into_sentences(section.content)
            current_chunk_sentences = []
            current_size = 0

            for sentence in sentences:
                sentence_size = len(sentence)

                if current_size + sentence_size > self.max_chunk_size and current_chunk_sentences:
                    # Créer un chunk
                    chunk_text = ' '.join(current_chunk_sentences)
                    chunk = {
                        "id": f"{document_id}-chunk-{start_chunk_index + len(chunks)}",
                        "document_id": document_id,
                        "text": chunk_text,
                        "start_pos": section.start_pos,
                        "end_pos": section.start_pos + len(chunk_text),
                        "metadata": {
                            "filepath": filepath,
                            "filename": os.path.basename(filepath),
                            "section_title": section.title,
                            "section_type": section.section_type,
                            "is_partial_section": True,
                            **(metadata or {})
                        }
                    }
                    chunks.append(chunk)

                    # Overlap
                    if self.overlap_sentences > 0:
                        current_chunk_sentences = current_chunk_sentences[-self.overlap_sentences:]
                        current_size = sum(len(s) + 1 for s in current_chunk_sentences)
                    else:
                        current_chunk_sentences = []
                        current_size = 0

                current_chunk_sentences.append(sentence)
                current_size += sentence_size + 1

            # Dernier chunk
            if current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences)
                chunk = {
                    "id": f"{document_id}-chunk-{start_chunk_index + len(chunks)}",
                    "document_id": document_id,
                    "text": chunk_text,
                    "start_pos": section.start_pos,
                    "end_pos": section.end_pos,
                    "metadata": {
                        "filepath": filepath,
                        "filename": os.path.basename(filepath),
                        "section_title": section.title,
                        "section_type": section.section_type,
                        "is_partial_section": True,
                        **(metadata or {})
                    }
                }
                chunks.append(chunk)

        else:
            # Pour les autres types (code, listes), découper mécaniquement
            # mais en préservant le type
            text = section.content
            for i in range(0, len(text), self.max_chunk_size - 200):  # 200 chars d'overlap
                chunk_text = text[i:i + self.max_chunk_size]
                chunk = {
                    "id": f"{document_id}-chunk-{start_chunk_index + len(chunks)}",
                    "document_id": document_id,
                    "text": chunk_text,
                    "start_pos": section.start_pos + i,
                    "end_pos": section.start_pos + i + len(chunk_text),
                    "metadata": {
                        "filepath": filepath,
                        "filename": os.path.basename(filepath),
                        "section_title": section.title,
                        "section_type": section.section_type,
                        "is_partial_section": True,
                        **(metadata or {})
                    }
                }
                chunks.append(chunk)

        return chunks

    def _create_fallback_chunks(
            self,
            text: str,
            document_id: str,
            filepath: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Fallback vers l'ancienne méthode si pas de structure détectée"""
        # Utiliser l'ancienne méthode mais avec des métadonnées enrichies
        chunks = []
        sentences = self._split_into_sentences(text)

        current_chunk_sentences = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size > self.max_chunk_size and current_chunk_sentences:
                # Créer un chunk
                chunk_text = ' '.join(current_chunk_sentences)
                chunk = {
                    "id": f"{document_id}-chunk-{len(chunks)}",
                    "document_id": document_id,
                    "text": chunk_text,
                    "start_pos": text.find(current_chunk_sentences[0]),
                    "end_pos": text.find(current_chunk_sentences[-1]) + len(current_chunk_sentences[-1]),
                    "metadata": {
                        "filepath": filepath,
                        "filename": os.path.basename(filepath),
                        "chunk_method": "fallback",
                        "no_structure_detected": True,
                        **(metadata or {})
                    }
                }
                chunks.append(chunk)

                # Overlap
                if self.overlap_sentences > 0:
                    current_chunk_sentences = current_chunk_sentences[-self.overlap_sentences:]
                    current_size = sum(len(s) + 1 for s in current_chunk_sentences)
                else:
                    current_chunk_sentences = []
                    current_size = 0

            current_chunk_sentences.append(sentence)
            current_size += sentence_size + 1

        # Dernier chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunk = {
                "id": f"{document_id}-chunk-{len(chunks)}",
                "document_id": document_id,
                "text": chunk_text,
                "start_pos": text.find(current_chunk_sentences[0]),
                "end_pos": text.find(current_chunk_sentences[-1]) + len(current_chunk_sentences[-1]),
                "metadata": {
                    "filepath": filepath,
                    "filename": os.path.basename(filepath),
                    "chunk_method": "fallback",
                    "no_structure_detected": True,
                    **(metadata or {})
                }
            }
            chunks.append(chunk)

        return chunks

    def _detect_code_language(self, line: str) -> Optional[str]:
        """Détecte le langage d'un bloc de code"""
        match = re.match(r'^```(\w+)', line)
        return match.group(1) if match else None