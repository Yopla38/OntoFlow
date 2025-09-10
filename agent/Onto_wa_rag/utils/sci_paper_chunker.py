"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

import re
from dataclasses import dataclass
from .semantic_chunker import SemanticChunker


@dataclass
class PaperSection:
    level: int          # 1 = #, 2 = ## ...
    title: str
    start: int
    end: int


class SciPaperSemanticChunker(SemanticChunker):
    headings_re = re.compile(r'^(#{1,6})\s+(.*)')

    def _parse_sections(self, lines):
        stack = []
        sections = []
        for idx, line in enumerate(lines):
            m = self.headings_re.match(line)
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                while stack and stack[-1].level >= level:
                    stack[-1].end = idx-1
                    stack.pop()
                sec = PaperSection(level, title, idx, len(lines)-1)
                sections.append(sec)
                stack.append(sec)
        for s in stack:   # terminer les fins
            s.end = len(lines)-1
        return sections

    async def create_paper_chunks(self, md:str, doc_id:str,
                                  filepath:str, metadata=None):
        lines = md.splitlines()
        sections = self._parse_sections(lines)
        chunks=[]
        idx=0
        for s in sections:
            text = "\n".join(lines[s.start:s.end+1])
            if len(text)<self.min_chunk_size:
                continue
            chunk = {
              "id": f"{doc_id}-chunk-{idx}",
              "document_id": doc_id,
              "text": text,
              "start_pos": s.start,
              "end_pos": s.end,
              "metadata": {
                   **(metadata or {}),
                   "section_title": s.title,
                   "section_level": s.level,
                   "filepath": filepath,
                   "filename": metadata.get("filename","")
              }
            }
            chunks.append(chunk)
            idx += 1
        return chunks
    