"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

PRONOUNS: Dict[str, Dict[str, List[str]]] = {
    "fr": {
        "personal": ["je","tu","il","elle","on","nous","vous","ils","elles"],
        "object": [
            "me","te","se","m'","t'","s'","l'","lui","leur","y","en","le","la","les"
        ],
        "demonstrative": ["celui","celle","ceux","celles","ceci","cela","ça","celui-ci","celle-ci"],
    },
    "en": {
        "personal": ["I","you","he","she","it","we","they"],
        "object": ["me","him","her","us","them","it"],
        "reflexive": ["myself","himself","herself","itself","ourselves","themselves"],
        "possessive": ["mine","yours","his","hers","ours","theirs"],
        "demonstrative": ["this","that","these","those"],
    },
    "es": {
        "personal": ["yo","tú","él","ella","nosotros","vosotros","ellos","ellas"],
        "object": ["me","te","lo","la","los","las","le"],
        "reflexive": ["se"],
        "demonstrative": ["este","esta","estos","estas","ese","esa","esos","esas","aquel","aquella","aquellos","aquellas"],
    },
}


@dataclass
class Pronoun:
    text: str
    position: int
    end_position: int
    language: str
    pos: Optional[str] = None


class MultilingualPronounDetector:
    def __init__(
        self,
        primary_language: str = "fr",
        include_categories: Optional[Dict[str, Sequence[str]]] = None,
        include_relative: bool = False,
    ) -> None:
        self.primary_language = primary_language
        self.include_categories = include_categories or {
            "fr": ("personal", "object", "demonstrative"),
            "en": ("personal", "object", "reflexive", "possessive", "demonstrative"),
            "es": ("personal", "object", "reflexive", "demonstrative"),
        }
        self.include_relative = include_relative
        self.patterns: Dict[str, re.Pattern] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        for lang, cats in PRONOUNS.items():
            allowed = set(self.include_categories.get(lang, cats.keys()))
            forms: List[str] = []
            for c, lst in cats.items():
                if c in allowed:
                    forms.extend(lst)
            if lang == "fr":
                for base in {"m","t","s","l","c"}:
                    forms += [f"{base}'", f"{base}’"]
            forms = sorted(set(forms), key=len, reverse=True)
            if lang == "fr":
                pat = r"(?<!\w)(?:" + "|".join(re.escape(p) for p in forms) + r")(?!\w)"
            else:
                pat = r"\b(?:" + "|".join(re.escape(p) for p in forms) + r")\b"
            self.patterns[lang] = re.compile(pat, re.IGNORECASE | re.UNICODE)

    def find_pronouns(self, text: str, nlp=None) -> List[Pronoun]:
        pronouns: List[Pronoun] = []
        languages = [self.primary_language] + [l for l in self.patterns if l != self.primary_language]
        candidates: List[Pronoun] = []
        for lang in languages:
            pat = self.patterns.get(lang)
            if not pat:
                continue
            for m in pat.finditer(text):
                # ne pas dupliquer la même position (langues multiples)
                if any(p.position == m.start() for p in candidates):
                    continue
                candidates.append(Pronoun(m.group(), m.start(), m.end(), lang))

        # Filtrage POS avec spaCy (critique pour FR: exclut les DET 'le/la/les')
        if nlp is not None:
            doc = nlp(text)
            kept: List[Pronoun] = []
            for p in candidates:
                span = doc.char_span(p.position, p.end_position, alignment_mode="expand")
                if span is None:
                    continue
                # Garder si au moins un token de la span est un PRON
                if any(t.pos_ == "PRON" for t in span):
                    kept.append(p)
            pronouns = kept
        else:
            pronouns = candidates

        pronouns.sort(key=lambda p: p.position)
        return pronouns