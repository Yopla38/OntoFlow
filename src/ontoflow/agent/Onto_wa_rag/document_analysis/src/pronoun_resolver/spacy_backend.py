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
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

try:
    import spacy  # type: ignore
    from spacy.cli import download as spacy_download  # type: ignore
except Exception:  # pragma: no cover
    spacy = None
    spacy_download = None

_LANG_MODEL = {
    "fr": ["fr_core_news_md", "fr_core_news_sm"],
    "en": ["en_core_web_md", "en_core_web_sm"],
    "es": ["es_core_news_md", "es_core_news_sm"],
}

_CACHE = {}


def ensure_model(lang: str):
    if spacy is None:
        logger.warning("spaCy not installed; NER/POS disabled.")
        return None
    if lang in _CACHE:
        return _CACHE[lang]

    models = _LANG_MODEL.get(lang, [])
    for name in models:
        try:
            nlp = spacy.load(name)
            _CACHE[lang] = nlp
            logger.info("Loaded spaCy model: %s", name)
            return nlp
        except Exception:
            continue

    # Try to download the first lightweight model
    if spacy_download and models:
        try:
            logger.info("Downloading spaCy model: %s", models[-1])
            spacy_download(models[-1])
            nlp = spacy.load(models[-1])
            _CACHE[lang] = nlp
            logger.info("Loaded spaCy model after download: %s", models[-1])
            return nlp
        except Exception as e:
            logger.warning("Could not download/load spaCy model for %s: %s", lang, e)

    logger.warning("No spaCy model available for %s; NER/POS disabled.")
    _CACHE[lang] = None
    return None


def extract_context(text: str, lang: str) -> Dict[str, List[str]]:
    nlp = ensure_model(lang)
    if not nlp:
        return {"entities": [], "pronouns": []}
    doc = nlp(text)
    ents = [e.text for e in doc.ents if e.label in {"PER", "ORG", "LOC", "GPE", "MISC"}]
    pro = [t.text for t in doc if t.pos_ == "PRON"]
    # dedupe while preserving order
    seen = set()
    ents = [x for x in ents if not (x in seen or seen.add(x))]
    seen.clear()
    pro = [x for x in pro if not (x in seen or seen.add(x))]
    return {"entities": ents[:30], "pronouns": pro[:50]}
