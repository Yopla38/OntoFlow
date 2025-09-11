"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

import pytest

from document_analysis.src.pronoun_resolver import PronounResolver, ResolutionConfig, PydanticLLMProvider


class DummyLLM:
    def init(self, mapping):
        self.mapping = mapping  # {pos: replacement}
    def call(self, messages, pydantic_model, **kwargs):
        # Extract positions mentioned in the prompt (simple heuristic for test)
        text = "\n".join(messages)
        reps = []
        for pos, val in self.mapping.items():
            reps.append({"pos": pos, "text": val, "confidence": 1.0})
        return pydantic_model(replacements=reps)

@pytest.mark.parametrize("lang,text,mapping,expected", [
    ("fr", "Pierre cuisine. Il l'aide.", {15: "Marie", 18: "le gâteau"}, "Pierre cuisine. Marie le gâteau aide."),
    ("en", "John reads. He likes it.", {12: "John", 22: "the book"}, "John reads. John likes the book."),
    ("es", "Ana cocina. Ella lo ama.", {12: "Ana", 18: "el postre"}, "Ana cocina. Ana el postre ama."),
])
def test_basic_resolution(lang, text, mapping, expected):
    backend = DummyLLM(mapping)
    provider = PydanticLLMProvider(backend)
    resolver = PronounResolver(provider, ResolutionConfig(lang=lang, batch_size=10))
    out = resolver.resolve(text)
    assert out == expected

def test_no_change_when_empty_replacements():
    backend = lambda messages, pydantic_model, **kw: pydantic_model(replacements=[])
    provider = PydanticLLMProvider(backend)
    resolver = PronounResolver(provider, ResolutionConfig(lang="fr"))
    text = "Il pleut."
    assert resolver.resolve(text) == text