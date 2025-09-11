"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """


from future import annotations
import argparse
import sys
from .resolver import PronounResolver, ResolutionConfig
from .llm import PydanticLLMProvider
from .models import ResolutionOutput

# Example backend adapter expecting a function fn(messages, pydantic_model, **kwargs)

def main():
    parser = argparse.ArgumentParser(description="Resolve pronouns in text using LLM + spaCy")
    parser.add_argument("text", nargs="?", help="Input text; if omitted, read from stdin")
    parser.add_argument("--lang", default="fr", help="Primary language (fr/en/es)")
    args = parser.parse_args()

    text = args.text or sys.stdin.read()

    # Placeholder backend: user should inject real backend here.
    def backend(messages, pydantic_model, **kwargs):
        # This mock returns no replacements; replace with your real provider
        return pydantic_model(replacements=[])

    resolver = PronounResolver(PydanticLLMProvider(backend), ResolutionConfig(lang=args.lang))
    print(resolver.resolve(text))

if __name__ == "main":
    main()