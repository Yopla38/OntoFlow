"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

from document_analysis.src.pronoun_resolver.utils import adjust_casing


def test_adjust_casing_upper():
    assert adjust_casing("IL", "marie") == "MARIE"


def test_adjust_casing_capitalized():
    assert adjust_casing("Il", "marie") == "Marie"


def test_adjust_casing_lower():
    assert adjust_casing("il", "Marie") == "Marie"