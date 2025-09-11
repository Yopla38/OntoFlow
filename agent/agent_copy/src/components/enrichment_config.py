"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

from typing import List

from agent.src.types.interfaces import EnrichmentPath


class EnrichmentConfig:
    """Configuration générique des enrichissements par rôle"""

    @staticmethod
    def get_role_enrichments(role_name: str) -> List[EnrichmentPath]:
        """Retourne les enrichissements nécessaires pour un rôle"""
        return ROLE_ENRICHMENTS.get(role_name, [])


# Cette configuration peut être chargée depuis un fichier externe
ROLE_ENRICHMENTS = {
    "ingenieur": [
        # Contexte du projet
        EnrichmentPath(
            source_role="architecte",
            pydantic_path=["analysis", "overview"],
            category="project_context",
            description="Contexte du projet"
        ),
        # Stack technique
        EnrichmentPath(
            source_role="architecte",
            pydantic_path=["analysis", "technical_stack"],
            category="technical_stack",
            description="Stack technique à utiliser"
        ),
        # Dépendances Python
        EnrichmentPath(
            source_role="architecte",
            pydantic_path=["dependencies", "python_packages"],
            category="dependencies",
            description="Dépendances Python requises"
        )
    ],
    "frontend_dev": [
        # Contexte du projet
        EnrichmentPath(
            source_role="architecte",
            pydantic_path=["analysis", "overview"],
            category="project_context",
            description="Contexte du projet"
        ),
        # Stack technique frontend
        EnrichmentPath(
            source_role="architecte",
            pydantic_path=["analysis", "technical_stack"],
            category="technical_stack",
            description="Stack technique à utiliser"
        )
    ],
    "backend_dev": [
        # Contexte du projet
        EnrichmentPath(
            source_role="architecte",
            pydantic_path=["analysis", "overview"],
            category="project_context",
            description="Contexte du projet"
        ),
        # Stack technique backend
        EnrichmentPath(
            source_role="architecte",
            pydantic_path=["analysis", "technical_stack"],
            category="technical_stack",
            description="Stack technique à utiliser"
        )
    ],
    "database_dev": [
        # Contexte du projet
        EnrichmentPath(
            source_role="architecte",
            pydantic_path=["analysis", "overview"],
            category="project_context",
            description="Contexte du projet"
        ),
        # Stack technique base de données
        EnrichmentPath(
            source_role="architecte",
            pydantic_path=["analysis", "technical_stack"],
            category="technical_stack",
            description="Stack technique à utiliser"
        )
    ],
    "designer_interface": [
        # Contexte du projet
        EnrichmentPath(
            source_role="architecte",
            pydantic_path=["analysis", "overview"],
            category="project_context",
            description="Contexte du projet"
        ),
        # Stack technique UI
        EnrichmentPath(
            source_role="architecte",
            pydantic_path=["analysis", "technical_stack"],
            category="technical_stack",
            description="Stack technique à utiliser"
        )
    ],
    "codeur": [
        # Contexte du projet
        EnrichmentPath(
            source_role="architecte",
            pydantic_path=["analysis", "overview"],
            category="project_context",
            description="Contexte du projet"
        ),
        # Stack technique
        EnrichmentPath(
            source_role="architecte",
            pydantic_path=["analysis", "technical_stack"],
            category="technical_stack",
            description="Stack technique à utiliser"
        ),
        # Dépendances Python
        EnrichmentPath(
            source_role="architecte",
            pydantic_path=["dependencies", "python_packages"],
            category="dependencies",
            description="Dépendances Python requises"
        )
    ],
    "developpeur": [
            # Contexte du projet
            EnrichmentPath(source_role="generateur_idees",
                pydantic_path=[""],
                category="idea",
                description="Nouvelle idée de développement")
        ],
    "developpeur_old": [
            # Contexte du projet
            EnrichmentPath(
                source_role="generateur_idees",
                pydantic_path=["improvement_proposal", "idea"],
                category="idea",
                description="Nouvelle idée de développement"
            ),
            EnrichmentPath(
                source_role="convertisseur_dial2tec",
                pydantic_path=["improvement_proposal", "idea"],
                category="idea",
                description="Convertion technique d'un dialogue"
            )
        ]
}
