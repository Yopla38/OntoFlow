"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# src/components/knowledge_base.py
import logging
from datetime import datetime
from typing import Dict, List, Any
from agent.src.types.interfaces import MemoryProvider


class KnowledgeBase:
    def __init__(self, memory_provider: MemoryProvider):
        self.memory_provider = memory_provider
        self.categories: Dict[str, List[str]] = {}

    async def add_knowledge(self, category: str, content: str):
        """Ajoute une connaissance à la base"""
        logging.info(f"Adding knowledge for category: {category}")

        await self.memory_provider.add({
            "category": category,
            "content": content,
            "timestamp": datetime.now()
        })
        logging.info("Knowledge added to memory provider")

    async def query_knowledge(self, query: List[str], category: str) -> List[Dict[str, Any]]:
        """
        Récupère les connaissances en suivant le chemin Pydantic spécifié.
        """
        logging.info(f"Querying knowledge for category: {category}, path: {query}")

        try:
            # Récupérer les données stockées pour cette catégorie
            results = await self.memory_provider.search(category)

            if not results:
                return []

            # Pour chaque résultat trouvé
            extracted_data = []
            for result in results:
                content = result.get('content', {})
                logging.info(f"Processing content...")

                # Suivre le chemin Pydantic pour extraire les données
                current_data = content
                for key in query:
                    if isinstance(current_data, dict) and key in current_data:
                        current_data = current_data[key]
                    else:
                        current_data = None
                        break

                if current_data is not None:
                    extracted_data.append(current_data)

            logging.info(f"Extracted data...")
            return extracted_data
        except Exception as e:
            logging.info(f"No data for {query}...")
            return [{}]

    async def store_pydantic_result(self, role: str, result: Any):
        """Stocke le modèle Pydantic complet"""
        logging.info(f"Storing pydantic result for role {role}")
        #logging.info(f"Result to store: {result}")

        if not hasattr(result, 'model_dump'):
            logging.warning("Result is not a Pydantic model!")
            return

        await self.add_knowledge(
            category=role,
            content=result.model_dump()
        )
        logging.info("Knowledge added successfully")

    def _flatten_dict(self, d: Dict[str, Any], parent_path: List[str] = None) -> List[tuple[List[str], Any]]:
        """Aplatit un dictionnaire en conservant les chemins d'accès"""
        items = []
        parent_path = parent_path or []

        for key, value in d.items():
            path = parent_path + [key]
            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, path))
            else:
                items.append((path, value))

        return items
