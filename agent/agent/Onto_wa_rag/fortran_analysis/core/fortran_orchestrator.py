"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

from fortran_analysis.core.entity_manager import EntityManager
from fortran_analysis.core.hybrid_fortran_parser import HybridFortranParser
import logging

logger = logging.getLogger(__name__)


class FortranAnalysisOrchestrator:
    """Orchestrateur qui fait le lien entre Parser et EntityManager"""

    def __init__(self, document_store, ontology_manager=None):
        self.parser = HybridFortranParser()
        self.entity_manager = EntityManager(document_store)
        self.ontology_manager = ontology_manager
        self.document_store = document_store
        self._initialized = False

    async def initialize(self):
        """Initialise l'orchestrateur"""
        if not self._initialized:
            await self.entity_manager.initialize()
            self._initialized = True
            logger.info("🔧 Orchestrateur Fortran initialisé")

    async def analyze_document(self, document_id: str, code: str, filename: str):
        """Analyse complète : parsing + indexation"""
        if not self._initialized:
            await self.initialize()

        logger.info(f"🔬 Analyse du document: {filename}")

        try:
            # 1. Parser le code
            entities = self.parser.parse_fortran_code(code, filename, self.ontology_manager)
            logger.info(f"📋 Parser trouvé {len(entities)} entités")

            # 2. Envoyer à EntityManager
            result = await self.entity_manager.add_entities_from_parser(entities, document_id)
            logger.info(f"📥 EntityManager: {result['added']} nouvelles, {result['updated']} mises à jour")

            # 3. Retourner les statistiques
            return {
                'entities_parsed': len(entities),
                'entities_added': result['added'],
                'entities_updated': result['updated'],
                'entities_indexed': len(self.entity_manager.entities),
                'parsing_stats': self.parser.get_parsing_stats()
            }

        except Exception as e:
            logger.error(f"❌ Erreur analyse document {filename}: {e}")
            return {
                'error': str(e),
                'entities_parsed': 0,
                'entities_added': 0,
                'entities_updated': 0,
                'entities_indexed': len(self.entity_manager.entities),
                'parsing_stats': {}
            }

    async def get_entity_manager(self) -> EntityManager:
        """Récupère l'EntityManager"""
        if not self._initialized:
            await self.initialize()
        return self.entity_manager

    async def get_stats(self) -> dict:
        """Statistiques de l'orchestrateur"""
        if not self._initialized:
            await self.initialize()

        return {
            'parser_stats': self.parser.get_parsing_stats(),
            'entity_manager_stats': self.entity_manager.get_stats()
        }

    async def refresh_all(self):
        """Rafraîchit tous les index"""
        if not self._initialized:
            await self.initialize()

        await self.entity_manager.rebuild_index()
        logger.info("🔄 Tous les index rafraîchis")