"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# tools_rag.py
import asyncio, json
from pathlib import Path
from typing import Any, Dict
from CONSTANT import ONTOLOGY_PATH_TTL
from RAG_context_provider import RagContextProvider


class RagTools:
    """Expose chaque tool sous forme de coroutine."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.rag = RagContextProvider(
            storage_dir=str(storage_dir / "rag_storage"),
            ontology_path=ONTOLOGY_PATH_TTL
        )
        self._initialized = False

    async def _ensure_init(self):
        if not self._initialized:
            await self.rag.initialize()
            self._initialized = True

    # ------------- tools -----------------
    async def rag_index_directory(self, directory: str) -> str:
        await self._ensure_init()
        await self.rag.index_directory(Path(directory))
        return f"Dossier {directory} indexé avec succès."

    async def rag_add_file(self, filepath: str) -> str:
        await self._ensure_init()
        await self.rag.index_file(Path(filepath))
        return f"Fichier {filepath} ajouté."

    async def rag_remove_file(self, filepath: str) -> str:
        await self._ensure_init()
        await self.rag.remove_file(Path(filepath))
        return f"Fichier {filepath} supprimé."

    async def rag_query(self, question: str, max_chunks: int = 8) -> Dict[str, Any]:
        await self._ensure_init()
        passages = await self.rag.query(question, max_chunks=max_chunks)
        return passages   # serialisable

    async def rag_function_json(self, question: str) -> Dict[str, Any]:
        await self._ensure_init()
        passages = await self.rag.query(question, max_chunks=15)
        if passages:
            return json.loads(await self.rag.passages_to_function_json(passages))
        return {"error": "aucun passage trouvé"}