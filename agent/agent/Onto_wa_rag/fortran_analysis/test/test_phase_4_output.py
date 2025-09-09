# test/test_phase_4_output.py
"""
Tests d'intégration pour valider la Phase 4 : Couche de Sortie.
Version finale et correcte, testant le système de bout en bout
sans création manuelle de chunks dans le test.
"""

import asyncio
import os
import tempfile
import shutil
from typing import List, Dict, Any, Tuple
from unittest.mock import MagicMock, AsyncMock

from utils.document_store import DocumentStore
# --- Imports de notre nouvelle architecture ---
from ..providers.smart_orchestrator import SmartOrchestrator
from ..output.graph_visualizer import GraphVisualizer
from ..output.text_generator import TextGenerator
from ..output.report_generator import ReportGenerator
from ..core.fortran_parser import UnifiedFortranParser  # Notre parser de la Phase 2


# --- Simulation du maillon manquant : DocumentProcessor ---
# C'est la classe qui utilise notre parser pour créer les chunks.
class MockDocumentProcessor:
    def __init__(self):
        # Le processeur utilise notre NOUVEAU parser unifié.
        self.parser = UnifiedFortranParser()

    async def process_document(self, filepath: str, document_id: str, metadata: Dict = None) -> Tuple[str, List[Dict]]:
        """
        Simule le traitement d'un document : lit, parse, et crée les chunks.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()

        # 1. Utiliser notre parser pour obtenir les entités
        parsed_result = await self.parser.parse_code_snippet(code, filepath)
        entities = parsed_result.entities

        # 2. Convertir les entités en chunks (la logique que j'avais mise au mauvais endroit)
        chunks = []
        lines = code.splitlines()
        for i, entity in enumerate(entities):
            # Le start_line et end_line de l'entité sont des entiers, l'erreur TypeError est corrigée.
            entity_code = "\n".join(lines[entity.start_line - 1: entity.end_line])
            chunk_metadata = entity.to_dict()
            chunk_metadata.update({
                'filepath': filepath,
                'filename': os.path.basename(filepath),
            })
            chunks.append({
                'id': f"{document_id}-chunk-{i}",
                'text': entity_code,
                'metadata': chunk_metadata
            })
        return document_id, chunks


async def test_phase_4_output_layer():
    """Test l'intégration de la couche de sortie avec le reste du système."""

    print("\n🧪 Tests d'intégration Phase 4 - Approche de Bout en Bout")
    print("=" * 70)

    # --- Création d'un environnement de test temporaire ---
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Créer un fichier Fortran source
        test_code = """
module module_a
    use constants
    implicit none
    contains
    subroutine sub_a()
        call sub_b("test")
    end subroutine sub_a
end module module_a

subroutine sub_b(arg)
    character(len=*), intent(in) :: arg
    implicit none
    print *, "Hello from B:", arg
end subroutine sub_b
        """
        source_filepath = os.path.join(temp_dir, "test_code.f90")
        with open(source_filepath, "w") as f:
            f.write(test_code)

        # 2. Setup des dépendances (DocumentStore et RAG Engine)
        mock_embedding_manager = MagicMock()
        mock_embedding_manager.storage_dir = os.path.join(temp_dir, "embeddings")
        mock_embedding_manager.load_embeddings = AsyncMock(return_value=True)
        mock_embedding_manager.create_embeddings = AsyncMock()
        mock_embedding_manager.save_embeddings = AsyncMock()

        # On instancie notre VRAI DocumentStore avec un processeur qui utilise notre NOUVELLE logique
        storage_dir = os.path.join(temp_dir, "storage")
        doc_store = DocumentStore(
            document_processor=MockDocumentProcessor(),
            embedding_manager=mock_embedding_manager,
            storage_dir=storage_dir
        )

        mock_rag_engine = MagicMock()

        # --- Exécution du flux normal du système ---

        print(f"1. Ajout du fichier source '{source_filepath}' au DocumentStore...")
        # C'est cette ligne qui déclenche le parsing et le chunking par notre nouvelle architecture.
        document_id = await doc_store.add_document_with_id(source_filepath, "doc1")
        print(f"   ✅ Document traité et 'chunké'. ID: {document_id}")

        # 2. Initialisation de l'orchestrateur
        print("\n2. Initialisation du SmartOrchestrator...")
        # L'orchestrateur lit maintenant les chunks qui ont été créés et stockés.
        orchestrator = SmartOrchestrator(doc_store, mock_rag_engine)
        await orchestrator.initialize()

        assert orchestrator._initialized, "L'orchestrateur n'a pas pu s'initialiser."
        print("   ✅ Orchestrateur initialisé. Le système a chargé et analysé les chunks.")
        assert 'sub_a' in orchestrator.entity_manager.name_to_entity, "L'entité 'sub_a' n'a pas été indexée."

        # 3. Instanciation de la couche de sortie
        visualizer = GraphVisualizer(orchestrator.analyzer, orchestrator.entity_manager)
        text_gen = TextGenerator(orchestrator)
        report_gen = ReportGenerator(orchestrator.entity_manager, orchestrator.analyzer)

        # --- Lancement des tests sur les composants de sortie ---

        print("\n3. Test TextGenerator...")
        context_text = await text_gen.get_contextual_text("sub_a", format_style="detailed")
        assert "sub_b" in context_text.lower(), "Le TextGenerator n'a pas affiché l'appel à sub_b."
        print("   ✅ Résumé textuel généré avec succès et contient la relation d'appel.")

        print("\n4. Test GraphVisualizer...")
        output_viz_file = os.path.join(temp_dir, "test_visualization.html")
        await visualizer.generate_visualization("module_a", output_file=output_viz_file, max_depth=2)
        assert os.path.exists(output_viz_file), "Le fichier de visualisation n'a pas été créé !"
        print(f"   ✅ Fichier de visualisation créé.")

        print("\n5. Test ReportGenerator...")
        output_report_file = os.path.join(temp_dir, "test_report.html")
        await report_gen.generate_project_report(output_file=output_report_file)
        assert os.path.exists(output_report_file), "Le fichier de rapport n'a pas été créé."
        print(f"   ✅ Fichier de rapport créé.")

        print("\n🎉 Tous les tests de la couche de sortie ont réussi de bout en bout !")


if __name__ == "__main__":
    asyncio.run(test_phase_4_output_layer())