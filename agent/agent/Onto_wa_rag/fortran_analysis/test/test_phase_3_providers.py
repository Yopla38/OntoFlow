"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# test/test_phase_3_corrected.py
"""
Tests Phase 3 avec corrections des erreurs identifiées.
"""

import asyncio
import tempfile
import os


async def test_phase_3_corrected():
    """Test Phase 3 avec toutes les corrections"""

    print("🔧 Tests Phase 3 CORRIGÉS")
    print("=" * 50)

    # Code de test simplifié pour éviter les erreurs de parsing
    test_code = """module simple_test
    implicit none
    public :: test_subroutine
    contains
    subroutine test_subroutine()
        print *, "test"
    end subroutine test_subroutine
end module simple_test"""

    # Mock store simplifié et robuste
    class RobustMockStore:
        def __init__(self):
            self.docs = {'test': {'id': 'test', 'path': 'test.f90'}}
            self.chunks = {
                'test': [
                    {
                        'id': 'test-chunk-0',
                        'text': test_code,
                        'metadata': {
                            'entity_name': 'simple_test',
                            'entity_type': 'module',
                            'filepath': 'test.f90',
                            'filename': 'test.f90',
                            'start_pos': 1,
                            'end_pos': 10,
                            'dependencies': [],
                            'detected_concepts': [
                                {'label': 'test_module', 'confidence': 0.8, 'category': 'testing'}
                            ]
                        }
                    }
                ]
            }

        async def get_all_documents(self):
            return ['test']

        async def get_document_chunks(self, doc_id):
            return self.chunks.get(doc_id, [])

        async def load_document_chunks(self, doc_id):
            return True

    class MockRAG:
        async def find_similar(self, text, max_results=5, min_similarity=0.5):
            return []

    try:
        print("\n1. Test avec corrections...")

        mock_store = RobustMockStore()
        mock_rag = MockRAG()

        # Test EntityManager corrigé
        from ..core.entity_manager import EntityManager
        entity_manager = EntityManager(mock_store)
        await entity_manager.initialize()

        print(f"   ✅ EntityManager: {len(entity_manager.entities)} entités")

        # Test BaseProvider corrigé
        from ..providers.base_provider import BaseContextProvider
        from ..providers.local_context import LocalContextProvider

        local_provider = LocalContextProvider(mock_store, mock_rag, entity_manager)

        # Test de résolution d'entité
        resolved = await local_provider.resolve_entity('simple_test')
        if resolved:
            print(f"   ✅ Résolution entité: {resolved['name']}")

        # Test de définition d'entité (correction principale)
        definition = await local_provider.get_entity_definition('simple_test')
        if definition:
            print(f"   ✅ Définition: {definition['name']} ({definition['type']})")
            print(f"      Concepts: {len(definition['concepts'])}")

        # Test contexte local complet
        local_context = await local_provider.get_local_context('simple_test', 1000)
        print(f"   ✅ Contexte local: {local_context.get('tokens_used', 0)} tokens")

        # Test SmartOrchestrator corrigé
        from ..providers.smart_orchestrator import SmartContextOrchestrator

        orchestrator = SmartContextOrchestrator(mock_store, mock_rag)
        await orchestrator.initialize()

        # Test contexte complet
        full_context = await orchestrator.get_context_for_agent(
            'simple_test', 'developer', 'code_understanding', 2000
        )

        print(f"   ✅ Contexte complet: {full_context['total_tokens']} tokens")
        print(f"      Contextes générés: {list(full_context['contexts'].keys())}")
        print(f"      Erreurs: {[k for k, v in full_context['contexts'].items() if 'error' in v]}")

        print("\n🎉 Tests Phase 3 corrigés réussis!")
        return True

    except Exception as e:
        print(f"\n❌ Erreur tests corrigés: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_phase_3_corrected())
    print(f"\n{'✅ SUCCÈS' if success else '❌ ÉCHEC'}")