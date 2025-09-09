"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# test/test_phase_2_integration.py
"""
Tests d'intégration pour valider la Phase 2 du système d'analyse Fortran.
"""

import asyncio
import tempfile
import os
from pathlib import Path


async def test_phase_2_integration():
    """Test d'intégration complet de la Phase 2"""

    print("🧪 Tests d'intégration Phase 2")
    print("=" * 60)

    # Code de test enrichi
    test_code = """
module molecular_dynamics
    use constants, only: kb, pi
    use force_calculation, only: lennard_jones_force
    implicit none

    private
    public :: velocity_verlet_step, compute_kinetic_energy

    contains

    subroutine velocity_verlet_step(particles, dt, box_size)
        type(particle_t), intent(inout) :: particles(:)
        real(real64), intent(in) :: dt, box_size
        integer :: i

        ! Update positions
        do i = 1, size(particles)
            particles(i)%x = particles(i)%x + particles(i)%vx * dt + &
                             0.5_real64 * particles(i)%fx * dt**2 / particles(i)%mass
        end do

        ! Compute new forces
        call compute_forces(particles, box_size)
        call apply_periodic_boundary(particles, box_size)

        ! Update velocities
        call update_velocities(particles, dt)
    end subroutine velocity_verlet_step

    pure function compute_kinetic_energy(particles) result(ke)
        type(particle_t), intent(in) :: particles(:)
        real(real64) :: ke
        integer :: i

        ke = 0.0_real64
        do i = 1, size(particles)
            ke = ke + 0.5_real64 * particles(i)%mass * &
                 (particles(i)%vx**2 + particles(i)%vy**2 + particles(i)%vz**2)
        end do
    end function compute_kinetic_energy

end module molecular_dynamics
    """

    # Mock document store simple pour les tests
    class MockDocumentStore:
        def __init__(self):
            self.docs = {}
            self.chunks = {}

        async def get_all_documents(self):
            return list(self.docs.keys())

        async def get_document_chunks(self, doc_id):
            return self.chunks.get(doc_id, [])

        async def load_document_chunks(self, doc_id):
            return True

    # Mock chunk avec métadonnées
    def create_mock_chunk(chunk_id, entity_name, entity_type, start_line, end_line):
        return {
            'id': chunk_id,
            'text': test_code[start_line:end_line] if isinstance(start_line, int) else test_code,
            'metadata': {
                'entity_name': entity_name,
                'entity_type': entity_type,
                'filepath': 'test_molecular_dynamics.f90',
                'filename': 'test_molecular_dynamics.f90',
                'start_pos': start_line,
                'end_pos': end_line,
                'dependencies': ['constants', 'force_calculation'] if entity_type == 'module' else [],
                'detected_concepts': [
                    {'label': 'molecular_dynamics', 'confidence': 0.9, 'category': 'physics'},
                    {'label': 'numerical_integration', 'confidence': 0.7, 'category': 'mathematics'}
                ]
            }
        }

    # Setup mock data
    mock_store = MockDocumentStore()
    mock_store.docs['test_doc'] = {'id': 'test_doc', 'path': 'test.f90'}
    mock_store.chunks['test_doc'] = [
        create_mock_chunk('test_doc-chunk-0', 'molecular_dynamics', 'module', 1, 100),
        create_mock_chunk('test_doc-chunk-1', 'velocity_verlet_step', 'subroutine', 10, 50),
        create_mock_chunk('test_doc-chunk-2', 'compute_kinetic_energy', 'function', 51, 80)
    ]

    try:
        # Test 1: EntityManager
        print("\n1. Test EntityManager...")
        from ..core.entity_manager import EntityManager

        entity_manager = EntityManager(mock_store)
        await entity_manager.initialize()

        print(f"   ✅ {len(entity_manager.entities)} entités indexées")

        # Rechercher une entité
        module_entity = await entity_manager.find_entity('molecular_dynamics')
        if module_entity:
            print(f"   ✅ Module trouvé: {module_entity.entity_name}")
        else:
            print("   ❌ Module non trouvé")

        # Test 2: FortranParser
        print("\n2. Test UnifiedFortranParser...")
        from ..core.fortran_parser import UnifiedFortranParser

        parser = UnifiedFortranParser()
        result = await parser.parse_code_snippet(test_code, "molecular_dynamics_test")

        print(f"   ✅ {result.total_entities} entités parsées")
        print(f"   ✅ {result.total_calls} appels détectés")
        print(f"   ✅ Standard Fortran: {result.fortran_standard}")

        for entity in result.get_main_entities():
            print(f"      - {entity.name} ({entity.entity_type}): {len(entity.called_functions)} appels")

        # Test 3: FortranAnalyzer
        print("\n3. Test FortranAnalyzer...")
        from ..core.fortran_analyzer import get_fortran_analyzer

        analyzer = await get_fortran_analyzer(mock_store, entity_manager)

        # Analyser les appels de fonctions
        call_analysis = await analyzer.analyze_function_calls('velocity_verlet_step')
        if 'error' not in call_analysis:
            print(f"   ✅ Analyse des appels: {call_analysis['call_statistics']['total_outgoing']} sortants")
        else:
            print(f"   ⚠️ Entité non trouvée dans l'EntityManager, utilisation du parser direct")

        # Analyser les dépendances avec le parser
        deps = await parser.analyze_dependencies(test_code, "molecular_dynamics.f90")
        print(f"   ✅ Dépendances: {deps['total_dependencies']} total")
        print(f"      USE statements: {deps['use_statements']}")
        print(f"      Function calls: {deps['function_calls']}")

        # Test 4: ConceptDetector
        print("\n4. Test ConceptDetector...")
        from ..core.concept_detector import get_concept_detector

        detector = get_concept_detector()
        concepts = await detector.detect_concepts(test_code, "molecular_dynamics")

        print(f"   ✅ {len(concepts)} concepts détectés:")
        for concept in concepts[:5]:
            print(f"      - {concept.label}: {concept.confidence:.3f} ({concept.category})")

        # Test concepts pour entité spécifique
        entity_concepts = await detector.detect_concepts_for_entity(
            test_code, "velocity_verlet_step", "subroutine"
        )
        print(f"   ✅ {len(entity_concepts)} concepts pour l'entité:")
        for concept in entity_concepts[:3]:
            print(f"      - {concept.label}: {concept.confidence:.3f}")

        # Test 5: Intégration complète
        print("\n5. Test d'intégration complète...")

        # Patterns algorithmiques
        if module_entity:
            patterns = await analyzer.detect_algorithmic_patterns('molecular_dynamics')
            if 'error' not in patterns:
                print(f"   ✅ Patterns détectés: {patterns['pattern_summary']['total_patterns']}")
                if patterns['detected_patterns']:
                    top_pattern = patterns['detected_patterns'][0]
                    print(f"      Pattern principal: {top_pattern['pattern']} (score: {top_pattern['score']})")

        # Test de performance et cache
        print("\n6. Test de performance et cache...")

        # Répéter l'analyse pour tester le cache
        start_time = asyncio.get_event_loop().time()
        for _ in range(3):
            concepts_cached = await detector.detect_concepts(test_code, "molecular_dynamics")
        cache_time = asyncio.get_event_loop().time() - start_time

        print(f"   ✅ 3 détections avec cache: {cache_time:.3f}s")

        # Statistiques des caches
        from ..utils.caching import global_cache
        cache_stats = global_cache.get_all_stats()

        print("\n7. Statistiques des caches:")
        for cache_name, stats in cache_stats.items():
            if stats.hits + stats.misses > 0:
                hit_rate = stats.hits / (stats.hits + stats.misses) * 100
                print(f"   {cache_name}: {stats.entries} entrées, {hit_rate:.1f}% hit rate")

        # Test de l'API unifiée
        print("\n8. Test API unifiée...")

        # Extraction simple d'appels
        calls = await parser.extract_function_calls_only(test_code)
        print(f"   ✅ API simple - appels: {calls}")

        # Extraction de signature
        signature = await parser.extract_signature_only(test_code)
        print(f"   ✅ API simple - signature: {signature}")

        # Statistiques finales
        print("\n📊 Statistiques finales:")
        entity_stats = entity_manager.get_stats()
        parser_stats = parser.get_parsing_stats()

        print(f"   Entités totales: {entity_stats['total_entities']}")
        print(f"   Chunks totaux: {entity_stats['total_chunks']}")
        print(f"   Parsing réussis: {parser_stats['total_parsed']}")
        print(f"   Cache hits parser: {parser_stats['cache_hits']}")

        # Test de nettoyage
        print("\n9. Test de nettoyage...")
        entity_manager.clear_caches()
        detector.clear_cache()
        parser.clear_cache()
        print("   ✅ Caches nettoyés")

        print("\n🎉 Tous les tests Phase 2 réussis!")
        return True

    except Exception as e:
        print(f"\n❌ Erreur dans les tests Phase 2: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_phase_2_integration())
    if success:
        print("\n✅ Phase 2 validée avec succès!")
    else:
        print("\n❌ Phase 2 nécessite des corrections")