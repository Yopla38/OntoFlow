"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# tests/test_phase_1_utils.py
"""
Tests corrigés pour valider les améliorations de la Phase 1
"""

import asyncio
from ..utils.fortran_patterns import extract_function_calls, extract_signature, get_fortran_processor
from ..core.hybrid_fortran_parser import get_fortran_analyzer


def test_corrected_patterns():
    """Test avec le code qui posait problème"""

    test_code = """
    module physics_mod
        use constants, only: pi
        implicit none

        contains

        subroutine compute_energy(x, y, energy)
            real, intent(in) :: x, y
            real, intent(out) :: energy

            energy = sqrt(x**2 + y**2)  ! Distance
            call normalize_vector(x, y)
        end subroutine compute_energy

        pure function distance(x1, y1, x2, y2) result(dist)
            real, intent(in) :: x1, y1, x2, y2
            real :: dist

            dist = my_custom_func(x1, x2) + other_calc(y1, y2)
        end function distance

    end module physics_mod
    """

    print("🧪 Tests corrigés - Phase 1")
    print("=" * 50)

    # Test 1: Extraction d'appels avec parser hybride
    print("\n1. Test extraction d'appels (hybride):")
    calls_hybrid = extract_function_calls(test_code, use_hybrid=True)
    print(f"   ✅ Appels détectés (hybride): {calls_hybrid}")

    # Test 2: Extraction d'appels avec fallback regex
    print("\n2. Test extraction d'appels (regex fallback):")
    calls_regex = extract_function_calls(test_code, use_hybrid=False)
    print(f"   ✅ Appels détectés (regex): {calls_regex}")

    # Test 3: Extraction de signature
    print("\n3. Test extraction de signature:")
    signature = extract_signature(test_code, use_hybrid=True)
    print(f"   ✅ Signature: {signature}")

    # Test 4: Analyse complète avec le parser hybride
    print("\n4. Test analyse complète:")
    analyzer = get_fortran_analyzer("hybrid")
    entities = analyzer.get_entities(test_code, "test.f90")

    print(f"   ✅ {len(entities)} entités détectées:")
    for entity in entities:
        print(f"      - {entity.name} ({entity.entity_type}): {len(entity.called_functions)} appels")
        if entity.called_functions:
            print(f"        Appels: {list(entity.called_functions)}")

    # Test 5: Statistiques
    print("\n5. Statistiques d'analyse:")
    stats = analyzer.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Validation des résultats attendus
    print("\n6. Validation:")
    expected_calls = {'normalize_vector', 'my_custom_func', 'other_calc'}
    found_calls = set(calls_hybrid)

    # sqrt doit être détecté mais peut être filtré selon la configuration
    if 'sqrt' in found_calls:
        print("   ✅ sqrt détecté (intrinsèque)")
        found_calls.discard('sqrt')  # L'enlever pour la comparaison

    # Vérifier que les mots-clés problématiques ne sont plus là
    bad_calls = {'intent', 'constants'}.intersection(found_calls)
    if bad_calls:
        print(f"   ❌ Faux positifs détectés: {bad_calls}")
    else:
        print("   ✅ Aucun faux positif (intent, constants)")

    # Vérifier les vrais appels
    missing_calls = expected_calls - found_calls
    extra_calls = found_calls - expected_calls - {'sqrt'}  # sqrt est optionnel

    if not missing_calls and not extra_calls:
        print("   ✅ Tous les appels corrects détectés")
    else:
        if missing_calls:
            print(f"   ⚠️ Appels manqués: {missing_calls}")
        if extra_calls:
            print(f"   ⚠️ Appels supplémentaires: {extra_calls}")

    # Vérifier la signature
    if signature != "Signature not found" and "compute_energy" in signature:
        print("   ✅ Signature correctement extraite")
    else:
        print(f"   ⚠️ Problème de signature: {signature}")


if __name__ == "__main__":
    test_corrected_patterns()