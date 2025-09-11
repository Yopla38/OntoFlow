"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# test_hybrid_parser.py
"""
Tests complets du nouveau parser Fortran hybride.
Compare les performances avec l'ancien parser et teste sur les vrais fichiers Fortran.
"""

import asyncio
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

# Imports du nouveau système
from fortran_analysis.core.hybrid_fortran_parser import (
    FortranAnalysisEngine,
    get_fortran_analyzer,
    HybridFortranParser
)
from fortran_analysis.core.fortran_parser import (
    UnifiedFortranParser,
    get_unified_parser,
    ParsedFortranFile
)
from fortran_analysis.core.entity_manager import EntityManager, UnifiedEntity
from fortran_analysis.core.concept_detector import ConceptDetector, get_concept_detector
from fortran_analysis.utils.caching import global_cache
from fortran_analysis.utils.fortran_patterns import get_fortran_processor

# Import de l'ancien parser pour comparaison
try:
    from utils.ontologie_fortran_chunker import OFPFortranSemanticChunker

    OLD_PARSER_AVAILABLE = True
except ImportError:
    OLD_PARSER_AVAILABLE = False
    OFPFortranSemanticChunker = None


class HybridParserTester:
    """
    Testeur complet pour le nouveau parser hybride.
    Compare avec l'ancien système et fait des tests de régression.
    """

    def __init__(self):
        self.test_files = {
            'constants_types.f90': '/home/yopla/dynamic_molecular/constants_types.f90',
            'math_utilities.f90': '/home/yopla/dynamic_molecular/math_utilities.f90',
            'simulation_main.f90': '/home/yopla/dynamic_molecular/simulation_main.f90',
            'integrator.f90': '/home/yopla/dynamic_molecular/integrator.f90',
            'force_calculations.f90': '/home/yopla/dynamic_molecular/force_calculations.f90'
        }

        # Nouveau système
        self.hybrid_parser = get_fortran_analyzer("hybrid")
        self.unified_parser = get_unified_parser("hybrid")
        self.concept_detector = get_concept_detector()
        self.fortran_processor = get_fortran_processor(use_hybrid=True)

        # Ancien système (si disponible)
        self.old_parser = None
        if OLD_PARSER_AVAILABLE:
            self.old_parser = OFPFortranSemanticChunker(
                min_chunk_size=200,
                max_chunk_size=2000,
                overlap_sentences=0
            )

        # Résultats des tests
        self.test_results = {
            'files_tested': [],
            'parsing_comparison': {},
            'performance_metrics': {},
            'entity_analysis': {},
            'concept_detection': {},
            'function_calls': {},
            'error_analysis': {},
            'regression_tests': {}
        }

    def print_separator(self, title: str, level: int = 1):
        """Affiche un séparateur avec titre"""
        chars = "=" if level == 1 else "-"
        width = 80 if level == 1 else 60
        print(f"\n{chars * width}")
        print(f"{'🧪' if level == 1 else '🔬'} {title}")
        print(f"{chars * width}")

    async def test_cross_file_dependencies(self):
        """Analyse les dépendances entre fichiers"""
        self.print_separator("DÉPENDANCES INTER-FICHIERS", 2)

        all_modules = {}  # nom_module → fichier
        all_entities = {}  # nom_entité → (fichier, type)
        dependencies_graph = {}  # fichier → [fichiers utilisés]

        # Première passe : collecter toutes les entités
        for filename, filepath in self.test_files.items():
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()

            entities = self.hybrid_parser.get_entities(code, filepath)

            for entity in entities:
                if entity.entity_type == 'module':
                    all_modules[entity.entity_name] = filename
                all_entities[entity.entity_name] = (filename, entity.entity_type)

        print(f"📦 Modules trouvés: {list(all_modules.keys())}")

        # Deuxième passe : analyser les dépendances
        for filename, filepath in self.test_files.items():
            print(f"\n🔍 Dépendances de {filename}:")

            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()

            entities = self.hybrid_parser.get_entities(code, filepath)
            used_files = set()

            for entity in entities:
                # Dépendances USE
                for dep in entity.dependencies:
                    if dep in all_modules:
                        target_file = all_modules[dep]
                        if target_file != filename:
                            used_files.add(target_file)
                            print(f"   📦 USE {dep} (depuis {target_file})")

                # Appels de fonctions
                for call in entity.called_functions:
                    if call in all_entities:
                        target_file, target_type = all_entities[call]
                        if target_file != filename:
                            used_files.add(target_file)
                            print(f"   📞 APPELLE {call} ({target_type} depuis {target_file})")

            dependencies_graph[filename] = list(used_files)

            if not used_files:
                print(f"   ⭐ {filename} est indépendant")

        # Résumé du graphe
        print(f"\n🌐 GRAPHE DE DÉPENDANCES:")
        for file, deps in dependencies_graph.items():
            if deps:
                print(f"   {file} → {', '.join(deps)}")

    async def run_all_tests(self):
        """Lance tous les tests sur les fichiers réels"""
        print("🚀 LANCEMENT DES TESTS DU PARSER HYBRIDE")
        print("=" * 80)

        start_time = time.time()

        # Vérifier la disponibilité des fichiers
        await self.check_test_files()

        # Tests principaux
        await self.test_basic_parsing()
        await self.test_entity_extraction()
        await self.test_function_call_detection()
        await self.test_dependency_analysis()
        await self.test_concept_detection()
        await self.test_performance_comparison()
        await self.test_regression_scenarios()

        # Analyse finale
        total_time = time.time() - start_time
        await self.generate_final_report(total_time)

    async def check_test_files(self):
        """Vérifie que tous les fichiers de test existent"""
        self.print_separator("VÉRIFICATION DES FICHIERS DE TEST")

        available_files = {}
        for name, filepath in self.test_files.items():
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"✅ {name} - {size} bytes")
                available_files[name] = filepath
            else:
                print(f"❌ {name} - FICHIER MANQUANT: {filepath}")

        self.test_files = available_files
        self.test_results['files_tested'] = list(available_files.keys())

        if not available_files:
            print("❌ AUCUN FICHIER DE TEST DISPONIBLE!")
            return False

        print(f"\n📊 {len(available_files)} fichiers disponibles pour les tests")
        return True

    async def test_basic_parsing(self):
        """Test de parsing de base avec le nouveau système"""
        self.print_separator("TESTS DE PARSING DE BASE", 2)

        parsing_results = {}

        for filename, filepath in self.test_files.items():
            print(f"\n🔍 Test de parsing: {filename}")

            try:
                # Lire le fichier
                with open(filepath, 'r', encoding='utf-8') as f:
                    code = f.read()

                # Parser avec le nouveau système
                start_time = time.time()
                parsed_result = await self.unified_parser.parse_file(filepath, code)
                parse_time = time.time() - start_time

                # Collecter les résultats
                result = {
                    'success': True,
                    'parse_time': parse_time,
                    'total_entities': parsed_result.total_entities,
                    'total_calls': parsed_result.total_calls,
                    'fortran_standard': parsed_result.fortran_standard,
                    'parse_method': parsed_result.parse_method,
                    'confidence': parsed_result.confidence,
                    'entities_by_type': {}
                }

                # Analyser les entités par type
                for entity in parsed_result.entities:
                    entity_type = entity.entity_type
                    if entity_type not in result['entities_by_type']:
                        result['entities_by_type'][entity_type] = 0
                    result['entities_by_type'][entity_type] += 1

                parsing_results[filename] = result

                print(f"   ✅ Succès - {parsed_result.total_entities} entités, "
                      f"{parsed_result.total_calls} appels ({parse_time:.3f}s)")
                print(f"   📊 Standard: {parsed_result.fortran_standard}, "
                      f"Méthode: {parsed_result.parse_method}")
                print(f"   🎯 Confiance: {parsed_result.confidence:.2f}")

                # Afficher le détail des entités
                for entity_type, count in result['entities_by_type'].items():
                    print(f"      {entity_type}: {count}")

            except Exception as e:
                print(f"   ❌ Erreur: {e}")
                parsing_results[filename] = {
                    'success': False,
                    'error': str(e),
                    'parse_time': 0
                }

        self.test_results['parsing_comparison'] = parsing_results

    async def test_entity_extraction(self):
        """Test d'extraction détaillée des entités"""
        self.print_separator("TESTS D'EXTRACTION D'ENTITÉS", 2)

        entity_results = {}

        for filename, filepath in self.test_files.items():
            print(f"\n🏗️ Analyse des entités: {filename}")

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    code = f.read()

                # Extraire les entités avec le parser hybride
                entities = self.hybrid_parser.get_entities(code, filepath)

                file_analysis = {
                    'total_entities': len(entities),
                    'entities_detail': [],
                    'hierarchical_relations': [],
                    'dependencies': [],
                    'signatures_found': 0
                }

                # Analyser chaque entité
                for entity in entities:
                    entity_detail = {
                        'name': entity.entity_name,
                        'type': entity.entity_type,
                        'start_line': entity.start_line,
                        'end_line': entity.end_line,
                        'has_signature': bool(entity.signature),
                        'dependencies_count': len(entity.dependencies),
                        'calls_count': len(entity.called_functions),
                        'confidence': entity.confidence,
                        'source_method': entity.source_method
                    }

                    file_analysis['entities_detail'].append(entity_detail)

                    if entity.signature:
                        file_analysis['signatures_found'] += 1

                    # Relations hiérarchiques
                    if entity.parent_entity:
                        file_analysis['hierarchical_relations'].append({
                            'child': entity.entity_name,
                            'parent': entity.parent_entity
                        })

                    # Dépendances
                    file_analysis['dependencies'].extend(list(entity.dependencies))

                entity_results[filename] = file_analysis

                print(f"   📊 {len(entities)} entités extraites")
                print(f"   🔗 {len(file_analysis['hierarchical_relations'])} relations hiérarchiques")
                print(f"   📝 {file_analysis['signatures_found']} signatures trouvées")
                print(f"   🔗 {len(set(file_analysis['dependencies']))} dépendances uniques")

                # Afficher les entités principales
                main_entities = [e for e in entities if
                                 e.entity_type in ['module', 'program', 'subroutine', 'function']]
                for entity in main_entities[:5]:  # Top 5
                    print(f"      📋 {entity.entity_type}: {entity.entity_name} "
                          f"(lignes {entity.start_line}-{entity.end_line})")

                print(f"\n   🔗 DÉTAIL DES RELATIONS:")
                for entity in entities[:5]:  # Top 5
                    print(f"      📋 {entity.entity_type}: {entity.entity_name}")

                    if entity.dependencies:
                        print(f"         📦 USE: {', '.join(list(entity.dependencies))}")

                    if entity.called_functions:
                        print(f"         📞 APPELLE: {', '.join(list(entity.called_functions))}")

                    if entity.parent_entity:
                        print(f"         👤 PARENT: {entity.parent_entity}")

            except Exception as e:
                print(f"   ❌ Erreur extraction entités: {e}")
                entity_results[filename] = {'error': str(e)}

        self.test_results['entity_analysis'] = entity_results

    async def test_dependency_analysis(self):
        """Test d'analyse des dépendances et appels - COMPARAISON ANCIEN vs NOUVEAU"""
        self.print_separator("ANALYSE DES DÉPENDANCES ET APPELS (ANCIEN vs NOUVEAU)", 2)

        if not self.old_parser:
            print("⚠️ Ancien parser non disponible - test impossible")
            return

        dependency_results = {}

        for filename, filepath in self.test_files.items():
            print(f"\n🔗 Analyse des dépendances: {filename}")
            print("=" * 50)

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    code = f.read()

                # === PARSER AVEC LES DEUX SYSTÈMES ===
                print("🔄 Parsing avec les deux systèmes...")

                # Nouveau parser
                new_entities = self.hybrid_parser.get_entities(code, filepath)

                # Ancien parser
                old_entities = self.old_parser.extract_fortran_structure(code, filepath)

                # === ANALYSE COMPARATIVE ===
                comparison = {
                    'file': filename,
                    'nouveau_parser': {
                        'entités_total': len(new_entities),
                        'use_dependencies': {},
                        'function_calls': {},
                        'internal_calls': {},
                        'entities_with_deps': 0,
                        'entities_with_calls': 0
                    },
                    'ancien_parser': {
                        'entités_total': len(old_entities),
                        'use_dependencies': {},
                        'function_calls': {},
                        'internal_calls': {},
                        'entities_with_deps': 0,
                        'entities_with_calls': 0
                    }
                }

                # === ANALYSER NOUVEAU PARSER ===
                print(f"\n🆕 NOUVEAU PARSER ({len(new_entities)} entités):")
                for entity in new_entities:
                    entity_name = entity.entity_name

                    # USE dependencies
                    if entity.dependencies:
                        comparison['nouveau_parser']['use_dependencies'][entity_name] = list(entity.dependencies)
                        comparison['nouveau_parser']['entities_with_deps'] += 1
                        print(f"   📦 {entity_name} ({entity.entity_type}) USE: {', '.join(list(entity.dependencies))}")

                    # Function calls
                    if entity.called_functions:
                        comparison['nouveau_parser']['function_calls'][entity_name] = list(entity.called_functions)
                        comparison['nouveau_parser']['entities_with_calls'] += 1
                        calls_list = list(entity.called_functions)
                        print(
                            f"   📞 {entity_name} APPELLE: {', '.join(calls_list[:5])}{'...' if len(calls_list) > 5 else ''}")

                    # Calls internes (si parent)
                    if entity.parent_entity:
                        parent_calls = comparison['nouveau_parser']['internal_calls'].get(entity.parent_entity, [])
                        parent_calls.append(entity_name)
                        comparison['nouveau_parser']['internal_calls'][entity.parent_entity] = parent_calls

                # === ANALYSER ANCIEN PARSER ===
                print(f"\n🔙 ANCIEN PARSER ({len(old_entities)} entités):")
                for entity in old_entities:
                    entity_name = entity.name  # Ancien parser utilise .name
                    entity_type = entity.entity_type.value  # Enum vers string

                    # USE dependencies
                    if hasattr(entity, 'dependencies') and entity.dependencies:
                        comparison['ancien_parser']['use_dependencies'][entity_name] = list(entity.dependencies)
                        comparison['ancien_parser']['entities_with_deps'] += 1
                        print(f"   📦 {entity_name} ({entity_type}) USE: {', '.join(list(entity.dependencies))}")

                    # Function calls - vérifier différents attributs possibles
                    calls_found = set()

                    # Vérifier called_functions
                    if hasattr(entity, 'called_functions') and entity.called_functions:
                        calls_found.update(entity.called_functions)

                    # Vérifier function_calls (au cas où)
                    if hasattr(entity, 'function_calls') and entity.function_calls:
                        calls_found.update(entity.function_calls)

                    # Vérifier dans metadata
                    if hasattr(entity, 'metadata') and entity.metadata:
                        if 'function_calls' in entity.metadata:
                            calls_found.update(entity.metadata['function_calls'])
                        if 'called_functions' in entity.metadata:
                            calls_found.update(entity.metadata['called_functions'])

                    if calls_found:
                        comparison['ancien_parser']['function_calls'][entity_name] = list(calls_found)
                        comparison['ancien_parser']['entities_with_calls'] += 1
                        calls_list = list(calls_found)
                        print(
                            f"   📞 {entity_name} APPELLE: {', '.join(calls_list[:5])}{'...' if len(calls_list) > 5 else ''}")

                    # Calls internes
                    if hasattr(entity, 'parent') and entity.parent:
                        parent_calls = comparison['ancien_parser']['internal_calls'].get(entity.parent, [])
                        parent_calls.append(entity_name)
                        comparison['ancien_parser']['internal_calls'][entity.parent] = parent_calls

                # === COMPARAISON DÉTAILLÉE ===
                print(f"\n📊 COMPARAISON DÉTAILLÉE:")
                print("-" * 40)

                # Entités avec dépendances
                new_deps = comparison['nouveau_parser']['entities_with_deps']
                old_deps = comparison['ancien_parser']['entities_with_deps']
                print(f"Entités avec USE deps:     Nouveau: {new_deps:2d} | Ancien: {old_deps:2d}")

                # Entités avec appels
                new_calls = comparison['nouveau_parser']['entities_with_calls']
                old_calls = comparison['ancien_parser']['entities_with_calls']
                print(f"Entités avec appels:       Nouveau: {new_calls:2d} | Ancien: {old_calls:2d}")

                # Total des dépendances
                new_total_deps = sum(len(deps) for deps in comparison['nouveau_parser']['use_dependencies'].values())
                old_total_deps = sum(len(deps) for deps in comparison['ancien_parser']['use_dependencies'].values())
                print(f"Total USE dependencies:    Nouveau: {new_total_deps:2d} | Ancien: {old_total_deps:2d}")

                # Total des appels
                new_total_calls = sum(len(calls) for calls in comparison['nouveau_parser']['function_calls'].values())
                old_total_calls = sum(len(calls) for calls in comparison['ancien_parser']['function_calls'].values())
                print(f"Total function calls:      Nouveau: {new_total_calls:2d} | Ancien: {old_total_calls:2d}")

                # === ANALYSE DES DIFFÉRENCES ===
                print(f"\n🔍 ANALYSE DES DIFFÉRENCES:")

                # Dépendances manquées
                new_deps_entities = set(comparison['nouveau_parser']['use_dependencies'].keys())
                old_deps_entities = set(comparison['ancien_parser']['use_dependencies'].keys())

                only_new = new_deps_entities - old_deps_entities
                only_old = old_deps_entities - new_deps_entities

                if only_new:
                    print(f"   ✅ Dépendances détectées SEULEMENT par nouveau: {', '.join(only_new)}")
                if only_old:
                    print(f"   ⚠️  Dépendances détectées SEULEMENT par ancien: {', '.join(only_old)}")
                if not only_new and not only_old:
                    print(f"   🎯 Même détection de dépendances!")

                # Appels manqués
                new_calls_entities = set(comparison['nouveau_parser']['function_calls'].keys())
                old_calls_entities = set(comparison['ancien_parser']['function_calls'].keys())

                only_new_calls = new_calls_entities - old_calls_entities
                only_old_calls = old_calls_entities - new_calls_entities

                if only_new_calls:
                    print(f"   ✅ Appels détectés SEULEMENT par nouveau: {', '.join(only_new_calls)}")
                if only_old_calls:
                    print(f"   ⚠️  Appels détectés SEULEMENT par ancien: {', '.join(only_old_calls)}")
                if not only_new_calls and not only_old_calls:
                    print(f"   🎯 Même détection d'appels!")

                # === QUI APPELLE QUI ===
                print(f"\n🎯 QUI APPELLE QUI:")

                # Construire le graphe inverse pour les deux parsers
                for parser_name, parser_data in [("NOUVEAU", comparison['nouveau_parser']),
                                                 ("ANCIEN", comparison['ancien_parser'])]:
                    print(f"\n   {parser_name}:")
                    who_calls_whom = {}

                    for caller, called_list in parser_data['function_calls'].items():
                        for called in called_list:
                            if called not in who_calls_whom:
                                who_calls_whom[called] = []
                            who_calls_whom[called].append(caller)

                    for called, callers in who_calls_whom.items():
                        print(f"      🎯 {called} est appelé par: {', '.join(callers)}")

                    if not who_calls_whom:
                        print(f"      ℹ️  Aucun appel interne détecté")

                dependency_results[filename] = comparison

            except Exception as e:
                print(f"   ❌ Erreur analyse dépendances: {e}")
                import traceback
                traceback.print_exc()
                dependency_results[filename] = {'error': str(e)}

        # === RÉSUMÉ GLOBAL ===
        print(f"\n{'=' * 60}")
        print(f"📋 RÉSUMÉ GLOBAL DES DÉPENDANCES")
        print(f"{'=' * 60}")

        total_files = len([r for r in dependency_results.values() if 'error' not in r])
        total_new_deps = 0
        total_old_deps = 0
        total_new_calls = 0
        total_old_calls = 0

        for result in dependency_results.values():
            if 'error' not in result:
                total_new_deps += len(result['nouveau_parser']['use_dependencies'])
                total_old_deps += len(result['ancien_parser']['use_dependencies'])
                total_new_calls += len(result['nouveau_parser']['function_calls'])
                total_old_calls += len(result['ancien_parser']['function_calls'])

        print(f"Fichiers analysés: {total_files}")
        print(f"")
        print(f"USE DEPENDENCIES:")
        print(f"  Nouveau parser: {total_new_deps} entités avec dépendances")
        print(f"  Ancien parser:  {total_old_deps} entités avec dépendances")
        print(f"  Différence:     {total_new_deps - total_old_deps:+d}")
        print(f"")
        print(f"FUNCTION CALLS:")
        print(f"  Nouveau parser: {total_new_calls} entités avec appels")
        print(f"  Ancien parser:  {total_old_calls} entités avec appels")
        print(f"  Différence:     {total_new_calls - total_old_calls:+d}")

        if total_new_deps > total_old_deps:
            print(f"✅ Le nouveau parser détecte PLUS de dépendances!")
        elif total_new_deps < total_old_deps:
            print(f"⚠️  L'ancien parser détecte plus de dépendances")
        else:
            print(f"🎯 Même niveau de détection des dépendances")

        if total_new_calls > total_old_calls:
            print(f"✅ Le nouveau parser détecte PLUS d'appels!")
        elif total_new_calls < total_old_calls:
            print(f"⚠️  L'ancien parser détecte plus d'appels")
        else:
            print(f"🎯 Même niveau de détection des appels")

        self.test_results['dependency_analysis'] = dependency_results

    async def test_function_call_detection(self):
        """Test de détection des appels de fonctions"""
        self.print_separator("TESTS DE DÉTECTION D'APPELS DE FONCTIONS", 2)

        call_results = {}

        for filename, filepath in self.test_files.items():
            print(f"\n📞 Analyse des appels: {filename}")

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    code = f.read()

                # Extraction avec le processeur Fortran
                start_time = time.time()
                calls = self.fortran_processor.extract_function_calls(code, filepath)
                extraction_time = time.time() - start_time

                # Analyse avec le parser hybride pour plus de détails
                entities = self.hybrid_parser.get_entities(code, filepath)

                # Analyser les appels par entité
                calls_by_entity = {}
                total_entity_calls = 0

                for entity in entities:
                    if entity.called_functions:
                        calls_by_entity[entity.entity_name] = {
                            'type': entity.entity_type,
                            'calls': list(entity.called_functions),
                            'call_count': len(entity.called_functions)
                        }
                        total_entity_calls += len(entity.called_functions)

                # Classification des appels
                all_calls = set(calls)
                intrinsic_calls = set()
                user_calls = set()

                # Import des patterns pour classification
                from fortran_analysis.utils.fortran_patterns import FortranPatterns

                for call in all_calls:
                    if call.lower() in FortranPatterns.FORTRAN_INTRINSICS:
                        intrinsic_calls.add(call)
                    else:
                        user_calls.add(call)

                call_analysis = {
                    'total_calls_found': len(calls),
                    'extraction_time': extraction_time,
                    'calls_by_entity': calls_by_entity,
                    'entity_calls_total': total_entity_calls,
                    'intrinsic_calls': list(intrinsic_calls),
                    'user_calls': list(user_calls),
                    'call_classification': {
                        'intrinsic_count': len(intrinsic_calls),
                        'user_count': len(user_calls),
                        'ratio_intrinsic': len(intrinsic_calls) / max(1, len(all_calls))
                    }
                }

                call_results[filename] = call_analysis

                print(f"   📊 {len(calls)} appels trouvés ({extraction_time:.3f}s)")
                print(f"   🔧 {len(intrinsic_calls)} intrinsèques, {len(user_calls)} utilisateur")
                print(f"   📈 {len(calls_by_entity)} entités avec appels")

                # Afficher les entités avec le plus d'appels
                sorted_entities = sorted(calls_by_entity.items(),
                                         key=lambda x: x[1]['call_count'], reverse=True)

                for entity_name, info in sorted_entities[:3]:
                    print(f"      🏆 {info['type']} {entity_name}: {info['call_count']} appels")
                    for call in info['calls'][:3]:  # Top 3 calls
                        print(f"         - {call}")

            except Exception as e:
                print(f"   ❌ Erreur détection appels: {e}")
                call_results[filename] = {'error': str(e)}

        self.test_results['function_calls'] = call_results

    async def test_concept_detection(self):
        """Test de détection des concepts sémantiques"""
        self.print_separator("TESTS DE DÉTECTION DE CONCEPTS", 2)

        concept_results = {}

        for filename, filepath in self.test_files.items():
            print(f"\n🧠 Analyse des concepts: {filename}")

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    code = f.read()

                # Détecter les concepts avec le nouveau détecteur
                entities = self.hybrid_parser.get_entities(code, filepath)

                file_concepts = {
                    'total_entities_analyzed': len(entities),
                    'entities_with_concepts': 0,
                    'all_concepts_detected': [],
                    'concept_by_entity': {},
                    'concept_statistics': {}
                }

                for entity in entities:
                    # Détecter les concepts pour cette entité
                    start_time = time.time()
                    detected_concepts = await self.concept_detector.detect_concepts_for_entity(
                        entity_code=code[
                                    entity.start_line:entity.end_line] if entity.start_line and entity.end_line else code[
                                                                                                                     :500],
                        entity_name=entity.entity_name,
                        entity_type=entity.entity_type
                    )
                    detection_time = time.time() - start_time

                    if detected_concepts:
                        file_concepts['entities_with_concepts'] += 1

                        entity_concept_info = {
                            'entity_type': entity.entity_type,
                            'concepts': [concept.to_dict() for concept in detected_concepts],
                            'detection_time': detection_time,
                            'top_concept': detected_concepts[0].label if detected_concepts else None,
                            'confidence': detected_concepts[0].confidence if detected_concepts else 0
                        }

                        file_concepts['concept_by_entity'][entity.entity_name] = entity_concept_info
                        file_concepts['all_concepts_detected'].extend([c.label for c in detected_concepts])

                # Statistiques des concepts
                from collections import Counter
                concept_counts = Counter(file_concepts['all_concepts_detected'])
                file_concepts['concept_statistics'] = {
                    'unique_concepts': len(concept_counts),
                    'most_common': dict(concept_counts.most_common(5)),
                    'coverage_rate': file_concepts['entities_with_concepts'] / max(1, len(entities))
                }

                concept_results[filename] = file_concepts

                print(f"   🎯 {file_concepts['entities_with_concepts']}/{len(entities)} entités avec concepts")
                print(f"   📚 {file_concepts['concept_statistics']['unique_concepts']} concepts uniques")
                print(f"   📈 Couverture: {file_concepts['concept_statistics']['coverage_rate']:.2%}")

                # Afficher les concepts les plus fréquents
                for concept, count in concept_counts.most_common(3):
                    print(f"      🏷️ {concept}: {count} occurrences")

                # Afficher les entités avec les meilleurs concepts
                sorted_entities = sorted(
                    file_concepts['concept_by_entity'].items(),
                    key=lambda x: x[1]['confidence'], reverse=True
                )

                for entity_name, info in sorted_entities[:2]:
                    print(f"      🌟 {entity_name} ({info['entity_type']}): {info['top_concept']} "
                          f"(confiance: {info['confidence']:.2f})")

            except Exception as e:
                print(f"   ❌ Erreur détection concepts: {e}")
                import traceback
                traceback.print_exc()
                concept_results[filename] = {'error': str(e)}

        self.test_results['concept_detection'] = concept_results

    async def test_performance_comparison(self):
        """Compare les performances avec détails des entités"""
        self.print_separator("COMPARAISON DÉTAILLÉE NOUVEAU vs ANCIEN", 2)

        if not OLD_PARSER_AVAILABLE:
            print("⚠️ Ancien parser non disponible")
            return

        for filename, filepath in self.test_files.items():
            print(f"\n{'=' * 60}")
            print(f"📁 FICHIER: {filename}")
            print(f"{'=' * 60}")

            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()

            # =================== NOUVEAU PARSER ===================
            print(f"\n🆕 NOUVEAU PARSER HYBRIDE:")
            print("-" * 40)

            start_time = time.time()
            try:
                new_entities = self.hybrid_parser.get_entities(code, filepath)
                new_time = time.time() - start_time

                print(f"✅ Succès: {len(new_entities)} entités en {new_time:.3f}s")
                print(f"📋 DÉTAIL DES ENTITÉS NOUVEAU PARSER:")

                for i, entity in enumerate(new_entities, 1):
                    print(f"   {i}. {entity.entity_type}: {entity.entity_name}")
                    print(f"      📍 Lignes: {entity.start_line}-{entity.end_line}")
                    print(f"      🔧 Méthode: {entity.source_method}")
                    print(f"      🎯 Confiance: {entity.confidence:.2f}")
                    if entity.signature:
                        print(f"      ✍️  Signature: {entity.signature}")
                    if entity.dependencies:
                        deps = list(entity.dependencies)[:3]
                        print(f"      🔗 Dépendances: {', '.join(deps)}...")
                    print()

            except Exception as e:
                print(f"❌ Erreur nouveau parser: {e}")
                new_entities = []
                new_time = 0

            # =================== ANCIEN PARSER ===================
            print(f"\n🔙 ANCIEN PARSER OFP:")
            print("-" * 40)

            start_time = time.time()
            try:
                old_entities = self.old_parser.extract_fortran_structure(code, filepath)
                old_time = time.time() - start_time

                print(f"✅ Succès: {len(old_entities)} entités en {old_time:.3f}s")
                print(f"📋 DÉTAIL DES ENTITÉS ANCIEN PARSER:")

                for i, entity in enumerate(old_entities, 1):
                    print(f"   {i}. {entity.entity_type.value}: {entity.name}")
                    print(f"      📍 Lignes: {entity.start_line}-{entity.end_line}")
                    if entity.parent:
                        print(f"      👆 Parent: {entity.parent}")
                    if entity.dependencies:
                        deps = list(entity.dependencies)[:3]
                        print(f"      🔗 Dépendances: {', '.join(deps)}...")
                    print()

            except Exception as e:
                print(f"❌ Erreur ancien parser: {e}")
                old_entities = []
                old_time = 0

            # =================== COMPARAISON ENTITÉ PAR ENTITÉ ===================
            print(f"\n📊 COMPARAISON ENTITÉ PAR ENTITÉ:")
            print("-" * 50)

            await self._compare_entities_detailed(new_entities, old_entities, filename)

            # =================== RÉSUMÉ PERFORMANCE ===================
            print(f"\n⚡ RÉSUMÉ PERFORMANCE:")
            print("-" * 30)
            print(f"🆕 Nouveau: {len(new_entities)} entités en {new_time:.3f}s")
            print(f"🔙 Ancien:  {len(old_entities)} entités en {old_time:.3f}s")

            if old_time > 0:
                ratio = new_time / old_time
                improvement = ((old_time - new_time) / old_time * 100)
                print(f"📈 Ratio vitesse: {ratio:.2f}x {'(plus lent)' if ratio > 1 else '(plus rapide)'}")
                print(f"🎯 Amélioration: {improvement:.1f}%")

            entity_ratio = len(new_entities) / max(1, len(old_entities))
            print(f"🏗️ Ratio entités: {entity_ratio:.2f}")

    async def _compare_entities_detailed(self, new_entities: List, old_entities: List, filename: str):
        """Compare les entités en détail"""

        # Créer des maps pour comparaison
        new_by_name = {e.entity_name: e for e in new_entities}
        old_by_name = {e.name: e for e in old_entities}

        all_names = set(new_by_name.keys()) | set(old_by_name.keys())

        print(f"🔍 Comparaison de {len(all_names)} entités uniques:")

        matches = 0
        missing_in_new = 0
        missing_in_old = 0
        line_mismatches = 0

        for name in sorted(all_names):
            new_entity = new_by_name.get(name)
            old_entity = old_by_name.get(name)

            if new_entity and old_entity:
                # Entité présente dans les deux
                matches += 1
                status = "✅"

                # Vérifier les numéros de lignes
                if (new_entity.start_line != old_entity.start_line or
                        new_entity.end_line != old_entity.end_line):
                    line_mismatches += 1
                    status = "⚠️"

                print(f"   {status} {name} ({old_entity.entity_type.value})")
                print(f"      🆕 Nouveau: lignes {new_entity.start_line}-{new_entity.end_line}")
                print(f"      🔙 Ancien:  lignes {old_entity.start_line}-{old_entity.end_line}")

                if status == "⚠️":
                    print(f"      ❗ DIFFÉRENCE DE LIGNES!")

            elif new_entity:
                # Seulement dans le nouveau
                missing_in_old += 1
                print(f"   ➕ {name} - NOUVEAU SEULEMENT")
                print(f"      🆕 Lignes: {new_entity.start_line}-{new_entity.end_line}")

            elif old_entity:
                # Seulement dans l'ancien
                missing_in_new += 1
                print(f"   ➖ {name} - ANCIEN SEULEMENT")
                print(f"      🔙 Lignes: {old_entity.start_line}-{old_entity.end_line}")

        print(f"\n📈 BILAN COMPARAISON:")
        print(f"   ✅ Correspondances: {matches}")
        print(f"   ➕ Nouveau seulement: {missing_in_old}")
        print(f"   ➖ Ancien seulement: {missing_in_new}")
        print(f"   ⚠️ Différences de lignes: {line_mismatches}")

        # Verdict
        if matches == len(old_entities) and missing_in_new == 0 and line_mismatches == 0:
            print(f"   🏆 PARFAIT: Correspondance exacte!")
        elif missing_in_new > 0:
            print(f"   ❌ PROBLÈME: Le nouveau parser manque {missing_in_new} entités")
        elif line_mismatches > 0:
            print(f"   ⚠️ ATTENTION: {line_mismatches} différences de numérotation")
        else:
            print(f"   ✅ BON: Couverture correcte")

    async def test_regression_scenarios(self):
        """Tests de régression sur des scénarios spécifiques"""
        self.print_separator("TESTS DE RÉGRESSION", 2)

        regression_tests = {
            'module_detection': [],
            'function_signature_extraction': [],
            'dependency_tracking': [],
            'type_definition_detection': [],
            'internal_function_detection': []
        }

        for filename, filepath in self.test_files.items():
            print(f"\n🔬 Tests de régression: {filename}")

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    code = f.read()

                entities = self.hybrid_parser.get_entities(code, filepath)

                # Test 1: Détection de modules
                modules = [e for e in entities if e.entity_type == 'module']
                module_test = {
                    'file': filename,
                    'modules_found': len(modules),
                    'module_names': [m.entity_name for m in modules],
                    'passed': len(modules) > 0 if 'module' in code.lower() else len(modules) == 0
                }
                regression_tests['module_detection'].append(module_test)

                # Test 2: Extraction de signatures de fonctions
                functions = [e for e in entities if e.entity_type in ['function', 'subroutine']]
                signatures_found = sum(1 for f in functions if f.signature and f.signature != "Signature not found")
                signature_test = {
                    'file': filename,
                    'functions_found': len(functions),
                    'signatures_extracted': signatures_found,
                    'extraction_rate': signatures_found / max(1, len(functions)),
                    'passed': signatures_found > 0 if functions else True
                }
                regression_tests['function_signature_extraction'].append(signature_test)

                # Test 3: Suivi des dépendances
                total_deps = sum(len(e.dependencies) for e in entities)
                dependency_test = {
                    'file': filename,
                    'entities_with_deps': sum(1 for e in entities if e.dependencies),
                    'total_dependencies': total_deps,
                    'passed': total_deps > 0 if 'use ' in code.lower() else True
                }
                regression_tests['dependency_tracking'].append(dependency_test)

                # Test 4: Détection de types définis
                types = [e for e in entities if e.entity_type == 'type_definition']
                type_test = {
                    'file': filename,
                    'types_found': len(types),
                    'type_names': [t.entity_name for t in types],
                    'passed': len(types) > 0 if 'type ::' in code or 'type,' in code else len(types) == 0
                }
                regression_tests['type_definition_detection'].append(type_test)

                # Test 5: Détection de fonctions internes
                internal_funcs = [e for e in entities if e.parent_entity]
                internal_test = {
                    'file': filename,
                    'internal_functions': len(internal_funcs),
                    'parent_relations': [(e.entity_name, e.parent_entity) for e in internal_funcs],
                    'passed': True  # Toujours passé, c'est un bonus
                }
                regression_tests['internal_function_detection'].append(internal_test)

                print(f"   📦 Modules: {module_test['modules_found']} "
                      f"({'✅' if module_test['passed'] else '❌'})")
                print(f"   📝 Signatures: {signatures_found}/{len(functions)} "
                      f"({'✅' if signature_test['passed'] else '❌'})")
                print(f"   🔗 Dépendances: {total_deps} "
                      f"({'✅' if dependency_test['passed'] else '❌'})")
                print(f"   🏷️ Types: {len(types)} "
                      f"({'✅' if type_test['passed'] else '❌'})")
                print(f"   📍 Fonctions internes: {len(internal_funcs)}")

            except Exception as e:
                print(f"   ❌ Erreur tests régression: {e}")

        self.test_results['regression_tests'] = regression_tests

    async def generate_final_report(self, total_time: float):
        """Génère le rapport final des tests"""
        self.print_separator("RAPPORT FINAL DES TESTS")

        # Calculer les statistiques globales
        total_files = len(self.test_results['files_tested'])
        successful_parses = sum(1 for r in self.test_results['parsing_comparison'].values()
                                if r.get('success', False))

        total_entities = sum(r.get('total_entities', 0)
                             for r in self.test_results['parsing_comparison'].values()
                             if r.get('success', False))

        total_calls = sum(r.get('total_calls_found', 0)
                          for r in self.test_results['function_calls'].values()
                          if 'error' not in r)

        # Statistiques de concepts
        entities_with_concepts = sum(r.get('entities_with_concepts', 0)
                                     for r in self.test_results['concept_detection'].values()
                                     if 'error' not in r)

        if 'dependency_analysis' in self.test_results:
            dep_results = self.test_results['dependency_analysis']
            total_use_deps = 0
            total_call_deps = 0

            for file_result in dep_results.values():
                if 'error' not in file_result:
                    total_use_deps += len(file_result['new_parser']['use_dependencies'])
                    total_call_deps += len(file_result['new_parser']['function_calls'])

            print(f"   🔗 USE dependencies: {total_use_deps}")
            print(f"   📞 Function calls: {total_call_deps}")

        print(f"📊 RÉSUMÉ GLOBAL:")
        print(f"   📁 Fichiers testés: {total_files}")
        print(f"   ✅ Parsing réussi: {successful_parses}/{total_files}")
        print(f"   🏗️ Entités extraites: {total_entities}")
        print(f"   📞 Appels détectés: {total_calls}")
        print(f"   🧠 Entités avec concepts: {entities_with_concepts}")
        print(f"   ⏱️ Temps total: {total_time:.2f}s")

        # Performance moyenne
        if self.test_results['performance_metrics']:
            avg_speed_improvement = sum(r.get('speed_improvement', 0)
                                        for r in self.test_results['performance_metrics'].values()
                                        if 'error' not in r) / max(1, len(self.test_results['performance_metrics']))
            print(f"   🚀 Amélioration moyenne: {avg_speed_improvement:.1f}%")

        # Tests de régression
        regression_summary = {}
        for test_type, tests in self.test_results['regression_tests'].items():
            if tests:
                passed = sum(1 for t in tests if t.get('passed', False))
                regression_summary[test_type] = f"{passed}/{len(tests)}"

        print(f"\n🔬 TESTS DE RÉGRESSION:")
        for test_type, result in regression_summary.items():
            print(f"   {test_type}: {result}")

        # Recommandations
        print(f"\n💡 RECOMMANDATIONS:")
        if successful_parses == total_files:
            print("   ✅ Tous les fichiers parsés avec succès!")
        else:
            print(f"   ⚠️ {total_files - successful_parses} fichiers en échec - analyser les erreurs")

        if total_entities > 0:
            print(f"   ✅ Extraction d'entités fonctionnelle")
        else:
            print(f"   ❌ Problème d'extraction d'entités")

        if entities_with_concepts > 0:
            print(f"   ✅ Détection de concepts opérationnelle")
        else:
            print(f"   ⚠️ Détection de concepts à améliorer")

        # Sauvegarder le rapport
        await self.save_test_report()

    async def save_test_report(self):
        """Sauvegarde le rapport de test en JSON"""
        try:
            report_file = f"hybrid_parser_test_report_{int(time.time())}.json"

            # Convertir les résultats en format sérialisable
            serializable_results = {}
            for key, value in self.test_results.items():
                try:
                    # Test de sérialisation
                    json.dumps(value)
                    serializable_results[key] = value
                except (TypeError, ValueError):
                    # Si ça ne marche pas, convertir en string
                    serializable_results[key] = str(value)

            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            print(f"\n💾 Rapport sauvegardé: {report_file}")

        except Exception as e:
            print(f"❌ Erreur sauvegarde rapport: {e}")

    async def run_interactive_test(self):
        """Mode de test interactif"""
        print("🎮 MODE INTERACTIF - Choisissez votre test:")
        print("1. Test de parsing basique")
        print("2. Analyse détaillée d'un fichier")
        print("3. Comparaison avec ancien parser")
        print("4. Test de performance")
        print("5. Visualisation du graphe de dépendances")
        print("6. Tous les tests")

        choice = input("\nVotre choix (1-5): ").strip()

        if choice == "1":
            await self.test_basic_parsing()
        elif choice == "2":
            filename = input("Nom du fichier (ex: math_utilities.f90): ").strip()
            if filename in self.test_files:
                await self.analyze_single_file(filename)
            else:
                print(f"❌ Fichier {filename} non trouvé")
        elif choice == "3":
            await self.test_performance_comparison()
            input("Test des liens externes... Appuyez sur une touche")
            await self.test_dependency_analysis()

        elif choice == "4":
            await self.benchmark_performance()
        elif choice == "5":
            await self.test_graph_visualization()
        elif choice == "6":
            await self.run_all_tests()
        else:
            print("❌ Choix invalide")

    async def test_graph_visualization(self):
        """Test de visualisation du graphe de dépendances"""
        self.print_separator("VISUALISATION DU GRAPHE DE DÉPENDANCES")

        print("🎨 Collecte des entités depuis tous les fichiers...")

        # Collecter toutes les entités depuis nos fichiers de test
        all_entities = []

        for filename, filepath in self.test_files.items():
            print(f"   📄 Parsing {filename}...")

            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()

            # Récupérer les entités avec notre parser hybride
            entities = self.hybrid_parser.get_entities(code, filepath)
            all_entities.extend(entities)

            print(f"      ✅ {len(entities)} entités extraites")

        print(f"\n📊 Total: {len(all_entities)} entités collectées")

        # Import du visualiseur simplifié
        from fortran_analysis.output.graph_visualizer import create_dependency_visualization_from_parser

        try:
            print("\n🎨 Options de visualisation:")
            print("1. Vue d'ensemble (toutes les entités)")
            print("2. Vue limitée (50 entités principales)")
            print("3. Vue focalisée sur une entité")

            viz_choice = input("Votre choix (1-3): ").strip()

            if viz_choice == "1":
                print("🌐 Génération de la vue complète...")
                output_file = create_dependency_visualization_from_parser(
                    all_entities,
                    "complete_dependencies.html"
                )

            elif viz_choice == "2":
                print("📋 Génération de la vue limitée (50 entités)...")
                # Prendre les 50 plus importantes
                important_entities = sorted(
                    all_entities,
                    key=lambda e: len(e.dependencies) + len(e.called_functions) + (
                        10 if e.entity_type == 'module' else 0),
                    reverse=True
                )[:50]

                output_file = create_dependency_visualization_from_parser(
                    important_entities,
                    "limited_dependencies.html"
                )

            elif viz_choice == "3":
                # Afficher les entités disponibles
                print("\n📋 Entités disponibles:")
                main_entities = [e for e in all_entities if
                                 e.entity_type in ['module', 'program', 'subroutine', 'function']]
                for i, entity in enumerate(main_entities[:10], 1):
                    print(f"   {i}. {entity.entity_name} ({entity.entity_type})")

                entity_name = input("\nNom de l'entité focus: ").strip()

                print(f"🎯 Génération de la vue focalisée sur {entity_name}...")
                output_file = create_dependency_visualization_from_parser(
                    all_entities,
                    f"focused_{entity_name}_dependencies.html",
                    focus_entity=entity_name
                )

            else:
                print("❌ Choix invalide")
                return

            print(f"\n✅ Visualisation générée: {output_file}")
            print(f"🌐 Le fichier s'ouvre automatiquement dans votre navigateur")

        except Exception as e:
            print(f"❌ Erreur lors de la visualisation: {e}")
            import traceback
            traceback.print_exc()

    async def analyze_single_file(self, filename: str):
        """Analyse détaillée d'un seul fichier"""
        filepath = self.test_files[filename]
        self.print_separator(f"ANALYSE DÉTAILLÉE - {filename}")

        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()

        print(f"📄 Fichier: {filepath}")
        print(f"📏 Taille: {len(code)} caractères, {len(code.splitlines())} lignes")

        # Parsing complet
        entities = self.hybrid_parser.get_entities(code, filepath)
        print(f"\n🏗️ {len(entities)} entités extraites:")

        for entity in entities:
            print(f"   📋 {entity.entity_type}: {entity.entity_name}")
            print(f"      Lignes {entity.start_line}-{entity.end_line}")
            if entity.signature:
                print(f"      Signature: {entity.signature}")
            if entity.dependencies:
                print(f"      Dépendances: {', '.join(list(entity.dependencies)[:3])}...")
            if entity.called_functions:
                print(f"      Appels: {', '.join(list(entity.called_functions)[:3])}...")
            print()

    async def benchmark_performance(self):
        """Benchmark de performance détaillé"""
        self.print_separator("BENCHMARK DE PERFORMANCE DÉTAILLÉ")

        iterations = 5
        results = {}

        for filename, filepath in self.test_files.items():
            print(f"\n⚡ Benchmark {filename} ({iterations} itérations):")

            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()

            times = []
            for i in range(iterations):
                start_time = time.time()
                entities = self.hybrid_parser.get_entities(code, filepath)
                execution_time = time.time() - start_time
                times.append(execution_time)
                print(f"   Itération {i + 1}: {execution_time:.3f}s ({len(entities)} entités)")

            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            results[filename] = {
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'times': times
            }

            print(f"   📊 Moyenne: {avg_time:.3f}s")
            print(f"   ⚡ Min: {min_time:.3f}s, Max: {max_time:.3f}s")


# Point d'entrée principal
async def main():
    """Point d'entrée principal des tests"""

    # Vérifier les imports
    print("🔧 Vérification des dépendances...")
    dependencies = {
        'f2py': True,
        'fparser': True,
        'ancien_parser': OLD_PARSER_AVAILABLE
    }

    try:
        import numpy.f2py
        print("✅ f2py disponible")
    except ImportError:
        dependencies['f2py'] = False
        print("❌ f2py non disponible")

    try:
        import fparser
        print("✅ fparser disponible")
    except ImportError:
        dependencies['fparser'] = False
        print("❌ fparser non disponible")

    if OLD_PARSER_AVAILABLE:
        print("✅ Ancien parser disponible pour comparaison")
    else:
        print("⚠️ Ancien parser non disponible")

    if not any([dependencies['f2py'], dependencies['fparser']]):
        print("❌ Aucun parser disponible! Installer numpy et fparser")
        return

    # Lancer les tests
    tester = HybridParserTester()

    # Mode interactif ou automatique
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        await tester.run_interactive_test()
    else:
        await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())