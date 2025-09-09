"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# core/hybrid_fortran_parser.py
"""
Parser Fortran hybride combinant f2py et fparser pour une analyse complète.
Remplace les méthodes d'extraction regex basiques par une analyse AST robuste.
"""
import asyncio
import re
import logging
import tempfile
import os
from collections import defaultdict
from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass, field

from ...CONSTANT import RED
from ..core.entity_manager import UnifiedEntity

MODE_FORTRAN = "f2008"  # f2003

# f2py pour la structure
try:
    import numpy.f2py.crackfortran as crackfortran

    F2PY_AVAILABLE = True
except ImportError:
    F2PY_AVAILABLE = False
    crackfortran = None

# fparser pour l'AST (IMPORTS CORRECTS)
try:
    from fparser.common.readfortran import FortranStringReader, FortranFileReader
    from fparser.two.parser import ParserFactory
    from fparser.two.utils import walk
    from fparser.two.Fortran2003 import (
        Program, Module, Subroutine_Subprogram, Function_Subprogram,
        Call_Stmt, Function_Reference, Name, Use_Stmt, Main_Program,
        Assignment_Stmt, Procedure_Designator, Part_Ref,
        Intrinsic_Function_Reference, Data_Ref, Primary
    )

    FPARSER_AVAILABLE = True
except ImportError:
    FPARSER_AVAILABLE = False
    walk = None

logger = logging.getLogger(__name__)


@dataclass
class ParsingStats:
    """Statistiques de parsing détaillées"""
    f2py_entities: int = 0
    fparser_calls: int = 0
    hybrid_success: int = 0
    parse_errors: int = 0
    fallback_used: int = 0
    total_entities: int = 0
    total_calls: int = 0

    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_entities': self.total_entities,
            'total_calls': self.total_calls,
            'f2py_entities': self.f2py_entities,
            'fparser_calls': self.fparser_calls,
            'hybrid_success': self.hybrid_success,
            'parse_errors': self.parse_errors,
            'fallback_used': self.fallback_used,
            'success_rate': self.hybrid_success / max(1, self.total_entities)
        }


# Définir un type pour la carte des sources pour plus de clarté
SourceMap = List[Tuple[str, int]] # Chaque item est (filepath, original_line_number)


class HybridFortranParser:
    """
    Parser Fortran hybride intelligent combinant f2py et fparser.

    Stratégie :
    1. f2py pour la structure complète (modules, fonctions, types)
    2. fparser pour l'analyse précise des appels de fonctions
    3. Fusion intelligente des résultats
    4. Fallback regex robuste si nécessaire
    """

    def __init__(self, prefer_method: str = "hybrid"):
        self.prefer_method = prefer_method
        self.logger = logging.getLogger(__name__)
        self.stats = ParsingStats()

        # Vérifier les dépendances
        if not F2PY_AVAILABLE and not FPARSER_AVAILABLE:
            raise ImportError(
                "Ni f2py ni fparser disponibles. "
                "Installez: pip install numpy fparser"
            )

        # Configuration des intrinsèques Fortran à filtrer
        self.fortran_intrinsics = {
            'sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'log10',
            'abs', 'max', 'min', 'sum', 'product', 'size', 'len',
            'real', 'int', 'dble', 'cmplx', 'mod', 'modulo',
            'allocated', 'associated', 'present', 'trim', 'adjustl'
        }

        # Mots-clés Fortran à éviter
        self.fortran_keywords = {
            'intent', 'in', 'out', 'inout', 'parameter', 'dimension',
            'allocatable', 'pointer', 'target', 'optional', 'save',
            'public', 'private', 'protected', 'volatile', 'asynchronous'
        }

    def parse_file(self, filepath: str, ontology_manager=None) -> Tuple[List[UnifiedEntity], str, SourceMap]:
        """
        Analyse un fichier Fortran et retourne :
        1. Les entités avec des numéros de ligne RELATIFS au code aplati.
        2. Le code source complet et pré-traité.
        3. La carte des sources pour la correction des lignes.
        """
        self.logger.info(f"🚀 Analyse du fichier : {filepath}")

        full_code, source_map = self._preprocess_fortran_includes(filepath)
        if not full_code:
            self.logger.error(f"Le fichier {filepath} est vide ou n'a pas pu être lu.")
            return [], "", []

        filename = os.path.basename(filepath)
        entities = self.parse_fortran_code(full_code, filename, ontology_manager)

        # NE PAS corriger les lignes ici. Laisser le consommateur le faire.
        return entities, full_code, source_map


    def _preprocess_fortran_includes(self, filepath: str, processed_files: Optional[Set[str]] = None) -> Tuple[
        str, SourceMap]:
        """
        Lit un fichier Fortran, gère les 'include', et retourne :
        1. Le code source aplati.
        2. Une "source map" qui mappe chaque ligne du code aplati à son origine.
        """
        if processed_files is None:
            processed_files = set()

        abs_filepath = os.path.abspath(filepath)
        if abs_filepath in processed_files:
            return "", []
        processed_files.add(abs_filepath)

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except FileNotFoundError:
            self.logger.warning(f"Fichier include non trouvé : {filepath}")
            return "", []

        base_dir = os.path.dirname(filepath)
        output_code = ""
        source_map: SourceMap = []  # Initialiser la carte des sources
        include_pattern = re.compile(r"^\s*include\s+['\"]([^'\"]+)['\"]", re.IGNORECASE)

        for line_num, line in enumerate(lines, 1):
            match = include_pattern.match(line.strip())
            if match:
                include_filename = match.group(1)
                include_filepath = os.path.join(base_dir, include_filename)
                self.logger.info(f"  -> Inclusion de '{include_filepath}'...")

                # Appel récursif qui retourne le code ET la carte de l'inclus
                included_code, included_map = self._preprocess_fortran_includes(include_filepath, processed_files)

                output_code += included_code
                source_map.extend(included_map)  # Ajouter la carte de l'inclus à la carte principale
            else:
                output_code += line
                source_map.append((abs_filepath, line_num))  # Ajouter l'origine de cette ligne

        return output_code, source_map

    def parse_fortran_code(self, code: str, filename: str = "unknown.f90", ontology_manager=None) -> List[
        UnifiedEntity]:
        """
        Point d'entrée principal pour le parsing du code Fortran.
        Stratégie: fparser-first, puis fallback.
        """
        self.stats = ParsingStats()  # Reset stats
        cleaned_code = self._preprocess_code(code)

        entities = []
        parse_method = "none"

        # Stratégie 1 (et la meilleure) : Analyse AST complète avec fparser
        if FPARSER_AVAILABLE:
            try:
                # Utilise la nouvelle méthode robuste
                entities = self._parse_with_fparser_ast(cleaned_code, filename, ontology_manager)
                parse_method = 'fparser_ast'
            except Exception as e:
                print(f"{RED} La méthode principale fparser_ast a échoué: {e}")
                entities = []  # Assurer que la liste est vide pour le fallback
        else:
            self.logger.warning("fparser n'est pas disponible.")

        # Stratégie 2 (Fallback) : f2py pour une structure basique si fparser a échoué
        if not entities and F2PY_AVAILABLE:
            self.logger.warning("fparser_ast n'a retourné aucune entité, tentative avec f2py.")
            try:
                entities = self._parse_with_f2py_only(cleaned_code, filename, ontology_manager)
                parse_method = 'f2py_fallback'
                self.stats.fallback_used += 1
            except Exception as e:
                self.logger.error(f"Le fallback f2py a aussi échoué: {e}")
                entities = []

        # Stratégie 3 (Dernier recours) : Regex
        if not entities:
            self.logger.error("Toutes les méthodes de parsing structuré ont échoué. Utilisation du fallback regex.")
            entities = self._regex_fallback(cleaned_code, filename)
            parse_method = 'regex_ultimate_fallback'
            self.stats.fallback_used += 1

        # Post-traitement et stats
        entities = self._post_process_entities(entities)
        self.stats.total_entities = len(entities)
        self.stats.total_calls = sum(len(e.called_functions) for e in entities)

        self.logger.info(f"✅ Parsing terminé via '{parse_method}': {len(entities)} entités, "
                         f"{self.stats.total_calls} appels détectés.")

        return entities

    def _preprocess_code(self, code: str, clean_empty_lines: bool = False) -> str:
        """Nettoie et normalise le code avant parsing"""
        # Normaliser les fins de ligne
        code = code.replace('\r\n', '\n').replace('\r', '\n')

        if not clean_empty_lines:
            return code

        # Supprimer les lignes vides excessives
        lines = code.split('\n')
        cleaned_lines = []
        prev_empty = False

        for line in lines:
            is_empty = not line.strip()
            if not (is_empty and prev_empty):  # Éviter les lignes vides consécutives
                cleaned_lines.append(line)
            prev_empty = is_empty

        return '\n'.join(cleaned_lines)

    def _parse_hybrid_corrected(self, code: str, filename: str, ontology_manager=None) -> List[UnifiedEntity]:
        """
        Méthode hybride corrigée avec fusion intelligente
        """
        self.logger.info(f"🔬 Parsing hybride de {filename}")

        # PHASE 1 : f2py pour la structure
        f2py_entities = []
        if F2PY_AVAILABLE:
            try:
                f2py_entities = self._parse_with_f2py_only(code, filename, ontology_manager)
                self.stats.f2py_entities = len(f2py_entities)
                self.logger.debug(f"f2py: {len(f2py_entities)} entités structurelles")
            except Exception as e:
                self.logger.warning(f"f2py parsing failed: {e}")
                self.stats.parse_errors += 1

        # PHASE 2 : fparser pour les appels détaillés
        function_calls_map = {}
        if FPARSER_AVAILABLE:
            try:
                function_calls_map = self._extract_calls_with_fparser_improved(code, filename)
                total_calls = sum(len(calls) for calls in function_calls_map.values())
                self.stats.fparser_calls = total_calls
                self.logger.debug(f"fparser: {total_calls} appels dans {len(function_calls_map)} entités")
            except Exception as e:
                self.logger.warning(f"fparser analysis failed: {e}")
                self.stats.parse_errors += 1

        # PHASE 3 : Fusion intelligente
        if f2py_entities and function_calls_map:
            entities = self._merge_f2py_and_fparser_corrected(f2py_entities, function_calls_map)
            self.stats.hybrid_success = len(entities)
            self.logger.info(f"✅ Fusion réussie: {len(entities)} entités")

        elif f2py_entities:
            entities = f2py_entities
            self.logger.info("📊 f2py seulement")

        elif function_calls_map:
            entities = self._create_entities_from_fparser_only(function_calls_map, code)
            self.logger.info("🔍 fparser seulement")

        else:
            entities = self._regex_fallback(code, filename)
            self.stats.fallback_used = len(entities)
            self.logger.warning("⚠️ Fallback regex utilisé")

        return entities

    def _extract_calls_with_fparser_improved(self, code: str, filename: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extraction améliorée des appels avec fparser
        """
        function_calls_map = defaultdict(list)

        try:
            reader = FortranStringReader(code)
            parser = ParserFactory().create(std=MODE_FORTRAN)
            ast = parser(reader)

            current_entity = "global"
            entity_stack = ["global"]

            for node in walk(ast):
                # 1. Tracking des entités courantes
                if isinstance(node, Module):
                    entity_name = self._get_entity_name_from_node(node)
                    if entity_name:
                        current_entity = entity_name
                        entity_stack = [entity_name]
                        function_calls_map.setdefault(current_entity, set())

                elif isinstance(node, (Subroutine_Subprogram, Function_Subprogram)):
                    entity_name = self._get_entity_name_from_node(node)
                    if entity_name:
                        current_entity = entity_name
                        if len(entity_stack) > 1:
                            entity_stack.append(current_entity)
                        else:
                            entity_stack = [entity_stack[0], current_entity]
                        function_calls_map.setdefault(current_entity, set())

                elif isinstance(node, Main_Program):
                    entity_name = self._get_entity_name_from_node(node) or "main_program"
                    current_entity = entity_name
                    entity_stack = [entity_name]
                    function_calls_map.setdefault(current_entity, set())

                # 2. Extraction des appels - VERSION AMÉLIORÉE
                elif isinstance(node, Call_Stmt):
                    called_name = self._extract_call_name(node)
                    if called_name and self._is_valid_call_name(called_name):
                        line, _ = self._get_line_numbers(node)
                        function_calls_map[current_entity].append({'name': called_name, 'line': line})
                        self.logger.debug(f"CALL: {current_entity} → {called_name}")

                elif isinstance(node, Function_Reference):
                    called_name = self._extract_function_reference_name(node)
                    if called_name and self._is_valid_call_name(called_name):
                        line, _ = self._get_line_numbers(node)
                        function_calls_map[current_entity].append({'name': called_name, 'line': line})
                        self.logger.debug(f"Function: {current_entity} → {called_name}")

                elif isinstance(node, Assignment_Stmt):
                    calls_in_assignment = self._extract_calls_from_assignment(node)
                    line, _ = self._get_line_numbers(node)
                    for called_name in calls_in_assignment:
                        if self._is_valid_call_name(called_name):
                            function_calls_map[current_entity].append({'name': called_name, 'line': line})
                            self.logger.debug(f"Assignment: {current_entity} → {called_name}")

        except Exception as e:
            self.logger.error(f"fparser extraction error: {e}")
            self.stats.parse_errors += 1

        return function_calls_map

    def _merge_f2py_and_fparser_corrected(self, f2py_entities: List[UnifiedEntity],
                                          calls_map: Dict[str, List[Dict[str, Any]]]) -> List[UnifiedEntity]:
        """Fusion intelligente entre structure f2py et appels fparser"""
        merged_entities = []
        matched_fparser_entities = set()

        for f2py_entity in f2py_entities:
            # CORRECTION: Copier l'UnifiedEntity existante
            merged_entity = UnifiedEntity(
                entity_name=f2py_entity.entity_name,
                entity_type=f2py_entity.entity_type,
                start_line=f2py_entity.start_line,
                end_line=f2py_entity.end_line,
                filepath=f2py_entity.filepath,
                filename=f2py_entity.filename,
                parent_entity=f2py_entity.parent_entity,
                dependencies=f2py_entity.dependencies.copy(),
                called_functions=f2py_entity.called_functions.copy(),
                signature=f2py_entity.signature,
                source_method='hybrid',
                confidence=0.9
            )

            # Stratégies de matching (reste identique)
            entity_calls = []

            if f2py_entity.entity_name in calls_map:
                entity_calls.extend(calls_map[f2py_entity.entity_name])
                matched_fparser_entities.add(f2py_entity.entity_name)

            # Match fuzzy...
            for fparser_name, calls in calls_map.items():
                if fparser_name in matched_fparser_entities:
                    continue
                if self._names_are_similar(f2py_entity.entity_name, fparser_name):
                    entity_calls.extend(calls)
                    matched_fparser_entities.add(fparser_name)
                    break

            merged_entity.called_functions = entity_calls
            merged_entities.append(merged_entity)

        # Entités orphelines
        for fparser_name, calls in calls_map.items():
            if fparser_name not in matched_fparser_entities:
                # CORRECTION: Utiliser UnifiedEntity.from_parser_result
                orphan_entity = UnifiedEntity.from_parser_result(
                    name=fparser_name,
                    entity_type='unknown_procedure',
                    start_line=1,
                    end_line=100,
                    source_method='fparser_orphan',
                    confidence=0.6
                )
                orphan_entity.called_functions = calls
                merged_entities.append(orphan_entity)

        return merged_entities

    def _names_are_similar(self, name1: str, name2: str) -> bool:
        """Détermine si deux noms d'entités sont similaires"""
        name1_lower = name1.lower()
        name2_lower = name2.lower()

        # Match exact
        if name1_lower == name2_lower:
            return True

        # L'un contient l'autre
        if name1_lower in name2_lower or name2_lower in name1_lower:
            return True

        # Match sur signature (pour subroutines/functions)
        if (f"subroutine {name1_lower}" in name2_lower or
                f"function {name1_lower}" in name2_lower or
                f"subroutine {name2_lower}" in name1_lower or
                f"function {name2_lower}" in name1_lower):
            return True

        return False

    def _post_process_entities(self, entities: List[UnifiedEntity]) -> List[UnifiedEntity]:
        """
        Post-traitement des entités extraites.
        VERSION ROBUSTE : gère à la fois les appels sous forme de 'str' et de 'dict'.
        """
        processed = []

        for entity in entities:
            clean_calls_list = []

            # Si called_functions n'est pas une liste, on ne fait rien pour éviter les erreurs.
            if not isinstance(entity.called_functions, list):
                # Optionnel: loguer un avertissement si le type est inattendu
                if entity.called_functions:  # Si ce n'est pas juste une liste/set vide
                    self.logger.warning(f"Type inattendu pour called_functions dans {entity.entity_name}: "
                                        f"{type(entity.called_functions)}. Appels ignorés pour cette entité.")
                entity.called_functions = []  # On s'assure que c'est une liste vide

            else:
                for call_info in entity.called_functions:
                    call_name = None

                    # ▼▼▼ LOGIQUE DE TOLÉRANCE ▼▼▼
                    if isinstance(call_info, dict):
                        # C'est le nouveau format, on extrait le nom
                        call_name = call_info.get('name')
                    elif isinstance(call_info, str):
                        # C'est l'ancien format, on utilise la chaîne directement
                        call_name = call_info
                        # On convertit au nouveau format pour la cohérence
                        call_info = {'name': call_name, 'line': 0}  # Ligne 0 = inconnue
                    # ▲▲▲ FIN DE LA LOGIQUE DE TOLÉRANCE ▲▲▲

                    if self._is_valid_call_name(call_name):
                        # On ajoute toujours le format dictionnaire
                        clean_calls_list.append(call_info)

                entity.called_functions = clean_calls_list

            clean_deps_list = []
            if not isinstance(entity.dependencies, list):
                # Convertir ancien format (set/string) vers nouveau format (list de dict)
                if isinstance(entity.dependencies, set):
                    for dep_name in entity.dependencies:
                        clean_deps_list.append({'name': dep_name, 'line': 0})
                entity.dependencies = clean_deps_list
            else:
                # Normaliser le format existant
                for dep_info in entity.dependencies:
                    if isinstance(dep_info, dict):
                        dep_name = dep_info.get('name')
                        if 'line' not in dep_info:
                            dep_info['line'] = 0
                    elif isinstance(dep_info, str):
                        dep_name = dep_info
                        dep_info = {'name': dep_name, 'line': 0}

                    if dep_name:  # Validation basique
                        clean_deps_list.append(dep_info)

                entity.dependencies = clean_deps_list


            if not entity.signature and entity.entity_type in ['subroutine', 'function']:
                entity.signature = f"{entity.entity_type} {entity.name}(...)"

            processed.append(entity)

        return processed

    def _extract_function_reference_name(self, node) -> Optional[str]:
        """Extrait le nom d'une référence de fonction."""
        try:
            from fparser.two.Fortran2003 import Name
            if hasattr(node, 'children'):
                for child in node.children:
                    if isinstance(child, Name):
                        return str(child)
        except:
            pass
        return None

    def _extract_call_name(self, call_node) -> Optional[str]:
        """
        Extrait le nom de la procédure depuis un nœud Call_Stmt de manière ROBUSTE.
        Cette version utilise `walk` pour trouver le premier nœud `Name`, ce qui est
        beaucoup plus fiable que de supposer une structure de children fixe.
        """
        try:
            # fparser.two.utils.walk est nécessaire pour cette approche
            from fparser.two.utils import walk
            from fparser.two.Fortran2003 import Name, Procedure_Designator

            # L'approche la plus fiable est de parcourir les sous-nœuds du Call_Stmt
            # et de prendre le PREMIER nom que l'on trouve. C'est universellement
            # le nom de la routine appelée.
            for sub_node in walk(call_node):
                if isinstance(sub_node, Name):
                    # On a trouvé le nom. On le retourne et on arrête immédiatement
                    # pour ne pas attraper les noms des arguments.
                    return str(sub_node)

        except ImportError:
            # Fallback si fparser n'est pas disponible (ne devrait pas arriver ici)
            self.logger.error("fparser.two.utils.walk non trouvé. L'extraction des appels sera limitée.")
        except Exception as e:
            self.logger.debug(f"Erreur lors de l'extraction robuste du nom de l'appel : {e}")

        # Si aucune méthode n'a fonctionné, on retourne None.
        return None

    # ===== Méthodes fallback =====

    def _parse_with_f2py_only(self, code: str, filename: str, ontology_manager=None) -> List[UnifiedEntity]:
        """Parse avec f2py uniquement (structure complète)"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.f90', delete=False) as tmp_file:
            tmp_file.write(code)
            tmp_filename = tmp_file.name

        try:
            f2py_tree = crackfortran.crackfortran([tmp_filename])
            entities = []

            for item in f2py_tree:
                entities.extend(self._extract_f2py_entities(item, None, ontology_manager, filename))

            return entities

        finally:
            os.unlink(tmp_filename)

    def _extract_f2py_entities(self, item: Dict[str, Any], parent_name: Optional[str] = None, ontology_manager=None,
                               filepath: str = "unknown.f90") -> List[UnifiedEntity]:
        """Extrait les entités depuis un arbre f2py"""
        entities = []

        name = item.get('name', 'unknown')
        block_type = item.get('block', 'unknown')

        # CORRECTION: Utiliser UnifiedEntity.from_parser_result
        entity = UnifiedEntity.from_parser_result(
            name=name,
            entity_type=block_type,
            start_line=self.safe_int_conversion(item.get('from', 1)),
            end_line=self.safe_int_conversion(item.get('upto', 1)),
            filepath=filepath,
            parent_entity=parent_name,
            source_method='f2py',
            confidence=0.8
        )

        entity.filename = os.path.basename(filepath) if filepath else ""

        # Extraire USE statements
        if 'use' in item:
            uses = item['use']
            if isinstance(uses, dict):
                entity.dependencies.update(uses.keys())

        # Construire signature
        if block_type in ['subroutine', 'function']:
            args = item.get('args', [])
            entity.signature = f"{block_type} {name}({', '.join(args)})"

        # Concepts
        if ontology_manager and block_type in ['module', 'subroutine', 'function']:
            entity.detected_concepts.append({'label': 'concept_detected_via_manager', 'confidence': 0.99})

        entities.append(entity)

        # Récursion pour les enfants
        if 'body' in item:
            for child_item in item['body']:
                child_entities = self._extract_f2py_entities(child_item, name, ontology_manager, filepath)
                entities.extend(child_entities)

        return entities

    async def _parse_with_f2py_unified(self, code: str, filename: str, ontology_manager=None) -> List[UnifiedEntity]:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.f90', delete=False) as tmp_file:
            tmp_file.write(code)
            tmp_filename = tmp_file.name

        loop = asyncio.get_event_loop()
        try:
            # crackfortran est synchrone, on l'exécute dans un thread pour ne pas bloquer l'event loop.
            f2py_tree = await loop.run_in_executor(None, lambda: crackfortran.crackfortran([tmp_filename]))
            entities = []
            for item in f2py_tree:
                unified_entities = await self._extract_f2py_unified(item, None, code, ontology_manager, filename)
                entities.extend(unified_entities)

            return entities
        finally:
            os.unlink(tmp_filename)

    def safe_int_conversion(self, value, default=1):
        """Conversion sécurisée depuis les données f2py"""
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                # Gérer le cas où c'est un nom de fichier
                if '/' in value or '.f90' in value:
                    return default
                return int(value)
            except ValueError:
                return default
        return default

    async def _extract_f2py_unified(self, item: Dict, parent_name: Optional[str] = None, full_code: str = "",
                                         ontology_manager=None, filename: str = "unknown.f90") -> List[UnifiedEntity]:
        """Version asynchrone qui gère la détection de concepts."""
        entities = []
        name = item.get('name', 'unknown')
        block_type = item.get('block', 'unknown')

        # === CRÉER UnifiedEntity DIRECTEMENT ===
        entity = UnifiedEntity.from_parser_result(
            name=name,
            entity_type=block_type,
            start_line=self.safe_int_conversion(item.get('from', 1)),
            end_line=self.safe_int_conversion(item.get('upto', 1)),
            filepath=filename,
            parent_entity=parent_name,
            source_method='f2py',
            confidence=0.8
        )

        entity.filename = os.path.basename(filename) if filename else ""

        # === REMPLIR LES DÉPENDANCES ===
        if 'use' in item and isinstance(item['use'], dict):
            entity.dependencies.update(item['use'].keys())

        # === CONSTRUIRE SIGNATURE ===
        if block_type in ['subroutine', 'function']:
            args = item.get('args', [])
            entity.signature = f"{block_type} {name}({', '.join(args)})"


        entities.append(entity)

        # Récursion pour les enfants
        if 'body' in item:
            child_tasks = []
            for child_item in item['body']:
                children = await self._extract_f2py_unified(child_item, name, full_code, ontology_manager, filename)
                entities.extend(children)

            return entities

    def _parse_with_fparser_ast(self, code: str, filename: str, ontology_manager=None) -> List[UnifiedEntity]:
        """
        Parser fparser ULTRA-COMPLET - Détecte TOUTES les entités Fortran et assigne
        les dépendances et les appels de manière robuste en se basant sur les numéros de ligne.
        """
        self.logger.info(f"🔬 Parser fparser with ast pour {filename}")

        try:
            from fparser.common.readfortran import FortranStringReader
            from fparser.two.parser import ParserFactory
            from fparser.two.utils import walk, FortranSyntaxError
            from fparser.two.Fortran2003 import (
                Module, Subroutine_Subprogram, Function_Subprogram, Main_Program,
                Use_Stmt, Call_Stmt, Function_Reference, Intrinsic_Function_Reference,
                Name, Derived_Type_Def, Assignment_Stmt, Primary, Data_Ref,
                Type_Declaration_Stmt, Entity_Decl, Attr_Spec, Interface_Block,
                Access_Stmt
            )

            reader = FortranStringReader(code, ignore_comments=False)
            parser = ParserFactory().create(std="f2008")

            try:
                ast = parser(reader)
            except (FortranSyntaxError, Exception) as e:
                self.logger.warning(f"Erreur fparser-ast ({e}), fallback vers regex")
                return self._regex_fallback(code, filename)

            if ast is None:
                self.logger.warning("fparser a retourné un AST vide. Fallback.")
                return self._regex_fallback(code, filename)

            # ======================================================================
            # PHASE 1 : COLLECTE EXHAUSTIVE DES ENTITÉS ET DES RELATIONS AVEC LIGNES
            # ======================================================================
            entities_data = []  # Pour les entités (modules, subroutines, etc.)
            uses_data = []  # Pour les 'USE' avec leur numéro de ligne
            calls_data = []  # Pour les 'CALL' et références de fonction avec leur ligne
            module_default_access = 'public'  # Le défaut Fortran pour un module
            explicit_access_map = {}  # Pour stocker les déclarations `PUBLIC :: a, b`

            for node in walk(ast):
                start_line, end_line = self._get_line_numbers(node)

                # === 1. ENTITÉS PRINCIPALES (Modules, Subroutines, Fonctions, Programmes) ===
                if isinstance(node, Module):
                    entity_name = self._get_entity_name_from_node(node)
                    module_default_access = 'public'
                    explicit_access_map = {}
                    if entity_name:
                        entities_data.append({
                            'name': entity_name, 'type': 'module',
                            'start_line': start_line, 'end_line': end_line, 'node': node
                        })
                elif isinstance(node, (Subroutine_Subprogram, Function_Subprogram)):
                    entity_name = self._get_entity_name_from_node(node)
                    if entity_name:
                        entity_type = 'subroutine' if isinstance(node, Subroutine_Subprogram) else 'function'
                        #  Extraction de la signature
                        arguments, return_type, signature_line = self._extract_rich_signature_data(node)
                        entities_data.append({
                            'name': entity_name, 'type': entity_type,
                            'start_line': start_line, 'end_line': end_line, 'node': node,
                            'arguments': arguments,  # <-- Ajout
                            'return_type': return_type,  # <-- Ajout
                            'signature': signature_line  # <-- Ajout
                        })

                elif isinstance(node, Main_Program):
                    entity_name = self._get_entity_name_from_node(node) or "main_program"
                    entities_data.append({
                        'name': entity_name, 'type': 'program',
                        'start_line': start_line, 'end_line': end_line, 'node': node
                    })

                # === 2. TYPES DÉFINIS ===
                elif isinstance(node, Derived_Type_Def):
                    type_name = self._get_type_name_from_node(node)
                    if type_name:
                        entities_data.append({
                            'name': type_name, 'type': 'type_definition',
                            'start_line': start_line, 'end_line': end_line, 'node': node
                        })

                # === 3. CONSTANTES PARAMETER ET VARIABLES ===
                elif isinstance(node, Type_Declaration_Stmt):
                    declared_entities = self._extract_declared_entities(node, start_line, end_line)
                    entities_data.extend(declared_entities)

                # === 4. INTERFACES ===
                elif isinstance(node, Interface_Block):
                    interface_name = self._get_interface_name(node)
                    if interface_name:
                        entities_data.append({
                            'name': interface_name, 'type': 'interface',
                            'start_line': start_line, 'end_line': end_line, 'node': node
                        })

                # === 5. DÉPENDANCES 'USE' ===
                elif isinstance(node, Use_Stmt):
                    mod_name = self._extract_use_module_name(node)
                    if mod_name:
                        use_line, _ = self._get_line_numbers(node)
                        uses_data.append({'name': mod_name, 'line': use_line})

                # === 6. APPELS DE FONCTIONS ET SUBROUTINES ===
                elif isinstance(node, Call_Stmt):
                    called_name = self._extract_call_name(node)
                    if called_name and self._is_valid_call_name(called_name):
                        call_line, _ = self._get_line_numbers(node)
                        calls_data.append({'name': called_name, 'line': call_line})

                elif isinstance(node, Function_Reference):
                    called_name = self._extract_function_reference_name(node)
                    if called_name and self._is_valid_call_name(called_name):
                        ref_line, _ = self._get_line_numbers(node)
                        calls_data.append({'name': called_name, 'line': ref_line})

                elif isinstance(node, Assignment_Stmt):
                    calls_in_assignment = self._extract_calls_from_assignment(node)
                    assign_line, _ = self._get_line_numbers(node)
                    for called_name in calls_in_assignment:
                        if self._is_valid_call_name(called_name):
                            calls_data.append({'name': called_name, 'line': assign_line})

                elif isinstance(node, Access_Stmt):

                    # On détermine si c'est PUBLIC ou PRIVATE en regardant le premier item

                    access_spec = str(node.items[0]).upper()

                    # Si c'est 'PRIVATE' ou 'PUBLIC' sans liste, ça change le défaut du module

                    if len(node.items) == 1:

                        if access_spec == 'PRIVATE':
                            module_default_access = 'private'

                        # (Pas besoin de gérer PUBLIC car c'est déjà le défaut)

                    else:  # Sinon, ça déclare des entités spécifiques

                        access_level_to_set = 'private' if access_spec == 'PRIVATE' else 'public'

                        # Les entités déclarées sont les items suivants

                        for item in node.items[1:]:
                            explicit_access_map[str(item).lower()] = access_level_to_set

            # ======================================================================
            # PHASE 2 : DÉTERMINATION DES PARENTS
            # ======================================================================
            self._determine_entity_parents(entities_data)
            self.logger.info(f"Trouvé {len(entities_data)} entités (dont constantes et variables)")

            # ======================================================================
            # PHASE 3 : CRÉATION DES OBJETS UNIFIEDENTITY
            # ======================================================================
            entities_map = {}
            for entity_data in entities_data:
                name = entity_data['name']
                if name in entities_map:
                    continue

                final_access = entity_data.get('access_level')  # Priorité 1
                if final_access is None:
                    final_access = explicit_access_map.get(name.lower())  # Priorité 2
                if final_access is None:
                    # Pour les entités dans un module, appliquer le défaut
                    if entity_data.get('parent'):
                        final_access = module_default_access

                entity = UnifiedEntity.from_parser_result(
                    name=name,
                    entity_type=entity_data['type'],
                    start_line=entity_data['start_line'],
                    end_line=entity_data['end_line'],
                    filepath=filename,
                    source_method='fparser_ast_complete',
                    confidence=1.0,
                    access_level=final_access,
                    arguments=entity_data.get('arguments', []),
                    return_type=entity_data.get('return_type'),
                    signature=entity_data.get('signature', '')  # Signature simple
                )
                if 'parent' in entity_data and entity_data['parent']:
                    entity.parent_entity = entity_data['parent']
                if 'value' in entity_data:
                    entity.signature = f"{entity_data['type']} {name} = {entity_data['value']}"

                entities_map[name] = entity

            all_entities = list(entities_map.values())

            # ======================================================================
            # PHASE 4 : ASSIGNATION ROBUSTE DES DÉPENDANCES ET APPELS
            # ======================================================================
            self._assign_dependencies_and_calls(all_entities, uses_data, calls_data)

            self.stats.hybrid_success = len(all_entities)
            self.logger.info(f"✅ Parser fortran : {len(all_entities)} entités")
            return all_entities

        except Exception as e:
            self.logger.error(f"Erreur majeure parser: {e}", exc_info=True)
            self.stats.parse_errors += 1
            return self._regex_fallback(code, filename)

    def _extract_rich_signature_data(self, subprogram_node) -> Tuple[List[Dict], Optional[str], str]:
        """
        Extrait les arguments, le type de retour et la ligne de signature
        d'un nœud Subroutine_Subprogram ou Function_Subprogram.

        Cette version utilise `walk` pour une robustesse maximale.
        """
        from fparser.two.Fortran2003 import (
            Subroutine_Stmt, Function_Stmt, Type_Declaration_Stmt,
            Entity_Decl, Attr_Spec, Name, Prefix_Spec
        )
        from fparser.two.utils import walk

        # --- Initialisation des valeurs par défaut ---
        arguments = []
        return_type = None
        signature_line = "Signature not found"
        arg_names_ordered = []

        # ======================================================================
        # ÉTAPE 1: Trouver le nœud de déclaration et extraire les noms d'arguments
        # ======================================================================
        # On parcourt le nœud de la sous-programmation pour trouver la déclaration
        # principale (ex: `subroutine my_sub(...)`). Il ne devrait y en avoir qu'une.
        stmt_node = None
        for node in walk(subprogram_node):
            if isinstance(node, (Subroutine_Stmt, Function_Stmt)):
                stmt_node = node
                break  # On a trouvé, on arrête de chercher.

        if not stmt_node:
            self.logger.warning(f"Impossible de trouver le nœud de déclaration pour {subprogram_node}.")
            return [], None, signature_line

        # On a trouvé le nœud. On peut maintenant extraire les informations de base.
        signature_line = str(stmt_node)

        # fparser fournit un attribut `.args` qui est une liste de chaînes de caractères. C'est la méthode la plus fiable.
        if hasattr(stmt_node, 'args'):
            arg_names_ordered = [name.lower() for name in stmt_node.args]

        # Si pas d'arguments, on peut s'arrêter ici.
        if not arg_names_ordered:
            return [], None, signature_line

        # Préparation du dictionnaire qui stockera les détails de chaque argument.
        arg_details = {name: {'name': name, 'type': 'unknown', 'attributes': []} for name in arg_names_ordered}

        # ======================================================================
        # ÉTAPE 2: Extraire le type de retour (pour les fonctions)
        # ======================================================================
        if isinstance(stmt_node, Function_Stmt) and hasattr(stmt_node, 'prefix') and stmt_node.prefix:
            # Le préfix contient le type de la fonction (ex: 'real(dp) function ...')
            # On nettoie les mots-clés comme 'recursive' pour ne garder que le type.
            type_parts = []
            prefix_str = str(stmt_node.prefix).lower()
            keywords_to_ignore = ['recursive', 'pure', 'elemental']
            # Simple split pour séparer les mots-clés du type
            for part in prefix_str.split():
                if part not in keywords_to_ignore:
                    type_parts.append(part)
            if type_parts:
                return_type = " ".join(type_parts)

        # ======================================================================
        # ÉTAPE 3: Parcourir TOUT le sous-programme pour trouver les déclarations de type
        # ======================================================================
        # C'est ici que `walk` est crucial. On parcourt tout le `subprogram_node` pour
        # trouver toutes les déclarations de variables (`Type_Declaration_Stmt`).
        for decl_node in walk(subprogram_node):
            if not isinstance(decl_node, Type_Declaration_Stmt):
                continue

            # On a trouvé un `Type_Declaration_Stmt`. Extrayons ses informations.
            # ex: real(dp), intent(in) :: my_var

            # 1. Le type : 'real(dp)'
            type_str = str(decl_node.items[0])

            # 2. Les attributs : ['intent(in)']
            attrs = []
            if decl_node.items[1]:  # Le nœud d'attributs peut être None
                attrs = [str(attr).upper() for attr in decl_node.items[1].items]

            # 3. La liste des variables déclarées avec ce type et ces attributs
            entity_list_node = decl_node.items[-1]
            if hasattr(entity_list_node, 'items'):
                for entity_decl in entity_list_node.items:
                    if isinstance(entity_decl, Entity_Decl):
                        var_name = str(entity_decl.items[0]).lower()

                        # Si cette variable est l'un de nos arguments...
                        if var_name in arg_details:
                            # ... on met à jour ses informations !
                            self.logger.debug(
                                f"Déclaration trouvée pour l'argument '{var_name}': type={type_str}, attrs={attrs}")
                            arg_details[var_name]['type'] = type_str
                            arg_details[var_name]['attributes'].extend(attrs)

        # ======================================================================
        # ÉTAPE 4: Assemblage final
        # ======================================================================
        # On reconstruit la liste dans le bon ordre.
        arguments = [arg_details[name] for name in arg_names_ordered]

        return arguments, return_type, signature_line

    def _assign_dependencies_and_calls(self, all_entities: List,
                                       uses_data: List[Dict], calls_data: List[Dict]) -> None:
        """
        Assigne les dépendances (USE) et les appels (CALL) aux bonnes entités
        en se basant sur les numéros de ligne pour une robustesse maximale.
        """

        def find_owner(line_number: int, candidates: List[UnifiedEntity]) -> Optional[UnifiedEntity]:
            owner = None
            smallest_scope = float('inf')

            for entity in candidates:
                if entity.start_line <= line_number <= entity.end_line:
                    scope_size = entity.end_line - entity.start_line
                    if scope_size < smallest_scope:
                        smallest_scope = scope_size
                        owner = entity
            return owner

        # ==========================================================
        # 1. Traitement des dépendances 'USE' → dependencies (avec lignes)
        # ==========================================================
        self.logger.info(f"Assignation de {len(uses_data)} dépendances 'USE'...")
        for use in uses_data:
            owner = find_owner(use['line'], all_entities)
            if owner:
                # ✅ NOUVELLE APPROCHE : Garder la ligne dans dependencies
                use_info = {'name': use['name'], 'line': use['line']}
                owner.dependencies.append(use_info)  # ← Liste de dict, comme called_functions
                self.logger.debug(
                    f"Dépendance assignée: '{owner.entity_name}' USEs '{use['name']}' (ligne {use['line']})")
            else:
                self.logger.warning(f"Aucun propriétaire trouvé pour 'USE {use['name']}' à la ligne {use['line']}")

        # ==========================================================
        # 2. Traitement des appels → called_functions (inchangé)
        # ==========================================================
        self.logger.info(f"Assignation de {len(calls_data)} appels de fonctions/subroutines...")
        for call in calls_data:
            owner = find_owner(call['line'], all_entities)
            if owner:
                call_info = {'name': call['name'], 'line': call['line']}
                owner.called_functions.append(call_info)
                self.logger.debug(
                    f"Appel assigné: '{owner.entity_name}' CALLs '{call['name']}' (ligne {call['line']})")
            else:
                self.logger.warning(
                    f"Aucun propriétaire trouvé pour l'appel à '{call['name']}' à la ligne {call['line']}")

    def old_old_assign_dependencies_and_calls(self, all_entities: List,
                                       uses_data: List[Dict], calls_data: List[Dict]) -> None:
        """
        Assigne les dépendances (USE) et les appels (CALL) aux bonnes entités
        en se basant sur les numéros de ligne pour une robustesse maximale.

        ✅ CORRECTION : Sépare correctement USE (→ dependencies) et CALL (→ called_functions)
        """

        # Fonction helper (interne à cette méthode) pour trouver l'entité
        # propriétaire d'un numéro de ligne donné.
        def find_owner(line_number: int, candidates: List[UnifiedEntity]) -> Optional[UnifiedEntity]:
            """
            Trouve l'entité la plus "petite" (la plus locale/imbriquée) qui contient
            le numéro de ligne spécifié.
            """
            owner = None
            smallest_scope = float('inf')

            for entity in candidates:
                # Étape 1 : Vérifier si la ligne est dans la portée de l'entité.
                if entity.start_line <= line_number <= entity.end_line:
                    # Étape 2 : Prendre la plus spécifique (plus petite étendue)
                    scope_size = entity.end_line - entity.start_line
                    if scope_size < smallest_scope:
                        smallest_scope = scope_size
                        owner = entity

            return owner

        # ==========================================================
        # 1. Traitement des dépendances 'USE' → dependencies
        # ==========================================================
        self.logger.info(f"Assignation de {len(uses_data)} dépendances 'USE'...")
        for use in uses_data:
            owner = find_owner(use['line'], all_entities)
            if owner:
                # ✅ CORRECTION : USE va dans dependencies, pas called_functions
                owner.dependencies.add(use['name'])  # ← Utiliser .add() car c'est un set
                self.logger.debug(
                    f"Dépendance assignée: '{owner.entity_name}' USEs '{use['name']}' (ligne {use['line']})")
            else:
                self.logger.warning(f"Aucun propriétaire trouvé pour 'USE {use['name']}' à la ligne {use['line']}")

        # ==========================================================
        # 2. Traitement des appels de fonctions et subroutines → called_functions
        # ==========================================================
        self.logger.info(f"Assignation de {len(calls_data)} appels de fonctions/subroutines...")
        for call in calls_data:
            owner = find_owner(call['line'], all_entities)
            if owner:
                # ✅ CORRECTION : CALL garde sa structure dict avec ligne
                call_info = {'name': call['name'], 'line': call['line']}
                owner.called_functions.append(call_info)  # ← Format dict cohérent
                self.logger.debug(
                    f"Appel assigné: '{owner.entity_name}' CALLs '{call['name']}' (ligne {call['line']})")
            else:
                self.logger.warning(
                    f"Aucun propriétaire trouvé pour l'appel à '{call['name']}' à la ligne {call['line']}")

    def old_assign_dependencies_and_calls(self, all_entities: List,
                                       uses_data: List[Dict], calls_data: List[Dict]) -> None:
        """
        Assigne les dépendances (USE) et les appels (CALL) aux bonnes entités
        en se basant sur les numéros de ligne pour une robustesse maximale.

        Cette méthode unifiée remplace les anciennes logiques d'affectation séparées
        et fragiles.
        """

        # Fonction helper (interne à cette méthode) pour trouver l'entité
        # propriétaire d'un numéro de ligne donné.
        def find_owner(line_number: int, candidates: List[UnifiedEntity]) -> Optional[UnifiedEntity]:
            """
            Trouve l'entité la plus "petite" (la plus locale/imbriquée) qui contient
            le numéro de ligne spécifié.
            """
            owner = None
            smallest_scope = float('inf')

            for entity in candidates:
                # Étape 1 : Vérifier si la ligne est dans la portée de l'entité.
                # Cela inclut les entités de haut niveau comme les modules et les
                # entités imbriquées comme les subroutines.
                if entity.start_line <= line_number <= entity.end_line:

                    # Étape 2 : Si plusieurs entités contiennent la ligne (ex: une
                    # subroutine dans un module), on veut la plus spécifique.
                    # On la trouve en choisissant celle qui a la plus petite
                    # étendue de lignes (end_line - start_line).
                    scope_size = entity.end_line - entity.start_line

                    if scope_size < smallest_scope:
                        smallest_scope = scope_size
                        owner = entity

            return owner

        # ==========================================================
        # 1. Traitement des dépendances 'USE'
        # ==========================================================
        self.logger.info(f"Assignation de {len(uses_data)} dépendances 'USE'...")
        for use in uses_data:
            # Pour chaque 'USE', trouver son propriétaire en fonction de sa ligne.
            owner = find_owner(use['line'], all_entities)
            if owner:
                # Assigner la dépendance à l'entité propriétaire.
                owner.called_functions.append(use)
                self.logger.debug(
                    f"Dépendance assignée: '{owner.entity_name}' USEs '{use['name']}' (ligne {use['line']})")
            else:
                self.logger.warning(f"Aucun propriétaire trouvé pour 'USE {use['name']}' à la ligne {use['line']}")

        # ==========================================================
        # 2. Traitement des appels de fonctions et subroutines
        # ==========================================================
        self.logger.info(f"Assignation de {len(calls_data)} appels de fonctions/subroutines...")
        for call in calls_data:
            # La logique est identique : trouver le propriétaire en fonction de la ligne.
            owner = find_owner(call['line'], all_entities)
            if owner:
                # Assigner l'appel à l'entité propriétaire.
                owner.called_functions.append(call['name'])
                self.logger.debug(
                    f"Appel assigné: '{owner.entity_name}' CALLs '{call['name']}' (ligne {call['line']})")
            else:
                self.logger.warning(
                    f"Aucun propriétaire trouvé pour l'appel à '{call['name']}' à la ligne {call['line']}")

    def _extract_declared_entities(self, type_decl_stmt, start_line: int, end_line: int) -> List[Dict]:
        """
        NOUVELLE MÉTHODE : Extrait les entités déclarées (constantes PARAMETER, variables, etc.)
        """
        entities = []

        try:
            from fparser.two.Fortran2003 import Entity_Decl, Attr_Spec, Name

            access_level = None
            # Vérifier si c'est un PARAMETER
            is_parameter = False
            if hasattr(type_decl_stmt, 'children') and len(type_decl_stmt.children) >= 2:
                attr_spec_list = type_decl_stmt.children[1]  # Deuxième enfant = attributs
                if attr_spec_list and hasattr(attr_spec_list, 'children'):
                    for attr_str in attr_spec_list.children:
                        if isinstance(attr_str, Attr_Spec) and str(attr_str).upper() == 'PARAMETER':
                            is_parameter = True
                            break
                        elif attr_str == 'PUBLIC':
                            access_level = 'public'
                            break
                        elif attr_str == 'PRIVATE':
                            access_level = 'private'
                            break

            # Extraire les entités déclarées
            if hasattr(type_decl_stmt, 'children') and len(type_decl_stmt.children) >= 3:
                entity_decl_list = type_decl_stmt.children[-1]  # Dernier enfant = liste des entités

                if hasattr(entity_decl_list, 'children'):
                    for entity_decl in entity_decl_list.children:
                        if isinstance(entity_decl, Entity_Decl):
                            entity_name = self._get_entity_name_from_entity_decl(entity_decl)
                            if entity_name:
                                # Extraire la valeur si c'est initialisé
                                value = self._get_entity_value(entity_decl)

                                entity_type = 'parameter' if is_parameter else 'variable'

                                entity_data = {
                                    'name': entity_name,
                                    'type': entity_type,
                                    'start_line': start_line,
                                    'end_line': end_line,
                                    'node': entity_decl,
                                    'access_level': access_level
                                }

                                if value:
                                    entity_data['value'] = value

                                entities.append(entity_data)

                                self.logger.debug(f"📝 {entity_type}: {entity_name}" + (f" = {value}" if value else ""))

        except Exception as e:
            self.logger.debug(f"Erreur extraction entités déclarées: {e}")

        return entities

    def _get_entity_name_from_entity_decl(self, entity_decl) -> Optional[str]:
        """Extrait le nom depuis un Entity_Decl."""
        try:
            from fparser.two.Fortran2003 import Name

            if hasattr(entity_decl, 'children') and entity_decl.children:
                first_child = entity_decl.children[0]
                if isinstance(first_child, Name):
                    return str(first_child)
        except Exception:
            pass
        return None

    def _get_entity_value(self, entity_decl) -> Optional[str]:
        """Extrait la valeur d'initialisation depuis un Entity_Decl."""
        try:
            if hasattr(entity_decl, 'children') and len(entity_decl.children) >= 4:
                # L'initialisation est typiquement le 4ème enfant
                initialization = entity_decl.children[3]
                if initialization and hasattr(initialization, 'children') and len(initialization.children) >= 2:
                    value_node = initialization.children[1]
                    return str(value_node).replace('_real64', '').replace('_REAL64', '')
        except Exception:
            pass
        return None

    def _get_interface_name(self, interface_node) -> Optional[str]:
        """Extrait le nom d'une interface."""
        try:
            from fparser.two.Fortran2003 import Name

            if hasattr(interface_node, 'children'):
                for child in interface_node.children:
                    if isinstance(child, Name):
                        return str(child)
        except Exception:
            pass
        return None

    def _determine_entity_parents(self, entities_data: List[Dict]) -> None:
        """
        Détermine les parents des entités basé sur l'imbrication des lignes.
        NOUVELLE MÉTHODE pour gérer correctement la hiérarchie.
        """
        # Trier par ligne de début pour traiter dans l'ordre
        entities_data.sort(key=lambda x: x['start_line'])

        for i, entity in enumerate(entities_data):
            entity['parent'] = None

            # Chercher le parent le plus proche (dernière entité qui contient cette entité)
            best_parent = None
            smallest_scope = float('inf')

            for j, potential_parent in enumerate(entities_data):
                if i == j:  # Pas soi-même
                    continue

                # Le parent doit contenir complètement l'entité
                if (potential_parent['start_line'] <= entity['start_line'] and
                        potential_parent['end_line'] >= entity['end_line']):

                    # Prendre le parent avec la plus petite portée (le plus proche)
                    scope_size = potential_parent['end_line'] - potential_parent['start_line']
                    if scope_size < smallest_scope:
                        smallest_scope = scope_size
                        best_parent = potential_parent['name']

            entity['parent'] = best_parent

            if best_parent:
                self.logger.debug(
                    f"📍 Hiérarchie détectée: {entity['name']} (lignes {entity['start_line']}-{entity['end_line']}) → parent: {best_parent}")

    def _assign_calls_to_entities(self, function_calls_map: Dict, entities_map: Dict,
                                  all_entities: List, uses_data: List) -> None:
        """
        Assigne les appels aux bonnes entités avec gestion améliorée.
        """

        # Fonction helper pour trouver le propriétaire d'une ligne
        def find_owner(line_number: int, candidates: List[UnifiedEntity]) -> Optional[UnifiedEntity]:
            owner = None
            smallest_scope = float('inf')
            for entity in candidates:
                if entity.start_line <= line_number <= entity.end_line:
                    scope_size = entity.end_line - entity.start_line
                    if scope_size < smallest_scope:
                        smallest_scope = scope_size
                        owner = entity
            return owner

        # Attribuer les dépendances USE
        for use in uses_data:
            owner = find_owner(use['line'], all_entities)
            if owner:
                owner.dependencies.add(use['name'])
            else:
                # Fallback : attribuer au premier module
                for entity in all_entities:
                    if entity.entity_type == 'module':
                        entity.dependencies.add(use['name'])
                        break

        # Attribuer les appels de fonctions
        for entity_name, calls in function_calls_map.items():
            if entity_name in entities_map:
                entities_map[entity_name].called_functions = calls
            else:
                # Entité orpheline
                orphan_entity = UnifiedEntity.from_parser_result(
                    name=entity_name,
                    entity_type='detected_procedure',
                    start_line=1,
                    end_line=1000,
                    filepath="unknown",
                    source_method='fparser_orphan',
                    confidence=0.7
                )
                orphan_entity.called_functions = calls
                all_entities.append(orphan_entity)

    def _extract_use_module_name(self, node) -> Optional[str]:
        """Extrait le nom du module depuis un Use_Stmt."""
        try:
            if hasattr(node, 'children'):
                for child in node.children:
                    if isinstance(child, Name):
                        return str(child)
        except Exception as e:
            self.logger.debug(f"Erreur extraction nom module USE: {e}")
        return None

    def _get_type_name_from_node(self, node) -> Optional[str]:
        """Extrait le nom d'un type défini depuis un nœud Derived_Type_Def."""
        try:
            from fparser.two.Fortran2003 import Name, Type_Name

            if hasattr(node, 'children'):
                for child in node.children:
                    if hasattr(child, 'children'):
                        for subchild in child.children:
                            if isinstance(subchild, (Name, Type_Name)):
                                return str(subchild)

            if hasattr(node, 'items'):
                for item in node.items:
                    if isinstance(item, (Name, Type_Name)):
                        return str(item)

        except Exception as e:
            self.logger.debug(f"Erreur extraction nom type: {e}")

        return None

    def _extract_calls_from_assignment(self, assignment_node) -> Set[str]:
        """
        Extrait les appels de fonctions depuis une assignation - MÉTHODE COMPLÈTE.
        Capture les patterns comme : result = function(args), var = intrinsic(x), etc.
        """
        calls = set()

        try:
            if hasattr(assignment_node, 'children') and len(assignment_node.children) >= 3:
                rhs = assignment_node.children[2]  # Côté droit de l'assignation

                # Parcourir récursivement le côté droit
                for subnode in walk(rhs):
                    if isinstance(subnode, Function_Reference):
                        func_name = self._extract_function_reference_name(subnode)
                        if func_name:
                            calls.add(func_name)

                    elif isinstance(subnode, Intrinsic_Function_Reference):
                        func_name = self._extract_function_reference_name(subnode)
                        if func_name:
                            calls.add(func_name)

                    elif isinstance(subnode, Primary):
                        # Chercher les intrinsèques dans Primary
                        intrinsic = self._extract_intrinsic_from_primary(subnode)
                        if intrinsic:
                            calls.add(intrinsic)

                    elif isinstance(subnode, Data_Ref):
                        # Parfois les appels sont dans Data_Ref
                        func_name = self._extract_function_from_data_ref(subnode)
                        if func_name:
                            calls.add(func_name)

        except Exception as e:
            self.logger.debug(f"Erreur extraction calls assignment: {e}")

        return calls

    def _extract_intrinsic_from_primary(self, primary_node) -> Optional[str]:
        """Extrait les fonctions intrinsèques depuis un nœud Primary."""
        try:
            primary_str = str(primary_node).lower()

            for intrinsic in self.fortran_intrinsics:
                if f"{intrinsic}(" in primary_str:
                    return intrinsic

        except Exception:
            pass

        return None

    def _extract_function_from_data_ref(self, data_ref_node) -> Optional[str]:
        """Extrait les appels de fonction depuis un Data_Ref."""
        try:
            if hasattr(data_ref_node, 'children'):
                for child in data_ref_node.children:
                    if isinstance(child, Name):
                        return str(child)
        except Exception:
            pass

        return None

    def _get_line_numbers(self, node) -> tuple[int, int]:
        """Extrait les numéros de ligne d'un nœud fparser - VERSION AMÉLIORÉE."""
        try:
            # Méthode 1: via item.span
            if hasattr(node, 'item') and node.item and hasattr(node.item, 'span'):
                return node.item.span

            # Méthode 2: via span direct
            if hasattr(node, 'span'):
                return node.span

            # Méthode 3: chercher dans les enfants
            if hasattr(node, 'children'):
                for child in node.children:
                    if hasattr(child, 'item') and child.item and hasattr(child.item, 'span'):
                        start_line, _ = child.item.span
                        # Pour la fin, chercher le dernier enfant avec span
                        end_line = start_line
                        for subchild in reversed(list(walk(node))):
                            if hasattr(subchild, 'item') and subchild.item and hasattr(subchild.item, 'span'):
                                _, end_line = subchild.item.span
                                break
                        return start_line, end_line

        except Exception as e:
            self.logger.debug(f"Erreur extraction ligne pour {type(node)}: {e}")

        return 1, 1  # Valeur par défaut

    def _is_valid_call_name(self, name: str) -> bool:
        """
        Validation sophistiquée des noms d'appels - VERSION AMÉLIORÉE.
        """
        if not name or len(name) < 2:
            return False

        name_lower = name.lower()

        # Filtrer les mots-clés Fortran
        if name_lower in self.fortran_keywords:
            return False

        # Garder les intrinsèques (ils sont utiles)
        if name_lower in self.fortran_intrinsics:
            return True

        # Filtrer les nombres
        if name.isdigit():
            return False

        # Filtrer les noms suspects
        if len(name) == 1 or not name.replace('_', '').isalnum():
            return False

        # Filtrer les mots-clés courants qui passent à travers
        forbidden = {'if', 'then', 'else', 'end', 'do', 'while', 'select', 'case'}
        if name_lower in forbidden:
            return False

        return True

    def _parse_with_fparser_only(self, code: str, filename: str) -> List[UnifiedEntity]:
        """Parse avec fparser uniquement"""
        calls_map = self._extract_calls_with_fparser_improved(code, filename)
        return self._create_entities_from_fparser_only(calls_map, code)

    def _create_entities_from_fparser_only(self, calls_map: Dict[str, Set[str]], code: str) -> List[UnifiedEntity]:
        """Crée des entités basiques depuis fparser seulement"""
        entities = []
        line_count = len(code.split('\n'))

        for entity_name, calls in calls_map.items():
            # CORRECTION: Utiliser UnifiedEntity.from_parser_result
            entity = UnifiedEntity.from_parser_result(
                name=entity_name,
                entity_type='detected_procedure',
                start_line=1,
                end_line=line_count,
                source_method='fparser',
                confidence=0.7
            )
            entity.called_functions = calls
            entities.append(entity)

        return entities

    def _get_entity_name_from_node(self, node) -> Optional[str]:
        """
        Extrait le nom d'une entité depuis un nœud fparser de manière ROBUSTE.
        CORRECTION : Accès direct aux children du nœud.
        """
        try:
            from fparser.two.Fortran2003 import (
                Module_Stmt, Subroutine_Stmt, Function_Stmt, Program_Stmt, Name
            )

            # Le nom est dans le nœud de déclaration direct (pas dans les descendants)
            if hasattr(node, 'children'):
                for child in node.children:
                    if isinstance(child, (Module_Stmt, Subroutine_Stmt, Function_Stmt, Program_Stmt)):
                        # Le nom est généralement le 2e élément (index 1) dans les items
                        if hasattr(child, 'items') and len(child.items) > 1:
                            potential_name = child.items[1]
                            if isinstance(potential_name, Name):
                                return str(potential_name)

                        # Alternative : parcourir les items pour trouver le premier Name
                        if hasattr(child, 'items'):
                            for item in child.items:
                                if isinstance(item, Name):
                                    return str(item)
                        break

            # Si pas trouvé dans children, essayer directement sur le nœud
            if isinstance(node, (Module_Stmt, Subroutine_Stmt, Function_Stmt, Program_Stmt)):
                if hasattr(node, 'items') and len(node.items) > 1:
                    potential_name = node.items[1]
                    if isinstance(potential_name, Name):
                        return str(potential_name)

        except Exception as e:
            self.logger.warning(f"Exception during name extraction for node {type(node)}: {e}")

        return None

    def _regex_fallback(self, code: str, filename: str) -> List[UnifiedEntity]:
        """Utilise l'ancien parser OFP comme fallback fiable"""
        #return self._simple_regex_fallback(code, filename)
        try:
            from utils.ontologie_fortran_chunker import OFPFortranSemanticChunker
            old_parser = OFPFortranSemanticChunker()
            old_entities = old_parser.extract_fortran_structure(code, filename)

            # Convertir en UnifiedEntity
            unified_entities = []
            for old_entity in old_entities:
                unified = UnifiedEntity.from_parser_result(
                    name=old_entity.name,
                    entity_type=old_entity.entity_type.value,
                    start_line=old_entity.start_line,
                    end_line=old_entity.end_line,
                    source_method='ofp_fallback',
                    confidence=0.9  # L'ancien parser est fiable
                )
                unified.dependencies = old_entity.dependencies.copy()
                unified_entities.append(unified)

            return unified_entities
        except Exception as e:
            # En dernier recours, regex basique
            return self._simple_regex_fallback(code, filename)

    def _simple_regex_fallback(self, code: str, filename: str) -> List[UnifiedEntity]:
        """Fallback regex robuste si tout échoue"""
        self.logger.warning(f"⚠️ Utilisation du fallback regex pour {filename}")

        entity = UnifiedEntity.from_parser_result(
            name="main_fallback",
            entity_type="unknown",
            start_line=1,
            end_line=len(code.split('\n')),
            filepath=filename,  # Ajouter filepath
            source_method='regex_fallback',
            confidence=0.3
        )

        # Extraction regex basique mais robuste
        calls = []

        # Nettoyer le code des commentaires
        lines = code.split('\n')
        clean_lines = []
        for line in lines:
            comment_pos = line.find('!')
            if comment_pos >= 0:
                line = line[:comment_pos]
            clean_lines.append(line)

        clean_code = '\n'.join(clean_lines)

        # Patterns basiques
        call_pattern = re.compile(r'\bcall\s+(\w+)', re.IGNORECASE)
        func_pattern = re.compile(r'=\s*(\w+)\s*\(', re.IGNORECASE)

        # Extraire et filtrer
        potential_calls = set()
        potential_calls.update(call_pattern.findall(clean_code))
        potential_calls.update(func_pattern.findall(clean_code))

        # Filtrer avec nos critères améliorés
        for call in potential_calls:
            if self._is_valid_call_name(call):
                calls.append({'name': call, 'line': 0})

        entity.called_functions = calls
        return [entity]

    def get_parsing_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques détaillées du parsing"""
        return self.stats.get_summary()


# ===== Interface publique unifiée =====

class FortranAnalysisEngine:
    """
    Interface unifiée pour l'analyse Fortran.
    Remplace les méthodes dispersées dans fortran_patterns.py
    """

    def __init__(self, method: str = "hybrid"):
        self.parser = HybridFortranParser(method)
        self._cache = {}  # Cache simple pour éviter le re-parsing

    def extract_function_calls(self, code: str, filename: str = "temp.f90") -> List[str]:
        """
        Extrait les appels de fonctions avec une analyse robuste.
        Interface compatible avec le code existant.
        """
        cache_key = f"calls_{hash(code)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        entities = self.parser.parse_fortran_code(code, filename)

        all_calls = set()
        for entity in entities:
            all_calls.update(entity.called_functions)

        result = list(all_calls)
        self._cache[cache_key] = result
        return result

    def extract_signature(self, code: str, filename: str = "temp.f90") -> str:
        """
        Extrait la signature principale avec analyse robuste.
        Interface compatible avec le code existant.
        """
        cache_key = f"sig_{hash(code)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        entities = self.parser.parse_fortran_code(code, filename)

        # Chercher la première signature valide
        for entity in entities:
            if (entity.entity_type in ['function', 'subroutine'] and
                    entity.signature and
                    entity.signature != "Signature not found"):
                self._cache[cache_key] = entity.signature
                return entity.signature

        result = "Signature not found"
        self._cache[cache_key] = result
        return result

    def get_entities(self, code: str, filename: str = "temp.f90", ontology_manager=None) -> List[UnifiedEntity]:
        """Récupère toutes les entités analysées - VERSION CORRIGÉE"""
        # Utiliser directement la méthode synchrone du parser
        return self.parser.parse_fortran_code(code, filename, ontology_manager)

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'analyse"""
        return self.parser.get_parsing_stats()

    def clear_cache(self):
        """Vide le cache d'analyse"""
        self._cache.clear()

    def analyze_file(self, filepath: str, ontology_manager=None) -> Tuple[List[UnifiedEntity], str, SourceMap]:
        """
        Analyse un fichier Fortran et propage l'entité, le code et la carte des sources.
        """
        self.clear_cache()

        # Utilise la nouvelle méthode du parser et propage le tuple
        entities, full_code, source_map = self.parser.parse_file(filepath, ontology_manager)

        return entities, full_code, source_map


# Fonction factory pour compatibilité
def get_fortran_analyzer(method: str = "hybrid") -> FortranAnalysisEngine:
    """
    Factory pour créer un analyseur Fortran.
    Point d'entrée principal pour le système.
    """
    return FortranAnalysisEngine(method)


