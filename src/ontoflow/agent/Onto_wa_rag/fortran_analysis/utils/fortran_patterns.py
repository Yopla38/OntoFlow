"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# utils/fortran_patterns.py (VERSION RÉÉCRITE)
"""
Patterns Fortran centralisés utilisant le parser hybride robuste.
Remplace les regex basiques par une analyse AST précise.
"""

import re
from typing import List, Dict, Set, Pattern, Optional, Union
from enum import Enum

# Import du parser hybride
try:
    from core.hybrid_fortran_parser import FortranAnalysisEngine, get_fortran_analyzer

    HYBRID_PARSER_AVAILABLE = True
except ImportError:
    HYBRID_PARSER_AVAILABLE = False
    FortranAnalysisEngine = None


class FortranConstructType(Enum):
    """Types de constructions Fortran (enrichi)"""
    MODULE = "module"
    PROGRAM = "program"
    SUBROUTINE = "subroutine"
    FUNCTION = "function"
    INTERNAL_FUNCTION = "internal_function"
    TYPE_DEFINITION = "type_definition"
    INTERFACE = "interface"
    USE_STATEMENT = "use_statement"
    VARIABLE_DECLARATION = "variable_declaration"
    PARAMETER = "parameter"
    COMMON_BLOCK = "common_block"
    PROCEDURE_STMT = "procedure_stmt"


class FortranPatterns:
    """
    Patterns Fortran centralisés avec support du parser hybride.
    Fournit à la fois l'analyse robuste et les fallbacks regex.
    """

    # =================== PATTERNS REGEX DE FALLBACK ===================
    # Utilisés uniquement si le parser hybride n'est pas disponible

    SIMPLE_CALL_PATTERNS: List[Pattern] = [
        re.compile(r'\bcall\s+(\w+)', re.IGNORECASE),
        re.compile(r'=\s*(\w+)\s*\(', re.IGNORECASE),
    ]

    ENTITY_PATTERNS: Dict[FortranConstructType, Pattern] = {
        FortranConstructType.MODULE: re.compile(r'^\s*module\s+(\w+)', re.IGNORECASE),
        FortranConstructType.PROGRAM: re.compile(r'^\s*program\s+(\w+)', re.IGNORECASE),
        FortranConstructType.SUBROUTINE: re.compile(r'^\s*subroutine\s+(\w+)', re.IGNORECASE),
        FortranConstructType.FUNCTION: re.compile(
            r'^\s*(?:pure\s+|elemental\s+|recursive\s+)*'
            r'(?:real|integer|logical|character|complex|type)?\s*(?:\([^)]*\))?\s*'
            r'function\s+(\w+)',
            re.IGNORECASE
        ),
        FortranConstructType.TYPE_DEFINITION: re.compile(r'^\s*type\s*(?:::)?\s*(\w+)', re.IGNORECASE),
    }

    SIGNATURE_PATTERNS: List[Pattern] = [
        re.compile(r'(subroutine\s+\w+\s*\([^)]*\))', re.IGNORECASE),
        re.compile(r'(function\s+\w+\s*\([^)]*\))', re.IGNORECASE),
        re.compile(r'(.*function\s+\w+\s*\([^)]*\))', re.IGNORECASE),
    ]

    # =================== MOTS-CLÉS ET INTRINSÈQUES ===================

    FORTRAN_KEYWORDS: Set[str] = {
        'if', 'then', 'else', 'endif', 'elseif', 'do', 'while', 'enddo', 'select',
        'case', 'where', 'forall', 'real', 'integer', 'logical', 'character', 'complex',
        'allocate', 'deallocate', 'nullify', 'write', 'read', 'print', 'open', 'close',
        'module', 'program', 'subroutine', 'function', 'end', 'contains',
        'use', 'implicit', 'none', 'type', 'class', 'procedure', 'interface',
        'intent', 'in', 'out', 'inout', 'parameter', 'dimension',
        'allocatable', 'pointer', 'target', 'optional', 'save',
        'public', 'private', 'protected', 'volatile', 'asynchronous'
    }

    FORTRAN_INTRINSICS: Set[str] = {
        'sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'log10',
        'abs', 'max', 'min', 'sum', 'product', 'size', 'len',
        'trim', 'adjustl', 'adjustr', 'present', 'associated', 'allocated',
        'huge', 'tiny', 'epsilon', 'precision', 'range', 'digits',
        'modulo', 'mod', 'int', 'nint', 'floor', 'ceiling', 'aint', 'anint',
        'real', 'dble', 'cmplx'
    }

    # =================== DÉTECTION DE STANDARD ===================

    FORTRAN_STANDARD_INDICATORS: Dict[str, List[str]] = {
        'fortran2018': [r'concurrent', r'co_', r'sync\s+all'],
        'fortran2008': [r'submodule', r'block', r'error\s+stop'],
        'fortran2003': [r'type\s*::', r'class\s*\(', r'abstract'],
        'fortran95': [r'forall', r'pure', 'elemental'],
        'fortran90': [r'module', r'interface', r'type\s+'],
        'fortran77': [r'^\s{5}[^\s]', r'common\s*/']
    }


class FortranTextProcessor:
    """
    Processeur de texte Fortran intelligent utilisant le parser hybride.
    Fallback vers regex si le parser n'est pas disponible.
    """

    def __init__(self, use_hybrid: bool = True):
        self.use_hybrid = use_hybrid and HYBRID_PARSER_AVAILABLE

        if self.use_hybrid:
            self.analyzer = get_fortran_analyzer("hybrid")
        else:
            self.analyzer = None

    @staticmethod
    def remove_comments(text: str) -> str:
        """
        Supprime les commentaires Fortran de manière robuste.
        Version améliorée et sécurisée.
        """
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Trouver le premier '!' qui n'est pas dans une chaîne
            in_string = False
            quote_char = None
            comment_pos = -1

            i = 0
            while i < len(line):
                char = line[i]

                if not in_string:
                    if char in ['"', "'"]:
                        in_string = True
                        quote_char = char
                    elif char == '!':
                        comment_pos = i
                        break
                else:
                    if char == quote_char:
                        # Vérifier si ce n'est pas échappé
                        if i == 0 or line[i - 1] != '\\':
                            in_string = False
                            quote_char = None

                i += 1

            if comment_pos >= 0:
                line = line[:comment_pos]

            cleaned_lines.append(line.rstrip())

        return '\n'.join(cleaned_lines)

    def extract_function_calls(self, code: str, filename: str = "temp.f90") -> List[str]:
        """
        Extrait les appels de fonctions avec le parser hybride ou fallback regex.
        Version corrigée qui utilise l'analyse AST robuste.
        """
        if self.use_hybrid and self.analyzer:
            # Utiliser le parser hybride (méthode recommandée)
            try:
                calls = self.analyzer.extract_function_calls(code, filename)
                # Filtrer les intrinsèques si désiré (optionnel)
                filtered_calls = [
                    call for call in calls
                    if call.lower() not in FortranPatterns.FORTRAN_INTRINSICS
                ]
                return filtered_calls
            except Exception as e:
                # Fallback vers regex en cas d'erreur
                return self._extract_calls_regex_fallback(code)
        else:
            # Fallback regex
            return self._extract_calls_regex_fallback(code)

    def _extract_calls_regex_fallback(self, code: str) -> List[str]:
        """Fallback regex pour l'extraction d'appels"""
        calls = set()

        # Nettoyer le code
        cleaned_code = self.remove_comments(code)

        # Appliquer les patterns regex basiques
        for pattern in FortranPatterns.SIMPLE_CALL_PATTERNS:
            matches = pattern.findall(cleaned_code)
            calls.update(matches)

        # Filtrer les résultats
        filtered_calls = []
        for call in calls:
            if (len(call) > 2 and
                    call.lower() not in FortranPatterns.FORTRAN_KEYWORDS and
                    not call.isdigit()):
                filtered_calls.append(call)

        return list(set(filtered_calls))

    def extract_signature(self, code: str, filename: str = "temp.f90") -> str:
        """
        Extrait la signature avec le parser hybride ou fallback regex.
        Version corrigée qui trouve effectivement les signatures.
        """
        if self.use_hybrid and self.analyzer:
            try:
                signature = self.analyzer.extract_signature(code, filename)
                if signature != "Signature not found":
                    return signature
            except Exception:
                pass

        # Fallback regex amélioré
        return self._extract_signature_regex_fallback(code)

    def _extract_signature_regex_fallback(self, code: str) -> str:
        """Fallback regex amélioré pour extraction de signature"""
        lines = code.split('\n')

        for line in lines[:10]:  # Chercher dans les 10 premières lignes
            line = line.strip()

            if not line or line.startswith('!'):
                continue

            for pattern in FortranPatterns.SIGNATURE_PATTERNS:
                match = pattern.search(line)
                if match:
                    return match.group(1).strip()

        return "Signature not found"

    def get_entities(self, code: str, filename: str = "temp.f90"):
        """Récupère toutes les entités (uniquement avec parser hybride)"""
        if self.use_hybrid and self.analyzer:
            return self.analyzer.get_entities(code, filename)
        else:
            return []

    def get_analysis_stats(self) -> Dict[str, any]:
        """Retourne les statistiques d'analyse"""
        if self.use_hybrid and self.analyzer:
            return self.analyzer.get_stats()
        else:
            return {"method": "regex_fallback", "hybrid_available": False}

    @staticmethod
    def detect_fortran_standard(code: str) -> str:
        """Détecte le standard Fortran utilisé"""
        code_lower = code.lower()

        for standard, patterns in FortranPatterns.FORTRAN_STANDARD_INDICATORS.items():
            if any(re.search(pattern, code_lower, re.MULTILINE) for pattern in patterns):
                return standard

        return 'fortran90'

    @staticmethod
    def extract_public_interface(module_code: str) -> List[str]:
        """Extrait l'interface publique d'un module"""
        interfaces = []

        # Chercher les déclarations public explicites
        public_pattern = re.compile(r'public\s*::\s*([^!\n]+)', re.IGNORECASE)
        matches = public_pattern.findall(module_code)

        for match in matches:
            names = [name.strip() for name in match.split(',')]
            interfaces.extend(names)

        # Si pas de public explicite, chercher les fonctions/subroutines
        if not interfaces:
            func_patterns = [
                re.compile(r'subroutine\s+(\w+)', re.IGNORECASE),
                re.compile(r'function\s+(\w+)', re.IGNORECASE)
            ]

            for pattern in func_patterns:
                matches = pattern.findall(module_code)
                interfaces.extend(matches)

        return interfaces[:10]


# ===== Instance globale et fonctions utilitaires =====

# Instance globale pour utilisation dans le système
_default_processor = None


def get_fortran_processor(use_hybrid: bool = True) -> FortranTextProcessor:
    """Factory pour obtenir un processeur Fortran"""
    global _default_processor
    if _default_processor is None:
        _default_processor = FortranTextProcessor(use_hybrid)
    return _default_processor


def extract_function_calls(code: str, filename: str = "temp.f90",
                           use_hybrid: bool = True) -> List[str]:
    """Fonction utilitaire pour extraire les appels (interface compatible)"""
    processor = get_fortran_processor(use_hybrid)
    return processor.extract_function_calls(code, filename)


def extract_signature(code: str, filename: str = "temp.f90",
                      use_hybrid: bool = True) -> str:
    """Fonction utilitaire pour extraire la signature (interface compatible)"""
    processor = get_fortran_processor(use_hybrid)
    return processor.extract_signature(code, filename)


def clean_fortran_code(code: str) -> str:
    """Fonction utilitaire pour nettoyer le code"""
    return FortranTextProcessor.remove_comments(code)


def is_fortran_keyword(word: str) -> bool:
    """Vérifie si un mot est un mot-clé Fortran"""
    return word.lower() in FortranPatterns.FORTRAN_KEYWORDS


def is_fortran_intrinsic(word: str) -> bool:
    """Vérifie si un mot est une fonction intrinsèque Fortran"""
    return word.lower() in FortranPatterns.FORTRAN_INTRINSICS