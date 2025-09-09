"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# core/fortran_parser.py
"""
Interface unifiée pour le parsing Fortran.
Utilise le parser hybride et fournit une API simple et cohérente.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from .hybrid_fortran_parser import FortranAnalysisEngine, get_fortran_analyzer, UnifiedEntity
from ..utils.caching import global_cache

logger = logging.getLogger(__name__)


@dataclass
class ParsedFortranFile:
    """Résultat du parsing d'un fichier Fortran"""
    filepath: str
    entities: List[UnifiedEntity]
    total_entities: int
    total_calls: int
    fortran_standard: str
    parse_method: str
    confidence: float

    def get_entities_by_type(self, entity_type: str) -> List[UnifiedEntity]:
        """Filtre les entités par type"""
        return [e for e in self.entities if e.entity_type == entity_type]

    def get_main_entities(self) -> List[UnifiedEntity]:
        """Récupère les entités principales (modules, programmes)"""
        main_types = {'module', 'program', 'subroutine', 'function'}
        return [e for e in self.entities if e.entity_type in main_types]

    def get_all_calls(self) -> Set[str]:
        """Récupère tous les appels de fonctions"""
        all_calls = set()
        for entity in self.entities:
            all_calls.update(entity.called_functions)
        return all_calls


class UnifiedFortranParser:
    """
    Parser Fortran unifié qui orchestre tous les outils d'analyse.
    Interface simple pour toute analyse de code Fortran.
    """

    def __init__(self,
                 prefer_method: str = "hybrid",
                 enable_caching: bool = True,
                 cache_ttl: int = 1800):

        self.prefer_method = prefer_method
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl

        # Analyseur principal
        self.analyzer = get_fortran_analyzer(prefer_method)

        # Statistiques de parsing
        self.total_parsed = 0
        self.cache_hits = 0
        self.parse_errors = 0

    async def parse_file(self, filepath: str, code: Optional[str] = None) -> ParsedFortranFile:
        """
        Parse un fichier Fortran complet.

        Args:
            filepath: Chemin du fichier
            code: Code Fortran (si None, lit le fichier)

        Returns:
            Résultat structuré du parsing
        """
        # Lire le code si nécessaire
        if code is None:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    code = f.read()
            except Exception as e:
                logger.error(f"Erreur lecture fichier {filepath}: {e}")
                self.parse_errors += 1
                return self._create_empty_result(filepath, f"Erreur lecture: {e}")

        # Vérifier le cache
        cache_key = f"parse_{filepath}_{hash(code)}"
        if self.enable_caching:
            cached_result = await global_cache.entities.get(cache_key)
            if cached_result:
                self.cache_hits += 1
                return self._deserialize_parse_result(cached_result)

        # Parser le code
        try:
            entities = self.analyzer.get_entities(code, filepath)
            stats = self.analyzer.get_stats()

            # Créer le résultat structuré
            result = ParsedFortranFile(
                filepath=filepath,
                entities=entities,
                total_entities=len(entities),
                total_calls=sum(len(e.called_functions) for e in entities),
                fortran_standard=self._detect_fortran_standard(code),
                parse_method=stats.get('method', self.prefer_method),
                confidence=self._calculate_confidence(entities, stats)
            )

            # Mettre en cache
            if self.enable_caching:
                serialized = self._serialize_parse_result(result)
                await global_cache.entities.set(cache_key, serialized, ttl=self.cache_ttl)

            self.total_parsed += 1
            logger.debug(f"✅ Parsed {filepath}: {len(entities)} entities")

            return result

        except Exception as e:
            logger.error(f"Erreur parsing {filepath}: {e}")
            self.parse_errors += 1
            return self._create_empty_result(filepath, f"Erreur parsing: {e}")

    async def parse_code_snippet(self, code: str, context: str = "snippet") -> ParsedFortranFile:
        """Parse un snippet de code Fortran"""
        return await self.parse_file(f"<{context}>", code)

    async def extract_entities_only(self, code: str, filepath: str = "temp.f90") -> List[UnifiedEntity]:
        """CORRIGÉ - Version async"""
        result = await self.parse_file(filepath, code)
        return result.entities

    async def extract_function_calls_only(self, code: str, filepath: str = "temp.f90") -> List[str]:
        """Extrait seulement les appels de fonctions (interface simple)"""
        cache_key = f"calls_{hash(code)}"

        if self.enable_caching:
            cached_calls = await global_cache.function_calls.get(cache_key)
            if cached_calls:
                return cached_calls

        calls = self.analyzer.extract_function_calls(code, filepath)

        if self.enable_caching:
            await global_cache.function_calls.set(cache_key, calls, ttl=self.cache_ttl)

        return calls

    async def extract_signature_only(self, code: str, filepath: str = "temp.f90") -> str:
        """Extrait seulement la signature (interface simple)"""
        return self.analyzer.extract_signature(code, filepath)

    async def analyze_dependencies(self, code: str, filepath: str = "temp.f90") -> Dict[str, Any]:
        """Analyse les dépendances d'un code Fortran"""
        result = await self.parse_file(filepath, code)

        dependencies = {
            'use_statements': set(),
            'function_calls': set(),
            'entity_dependencies': {}
        }

        for entity in result.entities:
            dependencies['use_statements'].update(entity.dependencies)
            dependencies['function_calls'].update(entity.called_functions)

            if entity.dependencies or entity.called_functions:
                dependencies['entity_dependencies'][entity.name] = {
                    'uses': list(entity.dependencies),
                    'calls': list(entity.called_functions)
                }

        return {
            'filepath': filepath,
            'use_statements': list(dependencies['use_statements']),
            'function_calls': list(dependencies['function_calls']),
            'entity_dependencies': dependencies['entity_dependencies'],
            'total_dependencies': len(dependencies['use_statements']) + len(dependencies['function_calls'])
        }

    def _detect_fortran_standard(self, code: str) -> str:
        """Détecte le standard Fortran (réutilise la logique existante)"""
        indicators = {
            'fortran2018': [r'concurrent', r'co_', r'sync\s+all'],
            'fortran2008': [r'submodule', r'block', r'error\s+stop'],
            'fortran2003': [r'type\s*::', r'class\s*\(', r'abstract'],
            'fortran95': [r'forall', r'pure', r'elemental'],
            'fortran90': [r'module', r'interface', r'type\s+'],
            'fortran77': [r'^\s{5}[^\s]', r'common\s*/']
        }

        import re
        code_lower = code.lower()
        for standard, patterns in indicators.items():
            if any(re.search(pattern, code_lower, re.MULTILINE) for pattern in patterns):
                return standard

        return 'fortran90'

    def _calculate_confidence(self, entities: List[UnifiedEntity], stats: Dict[str, Any]) -> float:
        """Calcule la confiance du parsing"""
        if not entities:
            return 0.0

        # Confiance basée sur la méthode utilisée
        method_confidence = {
            'hybrid': 0.9,
            'f2py': 0.8,
            'fparser': 0.7,
            'regex_fallback': 0.3
        }

        base_confidence = method_confidence.get(stats.get('method', 'hybrid'), 0.5)

        # Ajuster selon la qualité des entités
        avg_entity_confidence = sum(e.confidence for e in entities) / len(entities)

        return min(1.0, (base_confidence + avg_entity_confidence) / 2)

    def _create_empty_result(self, filepath: str, error: str) -> ParsedFortranFile:
        """Crée un résultat vide en cas d'erreur"""
        return ParsedFortranFile(
            filepath=filepath,
            entities=[],
            total_entities=0,
            total_calls=0,
            fortran_standard='unknown',
            parse_method='error',
            confidence=0.0
        )

    def _serialize_parse_result(self, result: ParsedFortranFile) -> Dict[str, Any]:
        """Sérialise un résultat de parsing pour le cache"""
        return {
            'filepath': result.filepath,
            'entities': [entity.to_dict() for entity in result.entities],
            'total_entities': result.total_entities,
            'total_calls': result.total_calls,
            'fortran_standard': result.fortran_standard,
            'parse_method': result.parse_method,
            'confidence': result.confidence
        }

    def _deserialize_parse_result(self, data: Dict[str, Any]) -> ParsedFortranFile:
        """Désérialise un résultat de parsing depuis le cache"""
        entities = []
        for entity_data in data['entities']:
            # CORRECTION: Utiliser UnifiedEntity.from_parser_result
            entity = UnifiedEntity.from_parser_result(
                name=entity_data['name'],
                entity_type=entity_data['entity_type'],
                start_line=entity_data['start_line'],
                end_line=entity_data['end_line'],
                filepath=entity_data.get('filepath', ''),
                parent_entity=entity_data.get('parent'),
                signature=entity_data.get('signature', ''),
                source_method=entity_data.get('source_method', ''),
                confidence=entity_data.get('confidence', 1.0)
            )
            # Restaurer les relations
            entity.dependencies = set(entity_data.get('dependencies', []))
            entity.called_functions = set(entity_data.get('called_functions', []))
            entities.append(entity)

        return ParsedFortranFile(
            filepath=data['filepath'],
            entities=entities,
            total_entities=data['total_entities'],
            total_calls=data['total_calls'],
            fortran_standard=data['fortran_standard'],
            parse_method=data['parse_method'],
            confidence=data['confidence']
        )

    def get_parsing_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de parsing"""
        hit_rate = self.cache_hits / max(1, self.total_parsed + self.cache_hits) * 100

        return {
            'total_parsed': self.total_parsed,
            'cache_hits': self.cache_hits,
            'parse_errors': self.parse_errors,
            'cache_hit_rate': round(hit_rate, 2),
            'analyzer_stats': self.analyzer.get_stats()
        }

    def clear_cache(self):
        """Vide le cache du parser"""
        self.analyzer.clear_cache()


# Instance globale
_global_parser = None


def get_unified_parser(method: str = "hybrid") -> UnifiedFortranParser:
    """Factory pour obtenir le parser unifié global"""
    global _global_parser
    if _global_parser is None:
        _global_parser = UnifiedFortranParser(method)
    return _global_parser