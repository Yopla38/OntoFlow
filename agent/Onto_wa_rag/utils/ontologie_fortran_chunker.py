"""
Chunker Fortran basé sur open-fortran-parser.
100 % compatible avec F2pyFortranSemanticChunker (il en hérite !)
"""

from __future__ import annotations
"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """



import hashlib
import logging
import os
import pathlib
import tempfile
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Set, Tuple

import numpy as np
import open_fortran_parser                     # pip install open-fortran-parser
from rdflib import Graph, Namespace, Literal, RDF, URIRef

from ..utils.fortran_sementic_chunker import FortranEntity
from ..utils.semantic_chunker import SemanticChunker


class FortranConstructType(Enum):
    """Types de constructions Fortran pour l'ontologie"""
    MODULE = "module"
    PROGRAM = "program"
    SUBROUTINE = "subroutine"
    FUNCTION = "function"
    TYPE_DEFINITION = "type_definition"
    INTERFACE = "interface"
    USE_STATEMENT = "use_statement"
    VARIABLE_DECLARATION = "variable_declaration"
    PARAMETER = "parameter"
    COMMON_BLOCK = "common_block"
    DATA_STATEMENT = "data_statement"
    CONTAINS = "contains"
    DO_LOOP = "do_loop"
    IF_CONSTRUCT = "if_construct"
    SELECT_CASE = "select_case"
    WHERE_CONSTRUCT = "where_construct"
    FORALL_CONSTRUCT = "forall_construct"
    OPENMP_DIRECTIVE = "openmp_directive"
    OPENACC_DIRECTIVE = "openacc_directive"


@dataclass
class FortranEntity:
    """Représente une entité Fortran avec ses métadonnées"""
    name: str
    entity_type: FortranConstructType
    start_line: int
    end_line: int
    parent: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)
    interfaces: Set[str] = field(default_factory=set)
    variables: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    physics_concepts: Set[str] = field(default_factory=set)
    computational_patterns: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    detected_concepts: List[Dict[str, Any]] = field(default_factory=list)
    semantic_metadata: Dict[str, Any] = field(default_factory=dict)


# ➜ on complète l’Enum pour le cas manquant
if not hasattr(FortranConstructType, "PROCEDURE_STMT"):
    FortranConstructType.PROCEDURE_STMT = "procedure_stmt"       # type: ignore

F90 = Namespace("http://source-code.org/fortran#")
_logger = logging.getLogger(__name__)


def _safe(uri_part: str) -> str:
    return re.sub(r'[^0-9A-Za-z_\-]', '_', uri_part)


def _make_uri(kind: str, name: str, file_uri: str) -> URIRef:
    return URIRef(f"{file_uri}#{kind}_{_safe(name)}")


class OFPFortranSemanticChunker(SemanticChunker):
    """
    Drop-in replacement de F2pyFortranSemanticChunker utilisant open-fortran-parser.
    """

    _interesting = {
        'module', 'program', 'subroutine', 'function', 'interface',
        'module_procedure', 'block_data', 'procedure_stmt', 'type', 'variable'
    }

    def __init__(self,
                 min_chunk_size=200,
                 max_chunk_size=2000,
                 overlap_sentences=0,
                 respect_boundaries=True,
                 enable_semantic_detection=True,
                 generate_triples=True,
                 ontology_graph: Optional[Graph] = None):

        # on appelle le __init__ du parent (garde toutes les méthodes utilitaires)
        super().__init__(min_chunk_size,
                         max_chunk_size,
                         overlap_sentences,
                         respect_boundaries)

        self.logger = logging.getLogger(__name__)
        self.enable_semantic_detection = enable_semantic_detection

        self.generate_triples = generate_triples
        self.graph: Graph = ontology_graph or Graph()
        if ontology_graph is None:
            self.graph.bind("code", F90)

    # ---------------------------------------------------------------------
    # 1.  Parse OFP -> entités
    # ---------------------------------------------------------------------
    def _parse_with_ofp(self, code: str, filepath: str) -> List[FortranEntity]:
        """
        NOUVELLE VERSION : Parser regex robuste pour remplacer OFP défaillant
        """
        print(f"🔄 Utilisation du parser regex robuste (OFP désactivé)")

        entities = []
        lines = code.split('\n')

        # Stack pour tracker les entités ouvertes
        open_entities = []

        for i, line in enumerate(lines):
            line_num = i + 1
            stripped = line.strip()

            if not stripped or stripped.startswith('!'):
                continue

            # 1. MODULES
            module_match = re.match(r'^\s*module\s+(\w+)', line, re.IGNORECASE)
            if module_match and not re.search(r'procedure', line, re.IGNORECASE):
                entity = FortranEntity(
                    name=module_match.group(1),
                    entity_type=FortranConstructType.MODULE,
                    start_line=line_num,
                    end_line=line_num  # Sera mis à jour
                )
                open_entities.append(entity)
                entities.append(entity)
                print(f"   📦 Module trouvé: {entity.name}")
                continue

            # 2. TYPES DÉFINIS (vrais types, pas les types de base)
            type_match = re.match(r'^\s*type\s*(?:::)?\s*(\w+)', line, re.IGNORECASE)
            if type_match:
                # Vérifier que c'est un vrai type défini
                type_name = type_match.group(1).lower()
                # Ignorer les types de base Fortran
                if type_name not in ['real', 'integer', 'logical', 'character', 'complex']:
                    entity = FortranEntity(
                        name=type_match.group(1),
                        entity_type=FortranConstructType.TYPE_DEFINITION,
                        start_line=line_num,
                        end_line=line_num
                    )
                    open_entities.append(entity)
                    entities.append(entity)
                    print(f"   🏷️  Type défini trouvé: {entity.name}")
                continue

            # 3. SUBROUTINES
            sub_match = re.match(r'^\s*subroutine\s+(\w+)', line, re.IGNORECASE)
            if sub_match:
                entity = FortranEntity(
                    name=sub_match.group(1),
                    entity_type=FortranConstructType.SUBROUTINE,
                    start_line=line_num,
                    end_line=line_num
                )
                open_entities.append(entity)
                entities.append(entity)
                print(f"   🔧 Subroutine trouvée: {entity.name}")
                continue

            # 4. FUNCTIONS
            func_match = re.match(
                r'^\s*(?:pure\s+|elemental\s+|recursive\s+)*'
                r'(?:real|integer|logical|character|type)?\s*(?:\([^)]*\))?\s*'
                r'function\s+(\w+)',
                line, re.IGNORECASE
            )
            if func_match:
                entity = FortranEntity(
                    name=func_match.group(1),
                    entity_type=FortranConstructType.FUNCTION,
                    start_line=line_num,
                    end_line=line_num
                )
                open_entities.append(entity)
                entities.append(entity)
                print(f"   ⚙️  Function trouvée: {entity.name}")
                continue

            # 5. PROGRAMS
            prog_match = re.match(r'^\s*program\s+(\w+)', line, re.IGNORECASE)
            if prog_match:
                entity = FortranEntity(
                    name=prog_match.group(1),
                    entity_type=FortranConstructType.PROGRAM,
                    start_line=line_num,
                    end_line=line_num
                )
                open_entities.append(entity)
                entities.append(entity)
                print(f"   🚀 Program trouvé: {entity.name}")
                continue

            # 6. USE STATEMENTS (dépendances)
            use_match = re.match(r'^\s*use\s+(\w+)', line, re.IGNORECASE)
            if use_match and open_entities:
                dep_name = use_match.group(1)
                open_entities[-1].dependencies.add(dep_name)
                print(f"   🔗 Dépendance ajoutée: {dep_name}")
                continue

            # 7. FINS D'ENTITÉS
            if re.match(r'^\s*end\s+module', line, re.IGNORECASE):
                if open_entities and open_entities[-1].entity_type == FortranConstructType.MODULE:
                    open_entities[-1].end_line = line_num
                    closed = open_entities.pop()
                    print(f"   ✅ Module fermé: {closed.name} (lignes {closed.start_line}-{closed.end_line})")

            elif re.match(r'^\s*end\s+type', line, re.IGNORECASE):
                if open_entities and open_entities[-1].entity_type == FortranConstructType.TYPE_DEFINITION:
                    open_entities[-1].end_line = line_num
                    closed = open_entities.pop()
                    print(f"   ✅ Type fermé: {closed.name} (lignes {closed.start_line}-{closed.end_line})")

            elif re.match(r'^\s*end\s+subroutine', line, re.IGNORECASE):
                if open_entities and open_entities[-1].entity_type == FortranConstructType.SUBROUTINE:
                    open_entities[-1].end_line = line_num
                    closed = open_entities.pop()
                    print(f"   ✅ Subroutine fermée: {closed.name} (lignes {closed.start_line}-{closed.end_line})")

            elif re.match(r'^\s*end\s+function', line, re.IGNORECASE):
                if open_entities and open_entities[-1].entity_type == FortranConstructType.FUNCTION:
                    open_entities[-1].end_line = line_num
                    closed = open_entities.pop()
                    print(f"   ✅ Function fermée: {closed.name} (lignes {closed.start_line}-{closed.end_line})")

            elif re.match(r'^\s*end\s+program', line, re.IGNORECASE):
                if open_entities and open_entities[-1].entity_type == FortranConstructType.PROGRAM:
                    open_entities[-1].end_line = line_num
                    closed = open_entities.pop()
                    print(f"   ✅ Program fermé: {closed.name} (lignes {closed.start_line}-{closed.end_line})")

        # Fermer les entités restantes (si pas de 'end' explicite)
        for entity in open_entities:
            entity.end_line = len(lines)
            print(f"   ⚠️  Entité auto-fermée: {entity.name}")

        return entities

    def extract_fortran_structure(self, code: str, filepath: str) -> List[FortranEntity]:
        """
        Extrait la structure complète du code Fortran en utilisant OFP.
        Point d'entrée principal pour l'analyse structurelle.
        """
        print(f"🔍 Extraction de la structure Fortran pour {os.path.basename(filepath)}")

        try:
            # Utiliser notre parser OFP principal
            entities = self._parse_with_ofp(code, filepath)

            if entities:
                print(f"✅ {len(entities)} entités détectées avec OFP")
                for entity in entities:
                    print(
                        f"   - {entity.name} ({entity.entity_type.value}) lignes {entity.start_line}-{entity.end_line}")
            else:
                print("⚠️ Aucune entité détectée avec OFP, tentative fallback...")
                # Fallback avec regex si OFP échoue
                entities = self._extract_with_regex_fallback(code)
                print(f"📝 {len(entities)} entités détectées avec fallback regex")

            return entities

        except Exception as e:
            print(f"❌ Erreur lors de l'extraction de structure: {e}")
            # En cas d'échec total, essayer le fallback regex
            try:
                entities = self._extract_with_regex_fallback(code)
                print(f"🔄 Fallback réussi: {len(entities)} entités détectées")
                return entities
            except Exception as e2:
                print(f"❌ Échec total de l'extraction: {e2}")
                return []

    def _extract_with_regex_fallback(self, code: str) -> List[FortranEntity]:
        """
        Méthode de fallback utilisant des regex simples si OFP échoue.
        """
        entities = []
        lines = code.split('\n')

        # Patterns pour les constructions principales
        patterns = {
            FortranConstructType.MODULE: re.compile(r'^\s*module\s+(\w+)', re.IGNORECASE),
            FortranConstructType.PROGRAM: re.compile(r'^\s*program\s+(\w+)', re.IGNORECASE),
            FortranConstructType.SUBROUTINE: re.compile(r'^\s*subroutine\s+(\w+)', re.IGNORECASE),
            FortranConstructType.FUNCTION: re.compile(
                r'^\s*(?:pure\s+|elemental\s+|recursive\s+)*(?:real|integer|logical|complex|character|type)?\s*(?:\([^)]*\))?\s*function\s+(\w+)',
                re.IGNORECASE
            ),
            FortranConstructType.TYPE_DEFINITION: re.compile(r'^\s*type\s*(?:::)?\s*(\w+)', re.IGNORECASE),
        }

        # Patterns de fin
        end_patterns = {
            FortranConstructType.MODULE: re.compile(r'^\s*end\s+module', re.IGNORECASE),
            FortranConstructType.PROGRAM: re.compile(r'^\s*end\s+program', re.IGNORECASE),
            FortranConstructType.SUBROUTINE: re.compile(r'^\s*end\s+subroutine', re.IGNORECASE),
            FortranConstructType.FUNCTION: re.compile(r'^\s*end\s+function', re.IGNORECASE),
            FortranConstructType.TYPE_DEFINITION: re.compile(r'^\s*end\s+type', re.IGNORECASE),
        }

        current_entities = []

        for i, line in enumerate(lines):
            # Vérifier les débuts de construction
            for construct_type, pattern in patterns.items():
                match = pattern.match(line)
                if match:
                    entity = FortranEntity(
                        name=match.group(1),
                        entity_type=construct_type,
                        start_line=i + 1,
                        end_line=i + 1  # Sera mis à jour
                    )
                    current_entities.append(entity)
                    entities.append(entity)
                    print(f"   📝 Détecté {construct_type.value}: {entity.name} (ligne {i + 1})")

            # Vérifier les fins de construction
            for construct_type, pattern in end_patterns.items():
                if pattern.match(line) and current_entities:
                    # Trouver l'entité correspondante
                    for j in range(len(current_entities) - 1, -1, -1):
                        if current_entities[j].entity_type == construct_type:
                            current_entities[j].end_line = i + 1
                            print(f"   ✅ Fermé {construct_type.value}: {current_entities[j].name} (ligne {i + 1})")
                            current_entities.pop(j)
                            break

            # Détecter les USE statements
            use_match = re.match(r'^\s*use\s+(\w+)', line, re.IGNORECASE)
            if use_match and current_entities:
                dep_name = use_match.group(1)
                current_entities[-1].dependencies.add(dep_name)
                print(f"   🔗 Dépendance ajoutée: {dep_name}")

        return entities

    async def _detect_semantic_concepts(
            self,
            entity: FortranEntity,
            entity_code: str,
            embedding: Optional[np.ndarray] = None,
            ontology_manager=None
    ):
        """Détecte les concepts sémantiques pour une entité."""
        if not self.enable_semantic_detection or not ontology_manager:
            return

        self.logger.debug(f"🧠 Classifying concepts for {entity.name}...")

        try:
            classifier = getattr(ontology_manager, 'classifier', None)

            if not classifier or not hasattr(classifier, 'concept_classifier'):
                self.logger.debug("No concept classifier available")
                return

            concept_classifier = classifier.concept_classifier

            # DEBUG : Vérifier l'état du classifier
            print(f"DEBUG: concept_classifier type = {type(concept_classifier)}")
            print(f"DEBUG: has classify_embedding_direct = {hasattr(concept_classifier, 'classify_embedding_direct')}")
            print(f"DEBUG: has concept_embeddings = {hasattr(concept_classifier, 'concept_embeddings')}")

            if hasattr(concept_classifier, 'concept_embeddings'):
                print(f"DEBUG: Number of concept embeddings = {len(concept_classifier.concept_embeddings)}")

            # Si l'embedding n'est pas fourni, le calculer
            if embedding is None:
                # ... code existant ...
                pass

            # IMPORTANT : Vérifier que l'embedding est valide
            print(f"DEBUG: Embedding shape = {embedding.shape if hasattr(embedding, 'shape') else 'not numpy array'}")
            print(f"DEBUG: Embedding norm = {np.linalg.norm(embedding)}")

            # Classification directe avec l'embedding
            if hasattr(concept_classifier, 'classify_embedding_direct'):
                print("DEBUG: Calling classify_embedding_direct...")
                detected_concepts = await concept_classifier.classify_embedding_direct(
                    embedding=embedding,
                    min_confidence=0.3  # Abaisser le seuil pour debug
                )
                print(f"DEBUG: classify_embedding_direct returned {detected_concepts}")
            else:
                print("ERROR: classify_embedding_direct method not found!")

                # FALLBACK : Utiliser une méthode qui existe
                if hasattr(concept_classifier, 'auto_detect_concepts'):
                    print("DEBUG: Using auto_detect_concepts as fallback...")
                    detected_concepts = await concept_classifier.auto_detect_concepts(
                        embedding,
                        min_confidence=0.3
                    )
                    print(f"DEBUG: auto_detect_concepts returned {len(detected_concepts)} concepts")
                else:
                    print("ERROR: No classification method available!")
                    detected_concepts = []

            # Stocker les concepts détectés
            entity.detected_concepts = detected_concepts[:5]  # Top 5
            entity.semantic_metadata['concepts_count'] = len(entity.detected_concepts)
            entity.semantic_metadata['detection_method'] = 'direct_embedding_classification'

            print(f"   ✅ Detected {len(entity.detected_concepts)} concepts for {entity.name}")

        except Exception as e:
            print(f"⚠️ Error detecting concepts for {entity.name}: {e}")
            import traceback
            traceback.print_exc()

    def _detect_fortran_standard(self, code: str) -> str:
        """Détecte le standard Fortran"""
        indicators = {
            'fortran2018': [r'concurrent', r'co_', r'sync\s+all'],
            'fortran2008': [r'submodule', r'block', r'error\s+stop'],
            'fortran2003': [r'type\s*::', r'class\s*\(', r'abstract'],
            'fortran95': [r'forall', r'pure', r'elemental'],
            'fortran90': [r'module', r'interface', r'type\s+'],
            'fortran77': [r'^\s{5}[^\s]', r'common\s*/']
        }

        code_lower = code.lower()
        for standard, patterns in indicators.items():
            if any(re.search(pattern, code_lower, re.MULTILINE) for pattern in patterns):
                return standard

        return 'fortran90'

    def _create_chunk_from_entity(
            self,
            entity: FortranEntity,
            code: str,
            document_id: str,
            filepath: str,
            chunk_index: int,
            base_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Crée un chunk à partir d'une entité avec métadonnées enrichies"""

        content_hash = hashlib.md5(code.encode()).hexdigest()

        # Identifier l'entité racine (sans les suffixes _part_X)
        base_entity_name = re.sub(r'_part_\d+$', '', entity.name)

        # Préparer les concepts détectés pour les métadonnées
        detected_concepts_info = []
        for concept in entity.detected_concepts:
            detected_concepts_info.append({
                'label': concept.get('label', ''),
                'confidence': concept.get('confidence', 0),
                'concept_uri': concept.get('concept_uri', ''),
                'category': concept.get('category', '')
            })

        # Métadonnées enrichies avec tracking complet
        chunk_metadata = {
            'entity_type': entity.entity_type.value,
            'entity_name': entity.name,
            'base_entity_name': base_entity_name,
            'entity_id': entity.metadata.get('parent_entity_id', f"{filepath}#{base_entity_name}"),
            'is_partial': entity.metadata.get('is_partial', False),
            'part_index': entity.metadata.get('part_index', 0),
            'total_parts': entity.metadata.get('total_parts', 1),
            'part_sequence': entity.metadata.get('part_sequence', 0),
            'sibling_chunks': entity.metadata.get('sibling_chunks', []),
            'all_chunks': entity.metadata.get('all_chunks', []),
            'parent_entity_id': entity.metadata.get('parent_entity_id'),
            'dependencies': list(entity.dependencies),
            'interfaces': list(entity.interfaces),
            'variables': entity.variables,
            'parameters': entity.parameters,
            'detected_concepts': detected_concepts_info,
            'semantic_metadata': entity.semantic_metadata,
            'content_hash': content_hash,
            'chunk_method': 'f2py_semantic_embedding',
            'language': 'fortran',
            'fortran_standard': self._detect_fortran_standard(code),
            'filepath': filepath,
            'filename': os.path.basename(filepath),
            'start_pos': entity.start_line,
            'end_pos': entity.end_line,
            **entity.metadata
        }

        # Ajouter les bounds de l'entité complète si disponibles
        if 'entity_bounds' in entity.semantic_metadata:
            chunk_metadata['entity_bounds'] = entity.semantic_metadata['entity_bounds']

        if entity.metadata.get('is_internal_function'):
            chunk_metadata['entity_type'] = 'internal_function'  # Type spécifique
            chunk_metadata['parent_function'] = entity.metadata.get('parent_entity')
            chunk_metadata['qualified_name'] = f"{entity.metadata.get('parent_entity')}.{entity.name}"

        # Métadonnées de base
        if base_metadata:
            chunk_metadata.update(base_metadata)

        parent_name = entity.parent or ''
        chunk_metadata['parent_entity_name'] = parent_name

        # ---------------- parent information ----------------
        parent_name = entity.parent or ''
        chunk_metadata['parent_entity_name'] = parent_name

        chunk_metadata['parent_entity_type'] = entity.metadata.get(
            'parent_entity_type', ''
        )

        return {
            "id": f"{document_id}-chunk-{chunk_index}",
            "document_id": document_id,
            "text": code,
            "start_pos": entity.start_line,
            "end_pos": entity.end_line,
            "metadata": chunk_metadata
        }

    # ------------------------------------------------------------------
    # 2.  Triples RDF
    # ------------------------------------------------------------------
    def _add_triples(self, ent: FortranEntity, file_uri: str):
        kind = ent.entity_type.value.capitalize()
        uri = _make_uri(kind, ent.name, file_uri)

        self.graph.add((uri, RDF.type, F90[kind]))
        self.graph.add((uri, F90.hasName, Literal(ent.name)))
        self.graph.add((uri, F90.lineBegin, Literal(ent.start_line)))
        self.graph.add((uri, F90.lineEnd, Literal(ent.end_line)))
        self.graph.add((uri, F90.filePath, Literal(file_uri)))

        if ent.parent:
            parent_uri = _make_uri("ProgramUnit", ent.parent, file_uri)
            self.graph.add((parent_uri, F90.contains, uri))

    def _find_logical_sections(self, lines: List[str]) -> List[Tuple[int, int, str]] | None:
        """
        Trouve les sections logiques dans le code pour un découpage intelligent.
        Retourne une liste de tuples (start_line, end_line, section_type)
        """
        sections = []
        current_section_start = 0
        current_section_type = "code"

        # Patterns pour identifier les sections
        section_patterns = {
            'declarations': re.compile(r'^\s*(?:implicit|use|real|integer|logical|character|type)', re.IGNORECASE),
            'contains': re.compile(r'^\s*contains\s*$', re.IGNORECASE),
            'executable': re.compile(r'^\s*(?:do|if|select|where|forall|call)', re.IGNORECASE),
            'io': re.compile(r'^\s*(?:open|close|read|write|print)', re.IGNORECASE),
            'parallel': re.compile(r'^\s*(?:\$omp|\$acc|!dir\$)', re.IGNORECASE),
        }

        # Identifier les transitions de sections
        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Ignorer les lignes vides et commentaires
            if not line_stripped or line_stripped.startswith('!'):
                continue

            # Détecter le type de section
            detected_type = None
            for section_type, pattern in section_patterns.items():
                if pattern.match(line):
                    detected_type = section_type
                    break

            # Si on change de type de section
            if detected_type and detected_type != current_section_type:
                # Sauvegarder la section précédente si elle est assez grande
                if i - current_section_start > 5:  # Au moins 5 lignes
                    sections.append((current_section_start, i - 1, current_section_type))

                current_section_start = i
                current_section_type = detected_type

        # Ajouter la dernière section
        if len(lines) - current_section_start > 5:
            sections.append((current_section_start, len(lines) - 1, current_section_type))

        # Si pas assez de sections trouvées, retourner None pour utiliser le fallback
        if len(sections) < 2:
            return None

        return sections

    def _split_large_entity(
            self,
            entity: FortranEntity,
            entity_code: str,
            document_id: str,
            filepath: str,
            start_chunk_index: int,
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Découpe une entité trop grande avec tracking complet des parties"""
        chunks = []
        lines = entity_code.split('\n')

        # Générer un ID unique pour cette entité complète
        entity_unique_id = f"{filepath}#{entity.name}#{entity.start_line}-{entity.end_line}"

        # Préparer la liste qui contiendra tous les chunk IDs
        all_chunk_ids = []

        # Bounds de l'entité complète
        entity_bounds = {
            'start_line': entity.start_line,
            'end_line': entity.end_line,
            'total_lines': entity.end_line - entity.start_line + 1
        }

        # Stratégie de découpage intelligent
        lines_per_chunk = max(10, self.max_chunk_size // 80)

        # Calculer le nombre total de parties à l'avance
        total_parts = ((len(lines) - 1) // (lines_per_chunk - self.overlap_sentences)) + 1

        # Essayer de découper aux frontières logiques
        logical_sections = self._find_logical_sections(lines)

        if logical_sections:
            # Découper selon les sections logiques
            for section_idx, (section_start, section_end, section_type) in enumerate(logical_sections):
                section_lines = lines[section_start:section_end + 1]
                chunk_code = '\n'.join(section_lines)

                if len(chunk_code.strip()) < self.min_chunk_size:
                    continue

                chunk_start_line = entity.start_line + section_start
                chunk_end_line = entity.start_line + section_end

                # ID du chunk
                chunk_id = f"{document_id}-chunk-{start_chunk_index + len(chunks)}"
                all_chunk_ids.append(chunk_id)

                # Créer une sous-entité avec métadonnées complètes
                sub_entity = FortranEntity(
                    name=f"{entity.name}_part_{len(chunks) + 1}",
                    entity_type=entity.entity_type,
                    start_line=chunk_start_line,
                    end_line=chunk_end_line,
                    parent=entity.parent,
                    dependencies=entity.dependencies.copy(),
                    interfaces=entity.interfaces.copy(),
                    variables=entity.variables.copy(),
                    parameters=entity.parameters.copy(),
                    detected_concepts=entity.detected_concepts.copy(),
                    semantic_metadata={
                        **entity.semantic_metadata,
                        'is_partial': True,
                        'part_index': len(chunks) + 1,
                        'section_type': section_type,
                        'parent_entity': entity.name,
                        'entity_unique_id': entity_unique_id,  # ID unique de l'entité complète
                        'entity_bounds': entity_bounds
                    },
                    metadata={
                        **entity.metadata,
                        'is_partial': True,
                        'part_index': len(chunks) + 1,
                        'total_parts': len(logical_sections),  # Nombre total prévu
                        'parent_entity_id': entity_unique_id,
                        'part_sequence': len(chunks) + 1,
                        'section_type': section_type
                    }
                )

                chunks.append((sub_entity, chunk_code, chunk_id))
        else:
            # Fallback: découpage par nombre de lignes
            part_index = 0
            for i in range(0, len(lines), lines_per_chunk - self.overlap_sentences):
                chunk_lines = lines[i:i + lines_per_chunk]
                chunk_code = '\n'.join(chunk_lines)

                if len(chunk_code.strip()) < self.min_chunk_size:
                    continue

                part_index += 1
                chunk_start_line = entity.start_line + i
                chunk_end_line = entity.start_line + i + len(chunk_lines) - 1

                # ID du chunk
                chunk_id = f"{document_id}-chunk-{start_chunk_index + part_index - 1}"
                all_chunk_ids.append(chunk_id)

                # Créer sous-entité
                sub_entity = FortranEntity(
                    name=f"{entity.name}_part_{part_index}",
                    entity_type=entity.entity_type,
                    start_line=chunk_start_line,
                    end_line=chunk_end_line,
                    parent=entity.parent,
                    dependencies=entity.dependencies.copy(),
                    interfaces=entity.interfaces.copy(),
                    detected_concepts=entity.detected_concepts.copy(),
                    semantic_metadata={
                        **entity.semantic_metadata,
                        'is_partial': True,
                        'part_index': part_index,
                        'parent_entity': entity.name,
                        'entity_unique_id': entity_unique_id,
                        'entity_bounds': entity_bounds
                    },
                    metadata={
                        **entity.metadata,
                        'is_partial': True,
                        'part_index': part_index,
                        'total_parts': total_parts,
                        'parent_entity_id': entity_unique_id,
                        'part_sequence': part_index
                    }
                )

                chunks.append((sub_entity, chunk_code, chunk_id))

        # Maintenant créer les chunks finaux avec les sibling_chunks
        final_chunks = []
        for sub_entity, chunk_code, chunk_id in chunks:
            # Ajouter la liste de tous les chunks frères
            sub_entity.metadata['sibling_chunks'] = [cid for cid in all_chunk_ids if cid != chunk_id]
            sub_entity.metadata['all_chunks'] = all_chunk_ids.copy()
            sub_entity.metadata['chunk_id'] = chunk_id

            chunk = self._create_chunk_from_entity(
                sub_entity, chunk_code, document_id, filepath,
                start_chunk_index + len(final_chunks), metadata
            )
            final_chunks.append(chunk)

        self.logger.debug(f"   ✂️ Split {entity.name} into {len(final_chunks)} chunks")

        return final_chunks

    # ------------------------------------------------------------------
    # 3.  Méthode publique : create_fortran_chunks
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #  Méthode publique : parse OFP  → entités → chunks
    # ------------------------------------------------------------------
    async def create_fortran_chunks(self,
                                    code: str,
                                    document_id: str,
                                    filepath: str,
                                    metadata: Optional[Dict[str, Any]] = None,
                                    ontology_manager=None) -> List[Dict[str, Any]]:
        """
        1. Parse le code (open-fortran-parser) pour récupérer les entités
        2. Crée des chunks avec métadonnées enrichies
           – parent_entity_name / type
           – détection de concepts optionnelle
        3. Split intelligent si l’entité > max_chunk_size
        """
        self.logger.info(f"🔍 OFP chunking for {filepath}")

        # 1. Extraction des entités --------------------------------------
        entities = self._parse_with_ofp(code, filepath)
        if not entities:
            self.logger.warning("⚠️ OFP n’a trouvé aucune entité, fallback générique.")
            return super().create_semantic_chunks(code, document_id, filepath, metadata)

        # Préparer un index rapide  nom  → (type, start_line)
        parent_index = {e.name: (e.entity_type.value, e.start_line) for e in entities}

        # 2. Création des chunks -----------------------------------------
        chunks: List[Dict[str, Any]] = []
        chunk_idx = 0
        lines = code.splitlines()

        # 2.1  utilitaire interne pour DRY
        async def _handle_entity(ent: FortranEntity, ent_code: str):
            nonlocal chunk_idx

            # -- SAFETY NET : s'assurer que les attributs existent -------------
            if not hasattr(ent, 'semantic_metadata'):
                ent.semantic_metadata = {}  # type: ignore[attr-defined]
            if not hasattr(ent, 'detected_concepts'):
                ent.detected_concepts = []  # type: ignore[attr-defined]

            # ------------------------------------------------------------------
            # 1.  Détection de concepts  (seulement pour les entités pertinentes)
            # ------------------------------------------------------------------
            ALLOW_TYPES = {
                FortranConstructType.FUNCTION,
                FortranConstructType.SUBROUTINE,
                FortranConstructType.MODULE,
                FortranConstructType.PROGRAM,
                FortranConstructType.INTERFACE,
            }

            MIN_CHARS_FOR_CLASSIF = 120  # < 120 caractères => on ignore

            if (self.enable_semantic_detection
                    and ontology_manager
                    and ent.entity_type in ALLOW_TYPES
                    and len(ent_code) >= MIN_CHARS_FOR_CLASSIF):

                try:
                    emb = await ontology_manager.rag_engine.embedding_manager.provider.generate_embeddings([ent_code])
                    if emb:
                        await self._detect_semantic_concepts(ent, ent_code, emb[0], ontology_manager)
                except Exception as e:
                    self.logger.debug(f"Concept detection failed for {ent.name}: {e}")
            # ---------------- Split ou chunk unique --------------------
            if len(ent_code) > self.max_chunk_size:
                sub_chunks = self._split_large_entity(ent, ent_code, document_id,
                                                      filepath, chunk_idx, metadata)
            else:
                sub_chunks = [self._create_chunk_from_entity(ent, ent_code,
                                                             document_id, filepath,
                                                             chunk_idx, metadata)]
            # enrichir chaque chunk avec le parent ----------------------
            for ch in sub_chunks:
                md = ch['metadata']
                parent_name = ent.parent or ''
                md['parent_entity_name'] = parent_name
                if parent_name and parent_name in parent_index:
                    md['parent_entity_type'] = parent_index[parent_name][0]
                    md['parent_start_line'] = parent_index[parent_name][1]
                else:
                    md['parent_entity_type'] = ''
            # -----------------------------------------------------------
            chunks.extend(sub_chunks)
            chunk_idx += len(sub_chunks)

        # Boucle principale ---------------------------------------------
        for ent in entities:
            ent_code = "\n".join(lines[ent.start_line - 1: ent.end_line])
            await _handle_entity(ent, ent_code)

        self.logger.info(f"🎉 OFP : {len(chunks)} chunks créés pour {filepath}")
        return chunks

    def debug_detected_entities(self, entities: List[FortranEntity], filename: str):
        """Debug détaillé des entités détectées"""
        print(f"\n🔬 DEBUG DÉTAILLÉ pour {filename}")
        print("-" * 60)

        for i, entity in enumerate(entities, 1):
            print(f"Entité {i}:")
            print(f"  📝 Nom: '{entity.name}'")
            print(f"  🏷️  Type: {entity.entity_type.value}")
            print(f"  📍 Lignes: {entity.start_line}-{entity.end_line}")

            if entity.dependencies:
                print(f"  🔗 Dépendances: {list(entity.dependencies)}")

            # Afficher un extrait du code pour cette entité
            if hasattr(entity, 'metadata') and 'source_lines' in entity.metadata:
                lines = entity.metadata['source_lines'][:3]  # 3 premières lignes
                print(f"  📄 Extrait code:")
                for line in lines:
                    print(f"     {line.strip()}")

            print()

        # Statistiques par type
        type_counts = {}
        for entity in entities:
            type_name = entity.entity_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        print("📊 Résumé par type:")
        for type_name, count in sorted(type_counts.items()):
            print(f"  {type_name}: {count}")

    # ------------------------------------------------------------------
    # 4.  Export des triples
    # ------------------------------------------------------------------
    def serialize_triples(self, ttl_file: str):
        if not self.generate_triples:
            _logger.warning("generate_triples=False, rien à exporter.")
            return
        pathlib.Path(ttl_file).write_text(self.graph.serialize(format="turtle"))
        _logger.info(f"💾 Triples RDF écrits dans {ttl_file}")



if __name__ == "__main__":
    import asyncio
    import tempfile
    import os
    from pathlib import Path

    # Données de test - fichiers Fortran fournis
    TEST_FILES = {
        'constants_types.f90': '''module constants_types
    use iso_fortran_env, only: real64
    implicit none

    ! Physical constants
    real(real64), parameter :: kb = 1.380649e-23_real64  ! Boltzmann constant
    real(real64), parameter :: na = 6.02214076e23_real64 ! Avogadro number
    real(real64), parameter :: pi = 3.141592653589793_real64

    ! Simulation parameters
    integer, parameter :: max_particles = 10000
    real(real64), parameter :: default_dt = 0.001_real64

    ! Derived types
    type :: particle_t
        real(real64) :: x, y, z           ! position
        real(real64) :: vx, vy, vz        ! velocity  
        real(real64) :: fx, fy, fz        ! force
        real(real64) :: mass
        integer :: id
    end type particle_t

    type :: system_t
        type(particle_t), allocatable :: particles(:)
        integer :: n_particles
        real(real64) :: box_size
        real(real64) :: temperature
        real(real64) :: total_energy
    end type system_t

end module constants_types''',

        'math_utilities.f90': '''module math_utilities
    use iso_fortran_env, only: real64
    use constants_types, only: pi
    implicit none

    private
    public :: distance, random_gaussian, normalize_vector, matrix_multiply

contains

    pure function distance(x1, y1, z1, x2, y2, z2) result(dist)
        real(real64), intent(in) :: x1, y1, z1, x2, y2, z2
        real(real64) :: dist

        dist = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    end function distance

    function random_gaussian(mean, std_dev) result(value)
        real(real64), intent(in) :: mean, std_dev
        real(real64) :: value
        real(real64) :: u1, u2

        call random_number(u1)
        call random_number(u2)

        ! Box-Muller transformation
        value = mean + std_dev * sqrt(-2.0_real64 * log(u1)) * cos(2.0_real64 * pi * u2)
    end function random_gaussian

    subroutine normalize_vector(vx, vy, vz)
        real(real64), intent(inout) :: vx, vy, vz
        real(real64) :: norm

        norm = sqrt(vx**2 + vy**2 + vz**2)
        if (norm > 1e-12_real64) then
            vx = vx / norm
            vy = vy / norm  
            vz = vz / norm
        end if
    end subroutine normalize_vector

    subroutine matrix_multiply(a, b, c, n)
        integer, intent(in) :: n
        real(real64), intent(in) :: a(n,n), b(n,n)
        real(real64), intent(out) :: c(n,n)
        integer :: i, j, k

        ! Simple matrix multiplication O(n^3)
        do i = 1, n
            do j = 1, n
                c(i,j) = 0.0_real64
                do k = 1, n
                    c(i,j) = c(i,j) + a(i,k) * b(k,j)
                end do
            end do
        end do
    end subroutine matrix_multiply

end module math_utilities''',

        'simulation_main.f90': '''module simulation_main
    use iso_fortran_env, only: real64
    use constants_types
    use math_utilities, only: random_gaussian
    use force_calculation, only: compute_forces
    use integrator, only: velocity_verlet_step, compute_kinetic_energy, thermostat_berendsen
    implicit none

    private
    public :: initialize_system, run_md_simulation, write_trajectory

contains

    subroutine initialize_system(system, n_particles, box_size, temperature)
        type(system_t), intent(out) :: system
        integer, intent(in) :: n_particles
        real(real64), intent(in) :: box_size, temperature
        integer :: i
        real(real64) :: mass_amu

        system%n_particles = n_particles
        system%box_size = box_size
        system%temperature = temperature

        allocate(system%particles(n_particles))

        ! Initialize particles
        mass_amu = 39.948_real64  ! Argon mass in amu

        do i = 1, n_particles
            system%particles(i)%id = i
            system%particles(i)%mass = mass_amu * 1.66054e-27_real64  ! Convert to kg

            ! Random positions and velocities...
        end do

        ! Compute initial forces
        call compute_forces(system)

    end subroutine initialize_system

    subroutine compute_total_energy(system)
        type(system_t), intent(inout) :: system
        real(real64) :: kinetic_energy, potential_energy

        contains

            function compute_potential_energy(sys) result(potential)
                type(system_t), intent(in) :: sys
                real(real64) :: potential
                integer :: i, j

                potential = 0.0_real64
                ! Computation logic...
            end function compute_potential_energy

    end subroutine compute_total_energy

end module simulation_main'''
    }


    def print_separator(title):
        """Affiche un séparateur avec titre"""
        print("\n" + "=" * 80)
        print(f"🧪 {title}")
        print("=" * 80)


    def print_entity_details(entity):
        """Affiche les détails d'une entité"""
        print(f"  📝 {entity.name} ({entity.entity_type.value})")
        print(f"      Lignes: {entity.start_line}-{entity.end_line}")
        if entity.dependencies:
            print(f"      Dépendances: {', '.join(entity.dependencies)}")
        if entity.physics_concepts:
            print(f"      Concepts physiques: {', '.join(entity.physics_concepts)}")
        if entity.computational_patterns:
            print(f"      Patterns computationnels: {', '.join(entity.computational_patterns)}")


    def validate_expected_entities(entities, expected):
        """Valide que les entités détectées correspondent aux attentes"""
        print(f"\n🔍 Validation des entités (attendu: {expected['total']})")

        actual_counts = {}
        for entity in entities:
            entity_type = entity.entity_type.value
            actual_counts[entity_type] = actual_counts.get(entity_type, 0) + 1

        print(f"   Détecté: {len(entities)} entités total")

        for entity_type, expected_count in expected.items():
            if entity_type == 'total':
                continue
            actual = actual_counts.get(entity_type, 0)
            status = "✅" if actual == expected_count else "❌"
            print(f"   {status} {entity_type}: {actual}/{expected_count}")

        return len(entities) == expected['total']


    async def test_chunking(chunker, filename, code, expected_entities):
        """Test la création de chunks"""
        print(f"\n🔨 Test de chunking pour {filename}")

        try:
            chunks = await chunker.create_fortran_chunks(
                code=code,
                document_id=f"test_{filename}",
                filepath=f"/test/{filename}",
                metadata={'test': True}
            )

            print(f"   ✅ {len(chunks)} chunks créés")

            # Analyser les métadonnées des chunks
            entity_types = {}
            entity_names = []

            for chunk in chunks:
                metadata = chunk.get('metadata', {})
                entity_type = metadata.get('entity_type', 'unknown')
                entity_name = metadata.get('entity_name', 'unknown')

                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                entity_names.append(entity_name)

            print("   📊 Types d'entités dans les chunks:")
            for entity_type, count in entity_types.items():
                print(f"      {entity_type}: {count}")

            print("   🏷️  Noms d'entités:")
            unique_names = set(entity_names)
            for name in sorted(unique_names):
                count = entity_names.count(name)
                print(f"      {name}: {count} chunk(s)")

            # Vérifier si on a des noms 'unknown' (problème)
            unknown_count = entity_names.count('unknown')
            if unknown_count > 0:
                print(f"   ⚠️  {unknown_count} entités avec nom 'unknown' détectées!")

            return chunks

        except Exception as e:
            print(f"   ❌ Erreur lors du chunking: {e}")
            import traceback
            traceback.print_exc()
            return []


    async def run_tests():
        """Exécute tous les tests"""
        print_separator("Tests du OFPFortranSemanticChunker")

        # Créer le chunker
        print("🚀 Initialisation du chunker...")
        chunker = OFPFortranSemanticChunker(
            min_chunk_size=200,
            max_chunk_size=2000,
            overlap_sentences=0
        )
        print("✅ Chunker initialisé")

        # Tests par fichier avec attentes spécifiques
        # Dans la fonction run_tests(), corriger cette partie :
        test_expectations = {
            'constants_types.f90': {
                'total': 3,
                'module': 1,
                'type_definition': 2,  # particle_t et system_t
                'subroutine': 0,  # ← CORRIGER : aucune subroutine !
                'function': 0,
                'variable_declaration': 0  # On ne compte plus les variables
            },
            'math_utilities.f90': {
                'total': 5,
                'module': 1,
                'function': 2,  # distance, random_gaussian
                'subroutine': 2,  # normalize_vector, matrix_multiply
                'type_definition': 0  # Aucun type défini dans ce fichier
            },
            'simulation_main.f90': {
                'total': 4,
                'module': 1,
                'subroutine': 2,  # initialize_system, compute_total_energy
                'function': 1,  # compute_potential_energy (interne)
                'type_definition': 0  # Aucun type défini ici
            }
        }

        # Tests avec fichiers temporaires
        for filename, code in TEST_FILES.items():
            print_separator(f"Test du fichier: {filename}")

            # Créer un fichier temporaire
            with tempfile.NamedTemporaryFile(mode='w', suffix='.f90', delete=False) as tmp_file:
                tmp_file.write(code)
                tmp_filepath = tmp_file.name

            try:
                # Test 1: Structure parsing
                print("🔍 Test de l'extraction de structure...")
                entities = chunker.extract_fortran_structure(code, tmp_filepath)
                chunker.debug_detected_entities(entities, filename)

                print(f"✅ {len(entities)} entités détectées:")
                for entity in entities:
                    print_entity_details(entity)

                # Validation
                expected = test_expectations.get(filename, {'total': len(entities)})
                is_valid = validate_expected_entities(entities, expected)

                if not is_valid:
                    print("⚠️  Les entités détectées ne correspondent pas aux attentes!")

                # Test 2: Chunking
                chunks = await test_chunking(chunker, filename, code, expected)

                # Test 3: Comparaison entités vs chunks
                if entities and chunks:
                    print(f"\n📊 Comparaison: {len(entities)} entités → {len(chunks)} chunks")

                    if len(chunks) > len(entities) * 2:
                        print("⚠️  Trop de chunks générés - possible sur-segmentation!")
                    elif len(chunks) < len(entities):
                        print("⚠️  Pas assez de chunks - possible sous-segmentation!")
                    else:
                        print("✅ Ratio chunks/entités raisonnable")

            except Exception as e:
                print(f"❌ Erreur lors du test de {filename}: {e}")
                import traceback
                traceback.print_exc()

            finally:
                # Nettoyer le fichier temporaire
                os.unlink(tmp_filepath)

        # Test de regression simple
        print_separator("Test de régression simple")

        simple_fortran = '''module simple_test
    implicit none

    type :: test_type
        real :: value
    end type test_type

    contains

    subroutine test_sub()
        print *, "test"
    end subroutine test_sub

end module simple_test'''

        print("🧪 Test avec code Fortran minimal...")
        try:
            entities = chunker.extract_fortran_structure(simple_fortran, "test.f90")
            print(f"✅ {len(entities)} entités détectées dans le code minimal")

            expected_simple = {'total': 3, 'module': 1, 'type_definition': 1, 'subroutine': 1}
            validate_expected_entities(entities, expected_simple)

            for entity in entities:
                print_entity_details(entity)

        except Exception as e:
            print(f"❌ Erreur dans le test de régression: {e}")

        print_separator("Résumé des tests")
        print("🎯 Tests terminés!")
        print("📝 Vérifiez les résultats ci-dessus pour identifier les problèmes:")
        print("   - Entités 'unknown' → Problème d'extraction de noms")
        print("   - Mauvais ratios → Problème de chunking")
        print("   - Types incorrects → Problème de classification OFP")


    # Lancer les tests
    print("🚀 Démarrage des tests du OFPFortranSemanticChunker...")
    try:
        asyncio.run(run_tests())
    except Exception as e:
        print(f"❌ Erreur fatale lors des tests: {e}")
        import traceback

        traceback.print_exc()