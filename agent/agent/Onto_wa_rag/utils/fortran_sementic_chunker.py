"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# utils/fortran_semantic_chunker.py
import os
import re
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
import hashlib
from datetime import datetime
from enum import Enum
import tree_sitter

# f2py parser pour python

try:
    import tree_sitter_fortran as tsfortran
    from tree_sitter import Language, Parser, Node
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    # Définir des classes vides pour éviter les erreurs
    class Node:
        pass
import fparser.two.parser as fp2parser
from fparser.two.utils import walk
from fparser.common.readfortran import FortranFileReader

from .semantic_chunker import SemanticChunker, DocumentSection


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


class FortranSemanticChunker(SemanticChunker):
    """Chunker spécialisé pour le code Fortran avec support BigDFT"""

    def __init__(
            self,
            min_chunk_size: int = 200,
            max_chunk_size: int = 2000,  # Plus grand pour le code
            overlap_sentences: int = 0,  # Pas d'overlap pour le code
            respect_boundaries: bool = True,
            tree_sitter_lib_path: Optional[str] = None,
            enable_physics_detection: bool = True,
            ontology_manager=None
    ):
        super().__init__(min_chunk_size, max_chunk_size, overlap_sentences, respect_boundaries)

        self.logger = logging.getLogger(__name__)
        self.enable_physics_detection = enable_physics_detection
        self.ontology_manager = ontology_manager

        # Initialiser Tree-sitter
        self._init_tree_sitter()

        # Initialiser la construiction des patterns avec l'ontologie
        self._init_ontology()

    def _init_ontology(self):
        # Patterns BigDFT spécifiques
        if self.ontology_manager:
            self.bigdft_patterns = self._build_patterns_from_ontology()
        else:
            self.bigdft_patterns = {
                'wavelet_operations': [
                    r'wavelet_transform', r'convolution_3d', r'magic_filter',
                    r'scf_wavelet', r'psolver_wavelet'
                ],
                'mpi_patterns': [
                    r'mpi_comm_', r'mpi_allreduce', r'mpi_bcast',
                    r'distributed_array', r'parallel_sum'
                ],
                'physics_concepts': {
                    'dft': [r'density', r'potential', r'hamiltonian', r'energy'],
                    'wavelet': [r'daubechies', r'scaling_function', r'multiresolution'],
                    'optimization': [r'bfgs', r'conjugate_gradient', r'diis'],
                    'pseudopotential': [r'psp', r'kleinman', r'goedecker', r'gth']
                }
            }

    def _init_tree_sitter(self, lib_path: Optional[str] = None):
        """Initialise Tree-sitter pour Fortran avec tree-sitter-fortran 0.5.1"""
        try:
            import tree_sitter_fortran as tsfortran
            from tree_sitter import Parser, Language

            # Convertir le PyCapsule en objet Language
            language_capsule = tsfortran.language()
            language = Language(language_capsule)

            self.ts_parser = Parser()
            self.ts_parser.language = language

            self.tree_sitter_available = True
            self.logger.info("Tree-sitter initialized successfully with tree-sitter-fortran")

        except ImportError as e:
            self.logger.warning(f"tree-sitter-fortran not available: {e}")
            self.tree_sitter_available = False
        except Exception as e:
            self.logger.warning(f"Tree-sitter initialization failed: {e}")
            self.tree_sitter_available = False

    def extract_fortran_structure(self, code: str, filepath: str) -> List[FortranEntity]:
        """Extrait la structure complète du code Fortran"""
        entities = []

        # Essayer d'abord avec FPARSER2
        try:
            entities.extend(self._extract_with_fparser2(code, filepath))
        except Exception as e:
            self.logger.debug(f"FPARSER2 parsing failed for {filepath}: {e}")

        # Compléter avec Tree-sitter si disponible
        if self.tree_sitter_available:
            try:
                ts_entities = self._extract_with_tree_sitter(code)
                # Merger intelligemment les résultats
                entities = self._merge_parsing_results(entities, ts_entities)
            except Exception as e:
                self.logger.debug(f"Tree-sitter parsing failed: {e}")

        # Si aucun parser n'a fonctionné, utiliser l'analyse regex
        if not entities:
            entities = self._extract_with_regex(code)

        return entities

    def _extract_with_fparser2(self, code: str, filepath: str) -> List[FortranEntity]:
        """Extraction sémantique avec FPARSER2"""
        entities = []

        try:
            # Parser le code
            reader = FortranFileReader(code)
            f2008_parser = fp2parser.FortranReader(reader)
            parse_tree = fp2parser.Fortran2008.program(reader=f2008_parser)

            # Parcourir l'AST
            for node in walk(parse_tree):
                entity = self._process_fparser2_node(node)
                if entity:
                    entities.append(entity)

        except Exception as e:
            self.logger.debug(f"FPARSER2 detailed error: {e}")
            raise

        return entities

    def _process_fparser2_node(self, node) -> Optional[FortranEntity]:
        """Traite un nœud FPARSER2 pour extraire une entité"""
        entity = None

        node_type = type(node).__name__

        if node_type == 'Module':
            entity = FortranEntity(
                name=str(node.children[1]),
                entity_type=FortranConstructType.MODULE,
                start_line=node.start_line,
                end_line=node.end_line
            )
            # Extraire les USE statements
            for child in walk(node):
                if type(child).__name__ == 'Use_Stmt':
                    entity.dependencies.add(str(child.children[1]))

        elif node_type == 'Subroutine_Subprogram':
            entity = FortranEntity(
                name=str(node.children[0].children[1]),
                entity_type=FortranConstructType.SUBROUTINE,
                start_line=node.start_line,
                end_line=node.end_line
            )
            # Extraire les arguments
            self._extract_dummy_args(node, entity)

        elif node_type == 'Function_Subprogram':
            entity = FortranEntity(
                name=str(node.children[0].children[1]),
                entity_type=FortranConstructType.FUNCTION,
                start_line=node.start_line,
                end_line=node.end_line
            )
            # Extraire le type de retour et les arguments
            self._extract_function_info(node, entity)

        elif node_type == 'Derived_Type_Def':
            entity = FortranEntity(
                name=str(node.children[0].children[1]),
                entity_type=FortranConstructType.TYPE_DEFINITION,
                start_line=node.start_line,
                end_line=node.end_line
            )
            # Extraire les composants du type
            self._extract_type_components(node, entity)

        return entity

    def _extract_with_tree_sitter(self, code: str) -> List[FortranEntity]:
        """Extraction syntaxique avec Tree-sitter"""
        entities = []

        tree = self.ts_parser.parse(bytes(code, 'utf8'))

        def visit_node(node: Node, parent_name: Optional[str] = None):
            if node.type == 'module':
                entity = self._create_entity_from_ts_node(
                    node, FortranConstructType.MODULE, code
                )
                if entity:
                    entities.append(entity)
                    parent_name = entity.name

            elif node.type == 'function':
                entity = self._create_entity_from_ts_node(
                    node, FortranConstructType.FUNCTION, code, parent_name
                )
                if entity:
                    entities.append(entity)

            elif node.type == 'subroutine':
                entity = self._create_entity_from_ts_node(
                    node, FortranConstructType.SUBROUTINE, code, parent_name
                )
                if entity:
                    entities.append(entity)

            elif node.type == 'derived_type_definition':
                entity = self._create_entity_from_ts_node(
                    node, FortranConstructType.TYPE_DEFINITION, code, parent_name
                )
                if entity:
                    entities.append(entity)

            elif node.type == 'use_statement':
                # Ajouter la dépendance au parent
                if parent_name and entities:
                    module_name = self._extract_ts_identifier(node, code)
                    for entity in entities:
                        if entity.name == parent_name:
                            entity.dependencies.add(module_name)

            # Parcourir les enfants
            for child in node.children:
                visit_node(child, parent_name)

        visit_node(tree.root_node)
        return entities

    def _create_entity_from_ts_node(
            self,
            node: Node,
            entity_type: FortranConstructType,
            code: str,
            parent: Optional[str] = None
    ) -> Optional[FortranEntity]:
        """Crée une entité à partir d'un nœud Tree-sitter"""
        name = self._extract_ts_identifier(node, code)
        if not name:
            return None

        return FortranEntity(
            name=name,
            entity_type=entity_type,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            parent=parent
        )

    def _extract_with_regex(self, code: str) -> List[FortranEntity]:
        """Extraction basique avec regex (fallback)"""
        entities = []
        lines = code.split('\n')

        patterns = {
            FortranConstructType.MODULE: re.compile(
                r'^\s*module\s+(\w+)', re.IGNORECASE
            ),
            FortranConstructType.SUBROUTINE: re.compile(
                r'^\s*subroutine\s+(\w+)', re.IGNORECASE
            ),
            FortranConstructType.FUNCTION: re.compile(
                r'^\s*(?:pure\s+|elemental\s+|recursive\s+)*(?:real|integer|logical|complex|character|type)\s*(?:\([^)]*\))?\s*function\s+(\w+)',
                re.IGNORECASE
            ),
            FortranConstructType.TYPE_DEFINITION: re.compile(
                r'^\s*type\s*(?:::)?\s*(\w+)', re.IGNORECASE
            ),
            FortranConstructType.PROGRAM: re.compile(
                r'^\s*program\s+(\w+)', re.IGNORECASE
            )
        }

        end_patterns = {
            FortranConstructType.MODULE: re.compile(r'^\s*end\s+module', re.IGNORECASE),
            FortranConstructType.SUBROUTINE: re.compile(r'^\s*end\s+subroutine', re.IGNORECASE),
            FortranConstructType.FUNCTION: re.compile(r'^\s*end\s+function', re.IGNORECASE),
            FortranConstructType.TYPE_DEFINITION: re.compile(r'^\s*end\s+type', re.IGNORECASE),
            FortranConstructType.PROGRAM: re.compile(r'^\s*end\s+program', re.IGNORECASE)
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

            # Vérifier les fins de construction
            for construct_type, pattern in end_patterns.items():
                if pattern.match(line) and current_entities:
                    # Trouver l'entité correspondante
                    for j in range(len(current_entities) - 1, -1, -1):
                        if current_entities[j].entity_type == construct_type:
                            current_entities[j].end_line = i + 1
                            current_entities.pop(j)
                            break

            # Détecter les USE statements
            use_match = re.match(r'^\s*use\s+(\w+)', line, re.IGNORECASE)
            if use_match and current_entities:
                current_entities[-1].dependencies.add(use_match.group(1))

        return entities

    async def _detect_physics_concepts(self, entities: List[FortranEntity], code: str, classifier=None):
        """
        Détecte les concepts physiques dans le code en utilisant le classifier ontologique.
        Utilise les méthodes existantes au lieu de patterns hardcodés.
        """
        print("🧠 Détection des concepts via classifier ontologique...")

        for entity in entities:
            try:
                # Extraire le code de l'entité
                entity_code = self._extract_entity_code(entity, code)

                if len(entity_code.strip()) < 50:  # Ignorer code trop court
                    continue

                # Utiliser les méthodes existantes du classifier pour détecter les concepts
                detected_concepts = await self._detect_concepts_with_classifier(
                    entity_code, classifier
                )

                # Organiser les concepts détectés par catégories
                self._organize_detected_concepts(entity, detected_concepts)

                # Détecter les patterns computationnels avec le classifier
                await self._detect_computational_patterns_with_classifier(
                    entity, entity_code, classifier
                )

                # Détecter les opportunités d'optimisation (garde la logique existante)
                self._detect_optimization_opportunities(entity, entity_code)

            except Exception as e:
                print(f"⚠️ Erreur détection concepts pour {entity.name}: {e}")
                continue

    async def _detect_concepts_with_classifier(self, entity_code: str, classifier) -> List[Dict[str, Any]]:
        """
        Utilise le classifier pour détecter les concepts dans le code.
        Réutilise les méthodes existantes smart_concept_detection ou auto_detect_concepts.
        """
        try:
            # Utiliser directement la méthode smart_concept_detection du classifier
            if hasattr(classifier, 'smart_concept_detection'):
                concepts = await classifier.smart_concept_detection(entity_code)
                return concepts

            # Fallback sur auto_detect_concepts
            elif hasattr(classifier, 'concept_classifier') and classifier.concept_classifier:
                # Générer l'embedding du code
                if hasattr(classifier, 'rag_engine') and classifier.rag_engine:
                    embeddings = await classifier.rag_engine.embedding_manager.provider.generate_embeddings(
                        [entity_code])
                    if embeddings and len(embeddings) > 0:
                        code_embedding = embeddings[0]
                        concepts = await classifier.concept_classifier.auto_detect_concepts(
                            code_embedding,
                            min_confidence=0.4,
                            max_concepts=10
                        )
                        return concepts

            return []

        except Exception as e:
            print(f"⚠️ Erreur lors de la détection avec classifier: {e}")
            return []

    def _organize_detected_concepts(self, entity: FortranEntity, detected_concepts: List[Dict[str, Any]]):
        """
        Organise les concepts détectés dans les bonnes catégories de l'entité.
        """
        for concept_info in detected_concepts:
            concept_uri = concept_info.get('concept_uri', '')
            concept_label = concept_info.get('label', '').lower()
            confidence = concept_info.get('confidence', 0)

            if confidence < 0.3:  # Seuil minimum de confiance
                continue

            # Classifier le concept selon sa place dans l'ontologie
            category = self._classify_concept_from_hierarchy(concept_uri, concept_label)

            if category:
                entity.physics_concepts.add(category)

                # Stocker aussi le concept spécifique dans les métadonnées
                if 'detected_concepts' not in entity.metadata:
                    entity.metadata['detected_concepts'] = []

                entity.metadata['detected_concepts'].append({
                    'concept_uri': concept_uri,
                    'label': concept_label,
                    'confidence': confidence,
                    'category': category
                })

    def _classify_concept_from_hierarchy(self, concept_uri: str, concept_label: str) -> Optional[str]:
        """
        Classifie un concept selon sa hiérarchie dans l'ontologie BigDFT.
        """
        # Classification basée sur le label et l'URI
        label_lower = concept_label.lower()
        uri_lower = concept_uri.lower() if concept_uri else ""

        # Mapping basé sur l'ontologie BigDFT
        if any(keyword in label_lower or keyword in uri_lower
               for keyword in ['density', 'potential', 'hamiltonian', 'energy', 'dft']):
            return 'dft'
        elif any(keyword in label_lower or keyword in uri_lower
                 for keyword in ['wavelet', 'daubechies', 'scaling', 'multiresolution']):
            return 'wavelet'
        elif any(keyword in label_lower or keyword in uri_lower
                 for keyword in ['scf', 'cycle', 'convergence', 'iterative']):
            return 'scf'
        elif any(keyword in label_lower or keyword in uri_lower
                 for keyword in ['pseudo', 'potential', 'goedecker', 'gth']):
            return 'pseudopotential'
        elif any(keyword in label_lower or keyword in uri_lower
                 for keyword in ['exchange', 'correlation']):
            return 'exchange_correlation'
        elif any(keyword in label_lower or keyword in uri_lower
                 for keyword in ['optimization', 'bfgs', 'minimize', 'diis']):
            return 'optimization'

        return None

    async def _detect_computational_patterns_with_classifier(self, entity: FortranEntity,
                                                             entity_code: str, classifier):
        """
        Détecte les patterns computationnels.
        """
        try:
            entity_code_lower = entity_code.lower()

            # Détecter patterns wavelet
            if any(pattern in entity_code_lower for pattern in [
                'wavelet', 'convolution', 'transform', 'scaling', 'daubechies'
            ]):
                entity.computational_patterns.add('wavelet_computation')

            # Détecter patterns parallèles
            if any(pattern in entity_code_lower for pattern in [
                'mpi_', 'openmp', 'parallel', 'distributed'
            ]):
                entity.computational_patterns.add('parallel_computation')

            # Détecter patterns algèbre linéaire
            if any(pattern in entity_code_lower for pattern in [
                'matrix', 'blas', 'lapack', 'linear'
            ]):
                entity.computational_patterns.add('linear_algebra')

        except Exception as e:
            print(f"⚠️ Erreur détection patterns computationnels: {e}")

    def _detect_physics_concepts_fallback(self, entities: List[FortranEntity], code: str):
        """
        Méthode de fallback utilisant les patterns hardcodés si pas de classifier.
        """
        print("⚠️ Utilisation des patterns de fallback")

        for entity in entities:
            entity_code = self._extract_entity_code(entity, code)
            entity_code_lower = entity_code.lower()

            # Utiliser les patterns de fallback
            fallback_patterns = {
                'dft': [r'density', r'potential', r'hamiltonian', r'energy'],
                'wavelet': [r'daubechies', r'scaling_function', r'multiresolution'],
                'optimization': [r'bfgs', r'conjugate_gradient', r'diis'],
                'pseudopotential': [r'psp', r'kleinman', r'goedecker', r'gth']
            }

            for concept_type, patterns in fallback_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, entity_code_lower):
                        entity.physics_concepts.add(concept_type)

            # Patterns computationnels de fallback
            if re.search(r'wavelet_transform|convolution_3d', entity_code_lower):
                entity.computational_patterns.add('wavelet_computation')

            if re.search(r'mpi_comm_|mpi_allreduce', entity_code_lower):
                entity.computational_patterns.add('parallel_computation')

            self._detect_optimization_opportunities(entity, entity_code)


    def old_detect_physics_concepts(self, entities: List[FortranEntity], code: str):
        """Détecte les concepts physiques dans le code"""
        code_lower = code.lower()

        for entity in entities:
            # Extraire le code de l'entité
            entity_code = self._extract_entity_code(entity, code)
            entity_code_lower = entity_code.lower()

            # Détecter les concepts physiques
            for concept_type, patterns in self.bigdft_patterns['physics_concepts'].items():
                for pattern in patterns:
                    if re.search(pattern, entity_code_lower):
                        entity.physics_concepts.add(concept_type)

            # Détecter les patterns computationnels
            if any(re.search(p, entity_code_lower) for p in self.bigdft_patterns['wavelet_operations']):
                entity.computational_patterns.add('wavelet_computation')

            if any(re.search(p, entity_code_lower) for p in self.bigdft_patterns['mpi_patterns']):
                entity.computational_patterns.add('parallel_computation')

            # Détecter les optimisations possibles
            self._detect_optimization_opportunities(entity, entity_code)

    def _detect_optimization_opportunities(self, entity: FortranEntity, code: str):
        """Détecte les opportunités d'optimisation"""
        opportunities = []

        # Boucles non vectorisées
        if re.search(r'do\s+\w+\s*=.*\n.*?\n\s*end\s*do', code, re.IGNORECASE | re.DOTALL):
            if not re.search(r'!\$OMP|!\$ACC|!DIR\$', code):
                opportunities.append('vectorization_opportunity')

        # Arrays temporaires
        if re.search(r'allocate\s*\([^)]+\)', code, re.IGNORECASE):
            opportunities.append('memory_optimization_opportunity')

        # Appels de fonction dans les boucles
        if re.search(r'do\s+.*?call\s+\w+.*?end\s*do', code, re.IGNORECASE | re.DOTALL):
            opportunities.append('function_inlining_opportunity')

        entity.metadata['optimization_opportunities'] = opportunities

    async def create_fortran_chunks(
            self,
            code: str,
            document_id: str,
            filepath: str,
            metadata: Optional[Dict[str, Any]] = None,
            ontology_manager=None
    ) -> List[Dict[str, Any]]:
        """Crée des chunks sémantiques pour le code Fortran avec debugging"""

        print(f"🔍 Debugging chunking for {filepath}")
        print(f"📏 Code size: {len(code)} characters")
        print(f"📊 Max chunk size: {self.max_chunk_size}")

        # Extraire la structure Fortran
        entities = self.extract_fortran_structure(code, filepath)

        print(f"🎯 Found {len(entities)} entities:")
        for i, entity in enumerate(entities):
            print(f"  {i + 1}. {entity.name} ({entity.entity_type.value}) lines {entity.start_line}-{entity.end_line}")

        if not entities:
            print("⚠️  No entities found, falling back to generic chunking")
            return self.create_semantic_chunks(code, document_id, filepath, metadata)

        # chunk par entité individuelle
        chunks = []
        chunk_index = 0

        # Pour un fichier avec PROGRAM + SUBROUTINES (cas typique Fortran),
        # chaque entité devient un chunk
        for entity in entities:
            print(f"🔨 Processing entity: {entity.name} ({entity.entity_type.value})")

            # Extraire le code de cette entité
            entity_code = self._extract_entity_code(entity, code)
            entity_size = len(entity_code)

            print(f"   📏 Entity size: {entity_size} characters")

            # Si l'entité est trop grande, la découper
            if entity_size > self.max_chunk_size:
                print(f"   ✂️  Entity too large, splitting...")
                entity_chunks = self._split_large_entity(
                    entity, entity_code, document_id, filepath, chunk_index, metadata
                )
                print(f"   ✅ Split into {len(entity_chunks)} chunks")
            else:
                # Créer un chunk unique pour cette entité
                chunk = await self._create_single_chunk(
                    entity, entity_code, document_id, filepath, chunk_index, metadata, ontology_manager
                )
                entity_chunks = [chunk]
                print(f"   ✅ Created 1 chunk")

            chunks.extend(entity_chunks)
            chunk_index += len(entity_chunks)

        print(f"🎉 Total chunks created: {len(chunks)}")
        return chunks

    def _split_large_entity(
            self,
            entity: FortranEntity,
            entity_code: str,
            document_id: str,
            filepath: str,
            start_chunk_index: int,
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Découpe une entité trop grande en plusieurs chunks"""
        chunks = []
        lines = entity_code.split('\n')

        # Stratégie simple : découper par blocs de lignes
        lines_per_chunk = self.max_chunk_size // 80  # ~80 chars par ligne en moyenne
        lines_per_chunk = max(10, lines_per_chunk)  # Minimum 10 lignes par chunk

        for i in range(0, len(lines), lines_per_chunk):
            chunk_lines = lines[i:i + lines_per_chunk]
            chunk_code = '\n'.join(chunk_lines)

            if len(chunk_code.strip()) < self.min_chunk_size:
                continue

            # Calculer les vraies positions de lignes pour ce chunk
            chunk_start_line = entity.start_line + i
            chunk_end_line = entity.start_line + i + len(chunk_lines) - 1

            # Créer une sous-entité
            sub_entity = FortranEntity(
                name=f"{entity.name}_part_{len(chunks) + 1}",
                entity_type=entity.entity_type,
                start_line=chunk_start_line,
                end_line=chunk_end_line,
                dependencies=entity.dependencies,
                physics_concepts=entity.physics_concepts,
                computational_patterns=entity.computational_patterns,
                metadata={**entity.metadata, 'is_partial': True, 'part_index': len(chunks) + 1}
            )

            # Préparer les métadonnées du chunk
            chunk_metadata = {
                'entity_type': sub_entity.entity_type.value,
                'entity_name': sub_entity.name,
                'dependencies': list(sub_entity.dependencies),
                'interfaces': list(sub_entity.interfaces),
                'physics_concepts': list(sub_entity.physics_concepts),
                'computational_patterns': list(sub_entity.computational_patterns),
                'variables': sub_entity.variables,
                'parameters': sub_entity.parameters,
                'content_hash': hashlib.md5(chunk_code.encode()).hexdigest(),
                'chunk_method': 'fortran_semantic',
                'language': 'fortran',
                'fortran_standard': self._detect_fortran_standard(chunk_code),
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'start_pos': chunk_start_line,  # ← POSITIONS DANS METADATA
                'end_pos': chunk_end_line,  # ← POSITIONS DANS METADATA
                **(metadata or {}),
                **sub_entity.metadata
            }

            # Ajouter les métadonnées BigDFT si détectées
            if sub_entity.physics_concepts:
                chunk_metadata['bigdft_domain'] = self._classify_bigdft_domain(sub_entity)

            chunk = {
                "id": f"{document_id}-chunk-{start_chunk_index + len(chunks)}",
                "document_id": document_id,
                "text": chunk_code,
                "start_pos": chunk_start_line,
                "end_pos": chunk_end_line,
                "metadata": chunk_metadata
            }
            chunks.append(chunk)

        return chunks

    async def _create_chunks_for_entity(
            self,
            main_entity: FortranEntity,
            sub_entities: List[FortranEntity],
            code: str,
            document_id: str,
            filepath: str,
            start_chunk_index: int,
            base_metadata: Optional[Dict[str, Any]] = None,
            ontology_manager: Optional = None
    ) -> List[Dict[str, Any]]:
        """Crée des chunks pour une entité et ses sous-entités"""
        chunks = []
        lines = code.split('\n')

        # Extraire le code de l'entité principale
        entity_code = '\n'.join(
            lines[main_entity.start_line - 1:main_entity.end_line]
        )

        # Si l'entité est petite, un seul chunk
        if len(entity_code) <= self.max_chunk_size:
            chunk = await self._create_single_chunk(
                main_entity,
                entity_code,
                document_id,
                filepath,
                start_chunk_index,
                base_metadata,
                ontology_manager
            )
            chunks.append(chunk)

        else:
            # Découper intelligemment
            if sub_entities:
                # Chunk par sous-entité
                chunks = await self._chunk_by_sub_entities(
                    main_entity,
                    sub_entities,
                    lines,
                    document_id,
                    filepath,
                    start_chunk_index,
                    base_metadata,
                    ontology_manager
                )
            else:
                # Découper par sections logiques
                chunks = await self._chunk_by_sections(
                    main_entity,
                    entity_code,
                    document_id,
                    filepath,
                    start_chunk_index,
                    base_metadata,
                    ontology_manager
                )

        return chunks

    async def _create_single_chunk(
            self,
            entity: FortranEntity,
            code: str,
            document_id: str,
            filepath: str,
            chunk_index: int,
            base_metadata: Optional[Dict[str, Any]] = None,
            ontology_manager=None
    ) -> Dict[str, Any]:
        """Crée un chunk unique pour une entité"""

        # Calculer le hash du contenu
        content_hash = hashlib.md5(code.encode()).hexdigest()

        # Détéction sur les concepts
        if self.enable_physics_detection and ontology_manager:
            classifier = getattr(ontology_manager, 'classifier', None)
            if classifier:
                print(f"🧠 Détection concepts pour chunk {entity.name}...")
                try:
                    # Détecter les concepts sur ce chunk (taille appropriée)
                    detected_concepts = await self._detect_concepts_with_classifier(code, classifier)

                    # Organiser les concepts détectés
                    self._organize_detected_concepts(entity, detected_concepts)

                    # Détecter les patterns computationnels
                    await self._detect_computational_patterns_with_classifier(entity, code, classifier)

                except Exception as e:
                    print(f"⚠️ Erreur détection concepts pour {entity.name}: {e}")
                    # Fallback
                    self._detect_physics_concepts_fallback_single(entity, code)
            else:
                # Fallback si pas de classifier
                self._detect_physics_concepts_fallback_single(entity, code)

        # Construire les métadonnées enrichies
        chunk_metadata = base_metadata or {}
        chunk_metadata.update({
            'entity_type': entity.entity_type.value,
            'entity_name': entity.name,
            'dependencies': list(entity.dependencies),
            'interfaces': list(entity.interfaces),
            'physics_concepts': list(entity.physics_concepts),
            'computational_patterns': list(entity.computational_patterns),
            'variables': entity.variables,
            'parameters': entity.parameters,
            'content_hash': content_hash,
            'chunk_method': 'fortran_semantic',
            'language': 'fortran',
            'fortran_standard': self._detect_fortran_standard(code),
            **entity.metadata
        })

        # Ajouter les métadonnées BigDFT si détectées
        if entity.physics_concepts:
            chunk_metadata['bigdft_domain'] = self._classify_bigdft_domain(entity)

        return {
            "id": f"{document_id}-chunk-{chunk_index}",
            "document_id": document_id,
            "text": code,
            "start_pos": entity.start_line,
            "end_pos": entity.end_line,
            "metadata": {
                "filepath": filepath,
                "filename": os.path.basename(filepath),
                "start_pos": entity.start_line,  # ← AJOUTER DANS METADATA
                "end_pos": entity.end_line,  # ← AJOUTER DANS METADATA
                **chunk_metadata
            }
        }

    def _detect_physics_concepts_fallback_single(self, entity: FortranEntity, entity_code: str):
        """
        Méthode de fallback pour un seul chunk/entité.
        """
        entity_code_lower = entity_code.lower()

        # Utiliser les patterns de fallback
        fallback_patterns = {
            'dft': [r'density', r'potential', r'hamiltonian', r'energy'],
            'wavelet': [r'daubechies', r'scaling_function', r'multiresolution'],
            'optimization': [r'bfgs', r'conjugate_gradient', r'diis'],
            'pseudopotential': [r'psp', r'kleinman', r'goedecker', r'gth']
        }

        for concept_type, patterns in fallback_patterns.items():
            for pattern in patterns:
                if re.search(pattern, entity_code_lower):
                    entity.physics_concepts.add(concept_type)

        # Patterns computationnels de fallback
        if re.search(r'wavelet_transform|convolution_3d', entity_code_lower):
            entity.computational_patterns.add('wavelet_computation')

        if re.search(r'mpi_comm_|mpi_allreduce', entity_code_lower):
            entity.computational_patterns.add('parallel_computation')

        self._detect_optimization_opportunities(entity, entity_code)

    async def _chunk_by_sub_entities(
            self,
            main_entity: FortranEntity,
            sub_entities: List[FortranEntity],
            lines: List[str],
            document_id: str,
            filepath: str,
            start_chunk_index: int,
            base_metadata: Optional[Dict[str, Any]] = None,
            ontology_manager: Optional = None
    ) -> List[Dict[str, Any]]:
        """Découpe un module en chunks basés sur ses sous-entités"""
        chunks = []

        # Chunk pour l'en-tête du module (déclarations, USE, etc.)
        header_end = min(sub_entities, key=lambda e: e.start_line).start_line - 1
        if header_end > main_entity.start_line:
            header_code = '\n'.join(lines[main_entity.start_line - 1:header_end])
            header_chunk = await self._create_single_chunk(
                FortranEntity(
                    name=f"{main_entity.name}_header",
                    entity_type=main_entity.entity_type,
                    start_line=main_entity.start_line,
                    end_line=header_end,
                    dependencies=main_entity.dependencies,
                    metadata={'section': 'header'}
                ),
                header_code,
                document_id,
                filepath,
                start_chunk_index,
                base_metadata,
                ontology_manager
            )
            chunks.append(header_chunk)

        # Chunks pour chaque sous-entité
        for i, sub_entity in enumerate(sub_entities):
            sub_code = '\n'.join(
                lines[sub_entity.start_line - 1:sub_entity.end_line]
            )
            sub_chunk = await self._create_single_chunk(
                sub_entity,
                sub_code,
                document_id,
                filepath,
                start_chunk_index + len(chunks),
                base_metadata,
                ontology_manager
            )
            chunks.append(sub_chunk)

        return chunks

    async def _chunk_by_sections(
            self,
            entity: FortranEntity,
            code: str,
            document_id: str,
            filepath: str,
            start_chunk_index: int,
            base_metadata: Optional[Dict[str, Any]] = None,
            ontology_manager: Optional = None
    ) -> List[Dict[str, Any]]:
        """Découpe une entité en sections logiques"""
        chunks = []
        lines = code.split('\n')

        # Identifier les sections
        sections = []
        current_section = {'start': 0, 'lines': [], 'type': 'declarations'}

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()

            # Détecter les changements de section
            if line_lower.startswith('contains'):
                if current_section['lines']:
                    current_section['end'] = i
                    sections.append(current_section)
                current_section = {'start': i, 'lines': [line], 'type': 'contains'}

            elif re.match(r'^\s*!\s*={3,}', line):  # Séparateur de section
                if current_section['lines']:
                    current_section['end'] = i
                    sections.append(current_section)
                current_section = {'start': i + 1, 'lines': [], 'type': 'section'}

            else:
                current_section['lines'].append(line)

        # Dernière section
        if current_section['lines']:
            current_section['end'] = len(lines)
            sections.append(current_section)

        # Créer les chunks
        for i, section in enumerate(sections):
            section_code = '\n'.join(section['lines'])
            if len(section_code.strip()) < self.min_chunk_size:
                continue

            section_entity = FortranEntity(
                name=f"{entity.name}_section_{i}",
                entity_type=entity.entity_type,
                start_line=entity.start_line + section['start'],
                end_line=entity.start_line + section['end'],
                dependencies=entity.dependencies if i == 0 else set(),
                metadata={'section_type': section['type'], 'section_index': i}
            )

            chunk = await self._create_single_chunk(
                section_entity,
                section_code,
                document_id,
                filepath,
                start_chunk_index + len(chunks),
                base_metadata,
                ontology_manager
            )
            chunks.append(chunk)

        return chunks if chunks else [
            await self._create_single_chunk(
                entity, code, document_id, filepath, start_chunk_index, base_metadata, ontology_manager
            )
        ]

    def _extract_entity_code(self, entity: FortranEntity, full_code: str) -> str:
        """Extrait le code d'une entité"""
        lines = full_code.split('\n')
        return '\n'.join(lines[entity.start_line - 1:entity.end_line])

    def _extract_ts_identifier(self, node: Node, code: str) -> Optional[str]:
        """Extrait l'identifiant d'un nœud Tree-sitter"""
        for child in node.children:
            if child.type == 'identifier':
                return code[child.start_byte:child.end_byte]
        return None

    def _merge_parsing_results(
            self,
            fparser_entities: List[FortranEntity],
            ts_entities: List[FortranEntity]
    ) -> List[FortranEntity]:
        """Fusionne intelligemment les résultats des deux parsers"""
        merged = fparser_entities.copy()

        # Ajouter les entités trouvées uniquement par Tree-sitter
        for ts_entity in ts_entities:
            found = False
            for fp_entity in fparser_entities:
                if (fp_entity.name == ts_entity.name and
                        fp_entity.entity_type == ts_entity.entity_type):
                    found = True
                    # Enrichir avec les infos de Tree-sitter
                    if ts_entity.start_line < fp_entity.start_line:
                        fp_entity.start_line = ts_entity.start_line
                    if ts_entity.end_line > fp_entity.end_line:
                        fp_entity.end_line = ts_entity.end_line
                    break

            if not found:
                merged.append(ts_entity)

        return sorted(merged, key=lambda e: e.start_line)

    def _detect_fortran_standard(self, code: str) -> str:
        """Détecte le standard Fortran utilisé"""
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

        return 'fortran90'  # Par défaut

    def _classify_bigdft_domain(self, entity: FortranEntity) -> str:
        """Classifie le domaine BigDFT de l'entité"""
        if 'wavelet' in entity.physics_concepts:
            return 'wavelet_solver'
        elif 'dft' in entity.physics_concepts:
            return 'electronic_structure'
        elif 'parallel_computation' in entity.computational_patterns:
            return 'parallel_infrastructure'
        elif 'optimization' in entity.physics_concepts:
            return 'geometry_optimization'
        else:
            return 'general'

    # Méthodes helper supplémentaires pour FPARSER2
    def _extract_dummy_args(self, node, entity: FortranEntity):
        """Extrait les arguments d'une subroutine/function"""
        # Implementation spécifique à FPARSER2
        pass

    def _extract_function_info(self, node, entity: FortranEntity):
        """Extrait les infos spécifiques aux fonctions"""
        # Implementation spécifique à FPARSER2
        pass

    def _extract_type_components(self, node, entity: FortranEntity):
        """Extrait les composants d'un type dérivé"""
        # Implementation spécifique à FPARSER2
        pass

    def _build_patterns_from_ontology(self) -> Dict[str, Any]:
        """
        Construit les patterns à partir de l'ontologie de manière entièrement générique.
        Exploite toutes les informations disponibles sans mots-clés hardcodés.

        Returns:
            Dictionnaire contenant les patterns organisés par catégories ontologiques
        """
        if not self.ontology_manager:
            self.logger.warning("No ontology manager available, using default patterns")
            return {}

        try:
            patterns = {
                'physics_concepts': {},
                'computational_patterns': {}
            }

            # 1. Extraire les patterns des concepts avec leur hiérarchie complète
            concept_patterns = self._extract_concept_patterns_from_ontology()
            patterns['physics_concepts'].update(concept_patterns)

            # 2. Extraire les patterns des relations avec leurs domaines/portées
            relation_patterns = self._extract_relation_patterns_from_ontology()
            patterns['computational_patterns'].update(relation_patterns)

            # 3. Extraire les patterns des domaines si disponibles
            domain_patterns = self._extract_domain_patterns_from_ontology()
            patterns['physics_concepts'].update(domain_patterns)

            self.logger.info(
                f"Generated patterns from ontology: "
                f"{len(patterns['physics_concepts'])} concept categories, "
                f"{len(patterns['computational_patterns'])} relation categories"
            )

            return patterns

        except Exception as e:
            self.logger.warning(f"Error building patterns from ontology: {e}")
            return {}

    def _extract_concept_patterns_from_ontology(self) -> Dict[str, List[str]]:
        """
        Extrait les patterns à partir de tous les concepts de l'ontologie de manière générique.

        Returns:
            Dictionnaire catégorie_hiérarchique -> liste de patterns
        """
        concept_patterns = {}

        # Parcourir tous les concepts
        for concept_uri, concept in self.ontology_manager.concepts.items():
            try:
                # Extraire toutes les informations sémantiques du concept
                semantics = self.ontology_manager.extract_concept_semantics(concept_uri)

                # Déterminer la catégorie basée sur la hiérarchie
                category = self._determine_concept_category_from_hierarchy(concept_uri)

                if category not in concept_patterns:
                    concept_patterns[category] = []

                # Générer les patterns à partir de toutes les informations disponibles
                patterns = self._generate_patterns_from_concept_semantics(semantics)
                concept_patterns[category].extend(patterns)

            except Exception as e:
                self.logger.debug(f"Error processing concept {concept_uri}: {e}")

        # Nettoyer les doublons et filtrer les patterns vides
        for category in concept_patterns:
            concept_patterns[category] = list(set(filter(None, concept_patterns[category])))

        return concept_patterns

    def _extract_relation_patterns_from_ontology(self) -> Dict[str, List[str]]:
        """
        Extrait les patterns à partir de toutes les relations de l'ontologie de manière générique.

        Returns:
            Dictionnaire catégorie_relation -> liste de patterns
        """
        relation_patterns = {}

        # Parcourir toutes les relations
        for relation_uri, relation in self.ontology_manager.relations.items():
            try:
                # Déterminer la catégorie basée sur le domaine et la portée
                category = self._determine_relation_category_from_domain_range(relation)

                if category not in relation_patterns:
                    relation_patterns[category] = []

                # Générer les patterns à partir de la relation et ses métadonnées
                patterns = self._generate_patterns_from_relation(relation)
                relation_patterns[category].extend(patterns)

            except Exception as e:
                self.logger.debug(f"Error processing relation {relation_uri}: {e}")

        # Nettoyer les doublons
        for category in relation_patterns:
            relation_patterns[category] = list(set(filter(None, relation_patterns[category])))

        return relation_patterns

    def _extract_domain_patterns_from_ontology(self) -> Dict[str, List[str]]:
        """
        Extrait les patterns à partir des domaines de l'ontologie de manière générique.

        Returns:
            Dictionnaire nom_domaine -> liste de patterns
        """
        domain_patterns = {}

        # Parcourir tous les domaines si disponibles
        for domain_name, domain in self.ontology_manager.domains.items():
            try:
                patterns = []

                # Patterns à partir du nom et de la description du domaine
                if domain.name:
                    patterns.extend(self._create_text_patterns(domain.name))

                if domain.description:
                    patterns.extend(self._create_text_patterns(domain.description))

                # Patterns à partir des concepts du domaine
                for concept in domain.concepts:
                    concept_semantics = self.ontology_manager.extract_concept_semantics(concept.uri)
                    patterns.extend(self._generate_patterns_from_concept_semantics(concept_semantics))

                if patterns:
                    domain_patterns[domain_name] = list(set(filter(None, patterns)))

            except Exception as e:
                self.logger.debug(f"Error processing domain {domain_name}: {e}")

        return domain_patterns

    def _determine_concept_category_from_hierarchy(self, concept_uri: str) -> str:
        """
        Détermine la catégorie d'un concept basée sur sa position hiérarchique.
        Entièrement générique - utilise la hiérarchie ontologique.

        Args:
            concept_uri: URI du concept

        Returns:
            Nom de catégorie basé sur la hiérarchie
        """
        try:
            # Obtenir la chaîne hiérarchique complète
            hierarchy_chain = self.ontology_manager.get_concept_hierarchy_chain(concept_uri)

            if not hierarchy_chain:
                # Fallback : utiliser le label du concept lui-même
                concept = self.ontology_manager.concepts.get(concept_uri)
                if concept and concept.label:
                    return self._clean_category_name(concept.label)
                return "general"

            # Utiliser le concept parent le plus haut comme catégorie
            # ou le concept lui-même s'il n'a pas de parents
            if len(hierarchy_chain) > 1:
                # Prendre le concept parent (plus général)
                category_name = hierarchy_chain[0]
            else:
                # Le concept lui-même
                category_name = hierarchy_chain[0]

            return self._clean_category_name(category_name)

        except Exception as e:
            self.logger.debug(f"Error determining category for {concept_uri}: {e}")
            return "general"

    def _determine_relation_category_from_domain_range(self, relation) -> str:
        """
        Détermine la catégorie d'une relation basée sur son domaine et sa portée.
        Entièrement générique - utilise les métadonnées ontologiques.

        Args:
            relation: Objet Relation

        Returns:
            Nom de catégorie basé sur le domaine/portée
        """
        try:
            # Construire le nom de catégorie à partir du domaine et de la portée
            category_parts = []

            # Ajouter les labels des concepts de domaine
            for domain_concept in relation.domain:
                if domain_concept.label:
                    category_parts.append(domain_concept.label)

            # Ajouter les labels des concepts de portée
            for range_concept in relation.range:
                if range_concept.label:
                    category_parts.append(range_concept.label)

            if category_parts:
                # Créer un nom de catégorie composite
                category_name = "_".join(category_parts[:2])  # Limiter à 2 éléments
                return self._clean_category_name(category_name)

            # Fallback : utiliser le label de la relation
            if relation.label:
                return self._clean_category_name(relation.label)

            return "general_relations"

        except Exception as e:
            self.logger.debug(f"Error determining relation category: {e}")
            return "general_relations"

    def _generate_patterns_from_concept_semantics(self, semantics: Dict[str, Any]) -> List[str]:
        """
        Génère des patterns regex à partir des informations sémantiques d'un concept.
        Entièrement générique - exploite toutes les métadonnées disponibles.

        Args:
            semantics: Dictionnaire des informations sémantiques

        Returns:
            Liste de patterns regex
        """
        patterns = []

        try:
            # Pattern à partir du label principal
            if semantics.get("label"):
                patterns.extend(self._create_text_patterns(semantics["label"]))

            # Patterns à partir des labels alternatifs
            for alt_label in semantics.get("alt_labels", []):
                patterns.extend(self._create_text_patterns(alt_label))

            # Patterns à partir de la description
            if semantics.get("description"):
                # Extraire des termes clés de la description
                key_terms = self._extract_key_terms_from_text(semantics["description"])
                for term in key_terms:
                    patterns.extend(self._create_text_patterns(term))

            # Patterns à partir des symboles
            for symbol in semantics.get("symbols", []):
                patterns.extend(self._create_text_patterns(symbol))

            # Patterns à partir des labels de parents (contexte hiérarchique)
            for parent_label in semantics.get("parent_labels", []):
                patterns.extend(self._create_text_patterns(parent_label))

        except Exception as e:
            self.logger.debug(f"Error generating patterns from semantics: {e}")

        return list(set(filter(None, patterns)))

    def _generate_patterns_from_relation(self, relation) -> List[str]:
        """
        Génère des patterns à partir d'une relation et ses métadonnées.
        Entièrement générique.

        Args:
            relation: Objet Relation

        Returns:
            Liste de patterns regex
        """
        patterns = []

        try:
            # Pattern à partir du label de la relation
            if relation.label:
                patterns.extend(self._create_text_patterns(relation.label))

            # Pattern à partir de la description
            if relation.description:
                key_terms = self._extract_key_terms_from_text(relation.description)
                for term in key_terms:
                    patterns.extend(self._create_text_patterns(term))

            # Patterns à partir des concepts de domaine
            for domain_concept in relation.domain:
                if domain_concept.label:
                    patterns.extend(self._create_text_patterns(domain_concept.label))

            # Patterns à partir des concepts de portée
            for range_concept in relation.range:
                if range_concept.label:
                    patterns.extend(self._create_text_patterns(range_concept.label))

        except Exception as e:
            self.logger.debug(f"Error generating patterns from relation: {e}")

        return list(set(filter(None, patterns)))

    def _create_text_patterns(self, text: str) -> List[str]:
        """
        Crée des patterns regex à partir d'un texte de manière générique.

        Args:
            text: Texte source

        Returns:
            Liste de patterns regex
        """
        if not text or len(text.strip()) < 2:
            return []

        patterns = []

        try:
            # Nettoyer le texte
            clean_text = text.strip().lower()

            # Pattern pour le texte complet
            full_pattern = self._text_to_regex_pattern(clean_text)
            if full_pattern:
                patterns.append(full_pattern)

            # Patterns pour les mots individuels (si texte multi-mots)
            words = re.findall(r'\b\w{3,}\b', clean_text)  # Mots de 3+ caractères
            for word in words:
                word_pattern = self._text_to_regex_pattern(word)
                if word_pattern:
                    patterns.append(word_pattern)

            # Pattern pour les acronymes potentiels
            acronym = ''.join([c for c in clean_text if c.isupper()])
            if len(acronym) >= 2:
                acronym_pattern = self._text_to_regex_pattern(acronym.lower())
                if acronym_pattern:
                    patterns.append(acronym_pattern)

        except Exception as e:
            self.logger.debug(f"Error creating patterns from text '{text}': {e}")

        return list(set(filter(None, patterns)))

    def _extract_key_terms_from_text(self, text: str, max_terms: int = 5) -> List[str]:
        """
        Extrait les termes clés d'un texte de manière générique.

        Args:
            text: Texte source
            max_terms: Nombre maximum de termes à extraire

        Returns:
            Liste de termes clés
        """
        if not text or len(text.strip()) < 10:
            return []

        try:
            # Nettoyer et extraire les mots significatifs
            clean_text = text.lower().strip()

            # Mots de 4+ caractères (éviter les mots vides génériques)
            significant_words = re.findall(r'\b\w{4,}\b', clean_text)

            # Filtrer les mots très communs (générique)
            common_words = {
                'this', 'that', 'with', 'from', 'they', 'them', 'were', 'been',
                'have', 'will', 'would', 'could', 'should', 'might', 'must',
                'also', 'only', 'just', 'like', 'such', 'than', 'more', 'most',
                'some', 'many', 'much', 'very', 'quite', 'rather', 'pretty'
            }

            filtered_words = [w for w in significant_words if w not in common_words]

            # Retourner les premiers termes uniques
            unique_terms = []
            for word in filtered_words:
                if word not in unique_terms:
                    unique_terms.append(word)
                if len(unique_terms) >= max_terms:
                    break

            return unique_terms

        except Exception as e:
            self.logger.debug(f"Error extracting key terms from text: {e}")
            return []

    def _text_to_regex_pattern(self, text: str) -> Optional[str]:
        """
        Convertit un texte en pattern regex sûr et générique.

        Args:
            text: Texte à convertir

        Returns:
            Pattern regex ou None
        """
        if not text or len(text.strip()) < 2:
            return None

        try:
            # Nettoyer le texte
            clean_text = re.sub(r'[^\w\s-]', '', text.strip().lower())
            if not clean_text:
                return None

            # Remplacer les espaces par des patterns flexibles
            clean_text = re.sub(r'\s+', r'\\s*', clean_text)

            # Remplacer les tirets par des patterns flexibles
            clean_text = re.sub(r'-', r'[\\s\\-_]*', clean_text)

            # Échapper pour regex et créer pattern avec frontières de mots
            escaped_text = re.escape(clean_text)
            escaped_text = escaped_text.replace(r'\\s\*', r'\s*')
            escaped_text = escaped_text.replace(r'\[\\s\\\\\-\_\]\*', r'[\s\-_]*')

            pattern = rf'\b{escaped_text}\b'

            # Tester que le pattern est valide
            re.compile(pattern)

            return pattern

        except Exception as e:
            self.logger.debug(f"Error creating regex pattern from '{text}': {e}")
            return None

    def _clean_category_name(self, name: str) -> str:
        """
        Nettoie un nom de catégorie pour qu'il soit utilisable comme clé.

        Args:
            name: Nom brut

        Returns:
            Nom nettoyé
        """
        if not name:
            return "general"

        try:
            # Nettoyer et normaliser
            clean_name = re.sub(r'[^\w\s-]', '', name.strip().lower())
            clean_name = re.sub(r'\s+', '_', clean_name)
            clean_name = re.sub(r'-+', '_', clean_name)
            clean_name = re.sub(r'_+', '_', clean_name)
            clean_name = clean_name.strip('_')

            # Limiter la longueur
            if len(clean_name) > 30:
                clean_name = clean_name[:30]

            return clean_name if clean_name else "general"

        except Exception:
            return "general"