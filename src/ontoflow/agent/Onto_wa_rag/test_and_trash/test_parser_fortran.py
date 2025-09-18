"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

import asyncio
import os
import logging
import re
import sys
from collections import defaultdict

# --- Configuration initiale pour les imports et le logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Maquillage (Mocking) des dépendances externes ---

class MockCache:
    """Un cache factice qui ne fait rien."""

    def __init__(self):
        self._cache = {}

    async def get(self, key):
        return self._cache.get(key)

    async def set(self, key, value, ttl=None):
        self._cache[key] = value

    async def delete(self, key):
        self._cache.pop(key, None)


class MockGlobalCache:
    """Structure factice pour le global_cache."""
    function_calls = MockCache()
    dependency_graphs = MockCache()


class MockDocumentStore:
    """Un 'document store' factice qui stocke les chunks en mémoire."""

    def __init__(self):
        self._chunks_by_doc = defaultdict(list)
        # On a besoin d'un index par ID pour la correction
        self._chunks_by_id = {}

    async def add_entity_as_chunk(self, entity):
        """Simule le découpage d'une entité en un chunk et son stockage."""
        doc_id = entity.filepath
        chunk_id = f"{doc_id}-chunk-{len(self._chunks_by_id)}"

        chunk = {
            'id': chunk_id,
            'entity_info': entity.to_dict()
        }

        self._chunks_by_doc[doc_id].append(chunk)
        self._chunks_by_id[chunk_id] = chunk

    async def get_all_documents(self):
        return list(self._chunks_by_doc.keys())

    async def get_chunks_by_document(self, doc_id: str):
        return self._chunks_by_doc.get(doc_id, [])


class MockChunkAccessManager:
    """Un gestionnaire d'accès aux chunks factice."""

    def __init__(self, document_store):
        self.document_store = document_store

    async def get_chunks_by_document(self, doc_id: str):
        return await self.document_store.get_chunks_by_document(doc_id)

    # ==================== CORRECTION ICI ====================
    async def get_entity_info_from_chunk(self, chunk_id: str):
        """
        Récupère les métadonnées de l'entité à partir d'un chunk ID.
        Dans notre simulation, cela signifie chercher le chunk dans le store par son ID.
        """
        # Chercher le chunk complet par son ID dans le store factice
        full_chunk_object = self.document_store._chunks_by_id.get(chunk_id)

        if full_chunk_object:
            # Retourner les métadonnées de l'entité contenues dans ce chunk
            return full_chunk_object.get('entity_info')

        return None
    # =======================================================


# --- Injection des Mocks ---
from fortran_analysis.core import entity_manager

entity_manager.global_cache = MockGlobalCache()

# --- Imports des classes principales ---
from fortran_analysis.core.hybrid_fortran_parser import FortranAnalysisEngine
from fortran_analysis.core.entity_manager import EntityManager

# --- Code Fortran d'exemple (identique) ---
FORTRAN_CODE = """
! test_code.f90
PROGRAM main_test
  USE my_module
  IMPLICIT NONE

  REAL :: x, y, z
  INTEGER, PARAMETER :: max_iterations = 100
  TYPE(my_type) :: data_point

  x = 10.0
  y = custom_function(x, max_iterations)

  CALL calculate_stuff(x, y, z)

  data_point%value = z
  PRINT *, "Final result:", data_point%value

CONTAINS

  SUBROUTINE external_routine(a)
    REAL, INTENT(IN) :: a
    PRINT *, "External routine called with", a
  END SUBROUTINE external_routine

END PROGRAM main_test

MODULE my_module
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: calculate_stuff, custom_function, my_type

  TYPE :: my_type
    REAL :: value
  END TYPE my_type

  INTERFACE
     FUNCTION external_sin(x)
       REAL, INTENT(IN) :: x
       REAL :: external_sin
     END FUNCTION external_sin
  END INTERFACE

CONTAINS

  SUBROUTINE calculate_stuff(in_val, factor, out_val)
    USE iso_c_binding, only: c_sqrt
    REAL, INTENT(IN) :: in_val, factor
    REAL, INTENT(OUT) :: out_val
    REAL :: intermediate

    intermediate = sin(in_val) + cos(factor)
    out_val = c_sqrt(intermediate**2)

    CALL external_routine(out_val)

  END SUBROUTINE calculate_stuff

  FUNCTION custom_function(val1, iterations)
    REAL, INTENT(IN) :: val1
    INTEGER, INTENT(IN) :: iterations
    REAL :: custom_function
    INTEGER :: i

    custom_function = 0.0
    DO i = 1, iterations
      custom_function = custom_function + external_sin(val1 / i)
    END DO

  END FUNCTION custom_function

END MODULE my_module
"""


def preprocess_fortran_includes(filepath: str, processed_files=None) -> str:
    """
    Lit un fichier Fortran et remplace récursivement les directives 'include'
    par le contenu des fichiers correspondants.
    """
    if processed_files is None:
        processed_files = set()

    # Éviter les inclusions circulaires infinies
    if filepath in processed_files:
        return ""
    processed_files.add(filepath)

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.warning(f"Fichier include non trouvé : {filepath}")
        return ""

    base_dir = os.path.dirname(filepath)
    output_code = ""

    # Expression régulière pour trouver `include 'filename'` (insensible à la casse)
    include_pattern = re.compile(r"^\s*include\s+['\"]([^'\"]+)['\"]", re.IGNORECASE)

    for line in lines:
        match = include_pattern.match(line.strip())
        if match:
            include_filename = match.group(1)
            include_filepath = os.path.join(base_dir, include_filename)
            logging.info(f"Traitement de l'include : {include_filepath}")
            # Appel récursif pour traiter les includes dans le fichier inclus
            output_code += preprocess_fortran_includes(include_filepath, processed_files)
        else:
            output_code += line

    return output_code


async def main(fortran_filepath: str):
    """
    Fonction principale pour tester le flux complet sur un fichier réel.
    """
    if not os.path.exists(fortran_filepath):
        print(f"ERREUR: Le fichier '{fortran_filepath}' n'existe pas.")
        return

    # --- 0. PRÉ-TRAITEMENT DU FICHIER SOURCE ---
    print(f"--- 0. Pré-traitement du fichier '{fortran_filepath}' pour gérer les 'include' ---")
    full_fortran_code = preprocess_fortran_includes(fortran_filepath)
    if not full_fortran_code:
        print("ERREUR: Impossible de lire ou de traiter le fichier source.")
        return
    print("✅ Pré-traitement terminé.\n")

    # --- 1. ANALYSE AVEC LE PARSER ---
    print("--- 1. Analyse du code avec HybridFortranParser ---")
    analyzer = FortranAnalysisEngine()
    parsed_entities = analyzer.get_entities(code=full_fortran_code, filename=fortran_filepath)
    print(f"✅ Parser a trouvé {len(parsed_entities)} entités.\n")

    # --- 2. STOCKAGE (SIMULÉ) ---
    print("--- 2. Simulation du stockage des entités dans un DocumentStore ---")
    mock_store = MockDocumentStore()
    for entity in parsed_entities:
        await mock_store.add_entity_as_chunk(entity)
    print(f"✅ {len(parsed_entities)} entités ajoutées au store factice en tant que chunks.\n")

    # --- 3. GESTION AVEC L'ENTITY MANAGER ---
    print("--- 3. Initialisation de l'EntityManager ---")
    entity_mgr = EntityManager(mock_store)
    entity_mgr.chunk_access = MockChunkAccessManager(mock_store)
    await entity_mgr.initialize()
    print(f"✅ EntityManager initialisé. Il gère {len(entity_mgr.entities)} entités.\n")

    # --- 4. INTERROGATION ET AFFICHAGE ---
    print("--- 4. Affichage des routines, fonctions, modules, etc., gérés par l'EntityManager ---")
    all_managed_entities = await entity_mgr.list_entities(limit=1000)

    structural_entities = [
        e for e in all_managed_entities
        if e.entity_type in ['program', 'module', 'subroutine', 'function', 'interface', 'type_definition']
    ]

    sorted_entities = sorted(structural_entities, key=lambda e: (e.start_line, e.entity_name))

    for entity in sorted_entities:
        print("===================================================")
        print(f"  Nom        : {entity.entity_name}")
        print(f"  Type       : {entity.entity_type}")
        print(f"  Fichier    : {entity.filepath}")
        # Note : le parent peut être incorrect car les numéros de ligne sont faussés par l'inclusion
        print(f"  Parent     : {entity.parent_entity or 'Aucun'}")

        if entity.dependencies:
            print(f"  Dépendances: {', '.join(sorted(list(entity.dependencies)))}")

        if entity.called_functions:
            print(f"  Appels     : {', '.join(sorted(list(entity.called_functions)))}")

        print("===================================================\n")

    stats = entity_mgr.get_stats()
    print("--- Statistiques de l'EntityManager ---")
    for key, value in stats.items():
        print(f"  {key:.<25} : {value}")
    print("-------------------------------------\n")


if __name__ == "__main__":
    # Vérifier qu'un nom de fichier est fourni en argument
    if len(sys.argv) < 2:
        print("Usage: python test_parser_fortran.py <chemin_vers_le_fichier_fortran>")
        sys.exit(1)

    filepath = sys.argv[1]
    asyncio.run(main(filepath))

