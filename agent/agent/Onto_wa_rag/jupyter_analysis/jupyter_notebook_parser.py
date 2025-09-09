# jupyter_notebook_parser.py
"""
------------------------------------------
Copyright: CEA Grenoble
Auteur: Yoann CURE (Inspiré par), Gemini (Implémentation)
Entité: IRIG
Année: 2025
Description: Parseur de qualité production pour notebooks Jupyter (.ipynb)
             destiné à alimenter un système RAG.
------------------------------------------
"""

import json
import ast
import os
import logging
from typing import List, Dict, Any, Tuple, Optional

from ..fortran_analysis.core.entity_manager import UnifiedEntity


logger = logging.getLogger(__name__)


class CodeVisitor(ast.NodeVisitor):
    """
    Un NodeVisitor pour parcourir l'AST d'une cellule de code et extraire
    les fonctions, classes, imports et appels.
    Cette classe est conçue pour être instanciée pour chaque portée d'analyse
    (cellule complète, corps de fonction, etc.).
    """

    def __init__(self, filepath: str, parent_entity_name: str, source_code: str):
        self.entities: List[UnifiedEntity] = []
        self.dependencies: List[Dict[str, Any]] = []
        self.calls: List[Dict[str, Any]] = []
        self.filepath = filepath
        self.parent_stack = [parent_entity_name]
        self.source_lines = source_code.splitlines()

    def _get_full_source_segment(self, node: ast.AST) -> str:
        """Extrait le code source complet d'un nœud, y compris les décorateurs."""
        if hasattr(node, 'decorator_list') and node.decorator_list:
            first_node = node.decorator_list[0]
        else:
            first_node = node

        # ast.get_source_segment est la méthode la plus fiable
        try:
            return ast.get_source_segment(
                '\n'.join(self.source_lines),
                node,
                padded=True
            )
        except (TypeError, IndexError):
            # Fallback pour les cas où get_source_segment échoue
            start = first_node.lineno - 1
            end = node.end_lineno
            return '\n'.join(self.source_lines[start:end])

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """
        Extrait le nom qualifié complet d'un appel de fonction de manière récursive.
        Exemples: 'print', 'np.array', 'pd.DataFrame.from_dict'
        """
        func = node.func
        parts = []
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value
        if isinstance(func, ast.Name):
            parts.append(func.id)
            return ".".join(reversed(parts))
        # Gère les appels chaînés comme plt.figure().add_subplot()
        if isinstance(func, ast.Call):
            return f"{self._get_call_name(func)}()"
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._process_function(node)

    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent = self.parent_stack[-1]
        full_source = self._get_full_source_segment(node)
        signature = full_source.split('\n')[0].strip()

        # Analyser le corps de la fonction pour trouver ses propres appels
        body_visitor = CodeVisitor(self.filepath, node.name, full_source)
        for sub_node in node.body:
            body_visitor.visit(sub_node)

        function_calls = body_visitor.calls

        entity = UnifiedEntity(
            entity_name=node.name,
            entity_type='async function' if isinstance(node, ast.AsyncFunctionDef) else 'function',
            start_line=node.lineno,
            end_line=node.end_lineno,
            filepath=self.filepath,
            parent_entity=parent,
            signature=signature,
            called_functions=function_calls,
            source_method='jupyter_ast',
            confidence=1.0
        )
        self.entities.append(entity)

        # Pas de self.generic_visit(node) ici pour éviter de traiter
        # les appels et imports à l'intérieur de la fonction au niveau supérieur.

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        parent = self.parent_stack[-1]
        full_source = self._get_full_source_segment(node)
        signature = full_source.split('\n')[0].strip()

        entity = UnifiedEntity(
            entity_name=node.name,
            entity_type='class',
            start_line=node.lineno,
            end_line=node.end_lineno,
            filepath=self.filepath,
            parent_entity=parent,
            signature=signature,
            source_method='jupyter_ast',
            confidence=1.0
        )
        self.entities.append(entity)

        # Visiter les enfants (méthodes, etc.) avec le nom de la classe comme parent
        self.parent_stack.append(node.name)
        self.generic_visit(node)
        self.parent_stack.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            dep_name = alias.asname or alias.name
            self.dependencies.append({'name': dep_name, 'line': node.lineno})

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or '.'
        for alias in node.names:
            full_name = f"{module}.{alias.name}"
            self.dependencies.append({'name': full_name, 'line': node.lineno})

    def visit_Call(self, node: ast.Call) -> None:
        call_name = self._get_call_name(node)
        if call_name:
            self.calls.append({'name': call_name, 'line': node.lineno})
        self.generic_visit(node)  # Continuer à visiter les arguments de l'appel


class JupyterNotebookParser:
    """
    Parseur pour notebooks Jupyter qui identifie le résumé du notebook
    et l'injecte directement dans chaque entité créée.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse_file(self, filepath: str) -> Tuple[List[UnifiedEntity], Dict[str, Any]]:
        """
        Analyse un notebook en deux passes internes :
        1. Parse toutes les cellules et identifie le résumé.
        2. Injecte le résumé dans toutes les entités du notebook.
        """
        self.logger.info(f"🚀 Démarrage de l'analyse du notebook : {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                notebook_content = json.load(f)
        except Exception as e:
            self.logger.error(f"Erreur lors de la lecture/parsing de {filepath}: {e}")
            return [], {}

        # === PASSE 1 : Analyse et Identification du Résumé ===
        all_entities = []
        summary_text_parts = []
        is_in_summary_section = True

        filename = os.path.basename(filepath)
        notebook_name = os.path.splitext(filename)[0]
        global_line_counter = 1

        # Création de l'entité notebook (sera enrichie plus tard)
        notebook_entity = UnifiedEntity(
            entity_name=notebook_name,
            entity_type="notebook",
            start_line=1,
            end_line=sum(len(c.get('source', [])) for c in notebook_content.get('cells', [])),
            filepath=filepath,
            filename=filename,
            source_code=json.dumps(notebook_content, indent=2)
        )
        all_entities.append(notebook_entity)

        for i, cell_data in enumerate(notebook_content.get('cells', [])):
            cell_entities, lines_in_cell = self._parse_cell(
                cell_data, i + 1, filepath, notebook_name, global_line_counter
            )

            if cell_entities:
                cell_root_entity = cell_entities[0]  # La première entité est la cellule elle-même

                if is_in_summary_section:
                    if cell_root_entity.entity_type == 'markdown_cell':
                        # Cette cellule fait partie du résumé
                        cell_root_entity.entity_role = 'summary'
                        summary_text_parts.append(cell_root_entity.source_code)
                    elif cell_root_entity.entity_type == 'code_cell':
                        # Fin de la section résumé
                        is_in_summary_section = False

                all_entities.extend(cell_entities)

            global_line_counter += lines_in_cell

        # === PASSE 2 : Enrichissement des Entités ===
        full_summary_text = "\n\n".join(summary_text_parts).strip()

        for entity in all_entities:
            entity.notebook_summary = full_summary_text

        # Liaison sémantique finale
        all_entities = self._link_markdown_to_code(all_entities)

        self.logger.info(f"✅ Analyse de {filename} terminée. {len(all_entities)} entités enrichies trouvées.")
        return all_entities, notebook_content

    def _parse_cell(self, cell_data: Dict, cell_number: int, filepath: str, parent_name: str, start_line_abs: int) -> \
    Tuple[List[UnifiedEntity], int]:
        cell_type = cell_data.get('cell_type')
        source_lines = cell_data.get('source', [])
        source_code = "".join(source_lines)
        num_lines = len(source_lines)

        if not source_code.strip():
            return [], num_lines

        cell_entity_name = f"{parent_name}_cell_{cell_number}"

        cell_entity = UnifiedEntity(
            entity_name=cell_entity_name,
            entity_type=f"{cell_type}_cell",
            start_line=start_line_abs,
            end_line=start_line_abs + num_lines - 1 if num_lines > 0 else start_line_abs,
            filepath=filepath,
            filename=os.path.basename(filepath),
            parent_entity=parent_name,
            source_code=source_code
        )

        entities = [cell_entity]

        if cell_type == 'code':
            try:
                tree = ast.parse(source_code)
                visitor = CodeVisitorWithOffset(filepath, cell_entity_name, source_code, start_line_abs)
                visitor.visit(tree)

                entities.extend(visitor.entities)
                cell_entity.dependencies = visitor.dependencies
                cell_entity.called_functions = visitor.calls
            except SyntaxError as e:
                self.logger.warning(
                    f"Erreur de syntaxe dans la cellule {cell_number} de {filepath} (ligne {e.lineno}). Analyse AST de la cellule ignorée.")

        return entities, num_lines

    def _link_markdown_to_code(self, entities: List[UnifiedEntity]) -> List[UnifiedEntity]:
        for i, entity in enumerate(entities):
            if entity.entity_type == 'markdown_cell' and entity.entity_role != 'summary':
                # On cherche la prochaine cellule de code
                for next_entity in entities[i + 1:]:
                    if next_entity.entity_type == 'code_cell':
                        entity.signature = f"Documentation for: {next_entity.entity_name}"
                        entity.entity_role = 'documentation'
                        break
        return entities


# visiteur qui gère l'offset de ligne et le code source
class CodeVisitorWithOffset(ast.NodeVisitor):
    def __init__(self, filepath: str, parent_entity_name: str, source_code: str, line_offset: int):
        self.entities: List[UnifiedEntity] = []
        self.dependencies: List[Dict[str, Any]] = []
        self.calls: List[Dict[str, Any]] = []
        self.filepath = filepath
        self.parent_stack = [parent_entity_name]
        self.source_code = source_code
        self.line_offset = line_offset - 1  # L'offset est 0-indexed (start_line - 1)

    def _get_source_segment(self, node: ast.AST) -> str:
        return ast.get_source_segment(self.source_code, node) or ""

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        # ... (Même logique que le précédent) ...
        func = node.func
        parts = []
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value
        if isinstance(func, ast.Name):
            parts.append(func.id)
            return ".".join(reversed(parts))
        if isinstance(func, ast.Call):
            # Pour les appels chaînés, on reconstruit
            base = self._get_call_name(func)
            if base:
                return f"{base}()"  # Représentation simplifiée
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        parent = self.parent_stack[-1]
        signature = self._get_source_segment(node).split('\n')[0].strip()

        entity = UnifiedEntity(
            entity_name=node.name, entity_type='function',
            start_line=node.lineno + self.line_offset,
            end_line=node.end_lineno + self.line_offset,
            filepath=self.filepath, parent_entity=parent,
            signature=signature, source_code=self._get_source_segment(node)
        )
        # Visiter le corps pour trouver les appels INTERNES à la fonction
        body_visitor = CodeVisitorWithOffset(self.filepath, node.name, self.source_code, self.line_offset)
        for sub_node in node.body:
            body_visitor.visit(sub_node)
        entity.called_functions = body_visitor.calls

        self.entities.append(entity)
        # NE PAS appeler generic_visit pour éviter de compter les appels internes au niveau de la cellule

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        parent = self.parent_stack[-1]
        signature = self._get_source_segment(node).split('\n')[0].strip()

        entity = UnifiedEntity(
            entity_name=node.name, entity_type='class',
            start_line=node.lineno + self.line_offset,
            end_line=node.end_lineno + self.line_offset,
            filepath=self.filepath, parent_entity=parent,
            signature=signature, source_code=self._get_source_segment(node)
        )
        self.entities.append(entity)

        self.parent_stack.append(node.name)
        # On visite l'intérieur de la classe
        for sub_node in node.body:
            self.visit(sub_node)
        self.parent_stack.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.dependencies.append({'name': alias.name, 'line': node.lineno + self.line_offset})

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or '.'
        for alias in node.names:
            self.dependencies.append({'name': f"{module}.{alias.name}", 'line': node.lineno + self.line_offset})

    def visit_Call(self, node: ast.Call) -> None:
        call_name = self._get_call_name(node)
        if call_name:
            self.calls.append({'name': call_name, 'line': node.lineno + self.line_offset})

        # MODIFICATION pour la précision des appels: ne pas visiter la fonction elle-même
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)


class JupyterAnalysisEngine:
    """
    Interface unifiée pour l'analyse de notebooks Jupyter.
    Gère le parsing et un cache simple pour la performance.
    """

    def __init__(self):
        self.parser = JupyterNotebookParser()
        self._cache = {}

    def get_entities(self, filepath: str, use_cache: bool = True) -> List[UnifiedEntity]:
        """
        Récupère toutes les entités d'un notebook avec gestion du cache.
        """
        if use_cache and filepath in self._cache:
            return self._cache[filepath]

        entities, _ = self.parser.parse_file(filepath)

        if use_cache:
            self._cache[filepath] = entities

        return entities

    def analyze_file(self, filepath: str) -> Tuple[List[UnifiedEntity], Dict[str, Any]]:
        """
        Analyse un fichier notebook et retourne les entités et le contenu brut.
        Cette méthode ne met pas en cache le contenu brut pour économiser la mémoire.
        """
        self.clear_cache()  # S'assurer que les données sont fraîches
        return self.parser.parse_file(filepath)

    def clear_cache(self):
        """Vide le cache d'analyse."""
        self._cache.clear()


# Fonction factory pour la cohérence avec votre projet existant
def get_jupyter_analyzer() -> JupyterAnalysisEngine:
    """
    Factory pour créer un analyseur de notebooks Jupyter.
    C'est le point d'entrée principal pour le système.
    """
    return JupyterAnalysisEngine()


if __name__ == "__main__":


    # Configurez un logger basique pour voir les messages
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    # Créez un notebook de test nommé 'test_notebook.ipynb'
    # avec des cellules de code (fonctions, classes, imports) et de markdown.

    analyzer = get_jupyter_analyzer()
    filepath = '/home/yopla/PycharmProjects/ci-agent/agent/Onto_wa_rag/jupyter_analysis/test.ipynb'

    # Obtenir toutes les entités structurées
    entities, raw_content = analyzer.analyze_file(filepath)

    print(f"\n--- Analyse de '{filepath}' ---")
    print(f"Nombre total d'entités trouvées : {len(entities)}")

    for entity in entities:
        print("\n------------------------------")
        print(f"Nom      : {entity.entity_name}")
        print(f"Type     : {entity.entity_type}")
        print(f"Parent   : {entity.parent_entity}")
        print(f"Lignes   : {entity.start_line} à {entity.end_line}")
        if entity.signature:
            print(f"Signature: {entity.signature}")
        if entity.dependencies:
            deps = [d['name'] for d in entity.dependencies]
            print(f"Dépendances: {deps}")
        if entity.called_functions:
            calls = [c['name'] for c in entity.called_functions]
            print(f"Appels   : {calls}")
        if entity.source_code:
            print(f"Source   : {entity.source_code}")