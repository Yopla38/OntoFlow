"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

import ast
import logging
import re
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict

from pydantic import BaseModel

from agent.src.agent import Agent
from agent.src.types.enums import AgentRole
from agent.src.types.interfaces import LLMProvider, MemoryProvider


class CodeLanguage(Enum):
    PYTHON = "python"
    HTML = "html"
    CSS = "css"
    JAVASCRIPT = "javascript"
    UNKNOWN = "unknown"


def detect_language(code: str) -> CodeLanguage:
    """
    Détecte le langage du code basé sur des patterns spécifiques.
    """
    # Nettoyage du code
    code = code.strip()

    # Patterns caractéristiques
    patterns = {
        CodeLanguage.HTML: (
            r'<!DOCTYPE html>|<html|<body|<div|<script|<style|<head',
            lambda x: bool(re.search(r'<[^>]+>', x))
        ),
        CodeLanguage.CSS: (
            r'{[\s\S]*}',
            lambda x: bool(re.search(r'[a-z-]+\s*:\s*[^;]+;', x))
        ),
        CodeLanguage.JAVASCRIPT: (
            r'function|const|let|var|=>|document\.|window\.',
            lambda x: bool(re.search(r'(const|let|var)\s+\w+\s*=|function\s+\w+\s*\(', x))
        ),
        CodeLanguage.PYTHON: (
            r'def|class|import|from|async|await',
            lambda x: bool(re.search(r'def\s+\w+\s*\(|class\s+\w+:', x))
        )
    }

    for lang, (pattern, validator) in patterns.items():
        if re.search(pattern, code) and validator(code):
            return lang

    return CodeLanguage.UNKNOWN


class MergeStrategy(ABC):
    @abstractmethod
    def merge(self, original_code: str, new_code: str) -> str:
        pass

    @abstractmethod
    def is_complete(self, original_code: str, new_code: str) -> bool:
        pass


class PythonMergeStrategy(MergeStrategy):
    def __init__(self):
        self.ast = ast

    def is_complete(self, original_code: str, new_code: str) -> bool:
        try:
            original_ast = ast.parse(original_code)
            new_ast = ast.parse(new_code)
            return self.is_code_complete(original_ast, new_ast)
        except Exception as e:
            logging.error(f"Erreur lors de la vérification de complétude: {str(e)}")
            return False

    def is_code_complete(self, original_ast: ast.Module, new_ast: ast.Module) -> bool:
        """
        Vérifie la complétude au niveau AST
        """
        orig_struct = self.get_code_structure(original_ast)
        new_struct = self.get_code_structure(new_ast)

        for class_or_top_level, items in orig_struct.items():
            if class_or_top_level == "top_level":
                for func_name in items:
                    if func_name not in new_struct["top_level"]:
                        return False
            else:
                if class_or_top_level not in new_struct:
                    return False
                for method_name in items:
                    if method_name not in new_struct[class_or_top_level]:
                        return False

        return True

    def get_code_structure(self, module_ast: ast.Module) -> dict:
        """
        Construit un dictionnaire décrivant :
          - Les fonctions globales (sous la clé "top_level")
          - Les classes et leurs méthodes
        """
        structure = {"top_level": set()}

        for node in module_ast.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                structure["top_level"].add(node.name)
            elif isinstance(node, ast.ClassDef):
                method_names = set()
                for subnode in node.body:
                    if isinstance(subnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_names.add(subnode.name)
                structure[node.name] = method_names

        return structure

    def merge(self, original_code: str, new_code: str) -> str:
        try:
            # Normalisation
            original_code = self._normalize_indentation(original_code)
            new_code = self._normalize_indentation(new_code)

            # Parsing
            original_ast = ast.parse(original_code)
            new_ast = ast.parse(new_code)

            # Fusion avec vérification de complexité
            merged_ast = self.merge_asts(original_ast, new_ast)
            merged_code = ast.unparse(merged_ast)
            return self._normalize_indentation(merged_code)

        except Exception as e:
            logging.error(f"Erreur lors de la fusion Python: {str(e)}")
            return original_code

    def merge_asts(self, original_ast: ast.Module, patch_ast: ast.Module) -> ast.Module:
        """Fusionne deux ASTs en vérifiant la complexité des méthodes"""
        # Copie de l'AST original
        merged_ast = ast.Module(body=[], type_ignores=[])

        # Map des définitions originales
        original_defs = {}
        for node in original_ast.body:
            if isinstance(node, ast.ClassDef):
                original_defs[node.name] = {
                    'type': 'class',
                    'node': node,
                    'methods': {
                        m.name: m for m in node.body
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                    }
                }
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                original_defs[node.name] = {'type': 'function', 'node': node}

        # Traitement des nouvelles définitions
        for node in patch_ast.body:
            if isinstance(node, ast.ClassDef):
                if node.name in original_defs and original_defs[node.name]['type'] == 'class':
                    # Classe existante
                    original_class = original_defs[node.name]['node']
                    original_methods = original_defs[node.name]['methods']

                    # Copie de la classe
                    new_class = ast.ClassDef(
                        name=original_class.name,
                        bases=original_class.bases,
                        keywords=original_class.keywords,
                        body=[],
                        decorator_list=original_class.decorator_list
                    )

                    # Traitement des méthodes
                    processed_methods = set()

                    # D'abord, traiter les méthodes du patch
                    for method in node.body:
                        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if method.name in original_methods:
                                # Vérifier la complexité avant de remplacer
                                if self.compare_method_complexity(original_methods[method.name], method):
                                    new_class.body.append(method)
                                else:
                                    new_class.body.append(original_methods[method.name])
                            else:
                                new_class.body.append(method)
                            processed_methods.add(method.name)

                    # Ajouter les méthodes originales non modifiées
                    for method_name, method in original_methods.items():
                        if method_name not in processed_methods:
                            new_class.body.append(method)

                    merged_ast.body.append(new_class)
                else:
                    # Nouvelle classe
                    merged_ast.body.append(node)
            else:
                merged_ast.body.append(node)

        # Ajouter les classes/fonctions originales non modifiées
        for name, def_info in original_defs.items():
            if not any(isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                       and n.name == name for n in merged_ast.body):
                merged_ast.body.append(def_info['node'])

        return merged_ast

    def compare_method_complexity(self, original_method: ast.FunctionDef, new_method: ast.FunctionDef) -> bool:
        """Compare la complexité de deux méthodes"""

        def count_significant_nodes(node) -> int:
            count = 0
            for child in ast.walk(node):
                if isinstance(child, (
                        ast.Expr, ast.Assign, ast.AugAssign,
                        ast.Return, ast.If, ast.For, ast.While,
                        ast.Try, ast.Call, ast.Compare
                )):
                    count += 1
            return count

        original_complexity = count_significant_nodes(original_method)
        new_complexity = count_significant_nodes(new_method)
        return new_complexity >= (original_complexity * 0.7)

    def _normalize_indentation(self, code: str) -> str:
        lines = code.splitlines()
        normalized_lines = []
        indent_level = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                normalized_lines.append('')
                continue

            if stripped.startswith('class '):
                indent_level = 0
                normalized_lines.append(stripped)
                if stripped.endswith(':'):
                    indent_level = 1
            elif stripped.startswith('def '):
                normalized_lines.append('    ' * indent_level + stripped)
                if stripped.endswith(':'):
                    indent_level += 1
            else:
                normalized_lines.append('    ' * indent_level + stripped)

        return '\n'.join(normalized_lines)


class HTMLMergeStrategy(MergeStrategy):
    def __init__(self):
        from bs4 import BeautifulSoup
        self.BeautifulSoup = BeautifulSoup

    def _format_html(self, html_str: str) -> str:
        soup = self.BeautifulSoup(html_str, 'html.parser')
        return soup.prettify()

    def merge(self, original_code: str, new_code: str) -> str:
        original_soup = self.BeautifulSoup(original_code, 'html.parser')
        new_soup = self.BeautifulSoup(new_code, 'html.parser')

        # Fusion des balises head
        if new_soup.head:
            for tag in new_soup.head.contents:
                if not any(str(t) == str(tag) for t in original_soup.head.contents):
                    original_soup.head.append(tag)

        # Fusion des balises body
        if new_soup.body:
            for tag in new_soup.body.contents:
                if not any(str(t) == str(tag) for t in original_soup.body.contents):
                    original_soup.body.append(tag)

        return self._format_html(str(original_soup))

    def is_complete(self, original_code: str, new_code: str) -> bool:
        original_soup = self.BeautifulSoup(original_code, 'html.parser')
        new_soup = self.BeautifulSoup(new_code, 'html.parser')

        return bool(new_soup.html and new_soup.head and new_soup.body)


class CSSMergeStrategy(MergeStrategy):
    def _format_css(self, rules: dict) -> str:
        formatted_rules = []
        for selector, properties in rules.items():
            formatted_properties = '\n    '.join(
                prop.strip() for prop in properties.split(';') if prop.strip()
            )
            formatted_rules.append(f"{selector} {{\n    {formatted_properties}\n}}")
        return '\n\n'.join(formatted_rules)

    def merge(self, original_code: str, new_code: str) -> str:
        original_rules = self._parse_css_rules(original_code)
        new_rules = self._parse_css_rules(new_code)
        merged_rules = {**original_rules, **new_rules}
        return self._format_css(merged_rules)

    def is_complete(self, original_code: str, new_code: str) -> bool:
        original_rules = self._parse_css_rules(original_code)
        new_rules = self._parse_css_rules(new_code)
        return len(new_rules) >= len(original_rules) * 0.7

    def _parse_css_rules(self, css: str) -> dict:
        rules = {}
        # Utilise regex pour extraire les sélecteurs et leurs propriétés
        pattern = r'([^{]+){([^}]+)}'
        matches = re.finditer(pattern, css)
        for match in matches:
            selector = match.group(1).strip()
            properties = match.group(2).strip()
            rules[selector] = properties
        return rules

    def _build_css(self, rules: dict) -> str:
        return '\n'.join(f'{selector} {{\n    {properties}\n}}'
                         for selector, properties in rules.items())


class JavaScriptMergeStrategy(MergeStrategy):
    def __init__(self):
        import esprima
        import escodegen  # Pour la génération de code
        self.esprima = esprima
        self.escodegen = escodegen

    def merge(self, original_code: str, new_code: str) -> str:
        try:
            # Parse les deux codes
            original_ast = self.esprima.parseScript(original_code)
            new_ast = self.esprima.parseScript(new_code)

            # Extrait les déclarations des deux codes
            original_declarations = self._get_declarations(original_ast)
            new_declarations = self._get_declarations(new_ast)

            # Fusionne les déclarations
            merged_declarations = self._merge_declarations(original_declarations, new_declarations)

            # Reconstruit le code
            return self._build_js_code(merged_declarations)

        except Exception as e:
            logging.error(f"Erreur lors de la fusion JavaScript: {str(e)}")
            return original_code

    def is_complete(self, original_code: str, new_code: str) -> bool:
        try:
            original_ast = self.esprima.parseScript(original_code)
            new_ast = self.esprima.parseScript(new_code)

            original_functions = self._get_function_names(original_ast)
            new_functions = self._get_function_names(new_ast)

            return all(func in new_functions for func in original_functions)
        except:
            return False

    def _get_declarations(self, ast) -> dict:
        """Extrait les déclarations de fonctions et variables"""
        declarations = {
            'functions': {},
            'variables': {}
        }

        for node in ast.body:
            # Pour les fonctions
            if node.type == 'FunctionDeclaration':
                declarations['functions'][node.id.name] = node
            # Pour les variables
            elif node.type in ['VariableDeclaration']:
                for declarator in node.declarations:
                    if hasattr(declarator.id, 'name'):
                        declarations['variables'][declarator.id.name] = node

        return declarations

    def _merge_declarations(self, original_decls: dict, new_decls: dict) -> dict:
        """Fusionne les déclarations en privilégiant les nouvelles versions"""
        merged = {
            'functions': {},
            'variables': {}
        }

        # Copie d'abord toutes les déclarations originales
        merged['functions'].update(original_decls['functions'])
        merged['variables'].update(original_decls['variables'])

        # Remplace ou ajoute les nouvelles déclarations
        merged['functions'].update(new_decls['functions'])
        merged['variables'].update(new_decls['variables'])

        return merged

    def _build_js_code(self, declarations: dict) -> str:
        """Reconstruit le code JavaScript à partir des déclarations"""
        # Crée un nouveau AST avec toutes les déclarations
        new_ast = {
            'type': 'Program',
            'body': []
        }

        # Ajoute les fonctions
        for func_node in declarations['functions'].values():
            new_ast['body'].append(func_node)

        # Ajoute les variables
        for var_node in declarations['variables'].values():
            new_ast['body'].append(var_node)

        # Génère le code
        try:
            return self.escodegen.generate(new_ast)
        except Exception as e:
            logging.error(f"Erreur lors de la génération du code: {str(e)}")
            # Fallback simple
            return self._build_js_code_fallback(declarations)

    def _build_js_code_fallback(self, declarations: dict) -> str:
        """Méthode de fallback pour la génération de code"""
        code_parts = []

        # Génère le code pour les fonctions
        for func_node in declarations['functions'].values():
            try:
                params = ', '.join(p.name for p in func_node.params)
                body = self.escodegen.generate(func_node.body)
                code_parts.append(f"function {func_node.id.name}({params}) {body}")
            except:
                logging.warning(f"Impossible de générer le code pour la fonction {func_node.id.name}")

        # Génère le code pour les variables
        for var_node in declarations['variables'].values():
            try:
                code_parts.append(self.escodegen.generate(var_node))
            except:
                logging.warning(f"Impossible de générer le code pour une variable")

        return '\n\n'.join(code_parts)

    def _get_function_names(self, ast) -> set:
        """Extrait les noms de toutes les fonctions"""
        names = set()
        for node in ast.body:
            if node.type == 'FunctionDeclaration':
                names.add(node.id.name)
        return names


class MergeStrategyFactory:
    @staticmethod
    def get_strategy(language: CodeLanguage) -> MergeStrategy:
        strategies = {
            CodeLanguage.PYTHON: PythonMergeStrategy(),
            CodeLanguage.HTML: HTMLMergeStrategy(),
            CodeLanguage.CSS: CSSMergeStrategy(),
            CodeLanguage.JAVASCRIPT: JavaScriptMergeStrategy(),
        }
        return strategies.get(language, PythonMergeStrategy())  # Python par défaut


def smart_merge_code(original_code: str, new_code: str) -> str:
    """
    Fusionne intelligemment deux morceaux de code en détectant leur langage.
    """
    try:
        # Détecte le langage
        language = detect_language(original_code)

        # Obtient la stratégie appropriée
        strategy = MergeStrategyFactory.get_strategy(language)

        # Vérifie si le nouveau code est complet
        if strategy.is_complete(original_code, new_code):
            return new_code

        # Sinon, effectue la fusion
        return strategy.merge(original_code, new_code)

    except Exception as e:
        logging.error(f"Erreur lors de la fusion : {str(e)}")
        traceback.print_exc()
        # En cas d'erreur, retourne le code original
        return original_code


class DiffAgregateurAgent(Agent):
    """
    Agent spécialisé dans l'agrégation de code.
    """

    def __init__(
        self,
        llm_provider: "LLMProvider",
        memory_provider: "MemoryProvider",
        memory_size: int = 1000
    ):
        # Vous pouvez définir ici un format de réponse JSON si nécessaire
        self.response_format: Optional[str] = None
        super().__init__(
            name="agregateur",
            role=AgentRole.EXECUTOR,
            llm_provider=llm_provider,
            memory_provider=memory_provider,
            system_prompt="",
            memory_size=memory_size
        )
        # Peut-être un pydantic_model si vous utilisez generate_response(...)
        self.pydantic_model: Optional[BaseModel] = None

    async def aggregate_code(self, original_code: str, file_data: Dict) -> str:
        """Version améliorée pour gérer les modifications ciblées"""
        # Pour un fichier complet (création)
        if file_data.get("modification_type") == "create":
            return file_data["code_field"]["content"]

        # Pour les modifications de fichiers existants
        if file_data.get("modification_type") == "update" and "code_changes" in file_data:
            result_code = original_code

            for change in file_data["code_changes"]:
                operation = change["operation"]
                context_before = change["location"]["context_before"]
                match_code = change["location"]["match_code"]
                context_after = change["location"]["context_after"]
                new_code = change["new_code"]

                # Trouver l'emplacement de la modification
                location = self._find_location(
                    result_code, context_before, match_code, context_after
                )

                if location:
                    start_idx, end_idx = location

                    if operation == "add":
                        # Ajouter du nouveau code
                        result_code = result_code[:start_idx] + new_code + result_code[start_idx:]

                    elif operation == "modify":
                        # Remplacer du code existant
                        result_code = result_code[:start_idx] + new_code + result_code[end_idx:]

                    elif operation == "delete":
                        # Supprimer du code
                        result_code = result_code[:start_idx] + result_code[end_idx:]

            return result_code

        # Fallback: utiliser la fusion intelligente existante
        return await smart_merge_code(original_code, file_data["code_field"]["content"])

    def _find_location(self, code, context_before, match_code, context_after):
        """Localise précisément où appliquer une modification"""
        # Cas d'ajout (pas de code à matcher)
        if not match_code:
            # Localiser la transition entre context_before et context_after
            search_pattern = re.escape(context_before) + r'(.*?)' + re.escape(context_after)
            match = re.search(search_pattern, code, re.DOTALL)
            if match:
                start_pos = match.start() + len(context_before)
                return (start_pos, start_pos)
        else:
            # Pour modification ou suppression
            # Construire un pattern avec le contexte et le code à matcher
            search_pattern = re.escape(context_before) + re.escape(match_code) + re.escape(context_after)
            match = re.search(search_pattern, code, re.DOTALL)

            if match:
                # Calculer la position de début et fin du code à matcher
                start_pos = match.start() + len(context_before)
                end_pos = start_pos + len(match_code)
                return (start_pos, end_pos)

            # Si le pattern complet ne correspond pas, essayer juste avec match_code
            # et vérifier le contexte approximativement
            match = re.search(re.escape(match_code), code, re.DOTALL)
            if match:
                start_pos = match.start()
                end_pos = match.end()

                # Vérifier que les contextes correspondent approximativement
                before_matches = context_before.strip() in code[max(0, start_pos - 200):start_pos].strip()
                after_matches = context_after.strip() in code[end_pos:min(len(code), end_pos + 200)].strip()

                if before_matches and after_matches:
                    return (start_pos, end_pos)

        return None

    @staticmethod
    async def old_aggregate_code(original_code: str, new_code: str) -> str:
        """
        Agrège deux morceaux de code.
        Ici, on utilise merge_code (implémenté localement) pour faire la fusion.
        """
        try:
            merged_code = smart_merge_code(original_code, new_code)
            return merged_code

        except Exception as e:
            # Gestion d’erreur minimale
            traceback.print_exc()
            raise e




if __name__ == "__main__":
    dico = """{
  "file_exists": true,
  "modification_type": "update",
  "code_changes": [
    {
      "operation": "add",
      "location": {
        "context_before": "import os\nimport openpyxl\nimport shutil\nfrom datetime import datetime\nfrom openpyxl.styles import Font, Alignment, PatternFill, Border, Side",
        "match_code": "",
        "context_after": "from openpyxl.utils import get_column_letter\nfrom openpyxl.chart import BarChart, LineChart, PieChart, Reference\nimport re"
      },
      "new_code": "from openpyxl.drawing.image import Image\nfrom PIL import Image as PILImage\nimport io"
    },
    {
      "operation": "add",
      "location": {
        "context_before": "if __name__ == '__main__':\n    # Exemple d'utilisation simple\n    processor = ProcessExcel('exemple.xlsx', './download')\n    processor.create_table('Produit,Quantité,Prix')\n    processor.add_row('Ordinateur,5,1200')",
        "match_code": "",
        "context_after": "    processor.add_row('Téléphone,10,800')\n    processor.add_row('Tablette,7,500')\n    processor.create_formula('D2', 'B3*C3')\n    html_path = processor.export_to_html()\n    print(f\"Fichier HTML généré : {html_path}\")"
      },
      "new_code": ""
    },
    {
      "operation": "add",
      "location": {
        "context_before": "        except Exception as e:\n            return f\"Erreur lors de la création du tableau croisé: {str(e)}\"\n\n",
        "match_code": "",
        "context_after": "if __name__ == '__main__':\n    # Exemple d'utilisation simple\n    processor = ProcessExcel('exemple.xlsx', './download')\n    processor.create_table('Produit,Quantité,Prix')"
      },
      "new_code": "    def insert_image(self, image_path: str, cell_reference: str, 
                   width: int = None, height: int = None, 
                   keep_aspect_ratio: bool = True, 
                   sheet_name: str = None) -> str:\n        \"\"\"Insère une image dans une feuille Excel à une position spécifiée.\n\n        Args:\n            image_path (str): Chemin de l'image à insérer\n            cell_reference (str): Référence de la cellule où insérer l'image (ex: \"A1\")\n            width (int, optional): Largeur souhaitée en pixels. Si None, utilise la largeur originale.\n            height (int, optional): Hauteur souhaitée en pixels. Si None, utilise la hauteur originale.\n            keep_aspect_ratio (bool): Conserver le ratio d'aspect si seulement width ou height est spécifié\n            sheet_name (str, optional): Nom de la feuille. Si None, utilise la feuille active.\n\n        Returns:\n            str: Chemin du fichier Excel mis à jour\n        \"\"\"\n        if not self.working_path or not self.workbook:\n            return \"Erreur: Aucun fichier Excel valide.\"\n\n        try:\n            # Utiliser la feuille active si aucune n'est spécifiée\n            if sheet_name is None:\n                sheet_name = self.active_sheet\n\n            # Vérifier si la feuille existe\n            if sheet_name not in self.workbook.sheetnames:\n                return f\"Erreur: La feuille '{sheet_name}' n'existe pas.\"\n\n            sheet = self.workbook[sheet_name]\n\n            # Vérifier si le fichier image existe\n            if not os.path.exists(image_path):\n                return f\"Erreur: L'image '{image_path}' n'existe pas.\"\n\n            # Ouvrir l'image avec PIL pour obtenir ses dimensions et éventuellement la redimensionner\n            try:\n                pil_image = PILImage.open(image_path)\n                img_format = pil_image.format\n                original_width, original_height = pil_image.size\n\n                # Calculer les dimensions finales en tenant compte du ratio d'aspect\n                final_width = width if width is not None else original_width\n                final_height = height if height is not None else original_height\n\n                # Ajuster les dimensions pour conserver le ratio d'aspect si demandé\n                if keep_aspect_ratio:\n                    if width is not None and height is None:\n                        # Calculer la hauteur proportionnelle\n                        final_height = int(original_height * (final_width / original_width))\n                    elif height is not None and width is None:\n                        # Calculer la largeur proportionnelle\n                        final_width = int(original_width * (final_height / original_height))\n\n                # Redimensionner l'image si nécessaire\n                if final_width != original_width or final_height != original_height:\n                    pil_image = pil_image.resize((final_width, final_height))\n\n                # Convertir l'image au format PNG si nécessaire\n                # openpyxl supporte mieux le format PNG\n                if img_format != 'PNG':\n                    img_buffer = io.BytesIO()\n                    pil_image.save(img_buffer, format='PNG')\n                    img_buffer.seek(0)\n                    img = Image(img_buffer)\n                else:\n                    img = Image(image_path)\n                    \n                # Ajuster les dimensions pour openpyxl\n                img.width = final_width\n                img.height = final_height\n\n                # Insérer l'image dans la cellule spécifiée\n                sheet.add_image(img, cell_reference)\n                \n                return self._save_workbook()\n                \n            except Exception as img_error:\n                return f\"Erreur lors du traitement de l'image: {str(img_error)}\"\n                \n        except Exception as e:\n            return f\"Erreur lors de l'insertion de l'image: {str(e)}\"\n"
    }
  ]
}"""
