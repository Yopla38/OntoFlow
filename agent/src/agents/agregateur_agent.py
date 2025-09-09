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
import json
import logging
import os
import re
import subprocess
import tempfile
import traceback
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Union, Any, Literal
from pydantic import BaseModel, Field, ConfigDict

from agent.src.agent import Agent
from agent.src.types.enums import AgentRole
from agent.src.types.interfaces import LLMProvider, MemoryProvider



class DiffContent(BaseModel):
    line: str = Field(
        ...,
        description="Le code à insérer/modifier (sans indentation)"
    )
    indent: int = Field(
        ...,
        description="Nombre d'espaces d'indentation (multiple de 4)",
        ge=0
    )


class DiffOperation(BaseModel):
    operation: Literal['a', 'c', 'd'] = Field(
        ...,
        description="""
        Type d'opération:
        - 'a': ajouter (add) du code
        - 'c': modifier (change) du code existant
        - 'd': supprimer (delete) du code
        """
    )
    source: str = Field(
        ...,
        description="Numéro(s) de ligne dans code 1 (ex: '5' ou '5,7')"
    )
    target: str = Field(
        ...,
        description="Numéro(s) de ligne où appliquer l'opération (ex: '5' ou '5,7')"
    )
    content: List[DiffContent] = Field(
        default_factory=list,
        description="Liste des lignes de code à ajouter/modifier avec leur indentation"
    )


class LLMDiffResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "reasoning": "Ajout d'une nouvelle ligne de print",
                    "diff": [{
                        "operation": "a",
                        "source": "2",
                        "target": "3",
                        "content": [
                            {"line": "print('World')", "indent": 4}
                        ]
                    }]
                },
                {
                    "reasoning": "code 2 complet",
                    "diff": []
                }
            ]
        }
    )

    reasoning: str = Field(
        ...,
        description="Explication des modifications ou 'code 2 complet' si code 2 est autonome"
    )
    diff: List[DiffOperation] = Field(
        default_factory=list,
        description="Liste des opérations de différences à appliquer"
    )


class AiderCodeAggregator:
    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialise l'agrégateur de code avec Aider

        Args:
            model_name: Le modèle à utiliser (gpt-4, claude-3-sonnet, deepseek, etc.)
            api_key: Clé API pour le modèle choisi
        """
        self.model_name = model_name

        if "gpt" in self.model_name.lower() or "o1" in self.model_name.lower():
            os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_SMALL_API_KEY', "")
            self.api_key = os.environ["OPENAI_API_KEY"]
        elif "claude" in self.model_name.lower() or "sonnet" in self.model_name.lower():
            os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_SMALL_API_KEY', "")
            self.api_key = os.environ["ANTHROPIC_API_KEY"]

        self._setup_environment()

    def _setup_environment(self):
        """Configure les variables d'environnement pour les clés API"""
        if self.api_key:
            if "gpt" in self.model_name.lower() or "o1" in self.model_name.lower():
                os.environ["OPENAI_API_KEY"] = self.api_key
            elif "claude" in self.model_name.lower() or "sonnet" in self.model_name.lower():
                os.environ["ANTHROPIC_API_KEY"] = self.api_key
            elif "deepseek" in self.model_name.lower():
                os.environ["DEEPSEEK_API_KEY"] = self.api_key

    async def aggregate_code(self, original_content: str, new_content: str,
                             instruction: Optional[str] = None,
                             file_extension: str = "py") -> str:
        """
        Agrège le code original avec le nouveau contenu en utilisant Aider

        Args:
            original_content: Le contenu de code original
            new_content: Le nouveau contenu à intégrer
            instruction: Instructions spécifiques pour l'agrégation
            file_extension: Extension du fichier (py, js, etc.)

        Returns:
            Le code agrégé résultant
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Créer le fichier principal
            main_file = temp_path / f"main.{file_extension}"
            with open(main_file, 'w', encoding='utf-8') as f:
                f.write(original_content)

            # Créer un fichier avec les instructions et le nouveau contenu
            context_file = temp_path / f"new_content.{file_extension}"
            with open(context_file, 'w', encoding='utf-8') as f:
                f.write(f"# Nouveau contenu à intégrer:\n{new_content}")

            # Préparer l'instruction
            if not instruction:
                instruction = f"""
Intègre le code du fichier new_content.{file_extension} dans main.{file_extension}.
Évite les duplications, résous les conflits potentiels et maintiens la cohérence.
Assure-toi que le code résultant est propre et fonctionnel.
Ignore les commentaires '# Nouveau contenu à intégrer:' du fichier new_content.
"""

            try:
                # Construire la commande Aider
                cmd = [
                    "aider",
                    "--model", self.model_name,
                    "--message", instruction,
                    "--yes",  # Auto-accept changes
                    "--no-git",  # Don't use git in temp directory
                    str(main_file),
                    str(context_file)
                ]

                # Exécuter Aider
                result = await self._run_aider_command(cmd, temp_dir)

                # Lire le résultat
                with open(main_file, 'r', encoding='utf-8') as f:
                    aggregated_content = f.read()

                return aggregated_content

            except Exception as e:
                raise Exception(f"Erreur lors de l'agrégation avec Aider: {str(e)}")

    async def _run_aider_command(self, cmd: list, cwd: str) -> subprocess.CompletedProcess:
        """Exécute une commande Aider de manière asynchrone"""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        # Décoder les bytes en string
        stdout_str = stdout.decode('utf-8') if stdout else ""
        stderr_str = stderr.decode('utf-8') if stderr else ""

        if process.returncode != 0:
            raise Exception(f"Aider a échoué: {stderr_str}")

        return process


class AgregateurAgent(Agent):
    """Agent spécialisé dans l'agrégation de code"""

    def __init__(
            self,
            llm_provider: LLMProvider,
            memory_provider: MemoryProvider,
            memory_size: int = 1000
    ):
        self.response_format = """
        {
          "reasoning": "Explication ou 'code 2 complet'",
          "diff": [
            {
              "operation": "a|c|d",  # add|change|delete
              "source": "n[,m]",     # ligne(s) source (code 1)
              "target": "p[,q]",     # ligne(s) cible (où insérer/modifier)
              "content": [           # lignes de code avec leur indentation
                {"line": "code", "indent": n},
                ...
              ]
            }
          ]
        }"""

        system_prompt = """Vous êtes un expert Python chargé de générer un diff intelligent.

        FORMAT D'ENTRÉE:
        code 1: Code source original (numéroté)
        code 2: Code modifié à intégrer

        SYNTAXE DIFF:
        - na,mb : Lignes source,target
        - a : Ajout (add)
        - c : Modification (change)
        - d : Suppression (delete)

        RÈGLES:
        1. Si code 2 est complet: {"reasoning": "code 2 complet"}
        2. Sinon, pour chaque modification:
           - Identifier le type d'opération (a|c|d)
           - Spécifier les lignes sources et cibles
           - Fournir le contenu avec indentation correcte

        EXEMPLE:
        code 1:
        ```python
        001 def hello():
        002     print("Hello")
        ```

        code 2:
        ```python
        def hello():
            print("Hello")
            print("World")
        ```

        RÉPONSE:
        {
          "reasoning": "Ajout d'une ligne d'affichage",
          "diff": [{
            "operation": "a",
            "source": "2",
            "target": "3",
            "content": [
              {"line": "print(\"World\")", "indent": 4}
            ]
          }]
        }
        """

        super().__init__(
            name="agregateur",
            role=AgentRole.EXECUTOR,
            llm_provider=llm_provider,
            memory_provider=memory_provider,
            system_prompt=system_prompt,
            memory_size=memory_size
        )
        self.pydantic_model = LLMDiffResponse

    async def aggregate_code(self, original_code: str, new_code: str) -> str:
        """
        Agrège deux morceaux de code
        """
        try:
            # Numérotation du code original
            numbered_original = numeroter_lignes(original_code)

            # Préparation du message
            message = (f"""code 1:\n```python\n{numbered_original}\n```\ncode 2:\n```python\n{new_code}\n```\n\n\n""" +
                       f"""Vous devez respecter ce format JSON dict pour la réponse : {self.response_format}\n""")

            # Obtention de la réponse structurée
            message = await self.llm_provider.generate_response(
                message,
                pydantic_model=self.pydantic_model
            )

            reponse = message

            if "reasoning" in reponse and "code 2 complet" in reponse["reasoning"]:
                logging.info(f"Le nouveau code est complet.")
                return new_code
            else:
                logging.info("Le code est partiel, nous devons utiliser un agrégateur.")

            differ = CodeDiffer(self.response_format)
            return differ.apply_diff(original_code, new_code)

        except Exception as e:
            logging.error(f"Erreur lors de l'agrégation: {str(e)}")
            traceback.print_exc()
            return original_code


    def _normalize_response_keys(self, response: Any) -> Any:
        """Normalise les clés de la réponse"""
        if hasattr(response, '__dict__'):
            data = response.model_dump()
            normalized_data = self._normalize_dict_keys(data)
            return self.pydantic_model(**normalized_data)
        return response

    def _normalize_dict_keys(self, data: Any) -> Any:
        """Normalise récursivement les clés d'un dictionnaire"""
        if isinstance(data, dict):
            return {
                k.lower(): self._normalize_dict_keys(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._normalize_dict_keys(item) for item in data]
        return data


class CodeDiffer:
    def __init__(self, response_format):
        self.response_format = response_format

    def generate_diff(self, code1, code2):
        # 1. Prétraitement
        code1_lines = self._preprocess(code1)
        code2_lines = self._preprocess(code2)

        # 2. Analyse des différences
        diff = []
        current_line = 0

        while current_line < len(code1_lines):
            if current_line >= len(code2_lines):
                # Suppression
                diff.append({
                    "operation": "d",
                    "source": f"{current_line + 1}",
                    "target": f"{current_line + 1}",
                    "content": []
                })
            elif code1_lines[current_line] != code2_lines[current_line]:
                # Modification
                indent = len(code2_lines[current_line]) - len(code2_lines[current_line].lstrip())
                diff.append({
                    "operation": "c",
                    "source": f"{current_line + 1}",
                    "target": f"{current_line + 1}",
                    "content": [{
                        "line": code2_lines[current_line].lstrip(),
                        "indent": indent
                    }]
                })
            current_line += 1

        # 3. Ajouts en fin de fichier
        for line in code2_lines[current_line:]:
            indent = len(line) - len(line.lstrip())
            diff.append({
                "operation": "a",
                "source": str(current_line),
                "target": str(current_line + 1),
                "content": [{
                    "line": line.lstrip(),
                    "indent": indent
                }]
            })
            current_line += 1

        return {"reasoning": "Modifications détectées", "diff": diff}

    def _preprocess(self, code):
        return [line for line in code.splitlines() if line.strip()]

    def apply_diff(self, code1, diff):
        """Applique les modifications du diff sur code1"""
        lines = code1.splitlines()
        result = []

        for modification in diff["diff"]:
            source = int(modification["source"].split(",")[0])

            if modification["operation"] == "a":
                # Ajout
                result.extend(lines[:source])
                for content in modification["content"]:
                    result.append(" " * content["indent"] + content["line"])
                result.extend(lines[source:])
            elif modification["operation"] == "c":
                # Modification
                result.extend(lines[:source - 1])
                for content in modification["content"]:
                    result.append(" " * content["indent"] + content["line"])
                result.extend(lines[source:])
            elif modification["operation"] == "d":
                # Suppression
                result.extend(lines[:source - 1])
                result.extend(lines[source:])

        return "\n".join(result)


def extract_json(texte):
    code_match = re.search(r"```json\n(.*?)```", texte, re.DOTALL | re.IGNORECASE)
    if code_match:
        texte = re.sub(r'("bloc de code": ")[^"]*\n[^"]*(")',
                       lambda m: m.group(1) + m.group(0).replace('\n', '\\n')[
                                              len(m.group(1)):-len(m.group(2))] + m.group(2), code_match.group(1))

        return texte


def extract_python(texte):
    code_match = re.search(r"```python\n(.*?)```", texte, re.DOTALL | re.IGNORECASE)
    if code_match:
        texte = re.sub(r'("bloc de code": ")[^"]*\n[^"]*(")',
                       lambda m: m.group(1) + m.group(0).replace('\n', '\\n')[
                                              len(m.group(1)):-len(m.group(2))] + m.group(2), code_match.group(1))

        return texte


def parse_line_range(line_str: str) -> tuple[int, int]:
    """
    Convertit '004' en (3,3) ou '007-008' en (6,7) pour un usage en index 0-based.
    """
    if '-' in line_str:
        start_str, end_str = line_str.split('-')
        start, end = int(start_str), int(end_str)
    else:
        start = end = int(line_str)
    # On convertit en 0-based
    return start - 1, end - 1


def completer_code(code: str, modification: Dict[str, Any]) -> str:
    """
    Applique les modifications décrites dans 'modification' au bloc de code fourni.
    Le paramètre 'modification' doit inclure une clé 'modifications'
    qui est une liste d'actions (insertion ou remplacement).

    Retourne le code modifié sous forme de chaîne.
    """
    print(modification)
    #modification = remove_leading_spaces(modification)
    #print(modification)

    # On découpe le code en lignes (en ignorant d’éventuels \n en fin/début de chaîne)
    lines = code.strip('\n').split('\n')

    # Parcours dans l'ordre des modifications.
    # (Si l'ordre a de l'importance et que les indices risquent d'être décalés,
    #  envisagez de trier ou de parcourir à l'envers selon votre logique.)
    for mod_item in modification.get("modifications", []):
        # 1) Cas d'insertion
        if mod_item.get('inserer_ligne') is not None:
            details = mod_item['inserer_ligne']['inserer']
            ligne_str = details['ligne']
            bloc_code = details['bloc_de_code']
            espaces = details['espaces']

            start_index, end_index = parse_line_range(ligne_str)
            # Pour une insertion, on se contente d'utiliser start_index
            # (end_index peut être ignoré ou servir à d'autres règles, selon votre logique).

            # On prépare le bloc de code à insérer, avec l'indentation voulue
            bloc_lines = bloc_code.split('\n')
            bloc_lines = [(' ' * espaces) + bl for bl in bloc_lines]

            # Insertion dans la liste de lignes
            # Note : lines[:start_index] prend tous les éléments avant start_index,
            #        lines[start_index:] prend tout ce qui suit.
            lines = lines[:start_index] + bloc_lines + lines[start_index:]

        # 2) Cas de remplacement
        elif mod_item.get('remplacer_ligne') is not None:
            details = mod_item['remplacer_ligne']['remplacement']
            ligne_str = details['ligne']
            bloc_code = details['bloc_de_code']
            espaces = details['espaces']

            start_index, end_index = parse_line_range(ligne_str)

            # Prépare le bloc de code de remplacement, avec l'indentation voulue
            bloc_lines = bloc_code.split('\n')
            bloc_lines = [(' ' * espaces) + bl for bl in bloc_lines]

            # Remplacement : on enlève les lignes de start_index à end_index (inclus)
            # puis on insère les nouvelles lignes
            lines = lines[:start_index] + bloc_lines + lines[end_index + 1:]

    # Reconstruction du code final
    return '\n'.join(lines)


# Fonction pour enlever les espaces en début de ligne dans 'bloc_de_code'
def remove_leading_spaces(data):
    for modification in data.get('modifications', []):
        for key in ['inserer_ligne', 'remplacer_ligne']:
            ligne = modification.get(key)
            if ligne:
                bloc_de_code = (ligne.get('inserer', {}).get('bloc_de_code') or
                                ligne.get('remplacement', {}).get('bloc_de_code'))
                if bloc_de_code:
                    # Diviser en lignes
                    lignes = bloc_de_code.split('\n')
                    if lignes:
                        # Enlever les espaces uniquement de la première ligne
                        lignes[0] = lignes[0].lstrip()
                        # Réassembler le bloc de code
                        nouveau_bloc = '\n'.join(lignes)

                        # Mettre à jour le bloc de code approprié
                        if 'inserer' in ligne:
                            ligne['inserer']['bloc_de_code'] = nouveau_bloc
                        if 'remplacement' in ligne:
                            ligne['remplacement']['bloc_de_code'] = nouveau_bloc
    return data

def _completer_code(code_initial: str, modifications: dict) -> str:
    """
    Insère des morceaux de code dans le code initial à des numéros de ligne spécifiques,
    en tenant compte des indentations spécifiées.
    """

    try:
        # Diviser le code initial en lignes
        code_lines = code_initial.split('\n')
        nb_lignes = len(code_lines)
        resultat = []

        # Créer un inventaire des modifications
        inserer_modifications = {}
        remplacer_modifications = {}

        cleaned_mods = []
        for mod in modifications["modifications"]:
            cleaned_mod = {}
            for key, value in mod.items():
                if value is not None:
                    if isinstance(value, dict):
                        cleaned_sub = {k: v for k, v in value.items() if v is not None}
                        if cleaned_sub:
                            cleaned_mod[key] = cleaned_sub
                    else:
                        cleaned_mod[key] = value
            if cleaned_mod:
                cleaned_mods.append(cleaned_mod)
        modifications["modifications"] = cleaned_mods

        # Traiter les modifications
        if "modifications" in modifications:
            modifications_list = modifications["modifications"]
            print(f"Liste des modifications: {modifications_list}")  # Debug

            for modif in modifications_list:
                print(f"Traitement de la modification: {modif}")  # Debug

                if "remplacer_ligne" in modif:
                    details = modif["remplacer_ligne"]["remplacement"]
                    print(f"Details remplacement: {details}")  # Debug

                    # Traiter les numéros de ligne
                    if isinstance(details['ligne'], str):
                        debut, fin = map(int, details['ligne'].split('-'))
                    else:
                        debut = fin = int(details['ligne'])

                    # Préparer le bloc de code avec la bonne indentation
                    indentation = int(details['espaces'])
                    bloc_code = details['bloc_de_code']
                    # Diviser le bloc en lignes et appliquer l'indentation
                    lignes = bloc_code.split('\n')
                    bloc_code_indente = [' ' * indentation + ligne.lstrip() for ligne in lignes]
                    remplacer_modifications[(debut, fin)] = bloc_code_indente

        # Appliquer les modifications
        i = 1
        while i <= len(code_lines):
            if i in inserer_modifications:
                resultat.append(code_lines[i - 1])
                resultat.extend(inserer_modifications[i])
                i += 1
            else:
                remplacement_trouve = False
                for (debut, fin), bloc_code in remplacer_modifications.items():
                    if debut <= i <= fin:
                        if i == debut:
                            resultat.extend(bloc_code)
                        remplacement_trouve = True
                        i += 1
                        break

                if not remplacement_trouve:
                    # Ne pas ajouter la numérotation des lignes
                    ligne = re.sub(r'^\d{3} ', '', code_lines[i - 1])
                    resultat.append(ligne)
                    i += 1

        code_final = '\n'.join(resultat)
        print(f"Code final généré:\n{code_final}")  # Debug
        return code_final

    except Exception as e:
        print(f"Erreur dans completer_code: {str(e)}")
        traceback.print_exc()
        return code_initial


def numeroter_lignes(code):
    lignes = code.split('\n')
    lignes_numerotees = [f"{i:03d} {ligne}" for i, ligne in enumerate(lignes, start=1)]
    return '\n'.join(lignes_numerotees)


def supprimer_numerotation(code: str) -> str:
    """
    Supprime uniquement la numérotation en début de ligne qui correspond exactement à trois chiffres suivis d'un seul espace,
    sans affecter le reste du contenu de la ligne y compris toute indentation supplémentaire après cet espace.

    :param code: Le code textuel comportant une numérotation de ligne au format spécifié.
    :return: Le code sans la numérotation des lignes.
    """
    lignes = code.split('\n')
    code_sans_numerotation = []

    for ligne in lignes:
        # Supprime la séquence de trois chiffres suivis d'exactement un espace, si elle se trouve au début de la ligne
        nouvelle_ligne = re.sub(r'^\d{3} ', '', ligne, 1)
        code_sans_numerotation.append(nouvelle_ligne)

    return '\n'.join(code_sans_numerotation)
