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
import subprocess
import tempfile
import os
import json
from pathlib import Path
from typing import Optional

from utils.anthropic_setup import get_anthropic_key


class AiderCodeAggregator:
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialise l'agrégateur de code avec Aider

        Args:
            model_name: Le modèle à utiliser (gpt-4, claude-3-sonnet, deepseek, etc.)
            api_key: Clé API pour le modèle choisi
        """
        self.model_name = model_name
        self.api_key = api_key
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


async def example_usage():
    # Initialiser l'agrégateur
    aggregator = AiderCodeAggregator(
        model_name="claude-3-5-haiku-20241022",  # ou "claude-3-sonnet", "deepseek", etc.
        api_key=get_anthropic_key()
    )

    original_code = """
def hello():
    print("Hello World")

if __name__ == "__main__":
    hello()
"""

    new_code = """
def goodbye():
    print("Goodbye World")

def main():
    hello()
    goodbye()
"""

    # Agrégation
    result = await aggregator.aggregate_code(
        original_code,
        new_code,
        "Intègre la fonction goodbye et améliore la fonction main"
    )

    print("Code agrégé:")
    print(result)


if __name__ == "__main__":
    asyncio.run(example_usage())