"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# src/agents/development_agent.py
import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import jsonpatch
from pydantic import BaseModel

from agent.src.agent import Agent
from agent.src.agents.diff_agregateur_agent import DiffAgregateurAgent
from agent.src.interface.html_manager import HTMLManager
from agent.src.server.form_server import FormServer
from agent.src.types.enums import AgentRole, AgentCapability
from agent.src.components.file_manager import FileManager
from agent.src.components.task_manager import TaskManager, Tache

from agent.src.types.interfaces import LLMProvider, MemoryProvider

from agent.src.agents.agregateur_agent import AiderCodeAggregator


class DevelopmentAgent(Agent):
    """
    Agent spécialisé pour les tâches de développement.

    Capacités :
    - Traitement des tâches selon le rôle
    - Gestion des fichiers
    - Génération d'interfaces HTML
    - Gestion des dépendances
    - Validation de code
    """

    def __init__(
            self,
            name: str,
            role_name: str,
            llm_provider: LLMProvider,
            memory_provider: MemoryProvider,
            system_prompt: str,
            project_path: str,
            html_manager: HTMLManager,
            project_name: str,
            file_manager: Optional[FileManager] = None,
            task_manager: Optional[TaskManager] = None,
            agregateur: Optional[AiderCodeAggregator] = None,
            memory_size: int = 1000,
            pydantic_model: Optional[BaseModel] = None,
            response_format: str = "",
            actionable_keywords_files: Optional[List] = None
    ):
        # Conversion du rôle et initialisation de la classe parent

        self.a_k_files = ['files', 'dependencies']
        if isinstance(actionable_keywords_files, list):
            self.a_k_files.extend(actionable_keywords_files)

        role = self._convert_role(role_name)
        super().__init__(
            name=name,
            role=role,
            llm_provider=llm_provider,
            memory_provider=memory_provider,
            system_prompt=system_prompt,
            memory_size=memory_size
        )

        # Composants de développement
        self.pydantic_model = pydantic_model
        self.html_manager = html_manager
        self.project_name = project_name
        self.project_path = project_path
        self.response_format = response_format

        # Gestionnaires
        self.file_manager = file_manager or FileManager(project_path)
        self.task_manager = task_manager or TaskManager()

        # Formulaire interactif
        self.form_server = FormServer()
        self.form_response = None
        self.form_event = asyncio.Event()

        # Agrégateur de code
        self.agregateur = agregateur

        # Configuration du logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.name}")
        self.logger.setLevel(logging.INFO)

    async def process_message(self, message: str) -> Any:
        """
        Traite un message et gère les interactions si nécessaire.

        Args:
            message: Message à traiter

        Returns:
            Résultat du traitement (peut être structuré via Pydantic)
        """
        try:
            self.logger.info(f"Traitement du message par {self.name}")

            message += """\n\nIMPORTANT\n# diff rules:

        Return edits similar to unified diffs that `diff -U0` would produce.
        
        Make sure you include the first 2 lines with the file paths.
        Don't include timestamps with the file paths.
        
        Start each hunk of changes with a `@@ ... @@` line.
        Don't include line numbers like `diff -U0` does.
        The user's patch tool doesn't need them.
        
        The user's patch tool needs CORRECT patches that apply cleanly against the current contents of the file!
        Think carefully and make sure you include and mark all lines that need to be removed or changed as `-` lines.
        Make sure you mark all new or modified lines with `+`.
        Don't leave out any lines or the diff patch won't apply correctly.
        
        Indentation matters in the diffs!
        
        Start a new hunk for each section of the file that needs changes.
        
        Only output hunks that specify changes with `+` or `-` lines.
        Skip any hunks that are entirely unchanging ` ` lines.
        
        Output hunks in whatever order makes the most sense.
        Hunks don't need to be in any particular order.
        
        When editing a function, method, loop, etc use a hunk to replace the *entire* code block.
        Delete the entire existing version with `-` lines and then add a new, updated version with `+` lines.
        This will help you generate correct code and correct diffs.
        
        To move code within a file, use 2 hunks: 1 to delete it from its current location, 1 to insert it in the new location."""
            if self.pydantic_model:
                # Obtenir la réponse structurée du LLM
                response = await self.llm_provider.generate_response(
                    message,
                    pydantic_model=self.pydantic_model
                )

                # Normaliser les clés de la réponse
                normalized_response = self._normalize_response_keys(response)

                # Générer le HTML pour visualisation/interaction
                html_file = self.html_manager.generate_html(
                    instance=normalized_response,
                    project_name=self.project_name,
                    agent_name=self.name,
                    prompt=self.system_prompt,
                    input_message=message
                )

                # Cherche les actionnable (a_k)
                # Traiter les fichiers et dépendanceq si présents
                # TODO que c'est moche cette approche
                for attr in self.a_k_files:
                    if hasattr(normalized_response, attr):  # Vérifie si 'files' existe
                        if attr == "files":
                            files = getattr(normalized_response, attr)

                            # Étape 1 : Installer les dépendances en premier
                            for file_obj in files:
                                if hasattr(file_obj, "dependencies"):  # Vérifie si 'dependencies' existe
                                    await self.handle_dependencies(file_obj.dependencies)

                            # Étape 2 : Traiter les fichiers après
                            await self._process_files(files)

                # Retourner le résultat normalisé
                return normalized_response

            return await super().process_message(message)

        except Exception as e:
            self.logger.error(f"Erreur dans process_message: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Surcharge pour ajouter la validation des fichiers"""
        try:
            # Utiliser le traitement de base de l'agent
            result = await super().process_task(task)

            # Ajouter la validation spécifique aux fichiers
            if any(word in result for word in self.a_k_files) in result and not self.can_perform_action(AgentCapability.MODIFY_FILES):
                raise PermissionError(f"Agent {self.name} non autorisé à modifier des fichiers")

            return result

        except Exception as e:
            self.logger.error(f"Erreur lors du traitement de la tâche: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def _process_files(self, files: List[Dict[str, Any]]) -> None:
        """Traite les fichiers à créer ou modifier avec gestion adaptative"""
        try:

            for file_info in files:
                if not self.can_perform_action(AgentCapability.MODIFY_FILES):
                    self.logger.warning(f"Agent {self.name} tente de modifier des fichiers sans autorisation")
                    continue

                file_path = file_info.file_path

                # Vérification si le fichier existe
                file_exists = await self.file_manager.file_exists(file_path)

                if file_exists:

                    # Pour les fichiers existants, utiliser l'agrégateur
                    original_content = await self.file_manager.read_file(file_path)
                    new_content = original_content # Par défaut, le contenu ne change pas

                    has_changes = False
                    # Vérifier si l'agent a fourni un patch dans code_changes
                    if hasattr(file_info,
                               'code_changes') and file_info.code_changes and file_info.code_changes.new_code:
                        patch_content = file_info.code_changes.new_code

                        # On utilise pathlib pour extraire l'extension SANS le point.
                        file_ext = Path(file_path).suffix.lstrip('.')

                        new_content = await self.agregateur.aggregate_code(original_content,
                                                                     patch_content,
                                                                     file_extension=file_ext)


                        has_changes = True

                    await self.file_manager.create_or_update_file(
                        file_path,
                        new_content
                    )
                else:
                    # Pour les nouveaux fichiers, utiliser directement le contenu
                    await self.file_manager.create_or_update_file(
                        file_path,
                        file_info.code_field.content
                    )

                self.logger.info(f"Fichier traité: {file_path}")

        except Exception as e:
            self.logger.error(f"Erreur lors du traitement des fichiers: {str(e)}")
            raise

    def apply_patch(self, original_content, file_info):
        """
        Applique un patch au contenu original en utilisant soit patch via subprocess
        pour les fichiers diff, soit jsonpatch pour JSON.

        Args:
            original_content (str): Le contenu original du fichier
            file_info (dict): Informations sur le fichier et les modifications à apporter

        Returns:
            str: Le contenu modifié après application du patch
        """
        import json
        import os
        import tempfile
        import subprocess
        import jsonpatch

        # Extraire les informations nécessaires
        code_changes = file_info.get('code_changes', {})
        if not code_changes:
            return original_content

        operation = code_changes.get('operation', '')
        patch_content = code_changes.get('new_code', '')

        if not patch_content:
            return original_content

        # Appliquer le patch selon le type d'opération
        if operation == "jsonpatch":
            return self._apply_json_patch(original_content, patch_content)
        elif operation == "diff":
            # Essayer d'abord avec subprocess
            try:
                result = self._apply_with_subprocess(original_content, patch_content)
                if result:
                    return result
            except Exception as e:
                self.logger.warning(f"Échec de l'application du patch via subprocess: {str(e)}")

            # Si subprocess échoue, essayer avec l'approche directe
            try:
                return self._apply_diff_direct(original_content, patch_content)
            except Exception as e:
                # Si tout échoue, lever une exception
                raise ValueError(f"Erreur lors de l'application du patch: {str(e)}")
        else:
            return original_content

    def _apply_json_patch(self, original_content, patch_content):
        """Applique un patch JSON au contenu original."""
        import json
        import jsonpatch

        try:
            # Convertir le contenu original en objet JSON
            json_obj = json.loads(original_content)

            # Convertir le patch_content en objet patch JSON
            patch_obj = json.loads(patch_content) if isinstance(patch_content, str) else patch_content

            # Appliquer le patch
            patched_obj = jsonpatch.apply_patch(json_obj, patch_obj)

            # Convertir l'objet patché en chaîne JSON
            return json.dumps(patched_obj, indent=2)
        except Exception as e:
            raise ValueError(f"Erreur lors de l'application du patch JSON: {str(e)}")

    def _apply_with_subprocess(self, original_content, patch_content):
        """Applique un patch diff au contenu original en utilisant la commande patch."""
        import os
        import tempfile
        import subprocess

        original_file_path = None
        patch_file_path = None

        try:
            # Assurer que le contenu se termine par une nouvelle ligne
            if not original_content.endswith('\n'):
                original_content += '\n'

            # Idem pour le patch
            if not patch_content.endswith('\n'):
                patch_content += '\n'

            # Créer des fichiers temporaires pour le contenu original et le patch
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as original_file:
                original_file.write(original_content)
                original_file_path = original_file.name

            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as patch_file:
                patch_file.write(patch_content)
                patch_file_path = patch_file.name

            # Exécuter la commande patch avec différentes options
            success = False
            last_error = None
            for p_level in [0, 1]:  # Essayer avec -p0 et -p1
                try:
                    result = subprocess.run(
                        ['patch', f'-p{p_level}', '--ignore-whitespace', '-i', patch_file_path,
                         '-f', '--no-backup-if-mismatch', original_file_path],
                        capture_output=True,
                        text=True,
                        check=False
                    )

                    if result.returncode == 0:
                        # Succès! Lire le contenu du fichier patché
                        with open(original_file_path, 'r') as patched_file:
                            patched_content = patched_file.read()
                        success = True
                        break
                    else:
                        last_error = result.stderr
                except Exception as e:
                    last_error = str(e)
                    continue  # Essayer le niveau suivant

            if success:
                return patched_content
            else:
                raise ValueError(f"Échec de l'application du patch: {last_error}")

        except Exception as e:
            raise ValueError(f"Erreur lors de l'utilisation de subprocess: {str(e)}")

        finally:
            # Nettoyer les fichiers temporaires
            for file_path in [original_file_path, patch_file_path]:
                if file_path and os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                    except Exception:
                        pass  # Ignorer les erreurs de nettoyage

    def _apply_diff_direct(self, original_content, patch_content):
        """
        Applique directement un patch diff au contenu original
        en extrayant les modifications et en les appliquant manuellement.
        Utilisé comme solution de secours si la commande patch échoue.
        """
        # Extraire les paires avant/après du patch
        before_after_pairs = self._extract_before_after_from_patch(patch_content)

        if not before_after_pairs:
            return original_content

        # Appliquer chaque paire dans l'ordre
        result = original_content
        for before, after in before_after_pairs:
            # Cas 1: before est vide, ajouter à la fin
            if not before.strip():
                result = result + after
                continue

            # Cas 2: remplacement direct
            if before in result:
                result = result.replace(before, after)
                continue

            # Cas 3: essayer avec des espaces normalisés
            normalized_before = '\n'.join(line.rstrip() for line in before.splitlines())
            normalized_content = '\n'.join(line.rstrip() for line in result.splitlines())

            if normalized_before in normalized_content:
                # Reconstruire les positions correctes dans le texte original
                start_pos = 0
                found = False

                for i in range(len(result) - len(before) + 1):
                    candidate = result[i:i + len(before)]
                    normalized_candidate = '\n'.join(line.rstrip() for line in candidate.splitlines())

                    if normalized_candidate == normalized_before:
                        start_pos = i
                        found = True
                        break

                if found:
                    result = result[:start_pos] + after + result[start_pos + len(before):]
                    continue

        return result

    def _extract_before_after_from_patch(self, patch_content):
        """
        Extrait les paires de texte avant/après à partir d'un patch.

        Returns:
            Liste de tuples (before, after) représentant chaque changement
        """
        # Diviser en lignes
        lines = patch_content.splitlines(True)

        # Supprimer les lignes d'en-tête du fichier si présentes
        start_index = 0
        if len(lines) >= 2 and lines[0].startswith("--- ") and lines[1].startswith("+++ "):
            start_index = 2

        # Ignorer les lignes d'en-tête de hunk (@@ ... @@)
        # et regrouper par hunks
        hunks = []
        current_hunk = []
        in_hunk = False

        for i in range(start_index, len(lines)):
            line = lines[i]

            if line.startswith("@@"):
                if current_hunk:
                    hunks.append(current_hunk)
                current_hunk = []
                in_hunk = True
                continue

            if in_hunk:
                current_hunk.append(line)

        if current_hunk:
            hunks.append(current_hunk)

        # Traiter chaque hunk pour extraire les paires avant/après
        pairs = []
        for hunk in hunks:
            before = []
            after = []

            for line in hunk:
                if not line:
                    continue

                if len(line) < 1:
                    continue

                op = line[0]
                content = line[1:] if len(line) > 1 else ""

                if op == " ":
                    before.append(content)
                    after.append(content)
                elif op == "-":
                    before.append(content)
                elif op == "+":
                    after.append(content)

            before_text = "".join(before)
            after_text = "".join(after)

            if before_text != after_text:  # Ne garder que les paires avec des changements
                pairs.append((before_text, after_text))

        return pairs


    def find_position_in_code(self, code: str, context_before: str, match_code: str, context_after: str) -> Optional[
        Tuple[int, int]]:
        """Trouve la position exacte dans le code où appliquer une modification"""
        import re

        # Pour l'ajout (sans code à matcher)
        if not match_code:
            if context_before and context_after:
                # Chercher la transition entre context_before et context_after
                pattern = re.escape(context_before) + r'(.*?)' + re.escape(context_after)
                match = re.search(pattern, code, re.DOTALL)
                if match:
                    start_pos = match.start(1)  # Début du groupe capturé
                    return (start_pos, start_pos)
            elif context_before:
                # Trouver la fin du contexte avant
                pos = code.find(context_before)
                if pos >= 0:
                    return (pos + len(context_before), pos + len(context_before))
        else:
            # Pour modification ou suppression
            if context_before and context_after:
                # Chercher avec contexte complet
                pattern = re.escape(context_before) + re.escape(match_code) + re.escape(context_after)
                match = re.search(pattern, code, re.DOTALL)
                if match:
                    start_pos = match.start() + len(context_before)
                    return (start_pos, start_pos + len(match_code))

            # Chercher juste le code à matcher
            pos = code.find(match_code)
            if pos >= 0:
                return (pos, pos + len(match_code))

        return None

    async def old_process_files(self, files: List[Dict[str, Any]]) -> None:
        """Traite les fichiers à créer ou modifier"""
        #  TODO Retourner l'état de la compilation pour traitement ultérieur et modification
        try:
            for file_info in files:
                if not self.can_perform_action(AgentCapability.MODIFY_FILES):
                    self.logger.warning(f"Agent {self.name} tente de modifier des fichiers sans autorisation")
                    continue
                if await self.file_manager.file_exists(file_info.file_path):
                    new_code = await self.agregateur.aggregate_code(await self.file_manager.read_file(file_info.file_path), file_info.code_field.content)
                else:
                    new_code = file_info.code_field.content

                await self.file_manager.create_or_update_file(
                    file_info.file_path,
                    new_code
                )
                # Vérifie s'ils sont executables
                # TODO vérifier l'extension
                succes, sortie = await self.file_manager.executor.execute_python_file(os.path.join(self.file_manager.project_path, file_info.file_path))
                if succes:
                    self.logger.info(f"Fichier correct: {file_info.file_path}")
                else:
                    self.logger.info(f"Problème d'implémentation du fichier: {file_info.file_path}")
                    self.logger.error(f"Erreur: {sortie}")
                self.logger.info(f"Fichier traité: {file_info.file_path}")

        except Exception as e:
            self.logger.error(f"Erreur lors du traitement des fichiers: {str(e)}")
            raise

    def process_code_changes(self, original_file_path, patch_file_path):

        """Subprocess pour diff"""
        result = subprocess.run(['patch', original_file_path, patch_file_path],
                                capture_output=True, text=True)

        return result.returncode == 0

    async def handle_dependencies(self, dependencies: List[str]) -> tuple[bool, str]:
        """
        Gère l'installation des dépendances Python.

        Args:
            dependencies: Liste des packages à installer

        Returns:
            (succès, message)
        """
        try:
            # Création ou mise à jour du requirements.txt
            self.logger.info(f"Création ou mise à jour du requirements.txt")

            requirements_path = "requirements.txt"
            requirements_content = "\n".join(dependencies)

            await self.file_manager.create_or_update_file(
                str(requirements_path),
                requirements_content
            )

            self.logger.info(f"Installation des dépendances...")
            # Normaliser le chemin
            normalized_path = os.path.normpath(str(requirements_path))
            absolute_path = os.path.join(self.file_manager.project_path, normalized_path)
            # Installation des dépendances
            success, output = await self.file_manager.executor.install_requirements(
                absolute_path
            )

            if not success:
                self.logger.error(f"Erreur lors de l'installation des dépendances: {output}")
                raise Exception(f"Erreur lors de l'installation des dépendances: {output}")

            return True, "Dépendances installées avec succès"

        except Exception as e:
            self.logger.error(f"Erreur lors de la gestion des dépendances: {str(e)}")
            return False, str(e)

    @staticmethod
    def _convert_role(role_name: str) -> AgentRole:
        """Convertit les noms de rôles en AgentRole enum"""
        role_mapping = {
            # Coordinateurs
            "architecte": AgentRole.COORDINATOR,
            "generateur_idees": AgentRole.EXECUTOR,
            "developpeur": AgentRole.EXECUTOR,
            # Spécialistes
            "ingenieur": AgentRole.SPECIALIST,

            # Exécuteurs
            "frontend_dev": AgentRole.EXECUTOR,
            "backend_dev": AgentRole.EXECUTOR,
            "database_dev": AgentRole.EXECUTOR,
            "designer_interface": AgentRole.EXECUTOR,
            "codeur": AgentRole.EXECUTOR,

            # Critiques
            "testeur": AgentRole.CRITIC,
            "inspecteur": AgentRole.CRITIC
        }

        if role_name not in role_mapping:
            raise ValueError(f"Rôle non supporté: {role_name}")

        return role_mapping[role_name]

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

    async def _wait_for_form_submission(self, html_file: str) -> Dict[str, Any]:
        """Attend la soumission du formulaire HTML"""
        try:
            server_task = asyncio.create_task(self.form_server.start())

            # Ouvrir le navigateur
            import webbrowser
            webbrowser.open(f'file://{html_file}')

            # Attendre la soumission
            response = await self.form_server.wait_for_submission()

            # Arrêter le serveur
            await self.form_server.stop()
            await server_task

            if not response:
                raise ValueError("Aucune réponse reçue du formulaire")

            return response

        except Exception as e:
            await self.form_server.stop()
            raise Exception(f"Erreur lors de l'attente du formulaire: {str(e)}")

    async def _create_enhanced_task_message(self, task_data: Dict[str, Any]) -> str:
        """Crée un message enrichi pour une tâche avec tout le contexte nécessaire"""
        # todo est il necessaire d'enrichir le contexte
        return str(task_data)
        """
        try:
            # Récupération du contenu des fichiers existants
            file_contents = {}
            for file_path in task_data.get('files', []):
                if await self.file_manager.file_exists(file_path):
                    content = await self.file_manager.read_file(file_path)
                    file_contents[file_path] = content

            # Construction du contexte enrichi
            enhanced_context = {
                "task": task_data,
                "context": {
                    "project_description": self.project_description if hasattr(self, 'project_description') else None,
                    "existing_files": file_contents,
                    "dependent_tasks": await self._get_dependent_tasks(task_data)
                },
                "environment": {
                    "project_structure": self.file_manager.get_project_structure()
                }
            }

            # Conversion en texte formaté pour le LLM
            return self._format_context_for_llm(enhanced_context)

        except Exception as e:
            logging.error(f"Erreur lors de la création du message enrichi: {str(e)}")
            raise
        """

    async def _get_dependent_tasks(self, task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Récupère les détails des tâches dont dépend la tâche actuelle"""
        dependent_tasks = []
        if not hasattr(self, 'task_manager'):
            return dependent_tasks

        for dep_id in task_data.get('dependencies', []):
            if task := self.task_manager.get_task(dep_id):
                dependent_tasks.append({
                    "id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "status": task.status
                })
        return dependent_tasks

    def _format_context_for_llm(self, context: Dict[str, Any]) -> str:
        """Formate le contexte enrichi en texte structuré pour le LLM"""
        sections = []

        # Section Tâche
        task = context['task']
        sections.extend([
            "=== DÉTAILS DE LA TÂCHE ===",
            f"ID: {task.get('id', 'N/A')}",
            f"Titre: {task.get('title', 'N/A')}",
            f"Description: {task.get('description', 'N/A')}"
        ])

        # Critères d'acceptation
        if task.get('acceptance_criteria'):
            sections.append("\nCritères d'acceptation:")
            sections.extend([f"- {criterion}" for criterion in task['acceptance_criteria']])

        # Section Fichiers
        if task.get('files'):
            sections.append("\n=== FICHIERS À TRAITER ===")
            sections.extend([f"- {file}" for file in task['files']])

        # Fichiers existants
        existing_files = context.get('context', {}).get('existing_files', {})
        if existing_files:
            sections.append("\n=== CONTENU DES FICHIERS EXISTANTS ===")
            for file_path, content in existing_files.items():
                sections.extend([
                    f"\nFichier: {file_path}",
                    "```",
                    content,
                    "```"
                ])

        # Tâches dépendantes
        dependent_tasks = context.get('context', {}).get('dependent_tasks', [])
        if dependent_tasks:
            sections.append("\n=== TÂCHES DÉPENDANTES ===")
            for dep_task in dependent_tasks:
                sections.extend([
                    f"\nTâche #{dep_task['id']}: {dep_task['title']}",
                    f"Status: {dep_task['status']}"
                ])

        # Structure du projet
        project_structure = context.get('environment', {}).get('project_structure')
        if project_structure:
            sections.append("\n=== STRUCTURE DU PROJET ===")
            sections.append(json.dumps(project_structure, indent=2))

        return "\n".join(sections)



