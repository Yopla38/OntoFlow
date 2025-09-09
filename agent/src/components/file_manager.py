"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

import json
import logging
import os
import re
import subprocess
import traceback
from datetime import datetime
from typing import Dict, Optional, List, Any

from agent.src.components.executor import VenvCommandExecutor
from agent.src.components.task_manager import Revision


class Fichier:
    def __init__(self, projet_path, chemin, contenu="", executeur: VenvCommandExecutor = None):
        chemin = chemin.rstrip('*')
        chemin = chemin.lstrip('### ')
        chemin = chemin.rstrip('**')
        self.chemin = chemin.lstrip('/')
        self.contenu = contenu
        self.revisions = []
        self.abs_path = projet_path
        self.chemin_absolu = os.path.join(self.abs_path, self.chemin)
        self.revisions_file_path = f"{self.chemin_absolu}.revisions.json"

        os.makedirs(os.path.dirname(self.chemin_absolu), exist_ok=True)

        self.load_revisions()  # Chargement des révisions

        if not os.path.exists(self.chemin_absolu):
            logging.info(f"Création du fichier {os.path.basename(chemin)}")
            with open(self.chemin_absolu, 'w') as file:
                if contenu != "":
                    file.write(self.contenu)
                pass
        else:
            self.charger()
        #except IOError as e:
        #    logging.error(f"Erreur lors de la sauvegarde du fichier '{self.chemin}': {e}")
        self.executor = executeur

    def sauvegarder(self):
        if self.chemin == os.path.basename("requirements.txt"):
            self.ajouter(self.contenu)
            return
        try:
            with open(self.chemin_absolu, 'w') as fichier:
                fichier.write(self.contenu)
        except IOError as e:
            print(f"Erreur lors de la sauvegarde du fichier '{self.chemin}': {e}")
            traceback.print_exc()

    def charger(self):
        try:
            with open(self.chemin_absolu, 'r') as fichier:
                self.contenu = fichier.read()
        except IOError as e:
            print(f"Erreur lors de la lecture du fichier '{os.path.basename(self.chemin)}': {e}")
            traceback.print_exc()

    def ajouter(self, contenu):
        # TODO Des fois le contenu peut être tout le fichier donc redondance des imports
        try:
            # Ouvrir le fichier en mode append ('a+') pour ajouter du contenu à la fin
            with open(self.chemin_absolu, 'a+') as f:
                # Vérifier si le fichier est vide
                f.seek(0, 2)  # Déplacer le curseur à la fin du fichier
                if f.tell() > 0:  # Vérifier si le fichier est vide
                    f.write('\n')  # Ajouter un retour à la ligne si le fichier n'est pas vide
                # Ajouter le contenu à la fin du fichier
                f.write(contenu)
            logging.info(f"Contenu ajouté avec succès dans le fichier {os.path.basename(self.chemin)}")
        except Exception as e:
            logging.error(f"Erreur lors de l'écriture dans le fichier {os.path.basename(self.chemin)} : {e}")
            traceback.print_exc()

    def executer(self, force_validation: bool = False) -> (bool, str):
        parts = self.chemin.split(".")
        extension = parts[-1]
        if extension == "py":  # on va tenter d'executer le code si c'est un .py
            validation, message = self.validate_and_exec_py(self.contenu, self.executor)
            validation = True if force_validation else validation
            return validation, message
        return None, None

    @staticmethod
    def contains_flask_server_start(code_string):
        # Recherche de l'initialisation de Flask
        if re.search(r"Flask\(__name__\)", code_string):
            # Recherche d'un appel à .run() qui est typique pour démarrer le serveur
            if re.search(r"\.run\(\)", code_string):
                return True
        return False

    def validate_and_exec_py(self, code, executor: VenvCommandExecutor, timeout=10):
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return False, f"Erreur de syntaxe: {e}"

        # Exécution du code dans un environnement contrôlé
        command = f"{self.executor.python_executable} -c \"{code}\""

        success, result = executor.send_bash_command(command)

        if success:
            return True, "\n".join(result)
        else:
            return False, "\n".join(result if result else ["Unknown error"])

    def _validate_and_exec_py(self, code, timeout=10):
        # Première étape: validation de syntaxe
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return False, f"Erreur de syntaxe: {e}"

        # Deuxième étape: exécuter le code dans un processus séparé
        try:
            process = subprocess.run(['python', '-c', code],
                                     text=True,
                                     capture_output=True,
                                     timeout=timeout)
        except subprocess.TimeoutExpired:
            if self.contains_flask_server_start(code):
                return True, "C'est un serveur: Délai d'exécution dépassé, consideration de ce code comme validé"
            return False, "Erreur: Délai d'exécution dépassé"
        except Exception as e:
            return False, f"Erreur pendant l'exécution: {e}"

        # Retourne un tuple avec un indicateur de succès et le résultat
        if process.returncode == 0:
            return True, process.stdout.strip()
        else:
            return False, process.stderr.strip()

    @property
    def contenu(self):
        return self._contenu

    @contenu.setter
    def contenu(self, value):
        if hasattr(self, '_contenu'):
            # Ajout de la révision actuelle avant de changer le contenu
            self.revisions.append(Revision(self._contenu))
            self.save_revisions()  # Sauvegarde après chaque modification
        self._contenu = value

    def save_revisions(self):
        #  TODO revoir le principe des fichiers de révision
        # Enregistrement des révisions dans un fichier JSON
        pass
        """
        with open(self.revisions_file_path, 'w') as file:
            json.dump([vars(rev) for rev in self.revisions], file, default=str)
        """

    def load_revisions(self):
        # Chargement des révisions à partir d'un fichier JSON
        if os.path.exists(self.revisions_file_path):
            try:
                with open(self.revisions_file_path, 'r') as file:
                    revisions_data = json.load(file)
                    self.revisions = [Revision(**rev) for rev in revisions_data]
            except Exception as e:
                print(f"Erreur {e} sur le fichier {self.revisions_file_path}")

    def restore_revision(self, index):
        if 0 <= index < len(self.revisions):
            self.contenu = self.revisions[index].content
        else:
            raise IndexError("Aucune révision trouvée à cet index")

    def list_revisions(self):
        for idx, revision in enumerate(self.revisions):
            print(f"Revision {idx}: {revision.timestamp} - {revision.content[:30]}...")


class FileManager:
    def __init__(self, project_path: str, venv_path: str = None):
        self.project_path = project_path
        self.venv_path = venv_path or os.path.join(project_path, ".venv")
        self.files: Dict[str, Fichier] = {}
        self.executor = VenvCommandExecutor(
            venv_path=self.venv_path,
            working_dir=self.project_path
        )

        # Initialisation de l'environnement virtuel
        self.initialize_venv()

    def get_all_files(self) -> Dict[str, str]:
        """Retourne le contenu de tous les fichiers gérés"""
        return {
            path: file.contenu
            for path, file in self.files.items()
            if not path.endswith('/')  # Exclure les dossiers
        }

    def get_files_by_extension(self, extension: str) -> Dict[str, str]:
        """Retourne les fichiers avec une extension spécifique"""
        return {
            path: file.contenu
            for path, file in self.files.items()
            if path.endswith(extension)
        }

    def get_project_structure(self) -> Dict[str, Any]:
        """Retourne la structure actuelle du projet basée sur les fichiers enregistrés"""
        structure = {}
        for file_path in self.files.keys():
            parts = file_path.split('/')
            current = structure
            for part in parts[:-1]:  # Traiter les dossiers
                if part not in current:
                    current[part] = {}
                current = current[part]
            if parts[-1]:  # Ajouter le fichier
                current[parts[-1]] = None
        return structure

    async def file_exists(self, file_path: str) -> bool:
        """Vérifie si un fichier existe"""
        absolute_path = os.path.join(self.project_path, file_path)
        exist = os.path.exists(absolute_path)
        if not exist:
            logging.info(f"Le fichier {absolute_path} n'existe pas!")
        return exist

    async def read_file(self, file_path: str) -> str:
        """Lit le contenu d'un fichier"""
        try:
            absolute_path = os.path.join(self.project_path, file_path)
            with open(absolute_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Erreur lors de la lecture du fichier {file_path}: {str(e)}")
            raise

    async def get_file_history(self, file_path: str) -> List[Dict[str, Any]]:
        """Récupère l'historique des modifications d'un fichier"""
        if file_path in self.files:
            return self.files[file_path].revisions
        return []

    async def initialize_venv(self):
        """Initialise l'environnement virtuel s'il n'existe pas"""
        try:
            logging.info(f"Initialisation de l'environnement virtuel dans {self.venv_path}")

            # Vérifier si l'environnement virtuel existe déjà
            if not os.path.exists(os.path.join(self.venv_path, "bin", "python")):
                logging.info("Création d'un nouvel environnement virtuel...")
                await self.executor.create_env()
                logging.info("Environnement virtuel créé avec succès")
            else:
                logging.info("L'environnement virtuel existe déjà")

            # Vérifier si l'environnement est fonctionnel
            success, output = await self.executor.send_bash_command(f"{self.venv_path}/bin/python --version")
            if success:
                logging.info(f"Environnement virtuel vérifié : {output}")
            else:
                logging.error(f"Erreur lors de la vérification de l'environnement : {output}")
                raise Exception("L'environnement virtuel n'est pas fonctionnel")

        except Exception as e:
            logging.error(f"Erreur lors de l'initialisation de l'environnement virtuel : {str(e)}")
            raise

    def add_file(self, path: str) -> Fichier:
        """Ajoute ou récupère un fichier"""
        if path not in self.files:
            self.files[path] = Fichier(
                projet_path=self.project_path,
                chemin=path,
                executeur=self.executor
            )
        return self.files[path]

    def get_file(self, path: str) -> Optional[Fichier]:
        """Récupère un fichier existant"""
        return self.files.get(path)

    def get_files_content(self, paths: List[str]) -> str:
        """Récupère le contenu de plusieurs fichiers"""
        content = []
        for path in paths:
            if file := self.get_file(path):
                content.append(f"Fichier : {path}\n```\n{file.contenu}\n```")
        return "\n\n".join(content)

    async def update_file(self, path: str, content: str):
        """Met à jour ou crée un fichier"""
        file = self.add_file(path)
        file.contenu = content
        file.sauvegarder()

    async def create_or_update_file(self, file_path: str, content: str) -> None:
        """Crée ou met à jour un fichier en respectant l'arborescence"""
        try:
            # Normaliser le chemin
            normalized_path = os.path.normpath(file_path)
            absolute_path = os.path.join(self.project_path, normalized_path)

            # TODO a faire proprement ...
            # Vérifier que le chemin final est bien dans le projet
            #if not os.path.commonpath([absolute_path, self.project_path]) == self.project_path:
            #    raise ValueError(f"Chemin non autorisé: {absolute_path}")

            # Créer les dossiers parents si nécessaire
            os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

            # Écrire le contenu
            with open(absolute_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Mettre à jour le dictionnaire des fichiers
            self.files[normalized_path] = Fichier(
                projet_path=self.project_path,
                chemin=normalized_path,
                contenu=content
            )

            logging.info(f"Fichier créé/mis à jour: {normalized_path}")

        except Exception as e:
            logging.error(f"Erreur lors de la création/mise à jour du fichier {file_path}: {str(e)}")
            raise

    def _get_project_structure(self) -> Dict[str, Any]:
        """Retourne la structure actuelle du projet"""
        structure = {}

        for root, dirs, files in os.walk(self.project_path):
            # Ignorer le dossier .venv et autres dossiers cachés
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            current = structure
            path = os.path.relpath(root, self.project_path)

            if path != '.':
                parts = path.split(os.sep)
                for part in parts:
                    current = current.setdefault(part, {})

            for file in files:
                if not file.startswith('.'):
                    current[file] = None

        return structure

    async def validate_python_file(self, file_path: str) -> tuple[bool, str]:
        """
        Valide un fichier Python.

        Args:
            file_path: Chemin du fichier à valider

        Returns:
            tuple[bool, str]: (succès, message)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Vérification de la syntaxe
            compile(content, file_path, 'exec')

            # Exécution dans l'environnement virtuel pour validation
            if self.executor:
                success, output = await self.executor.execute_python_file(file_path)
                return success, '\n'.join(output) if output else "Validation réussie"

            return True, "Validation syntaxique réussie"

        except SyntaxError as e:
            return False, f"Erreur de syntaxe : {str(e)}"
        except Exception as e:
            return False, f"Erreur de validation : {str(e)}"