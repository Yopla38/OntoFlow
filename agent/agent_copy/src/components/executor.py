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
import re
import subprocess
import time
import urllib.parse
from datetime import datetime
from functools import wraps
from typing import List

import pexpect
import os
import logging

class VenvCommandExecutor:
    def __init__(self, venv_path=None, working_dir=None, python_executable='python3'):
        self.python_executable = 'python' if not venv_path else os.path.join(venv_path, 'bin', 'python')
        self.working_dir = working_dir
        self.venv_path = os.path.join(os.getcwd(), venv_path) if venv_path else None
        self.input_output = []
        self.env = False
        # Configuration du logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}_VenvCommandExecutor")
        self.logger.setLevel(logging.INFO)

        self.local_python = python_executable
        self.env = os.environ.copy()  # Copie de l'environnement actuel

    async def create_env(self):
        """Création asynchrone de l'environnement virtuel"""
        try:
            self.logger.info(f"Création de l'environnement virtuel dans {self.venv_path}")

            # S'assurer que le dossier parent existe
            os.makedirs(os.path.dirname(self.venv_path), exist_ok=True)

            # Création de l'environnement virtuel
            create_venv_command = f"{self.local_python} -m venv '{self.venv_path}'"
            self.logger.info(f"Exécution de la commande : {create_venv_command}")

            success, output = await self.send_bash_command(create_venv_command)

            if not success:
                error_msg = ' '.join(output) if isinstance(output, list) else str(output)
                raise Exception(f"Erreur lors de la création du venv : {error_msg}")

            # Vérifier que l'environnement a bien été créé
            python_path = os.path.join(self.venv_path, 'bin', 'python')
            if not os.path.exists(python_path):
                raise Exception(f"Python non trouvé dans le venv : {python_path}")

            self.logger.info(f"Environnement virtuel créé avec succès dans {self.venv_path}")

            # Mise à jour optionnelle de pip
            upgrade_pip_command = f"{python_path} -m pip install --upgrade pip"
            success, output = await self.send_bash_command(upgrade_pip_command)
            if success:
                self.logger.info("Pip mis à jour avec succès")
            else:
                self.logger.warning("Impossible de mettre à jour pip, mais l'environnement est créé")

            return True

        except Exception as e:
            self.logger.error(f"Erreur lors de la création de l'environnement virtuel : {str(e)}")
            raise

    async def install_package(self, package: str) -> tuple[bool, str]:
        """Installe un package avec pip et gère les erreurs."""
        pip_path = os.path.join(self.venv_path, 'bin', 'pip')
        command = f"{pip_path} install {package}"

        success, output = await self.send_bash_command(command)

        if not success:
            # Vérifie si pip suggère un autre package (ex: cv2 -> opencv-python)
            match = re.search(r"Did you mean ([^\?]+)\?", output)
            if match:
                suggested_package = match.group(1).strip()
                self.logger.info(f"⚠️ '{package}' introuvable. Tentative d'installation de '{suggested_package}'...")
                return await self.install_package(suggested_package)  # Réessaie avec la suggestion

            # Log et retour de l'erreur sans interrompre le processus
            self.logger.error(f"❌ Échec de l'installation de {package}: {output}")
            return False, f"Erreur lors de l'installation de {package}: {output}"

        return True, f"{package} installé avec succès ✅"

    async def install_requirements(self, requirements_path: str) -> tuple[bool, str]:
        """Installation asynchrone des dépendances avec gestion des erreurs et continuité."""
        try:
            # Lire le fichier requirements.txt
            with open(requirements_path, "r") as f:
                packages = [line.strip() for line in f.readlines() if line.strip()]

            results = []
            for pkg in packages:
                try:
                    result = await self.install_package(pkg)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"❌ Erreur inattendue lors de l'installation de {pkg}: {str(e)}")
                    results.append((False, f"Erreur sur {pkg}: {str(e)}"))

            # Filtrer les erreurs
            errors = [msg for success, msg in results if not success]

            if errors:
                return False, "\n".join(errors)  # Retourne toutes les erreurs sans arrêter le processus

            return True, "Installation des dépendances réussie ✅"

        except Exception as e:
            return False, f"Erreur lors de l'installation : {str(e)}"

    async def verify_installation(self):
        """Vérifie que l'installation est correcte"""
        python_path = os.path.join(self.venv_path, 'bin', 'python')

        # Vérification de l'existence de Python
        if not os.path.exists(python_path):
            raise Exception(f"Python non trouvé dans le venv: {python_path}")

        # Vérification que Python peut être exécuté
        success, output = await self.send_bash_command(f"{python_path} --version")
        if not success:
            raise Exception(f"Impossible d'exécuter Python dans le venv: {output}")

        # Vérification des packages installés
        success, output = await self.send_bash_command(f"{python_path} -m pip list")
        if not success:
            raise Exception(f"Impossible de lister les packages: {output}")

    async def validate_dependencies(self, required_packages: List[str]) -> tuple[bool, List[str]]:
        """Vérifie si toutes les dépendances requises sont installées"""
        python_path = os.path.join(self.venv_path, 'bin', 'python')

        # Obtenir la liste des packages installés
        success, output = await self.send_bash_command(f"{python_path} -m pip freeze")
        if not success:
            return False, []

        installed_packages = {
            line.split('==')[0].lower(): line.split('==')[1]
            for line in output if '==' in line
        }

        missing_packages = []
        for package in required_packages:
            package_name = package.split('==')[0].lower()
            if package_name not in installed_packages:
                missing_packages.append(package)

        return len(missing_packages) == 0, missing_packages

    def start_env(self):
        self.process = pexpect.spawn(self.python_executable, timeout=None, encoding='utf-8', cwd=self.working_dir)
        self.process.expect_exact('>>> ')
        self.env = True

    def send_command(self, command):
        output = []
        # Découpe la commande en lignes
        lines = command.split('\n')
        for line in lines:
            if re.search(r"exit\(\)", line):
                self.input_output.append({'input': command, 'output': ['exit']})
                self.close()
                return 'exit'
            self.process.sendline(line)
            self.process.expect_exact('>>> ')
            lines = self.process.before.split('\r\n')
            output = lines[1:-1]
            for out in output:
                #self.write_log(output=out)  # imprime chaque sortie
                pass
            # Stocke la commande et sa sortie dans input_output
            self.input_output.append({'input': command, 'output': output})

        return '\n'.join(output)

    def close(self):
        self.process.close()
        self.env = False

    def write_log(self, command=None, output=None):
        if self.log_file:

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log = []
            if command is not None:
                log.append({
                    'timestamp': timestamp,
                    'command': command,
                })
            if output is not None or output != "\n" or output != "":
                log.append({
                    'timestamp': timestamp,
                    'output': output
                })
            with open(self.log_file, 'a') as f:
                json.dump(log, f, indent=4)
                f.write('\n')

    async def send_bash_command(self, command: str) -> tuple[bool, list[str]]:
        """Version asynchrone de l'exécution des commandes bash"""
        try:
            self.logger.debug(f"Exécution de la commande : {command}")

            # Création du processus sans le paramètre text
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Attente asynchrone des résultats
            stdout_bytes, stderr_bytes = await process.communicate()

            # Conversion des bytes en str
            stdout = stdout_bytes.decode('utf-8')
            stderr = stderr_bytes.decode('utf-8')

            # Une sortie vide n'est pas nécessairement une erreur
            if process.returncode == 0:
                output = stdout.strip().split('\n') if stdout.strip() else []
                if output:
                    self.logger.debug(f"Sortie de la commande : {output}")
                return True, output
            else:
                error = stderr.strip().split('\n') if stderr.strip() else []
                if error:
                    self.logger.error(f"Erreur de la commande : {error} sur la commande : {command}")
                return False, error

        except Exception as e:
            self.logger.error(f"Exception lors de l'exécution de la commande : {str(e)}")
            return False, [str(e)]

    async def execute_python_file(self, file_path: str) -> tuple[bool, List[str]]:
        """
        Exécute un fichier Python dans l'environnement virtuel.

        Args:
            file_path: Chemin du fichier à exécuter

        Returns:
            tuple[bool, List[str]]: (succès, sortie)
        """
        try:
            command = f"{self.python_executable} {file_path}"
            success, output = await self.send_bash_command(command)
            return success, output
        except Exception as e:
            return False, [str(e)]

if __name__ == "__main__":
    executor = VenvCommandExecutor(venv_path="/home/yopla/PycharmProjects/Autonome_agent/test d environnement/.venv", working_dir="/home/yopla/PycharmProjects/Autonome_agent/test d environnement")
    #executor.start_env()
    executor.create_env()
    #executor.close()
