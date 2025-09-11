"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# src/interface/html_manager.py
import shutil
from datetime import datetime
from pathlib import Path
import os
from typing import Optional, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape


class HTMLManager:
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.template_dir = self.workspace_root / "templates"
        self.projects_dir = self.workspace_root / "projects"

        # Création du dossier templates s'il n'existe pas
        self.template_dir.mkdir(parents=True, exist_ok=True)

        # Copie du template par défaut s'il n'existe pas
        self.default_template = self.template_dir / "template.html"
        if not self.default_template.exists():
            # Chemin vers le template source dans le package
            source_template = Path(__file__).parent.parent / "templates" / "template.html"
            if source_template.exists():
                shutil.copy(source_template, self.default_template)
            else:
                raise FileNotFoundError(f"Template source non trouvé: {source_template}")


        # Initialisation de Jinja2
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(['html', 'xml'])
        )

        # Ajout des filtres personnalisés
        self.env.filters['lower_keys'] = self._lower_keys

    @staticmethod
    def _lower_keys(d):
        """Convertit toutes les clés d'un dictionnaire en minuscules"""
        if not isinstance(d, dict):
            return d
        return {k.lower(): v for k, v in d.items()}

    def get_project_html_path(self, project_name: str, agent_name: str) -> Path:
        """Crée et retourne le chemin pour les fichiers HTML d'un agent dans un projet"""
        html_path = self.projects_dir / project_name / "logs" / "html" / agent_name
        html_path.mkdir(parents=True, exist_ok=True)
        return html_path

    def generate_html(
            self,
            instance: Any,
            project_name: str,
            agent_name: str,
            prompt: str = None,
            input_message: str = None,
            template_name: str = 'template.html'
    ) -> str:
        """
        Génère une page HTML pour une interaction d'agent

        Args:
            instance: Données à afficher
            project_name: Nom du projet
            agent_name: Nom de l'agent
            prompt: Prompt système de l'agent
            input_message: Message d'entrée reçu par l'agent
            template_name: Nom du template à utiliser
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{agent_name}_{timestamp}.html"

        html_path = self.get_project_html_path(project_name, agent_name)
        output_file = html_path / filename

        # Génération du HTML
        template = self.env.get_template(template_name)
        data = instance.model_dump() if hasattr(instance, 'model_dump') else instance

        html_content = template.render(
            data=data,
            agent_name=agent_name,
            project_name=project_name,
            timestamp=timestamp,
            system_prompt=prompt,
            input_message=input_message,
            editable=self._has_editable_field(data)
        )

        # Sauvegarde du fichier
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(output_file)

    @staticmethod
    def _has_editable_field(data: Any) -> bool:
        """Vérifie si les données contiennent des champs éditables"""
        if isinstance(data, dict):
            if data.get('editable'):
                return True
            return any(HTMLManager._has_editable_field(v) for v in data.values())
        elif isinstance(data, list):
            return any(HTMLManager._has_editable_field(item) for item in data)
        return False