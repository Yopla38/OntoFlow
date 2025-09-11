"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# src/factory/agent_factory.py
import glob
import importlib
import logging
import mimetypes
import os
from pathlib import Path
import json
from typing import Dict, Any, List, Optional

from agent.src.agent import Agent
from agent.src.components.tool import Tool
from agent.src.providers.llm_providers import OpenAIProvider, AnthropicProvider
from agent.src.providers.memory_providers import MongoDBMemory, LocalFileMemory, SQLiteMemory
from agent.src.types.enums import AgentRole
from agent.src.types.interfaces import LLMProvider, MemoryProvider


class AgentFactory:
    @staticmethod
    def load_descriptor(path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            descriptor = json.load(f)

        # Valider la version du descripteur
        if descriptor.get('version') != '1.0':
            raise ValueError("Unsupported descriptor version")

        return descriptor['agent']

    @staticmethod
    def create_from_descriptor(descriptor_path: str) -> Agent:
        descriptor = AgentFactory.load_descriptor(descriptor_path)

        # Créer le provider LLM
        llm_provider = AgentFactory._create_llm_provider(descriptor['llm'])

        # Créer le provider de mémoire
        memory_provider = AgentFactory._create_memory_provider(descriptor['memory'])

        # Créer l'agent
        agent = Agent(
            name=descriptor['name'],
            role=AgentRole[descriptor['role']],
            llm_provider=llm_provider,
            memory_provider=memory_provider,
            system_prompt=descriptor['system']['prompt']
        )

        # Configurer les outils
        AgentFactory._configure_tools(agent, descriptor['tools'])

        # Configurer la base de connaissances
        if descriptor.get('knowledge_base', {}).get('enabled', False):
            AgentFactory._configure_knowledge_base(agent, descriptor['knowledge_base'])

        # Configurer les contraintes
        agent.constraints = descriptor.get('constraints', {})

        # Configurer le monitoring
        if descriptor.get('monitoring', {}).get('enabled', False):
            AgentFactory._configure_monitoring(agent, descriptor['monitoring'])

        return agent

    @staticmethod
    def _create_llm_provider(llm_config: Dict[str, Any]) -> LLMProvider:
        provider_type = llm_config['provider']
        config = llm_config['config']

        # Résoudre les variables d'environnement
        for key, value in config.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                config[key] = os.environ.get(env_var)

        if provider_type == 'openai':
            return OpenAIProvider(
                model=config['model'],
                api_key=config['api_key']
            )
        elif provider_type == 'anthropic':
            return AnthropicProvider(
                model=config['model'],
                api_key=config['api_key']
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_type}")

    @staticmethod
    def _create_memory_provider(memory_config: Dict[str, Any]) -> MemoryProvider:
        provider_type = memory_config['provider']
        config = memory_config['config']

        if provider_type == 'mongodb':
            return MongoDBMemory(
                connection_string=config['connection_string'],
                database=config['database'],
                collection=config['collection']
            )
        elif provider_type == 'local':
            return LocalFileMemory(
                file_path=config['file_path']
            )
        elif provider_type == 'sqlite':
            return SQLiteMemory(
                db_path=config['db_path']
            )
        else:
            raise ValueError(f"Unsupported memory provider: {provider_type}")

    @staticmethod
    def _configure_tools(agent: Agent, tools_config: List[Dict[str, Any]]):
        for tool_config in tools_config:
            if tool_config['type'] == 'function':
                module = importlib.import_module(tool_config['module'])
                function = getattr(module, tool_config['function'])

                tool = Tool(
                    name=tool_config['name'],
                    description=tool_config['description'],
                    function=function,
                    required_params=tool_config.get('parameters', {})
                )
                agent.add_tool(tool)

    @staticmethod
    def _configure_knowledge_base(agent: Agent, kb_config: Dict[str, Any]):
        for source in kb_config['sources']:
            if source['type'] == 'file':
                files = glob.glob(source['path'])
                for file in files:
                    # Logique pour charger le contenu du fichier
                    content = AgentFactory._load_file_content(file)
                    agent.knowledge_base.add_knowledge(
                        category=Path(file).stem,
                        content=content
                    )

    @staticmethod
    def _load_file_content(file_path: str) -> Optional[str]:
        """
        Charge le contenu d'un fichier en fonction de son type MIME.
        Supporte différents formats de fichiers courants.

        Args:
            file_path (str): Chemin vers le fichier à charger

        Returns:
            Optional[str]: Contenu du fichier ou None si le format n'est pas supporté
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logging.error(f"File not found: {file_path}")
                return None

            # Déterminer le type MIME du fichier
            mime_type, _ = mimetypes.guess_type(str(file_path))

            if mime_type is None:
                # Essayer de deviner à partir de l'extension
                mime_type = {
                    '.txt': 'text/plain',
                    '.md': 'text/markdown',
                    '.json': 'application/json',
                    '.yaml': 'application/x-yaml',
                    '.yml': 'application/x-yaml',
                }.get(file_path.suffix.lower())

            if mime_type is None:
                logging.warning(f"Unknown file type for: {file_path}")
                return None

            # Texte brut et formats similaires
            if mime_type.startswith('text/'):
                return file_path.read_text(encoding='utf-8')

            # PDF
            elif mime_type == 'application/pdf':
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        return '\n'.join(page.extract_text() for page in reader.pages)
                except ImportError:
                    logging.error("PyPDF2 is required for PDF processing")
                    return None

            # Documents Word
            elif mime_type in ['application/msword',
                               'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                try:
                    import docx
                    doc = docx.Document(file_path)
                    return '\n'.join(paragraph.text for paragraph in doc.paragraphs)
                except ImportError:
                    logging.error("python-docx is required for Word document processing")
                    return None

            # JSON
            elif mime_type == 'application/json':
                import json
                return json.loads(file_path.read_text(encoding='utf-8'))

            # YAML
            elif mime_type in ['application/x-yaml', 'application/yaml']:
                try:
                    import yaml
                    return yaml.safe_load(file_path.read_text(encoding='utf-8'))
                except ImportError:
                    logging.error("PyYAML is required for YAML processing")
                    return None

            # CSV
            elif mime_type == 'text/csv':
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    return df.to_string()
                except ImportError:
                    logging.error("pandas is required for CSV processing")
                    return None

            # Excel
            elif mime_type in ['application/vnd.ms-excel',
                               'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
                try:
                    import pandas as pd
                    df = pd.read_excel(file_path)
                    return df.to_string()
                except ImportError:
                    logging.error("pandas and openpyxl are required for Excel processing")
                    return None

            # Images (extraction de texte via OCR)
            elif mime_type.startswith('image/'):
                try:
                    import pytesseract
                    from PIL import Image
                    image = Image.open(file_path)
                    return pytesseract.image_to_string(image)
                except ImportError:
                    logging.error("pytesseract and Pillow are required for image processing")
                    return None

            else:
                logging.warning(f"Unsupported MIME type: {mime_type}")
                return None

        except Exception as e:
            logging.error(f"Error loading file {file_path}: {str(e)}")
            return None

    @staticmethod
    def _get_file_chunk_iterator(content: str, chunk_size: int = 1000):
        """
        Découpe le contenu en morceaux de taille gérable.

        Args:
            content (str): Contenu à découper
            chunk_size (int): Taille maximale de chaque morceau

        Yields:
            str: Morceaux de contenu
        """
        words = content.split()
        current_chunk = []
        current_size = 0

        for word in words:
            word_size = len(word) + 1  # +1 pour l'espace
            if current_size + word_size > chunk_size:
                yield ' '.join(current_chunk)
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size

        if current_chunk:
            yield ' '.join(current_chunk)

    @staticmethod
    async def _process_file_content(agent: Agent, file_path: str, category: str):
        """
        Traite le contenu d'un fichier et l'ajoute à la base de connaissances.

        Args:
            agent (Agent): Instance de l'agent
            file_path (str): Chemin vers le fichier
            category (str): Catégorie de la connaissance
        """
        content = AgentFactory._load_file_content(file_path)
        if content is None:
            return

        # Pour les grands fichiers, traiter par morceaux
        for chunk in AgentFactory._get_file_chunk_iterator(content):
            await agent.knowledge_base.add_knowledge(category, chunk)

    @staticmethod
    def _configure_knowledge_base(agent: Agent, kb_config: Dict[str, Any]):
        """
        Configure la base de connaissances de l'agent avec les sources spécifiées.

        Args:
            agent (Agent): Instance de l'agent
            kb_config (Dict[str, Any]): Configuration de la base de connaissances
        """

        async def process_sources():
            for source in kb_config['sources']:
                if source['type'] == 'file':
                    files = glob.glob(source['path'])
                    for file in files:
                        category = kb_config.get('categories', ['default'])[0]
                        await AgentFactory._process_file_content(agent, file, category)
                elif source['type'] == 'url':
                    # Implémentation du chargement depuis URL
                    pass

        # Exécuter le traitement asynchrone
        import asyncio
        asyncio.create_task(process_sources())