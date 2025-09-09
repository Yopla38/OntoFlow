"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# Integration_fortran_RAG.py
import os
import hashlib
import json
import logging
import shutil
import uuid
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import asyncio

from .utils.clean_fortran_code_v2 import nettoyer_commentaires_fortran
from .context_provider.query_router import IntelligentQueryRouter
from .fortran_analysis.providers.Fortran_agent import FortranAgent
from .fortran_analysis.providers.consult import FortranEntityExplorer
from .main_app import display_intelligent_hierarchical_result
from .semantic_analysis.core.semantic_chunker import ConceptualHierarchicalEngine, \
    LevelBasedSearchEngine
from .CONSTANT import API_KEY_PATH, CHUNK_SIZE, CHUNK_OVERLAP, ONTOLOGY_PATH_TTL, MAX_CONCURRENT, MAX_RESULTS, \
    STORAGE_DIR, FORTRAN_AGENT_NB_STEP
from .ontology.classifier import OntologyClassifier
from .ontology.ontology_manager import OntologyManager
#from utils.f2p_fortran_semantic_chunker import F2pyFortranSemanticChunker
from .utils.ontologie_fortran_chunker import OFPFortranSemanticChunker as F2pyFortranSemanticChunker
from .utils.semantic_chunker import SemanticChunker
from .utils.document_processor import DocumentProcessor

# Imports pour le RAG
from .utils.rag_engine import RAGEngine
from .provider.llm_providers import OpenAIProvider
from .provider.get_key import get_openai_key


class OntoDocumentProcessor(DocumentProcessor):
    """
    Processeur de documents personnalisé qui utilise les chunkers appropriés
    selon le type de fichier et bypass le MarkdownConverter pour certains types
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, file_extensions: dict = None):
        super().__init__(chunk_size, chunk_overlap, use_semantic_chunking=True)
        self.file_extensions = file_extensions or {}

        # Initialiser les chunkers spécialisés
        self.fortran_processor = None  # Sera initialisé plus tard

        from .semantic_analysis.core.semantic_chunker import HierarchicalSemanticChunker
        self.hierarchical_chunker = HierarchicalSemanticChunker(
            min_chunk_size=200,
            max_chunk_size=chunk_size,
            overlap_sentences=2,
            ontology_manager=None,
            concept_classifier=None
        )

        self.ontology_manager = None

        # Types de fichiers qui doivent être lus directement (sans conversion)
        self.direct_read_types = {'fortran', 'code', 'text', 'config'}

    def set_ontology_components(self, ontology_manager, concept_classifier):
        """Configure les composants ontologiques"""
        self.ontology_manager = ontology_manager
        self.hierarchical_chunker.set_ontology_components(ontology_manager, concept_classifier)
        if hasattr(self, 'semantic_chunker') and self.semantic_chunker:
            self.semantic_chunker.set_ontology_manager(ontology_manager, concept_classifier)
        print("✅ Composants ontologiques configurés dans le processeur")

    async def initialize_fortran_module(self, document_store, rag_engine=None):
        """
        Initialise le module Fortran après que document_store soit disponible.
        """
        from .fortran_analysis.integration.document_processor_integration import get_fortran_processor

        if hasattr(document_store, 'initialize'):
            await document_store.initialize()

        self.fortran_processor = await get_fortran_processor(
            document_store, rag_engine, self.ontology_manager
        )
        print("✅ Module Fortran initialisé")

    async def _extract_text_with_metadata(self, filepath: str) -> Tuple[str, Dict[str, Any]]:

        extension = Path(filepath).suffix.lower()
        file_type = self.file_extensions.get(extension, 'text')
        metadata = {
            'file_size': os.path.getsize(filepath),
            'file_type': file_type,
            'modification_date': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat(),
            'extraction_date': datetime.now().isoformat(),
        }
        if file_type in self.direct_read_types:
            print(f"📖 Lecture directe du fichier {file_type}: {Path(filepath).name}")
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return content, metadata
            except Exception as e:
                print(f"❌ Erreur lors de la lecture de {filepath}: {e}")
                raise
        else:
            print(f"🔄 Conversion via MarkdownConverter: {Path(filepath).name}")
            return await super()._extract_text_with_metadata(filepath)

    async def process_document(self, filepath: str, document_id: str = None, additional_metadata: dict = None, processing_options: dict = None):
        if document_id is None:
            document_id = str(uuid.uuid4())
        extension = Path(filepath).suffix.lower()
        file_type = self.file_extensions.get(extension, 'text')
        print(f"🔍 Type de fichier détecté: {file_type} pour {Path(filepath).name}")
        text_content, doc_metadata = await self._extract_text_with_metadata(filepath)
        metadata = {**doc_metadata, **(additional_metadata or {})}

        if file_type == 'fortran':

            print(f"✂️  Utilisation du chunker Fortran sémantique")
            if not self.fortran_processor:
                raise RuntimeError("Module Fortran non initialisé. Appelez initialize_fortran_module() d'abord.")
            chunks = await self.fortran_processor.process_fortran_document(
                filepath, document_id, text_content, metadata
            )
        elif file_type in ['text', 'markdown']:
            print(f"✂️  Utilisation du chunker sémantique pour texte")
            chunks = await self.hierarchical_chunker.create_semantic_chunks(
                text_content, document_id, filepath, metadata
            )
        else:
            print(f"✂️  Utilisation du chunker générique")
            chunks = self._create_chunks(text_content, document_id, filepath)
            for chunk in chunks:
                chunk['metadata'].update(metadata)

        print(f"✂️  {len(chunks)} chunks créés pour {Path(filepath).name}")
        return document_id, chunks


class OntoRAG:
    """
    RAG optimisé pour les grands ensembles de code avec reconnaissance automatique
    des types de fichiers et chunking sémantique adapté.
    """

    def __init__(
            self,
            storage_dir: str = "ontorag_storage",
            api_key_path: str = API_KEY_PATH,
            model: str = "gpt-4o",
            chunk_size: int = CHUNK_SIZE,
            chunk_overlap: int = CHUNK_OVERLAP,
            ontology_path: Optional[str] = None
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.api_key_path = api_key_path
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Métadonnées des documents
        self.metadata_file = self.storage_dir / "documents_metadata.json"
        self.documents_metadata = self._load_metadata()

        # Extensions supportées
        self.file_extensions = {
            # Code Fortran
            '.f90': 'fortran', '.f95': 'fortran', '.f03': 'fortran',
            '.f08': 'fortran', '.f': 'fortran', '.for': 'fortran', '.ftn': 'fortran',

            # Documentation
            '.md': 'markdown', '.rst': 'text', '.txt': 'text',

            # PDFs et documents (seront convertis)
            '.pdf': 'pdf', '.doc': 'document', '.docx': 'document',

            # Code source autres
            '.py': 'code', '.c': 'code', '.cpp': 'code',
            '.h': 'code', '.hpp': 'code', '.java': 'code',

            # Configs
            '.yaml': 'config', '.yml': 'config', '.json': 'config',
            '.xml': 'config', '.ini': 'config'
        }

        # RAG Engine sera initialisé de manière asynchrone
        self.rag_engine = None
        self.logger = logging.getLogger(__name__)

        # Ontologie
        self.ontology_path = ontology_path
        self.ontology_manager = None
        self.classifier = None

        # NOUVEAU : Système de contexte
        self.context_provider = None
        # Agent pour la recherche
        self.agent_fortran = None

    def cleanup(self):
        """
        Nettoie les ressources pour se conformer à l'interface de l'orchestrateur.
        """
        self.logger.info("Nettoyage des ressources OntoRAG...")
        # Actuellement, il n'y a pas de ressources persistantes à fermer (comme des connexions).
        # Si vous en ajoutez à l'avenir, c'est ici qu'il faudra les gérer.
        pass

    async def initialize(self):
        """Initialise le système RAG de manière asynchrone"""
        if self.rag_engine is not None:
            return

        print("🚀 Initialisation d'OntoRAG...")

        # Initialiser les providers
        openai_key = get_openai_key(api_key_path=self.api_key_path)
        llm_provider = OpenAIProvider(model=self.model, api_key=openai_key)

        # Créer notre processeur personnalisé AVANT l'initialisation
        self.custom_processor = OntoDocumentProcessor(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            file_extensions=self.file_extensions
        )

        if self.ontology_path and os.path.exists(self.ontology_path):
            print(f"📚 Chargement de l'ontologie: {os.path.basename(self.ontology_path)}")

            # Initialiser l'OntologyManager
            onto_storage_dir = str(self.storage_dir / "ontology")
            self.ontology_manager = OntologyManager(storage_dir=onto_storage_dir)

            # Charger l'ontologie TTL
            success = self.ontology_manager.load_ontology(self.ontology_path)
            if not success:
                print("⚠️ Échec du chargement de l'ontologie")
            else:
                print(
                    f"✅ Ontologie chargée: {len(self.ontology_manager.concepts)} concepts, {len(self.ontology_manager.relations)} relations")

                # Initialiser le classifieur ontologique
                classifier_dir = str(self.storage_dir / "classifier")
                self.classifier = OntologyClassifier(
                    rag_engine=None,  # ← Sera assigné après
                    ontology_manager=self.ontology_manager,
                    storage_dir=classifier_dir,
                    use_hierarchical=False,
                    enable_concept_classification=True,
                    enable_relation_learning=False,
                    multiscale_mode=False
                )
        else:
            print(f"⚠️ Échec du chargement de l'ontologie, le fichier {self.ontology_path} n'existe pas")

        # Initialiser le RAG Engine
        self.rag_engine = RAGEngine(
            llm_provider=llm_provider,
            embedding_provider=llm_provider,
            storage_dir=str(self.storage_dir / "rag_storage"),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        # IMPORTANT : Assigner le rag_engine au classifier AVANT l'initialisation
        if self.classifier:
            self.classifier.rag_engine = self.rag_engine
            # Assigner aussi au concept_classifier
            if hasattr(self.classifier, 'concept_classifier') and self.classifier.concept_classifier:
                self.classifier.concept_classifier.rag_engine = self.rag_engine
                print("✅ RAG engine assigné au concept_classifier")
                if hasattr(self.classifier.concept_classifier, 'classify_embedding_direct'):
                    print("✅ classify_embedding_direct disponible")
            # Assigner au classifier hierarchique/simple
            if hasattr(self.classifier, 'classifier') and self.classifier.classifier:
                self.classifier.classifier.rag_engine = self.rag_engine
                print("✅ RAG engine assigné au classifier hiérarchique")

            await self.classifier.initialize()
            print("✅ Classifieur ontologique initialisé")

        # Assigner le processeur au RAG Engine
        self.rag_engine.processor = self.custom_processor
        self.rag_engine.document_store.processor = self.custom_processor

        await self.rag_engine.initialize()

        # MAINTENANT initialiser le classifier avec le rag_engine disponible
        if self.classifier:
            await self.classifier.initialize()
            print("✅ Classifieur ontologique initialisé")

            # Lier le classifier à l'ontology_manager
            self.ontology_manager.classifier = self.classifier
            print("✅ Classifier lié à l'ontology_manager")

            if self.ontology_manager:
                self.ontology_manager.rag_engine = self.rag_engine
                print("✅ RAG engine lié à l'ontology_manager")

            self.custom_processor.set_ontology_components(self.ontology_manager, self.classifier)

            # ← ASSIGNMENT ICI - après que tout soit initialisé
            self.custom_processor.ontology_manager = self.ontology_manager
            print("✅ Ontology_manager assigné au processeur")

        # Initialiser le système de contexte
        await self.custom_processor.initialize_fortran_module(self.rag_engine.document_store, self.rag_engine)

        # Moteur de recherche par niveau
        self.level_search_engine = LevelBasedSearchEngine(
            self.rag_engine.document_store,
            self.rag_engine.embedding_manager.provider
        )

        self.entity_explorer = FortranEntityExplorer(self.custom_processor.fortran_processor.entity_manager,
                                                     self.ontology_manager)
        self.router = IntelligentQueryRouter(self.rag_engine.llm_provider, self.entity_explorer)

        #  Nous pouvons changer le provider pour l'agent si necessaire
        self.agent_fortran = FortranAgent(llm_provider=self.rag_engine.llm_provider, explorer=self.entity_explorer,
                                          max_steps=FORTRAN_AGENT_NB_STEP)

        print("✅ OntoRAG initialisé avec succès!")


    def scan_directory(
            self,
            directory_path: str,
            file_filters: List[str] = None,
            recursive: bool = True,
            exclude_patterns: List[str] = None,
            project_name: str = None,
            version: str = "1.0"
    ) -> List[Dict[str, Any]]:
        """
        Scanne un répertoire et retourne la liste des fichiers à traiter

        Args:
            directory_path: Chemin du répertoire à scanner
            file_filters: Extensions à inclure (ex: ['f90', 'md', 'txt'])
            recursive: Si True, scan récursif des sous-répertoires
            exclude_patterns: Patterns à exclure (ex: ['.git', '__pycache__', '.pyc'])
            project_name: Nom du projet (auto-détecté si None)
            version: Version du projet

        Returns:
            Liste de dictionnaires compatibles avec add_documents_batch
        """

        directory_path = Path(directory_path).resolve()

        if not directory_path.exists():
            print(f"❌ Répertoire non trouvé : {directory_path}")
            return []

        if not directory_path.is_dir():
            print(f"❌ Le chemin n'est pas un répertoire : {directory_path}")
            return []

        # Filtres par défaut
        if file_filters is None:
            file_filters = ['f90', 'f95']
            """
            'f03', 'f08', 'f', 'for', 'ftn',  # Fortran
                            'md', 'rst', 'txt',  # Documentation
                            'py', 'c', 'cpp', 'h', 'hpp',  # Autres codes
                            'yaml', 'yml', 'json', 'xml']  # Configs
            """

        # Exclusions par défaut
        if exclude_patterns is None:
            exclude_patterns = [
                '.git', '.svn', '.hg',  # Contrôle de version
                '__pycache__', '.pytest_cache', 'node_modules',  # Cache
                '.vscode', '.idea', '.DS_Store',  # IDE/Système
                'build', 'dist', 'target', 'out',  # Build
                '.pyc', '.pyo', '.pyd', '.so', '.dll',  # Binaires
                'CMakeFiles', 'CMakeCache.txt'  # CMake

            ]

        # Auto-détecter le nom du projet si pas fourni
        if project_name is None:
            project_name = directory_path.name

        print(f"📁 Scan du répertoire : {directory_path}")
        print(f"   Filtres : {file_filters}")
        print(f"   Récursif : {recursive}")
        print(f"   Projet : {project_name}")

        found_files = []

        # Fonction de scan
        def scan_folder(folder_path: Path, current_depth: int = 0):
            try:
                for item in folder_path.iterdir():

                    # Vérifier les exclusions
                    if any(pattern in str(item) for pattern in exclude_patterns):
                        continue

                    if item.is_file():
                        # Vérifier l'extension
                        extension = item.suffix.lower().lstrip('.')
                        if extension in file_filters:
                            # Calculer le chemin relatif pour les métadonnées
                            relative_path = item.relative_to(directory_path)

                            file_info = {
                                "filepath": str(item),
                                "project_name": project_name,
                                "version": version,
                                "additional_metadata": {
                                    "relative_path": str(relative_path),
                                    "directory_depth": current_depth,
                                    "parent_directory": str(item.parent.name),
                                    "scanned_from": str(directory_path),
                                    "file_size": item.stat().st_size,
                                    "scan_method": "directory_scan"
                                }
                            }
                            found_files.append(file_info)

                    elif item.is_dir() and recursive:
                        # Scan récursif des sous-répertoires
                        scan_folder(item, current_depth + 1)

            except PermissionError:
                print(f"⚠️ Accès refusé : {folder_path}")
            except Exception as e:
                print(f"⚠️ Erreur scan {folder_path} : {e}")

        # Lancer le scan
        scan_folder(directory_path)

        # Statistiques
        file_stats = {}
        for file_info in found_files:
            ext = Path(file_info["filepath"]).suffix.lower().lstrip('.')
            file_stats[ext] = file_stats.get(ext, 0) + 1

        print(f"✅ Scan terminé : {len(found_files)} fichiers trouvés")
        for ext, count in sorted(file_stats.items()):
            print(f"   .{ext}: {count} fichiers")

        return found_files

    async def generate_global_summary(self, scope: str = 'all') -> Dict[str, Any]:
        """Génère un résumé global de tous les documents"""

        if not await self._ensure_initialized():
            return {"error": "RAG non initialisé"}

        # 1. Collecter les métadonnées de tous les documents
        all_docs = self.list_documents()

        # 2. Filtrer selon le scope
        filtered_docs = self._filter_documents_by_scope(all_docs, scope)

        if not filtered_docs:
            return {"error": f"Aucun document trouvé pour le scope '{scope}'"}

        print(f"📊 Analyse de {len(filtered_docs)} documents...")

        # 3. Analyser les concepts globaux
        global_concepts = await self._analyze_global_concepts(filtered_docs)

        # 4. Identifier les thèmes principaux
        themes = self._identify_main_themes(filtered_docs, global_concepts)

        # 5. Générer le résumé textuel
        summary_text = await self._generate_summary_text(filtered_docs, global_concepts, themes)

        # 6. Compiler les statistiques
        statistics = self._compile_global_statistics(filtered_docs, global_concepts)

        return {
            "summary": summary_text,
            "main_concepts": global_concepts[:20],  # Top 20
            "themes": themes,
            "statistics": statistics,
            "documents_analyzed": len(filtered_docs)
        }

    def _filter_documents_by_scope(self, docs: List[Dict], scope: str) -> List[Dict]:
        """Filtre les documents selon le scope"""

        if scope == 'all':
            return docs

        if scope.startswith('project:'):
            project_name = scope.split(':', 1)[1]
            return [doc for doc in docs if doc.get('project', '').lower() == project_name.lower()]

        if scope.startswith('type:'):
            file_type = scope.split(':', 1)[1]
            return [doc for doc in docs if doc.get('file_type', '').lower() == file_type.lower()]

        # Scope comme nom de projet par défaut
        return [doc for doc in docs if scope.lower() in doc.get('project', '').lower()]

    async def _analyze_global_concepts(self, documents: List[Dict]) -> List[Dict[str, Any]]:
        """Analyse les concepts dans tous les documents"""

        concept_frequency = {}  # concept_uri -> {count, total_confidence, labels}

        for doc in documents:
            doc_concepts = doc.get('ontology_concepts', [])

            # Aussi récupérer les concepts des chunks si disponible
            doc_id = doc.get('document_id')
            if doc_id and doc_id in self.rag_engine.document_store.document_chunks:
                chunks = self.rag_engine.document_store.document_chunks[doc_id]
                for chunk in chunks:
                    detected_concepts = chunk.get('metadata', {}).get('detected_concepts', [])
                    for concept in detected_concepts:
                        uri = concept.get('concept_uri', '')
                        label = concept.get('label', '')
                        confidence = concept.get('confidence', 0)

                        if uri:
                            if uri not in concept_frequency:
                                concept_frequency[uri] = {
                                    'count': 0,
                                    'total_confidence': 0,
                                    'labels': set(),
                                    'documents': set()
                                }

                            concept_frequency[uri]['count'] += 1
                            concept_frequency[uri]['total_confidence'] += confidence
                            concept_frequency[uri]['labels'].add(label)
                            concept_frequency[uri]['documents'].add(doc.get('filename', 'Unknown'))

        # Convertir en liste triée
        global_concepts = []
        for uri, stats in concept_frequency.items():
            if stats['count'] > 0:  # Au moins une occurrence
                avg_confidence = stats['total_confidence'] / stats['count']
                primary_label = list(stats['labels'])[0] if stats['labels'] else uri.split('#')[-1]

                global_concepts.append({
                    'concept_uri': uri,
                    'label': primary_label,
                    'frequency': stats['count'],
                    'avg_confidence': avg_confidence,
                    'document_count': len(stats['documents']),
                    'documents': list(stats['documents']),
                    'importance_score': stats['count'] * avg_confidence * len(stats['documents'])
                })

        # Trier par score d'importance
        global_concepts.sort(key=lambda x: x['importance_score'], reverse=True)
        return global_concepts

    def _identify_main_themes(self, documents: List[Dict], concepts: List[Dict]) -> Dict[str, List[str]]:
        """Identifie les thèmes principaux"""

        themes = {}

        # Thèmes par type de fichier
        file_types = {}
        for doc in documents:
            file_type = doc.get('file_type', 'unknown')
            if file_type not in file_types:
                file_types[file_type] = []
            file_types[file_type].append(doc.get('filename', 'Unknown'))

        for file_type, files in file_types.items():
            if len(files) > 1:  # Au moins 2 fichiers
                themes[f"Documents {file_type}"] = files

        # Thèmes par concepts dominants
        for concept in concepts[:5]:  # Top 5 concepts
            if concept['document_count'] > 1:  # Dans plusieurs documents
                theme_name = f"Thème: {concept['label']}"
                themes[theme_name] = concept['documents']

        # Thèmes par projet
        projects = {}
        for doc in documents:
            project = doc.get('project', 'Unknown')
            if project not in projects:
                projects[project] = []
            projects[project].append(doc.get('filename', 'Unknown'))

        for project, files in projects.items():
            if len(files) > 1 and project != 'Unknown':
                themes[f"Projet: {project}"] = files

        return themes

    async def _generate_summary_text(self, documents: List[Dict], concepts: List[Dict], themes: Dict) -> str:
        """Génère le texte de résumé"""

        # Créer un prompt pour le LLM
        doc_list = "\n".join(
            [f"- {doc.get('filename', 'Unknown')} ({doc.get('file_type', 'unknown')})" for doc in documents])

        main_concepts_text = "\n".join(
            [f"- {c['label']} (présent dans {c['document_count']} documents)" for c in concepts[:10]])

        themes_text = "\n".join([f"- {theme}: {len(files)} documents" for theme, files in themes.items()])

        prompt = f"""Génère un résumé exécutif des documents suivants :

    DOCUMENTS ANALYSÉS ({len(documents)} au total):
    {doc_list}

    CONCEPTS PRINCIPAUX DÉTECTÉS:
    {main_concepts_text}

    THÈMES IDENTIFIÉS:
    {themes_text}

    Crée un résumé structuré qui inclut :
    1. Vue d'ensemble du contenu
    2. Domaines scientifiques/techniques couverts
    3. Types de documents et leur répartition
    4. Concepts clés et leur importance
    5. Connections entre les différents documents

    Sois concis mais informatif, en 3-4 paragraphes maximum."""

        try:
            # Utiliser le LLM pour générer le résumé
            summary = await self.rag_engine.llm_provider.generate_response(prompt)
            return summary
        except Exception as e:
            print(f"⚠️ Erreur génération LLM: {e}")

            # Fallback : résumé automatique simple
            return self._generate_simple_summary(documents, concepts, themes)

    def _generate_simple_summary(self, documents: List[Dict], concepts: List[Dict], themes: Dict) -> str:
        """Génère un résumé simple en fallback"""

        file_types = {}
        projects = set()

        for doc in documents:
            file_type = doc.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
            projects.add(doc.get('project', 'Unknown'))

        summary = f"""RÉSUMÉ DES DOCUMENTS

    📊 STATISTIQUES:
    - {len(documents)} documents analysés
    - {len(concepts)} concepts uniques détectés
    - {len(projects)} projets représentés
    - Types de fichiers: {', '.join([f"{t}({n})" for t, n in file_types.items()])}

    🧠 CONCEPTS PRINCIPAUX:
    {chr(10).join([f"• {c['label']} - présent dans {c['document_count']} documents" for c in concepts[:5]])}

    🎯 THÈMES IDENTIFIÉS:
    {chr(10).join([f"• {theme}: {len(files)} documents" for theme, files in list(themes.items())[:5]])}

    Cette collection de documents couvre principalement {', '.join(list(projects))} avec un focus sur les concepts de {', '.join([c['label'] for c in concepts[:3]])}."""

        return summary

    def _compile_global_statistics(self, documents: List[Dict], concepts: List[Dict]) -> Dict[str, Any]:
        """Compile les statistiques globales"""

        file_types = {}
        projects = {}
        total_chunks = 0

        for doc in documents:
            # Types de fichiers
            file_type = doc.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1

            # Projets
            project = doc.get('project', 'Unknown')
            projects[project] = projects.get(project, 0) + 1

            # Chunks
            total_chunks += doc.get('total_chunks', 0)

        # Domaines des concepts
        domains = set()
        for concept in concepts:
            uri = concept['concept_uri']
            if '#' in uri:
                domain = uri.rsplit('#', 1)[0].split('/')[-1]
                domains.add(domain)

        return {
            'total_documents': len(documents),
            'total_chunks': total_chunks,
            'file_types': file_types,
            'projects': projects,
            'domains': list(domains),
            'main_concepts': [c['label'] for c in concepts[:10]],
            'avg_concepts_per_doc': len(concepts) / len(documents) if documents else 0
        }

    async def hierarchical_query(self, query: str, max_per_level: int = 3, mode: str = 'auto') -> Dict[str, Any]:
        """Requête hiérarchique intelligente avec auto-détection de contenu"""

        # print(f"🔍 Requête hiérarchique intelligente: {query}")

        # Créer le moteur unifié
        entity_manager = None
        if (hasattr(self.custom_processor, 'fortran_processor') and
                self.custom_processor.fortran_processor and
                hasattr(self.custom_processor.fortran_processor, 'entity_manager')):
            entity_manager = self.custom_processor.fortran_processor.entity_manager

        conceptual_engine = ConceptualHierarchicalEngine(
            self.rag_engine.document_store,
            self.rag_engine.embedding_manager.provider,
            entity_manager
        )

        # Recherche intelligente
        conceptual_results = await conceptual_engine.intelligent_hierarchical_search(
            query, max_per_level, mode
        )

        # Préparer les passages pour la génération de réponse
        all_passages = []
        source_descriptions = []
        source_index = 0

        hierarchical_results = conceptual_results.get('hierarchical_results', {})
        search_mode = conceptual_results.get('search_mode', 'unified')

        for conceptual_level, level_data in hierarchical_results.items():
            results = level_data['results']
            display_name = level_data['display_name']

            for result in results:
                source_index += 1

                # Adapter selon le type de résultat
                if 'entity' in result:
                    # Résultat Fortran
                    entity = result['entity']
                    source_desc = (
                        f"[Source {source_index}] {entity.filename} "
                        f"(lignes {entity.start_line}-{entity.end_line}) "
                        f"- {entity.entity_type}: {entity.entity_name}"
                    )

                    # Récupérer le texte de l'entité
                    text_content = await self._get_entity_text_content(entity)

                    passage = {
                        'text': text_content,
                        'similarity': result['similarity'],
                        'level': conceptual_level,
                        'source_index': source_index,
                        'source_info': result['source_info'],
                        'source_description': source_desc,
                        'content_type': 'fortran'
                    }

                else:
                    # Résultat texte
                    chunk = result['chunk']
                    source_info = result['source_info']
                    source_desc = (
                        f"[Source {source_index}] {source_info['filename']} "
                        f"(lignes {source_info['start_line']}-{source_info['end_line']}) "
                        f"- {source_info['section_title']}"
                    )

                    passage = {
                        'text': chunk['text'],
                        'similarity': result['similarity'],
                        'level': conceptual_level,
                        'source_index': source_index,
                        'source_info': source_info,
                        'source_description': source_desc,
                        'content_type': source_info.get('content_type', 'text')
                    }

                source_descriptions.append(source_desc)
                all_passages.append(passage)

        # Générer le prompt système adapté
        system_prompt = f"""Tu es un assistant expert qui répond aux questions en citant TOUJOURS ses sources.

    Mode de recherche utilisé: {search_mode}

    Tu as accès aux sources suivantes:
    {chr(10).join(source_descriptions)}

    INSTRUCTIONS IMPORTANTES:
    1. Réponds à la question en utilisant uniquement les informations fournies
    2. Cite OBLIGATOIREMENT tes sources en utilisant [Source N] dans ta réponse
    3. Adapte ton langage selon le type de contenu (technique pour Fortran, descriptif pour texte)
    4. Structure ta réponse selon la hiérarchie trouvée
    5. N'invente aucune information

    Exemple de citation: "D'après [Source 1] dans le module my_module (lignes 45-67)..."
    """

        # Générer la réponse
        answer = await self.rag_engine.generate_answer(query, all_passages, system_prompt)

        return {
            "answer": answer,
            "search_mode": search_mode,
            "hierarchical_results": hierarchical_results,
            "total_passages": len(all_passages),
            "conceptual_levels_found": list(hierarchical_results.keys())
        }

    async def _get_entity_text_content(self, entity) -> str:
        """Récupère le contenu textuel d'une entité Fortran"""
        try:
            if entity.chunks:
                # Concaténer le texte de tous les chunks
                text_parts = []
                for chunk_info in entity.chunks:
                    chunk_id = chunk_info.get('chunk_id')
                    if chunk_id:
                        chunk = await self._get_chunk_by_id(chunk_id)
                        if chunk:
                            text_parts.append(chunk.get('text', ''))

                return '\n'.join(text_parts) if text_parts else f"Entité {entity.entity_name} (contenu non disponible)"
            else:
                return f"Entité {entity.entity_name} de type {entity.entity_type}"
        except Exception as e:
            return f"Entité {entity.entity_name} (erreur récupération: {e})"

    async def _get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Récupère un chunk par son ID"""
        for doc_id, chunks in self.rag_engine.document_store.document_chunks.items():
            for chunk in chunks:
                if chunk['id'] == chunk_id:
                    return chunk
        return None

    async def diagnose_fortran_entity_manager(self) -> Dict[str, Any]:
        """Diagnostic de l'EntityManager Fortran"""
        try:
            return await self.rag_engine.diagnose_fortran_entity_manager()

        except Exception as e:
            return {"error": f"Erreur diagnostic: {str(e)}"}

    async def get_fortran_context(self, entity_name: str,
                                  agent_type: str = "developer",
                                  task_context: str = "code_understanding",
                                  context_type: str = "smart") -> Dict[str, Any]:
        """
        Récupère le contexte Fortran pour une entité via le module d'analyse.
        """
        try:
            # Accéder au processeur Fortran via le document processor
            fortran_processor = self.custom_processor.fortran_processor

            if not fortran_processor:
                return {"error": "Module Fortran non initialisé"}

            if context_type == "smart":
                # Utiliser l'orchestrateur intelligent
                return await fortran_processor.get_entity_context(entity_name, "smart")
            else:
                # Contexte spécifique
                return await fortran_processor.get_entity_context(entity_name, context_type)

        except Exception as e:
            return {"error": f"Erreur récupération contexte: {str(e)}"}

    async def search_fortran_entities(self, query: str) -> List[Dict[str, Any]]:
        """Recherche d'entités Fortran"""
        try:
            fortran_processor = self.custom_processor.fortran_processor
            if not fortran_processor:
                return []

            return await fortran_processor.search_entities(query)

        except Exception as e:
            print(f"Erreur recherche entités: {e}")
            return []

    async def generate_dependency_visualization(self,
                                                output_file: str = "dependencies.html",
                                                max_entities: int = 50,
                                                include_variables: bool = True,
                                                focus_entity: str = None) -> Optional[str]:
        """Génère une visualisation des dépendances"""
        try:
            from fortran_analysis.output.graph_visualizer import create_dependency_visualization_from_parser

            # Récupérer toutes les entités
            fortran_processor = self.custom_processor.fortran_processor
            if not fortran_processor or not fortran_processor.entity_manager:
                return None

            # Récupérer les entités depuis EntityManager
            all_entities = list(fortran_processor.entity_manager.entities.values())

            if not all_entities:
                print("Aucune entité trouvée pour la visualisation")
                return None

            # Générer la visualisation
            html_file = create_dependency_visualization_from_parser(
                entities=all_entities,
                output_file=output_file,
                focus_entity=focus_entity,
                include_variables=include_variables,
                max_entities=max_entities
            )

            return html_file

        except Exception as e:
            print(f"Erreur génération visualisation: {e}")
            return None

    async def refresh_fortran_index(self):
        """Réindexe le module Fortran"""
        try:
            fortran_processor = self.custom_processor.fortran_processor
            if fortran_processor and fortran_processor.entity_manager:
                await fortran_processor.entity_manager.rebuild_index()

        except Exception as e:
            print(f"Erreur réindexation: {e}")
            raise

    async def get_fortran_stats(self) -> Dict[str, Any]:
        """Statistiques du module Fortran"""
        try:
            fortran_processor = self.custom_processor.fortran_processor
            if not fortran_processor:
                return {"error": "Module Fortran non initialisé"}

            return fortran_processor.get_stats()

        except Exception as e:
            return {"error": f"Erreur récupération stats: {str(e)}"}

    async def get_text_generator(self):
        """Récupère ou crée le générateur de texte contextuel"""
        if not hasattr(self, 'text_generator') or self.text_generator is None:
            from fortran_analysis.output.text_generator import ContextualTextGenerator

            self.text_generator = ContextualTextGenerator(
                self.rag_engine.document_store,
                self.rag_engine
            )
            await self.text_generator.initialize()

        return self.text_generator

    async def generate_contextual_text(self,
                                       element_name: str,
                                       context_type: str = "complete",
                                       agent_perspective: str = "developer",
                                       task_context: str = "code_understanding",
                                       format_style: str = "detailed") -> str:
        """Interface pour générer du texte contextuel"""
        try:
            text_generator = await self.get_text_generator()

            return await text_generator.get_contextual_text(
                element_name=element_name,
                context_type=context_type,
                agent_perspective=agent_perspective,
                task_context=task_context,
                format_style=format_style
            )

        except Exception as e:
            return f"❌ Erreur génération texte: {str(e)}"

    # Méthodes de convenance
    async def get_quick_text(self, element_name: str) -> str:
        """Texte rapide en bullet points"""
        text_generator = await self.get_text_generator()
        return await text_generator.get_quick_context(element_name)

    async def get_full_text(self, element_name: str) -> str:
        """Texte complet détaillé"""
        text_generator = await self.get_text_generator()
        return await text_generator.get_full_context(element_name)

    async def get_developer_text(self, element_name: str, task: str = "code_understanding") -> str:
        """Texte optimisé développeur"""
        text_generator = await self.get_text_generator()
        return await text_generator.get_developer_context(element_name, task)

    async def get_reviewer_text(self, element_name: str) -> str:
        """Texte optimisé reviewer"""
        text_generator = await self.get_text_generator()
        return await text_generator.get_reviewer_context(element_name)

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Charge les métadonnées des documents"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Erreur lors du chargement des métadonnées: {e}")
        return {}

    def _save_metadata(self):
        """Sauvegarde les métadonnées des documents"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde des métadonnées: {e}")

    def _get_file_type(self, filepath: str) -> str:
        """Détermine le type de fichier basé sur l'extension"""
        extension = Path(filepath).suffix.lower()
        return self.file_extensions.get(extension, 'text')

    def _calculate_file_hash(self, filepath: str) -> str:
        """Calcule le hash MD5 d'un fichier"""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul du hash pour {filepath}: {e}")
            return ""
        return hash_md5.hexdigest()

    def _get_document_id(self, filepath: str) -> str:
        """Génère un ID unique pour un document"""
        path = Path(filepath).resolve()
        return str(path).replace('/', '_').replace('\\', '_').replace(':', '').replace(' ', '_')

    async def add_document(
            self,
            filepath: str,
            project_name: Optional[str] = None,
            version: Optional[str] = None,
            additional_metadata: Optional[Dict[str, Any]] = None,
            force_update: bool = False,
            processing_options: Dict = None
    ) -> bool:
        """Version optimisée avec classification intégrée au chunking"""

        if not await self._ensure_initialized():
            return False

        filepath = str(Path(filepath).resolve())
        if not os.path.exists(filepath):
            self.logger.error(f"Fichier non trouvé: {filepath}")
            return False

        doc_id = self._get_document_id(filepath)

        # Vérifier si le fichier a changé
        current_hash = self._calculate_file_hash(filepath)
        if not force_update and doc_id in self.documents_metadata:
            if self.documents_metadata[doc_id].get('file_hash') == current_hash:
                print(f"📄 {Path(filepath).name} - Aucun changement détecté")
                return True

        print(f"📝 Traitement de {Path(filepath).name}...")

        try:
            # Préparer les métadonnées
            additional_meta = {
                'project': project_name or "Unknown",
                'version': version or "1.0",
                'file_type': self._get_file_type(filepath),
                'file_hash': current_hash
            }
            if additional_metadata:
                additional_meta.update(additional_metadata)

            # Ajouter le document - la classification se fait PENDANT le chunking
            doc_id_result = await self.rag_engine.document_store.add_document_with_id(
                filepath, doc_id, additional_meta #processing_options=processing_options
            )

            if doc_id in self.rag_engine.document_store.document_chunks:
                chunks = self.rag_engine.document_store.document_chunks[doc_id]

                # Analyser les résolutions d'ambiguïté
                resolved_ambiguities = []
                for chunk in chunks:
                    detected_concepts = chunk.get('metadata', {}).get('detected_concepts', [])
                    for concept in detected_concepts:
                        if concept.get('ambiguity_resolved'):
                            resolved_ambiguities.append({
                                'label': concept['label'],
                                'chosen_uri': concept['concept_uri'],
                                'final_confidence': concept.get('final_confidence', concept['confidence']),
                                'context_score': concept.get('context_score', 0)
                            })

                if resolved_ambiguities:
                    print(f"🎯 Ambiguïtés résolues pour {Path(filepath).name}:")
                    for resolution in resolved_ambiguities[:3]:  # Top 3
                        context_info = f"contexte: {resolution['context_score']:+.2f}" if resolution[
                                                                                              'context_score'] != 0 else ""
                        chosen_concept = resolution['chosen_uri'].split('#')[-1]
                        print(
                            f"   - '{resolution['label']}' → {chosen_concept} (conf: {resolution['final_confidence']:.2f}, {context_info})")
            # Collecter les concepts depuis les chunks
            all_concepts = {}

            if doc_id in self.rag_engine.document_store.document_chunks:
                chunks = self.rag_engine.document_store.document_chunks[doc_id]

                for chunk in chunks:
                    detected_concepts = chunk.get('metadata', {}).get('detected_concepts', [])

                    for concept in detected_concepts:
                        uri = concept.get('concept_uri', '')
                        if uri:
                            if uri not in all_concepts:
                                all_concepts[uri] = {
                                    'label': concept.get('label', ''),
                                    'total_confidence': 0,
                                    'count': 0
                                }
                            all_concepts[uri]['total_confidence'] += concept.get('confidence', 0)
                            all_concepts[uri]['count'] += 1

            # Agréger les concepts au niveau document
            document_concepts = []
            for uri, stats in all_concepts.items():
                avg_confidence = stats['total_confidence'] / stats['count'] if stats['count'] > 0 else 0
                document_concepts.append({
                    'uri': uri,
                    'label': stats['label'],
                    'confidence': avg_confidence,
                    'frequency': stats['count']
                })

            # Trier par score (confidence × frequency)
            document_concepts.sort(
                key=lambda x: x['confidence'] * (x['frequency'] / len(chunks) if chunks else 1),
                reverse=True
            )

            # Mettre à jour les métadonnées
            self.documents_metadata[doc_id] = {
                'document_id': doc_id,
                'filepath': filepath,
                'filename': Path(filepath).name,
                'last_updated': datetime.now().isoformat(),
                'ontology_concepts': [c['label'] for c in document_concepts[:20]],
                'ontology_classified': True,
                'total_chunks': len(chunks) if doc_id in self.rag_engine.document_store.document_chunks else 0,
                **additional_meta
            }

            self._save_metadata()
            print(f"✅ {Path(filepath).name} ajouté avec succès!")
            print(
                f"📊 {len(document_concepts)} concepts uniques détectés dans {self.documents_metadata[doc_id]['total_chunks']} chunks")

            return True

        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajout de {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _add_document_with_chunks(
            self,
            filepath: str,
            doc_id: str,
            chunks: List[Dict[str, Any]],
            metadata: Dict[str, Any]
    ) -> bool:
        """Ajoute un document avec des chunks pré-traités"""
        try:
            # Copier le fichier
            filename = Path(filepath).name
            document_path = Path(self.rag_engine.document_store.documents_dir) / f"{doc_id}_{filename}"

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: shutil.copy2(filepath, str(document_path)))

            # Stocker les métadonnées du document
            self.rag_engine.document_store.documents[doc_id] = {
                "id": doc_id,
                "path": str(document_path),
                "original_path": filepath,
                "original_filename": filename,
                "chunks_count": len(chunks),
                "additional_metadata": metadata
            }

            # Stocker les chunks
            self.rag_engine.document_store.document_chunks[doc_id] = chunks

            # Sauvegarder les chunks sur disque
            chunks_path = await self.rag_engine.document_store.save_chunks(doc_id)

            # Créer et sauvegarder les embeddings
            await self.rag_engine.embedding_manager.create_embeddings(chunks)
            await self.rag_engine.embedding_manager.save_embeddings(doc_id)

            # Sauvegarder les métadonnées
            await self.rag_engine.document_store._save_metadata()

            return True

        except Exception as e:
            print(f"❌ Erreur lors de l'ajout du document avec chunks: {e}")
            return False

    async def query(
            self,
            question: str,
            max_results: int = 5,
            file_types: Optional[List[str]] = None,
            projects: Optional[List[str]] = None,
            use_ontology: bool = True
    ) -> Dict[str, Any]:
        """Effectue une requête dans le RAG"""
        if not await self._ensure_initialized():
            return {"error": "RAG non initialisé"}

        print(f"🔍 Recherche: {question}")

        try:
            if use_ontology and self.classifier:
                # Utiliser la nouvelle méthode simple
                result = await self.classifier.search_with_concepts(
                    query=question,
                    top_k=max_results,
                    concept_weight=0.3  # 30% concepts, 70% similarité textuelle
                )

                # Filtrer par type/projet si nécessaire
                if file_types or projects:
                    filtered_passages = []
                    for passage in result.get("passages", []):
                        metadata = passage.get("metadata", {})

                        if file_types and metadata.get("file_type") not in file_types:
                            continue
                        if projects and metadata.get("project") not in projects:
                            continue

                        filtered_passages.append(passage)

                    result["passages"] = filtered_passages

                return result
            else:
                # Recherche standard sans ontologie
                # TODO fonction à faire !
                result = await self.rag_engine.chat(question, max_results)
                return self._format_standard_result(result)

        except Exception as e:
            self.logger.error(f"Erreur lors de la requête: {e}")
            return {"error": f"Erreur lors de la requête: {str(e)}"}

    async def ask(self, query: str):
        """
        Point d'entrée unifié pour toutes les questions.
        Route la requête vers le bon outil et formate la sortie.
        """
        # Le routeur décide quoi faire
        result = await self.router.route_query(query)

        # Maintenant, on gère la sortie en fonction du type de résultat
        result_type = result.get("type")

        if result_type == "entity_list":
            print("\n--- LISTE D'ENTITÉS CORRESPONDANTES ---")
            entities_found = result.get("data", [])
            if not entities_found:
                print("  -> Aucune entité trouvée.")
            for item in entities_found:
                entity = item['entity']
                score = item['score']
                print(f"  - {entity.entity_type:<12} {entity.entity_name:<30} "
                      f"(Fichier: {entity.filename}, Score: {score:.2f})")

        elif result_type == "entity_report":
            report_data = result.get("data", {})

            # 1. Transformer le rapport en contexte textuel propre
            context_string = await self._format_report_for_llm(report_data)

            # TODO pourquoi faire _generate_natural... alors que nous le faisons plus bas ?
            # 2. Demander au LLM de synthétiser ce contexte pour répondre à la question
            final_answer = await self._generate_natural_language_response(query, context_string)

            # 3. Afficher la réponse en langage naturel
            print(f"🤖 Réponse: {final_answer}")

        elif result_type == "conceptual_search":
            # Le routeur a demandé de lancer une recherche conceptuelle
            print("Action: Lancement de la recherche conceptuelle...")
            conceptual_result = await self.hierarchical_query(query)
            # Affichez ce résultat avec votre fonction existante
            await display_intelligent_hierarchical_result(conceptual_result)

        elif result_type == "entity_relations":
            # Extraire les informations du résultat
            entity = result.get('entity', {})
            relation_type = result.get('relation_type')
            data = result.get('data')
            entity_name = entity.get('name', 'Inconnu')

            # Préparer le contexte pour le LLM générateur
            context = ""
            if not data:
                context = f"L'information demandée n'a pas été trouvée. L'entité '{entity_name}' n'a aucune relation de type '{relation_type}' dans la base de connaissance."
            else:
                context = f"Voici les informations trouvées concernant l'entité '{entity_name}' pour la question : '{query}'.\n"

                if relation_type == 'callers':
                    context += "Cette entité est appelée par les entités suivantes:\n"
                    # data est une liste de dictionnaires
                    for caller in data:
                        context += (f"- Nom: {caller.get('name')}, Type: {caller.get('type')}, "
                                    f"Fichier: {caller.get('file', 'N/A')}, "
                                    f"à la Ligne: {caller.get('call_line', '?')}\n")  # Utilise call_line

                elif relation_type == 'callees':
                    # data est un dictionnaire
                    calls = data.get('called_functions_or_subroutines', [])
                    deps = data.get('module_dependencies (USE)', [])
                    if calls:
                        context += "Cette entité appelle les fonctions/subroutines suivantes:\n"
                        # 'calls' est maintenant une liste de dictionnaires
                        for c in calls:
                            context += f"- Nom: {c.get('name')}, à la ligne {c.get('line', '?')}\n"
                    if deps:
                        context += f"Cette entité dépend (USE) des modules suivants: {', '.join(deps)}\n"

            # Générer la réponse finale en langage naturel
            final_answer = await self._generate_natural_language_response(query, context)
            print(f"🤖 Réponse: {final_answer}")

        elif result_type == "unknown" or result_type == "error":
            print(f"\n--- ERREUR OU REQUÊTE INCONNUE ---")
            print(f"  -> {result.get('message')}")

        else:
            print("Type de résultat inattendu.")

    async def _format_report_for_llm(self, report: dict) -> str:
        """
        Transforme le dictionnaire de rapport complet en une chaîne de caractères
        structurée et lisible, idéale pour servir de contexte à un LLM.
        """
        if not report or report.get("error"):
            return f"Le rapport n'a pas pu être généré. Erreur: {report.get('error', 'inconnue')}"

        # Utilisation d'une liste de chaînes pour construire le contexte
        context_parts = []

        entity_name = report.get("entity_name", "Inconnu")
        context_parts.append(f"### Rapport d'Analyse pour l'Entité : {entity_name}\n")

        # Section Résumé
        summary = report.get('summary', {})
        if summary:
            context_parts.append("#### Résumé\n")
            for key, value in summary.items():
                context_parts.append(f"- {key.replace('_', ' ').capitalize()}: {value}")
            context_parts.append("")  # Ligne vide pour l'espacement

        # Section Signature Riche
        signature = report.get('rich_signature')
        if signature:
            context_parts.append("#### Signature et Arguments\n```fortran\n" + signature + "\n```\n")

        # Section Relations Sortantes (ce que l'entité utilise)
        outgoing = report.get('outgoing_relations', {})
        if outgoing:
            context_parts.append("#### Relations Sortantes (ce que cette entité utilise)\n")
            calls = outgoing.get('called_functions_or_subroutines', [])
            deps = outgoing.get('module_dependencies (USE)', [])
            if calls:
                # On crée une liste de chaînes formatées "nom (ligne X)"
                formatted_calls = [f"{call.get('name', '')} (ligne {call.get('line', '?')})" for call in calls]
                context_parts.append(f"- Appelle : {', '.join(formatted_calls)}")
            else:
                context_parts.append("- Appelle : (Aucun)")

            context_parts.append(f"- Dépend de (USE) : {', '.join(deps) if deps else '(Aucun)'}")
            context_parts.append("")

        # Section Relations Entrantes (qui utilise cette entité)
        incoming = report.get('incoming_relations', [])
        if incoming:
            context_parts.append("#### Relations Entrantes (qui utilise cette entité)\n")
            callers = [f"{c['name']} (type: {c['type']})" for c in incoming]
            context_parts.append(f"- Est appelée par : {', '.join(callers) if callers else '(Personne)'}")
            context_parts.append("")

        # Section Contexte Global (Parent)
        global_ctx = report.get('global_context', {})
        parent = global_ctx.get('parent_entity', {})
        if parent and isinstance(parent, dict):
            context_parts.append(
                f"#### Contexte Global\n- Se situe dans : {parent.get('name')} (type: {parent.get('type')})\n")

        # Section Contexte Local (Enfants)
        local_ctx = report.get('local_context', {})
        children = local_ctx.get('children_entities', [])
        if children:
            child_names = [f"{c['name']} (type: {c['type']})" for c in children]
            context_parts.append(f"#### Contenu de l'entité\n- Contient les sous-entités : {', '.join(child_names)}\n")

        # Section Code Source (extrait)
        source = local_ctx.get('source_code', '')
        if source and "Non demandé" not in source:
            context_parts.append("#### Code Source\n")
            # On ne passe qu'un extrait pour ne pas surcharger le contexte du LLM
            source_lines = source.splitlines()
            snippet = "\n".join(source_lines)
            context_parts.append(f"```fortran\n{snippet}\n```")

        return "\n".join(context_parts)

    async def remove_document(self, filepath: str) -> bool:
        """Supprime un document du RAG"""
        if not await self._ensure_initialized():
            return False

        doc_id = self._get_document_id(filepath)

        if doc_id not in self.documents_metadata:
            self.logger.warning(f"Document non trouvé: {filepath}")
            return False

        try:
            success = await self.rag_engine.remove_document(doc_id)

            if success:
                del self.documents_metadata[doc_id]
                self._save_metadata()
                print(f"🗑️  Document supprimé: {Path(filepath).name}")

            return success

        except Exception as e:
            self.logger.error(f"Erreur lors de la suppression de {filepath}: {e}")
            return False

    async def _ensure_initialized(self) -> bool:
        """S'assure que le RAG est initialisé"""
        if self.rag_engine is None:
            await self.initialize()
        return self.rag_engine is not None

    def list_documents(self) -> List[Dict[str, Any]]:
        """Liste tous les documents dans le RAG"""
        return list(self.documents_metadata.values())

    def get_statistics(self) -> Dict[str, Any]:
        """Retourne des statistiques sur le RAG"""
        if not self.documents_metadata:
            return {"message": "Aucun document chargé"}

        file_types = {}
        projects = {}
        for metadata in self.documents_metadata.values():
            file_type = metadata.get('file_type', 'unknown')
            project = metadata.get('project', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
            projects[project] = projects.get(project, 0) + 1

        return {
            'total_documents': len(self.documents_metadata),
            'file_types': file_types,
            'projects': projects
        }

    def _flatten_concept_hierarchy(self, concepts_hierarchy: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aplatit une hiérarchie de concepts"""
        flattened = []
        for concept in concepts_hierarchy:
            flattened.append(concept)
            if "sub_concepts" in concept:
                flattened.extend(self._flatten_concept_hierarchy(concept["sub_concepts"]))
        return flattened

    async def add_documents_batch(
            self,
            documents_info: List[Dict[str, Any]],
            max_concurrent: int = 3,
            force_update: bool = False,
            preserve_order: bool = False,  # NOUVEAU : pour les dépendances
            processing_options: Dict = None
    ) -> Dict[str, bool]:
        """Ajoute plusieurs documents en parallèle avec gestion de concurrence améliorée"""

        if not await self._ensure_initialized():
            return {}

        # NOUVEAU : Trier par type de fichier (modules avant les autres)
        if preserve_order:
            documents_info = self._sort_documents_by_dependencies(documents_info)

        # NOUVEAU : Sémaphore pour l'EntityManager
        entity_manager_semaphore = asyncio.Semaphore(1)  # Un seul à la fois pour l'indexation
        processing_semaphore = asyncio.Semaphore(max_concurrent)

        async def add_single_doc(doc_info):
            async with processing_semaphore:
                filepath = doc_info["filepath"]
                try:
                    # NOUVEAU : Vérifier si c'est un fichier Fortran
                    is_fortran = self._get_file_type(filepath) == 'fortran'

                    if is_fortran:
                        # Pour Fortran, utiliser le sémaphore d'EntityManager
                        async with entity_manager_semaphore:
                            success = await self._add_document_with_entity_protection(
                                filepath, doc_info, force_update, processing_options=processing_options
                            )
                    else:
                        # Pour les autres fichiers, traitement standard
                        success = await self.add_document(
                            filepath,
                            project_name=doc_info.get("project_name"),
                            version=doc_info.get("version"),
                            additional_metadata=doc_info.get("additional_metadata"),
                            force_update=force_update
                        )

                    return filepath, success

                except Exception as e:
                    self.logger.error(f"Erreur pour {filepath}: {e}")
                    return filepath, False

        # Lancer en parallèle
        tasks = [add_single_doc(doc_info) for doc_info in documents_info]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Organiser les résultats
        final_results = {}
        success_count = 0

        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Exception durant le traitement: {result}")
            else:
                filepath, success = result
                final_results[filepath] = success
                if success:
                    success_count += 1

        # NOUVEAU : Reconstruction des relations après traitement parallèle
        if success_count > 0:
            await self._rebuild_entity_relationships()

        if success_count > 0:
            print("🔄 Synchronisation des index de recherche...")
            try:
                await self.custom_processor.fortran_processor.sync_orchestrator_with_entities()
                print("✅ Index de recherche synchronisés")
            except Exception as e:
                print(f"⚠️ Erreur synchronisation: {e}")

        print(f"📊 Traitement terminé: {success_count}/{len(documents_info)} fichiers ajoutés avec succès")
        return final_results

    async def _add_document_with_entity_protection(
            self,
            filepath: str,
            doc_info: Dict[str, Any],
            force_update: bool,
            processing_options: Dict = None
    ) -> bool:
        """Ajoute un document avec protection de l'EntityManager"""
        try:
            return await self.add_document(
                filepath,
                project_name=doc_info.get("project_name"),
                version=doc_info.get("version"),
                additional_metadata=doc_info.get("additional_metadata"),
                force_update=force_update,
                processing_options=processing_options
            )
        except Exception as e:
            self.logger.error(f"Erreur traitement protégé {filepath}: {e}")
            return False

    def _sort_documents_by_dependencies(self, documents_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Trie les documents par ordre de dépendance (modules en premier)"""
        fortran_modules = []
        fortran_others = []
        non_fortran = []

        for doc_info in documents_info:
            filepath = doc_info["filepath"]
            file_type = self._get_file_type(filepath)

            if file_type == 'fortran':
                # Heuristique simple : les fichiers avec "module" dans le nom en premier
                filename = Path(filepath).name.lower()
                if 'module' in filename or filepath.endswith('_mod.f90'):
                    fortran_modules.append(doc_info)
                else:
                    fortran_others.append(doc_info)
            else:
                non_fortran.append(doc_info)

        # Ordre : modules Fortran → autres Fortran → non-Fortran
        return fortran_modules + fortran_others + non_fortran

    async def _rebuild_entity_relationships(self):
        """Reconstruit les relations entre entités après traitement parallèle"""
        try:
            if (self.custom_processor and
                    self.custom_processor.fortran_processor and
                    self.custom_processor.fortran_processor.entity_manager):
                entity_manager = self.custom_processor.fortran_processor.entity_manager

                # Reconstruire les relations
                await entity_manager._build_relationships()
                await entity_manager._detect_and_group_split_entities()

                print("✅ Relations entre entités reconstruites")

        except Exception as e:
            self.logger.error(f"Erreur reconstruction relations: {e}")

    async def diagnose_fortran_entity_manager(self) -> Dict[str, Any]:
        """Diagnostic complet de l'EntityManager Fortran"""
        try:
            if not self.custom_processor:
                return {"error": "custom_processor non disponible"}

            if not self.custom_processor.fortran_processor:
                return {"error": "fortran_processor non disponible"}

            return await self.custom_processor.fortran_processor.diagnose_entity_manager()

        except Exception as e:
            return {"error": f"Erreur diagnostic: {str(e)}"}

    async def _generate_natural_language_response(self, original_query: str, context_data: str) -> str:
        """
        Prend des données de contexte et la question originale pour générer
        une réponse finale en langage naturel, en utilisant le provider LLM du RAG.

        Args:
            original_query: La question posée initialement par l'utilisateur.
            context_data: Les informations brutes récupérées par les outils, formatées en texte.

        Returns:
            Une chaîne de caractères contenant la réponse synthétisée.
        """
        # 1. Création d'un prompt très directif pour la tâche de synthèse
        prompt = f"""Tu es un architecte logiciel senior et un expert du langage Fortran. Ta mission est d'analyser le contexte technique fourni pour répondre à la question de l'utilisateur de manière synthétique et avec perspicacité. Ne te contente pas de lister les faits, explique leur signification.

        INSTRUCTIONS POUR ADAPTER TA RÉPONSE :
Analyse la "Question Originale" et choisis l'un des deux styles de réponse suivants :

1.  **Si la question est FACTUELLE (demande qui, quel, combien, où) :**
    - Réponds directement en incluant les détails de localisation importants (fichier, lignes) s'ils sont disponibles.
    - Sois concis mais informatif. Ne fais pas d'analyse profonde si elle n'est pas demandée.

    - **EXEMPLE:**
      - Question: "Qui appelle la routine 'Cleanup' ?"
      - Contexte: "Cette entité est appelée par les entités suivantes:\n- Nom: Finalize_Run, Type: subroutine, Fichier: /path/main.f90, à la Ligne: 525\n"
      - **Réponse attendue:** "La routine 'Cleanup' est appelée par la subroutine 'Finalize_Run', à la ligne 525 du fichier `/path/main.f90`."

2.  **Si la question est ANALYTIQUE (demande pourquoi, comment, à quoi sert) :**
        a.  **Détermine le But Principal :** En te basant sur le nom de l'entité, les fonctions qu'elle appelle (surtout `free`, `allocate`, `create`, `destroy`, `init`), et les commentaires dans le code, déduis la responsabilité première de cette entité.
        b.  **Identifie les Opérations Clés :** Décris les actions les plus importantes effectuées par le code, en te basant sur les fonctions appelées et les dépendances.
        c.  **Analyse le Contexte d'Appel :** Explique pourquoi cette entité est utilisée, en regardant qui l'appelle.
        d.  **Synthétise, Ne Récite Pas :** Combine ces points dans une réponse fluide. Commence par le but principal, puis donne les détails pertinents.
        e.  **Source Unique de Vérité :** Fonde TOUTE ton analyse EXCLUSIVEMENT sur le "Contexte Fourni". N'invente rien.

        ---
        [Contexte Fourni]
        {context_data}
        ---
        [Question Originale de l'Utilisateur]
        {original_query}
        ---

        [Ton Analyse d'Expert]
        """
        try:
            # 2. Appel du provider LLM existant
            # Nous passons le prompt complet dans un message utilisateur. C'est une
            # approche robuste pour les tâches de génération en une seule fois.
            response = await self.rag_engine.llm_provider.generate_response(
                messages=[
                    {"role": "user", "content": prompt}
                ],

                temperature=0.3
            )
            return response.strip()

        except Exception as e:
            logging.error(f"Erreur lors de la génération de la réponse en langage naturel: {e}", exc_info=True)
            return "Désolé, une erreur est survenue lors de la formulation de la réponse finale."

    async def get_entity_manager_stats(self) -> Dict[str, Any]:
        """Statistiques détaillées de l'EntityManager"""
        try:
            if (not self.custom_processor or
                    not self.custom_processor.fortran_processor or
                    not self.custom_processor.fortran_processor.entity_manager):
                return {"error": "EntityManager non disponible"}

            entity_manager = self.custom_processor.fortran_processor.entity_manager

            return {
                "total_entities": len(entity_manager.entities),
                "entities_by_type": dict([(t, len(ids)) for t, ids in entity_manager.type_to_entities.items()]),
                "entities_by_file": dict([(f, len(ids)) for f, ids in entity_manager.file_to_entities.items()]),
                "name_index_size": len(entity_manager.name_to_entity),
                "parent_child_relations": len(entity_manager.parent_to_children),
                "initialized": getattr(entity_manager, '_initialized', False),
                "sample_entities": [
                    {
                        "name": e.entity_name,
                        "type": e.entity_type,
                        "id": e.entity_id,
                        "file": Path(e.filepath).name if e.filepath else "unknown"
                    }
                    for e in list(entity_manager.entities.values())[:10]
                ]
            }

        except Exception as e:
            return {"error": f"Erreur stats: {str(e)}"}

    async def preprocess_fortran_directory(
            self,
            directory_path: str,
            backup_dir: str = ".fortran_backup",
            max_concurrent: int = 5
    ) -> Dict[str, str]:
        """
        Scanne un répertoire, nettoie les fichiers Fortran et écrase les originaux.
        Crée une sauvegarde des fichiers originaux avant modification.

        Args:
            directory_path: Le chemin du répertoire à traiter.
            backup_dir: Le nom du sous-répertoire où stocker les sauvegardes.
            max_concurrent: Le nombre de fichiers à traiter en parallèle.

        Returns:
            Un dictionnaire résumant le statut de chaque fichier traité.
        """
        root_path = Path(directory_path).resolve()
        backup_path = root_path / backup_dir
        backup_path.mkdir(exist_ok=True)

        self.logger.info(f"Lancement du prétraitement Fortran sur le répertoire: {root_path}")
        self.logger.info(f"Les sauvegardes seront stockées dans: {backup_path}")

        # Utiliser la méthode `scan_directory` existante pour trouver les fichiers Fortran
        fortran_files_info = self.scan_directory(
            directory_path=str(root_path),
            file_filters=['f90', 'f95', 'f03', 'f08', 'f', 'for', 'ftn'],
            recursive=True
        )

        if not fortran_files_info:
            self.logger.warning("Aucun fichier Fortran trouvé pour le prétraitement.")
            return {}

        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}

        async def clean_single_file(file_info):
            async with semaphore:
                filepath = Path(file_info["filepath"])
                relative_path = filepath.relative_to(root_path)

                try:
                    # 1. Lire le contenu original
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        original_content = f.read()

                    # 2. Créer une sauvegarde
                    backup_file_path = backup_path / relative_path
                    backup_file_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(backup_file_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)

                    # 3. Nettoyer le code
                    cleaned_content = nettoyer_commentaires_fortran(
                        original_content,
                        utiliser_fprettify=True,
                        options_fprettify={"indent": 2, "line-length": 100}
                    )

                    # 4. Écraser le fichier original avec le contenu nettoyé
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(cleaned_content)

                    self.logger.info(f"✅ Fichier nettoyé et sauvegardé: {relative_path}")
                    return str(relative_path), "success"

                except Exception as e:
                    self.logger.error(f"❌ Erreur lors du nettoyage de {relative_path}: {e}")
                    return str(relative_path), f"error: {e}"

        tasks = [clean_single_file(info) for info in fortran_files_info]
        task_results = await asyncio.gather(*tasks)

        for path, status in task_results:
            results[path] = status

        return results

    #  TODO non utilisé pour l'instant
    async def query_parallel_concepts(
            self,
            question: str,
            max_results: int = 5,
            max_concurrent_concepts: int = 5,
            use_ontology: bool = True
    ) -> Dict[str, Any]:
        """Requête optimisée avec traitement parallèle des concepts"""

        if not await self._ensure_initialized():
            return {"error": "RAG non initialisé"}

        if use_ontology and self.classifier:
            return await self.classifier.auto_concept_search_optimized(
                query=question,
                top_k=max_results,
                max_concurrent_searches=max_concurrent_concepts
            )
        else:
            return await self.rag_engine.chat(query=question, top_k=max_results)


# Exemple d'utilisation
async def example_usage():
    """Exemple d'utilisation d'OntoRAG"""

    rag = OntoRAG(
        storage_dir=STORAGE_DIR,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        ontology_path=ONTOLOGY_PATH_TTL
    )

    await rag.initialize()

    documents_info = [
        {
            "filepath": "/home/yopla/test_real_bigdft/bigdft-suite/liborbs/src/reformatting.f90",
            "project_name": "BigDFT",
            "version": "1.9"
        },
        {
            "filepath": "/home/yopla/test_real_bigdft/bigdft-suite/liborbs/src/scalprod.f90",
            "project_name": "BigDFT",
            "version": "1.9"
        },
        {
            "filepath": "/home/yopla/test_real_bigdft/bigdft-suite/liborbs/src/growshrink_hyb_optim.f90",
            "project_name": "BigDFT",
            "version": "1.9"
        }
    ]

    # Traitement parallèle
    results = await rag.add_documents_batch(documents_info, max_concurrent=MAX_CONCURRENT)
    print(f"Ajout terminé: {sum(results.values())}/{len(results)} succès")

    # Statistiques
    stats = rag.get_statistics()
    print("\n📊 Statistiques:", json.dumps(stats, indent=2))

    while True:
        # Requête
        query = input('Votre question : ')
        result = await rag.query(query, max_results=MAX_RESULTS)

        print(f"\n🤖 Réponse: {result.get('answer', 'Pas de réponse')}")

        # Afficher les sources
        print(f"\n📚 Sources utilisées:")
        for source in result.get('sources', []):
            print(f"\n  📄 {source['filename']} (lignes {source['start_line']}-{source['end_line']})")
            print(f"     Type: {source['entity_type']} - Nom: {source['entity_name']}")
            if source['detected_concepts']:
                print(f"     Concepts: {', '.join(source['detected_concepts'])}")
            if source['matched_concepts']:
                print(f"     Concepts matchés: {', '.join(source['matched_concepts'])}")
            print(f"     Score: {source['relevance_score']}")

if __name__ == "__main__":
    asyncio.run(example_usage())
