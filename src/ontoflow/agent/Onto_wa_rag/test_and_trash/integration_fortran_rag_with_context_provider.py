"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# onto_rag.py
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

import numpy as np

from CONSTANT import API_KEY_PATH, CHUNK_SIZE, CHUNK_OVERLAP, LLM_MODEL, ONTOLOGY_PATH_TTL, MAX_CONCURRENT, MAX_RESULTS, \
    STORAGE_DIR
from context_provider import SmartContextProvider, ContextualTextGenerator
from context_provider.Context_visualizer import create_fortran_dependency_visualization
from ontology.classifier import OntologyClassifier
from ontology.ontology_manager import OntologyManager
from utils.ontologie_fortran_chunker import OFPFortranSemanticChunker as F2pyFortranSemanticChunker
from utils.semantic_chunker import SemanticChunker
from utils.document_processor import DocumentProcessor

# Imports pour le RAG
from utils.rag_engine import RAGEngine
from provider.llm_providers import OpenAIProvider
from provider.get_key import get_openai_key


class OntoDocumentProcessor(DocumentProcessor):
    """
    Processeur de documents personnalisé qui utilise les chunkers appropriés
    selon le type de fichier et bypass le MarkdownConverter pour certains types
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, file_extensions: dict = None):
        super().__init__(chunk_size, chunk_overlap, use_semantic_chunking=True)
        self.file_extensions = file_extensions or {}

        # Initialiser les chunkers spécialisés
        self.fortran_chunker = F2pyFortranSemanticChunker(
            min_chunk_size=200,
            max_chunk_size=chunk_size,
            overlap_sentences=0,
        )

        # Types de fichiers qui doivent être lus directement (sans conversion)
        self.direct_read_types = {'fortran', 'code', 'text', 'config'}
        self.ontology_manager = None

    async def _extract_text_with_metadata(self, filepath: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extrait le texte et les métadonnées d'un document
        Override pour gérer les fichiers non supportés par MarkdownConverter
        """
        # Déterminer le type de fichier
        extension = Path(filepath).suffix.lower()
        file_type = self.file_extensions.get(extension, 'text')

        # Métadonnées de base
        metadata = {
            'file_size': os.path.getsize(filepath),
            'file_type': file_type,
            'modification_date': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat(),
            'extraction_date': datetime.now().isoformat(),
        }

        # Si c'est un type de fichier qui doit être lu directement
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
            # Utiliser le convertisseur par défaut pour les autres types
            print(f"🔄 Conversion via MarkdownConverter: {Path(filepath).name}")
            return await super()._extract_text_with_metadata(filepath)

    async def process_document(self, filepath: str, document_id: str = None, additional_metadata: dict = None):
        """
        Traite un document en utilisant le chunker approprié selon son type
        """
        # Si pas d'ID fourni, en générer un
        if document_id is None:
            document_id = str(uuid.uuid4())

        # Déterminer le type de fichier
        extension = Path(filepath).suffix.lower()
        file_type = self.file_extensions.get(extension, 'text')

        print(f"🔍 Type de fichier détecté: {file_type} pour {Path(filepath).name}")

        # Extraire le texte et les métadonnées (utilise notre méthode override)
        text_content, doc_metadata = await self._extract_text_with_metadata(filepath)

        # Fusionner les métadonnées
        metadata = {**doc_metadata, **(additional_metadata or {})}

        # Utiliser le chunker approprié selon le type de fichier
        if file_type == 'fortran':
            print(f"✂️  Utilisation du chunker Fortran sémantique")

            """
            chunks = await self.fortran_chunker.create_fortran_chunks_with_debug(
                text_content, document_id, filepath, metadata,
                self.ontology_manager, create_debug_file=True
            )
            """
            chunks = await self.fortran_chunker.create_fortran_chunks(
                text_content, document_id, filepath, metadata, self.ontology_manager
            )


        elif file_type in ['text', 'markdown']:
            print(f"✂️  Utilisation du chunker sémantique pour texte")
            chunks = self.semantic_chunker.create_semantic_chunks(
                text_content, document_id, filepath, metadata
            )

        else:
            print(f"✂️  Utilisation du chunker générique")
            # Utiliser le chunker par défaut pour les autres types
            chunks = self._create_chunks(text_content, document_id, filepath)
            # Ajouter les métadonnées à chaque chunk
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
            model: str = LLM_MODEL,
            chunk_size: int = CHUNK_SIZE,
            chunk_overlap: int = CHUNK_OVERLAP,
            ontology_path: Optional[str] = None
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

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

    async def initialize(self):
        """Initialise le système RAG de manière asynchrone"""
        if self.rag_engine is not None:
            return

        print("🚀 Initialisation d'OntoRAG...")

        # Initialiser les providers
        openai_key = get_openai_key(api_key_path=self.api_key_path)
        llm_provider = OpenAIProvider(model=self.model, api_key=openai_key)

        # Créer notre processeur personnalisé AVANT l'initialisation
        custom_processor = OntoDocumentProcessor(
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
                    enable_relation_learning=True,
                    multiscale_mode=True
                )

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
        self.rag_engine.processor = custom_processor
        self.rag_engine.document_store.processor = custom_processor

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

            # ← ASSIGNMENT ICI - après que tout soit initialisé
            custom_processor.ontology_manager = self.ontology_manager
            print("✅ Ontology_manager assigné au processeur")

        # NOUVEAU : Initialiser le système de contexte
        await self.initialize_context_provider()

        print("✅ OntoRAG initialisé avec succès!")

    async def quick_diagnose(self):
        """Diagnostic rapide des problèmes"""
        print("🔍 DIAGNOSTIC RAPIDE")
        print("=" * 30)

        if not self.context_provider:
            print("❌ Context provider non initialisé")
            return

        stats = self.context_provider.get_index_stats()
        print(f"Entités indexées: {stats['total_entities']}")
        print(f"USE dependencies: {stats['use_dependencies']}")

        # Vérifier les entités split
        all_names = list(self.context_provider.entity_index.name_to_chunks.keys())
        split_count = sum(1 for name in all_names if '_part_' in name)
        print(f"Entités divisées: {split_count}/{len(all_names)}")

        # Test rapide d'une entité
        if all_names:
            test_entity = all_names[0]
            try:
                context = await self.get_local_context(test_entity, 500)
                deps = len(context.get('immediate_dependencies', []))
                calls = len(context.get('called_functions', []))
                print(f"Test entité '{test_entity}': {deps} deps, {calls} calls")
            except Exception as e:
                print(f"❌ Erreur test: {e}")

    async def initialize_context_provider(self):
        """Initialise le fournisseur de contexte"""
        if self.context_provider is not None:
            return

        print("🧠 Initialisation du système de contexte...")

        self.context_provider = SmartContextProvider(
            self.rag_engine.document_store,
            self.rag_engine
        )
        self.text_generator = ContextualTextGenerator(self.context_provider)

        if hasattr(self, 'classifier') and self.classifier:
            print("🔗 Connexion du classifier ontologique au contexte provider")
            # Le context provider pourra accéder au classifier via rag_engine.classifier
            if not hasattr(self.rag_engine, 'classifier'):
                self.rag_engine.classifier = self.classifier

        await self.context_provider.initialize()

        stats = self.context_provider.get_index_stats()
        print(f"✅ Système de contexte initialisé - {stats['total_entities']} entités indexées")
        print(f"   📊 Modules: {stats['modules']}, Functions: {stats['functions']}, Subroutines: {stats['subroutines']}")

    # === NOUVELLES MÉTHODES POUR LE SYSTÈME DE CONTEXTE ===

    async def refresh_context_index(self):
        """Rafraîchit l'index du système de contexte après ajout de documents"""
        if not self.context_provider:
            await self.initialize_context_provider()

        print("🔄 Rafraîchissement de l'index du système de contexte...")
        # Forcer la reconstruction de l'index
        self.context_provider.entity_index._initialized = False
        await self.context_provider.entity_index.build_index()

        stats = self.context_provider.get_index_stats()
        print(f"✅ Index rafraîchi - {stats['total_entities']} entités indexées")
        return stats

    async def debug_chunks_metadata(self, limit: int = 5):
        """Debug des métadonnées des chunks pour diagnostiquer les problèmes"""
        debug_info = {"sample_chunks": []}

        all_docs = await self.rag_engine.document_store.get_all_documents()

        for doc_id in list(all_docs.keys())[:limit]:
            await self.rag_engine.document_store.load_document_chunks(doc_id)
            chunks = await self.rag_engine.document_store.get_document_chunks(doc_id)

            if chunks:
                chunk = chunks[0]
                metadata = chunk.get('metadata', {})

                debug_info["sample_chunks"].append({
                    "chunk_id": chunk.get('id', 'N/A'),
                    "entity_name": metadata.get('entity_name', 'MISSING'),
                    "entity_type": metadata.get('entity_type', 'MISSING'),
                    "filepath": metadata.get('filepath', 'MISSING'),
                    "available_keys": list(metadata.keys())
                })

        return debug_info

    async def _ensure_context_provider(self):
        """S'assure que le context provider est initialisé"""
        if not self.context_provider:
            await self.initialize_context_provider()

        # Si l'index est vide alors qu'on a des documents, le reconstruire
        stats = self.context_provider.get_index_stats()
        if stats['total_entities'] == 0 and len(self.documents_metadata) > 0:
            await self.refresh_context_index()

    async def get_agent_context(self, entity_name: str, agent_type: str = "developer",
                                task_context: str = "code_understanding", max_tokens: int = 4000):
        await self._ensure_context_provider()
        return await self.context_provider.get_context_for_agent(entity_name, agent_type, task_context, max_tokens)

    async def search_entities(self, query: str):
        await self._ensure_context_provider()
        return await self.context_provider.search_entities(query)

    async def test_context_llm(self, entity: str):
        await self._ensure_context_provider()
        # Contexte complet pour LLM
        context_text = await self.text_generator.get_full_context(entity)

        # Contexte rapide
        quick_context = await self.text_generator.get_quick_context(entity)

        # Contexte focalisé dépendances
        dep_context = await self.text_generator.get_dependency_context(entity)

        # Contexte sémantique
        semantic_context = await self.text_generator.get_semantic_context_text(entity)

        input(context_text)
        input(quick_context)
        input(dep_context)
        input(semantic_context)


    async def get_local_context(self, entity_name: str, max_tokens: int = 2000):
        await self._ensure_context_provider()
        return await self.context_provider.get_local_context(entity_name, max_tokens)

    async def get_global_context(self, entity_name: str, max_tokens: int = 3000):
        await self._ensure_context_provider()
        return await self.context_provider.get_global_context(entity_name, max_tokens)

    async def get_semantic_context(self, entity_name: str, max_tokens: int = 2000):
        await self._ensure_context_provider()
        return await self.context_provider.get_semantic_context(entity_name, max_tokens)

    def get_context_stats(self) -> Dict[str, Any]:
        """Statistiques du système de contexte"""
        if not self.context_provider:
            return {"error": "Context provider not initialized"}

        return self.context_provider.get_index_stats()

    def debug_entity(self, entity_name: str) -> Dict[str, Any]:
        """Informations de debug pour une entité"""
        if not self.context_provider:
            return {"error": "Context provider not initialized"}

        return self.context_provider.debug_entity(entity_name)

    # === MÉTHODES DE TEST DU SYSTÈME DE CONTEXTE ===

    async def test_context_system(self, test_entities: List[str] = None) -> Dict[str, Any]:
        """Test complet du système de contexte"""
        print("\n" + "=" * 80)
        print("🧪 TEST COMPLET DU SYSTÈME DE CONTEXTE")
        print("=" * 80)

        if not await self._ensure_initialized():
            return {"error": "RAG non initialisé"}

        if not self.context_provider:
            await self.initialize_context_provider()

        test_results = {
            "stats": {},
            "entity_tests": {},
            "performance": {},
            "errors": []
        }

        # 1. Statistiques générales
        print("\n📊 STATISTIQUES DU SYSTÈME:")
        print("-" * 40)
        stats = self.get_context_stats()
        test_results["stats"] = stats

        for key, value in stats.items():
            print(f"  {key:.<25} {value}")

        # 2. Test de recherche d'entités
        print("\n🔍 TEST DE RECHERCHE D'ENTITÉS:")
        print("-" * 40)

        search_terms = ["compute", "scf", "energy", "matrix", "solve"]
        if test_entities:
            search_terms.extend(test_entities)

        for term in search_terms[:3]:  # Limiter à 3 termes
            try:
                entities = await self.search_entities(term)
                print(f"  Recherche '{term}': {len(entities)} entités trouvées")

                for entity in entities[:2]:  # Top 2 par terme
                    print(f"    - {entity['name']} ({entity['type']}) in {Path(entity['file']).name}")

                if entities:
                    test_results["entity_tests"][term] = entities
            except Exception as e:
                error_msg = f"Erreur recherche '{term}': {e}"
                print(f"    ❌ {error_msg}")
                test_results["errors"].append(error_msg)

        # 3. Test des contextes sur des entités trouvées
        print("\n🎯 TEST DES DIFFÉRENTS CONTEXTES:")
        print("-" * 40)

        # Prendre quelques entités pour les tests
        test_entity_names = []
        for search_results in test_results["entity_tests"].values():
            for entity in search_results[:1]:  # 1 par recherche
                test_entity_names.append(entity['name'])

        if test_entities:
            test_entity_names.extend(test_entities)

        # Limiter à 2 entités pour éviter des tests trop longs
        test_entity_names = list(set(test_entity_names))[:2]

        for entity_name in test_entity_names:
            print(f"\n  🧪 Test pour entité: {entity_name}")

            entity_tests = {}

            # Test contexte développeur
            try:
                start_time = datetime.now()
                dev_context = await self.get_agent_context(
                    entity_name,
                    agent_type="developer",
                    task_context="code_understanding"
                )
                end_time = datetime.now()

                if "error" not in dev_context:
                    duration = (end_time - start_time).total_seconds()
                    tokens = dev_context.get('total_tokens', 0)
                    insights = len(dev_context.get('key_insights', []))
                    recommendations = len(dev_context.get('recommendations', []))

                    print(
                        f"    ✅ Contexte développeur: {tokens} tokens, {insights} insights, {recommendations} recommandations ({duration:.2f}s)")

                    entity_tests["developer"] = {
                        "success": True,
                        "tokens": tokens,
                        "insights": insights,
                        "recommendations": recommendations,
                        "duration": duration
                    }
                else:
                    print(f"    ❌ Contexte développeur: {dev_context['error']}")
                    entity_tests["developer"] = {"success": False, "error": dev_context['error']}

            except Exception as e:
                error_msg = f"Erreur contexte développeur pour {entity_name}: {e}"
                print(f"    ❌ {error_msg}")
                test_results["errors"].append(error_msg)
                entity_tests["developer"] = {"success": False, "error": str(e)}

            # Test contexte reviewer
            try:
                start_time = datetime.now()
                review_context = await self.get_agent_context(
                    entity_name,
                    agent_type="reviewer",
                    task_context="code_review"
                )
                end_time = datetime.now()

                if "error" not in review_context:
                    duration = (end_time - start_time).total_seconds()
                    tokens = review_context.get('total_tokens', 0)

                    print(f"    ✅ Contexte reviewer: {tokens} tokens ({duration:.2f}s)")

                    entity_tests["reviewer"] = {
                        "success": True,
                        "tokens": tokens,
                        "duration": duration
                    }
                else:
                    print(f"    ❌ Contexte reviewer: {review_context['error']}")
                    entity_tests["reviewer"] = {"success": False, "error": review_context['error']}

            except Exception as e:
                error_msg = f"Erreur contexte reviewer pour {entity_name}: {e}"
                print(f"    ❌ {error_msg}")
                test_results["errors"].append(error_msg)
                entity_tests["reviewer"] = {"success": False, "error": str(e)}

            # Test contextes individuels
            try:
                local_ctx = await self.get_local_context(entity_name, 1000)
                if "error" not in local_ctx:
                    deps = len(local_ctx.get('immediate_dependencies', []))
                    calls = len(local_ctx.get('called_functions', []))
                    print(f"    ✅ Contexte local: {deps} dépendances, {calls} appels")
                    entity_tests["local"] = {"success": True, "dependencies": deps, "calls": calls}
                else:
                    print(f"    ❌ Contexte local: {local_ctx['error']}")
                    entity_tests["local"] = {"success": False, "error": local_ctx['error']}

            except Exception as e:
                error_msg = f"Erreur contexte local pour {entity_name}: {e}"
                print(f"    ❌ {error_msg}")
                test_results["errors"].append(error_msg)
                entity_tests["local"] = {"success": False, "error": str(e)}

            test_results["entity_tests"][entity_name] = entity_tests

        # 4. Résumé des performances
        print("\n⚡ RÉSUMÉ DES PERFORMANCES:")
        print("-" * 40)

        total_tests = 0
        successful_tests = 0
        total_duration = 0

        for entity_name, entity_tests in test_results["entity_tests"].items():
            if isinstance(entity_tests, dict):
                for test_type, test_result in entity_tests.items():
                    if isinstance(test_result, dict):
                        total_tests += 1
                        if test_result.get("success", False):
                            successful_tests += 1
                            duration = test_result.get("duration", 0)
                            total_duration += duration

        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        avg_duration = total_duration / successful_tests if successful_tests > 0 else 0

        print(f"  Tests réussis: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
        print(f"  Durée moyenne: {avg_duration:.2f}s")
        print(f"  Erreurs: {len(test_results['errors'])}")

        test_results["performance"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "average_duration": avg_duration,
            "total_errors": len(test_results["errors"])
        }

        if test_results["errors"]:
            print(f"\n❌ ERREURS DÉTECTÉES:")
            for i, error in enumerate(test_results["errors"][:5], 1):
                print(f"  {i}. {error}")

        return test_results

    async def demo_context_usage(self, entity_name: str = None):
        """Démonstration interactive du système de contexte"""
        print("\n" + "=" * 80)
        print("🎪 DÉMONSTRATION DU SYSTÈME DE CONTEXTE")
        print("=" * 80)

        if not await self._ensure_initialized():
            print("❌ RAG non initialisé")
            return

        if not self.context_provider:
            await self.initialize_context_provider()

        # Si pas d'entité spécifiée, en trouver une
        if not entity_name:
            print("\n🔍 Recherche d'entités pour la démonstration...")
            entities = await self.search_entities("compute")
            if not entities:
                entities = await self.search_entities("scf")
            if not entities:
                # Chercher n'importe quelle entité
                stats = self.get_context_stats()
                if stats.get('total_entities', 0) > 0:
                    # Prendre la première entité trouvée
                    all_docs = await self.rag_engine.document_store.get_all_documents()
                    for doc_id in list(all_docs.keys())[:1]:
                        chunks = await self.rag_engine.document_store.get_document_chunks(doc_id)
                        if chunks:
                            entity_name = chunks[0]['metadata'].get('entity_name')
                            if entity_name:
                                entities = [{"name": entity_name, "type": "unknown", "file": "unknown"}]
                                break

            if entities:
                entity_name = entities[0]['name']
                print(f"📍 Entité sélectionnée: {entity_name}")
            else:
                print("❌ Aucune entité trouvée pour la démonstration")
                return

        # Démonstration des différents contextes
        print(f"\n🎯 DÉMONSTRATION POUR L'ENTITÉ: {entity_name}")
        print("=" * 60)

        # 1. Contexte pour développeur
        print("\n1️⃣  CONTEXTE DÉVELOPPEUR (code_understanding)")
        print("-" * 50)
        try:
            dev_context = await self.get_agent_context(
                entity_name,
                agent_type="developer",
                task_context="code_understanding"
            )
            self._display_context_summary(dev_context, "Développeur")
        except Exception as e:
            print(f"❌ Erreur: {e}")

        # 2. Contexte pour reviewer
        print("\n2️⃣  CONTEXTE REVIEWER (code_review)")
        print("-" * 50)
        try:
            review_context = await self.get_agent_context(
                entity_name,
                agent_type="reviewer",
                task_context="code_review"
            )
            self._display_context_summary(review_context, "Reviewer")
        except Exception as e:
            print(f"❌ Erreur: {e}")

        # 3. Contexte pour analyzer
        print("\n3️⃣  CONTEXTE ANALYZER (bug_detection)")
        print("-" * 50)
        try:
            analyzer_context = await self.get_agent_context(
                entity_name,
                agent_type="analyzer",
                task_context="bug_detection"
            )
            self._display_context_summary(analyzer_context, "Analyzer")
        except Exception as e:
            print(f"❌ Erreur: {e}")

        # 4. Contextes individuels détaillés
        print("\n4️⃣  CONTEXTES INDIVIDUELS DÉTAILLÉS")
        print("-" * 50)

        # Contexte local
        try:
            local_ctx = await self.get_local_context(entity_name, 2000)
            self._display_local_context(local_ctx)
        except Exception as e:
            print(f"❌ Erreur contexte local: {e}")

        # Contexte global
        try:
            global_ctx = await self.get_global_context(entity_name, 2000)
            self._display_global_context(global_ctx)
        except Exception as e:
            print(f"❌ Erreur contexte global: {e}")

        # Contexte sémantique
        try:
            semantic_ctx = await self.get_semantic_context(entity_name, 1500)
            self._display_semantic_context(semantic_ctx)
        except Exception as e:
            print(f"❌ Erreur contexte sémantique: {e}")

    def _display_context_summary(self, context: Dict[str, Any], agent_type: str):
        """Affiche un résumé d'un contexte d'agent"""
        if "error" in context:
            print(f"❌ Erreur: {context['error']}")
            return

        print(f"📊 Résumé du contexte {agent_type}:")
        print(f"  Entité: {context.get('entity', 'N/A')}")
        print(f"  Tokens utilisés: {context.get('total_tokens', 0)}")

        # Insights
        insights = context.get('key_insights', [])
        print(f"  💡 Insights clés ({len(insights)}):")
        for insight in insights[:3]:
            print(f"    - {insight}")

        # Recommandations
        recommendations = context.get('recommendations', [])
        print(f"  📋 Recommandations ({len(recommendations)}):")
        for rec in recommendations[:3]:
            print(f"    - {rec}")

        # Informations de génération
        gen_info = context.get('generation_info', {})
        strategy = gen_info.get('strategy_used', {})
        if strategy:
            print(f"  ⚙️  Stratégie: Local({strategy.get('local_weight', 0):.1f}) "
                  f"Global({strategy.get('global_weight', 0):.1f}) "
                  f"Semantic({strategy.get('semantic_weight', 0):.1f})")

    def _display_local_context(self, context: Dict[str, Any]):
        """Affiche le contexte local en détail"""
        if "error" in context:
            print(f"❌ Contexte local - Erreur: {context['error']}")
            return

        print(f"\n📍 CONTEXTE LOCAL - {context.get('entity', 'N/A')}")

        # Définition principale
        main_def = context.get('main_definition', {})
        if main_def:
            print(f"  Type: {main_def.get('type', 'N/A')}")
            print(f"  Fichier: {Path(main_def.get('location', {}).get('file', '')).name}")
            print(f"  Signature: {main_def.get('signature', 'N/A')}")

        # Dépendances
        deps = context.get('immediate_dependencies', [])
        print(f"  📦 Dépendances ({len(deps)}):")
        for dep in deps[:3]:
            print(f"    - {dep.get('name', 'N/A')} ({dep.get('type', 'N/A')})")

        # Fonctions appelées
        calls = context.get('called_functions', [])
        print(f"  📞 Fonctions appelées ({len(calls)}):")
        for call in calls[:3]:
            print(f"    - {call.get('name', 'N/A')} ({call.get('source', 'N/A')})")

    def _display_global_context(self, context: Dict[str, Any]):
        """Affiche le contexte global en détail"""
        if "error" in context:
            print(f"❌ Contexte global - Erreur: {context['error']}")
            return

        print(f"\n🌍 CONTEXTE GLOBAL - {context.get('entity', 'N/A')}")

        # Vue d'ensemble du projet
        overview = context.get('project_overview', {})
        if overview:
            stats = overview.get('statistics', {})
            print(f"  📊 Projet: {stats.get('total_entities', 0)} entités, "
                  f"{stats.get('modules', 0)} modules, {stats.get('files', 0)} fichiers")

            arch_style = overview.get('architectural_style', '')
            if arch_style:
                print(f"  🏗️  Architecture: {arch_style}")

        # Analyse d'impact
        impact = context.get('impact_analysis', {})
        if impact:
            risk_level = impact.get('risk_level', 'unknown')
            affected_modules = len(impact.get('affected_modules', []))
            print(f"  ⚠️  Impact: Niveau {risk_level}, {affected_modules} modules affectés")

        # Modules liés
        related = context.get('related_modules', [])
        print(f"  🔗 Modules liés ({len(related)}):")
        for module in related[:3]:
            similarity = module.get('similarity', 0)
            print(f"    - {module.get('module', 'N/A')} (similarité: {similarity:.2f})")

    def _display_semantic_context(self, context: Dict[str, Any]):
        """Affiche le contexte sémantique en détail"""
        if "error" in context:
            print(f"❌ Contexte sémantique - Erreur: {context['error']}")
            return

        print(f"\n🧠 CONTEXTE SÉMANTIQUE - {context.get('entity', 'N/A')}")

        # Concepts principaux
        concepts = context.get('main_concepts', [])
        print(f"  🏷️  Concepts principaux ({len(concepts)}):")
        for concept in concepts:
            label = concept.get('label', 'N/A')
            confidence = concept.get('confidence', 0)
            print(f"    - {label} (confiance: {confidence:.2f})")

        # Entités similaires
        similar = context.get('similar_entities', [])
        print(f"  🔄 Entités similaires ({len(similar)}):")
        for entity in similar[:3]:
            similarity = entity.get('similarity', 0)
            print(f"    - {entity.get('name', 'N/A')} (similarité: {similarity:.2f})")

        # Patterns algorithmiques
        patterns = context.get('algorithmic_patterns', [])
        print(f"  🔬 Patterns algorithmiques ({len(patterns)}):")
        for pattern in patterns[:3]:
            pattern_name = pattern.get('pattern', 'N/A')
            score = pattern.get('score', 0)
            print(f"    - {pattern_name} (score: {score})")

    async def generate_module_readme(self, module_name: str) -> str:
        """Génère un README.md complet pour un module"""
        if not await self._ensure_initialized():
            return "Erreur: RAG non initialisé"

        # Créer le context provider de documentation
        doc_provider = DocumentationContextProvider(self)

        # Obtenir le contexte complet
        context = await doc_provider.get_module_readme_context(module_name)

        # Formatter le contexte pour le prompt
        context_str = self._format_context_for_readme(context)

        # Générer le README via LLM
        readme_prompt = README_TEMPLATE.format(context=context_str)

        print(readme_prompt)
        input("...wait...")
        self.rag_engine.llm_provider.set_system_prompt("Tu es un expert en documentation technique spécialisé en Fortran.")
        readme = await self.rag_engine.llm_provider.generate_response(
            messages=readme_prompt,
            max_tokens=4000
        )

        return readme

    def _format_context_for_readme(self, context: Dict[str, Any]) -> str:
        """Formate le contexte en texte structuré pour le prompt"""

        lines = [
            f"MODULE: {context['module_name']}",
            "",
            "DESCRIPTION:",
            f"  Purpose: {context['description']['purpose']}",
            f"  File: {context['description']['file_location']}",
            f"  Main functionality: {len(context['description']['main_functionality'])} procedures",
            "",
            "ARCHITECTURE:",
            f"  Role: {context['architecture']['role_in_project']}",
            f"  Impact level: {context['architecture']['impact_level']}",
            f"  Affects {context['architecture']['affected_modules']} other modules",
            "",
            "PUBLIC INTERFACE:",
        ]

        # Interface publique
        for proc in context['public_interface']['public_procedures']:
            lines.append(f"  - {proc['type']}: {proc['name']}")
            if proc['signature']:
                lines.append(f"    Signature: {proc['signature']}")

        # Dépendances
        lines.extend([
            "",
            "DEPENDENCIES:",
            f"  Coupling level: {context['dependencies']['coupling_level']}"
        ])

        for dep in context['dependencies']['direct_dependencies']:
            lines.append(f"  - {dep['name']} ({dep['type']})")

        # Concepts scientifiques
        if context['scientific_concepts']['main_concepts']:
            lines.extend([
                "",
                "SCIENTIFIC CONCEPTS:"
            ])
            for concept in context['scientific_concepts']['main_concepts']:
                lines.append(f"  - {concept['name']} (confidence: {concept['confidence']:.2f})")

        # Patterns algorithmiques
        if context['scientific_concepts']['algorithmic_patterns']:
            lines.extend([
                "",
                "ALGORITHMIC PATTERNS:"
            ])
            for pattern in context['scientific_concepts']['algorithmic_patterns']:
                lines.append(f"  - {pattern['pattern']}: {pattern['description']}")

        return "\n".join(lines)

    # === MÉTHODES EXISTANTES (inchangées) ===

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
            force_update: bool = False
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
                filepath, doc_id, additional_meta
            )

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
            documents_info: List[Dict[str, Any]],  # [{"filepath": ..., "project": ..., "version": ...}, ...]
            max_concurrent: int = 3,
            force_update: bool = False
    ) -> Dict[str, bool]:
        """Ajoute plusieurs documents en parallèle"""

        if not await self._ensure_initialized():
            return {}

        # Préparer les tâches
        semaphore = asyncio.Semaphore(max_concurrent)

        async def add_single_doc(doc_info):
            async with semaphore:
                filepath = doc_info["filepath"]
                try:
                    success = await self.add_document(
                        filepath,
                        project_name=doc_info.get("project_name"),
                        version=doc_info.get("version"),
                        additional_metadata=doc_info.get("additional_metadata"),
                        force_update=force_update
                    )
                    return filepath, success
                except Exception as e:
                    print(f"Erreur pour {filepath}: {e}")
                    return filepath, False

        # Lancer en parallèle
        tasks = [add_single_doc(doc_info) for doc_info in documents_info]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Organiser les résultats
        final_results = {}
        for result in results:
            if isinstance(result, Exception):
                print(f"Exception: {result}")
            else:
                filepath, success = result
                final_results[filepath] = success

        return final_results

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


class DocumentationContextProvider:
    """Orchestrateur spécialisé pour la génération de documentation"""

    def __init__(self, onto_rag_instance):
        self.onto_rag = onto_rag_instance
        self.rag_engine = onto_rag_instance.rag_engine

    async def get_module_readme_context(self, module_name: str) -> Dict[str, Any]:
        """Génère un contexte complet pour un README de module"""

        print(f"📚 Génération du contexte README pour le module: {module_name}")

        # 1. CONTEXTE ARCHITECTURAL (Global) - CORRECTION ICI
        print("🏗️ Analyse architecturale...")
        global_context = await self.onto_rag.get_global_context(module_name, 2000)

        # 2. INTERFACE ET DÉPENDANCES (Local) - CORRECTION ICI
        print("🔌 Analyse de l'interface...")
        local_context = await self.onto_rag.get_local_context(module_name, 2000)

        # 3. CONCEPTS SCIENTIFIQUES (Semantic) - CORRECTION ICI
        print("🧬 Analyse des concepts...")
        semantic_context = await self.onto_rag.get_semantic_context(module_name, 1500)

        # 4. RECHERCHE ENRICHIE avec le classifier
        print("🎯 Recherche contextuelle...")
        if hasattr(self.onto_rag, 'classifier') and self.onto_rag.classifier:
            search_result = await self.onto_rag.classifier.search_with_concepts(
                query=f"module {module_name} documentation interface usage",
                top_k=8,
                concept_weight=0.3
            )
        else:
            search_result = {}

        # 5. ASSEMBLAGE INTELLIGENT
        readme_context = await self._assemble_readme_context(
            module_name, global_context, local_context, semantic_context, search_result
        )

        return readme_context

    async def _assemble_readme_context(self, module_name: str,
                                       global_ctx: Dict, local_ctx: Dict,
                                       semantic_ctx: Dict, search_result: Dict) -> Dict[str, Any]:
        """Assemble toutes les informations en un contexte structuré pour README"""

        context = {
            "module_name": module_name,
            "description": await self._extract_module_description(local_ctx, search_result),
            "architecture": await self._extract_architectural_info(global_ctx),
            "public_interface": await self._extract_public_interface(local_ctx),
            "dependencies": await self._extract_dependencies_info(local_ctx, global_ctx),
            "scientific_concepts": await self._extract_scientific_concepts(semantic_ctx),
            "usage_examples": await self._extract_usage_examples(search_result),
            "internal_architecture": await self._extract_internal_architecture(local_ctx),
            "maintenance_info": await self._extract_maintenance_info(global_ctx),
            "related_modules": await self._extract_related_modules(global_ctx, semantic_ctx)
        }

        return context

    async def _extract_module_description(self, local_ctx: Dict, search_result: Dict) -> Dict[str, Any]:
        """Extrait la description du module"""
        description = {
            "purpose": "Unknown",
            "main_functionality": [],
            "file_location": "",
            "concepts_detected": []
        }

        if 'main_definition' in local_ctx:
            main_def = local_ctx['main_definition']
            description.update({
                "file_location": main_def.get('location', {}).get('file', ''),
                "signature": main_def.get('signature', ''),
                "concepts_detected": [c.get('label', '') for c in main_def.get('concepts', [])]
            })

        # Extraire la fonctionnalité depuis les enfants
        children = local_ctx.get('children_context', [])
        description["main_functionality"] = [
            f"{child.get('type', 'procedure')} {child.get('name', 'unknown')}"
            for child in children[:10]  # Top 10
        ]

        # Générer une description intelligente
        if description["concepts_detected"]:
            concepts_str = ", ".join(description["concepts_detected"][:3])
            description["purpose"] = f"Module implementing {concepts_str} functionality"
        elif description["main_functionality"]:
            func_count = len(description["main_functionality"])
            description["purpose"] = f"Module providing {func_count} procedures for computational tasks"

        return description

    async def _extract_architectural_info(self, global_ctx: Dict) -> Dict[str, Any]:
        """Extrait les informations architecturales"""
        arch_info = {
            "role_in_project": "Unknown",
            "impact_level": "low",
            "affected_modules": 0,
            "architectural_style": "unknown"
        }

        if 'impact_analysis' in global_ctx:
            impact = global_ctx['impact_analysis']
            arch_info.update({
                "impact_level": impact.get('risk_level', 'low'),
                "affected_modules": len(impact.get('affected_modules', [])),
                "recommendations": impact.get('recommendations', [])
            })

        if 'project_overview' in global_ctx:
            overview = global_ctx['project_overview']
            arch_info["architectural_style"] = overview.get('architectural_style', 'unknown')

        # Interpréter le rôle
        if arch_info["impact_level"] == "high":
            arch_info["role_in_project"] = "Core/Foundation module"
        elif arch_info["affected_modules"] > 3:
            arch_info["role_in_project"] = "Shared utility module"
        else:
            arch_info["role_in_project"] = "Specialized module"

        return arch_info

    async def _extract_public_interface(self, local_ctx: Dict) -> Dict[str, Any]:
        """Extrait l'interface publique du module"""
        interface = {
            "public_procedures": [],
            "public_types": [],
            "public_parameters": [],
            "exported_interfaces": []
        }

        # Analyser les enfants pour identifier l'interface publique
        children = local_ctx.get('children_context', [])

        for child in children:
            child_type = child.get('type', '').lower()
            child_name = child.get('name', 'unknown')
            child_signature = child.get('signature', '')

            if child_type in ['subroutine', 'function']:
                interface["public_procedures"].append({
                    "name": child_name,
                    "type": child_type,
                    "signature": child_signature,
                    "summary": child.get('summary', '')[:100] + "..." if child.get('summary') else ""
                })
            elif child_type in ['type', 'type_definition']:
                interface["public_types"].append({
                    "name": child_name,
                    "signature": child_signature
                })

        return interface

    async def _extract_dependencies_info(self, local_ctx: Dict, global_ctx: Dict) -> Dict[str, Any]:
        """Extrait les informations de dépendances"""
        deps_info = {
            "direct_dependencies": [],
            "dependency_analysis": "",
            "coupling_level": "unknown"
        }

        # Dépendances directes
        immediate_deps = local_ctx.get('immediate_dependencies', [])
        deps_info["direct_dependencies"] = [
            {
                "name": dep.get('name', 'unknown'),
                "type": dep.get('type', 'module'),
                "summary": dep.get('summary', '')[:100] + "..." if dep.get('summary') else ""
            }
            for dep in immediate_deps[:5]  # Top 5
        ]

        # Analyse du couplage
        deps_count = len(immediate_deps)
        if deps_count == 0:
            deps_info["coupling_level"] = "Independent (no dependencies)"
        elif deps_count <= 2:
            deps_info["coupling_level"] = "Low coupling"
        elif deps_count <= 4:
            deps_info["coupling_level"] = "Moderate coupling"
        else:
            deps_info["coupling_level"] = f"High coupling ({deps_count} dependencies)"

        return deps_info

    async def _extract_scientific_concepts(self, semantic_ctx: Dict) -> Dict[str, Any]:
        """Extrait les concepts scientifiques"""
        scientific_info = {
            "main_concepts": [],
            "algorithmic_patterns": [],
            "scientific_domain": "Computational",
            "mathematical_methods": []
        }

        # Concepts principaux
        main_concepts = semantic_ctx.get('main_concepts', [])
        scientific_info["main_concepts"] = [
            {
                "name": concept.get('label', 'unknown'),
                "confidence": concept.get('confidence', 0)
            }
            for concept in main_concepts[:5]
        ]

        # Patterns algorithmiques
        patterns = semantic_ctx.get('algorithmic_patterns', [])
        scientific_info["algorithmic_patterns"] = [
            {
                "pattern": pattern.get('pattern', 'unknown'),
                "description": pattern.get('description', ''),
                "score": pattern.get('score', 0)
            }
            for pattern in patterns[:3]
        ]

        # Déterminer le domaine scientifique
        concept_labels = [c.get('label', '').lower() for c in main_concepts]

        if any('quantum' in label or 'electron' in label for label in concept_labels):
            scientific_info["scientific_domain"] = "Quantum Chemistry/Physics"
        elif any('molecular' in label or 'dynamics' in label for label in concept_labels):
            scientific_info["scientific_domain"] = "Molecular Dynamics"
        elif any('matrix' in label or 'linear' in label for label in concept_labels):
            scientific_info["scientific_domain"] = "Linear Algebra/Numerical Methods"
        elif any('fft' in label or 'fourier' in label for label in concept_labels):
            scientific_info["scientific_domain"] = "Signal Processing/Spectral Methods"

        return scientific_info

    async def _extract_usage_examples(self, search_result: Dict) -> Dict[str, Any]:
        """Extrait des exemples d'utilisation depuis les passages trouvés"""
        usage = {
            "example_calls": [],
            "typical_workflow": [],
            "integration_patterns": []
        }

        if 'sources' in search_result:
            for source in search_result['sources'][:3]:  # Top 3 sources
                entity_name = source.get('entity_name', '')
                entity_type = source.get('entity_type', '')

                if entity_type in ['subroutine', 'function']:
                    usage["example_calls"].append(f"call {entity_name}(...)")
                elif entity_type == 'module':
                    usage["integration_patterns"].append(f"use {entity_name}")

        return usage

    async def _extract_internal_architecture(self, local_ctx: Dict) -> Dict[str, Any]:
        """Extrait l'architecture interne"""
        internal = {
            "internal_procedures": [],
            "data_structures": [],
            "complexity_analysis": ""
        }

        # Analyser les fonctions appelées pour comprendre la structure interne
        called_functions = local_ctx.get('called_functions', [])
        internal["internal_procedures"] = [
            {
                "name": func.get('name', 'unknown'),
                "source": func.get('source', 'unknown')
            }
            for func in called_functions[:5]
        ]

        # Analyse de complexité
        calls_count = len(called_functions)
        if calls_count == 0:
            internal["complexity_analysis"] = "Simple module with minimal internal dependencies"
        elif calls_count <= 3:
            internal["complexity_analysis"] = "Moderate complexity with few internal calls"
        else:
            internal["complexity_analysis"] = f"Complex module with {calls_count} internal function calls"

        return internal

    async def _extract_maintenance_info(self, global_ctx: Dict) -> Dict[str, Any]:
        """Extrait les informations de maintenance"""
        maintenance = {
            "impact_analysis": {},
            "testing_recommendations": [],
            "refactoring_opportunities": []
        }

        if 'impact_analysis' in global_ctx:
            impact = global_ctx['impact_analysis']
            maintenance["impact_analysis"] = {
                "risk_level": impact.get('risk_level', 'unknown'),
                "affected_modules": impact.get('affected_modules', []),
                "recommendations": impact.get('recommendations', [])
            }

        # Générer des recommandations de test
        risk_level = maintenance["impact_analysis"].get('risk_level', 'low')
        if risk_level == 'high':
            maintenance["testing_recommendations"] = [
                "Comprehensive unit testing required",
                "Integration testing with dependent modules",
                "Performance regression testing"
            ]
        elif risk_level == 'medium':
            maintenance["testing_recommendations"] = [
                "Unit testing for public interface",
                "Basic integration testing"
            ]
        else:
            maintenance["testing_recommendations"] = [
                "Basic unit testing sufficient"
            ]

        return maintenance

    async def _extract_related_modules(self, global_ctx: Dict, semantic_ctx: Dict) -> Dict[str, Any]:
        """Extrait les modules liés"""
        related = {
            "similar_modules": [],
            "dependent_modules": [],
            "semantic_relatives": []
        }

        # Modules similaires depuis le contexte global
        if 'related_modules' in global_ctx:
            related["similar_modules"] = [
                {
                    "name": mod.get('module', 'unknown'),
                    "similarity": mod.get('similarity', 0),
                    "concepts": mod.get('concepts', [])
                }
                for mod in global_ctx['related_modules'][:3]
            ]

        # Entités similaires depuis le contexte sémantique
        if 'similar_entities' in semantic_ctx:
            related["semantic_relatives"] = [
                {
                    "name": entity.get('name', 'unknown'),
                    "similarity": entity.get('similarity', 0),
                    "file": entity.get('file', 'unknown')
                }
                for entity in semantic_ctx['similar_entities'][:3]
            ]

        return related


# PROMPT TEMPLATE pour README
README_TEMPLATE = """Tu es un expert en documentation technique. Génère un README.md professionnel pour ce module Fortran.

CONTEXTE DU MODULE:
{context}

INSTRUCTIONS:
1. Crée un README.md complet et professionnel
2. Utilise UNIQUEMENT les informations fournies dans le contexte
3. Structure avec des sections claires et des exemples concrets
4. Inclus des badges de statut si approprié
5. Ajoute une section "Usage" avec des exemples réalistes basés sur l'interface publique
6. Mentionne les concepts scientifiques détectés
7. Inclus une section sur l'architecture et les dépendances
8. Ajoute des recommandations de maintenance si pertinentes

FORMAT ATTENDU:
- Titre et description
- Badges de statut
- Table des matières
- Description détaillée
- Installation/Compilation
- Usage avec exemples
- Interface publique (API)
- Architecture et dépendances  
- Concepts scientifiques
- Maintenance et développement
- Modules liés
- Contribution guidelines

Utilise la syntaxe Markdown et rends le README attractif et informatif."""


# Exemple d'utilisation modifié avec tests du système de contexte
async def example_usage():
    """Exemple d'utilisation d'OntoRAG avec tests du système de contexte"""

    rag = OntoRAG(
        storage_dir=STORAGE_DIR,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        ontology_path=ONTOLOGY_PATH_TTL
    )

    await rag.initialize()

    documents_info = [
        {
            "filepath": "/home/yopla/dynamic_molecular/simulation_main.f90",
            "project_name": "BigDFT",
            "version": "1.9"
        },
        {
            "filepath": "/home/yopla/dynamic_molecular/integrator.f90",
            "project_name": "BigDFT",
            "version": "1.9"
        },
        {
            "filepath": "/home/yopla/dynamic_molecular/force_calculations.f90",
            "project_name": "BigDFT",
            "version": "1.9"
        },
        {
            "filepath": "/home/yopla/dynamic_molecular/math_utilities.f90",
            "project_name": "BigDFT",
            "version": "1.9"
        },
        {
            "filepath": "/home/yopla/dynamic_molecular/constants_types.f90",
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

    # NOUVEAU : Test du système de contexte
    print("\n" + "=" * 80)
    print("🧪 TESTS DU SYSTÈME DE CONTEXTE")
    print("=" * 80)

    # Test complet du système
    test_results = await rag.test_context_system()

    if test_results.get("performance", {}).get("success_rate", 0) > 50:
        print("\n✅ Tests du système de contexte réussis - Démarrage de la démonstration")

        # Démonstration interactive
        await rag.demo_context_usage()
    else:
        print("\n⚠️ Tests du système de contexte partiellement réussis")

    # Interface interactive pour les requêtes ET les contextes
    print("\n" + "=" * 80)
    print("💬 INTERFACE INTERACTIVE")
    print("=" * 80)
    print("Commandes disponibles:")
    print("  - question normale : requête RAG classique")
    print("  - /context <entité> : contexte pour développeur")
    print("  - /review <entité> : contexte pour reviewer")
    print("  - /analyze <entité> : contexte pour analyzer")
    print("  - /search <terme> : recherche d'entités")
    print("  - /debug <entité> : informations de debug")
    print("  - /refresh : refait l'indexation des contexts")
    print("  - /debug-chunks : informations de debug")
    print("  - /stats : statistiques du système")
    print("  - /quick : diagnostique")
    print("  - /demo <entité> : démonstration complète")
    print("  - /visualization : page html de graphique de dépendances")
    print("  - /test_context : test")
    print("  - /quit : quitter")


    while True:
        try:
            query = input('\n💬 Votre commande : ').strip()

            if query.lower() in ['/quit', '/exit', 'quit', 'exit']:
                print("👋 Au revoir !")
                break

            elif query.startswith('/context '):
                entity_name = query[9:].strip()
                if entity_name:
                    context = await rag.get_agent_context(entity_name, "developer", "code_understanding")
                    rag._display_context_summary(context, "Développeur")
                else:
                    print("❌ Veuillez spécifier un nom d'entité")

            elif query.startswith('/review '):
                entity_name = query[8:].strip()
                if entity_name:
                    context = await rag.get_agent_context(entity_name, "reviewer", "code_review")
                    rag._display_context_summary(context, "Reviewer")
                else:
                    print("❌ Veuillez spécifier un nom d'entité")

            elif query.startswith('/analyze '):
                entity_name = query[9:].strip()
                if entity_name:
                    context = await rag.get_agent_context(entity_name, "analyzer", "bug_detection")
                    rag._display_context_summary(context, "Analyzer")
                else:
                    print("❌ Veuillez spécifier un nom d'entité")

            elif query.startswith('/search '):
                search_term = query[8:].strip()
                if search_term:
                    entities = await rag.search_entities(search_term)
                    print(f"🔍 Trouvé {len(entities)} entités pour '{search_term}':")
                    for i, entity in enumerate(entities[:10], 1):
                        print(f"  {i}. {entity['name']} ({entity['type']}) in {Path(entity['file']).name}")
                else:
                    print("❌ Veuillez spécifier un terme de recherche")

            elif query.startswith('/visualization'):
                visualizer = await create_fortran_dependency_visualization(
                    onto_rag_instance=rag,
                    output_file="my_dependencies.html",
                    hierarchical=False
                )
                print('Ouverture du browser et génération de la page html my_dependencies.html')

            elif query.startswith('/test_context '):
                entity_name = query[10:].strip()
                if entity_name:
                    await rag.test_context_llm(entity_name)
                else:
                    print("❌ Veuillez spécifier un nom d'entité")

            elif query.startswith('/debug '):
                entity_name = query[7:].strip()
                if entity_name:
                    debug_info = rag.debug_entity(entity_name)
                    print(f"🐛 Debug pour '{entity_name}':")
                    print(json.dumps(debug_info, indent=2))
                else:
                    print("❌ Veuillez spécifier un nom d'entité")

            elif query == '/stats':
                context_stats = rag.get_context_stats()
                print("📊 Statistiques du système de contexte:")
                print(json.dumps(context_stats, indent=2))

            elif query.startswith('/demo '):
                entity_name = query[6:].strip()
                await rag.demo_context_usage(entity_name)

            elif query == '/demo':
                await rag.demo_context_usage()

            elif query == '/refresh':
                if rag.context_provider:
                    await rag.refresh_context_index()
                else:
                    await rag.initialize_context_provider()

            elif query == '/debug-chunks':
                chunks_debug = await rag.debug_chunks_metadata()
                print("🔍 Debug des chunks:")
                print(json.dumps(chunks_debug, indent=2))

            elif query.startswith('/semantic '):
                entity_name = query[10:].strip()
                if entity_name:
                    context = await rag.get_semantic_context(entity_name, 2000)
                    rag._display_semantic_context(context)
                else:
                    print("❌ Veuillez spécifier un nom d'entité")

            elif query == '/quick':
                await rag.quick_diagnose()

            elif query.startswith('/readme '):
                module_name = query[8:].strip()
                if module_name:
                    print(f"📚 Génération du README pour le module: {module_name}")
                    readme = await rag.generate_module_readme(module_name)

                    # Sauvegarder le fichier
                    filename = f"README_{module_name}.md"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(readme)

                    print(f"✅ README généré: {filename}")
                    print("\n" + "=" * 50)
                    print(readme[:1000] + "..." if len(readme) > 1000 else readme)
                    print("=" * 50)
                else:
                    print("❌ Veuillez spécifier un nom de module")

            elif query.startswith('/'):
                print("❌ Commande inconnue. Tapez une question ou utilisez /quit pour quitter")

            else:
                # Requête RAG classique
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

        except KeyboardInterrupt:
            print("\n👋 Au revoir !")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")


if __name__ == "__main__":
    asyncio.run(example_usage())