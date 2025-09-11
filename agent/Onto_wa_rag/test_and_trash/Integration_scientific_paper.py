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
from ontology.classifier import OntologyClassifier
from ontology.ontology_manager import OntologyManager
#from utils.f2p_fortran_semantic_chunker import F2pyFortranSemanticChunker
from utils.ontologie_fortran_chunker import OFPFortranSemanticChunker as F2pyFortranSemanticChunker
from utils.sci_paper_chunker import SciPaperSemanticChunker
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

        self.paper_chunker = SciPaperSemanticChunker(
            min_chunk_size=400, max_chunk_size=chunk_size)

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

            chunks = await self.fortran_chunker.create_fortran_chunks_with_debug(
                text_content, document_id, filepath, metadata,
                self.ontology_manager, create_debug_file=True
            )
            """
            chunks = await self.fortran_chunker.create_fortran_chunks(
                text_content, document_id, filepath, metadata, self.ontology_manager
            )
            """

        elif file_type in ['pdf', 'document', 'markdown', 'text']:
            print("✂️  Chunker scientifique")
            chunks = await self.paper_chunker.create_paper_chunks(
                text_content, document_id, filepath, metadata)

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

        print("✅ OntoRAG initialisé avec succès!")

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
