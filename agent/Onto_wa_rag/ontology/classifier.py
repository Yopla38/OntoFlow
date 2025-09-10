"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# ontology/classifier.py
import os
import asyncio
import pickle
import re

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set

from tqdm import tqdm

from ..CONSTANT import RELATION_MODEL_TYPE, GREEN, YELLOW, RESET, BLUE
from .concept_hopfield import ConceptHopfieldClassifier
from .hopfield_network import HopfieldClassifier

# ontology/classifier.py
from typing import List, Dict, Any, Optional, Tuple, Set

from .hopfield_network import HopfieldClassifier
from .hierarchical_hopfield import HierarchicalHopfieldClassifier


# --- utils pour connaître la spécificité (déjà présent) -------------
SPECIFICITY_RANK = {
    'module': 1, 'program': 1,
    'type_definition': 2, 'interface': 3,
    'subroutine': 4, 'function': 4,
    'internal_function': 5
}

# Pour documentation scientifique
SPECIFICITY_RANK.update({
    'article': 1,
    'section': 2,
    'figure': 2,
    'reference_list': 2,
    'citation': 3
})


def _spec(chunk_meta):
    return SPECIFICITY_RANK.get(chunk_meta.get('entity_type', ''), 3)


class OntologyClassifier:
    """
    Système de classification ontologique pour le RAG.
    """

    def __init__(
            self,
            rag_engine,
            ontology_manager,
            storage_dir: str = "ontology_classifier",
            use_hierarchical: bool = True,  # Utiliser le classifieur hiérarchique par défaut
            enable_concept_classification: bool = True,  # Activer la classification par concepts
            enable_relation_learning: bool = False,
            multiscale_mode: bool = True,
            wavelet_config=None
    ):
        """
        Initialise le classifieur ontologique intégré au RAG.

        Args:
            rag_engine: Moteur RAG existant
            ontology_manager: Gestionnaire d'ontologie
            storage_dir: Répertoire de stockage pour les modèles et données
            use_hierarchical: Si True, utilise le classifieur hiérarchique multiniveau
            enable_concept_classification: Si True, active la classification par concepts
        """
        if wavelet_config is None:
            wavelet_config = {'wavelet': 'coif3', 'levels': 3}

        self.rag_engine = rag_engine
        self.ontology_manager = ontology_manager
        self.storage_dir = storage_dir
        self.use_hierarchical = use_hierarchical

        # Créer les répertoires de stockage
        os.makedirs(storage_dir, exist_ok=True)

        # Initialiser le classifieur approprié
        if use_hierarchical:
            hierarchical_dir = os.path.join(storage_dir, "hierarchical_models")
            self.classifier = HierarchicalHopfieldClassifier(
                rag_engine=rag_engine,
                ontology_manager=ontology_manager,
                storage_dir=hierarchical_dir
            )
        else:
            # Fallback vers le classifieur simple (non-hiérarchique)
            hopfield_dir = os.path.join(storage_dir, "hopfield_models")
            self.classifier = HopfieldClassifier(
                rag_engine=rag_engine,
                ontology_manager=ontology_manager,
                storage_dir=hopfield_dir,
                beta=22
            )

        # Cache des classifications
        self.classification_cache = {}

        self.enable_concept_classification = enable_concept_classification
        if enable_concept_classification:
            concept_dir = os.path.join(storage_dir, "concept_models")
            self.concept_classifier = ConceptHopfieldClassifier(
                rag_engine=rag_engine,
                ontology_manager=ontology_manager,
                storage_dir=concept_dir,
                multiscale_mode=multiscale_mode,
                wavelet_config=wavelet_config
            )

            # Cache pour les résultats de classification par concepts
            self.concept_classification_cache = {}
            self._extra_biblio_concepts = [
                "http://example.org/biblio#Article",
                "http://example.org/biblio#Section",
                "http://example.org/biblio#Figure",
                "http://example.org/biblio#Citation",
                "http://example.org/biblio#ReferenceList"
            ]

        else:
            self.concept_classifier = None

        # Ajouter le gestionnaire de relations si activé
        self.enable_relation_learning = enable_relation_learning
        if enable_relation_learning:
            relation_dir = os.path.join(storage_dir, "relation_models")
            from .simplified_hopfield_relation import SimplifiedRelationManager
            self.relation_manager = SimplifiedRelationManager(
                ontology_manager=ontology_manager,
                storage_dir=relation_dir,
            )
        else:
            self.relation_manager = None

    async def initialize(self) -> None:
        """Initialise le classifieur en chargeant les modèles."""
        await self.classifier.initialize()

        # Initialiser le classifieur de concepts si activé
        if self.concept_classifier:
            await self.concept_classifier.initialize()

        # Initialiser le gestionnaire de relations si activé
        if self.enable_relation_learning and self.relation_manager:
            await self.relation_manager.initialize()

        # ------------------------------------------------------------------
        #  Injecte les concepts bibliographiques si absents
        # ------------------------------------------------------------------
        if self.concept_classifier:
            missing = [uri for uri in self._extra_biblio_concepts
                       if uri not in self.concept_classifier.concept_embeddings]
            if missing:
                print(f"📚  Ajout des concepts biblio : {len(missing)} à entraîner")
                # TODO Fonction inexistante. A FAIRE
                try:
                    await self.concept_classifier.add_new_concepts(missing)
                except Exception as e:
                    print(f"❌ Aucune injection de concept {e}")

    async def full_initialize(self, train_relations: bool = True, min_relation_examples: int = 5) -> None:
        """
        Initialisation complète incluant l'entraînement des relations.

        Args:
            train_relations: Si True, entraîne les relations après l'initialisation
            min_relation_examples: Nombre minimum d'exemples pour entraîner une relation
        """
        print("🚀 Initialisation complète du système ontologique...")

        # 1. Initialisation de base
        await self.initialize()

        # 2. Vérifier que les concepts sont entraînés
        if self.concept_classifier and not self.concept_classifier.concept_embeddings:
            print("⚠️ Les concepts ne sont pas encore entraînés. Entraînement requis avant les relations.")
            return

        # 3. Entraîner les relations si demandé
        if train_relations and self.enable_relation_learning:
            print("🔗 Entraînement des relations à partir des documents...")

            try:
                results = await self.train_relations_from_documents(min_examples=min_relation_examples)

                trained_count = sum(1 for success in results.values() if success)
                total_count = len(results)

                print(f"✅ Entraînement des relations terminé: {trained_count}/{total_count} relations apprises")

                # Afficher les statistiques
                if trained_count > 0:
                    stats = await self.get_relation_statistics()
                    print(f"📊 Statistiques des relations: {stats}")

            except Exception as e:
                print(f"❌ Erreur lors de l'entraînement des relations: {e}")

        print("✅ Initialisation complète terminée")

    async def classify_text_direct(self, text: str, min_confidence: float = 0.3) -> List[Dict[str, Any]]:
        """
        Prend un texte brut, calcule son embedding, et le classifie pour trouver les concepts pertinents.
        C'est le point d'entrée pour l'enrichissement des chunks.
        """

        # 1. Vérifier que le classifieur de concepts est bien activé et disponible.
        if not self.concept_classifier:
            # Cette condition correspond exactement à votre log d'erreur, mais ne devrait plus se déclencher.
            return []

        # 2. Obtenir l'embedding pour le texte fourni.
        #    On utilise une méthode helper pour garder le code propre.
        embedding = await self._get_text_embedding(text)
        if embedding is None:
            # Si on n'a pas pu générer d'embedding, on ne peut pas continuer.
            return []

        # 3. Utiliser la méthode `auto_detect_concepts` du `ConceptHopfieldClassifier`.
        #    C'est la méthode la plus appropriée car elle est conçue pour
        #    classifier un embedding "à la volée".
        try:
            detected_concepts = await self.concept_classifier.auto_detect_concepts(
                query_embedding=embedding,
                min_confidence=min_confidence
            )
            if detected_concepts:
                concept_labels = [concept.get('label', 'SansLabel') for concept in detected_concepts]
                print(
                    f"✅ Concepts pour le chunk: {', '.join(concept_labels)}"
                )
            return detected_concepts
        except Exception as e:
            print(f"Erreur durant auto_detect_concepts: {e}")
            return []


    # ---------------------- RELATIONS ----------------------
    async def extract_possible_relations(self, text: str, confidence_threshold: float = 0.65) -> List[Dict[str, Any]]:
        """
        Extrait les relations possibles entre concepts dans un texte.
        """
        if not self.enable_relation_learning or not self.relation_manager:
            return []

        # 1. Détecter les concepts dans le texte
        query_embedding = await self._get_text_embedding(text)
        if query_embedding is None:
            return []

        if not self.concept_classifier:
            return []

        concept_matches = await self.concept_classifier.auto_detect_concepts(
            query_embedding,
            min_confidence=confidence_threshold
        )

        if len(concept_matches) < 2:  # Il faut au moins 2 concepts pour une relation
            return []

        # 2. Récupérer les embeddings des concepts détectés
        concept_embeddings = {}
        for concept in concept_matches:
            concept_uri = concept["concept_uri"]
            if concept_uri in self.concept_classifier.concept_embeddings:
                concept_embeddings[concept_uri] = self.concept_classifier.concept_embeddings[concept_uri]

        # 3. Générer tous les triplets possibles entre les concepts détectés
        possible_relations = []

        # Pour chaque paire de concepts
        for i, subject_concept in enumerate(concept_matches):
            for j, object_concept in enumerate(concept_matches):
                if i != j:  # Éviter les auto-relations
                    subject_uri = subject_concept["concept_uri"]
                    object_uri = object_concept["concept_uri"]

                    # Tester chaque relation entraînée
                    for relation_uri, transform in self.relation_manager.transformations.items():
                        if transform.is_trained:
                            # Prédire si cette relation existe
                            predictions = transform.predict_objects(
                                subject_uri,
                                concept_embeddings,
                                top_k=5,
                                threshold=0.4
                            )

                            # Vérifier si l'objet prédit correspond
                            for pred in predictions:
                                if pred["concept_uri"] == object_uri:
                                    possible_relations.append({
                                        "subject": subject_concept["label"],
                                        "subject_uri": subject_uri,
                                        "relation": transform.label,
                                        "relation_uri": relation_uri,
                                        "object": object_concept["label"],
                                        "object_uri": object_uri,
                                        "confidence": pred["confidence"],
                                        "source": "learned_relation"
                                    })

        # Trier par confiance
        possible_relations.sort(key=lambda x: x["confidence"], reverse=True)
        return possible_relations[:10]  # Top 10

    async def learn_relations_from_examples(self, relation_examples: Dict[str, List[Tuple[str, str, str]]],
                                            force_relearn: bool = False) -> Dict[str, bool]:
        """
        Apprend les transformations de relations à partir d'exemples de triplets.
        """
        if not self.enable_relation_learning or not self.relation_manager:
            return {}

        # Convertir les triplets en exemples (sujet, objet) par relation
        examples_by_relation = {}
        for relation_uri, triplets in relation_examples.items():
            examples_by_relation[relation_uri] = [(triplet[0], triplet[2]) for triplet in triplets]

        # Utiliser les embeddings des concepts
        concept_embeddings = {}
        if self.concept_classifier and self.concept_classifier.concept_embeddings:
            concept_embeddings = self.concept_classifier.concept_embeddings

        # Apprendre les relations
        results = await self.relation_manager.learn_relations(
            triples=[(s, r, o) for r, triplets in relation_examples.items() for s, _, o in triplets],
            concept_embeddings=concept_embeddings,
            min_examples=3,
            force_relearn=force_relearn
        )

        return results

    async def get_related_concepts(self, concept_uri: str, relation_uri: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Récupère les concepts liés à un concept via une relation.
        """
        if not self.enable_relation_learning or not self.relation_manager:
            return []

        concept_embeddings = {}
        if self.concept_classifier and self.concept_classifier.concept_embeddings:
            concept_embeddings = self.concept_classifier.concept_embeddings

        results = self.relation_manager.get_related_concepts(
            concept_uri, relation_uri, concept_embeddings, top_k
        )

        return results

    async def infer_new_relations(self, subject_uri: str = None, confidence_threshold: float = 0.5) -> List[
        Dict[str, Any]]:
        """
        Infère de nouvelles relations pour un concept ou tous les concepts.
        """
        if not self.enable_relation_learning or not self.relation_manager:
            return []

        concept_embeddings = {}
        if self.concept_classifier and self.concept_classifier.concept_embeddings:
            concept_embeddings = self.concept_classifier.concept_embeddings

        if subject_uri:
            # Inférer pour un concept spécifique
            return self.relation_manager.infer_new_relations(
                subject_uri, concept_embeddings, confidence_threshold
            )
        else:
            # Inférer pour tous les concepts
            all_inferences = []
            for concept_uri in concept_embeddings.keys():
                inferences = self.relation_manager.infer_new_relations(
                    concept_uri, concept_embeddings, confidence_threshold
                )
                all_inferences.extend(inferences)

            # Trier par confiance
            all_inferences.sort(key=lambda x: x["confidence"], reverse=True)
            return all_inferences[:50]  # Top 50

    async def search_by_relation(self, query: str, subject_concept: str, relation_uri: str,
                                 top_k: int = 5) -> Dict[str, Any]:
        """
        Recherche basée sur une relation spécifique entre concepts.
        """
        if not self.enable_relation_learning or not self.relation_manager:
            return {"error": "Apprentissage de relations non disponible"}

        # Résoudre l'URI du concept sujet
        subject_uri = self.ontology_manager.resolve_uri(subject_concept)

        # Obtenir les concepts liés via la relation
        related_concepts = await self.get_related_concepts(subject_uri, relation_uri, top_k=10)

        if not related_concepts:
            return {"error": f"Aucun concept lié trouvé pour {subject_concept} via {relation_uri}"}

        # Rechercher des documents pertinents pour ces concepts liés
        relevant_documents = []

        for related_concept in related_concepts:
            related_uri = related_concept["concept_uri"]

            # Rechercher des documents contenant ce concept
            concept_result = await self.search_by_concept(
                query=query,
                concept_uri=related_uri,
                include_subconcepts=True,
                top_k=3,
                confidence_threshold=0.4
            )

            if "passages" in concept_result:
                for passage in concept_result["passages"]:
                    passage["related_concept"] = related_concept["label"]
                    passage["relation_confidence"] = related_concept["confidence"]
                    relevant_documents.append(passage)

        if not relevant_documents:
            return {"error": "Aucun document pertinent trouvé"}

        # Trier par pertinence
        relevant_documents.sort(key=lambda x: x["similarity"], reverse=True)
        top_passages = relevant_documents[:top_k]

        # Générer une réponse contextuelle
        subject_label = self.ontology_manager.concepts.get(subject_uri, {}).get('label', subject_concept)
        relation_label = self.relation_manager.transformations.get(relation_uri,
                                                                   {}).label if relation_uri in self.relation_manager.transformations else relation_uri

        system_prompt = f"""
        Tu réponds à une question concernant les concepts liés à "{subject_label}" 
        via la relation "{relation_label}".

        Concepts liés trouvés: {', '.join([rc['label'] for rc in related_concepts[:5]])}

        Concentre-toi sur ces relations spécifiques dans ta réponse.
        """

        answer = await self.rag_engine.generate_answer(query, top_passages, system_prompt)

        return {
            "answer": answer,
            "passages": top_passages,
            "subject_concept": subject_label,
            "relation": relation_label,
            "related_concepts": [rc["label"] for rc in related_concepts],
            "relation_confidence": [rc["confidence"] for rc in related_concepts]
        }

    async def discover_concept_relations(self, concept_uri: str, min_confidence: float = 0.6) -> Dict[str, Any]:
        """
        Découvre toutes les relations possibles pour un concept donné.
        """
        if not self.enable_relation_learning or not self.relation_manager:
            return {"error": "Apprentissage de relations non disponible"}

        # Résoudre l'URI
        full_concept_uri = self.ontology_manager.resolve_uri(concept_uri)

        # Obtenir les embeddings
        concept_embeddings = {}
        if self.concept_classifier and self.concept_classifier.concept_embeddings:
            concept_embeddings = self.concept_classifier.concept_embeddings

        if full_concept_uri not in concept_embeddings:
            return {"error": f"Concept {concept_uri} non trouvé dans les embeddings"}

        # Découvrir les relations
        discovered_relations = {}

        for relation_uri, transform in self.relation_manager.transformations.items():
            if not transform.is_trained:
                continue

            # Prédire les objets pour cette relation
            predictions = transform.predict_objects(
                full_concept_uri,
                concept_embeddings,
                top_k=5,
                threshold=min_confidence
            )

            if predictions:
                discovered_relations[relation_uri] = {
                    "relation_label": transform.label,
                    "objects": [
                        {
                            "concept_uri": pred["concept_uri"],
                            "label": pred["label"],
                            "confidence": pred["confidence"]
                        }
                        for pred in predictions
                    ]
                }

        # Obtenir le label du concept
        concept = self.ontology_manager.concepts.get(full_concept_uri)
        concept_label = concept.label if concept else full_concept_uri.split('#')[-1]

        return {
            "concept": concept_label,
            "concept_uri": full_concept_uri,
            "relations": discovered_relations,
            "total_relations": len(discovered_relations)
        }

    async def get_relation_statistics(self) -> Dict[str, Any]:
        """
        Obtient des statistiques sur les relations apprises.
        """
        if not self.enable_relation_learning or not self.relation_manager:
            return {"error": "Apprentissage de relations non disponible"}

        return self.relation_manager.get_statistics()

    async def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Obtient l'embedding d'un texte.

        Args:
            text: Texte à encoder

        Returns:
            Embedding du texte ou None si erreur
        """
        try:
            embeddings = await self.rag_engine.embedding_manager.provider.generate_embeddings([text])
            if embeddings and len(embeddings) > 0:
                embedding = embeddings[0]
                # Normaliser
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding
        except Exception as e:
            print(f"Erreur lors de la génération de l'embedding du texte: {e}")

        return None

    # --------------------------------------------------------
    async def classify_document(self, document_id: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Classifie un document dans la hiérarchie ontologique.

        Args:
            document_id: ID du document
            force_refresh: Si True, force une nouvelle classification

        Returns:
            Résultats de classification avec hiérarchie de domaines
        """
        # Vérifier le cache
        if not force_refresh and document_id in self.classification_cache:
            return self.classification_cache[document_id]

        # Récupérer l'embedding du document
        document_embedding = await self._get_document_embedding(document_id)

        if document_embedding is None:
            return {"document_id": document_id, "domains": [], "error": "Embedding non trouvé"}

        # Classification avec le classifieur approprié
        if self.use_hierarchical:
            # Le classifieur hiérarchique renvoie déjà une structure complète
            domains_hierarchy = await self.classifier.classify_document(document_embedding)

            # Résultat final
            result = {
                "document_id": document_id,
                "domains": domains_hierarchy
            }
        else:
            # Classifier avec le classifieur simple
            classification_results = await self.classifier.classify_document(document_embedding)

            if not classification_results:
                return {"document_id": document_id, "domains": [], "error": "Classification échouée"}

            # Construire manuellement la hiérarchie des domaines
            domains_hierarchy = []
            for result in classification_results:
                domain_name = result["domain"]
                confidence = result["confidence"]

                # Récupérer le domaine
                domain = self.ontology_manager.domains.get(domain_name)
                if not domain:
                    continue

                # Construire la hiérarchie des parents
                hierarchy = self._build_domain_hierarchy(domain)

                domains_hierarchy.append({
                    "domain": domain_name,
                    "confidence": confidence,
                    "hierarchy": hierarchy
                })

            # Résultat final
            result = {
                "document_id": document_id,
                "domains": domains_hierarchy
            }

        # Mettre en cache
        self.classification_cache[document_id] = result

        return result

    async def classify_document_concepts(
            self,
            document_id: str,
            force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Classifie un document selon les concepts de l'ontologie."""
        # Vérifier que le classifieur de concepts existe
        if not self.concept_classifier:
            return {"document_id": document_id, "concepts": [], "error": "Classification par concepts non disponible"}

        # Vérifier le cache
        if not force_refresh and document_id in self.concept_classification_cache:
            return self.concept_classification_cache[document_id]

        # Récupérer l'embedding du document
        document_embedding = await self._get_document_embedding(document_id)

        if document_embedding is None:
            return {"document_id": document_id, "concepts": [], "error": "Embedding non trouvé"}

        # Classification des concepts
        concepts_hierarchy = await self.concept_classifier.classify_document(document_embedding)

        # Résultat final
        result = {
            "document_id": document_id,
            "concepts": concepts_hierarchy
        }

        # Mettre en cache
        self.concept_classification_cache[document_id] = result

        return result

    async def search_by_concept(
            self,
            query: str,
            concept_uri: str,
            include_subconcepts: bool = True,
            top_k: int = 5,
            confidence_threshold: float = 0.4
    ) -> Dict[str, Any]:
        """Effectue une recherche limitée à un concept spécifique au niveau CHUNK."""
        if not self.concept_classifier:
            return {"error": "Classification par concepts non disponible"}

        # Résoudre l'URI du concept
        full_concept_uri = self.ontology_manager.resolve_uri(concept_uri)

        # Vérifier que le concept existe
        if full_concept_uri not in self.ontology_manager.concepts:
            return {"error": f"Concept {concept_uri} non trouvé"}

        # Récupérer le concept et ses sous-concepts si demandé
        concepts = [full_concept_uri]
        if include_subconcepts:
            subconcepts = self.concept_classifier.get_all_subconcepts(full_concept_uri)
            concepts.extend(subconcepts)
            print(f"🔍 Recherche pour concept {full_concept_uri} avec {len(subconcepts)} sous-concepts")

        # NOUVEAU : Rechercher directement dans les chunks
        relevant_chunks = []

        # Parcourir tous les documents et leurs chunks
        all_documents = await self.rag_engine.get_all_documents()

        for doc_id in all_documents.keys():
            # Charger les chunks du document
            await self.rag_engine.document_store.load_document_chunks(doc_id)
            chunks = await self.rag_engine.document_store.get_document_chunks(doc_id)

            if not chunks:
                continue

            # Vérifier chaque chunk
            for chunk in chunks:
                # Récupérer les concepts détectés dans ce chunk
                detected_concepts = chunk.get('metadata', {}).get('detected_concepts', [])

                # Vérifier si un des concepts recherchés est présent
                for detected in detected_concepts:
                    concept_uri = detected.get('concept_uri', '')
                    confidence = detected.get('confidence', 0)

                    if concept_uri in concepts and confidence >= confidence_threshold:
                        relevant_chunks.append({
                            'chunk': chunk,
                            'doc_id': doc_id,
                            'matched_concept': concept_uri,
                            'confidence': confidence
                        })
                        break  # Un seul match par chunk

        if not relevant_chunks:
            print(f"⚠️ Aucun chunk trouvé pour le concept {concept_uri}")
            return {"error": f"Aucun document pertinent trouvé pour le concept {concept_uri}"}

        print(f"✅ {len(relevant_chunks)} chunks trouvés pour le concept")

        # Générer l'embedding de la requête
        query_embedding = (await self.rag_engine.embedding_manager.provider.generate_embeddings([query]))[0]

        # Calculer les similarités et préparer les passages
        all_passages = []

        for item in relevant_chunks:
            chunk = item['chunk']
            chunk_id = chunk['id']

            # Récupérer l'embedding du chunk
            chunk_embedding = self.rag_engine.embedding_manager.get_embedding(chunk_id)

            if chunk_embedding is not None:
                # Calculer la similarité
                similarity = float(np.dot(query_embedding, chunk_embedding / np.linalg.norm(chunk_embedding)))

                passage = {
                    'chunk_id': chunk_id,
                    'document_id': item['doc_id'],
                    'text': chunk['text'],
                    'similarity': similarity,
                    'metadata': chunk['metadata'],
                    'matched_concept': item['matched_concept'],
                    'concept_confidence': item['confidence']
                }

                all_passages.append(passage)

        # Trier par similarité
        all_passages.sort(key=lambda x: x['similarity'], reverse=True)
        top_passages = all_passages[:top_k]

        # Générer une réponse
        answer = await self.rag_engine.generate_answer(query, top_passages)

        # Récupérer le label du concept
        concept = self.ontology_manager.concepts.get(full_concept_uri)
        concept_label = concept.label if concept else full_concept_uri.split('#')[-1]

        return {
            "answer": answer,
            "passages": top_passages,
            "concept": concept_label,
            "concept_uri": full_concept_uri,
            "include_subconcepts": include_subconcepts,
            "chunks_found": len(relevant_chunks)
        }

    async def _get_document_embedding(self, document_id: str) -> Optional[np.ndarray]:
        """
        Récupère l'embedding d'un document à partir du RAG.

        Args:
            document_id: ID du document

        Returns:
            Embedding du document ou None si non trouvé
        """
        # Vérifier que le document existe
        document = await self.rag_engine.document_store.get_document(document_id)
        if not document:
            return None

        # Charger les chunks du document si nécessaire
        await self.rag_engine.document_store.load_document_chunks(document_id)

        # Récupérer les chunks du document
        doc_chunks = await self.rag_engine.document_store.get_document_chunks(document_id)
        if not doc_chunks:
            return None

        # Collecter les embeddings des chunks
        chunk_embeddings = []
        for chunk in doc_chunks:
            chunk_id = chunk["id"]
            embedding = self.rag_engine.embedding_manager.get_embedding(chunk_id)
            if embedding is not None:
                # Convertir en numpy si ce n'est pas déjà le cas
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                chunk_embeddings.append(embedding)

        if not chunk_embeddings:
            return None

        # Calculer l'embedding du document comme la moyenne des embeddings des chunks
        doc_embedding = np.mean(chunk_embeddings, axis=0)

        # Normaliser l'embedding
        norm = np.linalg.norm(doc_embedding)
        if norm > 0:
            doc_embedding = doc_embedding / norm

        return doc_embedding

    def _build_domain_hierarchy(self, domain):
        """Construit la hiérarchie des domaines parents."""
        hierarchy = []
        current = domain

        while current:
            hierarchy.append(current.name)
            current = current.parent_domain

        return hierarchy

    async def add_domain_to_hierarchy(
            self,
            domain_name: str,
            domain_description: str = None,
            parent_domain: str = None
    ) -> bool:
        """
        Ajoute un domaine à la hiérarchie ontologique.

        Args:
            domain_name: Nom du domaine
            domain_description: Description du domaine
            parent_domain: Nom du domaine parent

        Returns:
            True si l'ajout a réussi, False sinon
        """
        # Créer d'abord le domaine dans l'ontologie avec sa hiérarchie
        domain = self.ontology_manager.create_domain(domain_name, domain_description)

        # Si un parent est spécifié, établir la relation
        if parent_domain and parent_domain in self.ontology_manager.domains:
            parent = self.ontology_manager.domains[parent_domain]
            parent.add_subdomain(domain)

        # Ajouter le domaine au classifieur hiérarchique
        if self.use_hierarchical:
            return await self.classifier.add_domain_to_hierarchy(
                domain_name=domain_name,
                domain_description=domain_description,
                parent_domain=parent_domain
            )
        else:
            # Version non hiérarchique (fallback)
            print("⚠️ Mode non hiérarchique utilisé - utilisation simplifiée")
            return True

    async def train_from_examples(self, domain_name: str, document_ids: List[str],
                                  description: str = None, parent_domain: str = None) -> bool:
        """
        Entraîne le classifieur pour un domaine à partir d'exemples.

        Args:
            domain_name: Nom du domaine
            document_ids: Liste des IDs de documents exemples
            description: Description du domaine
            parent_domain: Nom du domaine parent (pour la hiérarchie)

        Returns:
            True si l'entraînement a réussi, False sinon
        """
        # Créer d'abord le domaine dans l'ontologie avec sa hiérarchie
        domain = self.ontology_manager.create_domain(domain_name, description)

        # Si un parent est spécifié, établir la relation
        if parent_domain and parent_domain in self.ontology_manager.domains:
            parent = self.ontology_manager.domains[parent_domain]
            parent.add_subdomain(domain)

        # Récupérer les embeddings des documents
        document_embeddings = []
        for doc_id in document_ids:
            embedding = await self._get_document_embedding(doc_id)
            if embedding is not None:
                document_embeddings.append((doc_id, embedding))

        if not document_embeddings:
            print(f"Aucun embedding trouvé pour les documents du domaine {domain_name}")
            return False

        # Entraîner le classifieur
        if self.use_hierarchical:
            # Pour le classifieur hiérarchique, nous devons adapter aux méthodes disponibles
            # Utilisez la méthode create_domain_for_level qui gère la hiérarchie
            success = await self.classifier.create_domain_for_level(
                domain_name=domain_name,
                document_embeddings=document_embeddings,
                parent_domain=parent_domain
            )
        else:
            # L'entraînement simple ne prend pas en compte la hiérarchie
            success = await self.classifier.create_domain_from_documents(
                domain_name=domain_name,
                document_ids=[doc_id for doc_id, _ in document_embeddings],
                description=description
            )

        return success

    async def search_in_domain(
            self,
            query: str,
            domain_name: str,
            include_subdomains: bool = True,
            top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Effectue une recherche limitée à un domaine spécifique.

        Args:
            query: Requête de recherche
            domain_name: Nom du domaine
            include_subdomains: Si True, inclut les documents des sous-domaines
            top_k: Nombre de résultats à retourner

        Returns:
            Résultats de recherche
        """
        # Vérifier que le domaine existe
        if domain_name not in self.ontology_manager.domains:
            return {"error": f"Domaine {domain_name} non trouvé"}

        # Récupérer le domaine et ses sous-domaines si demandé
        domains = [domain_name]
        if include_subdomains:
            domains.extend(self._get_all_subdomains(domain_name))

        # MODIFICATION: Classifier tous les documents pour trouver ceux appartenant au domaine
        all_documents = await self.rag_engine.get_all_documents()
        document_ids = list(all_documents.keys())

        # Documents pertinents pour les domaines sélectionnés
        relevant_documents = []
        confidence_threshold = 0.6  # Seuil de confiance pour l'appartenance au domaine

        for doc_id in document_ids:
            # Classifier le document
            classification = await self.classify_document(doc_id)

            # Vérifier si le document appartient à l'un des domaines ciblés
            for domain_result in classification.get("domains", []):
                # Vérifier le domaine principal
                if domain_result["domain"] in domains and domain_result["confidence"] >= confidence_threshold:
                    relevant_documents.append(doc_id)
                    break

                # Vérifier les sous-domaines
                for sub_domain in domain_result.get("sub_domains", []):
                    if sub_domain["domain"] in domains and sub_domain["confidence"] >= confidence_threshold:
                        relevant_documents.append(doc_id)
                        break

        if not relevant_documents:
            return {"error": f"Aucun document pertinent trouvé pour le domaine {domain_name}"}

        # Générer l'embedding de la requête
        query_embedding = (await self.rag_engine.embedding_manager.provider.generate_embeddings([query]))[0]

        # Rechercher dans les documents pertinents
        all_passages = []
        for doc_id in relevant_documents:
            passages = await self.rag_engine.search_with_embedding(
                query_embedding,
                document_id=doc_id,
                top_k=3,  # passages par document
                skip_loading=False
            )
            all_passages.extend(passages)

        # Trier tous les passages par similarité
        all_passages.sort(key=lambda x: x["similarity"], reverse=True)
        top_passages = all_passages[:top_k]

        # Générer une réponse
        answer = await self.rag_engine.generate_answer(query, top_passages)

        return {
            "answer": answer,
            "passages": top_passages,
            "domain": domain_name,
            "include_subdomains": include_subdomains,
            "domains_included": domains,
            "documents_used": relevant_documents
        }

    def _get_all_subdomains(self, domain_name: str) -> List[str]:
        """Récupère récursivement tous les sous-domaines d'un domaine."""
        if domain_name not in self.ontology_manager.domains:
            return []

        domain = self.ontology_manager.domains[domain_name]
        subdomains = []

        for subdomain in domain.subdomains:
            subdomains.append(subdomain.name)
            # Récursivement ajouter les sous-domaines des sous-domaines
            subdomains.extend(self._get_all_subdomains(subdomain.name))

        return subdomains

    async def get_domain_statistics(self, domain_name: str = None) -> Dict[str, Any]:
        """
        Obtient des statistiques sur les domaines et leurs documents.

        Args:
            domain_name: Nom du domaine (si None, statistiques pour tous les domaines)

        Returns:
            Statistiques des domaines
        """
        stats = {}

        # Si un domaine spécifique est demandé
        if domain_name:
            if domain_name not in self.ontology_manager.domains:
                return {"error": f"Domaine {domain_name} non trouvé"}

            domains_to_analyze = [domain_name]
            domains_to_analyze.extend(self._get_all_subdomains(domain_name))
        else:
            # Tous les domaines
            domains_to_analyze = list(self.ontology_manager.domains.keys())

        # Analyser chaque domaine
        for d_name in domains_to_analyze:
            domain = self.ontology_manager.domains[d_name]

            # Statistiques de base
            domain_stats = {
                "document_count": len(domain.documents),
                "subdomain_count": len(domain.subdomains),
                "subdomain_names": [sd.name for sd in domain.subdomains],
                "parent_domain": domain.parent_domain.name if domain.parent_domain else None,
                "concept_count": len(domain.concepts)
            }

            stats[d_name] = domain_stats

        return {
            "domains": stats,
            "total_domains": len(stats),
            "total_documents": sum(s["document_count"] for s in stats.values())
        }

    def _flatten_hierarchy(self, concepts_hierarchy: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aplatit une hiérarchie de concepts en une liste simple.

        Args:
            concepts_hierarchy: Liste hiérarchique de concepts

        Returns:
            Liste aplatie de concepts
        """
        flattened = []

        def flatten_recursive(concept_list):
            for concept in concept_list:
                # Ajouter le concept actuel
                flattened.append({
                    "concept_uri": concept["concept_uri"],
                    "label": concept["label"],
                    "confidence": concept["confidence"]
                })

                # Récursivement aplatir les sous-concepts
                if "sub_concepts" in concept and concept["sub_concepts"]:
                    flatten_recursive(concept["sub_concepts"])

        flatten_recursive(concepts_hierarchy)
        return flattened

    async def smart_concept_detection(self, query: str) -> List[Dict[str, Any]]:
        """
        Détection intelligente des concepts combinant approches ascendante et descendante.

        Args:
            query: Requête de l'utilisateur

        Returns:
            Liste des concepts pertinents
        """
        # Vérifier que le classifieur de concepts existe
        if not self.concept_classifier:
            return []

        # Générer l'embedding de la requête
        query_embedding = await self.rag_engine.embedding_manager.provider.generate_embeddings([query])
        query_embedding = query_embedding[0]

        # Approche 1: Classification directe de la requête (top-down)
        concepts_top_down = await self.concept_classifier.classify_document(query_embedding)

        # Approche 2: Recherche ascendante (bottom-up)
        concepts_bottom_up = await self.concept_classifier.auto_detect_concepts(query_embedding)

        # Fusionner et prioriser les résultats
        all_concepts = {}

        # Traiter d'abord les résultats de l'approche bottom-up (plus spécifiques)
        for concept in concepts_bottom_up:
            all_concepts[concept["concept_uri"]] = concept

        # Ajouter ou fusionner les résultats de l'approche top-down
        for concept in self._flatten_hierarchy(concepts_top_down):
            uri = concept["concept_uri"]
            if uri in all_concepts:
                # Prendre le score le plus élevé
                all_concepts[uri]["confidence"] = max(
                    all_concepts[uri]["confidence"],
                    concept["confidence"]
                )
            else:
                all_concepts[uri] = concept

        # Trier par pertinence
        result = list(all_concepts.values())
        result.sort(key=lambda x: x["confidence"], reverse=True)

        return result[:10]  # Retourner les 5 meilleurs concepts

    async def search_with_concepts(
            self,
            query: str,
            top_k: int = 5,
            concept_weight: float = 0.3,
            min_concept_confidence: float = 0.3,
            entity_boost: float = 1.5,
            group_entity_parts: bool = True  # Groupe les chunk de la meme entité
    ) -> Dict[str, Any]:

            print(f"🔍 Recherche: {query}")

            # 1. Générer l'embedding de la requête
            query_embedding = await self._get_text_embedding(query)
            if query_embedding is None:
                return {"error": "Impossible de générer l'embedding de la requête"}

            # 2. Détecter les concepts dans la requête
            query_concepts = []
            if self.concept_classifier:
                query_concepts = await self.concept_classifier.auto_detect_concepts(
                    query_embedding,
                    min_confidence=min_concept_confidence
                )
                if query_concepts:
                    print(f"📋 Concepts détectés dans la requête: {', '.join([c['label'] for c in query_concepts[:5]])}")

            # 3. Collecter tous les chunks avec leurs scores (code existant)
            chunk_scores = []
            all_documents = await self.rag_engine.get_all_documents()

            for doc_id in all_documents.keys():
                await self.rag_engine.document_store.load_document_chunks(doc_id)
                chunks = await self.rag_engine.document_store.get_document_chunks(doc_id)

                if not chunks:
                    continue

                for chunk in chunks:
                    chunk_id = chunk['id']
                    chunk_embedding = self.rag_engine.embedding_manager.get_embedding(chunk_id)
                    if chunk_embedding is None:
                        continue

                    # Calculer la similarité textuelle
                    text_similarity = float(np.dot(query_embedding, chunk_embedding))

                    # Calculer le score conceptuel
                    concept_score = 0.0
                    chunk_concepts = chunk.get('metadata', {}).get('detected_concepts', [])
                    matched_concepts = []

                    if chunk_concepts and query_concepts:
                        for chunk_concept in chunk_concepts:
                            for query_concept in query_concepts:
                                if chunk_concept['concept_uri'] == query_concept['concept_uri']:
                                    concept_score = max(concept_score,
                                                        chunk_concept['confidence'] * query_concept['confidence'])
                                    matched_concepts.append(chunk_concept['label'])
                                elif self._are_concepts_related(chunk_concept['concept_uri'],
                                                                query_concept['concept_uri']):
                                    concept_score = max(concept_score,
                                                        0.7 * chunk_concept['confidence'] * query_concept['confidence'])
                                    matched_concepts.append(f"{chunk_concept['label']} (related)")

                    # Score combiné
                    combined_score = (1 - concept_weight) * text_similarity + concept_weight * concept_score

                    # Bonus si le nom de l'entité correspond
                    entity_name = chunk.get('metadata', {}).get('entity_name', '').lower()
                    query_words = query.lower().split()
                    if any(word in entity_name for word in query_words if len(word) > 3):
                        combined_score += 0.1

                    chunk_scores.append({
                        'chunk': chunk,
                        'chunk_id': chunk_id,
                        'doc_id': doc_id,
                        'text_similarity': text_similarity,
                        'concept_score': concept_score,
                        'combined_score': combined_score,
                        'chunk_concepts': chunk_concepts[:5] if chunk_concepts else [],
                        'matched_concepts': matched_concepts
                    })

            # 3.5 NOUVEAU : Analyser la spécificité de la requête
            query_analysis = self._analyze_query_specificity(query, chunk_scores)

            # -----------------------------------------------------------------
            #  Ajustement des scores en fonction du nom d’entité
            #  (exact match  +35 %     /    match partiel  –75 %)
            # -----------------------------------------------------------------
            targets = [t.lower() for t in query_analysis['query_targets']]

            for item in chunk_scores:
                meta = item['chunk']['metadata']
                entity_name = meta.get('base_entity_name',
                                       meta.get('entity_name', '')).lower()

                if not entity_name:
                    continue

                exact = any(t == entity_name for t in targets)
                partial = (not exact) and any(t in entity_name for t in targets)

                if exact:
                    item['combined_score'] *= 1.35  # bonus
                elif partial:
                    item['combined_score'] *= 0.25  # malus

            if query_analysis['is_specific_query']:
                print(f"🎯 Requête spécifique détectée : {query_analysis['query_targets']}")
                print(f"   Niveau de spécificité : {query_analysis['max_specificity_level']}")


            # 4. NOUVELLE LOGIQUE : Regroupement des entités
            if group_entity_parts:
                chunk_scores = self._group_entity_parts(chunk_scores, top_k * 2)

            # 4.5 NOUVEAU : Filtrage par spécificité
            if query_analysis['is_specific_query']:
                print(f"🔍 Filtrage par spécificité...")
                chunk_scores = self._filter_by_specificity(chunk_scores, query_analysis)
                print(f"   Chunks après filtrage : {len(chunk_scores)}")

                # Ajout d’un mini-contexte parent (10-25 lignes max)
                chunk_scores = self._inject_parent_mini_context(chunk_scores, max_lines=25)

                trimmed = []
                for cs in chunk_scores:
                    meta = cs['chunk']['metadata']
                    if (len(cs['chunk']['text'].splitlines()) > 250 and
                            any(t in meta.get('entity_name', '').lower()
                                for t in query_analysis['query_targets'])):
                        foc = self._extract_internal_function_context(cs, query_analysis['query_targets'][0])
                        if foc: trimmed.append(foc)
                        continue
                    trimmed.append(cs)
                chunk_scores = trimmed

            # 5. Trier et sélectionner les meilleurs
            chunk_scores.sort(key=lambda x: x['combined_score'], reverse=True)

            # Pour les requêtes spécifiques, ajuster le top_k
            effective_top_k = top_k
            if query_analysis['is_specific_query']:
                chunk_scores = self._drop_parent_chunks(chunk_scores)
                chunk_scores = self._keep_only_target_chunks(chunk_scores,
                                                             query_analysis['query_targets'])
                print(f"✅ After keep_only_target_chunks : {len(chunk_scores)}")
                # Pour une requête spécifique, on peut se permettre moins de contexte
                effective_top_k = min(top_k, max(3, top_k // 2))

            enhanced_chunks = self._boost_related_entity_chunks(chunk_scores, entity_boost)
            top_chunks = enhanced_chunks[:effective_top_k]

            if not top_chunks:
                return {"error": "Aucun résultat trouvé"}

            # 6. Préparer les passages avec sources regroupées
            passages = []
            sources_info = []

            # NOUVELLE LOGIQUE : Créer des sources regroupées par entité
            entity_groups = self._create_entity_groups(top_chunks)

            source_index = 0
            for entity_id, entity_info in entity_groups.items():
                source_index += 1

                # Si l'entité a plusieurs parties, les regrouper
                if len(entity_info['chunks']) > 1:
                    # Source regroupée
                    grouped_source = self._create_grouped_source(entity_info, source_index)
                    sources_info.append(grouped_source)

                    # Créer un passage regroupé
                    grouped_passage = self._create_grouped_passage(entity_info, source_index)
                    passages.append(grouped_passage)
                else:
                    # Source individuelle (code existant)
                    item = entity_info['chunks'][0]
                    chunk = item['chunk']
                    metadata = chunk.get('metadata', {})

                    source_info = {
                        'index': source_index,
                        'file': metadata.get('filename', 'Unknown'),
                        'filepath': metadata.get('filepath', ''),
                        'entity': metadata.get('entity_name', 'Unknown'),
                        'entity_type': metadata.get('entity_type', 'code'),
                        'start_line': metadata.get('start_pos', 0),
                        'end_line': metadata.get('end_pos', 0),
                        'concepts': [c['label'] for c in item['chunk_concepts']],
                        'matched_concepts': item['matched_concepts'],
                        'parent_entity': metadata.get('parent_entity_name', ''),
                        'parent_entity_type': metadata.get('parent_entity_type', ''),
                        'section_title': metadata.get('section_title', ''),
                        'score': round(item['combined_score'], 3)
                    }
                    sources_info.append(source_info)

                    passages.append({
                        'chunk_id': item['chunk_id'],
                        'document_id': item['doc_id'],
                        'text': chunk['text'],
                        'similarity': item['text_similarity'],
                        'metadata': metadata,
                        'source_info': source_info,
                        'document_name': metadata.get('filename', 'Document inconnu')
                    })

            # 7. Créer le prompt système enrichi
            source_descriptions = []
            for src in sources_info:
                if src.get('is_grouped', False):
                    # Description pour source groupée
                    desc = f"[Source {src['index']}] {src['file']} - Entité complète '{src['entity']}' ({src['entity_type']})"
                    desc += f" - {src['total_parts']} parties (lignes {src['start_line']}-{src['end_line']})"
                    if src['matched_concepts']:
                        desc += f" - Concepts: {', '.join(src['matched_concepts'])}"
                    parent = src.get('parent_entity')
                    if parent:
                        desc += f"  (dans le {src.get('parent_entity_type', 'parent')} '{parent}')"
                    if src.get('section_title'):
                        desc += f" - section \"{src['section_title']}\""
                else:
                    # Description classique
                    desc = f"[Source {src['index']}] {src['file']} (lignes {src['start_line']}-{src['end_line']})"
                    if src['entity'] != 'Unknown':
                        desc += f" - {src['entity_type']} '{src['entity']}'"
                    parent = src.get('parent_entity')
                    if parent:
                        desc += f"  (dans le {src.get('parent_entity_type', 'parent')} '{parent}')"
                    if src.get('section_title'):
                        desc += f" - section \"{src['section_title']}\""
                    if src['matched_concepts']:
                        desc += f" - Concepts: {', '.join(src['matched_concepts'])}"
                source_descriptions.append(desc)

            if query_analysis['is_specific_query']:
                system_prompt = f"""Tu es un assistant expert en code Fortran et programmation scientifique.

            La question porte spécifiquement sur : {', '.join(query_analysis['query_targets'])}

            Tu as accès aux sources suivantes:
            {chr(10).join(source_descriptions)}

            Instructions:
            1. Concentre-toi sur l'entité spécifiquement demandée
            2. Les sources marquées [contexte minimal] donnent juste le contexte du module parent
            3. Cite TOUJOURS tes sources en utilisant [Source N]
            4. Sois précis et technique dans tes explications, mais n'invente rien et n'utilise pas ta connaissance
            5. Si la question est spécifique, évite les généralités sur le module complet
            """
            else:
                system_prompt = f"""Tu es un assistant expert en code Fortran et programmation scientifique.

    Tu as accès aux sources suivantes:
    {chr(10).join(source_descriptions)}

    Lorsque tu réponds:
    1. Cite TOUJOURS tes sources en utilisant [Source N] dans ta réponse
    2. Les sources groupées contiennent l'entité complète (toutes les parties)
    3. Mentionne les fichiers et numéros de ligne pertinents
    4. Indique les concepts détectés qui sont pertinents
    5. Si plusieurs sources contiennent des informations complémentaires, cite-les toutes
    6. Sois précis et technique dans tes explications mais n'invente rien et n'utilise pas ta connaissance

    Exemple de citation: "D'après [Source 1] qui contient l'entité complète inspect_rototranslation..."
    """

            # 8. Générer la réponse
            answer = await self.rag_engine.generate_answer(query, passages, system_prompt)

            # 9. Formatter les sources pour l'affichage
            formatted_sources = []
            for src in sources_info:
                if src.get('is_grouped', False):
                    formatted_sources.append({
                        'filename': src['file'],
                        'filepath': src['filepath'],
                        'entity_type': src['entity_type'],
                        'entity_name': src['entity'],
                        'start_line': src['start_line'],
                        'end_line': src['end_line'],
                        'detected_concepts': src['concepts'],
                        'matched_concepts': src['matched_concepts'],
                        'relevance_score': src['score'],
                        'is_grouped': True,
                        'total_parts': src['total_parts'],
                        'parts_info': src['parts_info']
                    })
                else:
                    formatted_sources.append({
                        'filename': src['file'],
                        'filepath': src['filepath'],
                        'entity_type': src['entity_type'],
                        'entity_name': src['entity'],
                        'start_line': src['start_line'],
                        'end_line': src['end_line'],
                        'detected_concepts': src['concepts'],
                        'matched_concepts': src['matched_concepts'],
                        'relevance_score': src['score']
                    })

            return {
                "answer": answer,
                "sources": formatted_sources,
                "passages": passages,
                "query_concepts": [c['label'] for c in query_concepts[:5]],
                "search_stats": {
                    "total_chunks_evaluated": len(chunk_scores),
                    "chunks_with_concepts": sum(1 for c in chunk_scores if c['concept_score'] > 0),
                    "best_match_score": round(top_chunks[0]['combined_score'], 3) if top_chunks else 0,
                    "entity_groups_found": len(entity_groups)
                }
            }

    def _analyze_query_specificity(self, query: str, chunk_scores: List[Dict]) -> Dict[str, Any]:
        """
        Analyse la spécificité de la requête pour déterminer le niveau de contexte nécessaire.
        """
        query_lower = query.lower()

        # Détecter les entités spécifiquement mentionnées
        mentioned_entities = {}

        # NOUVEAU : D'abord chercher dans le contenu réel des chunks pour les fonctions
        for chunk_data in chunk_scores[:30]:  # Limiter la recherche aux top chunks
            chunk_text = chunk_data['chunk'].get('text', '').lower()
            metadata = chunk_data['chunk'].get('metadata', {})

            # Extraire tous les noms de fonctions du chunk
            function_patterns = [
                r'function\s+(\w+)',
                r'subroutine\s+(\w+)',
                r'pure\s+function\s+(\w+)',
                r'elemental\s+function\s+(\w+)',
                r'recursive\s+function\s+(\w+)'
            ]

            found_functions = set()
            for pattern in function_patterns:
                matches = re.findall(pattern, chunk_text, re.IGNORECASE)
                found_functions.update(matches)

            # Vérifier si une des fonctions trouvées est dans la requête
            for func_name in found_functions:
                if func_name.lower() in query_lower:
                    # C'est la fonction recherchée !
                    is_internal = metadata.get('is_internal_function', False)

                    mentioned_entities[func_name] = {
                        'type': metadata.get('entity_type', 'function'),
                        'specificity_level': 5 if is_internal else 4,
                        'is_internal': is_internal,
                        'parent': metadata.get('parent_entity', ''),
                        'chunks': [chunk_data],
                        'found_in_content': True
                    }

                    print(f"🔍 Fonction {'interne' if is_internal else ''} détectée: {func_name}")
                    break

        # Ensuite chercher dans les métadonnées (code existant mais modifié)
        for chunk_data in chunk_scores:
            metadata = chunk_data['chunk'].get('metadata', {})

            # Vérifier dans base_entity_name ET entity_name
            entity_name = metadata.get('base_entity_name', metadata.get('entity_name', '')).lower()

            if entity_name and entity_name in query_lower:
                # Ne pas écraser si déjà trouvé dans le contenu
                if entity_name not in mentioned_entities:
                    if metadata.get('is_internal_function', False):
                        mentioned_entities[entity_name] = {
                            'type': metadata.get('entity_type', ''),
                            'specificity_level': 5,
                            'is_internal': True,
                            'parent': metadata.get('parent_entity', ''),
                            'chunks': [chunk_data]
                        }
                    else:
                        mentioned_entities[entity_name] = {
                            'type': metadata.get('entity_type', ''),
                            'specificity_level': self._get_specificity_level(metadata.get('entity_type', '')),
                            'chunks': [chunk_data]
                        }

        # Déterminer le niveau de spécificité requis
        max_specificity = 0
        if mentioned_entities:
            max_specificity = max(
                entity['specificity_level']
                for entity in mentioned_entities.values()
            )

        # --- détection DOI ---
        doi_match = re.search(r'10\.\d{4,9}/[\w\-.;()/:]+', query, re.I)
        if doi_match:
            doi = doi_match.group(0).lower()
            mentioned_entities[doi] = {
                'type': 'article',
                'specificity_level': 5,  # très spécifique
                'chunks': [],
            }
            max_specificity = 5

        return {
            'mentioned_entities': mentioned_entities,
            'max_specificity_level': max_specificity,
            'is_specific_query': bool(mentioned_entities),
            'query_targets': list(mentioned_entities.keys()),
            'has_internal_functions': any(
                entity.get('is_internal', False)
                for entity in mentioned_entities.values()
            )
        }

    def _get_specificity_level(self, entity_type: str) -> int:
        """
        Retourne le niveau de spécificité d'un type d'entité.
        Plus le nombre est élevé, plus l'entité est spécifique.
        """
        specificity_map = {
            'module': 1,
            'program': 1,
            'type_definition': 2,
            'interface': 3,
            'subroutine': 4,
            'function': 4,
            'variable_declaration': 5,
            'parameter': 5,
        }
        return specificity_map.get(entity_type, 3)

    def _filter_by_specificity(
            self,
            chunk_scores: List[Dict],
            query_analysis: Dict[str, Any],
            max_parent_context_lines: int = 50
    ) -> List[Dict]:
        """
        Filtre les chunks en fonction de la spécificité de la requête.
        """
        if not query_analysis['is_specific_query']:
            return chunk_scores

        filtered_chunks = []
        seen_entities = set()
        max_specificity = query_analysis['max_specificity_level']

        print(f"🔍 Filtrage avec query_targets: {query_analysis['query_targets']}")

        # NOUVEAU : D'abord collecter les chunks qui contiennent les fonctions recherchées
        for chunk_data in chunk_scores:
            metadata = chunk_data['chunk'].get('metadata', {})
            chunk_text = chunk_data['chunk'].get('text', '').lower()

            # Vérifier si le chunk contient une des fonctions recherchées
            chunk_contains_target = False

            # 1. Vérifier dans le texte du chunk
            for target in query_analysis['query_targets']:
                target_lower = target.lower()

                # Patterns pour détecter la fonction dans le texte
                function_patterns = [
                    f"function {target_lower}",
                    f"subroutine {target_lower}",
                    f"pure function {target_lower}",
                    f"elemental function {target_lower}",
                    f"end function {target_lower}",
                    f"end subroutine {target_lower}"
                ]

                for pattern in function_patterns:
                    if pattern in chunk_text:
                        chunk_contains_target = True
                        print(f"   ✅ Chunk contient '{pattern}'")
                        break

                if chunk_contains_target:
                    break

            # 2. Vérifier dans les métadonnées
            if not chunk_contains_target:
                # Vérifier entity_name, base_entity_name
                entity_name = metadata.get('entity_name', '').lower()
                base_entity_name = metadata.get('base_entity_name', '').lower()

                for target in query_analysis['query_targets']:
                    target_lower = target.lower()
                    if (target_lower == entity_name or
                            target_lower == base_entity_name or
                            entity_name.endswith(f"_{target_lower}") or
                            base_entity_name.endswith(f"_{target_lower}")):
                        chunk_contains_target = True
                        print(f"   ✅ Métadonnées correspondent: {entity_name or base_entity_name}")
                        break

                # 3. Vérifier searchable_names si disponible
                if not chunk_contains_target and 'searchable_names' in metadata:
                    for name in metadata['searchable_names']:
                        if any(target.lower() in name.lower() for target in query_analysis['query_targets']):
                            chunk_contains_target = True
                            print(f"   ✅ Searchable name correspond: {name}")
                            break

            # Si le chunk contient la cible, l'inclure
            if chunk_contains_target:
                filtered_chunks.append(chunk_data)
                seen_entities.add(metadata.get('entity_name', 'unknown'))

                # Si c'est une partie d'entité, essayer de récupérer les autres parties
                if metadata.get('is_partial', False):
                    parent_entity_id = metadata.get('parent_entity_id')
                    if parent_entity_id:
                        print(f"   📦 Recherche des autres parties de {parent_entity_id}")

                        # Chercher les autres chunks de la même entité
                        for other_chunk in chunk_scores:
                            other_meta = other_chunk['chunk'].get('metadata', {})
                            if (other_meta.get('parent_entity_id') == parent_entity_id and
                                    other_chunk not in filtered_chunks):
                                filtered_chunks.append(other_chunk)
                                print(f"      + Ajout partie {other_meta.get('part_index', '?')}")

        # Si on n'a trouvé aucun chunk direct, chercher plus largement
        if not filtered_chunks:
            print("⚠️ Aucun chunk direct trouvé, recherche élargie...")

            # Chercher dans les chunks avec un score élevé
            for chunk_data in chunk_scores[:20]:  # Top 20 chunks
                chunk_text = chunk_data['chunk'].get('text', '').lower()

                # Recherche plus souple
                for target in query_analysis['query_targets']:
                    if target.lower() in chunk_text:
                        filtered_chunks.append(chunk_data)
                        print(f"   ✅ Chunk contient le terme '{target}'")
                        break

        # Si toujours rien, prendre les meilleurs chunks
        if not filtered_chunks:
            print("⚠️ Aucun chunk trouvé avec filtrage, utilisation des top chunks")
            filtered_chunks = chunk_scores[:5]

        print(f"📊 Chunks après filtrage: {len(filtered_chunks)}")
        return filtered_chunks

    def _extract_internal_function_context(
            self,
            parent_chunk_data: Dict,
            function_name: str
    ) -> Optional[Dict]:
        """
        Extrait uniquement le contexte de la fonction interne depuis le chunk parent.
        """
        chunk = parent_chunk_data['chunk']
        text = chunk['text']
        lines = text.split('\n')

        # Trouver le début de la fonction
        func_start = None
        func_pattern = re.compile(
            rf'(pure\s+|elemental\s+|recursive\s+)*'
            rf'(real|integer|logical|character|type)?\s*(\([^)]*\))?\s*'
            rf'function\s+{re.escape(function_name)}',
            re.IGNORECASE
        )

        for i, line in enumerate(lines):
            if func_pattern.search(line):
                func_start = i
                break

        if func_start is None:
            return None

        # Trouver la fin de la fonction
        func_end = None
        end_pattern = re.compile(r'^\s*end\s+function', re.IGNORECASE)

        for i in range(func_start + 1, len(lines)):
            if end_pattern.match(lines[i]):
                func_end = i
                break

        if func_end is None:
            # Prendre au maximum 50 lignes
            func_end = min(func_start + 50, len(lines) - 1)

        # Extraire le contexte
        context_lines = lines[func_start:func_end + 1]

        # Ajouter un peu de contexte avant (déclarations du parent)
        pre_context = []
        for i in range(max(0, func_start - 10), func_start):
            line = lines[i].strip()
            if line and not line.startswith('!'):
                pre_context.append(lines[i])

        if pre_context:
            context_lines = ["    ! ... [Context from parent function] ..."] + pre_context + [""] + context_lines

        # Créer un nouveau chunk focalisé
        focused_chunk = parent_chunk_data.copy()
        focused_chunk['chunk'] = chunk.copy()
        focused_chunk['chunk']['text'] = '\n'.join(context_lines)
        focused_chunk['chunk']['metadata'] = chunk['metadata'].copy()
        focused_chunk['chunk']['metadata']['is_extracted_internal'] = True
        focused_chunk['chunk']['metadata']['internal_function_name'] = function_name
        focused_chunk['chunk']['metadata']['extraction_context'] = 'internal_function'

        # Ajuster les positions
        parent_start = chunk['metadata'].get('start_pos', 0)
        focused_chunk['chunk']['metadata']['start_pos'] = parent_start + func_start
        focused_chunk['chunk']['metadata']['end_pos'] = parent_start + func_end

        return focused_chunk

    def _create_minimal_parent_context(self, parent_chunk_data: Dict) -> Optional[Dict]:
        """
        Crée un chunk de contexte minimal pour une entité parent (ex: module).
        """
        chunk = parent_chunk_data['chunk']
        text = chunk['text']
        lines = text.split('\n')

        # Extraire seulement le début du module (signature, imports, interface)
        context_lines = []
        in_contains = False
        line_count = 0
        max_lines = 50

        for line in lines:
            line_lower = line.strip().lower()

            # Arrêter à 'contains' ou après max_lines
            if line_lower == 'contains' or line_count >= max_lines:
                if line_lower == 'contains':
                    context_lines.append(line)
                    context_lines.append("    ! ... [Module content truncated for context] ...")
                break

            # Inclure les lignes importantes du début
            if (line_lower.startswith(('module', 'use', 'implicit', 'interface', 'type', 'parameter')) or
                    line.strip().startswith('!') or  # Commentaires importants
                    not line.strip()):  # Lignes vides pour la lisibilité
                context_lines.append(line)
                if line.strip():  # Ne compter que les lignes non vides
                    line_count += 1

        if not context_lines:
            return None

        # Créer un nouveau chunk avec le contexte minimal
        minimal_chunk = parent_chunk_data.copy()
        minimal_chunk['chunk'] = chunk.copy()
        minimal_chunk['chunk']['text'] = '\n'.join(context_lines)
        minimal_chunk['chunk']['metadata'] = chunk['metadata'].copy()
        minimal_chunk['chunk']['metadata']['is_minimal_context'] = True
        minimal_chunk['chunk']['metadata']['original_lines'] = len(lines)
        minimal_chunk['chunk']['metadata']['context_lines'] = len(context_lines)

        # Réduire le score pour que ce contexte soit en bas de liste
        minimal_chunk['combined_score'] *= 0.5

        return minimal_chunk

    # ------------------------------------------------------------------
    #  FILTRE qui supprime les sur-structures inclusives
    # ------------------------------------------------------------------
    def _drop_parent_chunks(self, chunks: list) -> list:
        """
        Élimine tout chunk dont la plage (start_line, end_line) englobe
        entièrement au moins UN autre chunk du *même fichier*.

        1. on travaille sur des entiers (start_pos / end_pos)  – si ces clés
           n’existent pas on tombe sur start_line / end_line.
        2. on conserve toujours le chunk au plus petit intervalle
           (= le plus spécifique).
        """
        if len(chunks) < 2:
            return chunks

        # Helper pour récupérer positions + path
        def _info(c):
            md = c['chunk']['metadata']
            start = md.get('start_pos') or md.get('start_line', 0)
            end = md.get('end_pos') or md.get('end_line', 0)
            path = md.get('filepath')
            return int(start), int(end), path

        # On trie par taille décroissante (parents d’abord)
        chunks_sorted = sorted(chunks,
                               key=lambda c: (_info(c)[2],  # filepath
                                              -(_info(c)[1] - _info(c)[0])),  # taille négative
                               )

        keep = []
        for i, ci in enumerate(chunks_sorted):
            s_i, e_i, f_i = _info(ci)
            is_parent = False

            # Un seul test suffit : si un chunk plus petit (déjà conservé)
            # est inclus dans ci, alors ci est parent → on le jette.
            for ck in keep:
                s_k, e_k, f_k = _info(ck)
                if f_k == f_i and s_i <= s_k and e_k <= e_i:
                    is_parent = True
                    break

            if not is_parent:
                keep.append(ci)

        # On rend la liste dans le même ordre que l’entrée
        ids_kept = {c['chunk_id'] for c in keep}
        return [c for c in chunks if c['chunk_id'] in ids_kept]

    def _primary_target_name(self, chunks: list, query_targets: list) -> str:
        """
        Renvoie le nom d’entité (base_entity_name) le plus pertinent :
        celui qui apparaît dans un chunk de spécificité ≥ 4 et ayant
        le meilleur combined_score.  Si aucun, retourne ''.
        """
        targets = [t.lower() for t in query_targets]
        best_name = ''
        best_score = -1
        for c in chunks:
            md = c['chunk']['metadata']
            spec = SPECIFICITY_RANK.get(md.get('entity_type', ''), 3)
            if spec < 4:  # on ignore module, type, …
                continue
            name = md.get('base_entity_name', md.get('entity_name', '')).lower()
            if name in targets and c['combined_score'] > best_score:
                best_name = name
                best_score = c['combined_score']
        return best_name

    def _keep_only_target_chunks(self, chunks: list, query_targets: list) -> list:
        """
        Conserve uniquement :
            • les chunks dont base_entity_name == primary_target
            • les mini-contextes flaggés is_minimal_context
        """
        primary = self._primary_target_name(chunks, query_targets)
        if not primary:  # sécurité : si rien trouvé, on ne filtre pas
            return chunks

        kept = []
        for c in chunks:
            md = c['chunk']['metadata']
            if md.get('is_minimal_context'):
                kept.append(c)
                continue
            name = md.get('base_entity_name', md.get('entity_name', '')).lower()
            if name == primary:
                kept.append(c)
        return kept

    def _group_entity_parts(self, chunk_scores: list, expand_limit: int = 20) -> list:
        """
        Ne regroupe QUE les entités de même spécificité que la meilleure
        (fonction ou subroutine dans notre cas). On évite ainsi d’aspirer
        tout le module.
        """
        if not chunk_scores:
            return chunk_scores

        # 1. Meilleure spécificité rencontrée dans les top résultats
        top = sorted(chunk_scores, key=lambda x: x['combined_score'], reverse=True)[:expand_limit]
        max_spec = max(_spec(c['chunk']['metadata']) for c in top)

        # Ne jamais expandre les entités dont la spécificité < 4
        if max_spec >= 4:
            entity_min_spec = 4
        else:
            entity_min_spec = max_spec

        entities_to_expand = {}
        for c in top:
            meta = c['chunk']['metadata']
            if _spec(meta) < entity_min_spec:
                continue  # on ignore modules, types, etc.

            ent_id = meta.get('parent_entity_id') if meta.get('is_partial') else meta.get('entity_id')
            if ent_id:
                entities_to_expand.setdefault(ent_id, set()).add(c['chunk_id'])

        # 2. index rapide
        by_id = {c['chunk_id']: c for c in chunk_scores}
        kept = []
        done = set()

        for ent_id, initial_ids in entities_to_expand.items():
            # parties déjà présentes
            for cid in initial_ids:
                if cid in by_id and cid not in done:
                    kept.append(by_id[cid]);
                    done.add(cid)

            # parties manquantes mais listées dans les métadonnées
            sample = by_id[next(iter(initial_ids))]['chunk']['metadata']
            for cid in sample.get('all_chunks', []):
                if cid in by_id and cid not in done:
                    kept.append(by_id[cid]);
                    done.add(cid)

        # 3. on ajoute le reste (pas d’expansion)
        for c in chunk_scores:
            if c['chunk_id'] not in done:
                kept.append(c);
                done.add(c['chunk_id'])

        kept.sort(key=lambda x: x['combined_score'], reverse=True)
        return kept

    def _inject_parent_mini_context(self, selected_chunks, max_lines=25):
        """
        Pour chaque chunk sélectionné, ajoute un MINIMUM de contexte venant
        du parent (module / programme) si pas déjà présent.
        """
        new_chunks = []
        for c in selected_chunks:
            new_chunks.append(c)  # chunk cible
            meta = c['chunk']['metadata']
            if meta.get('is_partial'):  # on remonte au parent complet
                parent_id = meta.get('parent_entity_id')
                # trouve le premier chunk du parent
                parent_chunks = [ch for ch in selected_chunks
                                 if ch['chunk']['metadata'].get('entity_id') == parent_id]
                if not parent_chunks:
                    # fabriquer un mini-chunk
                    txt = c['chunk']['text']
                    header = "\n".join(txt.splitlines()[:max_lines])
                    mini = c.copy()
                    mini['chunk'] = mini['chunk'].copy()
                    mini['chunk']['text'] = header + "\n! [...]"
                    mini['combined_score'] *= 0.5
                    mini['chunk']['metadata'] = mini['chunk']['metadata'].copy()
                    mini['chunk']['metadata']['is_minimal_context'] = True
                    new_chunks.append(mini)
        return new_chunks

    def _create_entity_groups(self, top_chunks: List[Dict]) -> Dict[str, Dict]:
        """
        Crée des groupes d'entités en utilisant les métadonnées de tracking.
        """
        entity_groups = {}

        for chunk_data in top_chunks:
            metadata = chunk_data['chunk'].get('metadata', {})

            # Utiliser parent_entity_id pour les parties, entity_id pour les entités complètes
            if metadata.get('is_partial', False):
                entity_key = metadata.get('parent_entity_id')
            else:
                entity_key = metadata.get('entity_id')

            if not entity_key:
                # Fallback sur l'ancien système
                entity_key = metadata.get('base_entity_name') or metadata.get('entity_name', 'unknown')

            if entity_key not in entity_groups:
                # Récupérer les bounds de l'entité complète si disponibles
                entity_bounds = metadata.get('entity_bounds', {})

                entity_groups[entity_key] = {
                    'entity_name': metadata.get('base_entity_name') or metadata.get('entity_name', 'unknown'),
                    'entity_type': metadata.get('entity_type', 'code'),
                    'file': metadata.get('filename', 'Unknown'),
                    'filepath': metadata.get('filepath', ''),
                    'chunks': [],
                    'all_concepts': set(),
                    'all_matched_concepts': set(),
                    'best_score': 0,
                    'total_score': 0,
                    'entity_start': entity_bounds.get('start_line'),
                    'entity_end': entity_bounds.get('end_line'),
                    'expected_parts': metadata.get('total_parts', 1),
                    'parent_entity_id': entity_key
                }

            # Ajouter le chunk au groupe
            entity_groups[entity_key]['chunks'].append(chunk_data)

            # Mettre à jour les bounds si on trouve des valeurs plus précises
            if 'entity_bounds' in metadata:
                bounds = metadata['entity_bounds']
                entity_groups[entity_key]['entity_start'] = bounds.get('start_line')
                entity_groups[entity_key]['entity_end'] = bounds.get('end_line')

            # Aggréger les informations
            entity_groups[entity_key]['all_concepts'].update(
                c['label'] for c in chunk_data.get('chunk_concepts', [])
            )
            entity_groups[entity_key]['all_matched_concepts'].update(
                chunk_data.get('matched_concepts', [])
            )
            entity_groups[entity_key]['best_score'] = max(
                entity_groups[entity_key]['best_score'],
                chunk_data['combined_score']
            )
            entity_groups[entity_key]['total_score'] += chunk_data['combined_score']

        # Vérifier l'intégrité des groupes
        for entity_key, group in entity_groups.items():
            actual_parts = len(group['chunks'])
            expected_parts = group['expected_parts']

            if actual_parts < expected_parts:
                print(f"   ⚠️ Entité {group['entity_name']}: {actual_parts}/{expected_parts} parties trouvées")

            # Trier les chunks par part_sequence ou start_pos
            group['chunks'].sort(
                key=lambda x: x['chunk'].get('metadata', {}).get('part_sequence',
                                                                 x['chunk'].get('metadata', {}).get('start_pos', 0))
            )

        return entity_groups

    def _create_grouped_source(self, entity_info: Dict, source_index: int) -> Dict:
        """
        Crée une source regroupée avec les vraies limites de l'entité.
        """
        chunks = entity_info['chunks']

        # Utiliser les bounds de l'entité complète si disponibles
        if entity_info.get('entity_start') and entity_info.get('entity_end'):
            start_line = entity_info['entity_start']
            end_line = entity_info['entity_end']
        else:
            # Fallback: calculer depuis les chunks
            start_line = min(c['chunk'].get('metadata', {}).get('start_pos', 0) for c in chunks)
            end_line = max(c['chunk'].get('metadata', {}).get('end_pos', 0) for c in chunks)

        # Créer les informations des parties
        parts_info = []
        for chunk_data in chunks:
            metadata = chunk_data['chunk'].get('metadata', {})
            parts_info.append({
                'part_index': metadata.get('part_index', 0),
                'part_sequence': metadata.get('part_sequence', 0),
                'start_line': metadata.get('start_pos', 0),
                'end_line': metadata.get('end_pos', 0),
                'score': round(chunk_data['combined_score'], 3),
                'entity_name': metadata.get('entity_name', 'unknown')
            })

        # Trier par sequence
        parts_info.sort(key=lambda x: x['part_sequence'] or x['part_index'])

        return {
            'index': source_index,
            'file': entity_info['file'],
            'filepath': entity_info['filepath'],
            'entity': entity_info['entity_name'],
            'entity_type': entity_info['entity_type'],
            'start_line': start_line,
            'end_line': end_line,
            'concepts': list(entity_info['all_concepts']),
            'matched_concepts': list(entity_info['all_matched_concepts']),
            'score': round(entity_info['best_score'], 3),
            'is_grouped': True,
            'total_parts': len(chunks),
            'expected_parts': entity_info['expected_parts'],
            'parts_info': parts_info,
            'complete': len(chunks) == entity_info['expected_parts']
        }

    def _create_grouped_passage(self, entity_info: Dict, source_index: int) -> Dict:
        """
        Crée un passage regroupé pour une entité multi-parties.
        """
        chunks = entity_info['chunks']

        # Trier par position et concaténer le texte
        chunks.sort(key=lambda x: x['chunk'].get('metadata', {}).get('start_pos', 0))

        full_text_parts = []
        for i, chunk_data in enumerate(chunks):
            chunk = chunk_data['chunk']
            metadata = chunk.get('metadata', {})
            entity_name = metadata.get('entity_name', 'unknown')

            # Ajouter un marqueur pour chaque partie
            if len(chunks) > 1:
                full_text_parts.append(f"=== Partie {i + 1} de {entity_name} ===")

            full_text_parts.append(chunk['text'])

        full_text = '\n\n'.join(full_text_parts)

        # Métadonnées combinées
        first_chunk = chunks[0]['chunk']
        combined_metadata = first_chunk.get('metadata', {}).copy()
        combined_metadata.update({
            'is_grouped_entity': True,
            'total_parts': len(chunks),
            'combined_start_pos': min(c['chunk'].get('metadata', {}).get('start_pos', 0) for c in chunks),
            'combined_end_pos': max(c['chunk'].get('metadata', {}).get('end_pos', 0) for c in chunks)
        })

        return {
            'chunk_id': f"grouped_{entity_info['entity_name']}_{source_index}",
            'document_id': first_chunk.get('document_id', 'unknown'),
            'text': full_text,
            'similarity': entity_info['best_score'],
            'metadata': combined_metadata,
            'source_info': {
                'index': source_index,
                'is_grouped': True,
                'total_parts': len(chunks)
            },
            'document_name': entity_info['file']
        }

    def _boost_related_entity_chunks(
            self,
            chunk_scores: List[Dict],
            entity_boost: float = 1.5
    ) -> List[Dict]:
        """
        Booste les chunks qui appartiennent aux mêmes entités que les chunks les mieux classés.
        """
        if not chunk_scores:
            return chunk_scores

        # Identifier les entités des top chunks (top 20%)
        top_threshold = max(3, len(chunk_scores) // 5)
        top_chunks = chunk_scores[:top_threshold]

        # Collecter les entity_id des top chunks
        top_entity_ids = set()
        entity_scores = {}  # entity_id -> meilleur score

        for chunk_data in top_chunks:
            metadata = chunk_data['chunk'].get('metadata', {})
            entity_id = metadata.get('entity_id')

            if entity_id:
                top_entity_ids.add(entity_id)
                current_score = chunk_data['combined_score']

                if entity_id not in entity_scores or current_score > entity_scores[entity_id]:
                    entity_scores[entity_id] = current_score

        if not top_entity_ids:
            return chunk_scores

        print(f"🎯 Boosting chunks for {len(top_entity_ids)} top entities")

        # Appliquer le boost aux chunks des mêmes entités
        for chunk_data in chunk_scores:
            metadata = chunk_data['chunk'].get('metadata', {})
            entity_id = metadata.get('entity_id')

            if entity_id in top_entity_ids:
                # Calculer le boost basé sur le meilleur score de l'entité
                base_entity_score = entity_scores[entity_id]

                # Plus l'entité a un bon score de base, plus on booste ses autres chunks
                boost_factor = entity_boost if base_entity_score > 0.4 else (entity_boost * 0.7)

                # Appliquer le boost (mais pas au chunk qui a déjà le meilleur score)
                if chunk_data['combined_score'] < base_entity_score * 0.9:
                    original_score = chunk_data['combined_score']
                    chunk_data['combined_score'] *= boost_factor

                    # Marquer comme boosté pour l'affichage
                    chunk_data['entity_boosted'] = True
                    chunk_data['original_score'] = original_score

                    # Ajouter info sur l'entité dans matched_concepts
                    entity_name = metadata.get('base_entity_name', metadata.get('entity_name', ''))
                    if entity_name and entity_name not in chunk_data.get('matched_concepts', []):
                        chunk_data.setdefault('matched_concepts', []).append(f"Entity: {entity_name}")

        # Re-trier après boost
        chunk_scores.sort(key=lambda x: x['combined_score'], reverse=True)

        return chunk_scores

    def _are_concepts_related(self, concept1_uri: str, concept2_uri: str) -> bool:
        """
        Vérifie si deux concepts sont liés (parent/enfant ou frères).
        """
        if not self.concept_classifier:
            return False

        # Vérifier si l'un est parent de l'autre
        if hasattr(self.concept_classifier, 'is_parent_of'):
            if (self.concept_classifier.is_parent_of(concept1_uri, concept2_uri) or
                    self.concept_classifier.is_parent_of(concept2_uri, concept1_uri)):
                return True

        # Vérifier s'ils ont un parent commun
        concept1 = self.ontology_manager.concepts.get(concept1_uri)
        concept2 = self.ontology_manager.concepts.get(concept2_uri)

        if concept1 and concept2:
            if hasattr(concept1, 'parents') and hasattr(concept2, 'parents'):
                # Parents communs
                parents1 = {p.uri for p in concept1.parents} if concept1.parents else set()
                parents2 = {p.uri for p in concept2.parents} if concept2.parents else set()
                if parents1.intersection(parents2):
                    return True

        return False

    async def auto_concept_search(
            self,
            query: str,
            top_k: int = 5,
            min_confidence: float = 0.55,
            include_semantic_relations: bool = True,
            include_learned_relations: bool = True,
            max_concepts: int = 10,
            use_structure_boost=False
    ) -> Dict[str, Any]:
        """
        Recherche intelligente qui détecte automatiquement les concepts pertinents
        et utilise les relations apprises.

        Args:
            query: Requête de l'utilisateur
            top_k: Nombre maximum de passages à retourner
            min_confidence: Seuil minimal de confiance pour les concepts
            include_semantic_relations: Si True, inclut les relations sémantiques dans le prompt
            include_learned_relations: Si True, inclut les relations apprises
            max_concepts: Nombre maximum de concepts à considérer
            use_structure_boost: Si True, utilise le boost structurel pour la recherche
        Returns:
            Résultats de la recherche
        """

        # 1. Détecter les concepts pertinents avec l'approche intelligente
        relevant_concepts = await self.smart_concept_detection(query)

        if not relevant_concepts:
            # Si aucun concept n'est suffisamment pertinent, faire une recherche standard
            print("Aucun concept spécifique détecté. Recherche standard...")
            return await self.rag_engine.chat(query, top_k)

        print(f"Concepts détectés: {', '.join([c['label'] for c in relevant_concepts])}")

        # 2. Enrichir avec les relations apprises si activé
        enriched_concepts = list(relevant_concepts)
        discovered_relations = []

        if include_learned_relations and self.enable_relation_learning and self.relation_manager:
            print("🔗 Enrichissement avec les relations apprises...")

            for concept_info in relevant_concepts[:3]:  # Limiter aux 3 premiers concepts
                concept_uri = concept_info["concept_uri"]

                try:
                    # Découvrir les relations pour ce concept
                    relations_info = await self.discover_concept_relations(concept_uri, min_confidence=0.5)

                    if "relations" in relations_info and relations_info["relations"]:
                        discovered_relations.append({
                            "concept": concept_info["label"],
                            "concept_uri": concept_uri,
                            "relations": relations_info["relations"]
                        })

                        # Ajouter les concepts liés à la recherche
                        for relation_uri, relation_data in relations_info["relations"].items():
                            for obj_info in relation_data["objects"][:2]:  # Top 2 objets par relation
                                enriched_concepts.append({
                                    "concept_uri": obj_info["concept_uri"],
                                    "label": obj_info["label"],
                                    "confidence": obj_info["confidence"] * 0.8,  # Réduire légèrement
                                    "source": "learned_relation",
                                    "via_relation": relation_data["relation_label"],
                                    "from_concept": concept_info["label"]
                                })
                except Exception as e:
                    print(f"⚠️ Erreur lors de la découverte des relations pour {concept_info['label']}: {e}")

        # 3. Optimisation: générer l'embedding de la requête une seule fois
        query_embedding = await self.rag_engine.embedding_manager.provider.generate_embeddings([query])
        query_embedding = query_embedding[0]

        # 4. Effectuer la recherche pour chaque concept
        all_passages = []
        all_documents = set()
        concepts_used = []

        # Limiter au nombre de concepts maximum
        enriched_concepts = enriched_concepts[:max_concepts]

        for concept_info in enriched_concepts:
            concept_uri = concept_info["concept_uri"]
            concept_label = concept_info["label"]
            concept_confidence = concept_info["confidence"]
            concept_source = concept_info.get("source", "direct")

            # Affichage selon la source
            if concept_source == "learned_relation":
                print(
                    f"Recherche pour '{concept_label}' (via relation '{concept_info.get('via_relation', 'unknown')}' depuis '{concept_info.get('from_concept', 'unknown')}')")
            else:
                print(f"Recherche pour '{concept_label}' (confiance: {concept_confidence:.2f})")

            try:
                passages = []
                documents_used = []

                if use_structure_boost and hasattr(self.rag_engine, 'retriever'):
                    # Utiliser la nouvelle méthode de recherche avec boost
                    passages = await self.rag_engine.retriever.retrieve_with_structure_boost(
                        query=query,
                        document_id=None,
                        top_k=max(3, top_k // 2),
                        boost_headers=1.2,
                        boost_first_paragraph=1.1
                    )
                else:
                    # Rechercher avec le concept spécifique
                    result = await self.search_by_concept(
                        query=query,
                        concept_uri=concept_uri,
                        include_subconcepts=True,
                        top_k=max(2, top_k // 3),
                        confidence_threshold=0.3
                    )

                    if "passages" in result:
                        passages = result["passages"]
                    if "documents_used" in result:
                        documents_used = result["documents_used"]

                # Enrichir les passages avec les métadonnées
                for passage in passages:
                    passage["concept_confidence"] = concept_confidence
                    passage["concept"] = concept_label
                    passage["concept_source"] = concept_source

                    # Score adapté selon la source
                    if concept_source == "learned_relation":
                        passage["combined_score"] = passage["similarity"] * 0.6 + concept_confidence * 0.4
                        passage["via_relation"] = concept_info.get("via_relation", "")
                        passage["from_concept"] = concept_info.get("from_concept", "")
                    else:
                        passage["combined_score"] = passage["similarity"] * 0.7 + concept_confidence * 0.3

                    all_passages.append(passage)

                # Collecter les documents utilisés
                all_documents.update(documents_used)

                # Ajouter aux concepts utilisés
                concepts_used.append({
                    "label": concept_label,
                    "confidence": concept_confidence,
                    "uri": concept_uri,
                    "source": concept_source
                })

            except Exception as e:
                print(f"⚠️ Erreur lors de la recherche pour le concept {concept_label}: {str(e)}")

        # 5. Gestion du fallback si aucun passage trouvé
        if not all_passages:
            print("Aucun passage pertinent trouvé avec les concepts détectés. Recherche standard...")

            # Recherche standard avec l'embedding pré-calculé
            try:
                passages = await self.rag_engine.search_with_embedding(
                    query_embedding,
                    document_id=None,
                    top_k=top_k
                )

                if not passages:
                    # Dernier recours: recherche complète
                    return await self.rag_engine.chat(query, top_k)

                # Générer la réponse avec les passages trouvés
                answer = await self.rag_engine.generate_answer(query, passages)

                return {
                    "answer": answer,
                    "passages": passages,
                    "fallback_search": True,
                    "concepts_detected": [],
                    "discovered_relations": []
                }
            except Exception as e:
                print(f"⚠️ Erreur lors de la recherche de fallback: {e}")
                return await self.rag_engine.chat(query, top_k)

        # 6. Trier par score combiné et prendre les top_k
        all_passages.sort(key=lambda x: x.get("combined_score", x.get("similarity", 0)), reverse=True)
        top_passages = all_passages[:top_k]

        # 7. Construire les hiérarchies des concepts utilisés
        concept_hierarchies = []
        for concept_info in concepts_used:
            concept_uri = concept_info["uri"]

            try:
                # Obtenir la chaîne hiérarchique complète
                hierarchy = self.ontology_manager.get_concept_hierarchy_chain(concept_uri)
                hierarchy_str = " → ".join(hierarchy)

                concept_hierarchies.append({
                    "label": concept_info["label"],
                    "hierarchy": hierarchy_str,
                    "confidence": concept_info["confidence"],
                    "uri": concept_uri,
                    "source": concept_info["source"]
                })
            except Exception as e:
                print(f"⚠️ Erreur lors de la construction de la hiérarchie pour {concept_info['label']}: {e}")
                # Fallback sans hiérarchie
                concept_hierarchies.append({
                    "label": concept_info["label"],
                    "hierarchy": concept_info["label"],
                    "confidence": concept_info["confidence"],
                    "uri": concept_uri,
                    "source": concept_info["source"]
                })

        # 8. Créer le système prompt enrichi
        context_description = ""

        # Concepts principaux
        main_concepts = [h for h in concept_hierarchies if h["source"] == "direct"]
        if main_concepts:
            context_description += "Concepts principaux détectés:\n"
            for hier in main_concepts:
                context_description += f"- {hier['label']} (contexte: {hier['hierarchy']})\n"

        # Concepts liés par relations
        related_concepts = [h for h in concept_hierarchies if h["source"] == "learned_relation"]
        if related_concepts:
            context_description += "\nConcepts liés découverts via les relations apprises:\n"
            for hier in related_concepts:
                context_description += f"- {hier['label']} (contexte: {hier['hierarchy']})\n"

        system_prompt = f"""
    Tu es un assistant spécialisé répondant à une question qui concerne les concepts suivants:

    {context_description}
    """

        # 9. Ajouter les relations sémantiques si demandé
        if include_semantic_relations:
            semantic_relations = []

            for concept_info in concept_hierarchies:
                concept_uri = concept_info["uri"]
                related_concepts_semantic = []

                # Rechercher dans les axiomes
                try:
                    for axiom_type, source, target in self.ontology_manager.axioms:
                        if axiom_type.startswith("semantic_"):
                            rel_type = axiom_type.replace("semantic_", "")

                            if source == concept_uri:
                                target_concept = self.ontology_manager.concepts.get(target)
                                if target_concept:
                                    related_concepts_semantic.append(
                                        f"{concept_info['label']} {rel_type} {target_concept.label}")
                            elif target == concept_uri:
                                source_concept = self.ontology_manager.concepts.get(source)
                                if source_concept:
                                    related_concepts_semantic.append(
                                        f"{source_concept.label} {rel_type} {concept_info['label']}")

                    if related_concepts_semantic:
                        semantic_relations.append({
                            "concept": concept_info["label"],
                            "relations": related_concepts_semantic[:3]  # Limiter à 3 relations par concept
                        })
                except Exception as e:
                    print(f"⚠️ Erreur lors de la recherche des relations sémantiques pour {concept_info['label']}: {e}")

            if semantic_relations:
                system_prompt += "\nInformations supplémentaires sur les relations entre ces concepts:\n"
                for rel_info in semantic_relations:
                    system_prompt += f"\nRelations pour {rel_info['concept']}:\n"
                    for rel in rel_info['relations']:
                        system_prompt += f"- {rel}\n"

        # 10. Ajouter les relations apprises si disponibles
        if discovered_relations:
            system_prompt += "\nRelations apprises découvertes:\n"
            for rel_info in discovered_relations:
                system_prompt += f"\nPour {rel_info['concept']}:\n"
                for relation_uri, relation_data in rel_info['relations'].items():
                    system_prompt += f"- {relation_data['relation_label']}: "
                    objects = [obj['label'] for obj in relation_data['objects'][:3]]
                    system_prompt += ", ".join(objects) + "\n"

        # 11. Finaliser le prompt
        system_prompt += """
    Réponds en utilisant uniquement les informations du contexte fourni.
    Interprète chaque concept dans le contexte spécifique de sa hiérarchie,
    car le même terme peut avoir différentes significations selon le domaine.
    Si des relations apprises sont mentionnées, utilise-les pour enrichir ta réponse.
    """

        # 12. Générer la réponse
        try:
            answer = await self.rag_engine.generate_answer(query, top_passages, system_prompt)
        except Exception as e:
            print(f"⚠️ Erreur lors de la génération de la réponse: {e}")
            # Fallback sans système prompt
            answer = await self.rag_engine.generate_answer(query, top_passages)

        # 13. Retourner les résultats complets
        return {
            "answer": answer,
            "passages": top_passages,
            "concepts_detected": concept_hierarchies,
            "discovered_relations": discovered_relations,
            "documents_used": list(all_documents),
            "system_prompt": system_prompt,
            "total_concepts_used": len(concept_hierarchies),
            "enriched_with_relations": include_learned_relations and len(discovered_relations) > 0
        }

    async def train_relations_from_documents(self, min_examples: int = 5) -> Dict[str, bool]:
        """
        Entraîne les relations automatiquement à partir des documents classifiés.
        """
        if not self.enable_relation_learning or not self.relation_manager:
            return {}

        print("🔄 Extraction des triplets à partir des documents...")

        # Collecter tous les triplets des documents
        all_triples = []
        all_documents = await self.rag_engine.get_all_documents()

        for doc_id, doc_data in all_documents.items():
            # Analyser le contenu du document
            if 'content' in doc_data:
                content = doc_data['content']

                # Extraire les relations possibles du contenu
                relations = await self.extract_possible_relations(content, confidence_threshold=0.7)

                for relation in relations:
                    triple = (
                        relation["subject_uri"],
                        relation["relation_uri"],
                        relation["object_uri"]
                    )
                    all_triples.append(triple)

        print(f"📊 {len(all_triples)} triplets extraits")

        # Organiser par relation
        examples_by_relation = {}
        for s, r, o in all_triples:
            if r not in examples_by_relation:
                examples_by_relation[r] = []
            examples_by_relation[r].append((s, r, o))

        # Apprendre les relations
        return await self.learn_relations_from_examples(examples_by_relation)

    async def get_concept_neighborhood(self, concept_uri: str, max_distance: int = 2) -> Dict[str, Any]:
        """
        Obtient le voisinage d'un concept via les relations apprises.
        """
        if not self.enable_relation_learning or not self.relation_manager:
            return {"error": "Apprentissage de relations non disponible"}

        full_concept_uri = self.ontology_manager.resolve_uri(concept_uri)
        neighborhood = {"center": full_concept_uri, "neighbors": {}}

        visited = set()
        queue = [(full_concept_uri, 0)]

        while queue:
            current_concept, distance = queue.pop(0)

            if distance >= max_distance or current_concept in visited:
                continue

            visited.add(current_concept)

            # Découvrir les relations pour ce concept
            relations_info = await self.discover_concept_relations(current_concept, min_confidence=0.5)

            if "relations" in relations_info:
                for relation_uri, relation_data in relations_info["relations"].items():
                    if distance not in neighborhood["neighbors"]:
                        neighborhood["neighbors"][distance] = []

                    for obj_info in relation_data["objects"]:
                        neighbor_uri = obj_info["concept_uri"]

                        neighborhood["neighbors"][distance].append({
                            "concept_uri": neighbor_uri,
                            "label": obj_info["label"],
                            "relation": relation_data["relation_label"],
                            "confidence": obj_info["confidence"],
                            "distance": distance + 1
                        })

                        # Ajouter à la queue pour exploration plus profonde
                        if neighbor_uri not in visited:
                            queue.append((neighbor_uri, distance + 1))

        return neighborhood

    async def bootstrap_relations_from_ontology(self, min_similarity: float = 0.75,
                                                max_examples_per_relation: int = 20) -> Dict[str, bool]:
        """
        Bootstrap initial des relations en utilisant la structure de l'ontologie
        et la similarité avec les documents.

        Cette méthode ne doit PAS utiliser la vérité terrain !
        """
        if not self.enable_relation_learning or not self.relation_manager:
            return {}

        print(f"{BLUE}🚀 Bootstrap des relations à partir de la structure ontologique...{RESET}")

        # 1. Extraire tous les triplets possibles de l'ontologie (structure, pas vérité terrain)
        ontology_triplets = await self._extract_ontology_structure_triplets()

        if not ontology_triplets:
            print(f"{YELLOW}⚠️ Aucun triplet structurel trouvé dans l'ontologie{RESET}")
            return {}

        print(f"📊 {len(ontology_triplets)} triplets structurels extraits de l'ontologie")

        # 2. Récupérer tous les documents traités
        all_documents = await self.rag_engine.get_all_documents()
        if not all_documents:
            print(f"{YELLOW}⚠️ Aucun document disponible pour le bootstrap{RESET}")
            return {}

        # 3. Analyser chaque document pour trouver des correspondances
        relation_candidates = {}

        for doc_id, doc_data in tqdm(all_documents.items(), desc="Analyse des documents"):
            try:
                # Récupérer les chunks du document
                await self.rag_engine.document_store.load_document_chunks(doc_id)
                doc_chunks = await self.rag_engine.document_store.get_document_chunks(doc_id)

                if not doc_chunks:
                    continue

                # Pour chaque chunk
                for chunk in doc_chunks:
                    chunk_text = chunk.get('content', '')
                    if len(chunk_text) < 50:  # Ignorer les chunks trop courts
                        continue

                    # Obtenir l'embedding du chunk
                    chunk_embedding = self.rag_engine.embedding_manager.get_embedding(chunk['id'])
                    if chunk_embedding is None:
                        continue

                    # Comparer avec chaque triplet ontologique
                    for triplet in ontology_triplets:
                        similarity = await self._compare_triplet_with_chunk(
                            triplet, chunk_text, chunk_embedding
                        )

                        if similarity >= min_similarity:
                            relation_uri = triplet['relation_uri']

                            if relation_uri not in relation_candidates:
                                relation_candidates[relation_uri] = []

                            relation_candidates[relation_uri].append({
                                'subject': triplet['subject'],
                                'object': triplet['object'],
                                'chunk_text': chunk_text,
                                'similarity': similarity,
                                'doc_id': doc_id
                            })

            except Exception as e:
                print(f"⚠️ Erreur lors de l'analyse du document {doc_id}: {e}")
                continue

        # 4. Créer les exemples d'apprentissage pour chaque relation
        learning_examples = {}

        for relation_uri, candidates in relation_candidates.items():
            if len(candidates) < 3:  # Besoin d'au moins 3 exemples
                continue

            # Trier par similarité et prendre les meilleurs
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            best_candidates = candidates[:max_examples_per_relation]

            # Convertir en format d'apprentissage (sujet, objet)
            learning_examples[relation_uri] = [
                (cand['subject'], cand['object'])
                for cand in best_candidates
            ]

            print(f"  🎯 {len(best_candidates)} exemples trouvés pour {relation_uri.split('#')[-1]}")

        # 5. Apprendre les relations avec ces exemples
        if not learning_examples:
            print(f"{YELLOW}⚠️ Aucune correspondance suffisante trouvée{RESET}")
            return {}

        print(f"{GREEN}🎓 Apprentissage de {len(learning_examples)} relations avec bootstrap...{RESET}")

        # Récupérer les embeddings des concepts
        concept_embeddings = {}
        if self.concept_classifier and self.concept_classifier.concept_embeddings:
            concept_embeddings = self.concept_classifier.concept_embeddings

        # Apprendre les relations
        results = await self.relation_manager.learn_relations(
            triples=[(s, r, o) for r, examples in learning_examples.items() for s, o in examples],
            concept_embeddings=concept_embeddings,
            min_examples=3,
            force_relearn=False
        )

        success_count = sum(1 for success in results.values() if success)
        print(f"{GREEN}✓ Bootstrap terminé: {success_count}/{len(results)} relations apprises{RESET}")

        return results

    async def _extract_ontology_structure_triplets(self) -> List[Dict[str, str]]:
        """
        Extrait les triplets structurels de l'ontologie (hiérarchie, domaines, etc.)
        SANS utiliser la vérité terrain.
        """
        triplets = []

        # 1. Relations hiérarchiques entre concepts
        for concept_uri, concept in self.ontology_manager.concepts.items():
            if hasattr(concept, 'parents') and concept.parents:
                for parent in concept.parents:
                    if hasattr(parent, 'uri') and hasattr(parent, 'label'):
                        triplets.append({
                            'subject': concept.label,
                            'relation_uri': 'http://www.w3.org/2000/01/rdf-schema#subClassOf',
                            'object': parent.label,
                            'relation_type': 'hierarchical'
                        })

        # 2. Relations de domaine et range
        for relation_uri, relation in self.ontology_manager.relations.items():
            if hasattr(relation, 'domain') and hasattr(relation, 'range'):
                # Créer des triplets domaine -> relation -> range
                for domain_concept in relation.domain:
                    for range_concept in relation.range:
                        if (hasattr(domain_concept, 'label') and hasattr(range_concept, 'label') and
                                domain_concept.label and range_concept.label):
                            triplets.append({
                                'subject': domain_concept.label,
                                'relation_uri': relation_uri,
                                'object': range_concept.label,
                                'relation_type': 'domain_range'
                            })

        # 3. Axiomes ontologiques définis
        for axiom_type, source, target in self.ontology_manager.axioms:
            if axiom_type.startswith('semantic_'):
                # Convertir les URIs en labels
                source_label = self._uri_to_label(source)
                target_label = self._uri_to_label(target)

                if source_label and target_label:
                    triplets.append({
                        'subject': source_label,
                        'relation_uri': f"http://example.org/ontology#{axiom_type}",
                        'object': target_label,
                        'relation_type': 'semantic_axiom'
                    })

        # 4. Relations prédéfinies dans l'ontologie
        for relation_uri, relation in self.ontology_manager.relations.items():
            if hasattr(relation, 'label') and relation.label:
                # Chercher des exemples implicites dans les descriptions
                if hasattr(relation, 'comment') or hasattr(relation, 'description'):
                    # Parser la description pour extraire des exemples
                    examples = self._parse_relation_description(relation)
                    for subject, object_val in examples:
                        triplets.append({
                            'subject': subject,
                            'relation_uri': relation_uri,
                            'object': object_val,
                            'relation_type': 'description_example'
                        })

        print(f"📋 Types de triplets extraits:")
        types_count = {}
        for triplet in triplets:
            rel_type = triplet['relation_type']
            types_count[rel_type] = types_count.get(rel_type, 0) + 1

        for rel_type, count in types_count.items():
            print(f"  - {rel_type}: {count} triplets")

        return triplets

    async def _compare_triplet_with_chunk(self, triplet: Dict[str, str], chunk_text: str,
                                          chunk_embedding: np.ndarray) -> float:
        """
        Compare un triplet ontologique avec un chunk de document.
        """
        try:
            # 1. Créer une représentation textuelle du triplet
            triplet_text = f"{triplet['subject']} {triplet.get('relation_label', 'related to')} {triplet['object']}"

            # 2. Générer l'embedding du triplet
            triplet_embeddings = await self.rag_engine.embedding_manager.provider.generate_embeddings([triplet_text])
            if not triplet_embeddings:
                return 0.0

            triplet_embedding = triplet_embeddings[0]

            # 3. Calculer la similarité cosinus
            chunk_norm = chunk_embedding / (np.linalg.norm(chunk_embedding) + 1e-8)
            triplet_norm = triplet_embedding / (np.linalg.norm(triplet_embedding) + 1e-8)

            similarity = np.dot(chunk_norm, triplet_norm)

            # 4. Bonus si les entités apparaissent explicitement dans le texte
            text_lower = chunk_text.lower()
            subject_bonus = 0.1 if triplet['subject'].lower() in text_lower else 0.0
            object_bonus = 0.1 if triplet['object'].lower() in text_lower else 0.0

            # 5. Bonus si les mots-clés de la relation apparaissent
            relation_keywords = self._extract_relation_keywords(triplet.get('relation_uri', ''))
            keyword_bonus = 0.05 * sum(1 for keyword in relation_keywords if keyword.lower() in text_lower)

            final_similarity = similarity + subject_bonus + object_bonus + keyword_bonus

            return min(1.0, final_similarity)

        except Exception as e:
            print(f"⚠️ Erreur lors de la comparaison: {e}")
            return 0.0

    def _uri_to_label(self, uri: str) -> Optional[str]:
        """Convertit une URI en label lisible."""
        # Chercher dans les concepts
        concept = self.ontology_manager.concepts.get(uri)
        if concept and hasattr(concept, 'label') and concept.label:
            return concept.label

        # Chercher dans les relations
        relation = self.ontology_manager.relations.get(uri)
        if relation and hasattr(relation, 'label') and relation.label:
            return relation.label

        # Fallback: extraire de l'URI
        if '#' in uri:
            return uri.split('#')[-1].replace('_', ' ')
        return uri.split('/')[-1].replace('_', ' ')

    def _parse_relation_description(self, relation) -> List[Tuple[str, str]]:
        """Parse les descriptions de relations pour extraire des exemples."""
        examples = []

        description = ""
        if hasattr(relation, 'comment') and relation.comment:
            description += relation.comment + " "
        if hasattr(relation, 'description') and relation.description:
            description += relation.description

        if not description:
            return examples

        # Patterns pour extraire des exemples (très basique)
        import re

        # Pattern: "X relation Y" ou "X verb Y"
        patterns = [
            r'(\w+(?:\s+\w+)*)\s+(?:is|are|has|have|was|were)\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+(?:like|such as|including)\s+(\w+(?:\s+\w+)*)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            for subject, obj in matches:
                if len(subject.split()) <= 3 and len(obj.split()) <= 3:  # Limiter aux phrases courtes
                    examples.append((subject.strip(), obj.strip()))

        return examples[:5]  # Max 5 exemples par relation

    def _extract_relation_keywords(self, relation_uri: str) -> List[str]:
        """Extrait des mots-clés de l'URI de relation."""
        keywords = []

        # Extraire de l'URI
        if '#' in relation_uri:
            name = relation_uri.split('#')[-1]
        else:
            name = relation_uri.split('/')[-1]

        # Décomposer en mots
        words = re.findall(r'[A-Z][a-z]*|[a-z]+', name)
        keywords.extend([word.lower() for word in words if len(word) > 2])

        # Ajouter des synonymes basiques
        synonyms = {
            'has': ['contain', 'include', 'possess'],
            'is': ['be', 'represent', 'constitute'],
            'part': ['component', 'element', 'piece'],
            'type': ['kind', 'category', 'class'],
            'location': ['place', 'position', 'site'],
            'time': ['date', 'period', 'moment'],
        }

        for keyword in list(keywords):
            if keyword in synonyms:
                keywords.extend(synonyms[keyword])

        return keywords

    async def bootstrap_relations_from_ontology(self, min_similarity: float = 0.75,
                                                max_examples_per_relation: int = 20) -> Dict[str, bool]:
        """
        Bootstrap initial des relations en utilisant la structure de l'ontologie
        et la similarité avec les documents.

        Cette méthode ne doit PAS utiliser la vérité terrain !
        """
        if not self.enable_relation_learning or not self.relation_manager:
            return {}

        print(f"{BLUE}🚀 Bootstrap des relations à partir de la structure ontologique...{RESET}")

        # 1. Extraire tous les triplets possibles de l'ontologie (structure, pas vérité terrain)
        ontology_triplets = await self._extract_ontology_structure_triplets()

        if not ontology_triplets:
            print(f"{YELLOW}⚠️ Aucun triplet structurel trouvé dans l'ontologie{RESET}")
            return {}

        print(f"📊 {len(ontology_triplets)} triplets structurels extraits de l'ontologie")

        # 2. Récupérer tous les documents traités
        all_documents = await self.rag_engine.get_all_documents()
        if not all_documents:
            print(f"{YELLOW}⚠️ Aucun document disponible pour le bootstrap{RESET}")
            return {}

        # 3. Analyser chaque document pour trouver des correspondances
        relation_candidates = {}

        for doc_id, doc_data in tqdm(all_documents.items(), desc="Analyse des documents"):
            try:
                # Récupérer les chunks du document
                await self.rag_engine.document_store.load_document_chunks(doc_id)
                doc_chunks = await self.rag_engine.document_store.get_document_chunks(doc_id)

                if not doc_chunks:
                    continue

                # Pour chaque chunk
                for chunk in doc_chunks:
                    chunk_text = chunk.get('content', '')
                    if len(chunk_text) < 50:  # Ignorer les chunks trop courts
                        continue

                    # Obtenir l'embedding du chunk
                    chunk_embedding = self.rag_engine.embedding_manager.get_embedding(chunk['id'])
                    if chunk_embedding is None:
                        continue

                    # Comparer avec chaque triplet ontologique
                    for triplet in ontology_triplets:
                        similarity = await self._compare_triplet_with_chunk(
                            triplet, chunk_text, chunk_embedding
                        )

                        if similarity >= min_similarity:
                            relation_uri = triplet['relation_uri']

                            if relation_uri not in relation_candidates:
                                relation_candidates[relation_uri] = []

                            relation_candidates[relation_uri].append({
                                'subject': triplet['subject'],
                                'object': triplet['object'],
                                'chunk_text': chunk_text,
                                'similarity': similarity,
                                'doc_id': doc_id
                            })

            except Exception as e:
                print(f"⚠️ Erreur lors de l'analyse du document {doc_id}: {e}")
                continue

        # 4. Créer les exemples d'apprentissage pour chaque relation
        learning_examples = {}

        for relation_uri, candidates in relation_candidates.items():
            if len(candidates) < 3:  # Besoin d'au moins 3 exemples
                continue

            # Trier par similarité et prendre les meilleurs
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            best_candidates = candidates[:max_examples_per_relation]

            # Convertir en format d'apprentissage (sujet, objet)
            learning_examples[relation_uri] = [
                (cand['subject'], cand['object'])
                for cand in best_candidates
            ]

            print(f"  🎯 {len(best_candidates)} exemples trouvés pour {relation_uri.split('#')[-1]}")

        # 5. Apprendre les relations avec ces exemples
        if not learning_examples:
            print(f"{YELLOW}⚠️ Aucune correspondance suffisante trouvée{RESET}")
            return {}

        print(f"{GREEN}🎓 Apprentissage de {len(learning_examples)} relations avec bootstrap...{RESET}")

        # Récupérer les embeddings des concepts
        concept_embeddings = {}
        if self.concept_classifier and self.concept_classifier.concept_embeddings:
            concept_embeddings = self.concept_classifier.concept_embeddings

        # Apprendre les relations
        results = await self.relation_manager.learn_relations(
            triples=[(s, r, o) for r, examples in learning_examples.items() for s, o in examples],
            concept_embeddings=concept_embeddings,
            min_examples=3,
            force_relearn=False
        )

        success_count = sum(1 for success in results.values() if success)
        print(f"{GREEN}✓ Bootstrap terminé: {success_count}/{len(results)} relations apprises{RESET}")

        return results

    async def _extract_ontology_structure_triplets(self) -> List[Dict[str, str]]:
        """
        Extrait les triplets structurels de l'ontologie (hiérarchie, domaines, etc.)
        SANS utiliser la vérité terrain.
        """
        triplets = []

        # 1. Relations hiérarchiques entre concepts
        for concept_uri, concept in self.ontology_manager.concepts.items():
            if hasattr(concept, 'parents') and concept.parents:
                for parent in concept.parents:
                    if hasattr(parent, 'uri') and hasattr(parent, 'label'):
                        triplets.append({
                            'subject': concept.label,
                            'relation_uri': 'http://www.w3.org/2000/01/rdf-schema#subClassOf',
                            'object': parent.label,
                            'relation_type': 'hierarchical'
                        })

        # 2. Relations de domaine et range
        for relation_uri, relation in self.ontology_manager.relations.items():
            if hasattr(relation, 'domain') and hasattr(relation, 'range'):
                # Créer des triplets domaine -> relation -> range
                for domain_concept in relation.domain:
                    for range_concept in relation.range:
                        if (hasattr(domain_concept, 'label') and hasattr(range_concept, 'label') and
                                domain_concept.label and range_concept.label):
                            triplets.append({
                                'subject': domain_concept.label,
                                'relation_uri': relation_uri,
                                'object': range_concept.label,
                                'relation_type': 'domain_range'
                            })

        # 3. Axiomes ontologiques définis
        for axiom_type, source, target in self.ontology_manager.axioms:
            if axiom_type.startswith('semantic_'):
                # Convertir les URIs en labels
                source_label = self._uri_to_label(source)
                target_label = self._uri_to_label(target)

                if source_label and target_label:
                    triplets.append({
                        'subject': source_label,
                        'relation_uri': f"http://example.org/ontology#{axiom_type}",
                        'object': target_label,
                        'relation_type': 'semantic_axiom'
                    })

        # 4. Relations prédéfinies dans l'ontologie
        for relation_uri, relation in self.ontology_manager.relations.items():
            if hasattr(relation, 'label') and relation.label:
                # Chercher des exemples implicites dans les descriptions
                if hasattr(relation, 'comment') or hasattr(relation, 'description'):
                    # Parser la description pour extraire des exemples
                    examples = self._parse_relation_description(relation)
                    for subject, object_val in examples:
                        triplets.append({
                            'subject': subject,
                            'relation_uri': relation_uri,
                            'object': object_val,
                            'relation_type': 'description_example'
                        })

        print(f"📋 Types de triplets extraits:")
        types_count = {}
        for triplet in triplets:
            rel_type = triplet['relation_type']
            types_count[rel_type] = types_count.get(rel_type, 0) + 1

        for rel_type, count in types_count.items():
            print(f"  - {rel_type}: {count} triplets")

        return triplets

    async def _compare_triplet_with_chunk(self, triplet: Dict[str, str], chunk_text: str,
                                          chunk_embedding: np.ndarray) -> float:
        """
        Compare un triplet ontologique avec un chunk de document.
        """
        try:
            # 1. Créer une représentation textuelle du triplet
            triplet_text = f"{triplet['subject']} {triplet.get('relation_label', 'related to')} {triplet['object']}"

            # 2. Générer l'embedding du triplet
            triplet_embeddings = await self.rag_engine.embedding_manager.provider.generate_embeddings([triplet_text])
            if not triplet_embeddings:
                return 0.0

            triplet_embedding = triplet_embeddings[0]

            # 3. Calculer la similarité cosinus
            chunk_norm = chunk_embedding / (np.linalg.norm(chunk_embedding) + 1e-8)
            triplet_norm = triplet_embedding / (np.linalg.norm(triplet_embedding) + 1e-8)

            similarity = np.dot(chunk_norm, triplet_norm)

            # 4. Bonus si les entités apparaissent explicitement dans le texte
            text_lower = chunk_text.lower()
            subject_bonus = 0.1 if triplet['subject'].lower() in text_lower else 0.0
            object_bonus = 0.1 if triplet['object'].lower() in text_lower else 0.0

            # 5. Bonus si les mots-clés de la relation apparaissent
            relation_keywords = self._extract_relation_keywords(triplet.get('relation_uri', ''))
            keyword_bonus = 0.05 * sum(1 for keyword in relation_keywords if keyword.lower() in text_lower)

            final_similarity = similarity + subject_bonus + object_bonus + keyword_bonus

            return min(1.0, final_similarity)

        except Exception as e:
            print(f"⚠️ Erreur lors de la comparaison: {e}")
            return 0.0

    def _uri_to_label(self, uri: str) -> Optional[str]:
        """Convertit une URI en label lisible."""
        # Chercher dans les concepts
        concept = self.ontology_manager.concepts.get(uri)
        if concept and hasattr(concept, 'label') and concept.label:
            return concept.label

        # Chercher dans les relations
        relation = self.ontology_manager.relations.get(uri)
        if relation and hasattr(relation, 'label') and relation.label:
            return relation.label

        # Fallback: extraire de l'URI
        if '#' in uri:
            return uri.split('#')[-1].replace('_', ' ')
        return uri.split('/')[-1].replace('_', ' ')

    def _parse_relation_description(self, relation) -> List[Tuple[str, str]]:
        """Parse les descriptions de relations pour extraire des exemples."""
        examples = []

        description = ""
        if hasattr(relation, 'comment') and relation.comment:
            description += relation.comment + " "
        if hasattr(relation, 'description') and relation.description:
            description += relation.description

        if not description:
            return examples

        # Patterns pour extraire des exemples (très basique)
        import re

        # Pattern: "X relation Y" ou "X verb Y"
        patterns = [
            r'(\w+(?:\s+\w+)*)\s+(?:is|are|has|have|was|were)\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+(?:like|such as|including)\s+(\w+(?:\s+\w+)*)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            for subject, obj in matches:
                if len(subject.split()) <= 3 and len(obj.split()) <= 3:  # Limiter aux phrases courtes
                    examples.append((subject.strip(), obj.strip()))

        return examples[:5]  # Max 5 exemples par relation

    def _extract_relation_keywords(self, relation_uri: str) -> List[str]:
        """Extrait des mots-clés de l'URI de relation."""
        keywords = []

        # Extraire de l'URI
        if '#' in relation_uri:
            name = relation_uri.split('#')[-1]
        else:
            name = relation_uri.split('/')[-1]

        # Décomposer en mots
        words = re.findall(r'[A-Z][a-z]*|[a-z]+', name)
        keywords.extend([word.lower() for word in words if len(word) > 2])

        # Ajouter des synonymes basiques
        synonyms = {
            'has': ['contain', 'include', 'possess'],
            'is': ['be', 'represent', 'constitute'],
            'part': ['component', 'element', 'piece'],
            'type': ['kind', 'category', 'class'],
            'location': ['place', 'position', 'site'],
            'time': ['date', 'period', 'moment'],
        }

        for keyword in list(keywords):
            if keyword in synonyms:
                keywords.extend(synonyms[keyword])

        return keywords