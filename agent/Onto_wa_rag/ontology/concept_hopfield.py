"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# ontology/concept_hopfield.py
import asyncio
import os
import re
from typing import List, Dict, Any

import numpy as np
import rdflib
import pywt

from ..ontology.hopfield_network import ModernHopfieldNetwork


class WaveletDecomposer:
    """
    Gestionnaire de décomposition en ondelettes pour les embeddings de concepts.
    Adapté pour les réseaux de Hopfield hiérarchiques.
    """

    def __init__(
            self,
            wavelet: str = 'db3',
            levels: int = 5,
            normalize_levels: bool = True
    ):
        """
        Initialise le décomposeur d'ondelettes.

        Args:
            wavelet: Type d'ondelette (db1, db3, haar, etc.)
            levels: Nombre de niveaux de décomposition
            normalize_levels: Si True, normalise chaque niveau
        """
        self.wavelet = wavelet
        self.levels = levels
        self.normalize_levels = normalize_levels

        # Cache pour les décompositions
        self.decompositions_cache = {}  # pattern_id -> {level: decomposition}

    def decompose_embedding(self, embedding: np.ndarray, pattern_id: str = None) -> Dict[int, np.ndarray]:
        """
        Décompose un embedding en plusieurs niveaux d'ondelettes.

        Args:
            embedding: Embedding à décomposer
            pattern_id: ID pour le cache (optionnel)

        Returns:
            Dictionnaire {niveau: embedding_décomposé}
        """
        # Vérifier le cache si un ID est fourni
        if pattern_id and pattern_id in self.decompositions_cache:
            return self.decompositions_cache[pattern_id]

        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # Normaliser l'embedding original si demandé
        if self.normalize_levels:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        decompositions = {0: embedding}  # Niveau 0 = original
        working_embedding = embedding.copy()

        for level in range(1, self.levels + 1):
            if len(working_embedding) < 2:
                # Si trop petit, réutiliser le niveau précédent
                decompositions[level] = decompositions[level - 1]
                continue

            try:
                # Décomposition en ondelettes
                approx, detail = pywt.dwt(working_embedding, self.wavelet)

                # Normaliser l'approximation si demandé
                if self.normalize_levels and len(approx) > 0:
                    norm = np.linalg.norm(approx)
                    if norm > 0:
                        approx = approx / norm

                decompositions[level] = approx
                working_embedding = approx

            except Exception as e:
                print(f"Erreur décomposition niveau {level}: {e}")
                decompositions[level] = decompositions[level - 1]

        # Mettre en cache si un ID est fourni
        if pattern_id:
            self.decompositions_cache[pattern_id] = decompositions

        return decompositions

    def clear_cache(self):
        """Vide le cache des décompositions."""
        self.decompositions_cache.clear()


class ConceptHopfieldClassifier:
    """Système de classification des concepts utilisant des réseaux de Hopfield hiérarchiques."""

    def __init__(
            self,
            rag_engine,
            ontology_manager,
            storage_dir: str = "concept_hopfield_models",
            domain_specific: bool = False,  # option pour séparer par domaine
            current_domain: str = None,  # domaine actuellement traité
            multiscale_mode: bool = False,
            wavelet_config: Dict[str, Any] = None
    ):
        """
        Système de classification des concepts.

        Args:
            rag_engine: Moteur RAG
            ontology_manager: Gestionnaire d'ontologie
            storage_dir: Répertoire de stockage
            domain_specific: Mode spécifique au domaine
            current_domain: Domaine actuel
            multiscale_mode: Activer le mode multiscale avec ondelettes
            wavelet_config: Configuration des ondelettes
        """
        self.rag_engine = rag_engine
        self.ontology_manager = ontology_manager
        self.storage_dir = storage_dir
        self.domain_specific = domain_specific
        self.current_domain = current_domain

        #  Configuration multiscale
        self.multiscale_mode = multiscale_mode
        if wavelet_config is None:
            wavelet_config = {'wavelet': 'coif3', 'levels': 3}
        self.wavelet_config = wavelet_config

        # Créer le répertoire de stockage
        os.makedirs(storage_dir, exist_ok=True)

        # Si mode spécifique au domaine, ajouter le sous-répertoire du domaine
        if domain_specific and current_domain:
            self.domain_storage_dir = os.path.join(storage_dir, current_domain)
            os.makedirs(self.domain_storage_dir, exist_ok=True)
        else:
            self.domain_storage_dir = storage_dir

        # UN réseau de Hopfield par niveau
        self.level_networks = {}  # niveau -> réseau de Hopfield

        # Structure hiérarchique des concepts
        self.concept_hierarchy = {}  # URI concept -> [sous-concepts URIs]
        self.concept_to_level = {}  # URI concept -> niveau
        self.level_to_concepts = {}  # niveau -> [URI concepts]
        self.max_level = 1

        # Embeddings représentatifs de chaque concept
        self.concept_embeddings = {}  # URI concept -> embedding

    async def initialize(self, auto_build: bool = True, max_concepts_per_level: int = 20, domain_filter: str = None):
        """Initialise la structure hiérarchique et charge les réseaux existants."""
        print("Initialisation du classifieur de concepts hiérarchique...")

        # 1. Analyser la structure hiérarchique des concepts dans l'ontologie
        await self._build_concept_hierarchy(domain_filter)

        # 2. Charger ou créer les réseaux de Hopfield pour chaque niveau
        for level in range(1, self.max_level + 1):
            # Utiliser le répertoire spécifique au domaine si nécessaire
            if self.domain_specific and self.current_domain:
                level_dir = os.path.join(self.domain_storage_dir, f"level_{level}")
            else:
                level_dir = os.path.join(self.storage_dir, f"level_{level}")

            os.makedirs(level_dir, exist_ok=True)

            # Créer un réseau pour ce niveau
            network = ModernHopfieldNetwork(
                beta=22.0,
                normalize_patterns=True,
                storage_dir=level_dir,
                multiscale_mode=self.multiscale_mode,
                wavelet_config=self.wavelet_config
            )

            # Tenter de charger un réseau existant
            if self.multiscale_mode:
                loaded = network.load_multiscale()
            else:
                loaded = network.load()

            if loaded:
                mode_str = "multiscale" if self.multiscale_mode else "classique"
                print(f"✓ Réseau de concepts niveau {level} chargé (mode {mode_str})")
            else:
                print(f"✓ Nouveau réseau de concepts créé pour le niveau {level}")

            self.level_networks[level] = network

        # 3. Charger les embeddings des concepts s'ils existent
        await self._load_concept_embeddings()

        # 4. Construction automatique des embeddings si demandé
        if auto_build and not self.concept_embeddings:
            print("Aucun embedding de concept trouvé, construction automatique...")
            await self.build_concept_embeddings(max_concepts_per_level=max_concepts_per_level)

    async def classify_embedding_direct(self, embedding: np.ndarray, min_confidence: float = 0.5, top_k: int =10) -> List[
        Dict[str, Any]]:
        """
        Classifie directement un embedding en le comparant aux concepts.
        AUCUNE recherche dans la base, juste comparaison embedding vs concepts.

        Args:
            embedding: Embedding normalisé à classifier
            min_confidence: Seuil minimal de confiance

        Returns:
            Liste des concepts détectés avec leur confiance
        """
        if not self.concept_embeddings:
            self.logger.warning("Aucun embedding de concept disponible")
            return []

        # S'assurer que l'embedding est normalisé
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Calculer les similarités avec tous les concepts
        concept_scores = []

        for concept_uri, concept_embedding in self.concept_embeddings.items():
            # Similarité cosinus directe
            similarity = float(np.dot(embedding, concept_embedding))

            if similarity >= min_confidence:
                concept = self.ontology_manager.concepts.get(concept_uri, {})
                concept_label = concept.label if hasattr(concept, 'label') else concept_uri.split('#')[-1]

                concept_scores.append({
                    'concept_uri': concept_uri,
                    'label': concept_label,
                    'confidence': similarity,
                    'source': 'direct_classification'
                })

        # Trier par confiance décroissante
        concept_scores.sort(key=lambda x: x['confidence'], reverse=True)

        return concept_scores[:top_k]  # Top 10 concepts

    async def _build_concept_hierarchy(self, domain_filter: str = None):
        """
        Analyse l'ontologie et construit la hiérarchie des concepts.
        Filtre optionnellement par domaine.

        Args:
            domain_filter: Si spécifié, ne prend que les concepts du domaine indiqué
        """
        # Réinitialiser les structures
        self.concept_hierarchy = {}
        self.concept_to_level = {}
        self.level_to_concepts = {}

        # Collecter les concepts du domaine spécifié
        domain_concepts = {}

        if domain_filter:
            print(f"Filtrage des concepts pour le domaine '{domain_filter}'...")
            # Ne garder que les concepts qui appartiennent au domaine
            for uri, concept in self.ontology_manager.concepts.items():
                if domain_filter in uri:
                    domain_concepts[uri] = concept

            if not domain_concepts:
                print(f"⚠️ Aucun concept trouvé pour le domaine '{domain_filter}'")
                # Fallback - essayer avec une correspondance moins stricte
                for uri, concept in self.ontology_manager.concepts.items():
                    # Extraire l'ID numérique (ex: "18" dans "18_scientist")
                    match = re.search(r'(\d+)_', domain_filter)
                    if match and match.group(1) in uri:
                        domain_concepts[uri] = concept
        else:
            # Sans filtre, prendre tous les concepts
            domain_concepts = self.ontology_manager.concepts

        # Précharger les labels pour les afficher dans les logs
        concept_labels = {}
        for uri, concept in domain_concepts.items():
            label = self._get_concept_label(uri)
            concept_labels[uri] = label

        print(f"Construction de la hiérarchie pour {len(domain_concepts)} concepts" +
              (f" du domaine '{domain_filter}'" if domain_filter else ""))

        # Construire la hiérarchie des concepts en deux étapes

        # 1. Initialiser les entrées pour tous les concepts filtrés
        for uri in domain_concepts:
            if uri not in self.concept_hierarchy:
                self.concept_hierarchy[uri] = []

        # 2. Analyser les relations parent-enfant entre concepts filtrés
        for uri, concept in domain_concepts.items():
            # Collecter les enfants depuis les relations de concept
            if hasattr(concept, 'children') and concept.children:
                for child in concept.children:
                    child_uri = child.uri
                    if child_uri in domain_concepts:  # Ne considérer que les enfants dans le filtre
                        self.concept_hierarchy[uri].append(child_uri)

                        # S'assurer que l'enfant a une entrée
                        if child_uri not in self.concept_hierarchy:
                            self.concept_hierarchy[child_uri] = []

            # Collecter les parents - important pour établir la hiérarchie
            if hasattr(concept, 'parents') and concept.parents:
                for parent in concept.parents:
                    parent_uri = parent.uri
                    if parent_uri in domain_concepts:  # Ne considérer que les parents dans le filtre
                        if parent_uri not in self.concept_hierarchy:
                            self.concept_hierarchy[parent_uri] = []

                        # Ajouter l'enfant au parent
                        if uri not in self.concept_hierarchy[parent_uri]:
                            self.concept_hierarchy[parent_uri].append(uri)

        # 3. Extraire les relations subClassOf des axiomes
        for axiom_type, source, target in self.ontology_manager.axioms:
            if axiom_type == "subsumption":
                # Ne considérer que les relations entre concepts filtrés
                if source in domain_concepts and target in domain_concepts:
                    if target in self.concept_hierarchy and source not in self.concept_hierarchy[target]:
                        self.concept_hierarchy[target].append(source)

                    # S'assurer que le sous-concept a une entrée
                    if source not in self.concept_hierarchy:
                        self.concept_hierarchy[source] = []

                    # Afficher les relations avec labels au lieu des URIs
                    source_label = concept_labels.get(source, source.split('#')[-1])
                    target_label = concept_labels.get(target, target.split('#')[-1])
                    print(f"Relation subClassOf: {source_label} -> {target_label}")

        # 4. Déterminer le niveau de chaque concept
        for uri in self.concept_hierarchy:
            level = self._calculate_concept_level(uri)
            self.concept_to_level[uri] = level

            if level not in self.level_to_concepts:
                self.level_to_concepts[level] = []

            self.level_to_concepts[level].append(uri)

            # Mettre à jour le niveau maximum
            self.max_level = max(self.max_level, level)

        # Afficher les statistiques
        concepts_count = len(self.concept_hierarchy)
        print(f"✓ Hiérarchie de concepts construite avec {concepts_count} concepts sur {self.max_level} niveaux")

        for level, concepts in self.level_to_concepts.items():
            print(f"  - Niveau {level}: {len(concepts)} concepts")

        # Si aucun concept n'a été trouvé, ajouter un fallback
        if not self.concept_hierarchy and domain_filter:
            print(f"⚠️ Aucune hiérarchie de concepts trouvée pour '{domain_filter}', utilisation du mode fallback")
            # Ajouter un concept générique pour éviter les erreurs
            fallback_uri = f"http://fallback.org/{domain_filter}/concept"
            self.concept_hierarchy[fallback_uri] = []
            self.concept_to_level[fallback_uri] = 1
            if 1 not in self.level_to_concepts:
                self.level_to_concepts[1] = []
            self.level_to_concepts[1].append(fallback_uri)
            self.max_level = 1

    def _get_concept_label(self, uri: str) -> str:
        """
        Extrait un label lisible pour un concept.

        Args:
            uri: URI du concept

        Returns:
            Label humainement lisible
        """
        # Vérifier si le concept a déjà un label
        concept = self.ontology_manager.concepts.get(uri)
        if concept and hasattr(concept, 'label') and concept.label:
            return concept.label

        # Chercher un label dans le graphe RDF
        uri_ref = rdflib.URIRef(uri)

        # Propriété skos:prefLabel (standard EMMO)
        for _, _, label in self.ontology_manager.graph.triples(
                (uri_ref, rdflib.URIRef("http://www.w3.org/2004/02/skos/core#prefLabel"), None)):
            return str(label)

        # Propriété rdfs:label
        for _, _, label in self.ontology_manager.graph.triples((uri_ref, rdflib.RDFS.label, None)):
            return str(label)

        # Propriété EMMO elucidation
        emmo_elucidation = rdflib.URIRef("https://w3id.org/emmo#EMMO_967080e5_2f42_4eb2_a3a9_c58143e835f9")
        for _, _, desc in self.ontology_manager.graph.triples((uri_ref, emmo_elucidation, None)):
            # Si on a une description mais pas de label, utiliser les premiers mots
            words = str(desc).split()
            if words:
                return " ".join(words[:3]) + "..."

                # En dernier recours, extraire de l'URI
        if '#' in uri:
            return uri.split('#')[-1]
        return uri.split('/')[-1]

    async def _load_concept_embeddings(self):
        """Charge les embeddings de concepts existants."""
        import pickle

        embeddings_path = os.path.join(self.storage_dir, "concept_embeddings.pkl")

        if os.path.exists(embeddings_path):
            try:
                with open(embeddings_path, 'rb') as f:
                    self.concept_embeddings = pickle.load(f)
                print(f"✓ Embeddings de {len(self.concept_embeddings)} concepts chargés")
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement des embeddings de concepts: {e}")
                self.concept_embeddings = {}

    async def _save_concept_embeddings(self):
        """Sauvegarde les embeddings des concepts."""
        import pickle

        embeddings_path = os.path.join(self.storage_dir, "concept_embeddings.pkl")

        try:
            with open(embeddings_path, 'wb') as f:
                pickle.dump(self.concept_embeddings, f)
        except Exception as e:
            print(f"⚠️ Erreur lors de la sauvegarde des embeddings de concepts: {e}")

    async def _classify_at_level(
            self,
            document_embedding: np.ndarray,
            level: int,
            top_k: int = 3,
            parent_concept: str = None,
            confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Classifie un document à un niveau spécifique de la hiérarchie.

        Args:
            document_embedding: Embedding du document
            level: Niveau hiérarchique (1, 2, 3, ...)
            top_k: Nombre maximum de résultats
            parent_concept: URI du concept parent (pour filtrer les résultats)
            confidence_threshold: Seuil minimal de confiance

        Returns:
            Liste des concepts les plus probables à ce niveau
        """
        # Vérifier que le niveau existe
        if level not in self.level_networks:
            return []

        # Récupérer le réseau pour ce niveau
        network = self.level_networks[level]

        # Utiliser la recherche progressive si en mode multiscale
        if self.multiscale_mode:
            closest_patterns = network.get_closest_patterns_progressive(
                document_embedding,
                top_k=top_k * 2,  # Récupérer plus de candidats
                threshold=confidence_threshold
            )
        else:
            # Mode classique
            evaluation = network.evaluate_query(document_embedding)
            closest_patterns = evaluation["closest_patterns"]

        # Filtrer les résultats
        results = []

        for pattern in closest_patterns:
            if "label" not in pattern:
                continue

            concept_uri = pattern["label"]
            confidence = pattern["confidence"]

            # Vérifier si ce concept est valide à ce niveau
            if concept_uri not in self.concept_to_level or self.concept_to_level[concept_uri] != level:
                continue

            # Si un parent est spécifié, vérifier que ce concept est bien un enfant
            if parent_concept:
                is_child = False

                # Vérifier dans notre hiérarchie calculée
                if concept_uri in self.concept_hierarchy.get(parent_concept, []):
                    is_child = True

                # Vérifier dans la structure de l'ontologie
                if not is_child:
                    concept = self.ontology_manager.concepts.get(concept_uri)
                    if concept and any(parent.uri == parent_concept for parent in concept.parents):
                        is_child = True

                if not is_child:
                    continue

            # Vérifier le seuil de confiance
            if confidence < confidence_threshold:
                continue

            # Récupérer le concept pour son label
            concept = self.ontology_manager.concepts.get(concept_uri)
            label = concept.label if concept else concept_uri.split('/')[-1].split('#')[-1]

            results.append({
                "concept_uri": concept_uri,
                "label": label,
                "confidence": confidence,
                "similarity": pattern["similarity"]
            })

        # Trier par confiance décroissante et limiter au top_k
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k]

    async def _classify_recursive(
            self,
            document_embedding: np.ndarray,
            parent_concept_uri: str,
            parent_result: Dict[str, Any],
            level: int,
            top_k: int = 3,
            threshold: float = 0.5
    ):
        """
        Classification récursive à travers les niveaux hiérarchiques.

        Args:
            document_embedding: Embedding du document
            parent_concept_uri: URI du concept parent actuel
            parent_result: Dictionnaire du résultat parent à compléter
            level: Niveau actuel
            top_k: Nombre maximum de résultats par niveau
            threshold: Seuil minimal de confiance
        """
        # Vérifier si ce niveau existe
        if level > self.max_level:
            return

        # Vérifier si le parent a des enfants
        children = self.concept_hierarchy.get(parent_concept_uri, [])
        if not children:
            # Vérifier dans l'ontologie via les relations directes
            parent_concept = self.ontology_manager.concepts.get(parent_concept_uri)
            if parent_concept and parent_concept.children:
                children = [child.uri for child in parent_concept.children]

            if not children:
                return

        # Classifier à ce niveau sous ce parent
        level_results = await self._classify_at_level(
            document_embedding,
            level,
            top_k,
            parent_concept_uri,
            threshold
        )

        # Ajouter les résultats à la structure parent
        for result in level_results:
            concept_uri = result["concept_uri"]
            confidence = result["confidence"]
            label = result["label"]

            # Créer la structure pour ce sous-concept
            sub_result = {
                "concept_uri": concept_uri,
                "label": label,
                "confidence": confidence,
                "hierarchy": parent_result["hierarchy"] + [concept_uri],
                "sub_concepts": []
            }

            # Récursivement explorer les niveaux inférieurs
            await self._classify_recursive(
                document_embedding,
                concept_uri,
                sub_result,
                level + 1,
                top_k,
                threshold
            )

            # Ajouter au parent
            parent_result["sub_concepts"].append(sub_result)

    def get_all_subconcepts(self, concept_uri: str) -> List[str]:
        """
        Récupère récursivement tous les sous-concepts d'un concept.

        Args:
            concept_uri: URI du concept

        Returns:
            Liste des URIs des sous-concepts
        """
        # Set pour éviter les doublons
        all_subconcepts = set()

        # Fonction récursive pour explorer la hiérarchie
        def collect_subconcepts(uri, visited=None):
            if visited is None:
                visited = set()

            # Éviter les cycles
            if uri in visited:
                return

            visited.add(uri)

            # Ajouter les sous-concepts directs
            if uri in self.concept_hierarchy:
                children = self.concept_hierarchy[uri]
                for child in children:
                    all_subconcepts.add(child)
                    collect_subconcepts(child, visited)

            # Vérifier également les enfants dans l'ontologie
            concept = self.ontology_manager.concepts.get(uri)
            if concept and hasattr(concept, 'children'):
                for child in concept.children:
                    child_uri = child.uri
                    all_subconcepts.add(child_uri)
                    collect_subconcepts(child_uri, visited)

        # Démarrer la collecte
        collect_subconcepts(concept_uri)

        # Convertir en liste et retourner
        return list(all_subconcepts)

    def old_get_all_subconcepts(self, concept_uri: str) -> List[str]:
        """
        Récupère récursivement tous les sous-concepts d'un concept.

        Args:
            concept_uri: URI du concept

        Returns:
            Liste des URIs des sous-concepts
        """
        if concept_uri not in self.concept_hierarchy:
            return []

        subconcepts = []

        # Ajouter les sous-concepts directs
        subconcepts.extend(self.concept_hierarchy[concept_uri])

        # Ajouter récursivement les sous-concepts des sous-concepts
        for subconcept_uri in self.concept_hierarchy[concept_uri]:
            subconcepts.extend(self.get_all_subconcepts(subconcept_uri))

        return list(set(subconcepts))  # Éliminer les doublons

    async def _generate_concept_embedding(self, concept_uri: str) -> np.ndarray:
        """Génère un embedding pour un concept avec gestion d'erreurs robuste."""
        try:
            # Extraire le concept
            concept = self.ontology_manager.concepts.get(concept_uri)
            if not concept:
                raise ValueError(f"Concept non trouvé: {concept_uri}")

            # Récupérer le label - avec gestion de None
            label = getattr(concept, 'label', None) or self._extract_concept_label_safe(concept_uri)

            # Récupérer la description - avec gestion de None
            description = getattr(concept, 'description', None) or self._extract_concept_description_safe(concept_uri)

            # Construire un texte significatif pour l'embedding
            concept_text = f"{label}"

            if description:
                concept_text += f": {description}"

            # Récupérer les parents avec vérifications
            parent_labels = []
            if hasattr(concept, 'parents') and concept.parents:
                for parent in concept.parents:
                    # Vérifier que parent.uri existe
                    if hasattr(parent, 'uri') and parent.uri:
                        parent_label = getattr(parent, 'label', None) or self._extract_concept_label_safe(parent.uri)
                        if parent_label:
                            parent_labels.append(parent_label)

            if parent_labels:
                concept_text += f" (type de: {', '.join(parent_labels)})"

            # IMPORTANT: Afficher ce qui va être utilisé pour l'embedding
            print(f"Texte pour embedding de '{label}': {concept_text[:150]}...")

            # Générer l'embedding
            embeddings = await self.rag_engine.embedding_manager.provider.generate_embeddings([concept_text])
            embedding = embeddings[0]

            # Normaliser avec vérification
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            print(f"❌ Erreur lors de la génération de l'embedding pour {concept_uri}: {e}")
            # Créer un embedding aléatoire comme fallback
            # TODO bof !!!
            random_embedding = np.random.randn(3072)  # Taille commune pour OpenAI embeddings
            return random_embedding / np.linalg.norm(random_embedding)

    def _extract_concept_description_safe(self, uri: str) -> str:
        """Extrait la description d'un concept de façon sécurisée."""
        try:
            uri_ref = rdflib.URIRef(uri)

            # EMMO elucidation (description spécifique à EMMO)
            try:
                emmo_elucidation = rdflib.URIRef("https://w3id.org/emmo#EMMO_967080e5_2f42_4eb2_a3a9_c58143e835f9")
                for _, _, desc in self.ontology_manager.graph.triples((uri_ref, emmo_elucidation, None)):
                    if desc:
                        return str(desc)
            except Exception:
                pass

            # Commentaire standard
            try:
                for _, _, desc in self.ontology_manager.graph.triples((uri_ref, rdflib.RDFS.comment, None)):
                    if desc:
                        return str(desc)
            except Exception:
                pass

            # SKOS definition
            try:
                skos_definition = rdflib.URIRef("http://www.w3.org/2004/02/skos/core#definition")
                for _, _, desc in self.ontology_manager.graph.triples((uri_ref, skos_definition, None)):
                    if desc:
                        return str(desc)
            except Exception:
                pass

            # Réutiliser la méthode existante comme fallback
            return self._get_concept_description(uri)

        except Exception:
            # Fallback ultime
            return ""

    def _extract_concept_label_safe(self, uri: str) -> str:
        """Extrait le label d'un concept de façon sécurisée."""
        try:
            uri_ref = rdflib.URIRef(uri)

            # Essayer skos:prefLabel
            try:
                skos_prefLabel = rdflib.URIRef("http://www.w3.org/2004/02/skos/core#prefLabel")
                for _, _, value in self.ontology_manager.graph.triples((uri_ref, skos_prefLabel, None)):
                    if value:
                        return str(value)
            except Exception:
                pass

            # Essayer rdfs:label
            try:
                for _, _, value in self.ontology_manager.graph.triples((uri_ref, rdflib.RDFS.label, None)):
                    if value:
                        return str(value)
            except Exception:
                pass

            # Extraire de l'URI
            if '#' in uri:
                return uri.split('#')[-1]
            return uri.split('/')[-1]

        except Exception:
            # Fallback ultime
            return uri.split('/')[-1] if '/' in uri else uri

    def _get_concept_description(self, uri: str) -> str:
        """Extrait la description d'un concept."""
        # Vérifier si le concept a déjà une description
        concept = self.ontology_manager.concepts.get(uri)
        if concept and hasattr(concept, 'description') and concept.description:
            return concept.description

        uri_ref = rdflib.URIRef(uri)

        # EMMO elucidation (description spécifique à EMMO)
        emmo_elucidation = rdflib.URIRef("https://w3id.org/emmo#EMMO_967080e5_2f42_4eb2_a3a9_c58143e835f9")
        for _, _, desc in self.ontology_manager.graph.triples((uri_ref, emmo_elucidation, None)):
            return str(desc)

        # Commentaire standard
        for _, _, desc in self.ontology_manager.graph.triples((uri_ref, rdflib.RDFS.comment, None)):
            return str(desc)

        # SKOS definition
        for _, _, desc in self.ontology_manager.graph.triples(
                (uri_ref, rdflib.URIRef("http://www.w3.org/2004/02/skos/core#definition"), None)):
            return str(desc)

        return ""

    def _get_concept_symbols(self, uri: str) -> List[str]:
        """Extrait les symboles d'un concept (utile pour les unités EMMO)."""
        symbols = []
        uri_ref = rdflib.URIRef(uri)

        # EMMO hasSymbolData
        emmo_symbol = rdflib.URIRef("https://w3id.org/emmo#EMMO_33ae2d07_5526_4555_a0b4_8f4c031b5652")
        for _, _, symbol in self.ontology_manager.graph.triples((uri_ref, emmo_symbol, None)):
            symbols.append(str(symbol))

        # EMMO hasConventionalSymbol
        emmo_conv_symbol = rdflib.URIRef("https://w3id.org/emmo#EMMO_7f1dec83_d85e_4e1b_b7bd_c9442d4f5a64")
        for _, _, symbol in self.ontology_manager.graph.triples((uri_ref, emmo_conv_symbol, None)):
            symbols.append(str(symbol))

        return symbols

    def _calculate_concept_level(self, concept_uri: str, visited=None) -> int:
        """
        Calcule le niveau hiérarchique d'un concept.
        Gère les cycles et l'héritage multiple en prenant le chemin le plus court.
        """
        if visited is None:
            visited = set()

        if concept_uri in visited:
            # Cycle détecté, considérer comme niveau 1
            return 1

        visited.add(concept_uri)

        # Récupérer le concept
        concept = self.ontology_manager.concepts.get(concept_uri)
        if not concept:
            return 1

        # Si pas de parents, c'est un concept de niveau 1
        if not concept.parents:
            return 1

        # En cas d'héritage multiple, prendre le chemin le plus court vers la racine
        parent_levels = []
        for parent in concept.parents:
            parent_level = self._calculate_concept_level(parent.uri, visited.copy())
            parent_levels.append(parent_level)

        # Niveau = niveau du parent + 1 (en prenant le plus court chemin)
        return min(parent_levels) + 1 if parent_levels else 1

    async def add_concept_to_hierarchy(self, concept_uri: str) -> bool:
        """Ajoute un concept à la hiérarchie en générant son embedding."""
        # Vérifier que le concept existe
        if concept_uri not in self.ontology_manager.concepts:
            print(f"⚠️ Concept '{concept_uri}' non trouvé dans l'ontologie")
            return False

        # 1. Déterminer le niveau du concept
        concept_level = self._calculate_concept_level(concept_uri)

        # Mettre à jour la hiérarchie
        self.concept_to_level[concept_uri] = concept_level

        if concept_level not in self.level_to_concepts:
            self.level_to_concepts[concept_level] = []

        if concept_uri not in self.level_to_concepts[concept_level]:
            self.level_to_concepts[concept_level].append(concept_uri)

        self.max_level = max(self.max_level, concept_level)

        # 2. Générer l'embedding du concept
        try:
            concept_embedding = await self._generate_concept_embedding(concept_uri)
        except Exception as e:
            print(f"⚠️ Erreur lors de la génération de l'embedding pour '{concept_uri}': {e}")
            return False

        # Stocker l'embedding du concept
        self.concept_embeddings[concept_uri] = concept_embedding

        # 3. S'assurer que le réseau pour ce niveau existe
        if concept_level not in self.level_networks:
            level_dir = os.path.join(self.storage_dir, f"level_{concept_level}")
            os.makedirs(level_dir, exist_ok=True)

            self.level_networks[concept_level] = ModernHopfieldNetwork(
                beta=22.0,
                normalize_patterns=True,
                storage_dir=level_dir,
                multiscale_mode=self.multiscale_mode,
                wavelet_config=self.wavelet_config
            )

        # 4. Ajouter l'embedding conceptuel au réseau de son niveau
        network = self.level_networks[concept_level]
        network.store_patterns(
            np.array([concept_embedding]),
            [concept_uri]
        )

        # Sauvegarder selon le mode
        if self.multiscale_mode:
            network.save_multiscale()
        else:
            network.save()

        # 5. Sauvegarder tous les embeddings de concepts
        await self._save_concept_embeddings()

        mode_str = "multiscale" if self.multiscale_mode else "classique"
        print(f"✓ Concept '{concept_uri}' ajouté au niveau {concept_level} (mode {mode_str})")

        return True

    async def classify_document(
            self,
            document_embedding: np.ndarray,
            top_k: int = 3,
            threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Classifie un document en cascade à travers les niveaux hiérarchiques de concepts."""
        if not self.level_networks or 1 not in self.level_networks:
            return []

        # Normaliser l'embedding du document
        if np.linalg.norm(document_embedding) > 0:
            document_embedding = document_embedding / np.linalg.norm(document_embedding)

        # Résultats finaux
        final_results = []

        # Commencer par le niveau 1
        level1_results = await self._classify_at_level(
            document_embedding, 1, top_k, None, threshold
        )

        # Pour chaque concept de niveau 1 détecté
        for concept_result in level1_results:
            concept_uri = concept_result["concept_uri"]
            concept_confidence = concept_result["confidence"]

            # Structure pour ce résultat
            hierarchical_result = {
                "concept_uri": concept_uri,
                "label": self.ontology_manager.concepts[concept_uri].label,
                "confidence": concept_confidence,
                "hierarchy": [concept_uri],
                "sub_concepts": []
            }

            # Récursivement explorer les sous-niveaux
            await self._classify_recursive(
                document_embedding,
                concept_uri,
                hierarchical_result,
                level=2,
                top_k=top_k,
                threshold=threshold
            )

            final_results.append(hierarchical_result)

        return final_results

    async def build_concept_embeddings(
            self,
            max_concepts_per_level: int = None,
            max_depth: int = None,
            min_concepts_per_level: int = 1
    ) -> int:
        """
        Construit automatiquement les embeddings pour les concepts de l'ontologie.

        Args:
            max_concepts_per_level: Nombre maximum de concepts à traiter par niveau
            max_depth: Profondeur maximale à explorer (None = tous les niveaux)
            min_concepts_per_level: Nombre minimum de concepts à traiter par niveau

        Returns:
            Nombre total de concepts traités
        """
        # Vérifier que la hiérarchie est construite
        if not self.concept_to_level:
            await self._build_concept_hierarchy()

        # Calculer la profondeur maximale à explorer
        if max_depth is None or max_depth > self.max_level:
            max_depth = self.max_level

        total_processed = 0

        # Traiter niveau par niveau pour garantir que les parents sont traités avant leurs enfants
        for level in range(1, max_depth + 1):
            if level not in self.level_to_concepts:
                continue

            concepts_at_level = self.level_to_concepts[level]

            # Limiter le nombre de concepts si demandé
            if max_concepts_per_level and len(concepts_at_level) > max_concepts_per_level:
                # Sélectionner les concepts les plus "importants" à ce niveau
                # Ici, on pourrait implémenter différentes stratégies de sélection
                # Par défaut, on prend simplement un échantillon représentatif
                # TODO A Voir si c'est pertinant
                import random
                concepts_to_process = random.sample(concepts_at_level, max_concepts_per_level)
            else:
                concepts_to_process = concepts_at_level

            # Assurer un nombre minimum de concepts par niveau
            if len(concepts_to_process) < min_concepts_per_level:
                continue

            print(f"Traitement du niveau {level}: {len(concepts_to_process)}/{len(concepts_at_level)} concepts")

            # Traiter chaque concept à ce niveau
            for i, concept_uri in enumerate(concepts_to_process):
                success = await self.add_concept_to_hierarchy(concept_uri)
                if success:
                    total_processed += 1
                    label = self._get_concept_label(concept_uri)

                    # NOUVEAU: Montrer plus de détails pour les premiers concepts et périodiquement
                    if i < 3 or i % 10 == 0:
                        description = self._get_concept_description(concept_uri)
                        print(f"  ✓ [{i + 1}/{len(concepts_to_process)}] Concept ajouté: {label}")
                        print(f"    Description: {description[:100]}..." if len(
                            description) > 100 else f"    Description: {description}")
                    else:
                        print(f"  ✓ Concept ajouté: {label}")

        print(f"✓ Construction automatique terminée: {total_processed} concepts traités")
        return total_processed

    async def auto_detect_concepts(
            self,
            query_embedding: np.ndarray,
            min_confidence: float = 0.65,
            max_concepts: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Détecte automatiquement les concepts pertinents pour une requête.
        CORRECTION: S'assurer qu'aucune coroutine ne reste non-awaited.
        """
        try:
            # Normaliser l'embedding
            if np.linalg.norm(query_embedding) > 0:
                query_embedding = query_embedding / np.linalg.norm(query_embedding)

            # Préparer la liste pour les résultats
            concept_matches = []

            # Commencer par le niveau le plus profond et remonter
            for level in range(self.max_level, 0, -1):
                if level not in self.level_networks:
                    continue

                # ✅ CORRECTION: Vérification que le réseau n'est pas une coroutine
                network = self.level_networks[level]
                if asyncio.iscoroutine(network):
                    self.logger.warning(f"Réseau niveau {level} est une coroutine non-awaited")
                    continue

                # Obtenir le réseau Hopfield pour ce niveau
                try:
                    # ✅ CORRECTION: S'assurer que evaluate_query est synchrone
                    evaluation = network.evaluate_query(query_embedding)

                    # Vérifier que evaluation n'est pas une coroutine
                    if asyncio.iscoroutine(evaluation):
                        self.logger.warning(f"Evaluation niveau {level} retourne une coroutine")
                        evaluation = await evaluation  # Await si nécessaire

                except Exception as e:
                    self.logger.warning(f"Erreur évaluation niveau {level}: {e}")
                    continue

                # Récupérer les concepts les plus similaires
                closest_concepts = evaluation.get("closest_patterns", [])

                # ✅ CORRECTION: Nettoyer les résultats de toute coroutine
                valid_concepts = []
                for concept in closest_concepts:
                    # Vérifier que concept n'est pas une coroutine
                    if asyncio.iscoroutine(concept):
                        self.logger.warning("Concept dans closest_patterns est une coroutine")
                        continue

                    if not isinstance(concept, dict) or "label" not in concept:
                        continue

                    confidence = concept.get("confidence", 0.0)
                    if confidence < min_confidence:
                        continue

                    concept_uri = concept["label"]

                    # ✅ CORRECTION: Vérifier que l'URI n'est pas une coroutine
                    if asyncio.iscoroutine(concept_uri):
                        self.logger.warning("Concept URI est une coroutine")
                        continue

                    if concept_uri in self.ontology_manager.concepts:
                        concept_obj = self.ontology_manager.concepts[concept_uri]

                        # ✅ CORRECTION: S'assurer que les propriétés ne sont pas des coroutines
                        label = concept_obj.label if hasattr(concept_obj, 'label') and concept_obj.label else \
                        concept_uri.split('#')[-1]

                        if asyncio.iscoroutine(label):
                            label = str(concept_uri.split('#')[-1])

                        valid_concepts.append({
                            "concept_uri": str(concept_uri),  # Forcer en string
                            "label": str(label),  # Forcer en string
                            "confidence": float(confidence),  # Forcer en float
                            "level": int(level)  # Forcer en int
                        })

                # Si on a trouvé suffisamment de concepts, on peut s'arrêter
                if len(valid_concepts) >= max_concepts:
                    concept_matches.extend(valid_concepts[:max_concepts])
                    break

                concept_matches.extend(valid_concepts)

                if len(concept_matches) >= max_concepts:
                    break

            # ✅ CORRECTION: Nettoyage final pour s'assurer qu'aucune coroutine ne subsiste
            cleaned_matches = []
            for match in concept_matches[:max_concepts]:
                if not asyncio.iscoroutine(match) and isinstance(match, dict):
                    # Double vérification que tous les champs sont sérialisables
                    cleaned_match = {}
                    for key, value in match.items():
                        if not asyncio.iscoroutine(value):
                            if isinstance(value, (str, int, float, bool)):
                                cleaned_match[key] = value
                            else:
                                cleaned_match[key] = str(value)  # Forcer en string si type complexe

                    cleaned_matches.append(cleaned_match)

            return cleaned_matches

        except Exception as e:
            self.logger.error(f"Erreur dans auto_detect_concepts: {e}", exc_info=True)
            return []  # Retourner une liste vide plutôt qu'une exception

    def _clean_result_for_serialization(self, obj):
        """
        Nettoie récursivement un objet de toute coroutine pour la sérialisation.
        """
        import asyncio

        if asyncio.iscoroutine(obj):
            self.logger.warning("Coroutine détectée et supprimée")
            return None

        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                if not asyncio.iscoroutine(key) and not asyncio.iscoroutine(value):
                    cleaned_value = self._clean_result_for_serialization(value)
                    if cleaned_value is not None:
                        cleaned[str(key)] = cleaned_value
            return cleaned

        elif isinstance(obj, (list, tuple)):
            cleaned = []
            for item in obj:
                cleaned_item = self._clean_result_for_serialization(item)
                if cleaned_item is not None:
                    cleaned.append(cleaned_item)
            return cleaned if isinstance(obj, list) else tuple(cleaned)

        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj

        else:
            # Pour les types complexes, convertir en string
            try:
                return str(obj)
            except:
                return None

    async def old_auto_detect_concepts(
            self,
            query_embedding: np.ndarray,
            min_confidence: float = 0.65,
            max_concepts: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Détecte automatiquement les concepts pertinents pour une requête.
        Utilise une approche ascendante en commençant par les concepts spécifiques.

        Args:
            query_embedding: Embedding de la requête
            min_confidence: Seuil minimal de confiance
            max_concepts: Nombre maximum de concepts à retourner

        Returns:
            Liste des concepts pertinents avec leur score
        """
        # Normaliser l'embedding
        if np.linalg.norm(query_embedding) > 0:
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Préparer la liste pour les résultats
        concept_matches = []

        # Commencer par le niveau le plus profond et remonter
        for level in range(self.max_level, 0, -1):
            if level not in self.level_networks:
                continue

            # Obtenir le réseau Hopfield pour ce niveau
            network = self.level_networks[level]

            # Évaluer l'embedding de la requête avec le réseau
            evaluation = network.evaluate_query(query_embedding)

            # Récupérer les concepts les plus similaires
            closest_concepts = evaluation["closest_patterns"]

            # Filtrer selon le seuil de confiance
            valid_concepts = []
            for concept in closest_concepts:
                if "label" in concept and concept["confidence"] >= min_confidence:
                    # Vérifier que c'est bien un concept (URI dans l'ontologie)
                    concept_uri = concept["label"]
                    if concept_uri in self.ontology_manager.concepts:
                        concept_obj = self.ontology_manager.concepts[concept_uri]
                        valid_concepts.append({
                            "concept_uri": concept_uri,
                            "label": concept_obj.label if concept_obj.label else concept_uri.split('#')[-1],
                            "confidence": concept["confidence"],
                            "level": level
                        })

            # Si on a trouvé suffisamment de concepts, on peut s'arrêter
            if len(valid_concepts) >= max_concepts:
                concept_matches.extend(valid_concepts[:max_concepts])
                break

            # Sinon, on ajoute ce qu'on a trouvé et on continue à monter dans la hiérarchie
            concept_matches.extend(valid_concepts)

            # Si on a déjà trouvé assez de concepts, on s'arrête
            if len(concept_matches) >= max_concepts:
                break

        # Trier par confiance et limiter au nombre maximum
        concept_matches.sort(key=lambda x: x["confidence"], reverse=True)
        return concept_matches[:max_concepts]

    async def classify_documents_batch(self, document_embeddings: Dict[str, np.ndarray]) -> Dict[
        str, List[Dict[str, Any]]]:
        """Classifie plusieurs documents en parallèle"""
        # Créer des tâches pour chaque document
        tasks = []
        doc_ids = []

        for doc_id, embedding in document_embeddings.items():
            task = self.classify_document(embedding)
            tasks.append(task)
            doc_ids.append(doc_id)

        # Exécuter en parallèle
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Organiser les résultats
        batch_results = {}
        for doc_id, result in zip(doc_ids, results):
            if isinstance(result, Exception):
                print(f"Erreur classification pour {doc_id}: {result}")
                batch_results[doc_id] = []
            else:
                batch_results[doc_id] = result

        return batch_results

    async def classify_document_pure(self, document_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """
        Classifie un document en comparant directement son embedding
        avec les embeddings des concepts, SANS recherche dans la base.
        """
        if not self.concept_embeddings:
            print("⚠️ Aucun embedding de concept disponible")
            return []

        # Normaliser l'embedding du document
        doc_norm = document_embedding / (np.linalg.norm(document_embedding) + 1e-8)

        # Calculer les similarités avec TOUS les concepts
        concept_scores = []

        for concept_uri, concept_embedding in self.concept_embeddings.items():
            # Calculer la similarité cosinus
            similarity = float(np.dot(doc_norm, concept_embedding))

            # Récupérer les infos du concept
            concept = self.ontology_manager.concepts.get(concept_uri, {})
            concept_label = concept.label if hasattr(concept, 'label') else concept_uri.split('#')[-1]

            concept_scores.append({
                'concept_uri': concept_uri,
                'label': concept_label,
                'confidence': similarity
            })

        # Trier par score décroissant
        concept_scores.sort(key=lambda x: x['confidence'], reverse=True)

        # Construire la hiérarchie pour les meilleurs concepts
        threshold = 0.5  # Seuil de confiance minimal
        top_concepts = [c for c in concept_scores if c['confidence'] >= threshold][:10]

        return self._build_concept_hierarchy(top_concepts)

    async def auto_detect_concepts_batch(self, embeddings: Dict[str, np.ndarray], min_confidence: float = 0.6) -> Dict[
        str, List[Dict[str, Any]]]:
        """Détecte les concepts pour plusieurs embeddings en parallèle"""
        tasks = []
        ids = []

        for embed_id, embedding in embeddings.items():
            task = self.auto_detect_concepts(embedding, min_confidence)
            tasks.append(task)
            ids.append(embed_id)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        batch_results = {}
        for embed_id, result in zip(ids, results):
            if isinstance(result, Exception):
                print(f"Erreur détection concepts pour {embed_id}: {result}")
                batch_results[embed_id] = []
            else:
                batch_results[embed_id] = result

        return batch_results