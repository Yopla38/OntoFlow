"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# ontology/hopfield_network.py
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from sklearn.preprocessing import normalize
import torch
import torch.nn.functional as F


class ModernHopfieldNetwork:
    """
    Implémentation d'un réseau de Hopfield moderne selon l'article
    "Hopfield Networks is All You Need" (Ramsauer et al., 2020).
    """

    def __init__(
            self,
            beta: float = 22.0,  # Paramètre de température inverse
            normalize_patterns: bool = True,  # Normaliser les patterns
            storage_dir: str = None,  # Répertoire pour sauvegarder le réseau
            multiscale_mode: bool = False,
            wavelet_config: Dict[str, Any] = None
    ):
        """
        Initialise un réseau de Hopfield moderne.

        Args:
            beta: Paramètre de température inverse (plus élevé = plus précis)
            normalize_patterns: Si True, normalise les patterns (recommandé)
            storage_dir: Répertoire de stockage pour la persistance
            multiscale_mode: Si True, active le mode multiscale avec ondelettes
            wavelet_config: Configuration des ondelettes {wavelet: 'db3', levels: 5}
        """
        self.beta = beta
        self.normalize_patterns = normalize_patterns
        self.storage_dir = storage_dir
        self.multiscale_mode = multiscale_mode

        # Pour le mode GPU si disponible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configuration des ondelettes par défaut
        if wavelet_config is None:
            wavelet_config = {'wavelet': 'coif3', 'levels': 3}
        self.wavelet_config = wavelet_config

        # Patterns stockés (mode classique)
        self.stored_patterns = []
        self.pattern_labels = []  # Étiquettes pour les patterns (ex: IDs document)

        # Stockage multiscale
        if self.multiscale_mode:
            # Import conditionnel pour éviter les erreurs si pywt n'est pas installé
            """
            try:
                import pywt
                from concept_hopfield import WaveletDecomposer
                self.wavelet_decomposer = WaveletDecomposer(**wavelet_config)
                # Structure : {level: {patterns: tensor, labels: list}}
                self.multiscale_patterns = {}
                for level in range(wavelet_config.get('levels', 3) + 1):
                    self.multiscale_patterns[level] = {
                        'patterns': torch.tensor([], dtype=torch.float32, device=self.device),
                        'labels': []
                    }
            except ImportError:
                print("⚠️ PyWavelets non disponible, désactivation du mode multiscale")
                self.multiscale_mode = False
            """
            import pywt
            from ontology.concept_hopfield import WaveletDecomposer
            self.wavelet_decomposer = WaveletDecomposer(**wavelet_config)
            # Structure : {level: {patterns: tensor, labels: list}}
            self.multiscale_patterns = {}
            for level in range(wavelet_config.get('levels', 3) + 1):
                self.multiscale_patterns[level] = {
                    'patterns': torch.tensor([], dtype=torch.float32, device=self.device),
                    'labels': []
                }

        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)

    """ ------------------------  WAVELETS ------------------------------ """

    def store_patterns_multiscale(self, patterns: np.ndarray, labels: Optional[List[str]] = None) -> None:
        """Stocke des patterns en mode multiscale avec décomposition en ondelettes."""
        if not self.multiscale_mode:
            # Fallback vers la méthode classique
            return self.store_patterns(patterns, labels)

        if patterns.size == 0:
            return

        # Assurer que les patterns sont en 2D
        if patterns.ndim == 1:
            patterns = patterns.reshape(1, -1)

        # Normaliser si demandé
        if self.normalize_patterns:
            patterns = normalize(patterns, axis=1, norm='l2')

        # Décomposer chaque pattern et l'ajouter à tous les niveaux
        for i, pattern in enumerate(patterns):
            pattern_id = labels[i] if labels and i < len(labels) else f"pattern_{i}"

            # Décomposer en ondelettes
            decompositions = self.wavelet_decomposer.decompose_embedding(pattern, pattern_id)

            # Stocker à chaque niveau
            for level, decomposed_pattern in decompositions.items():
                if level in self.multiscale_patterns:
                    # Convertir en tenseur PyTorch
                    pattern_tensor = torch.tensor(decomposed_pattern, dtype=torch.float32,
                                                  device=self.device).unsqueeze(0)

                    # Ajouter aux patterns existants
                    if self.multiscale_patterns[level]['patterns'].size(0) == 0:
                        self.multiscale_patterns[level]['patterns'] = pattern_tensor
                    else:
                        self.multiscale_patterns[level]['patterns'] = torch.cat([
                            self.multiscale_patterns[level]['patterns'],
                            pattern_tensor
                        ], dim=0)

                    # Ajouter le label
                    self.multiscale_patterns[level]['labels'].append(pattern_id)

    def get_closest_patterns_progressive(
            self,
            query: np.ndarray,
            top_k: int = 1,
            start_level: int = None,
            threshold: float = 0.7,
            max_candidates_per_level: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Recherche progressive multiscale inspirée de WA_RAG.

        Args:
            query: Pattern de requête
            top_k: Nombre de résultats finaux
            start_level: Niveau de départ (None = niveau max)
            threshold: Seuil de confiance pour arrêt anticipé
            max_candidates_per_level: Nombre max de candidats à propager
        """
        if not self.multiscale_mode:
            # Fallback vers la méthode classique
            return self.get_closest_patterns(query, top_k)

        # Préparer la requête
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if self.normalize_patterns:
            query = normalize(query, axis=1, norm='l2')

        # Décomposer la requête en ondelettes
        query_decompositions = self.wavelet_decomposer.decompose_embedding(query.flatten())

        # Déterminer le niveau de départ
        if start_level is None:
            start_level = max(self.multiscale_patterns.keys())

        # Commencer avec tous les patterns comme candidats
        available_patterns = set()
        for level_data in self.multiscale_patterns.values():
            if len(level_data['labels']) > 0:
                available_patterns.update(level_data['labels'])

        candidates = list(available_patterns)
        candidate_scores = {}

        # Recherche progressive du niveau grossier au niveau fin
        for level in range(start_level, -1, -1):
            if level not in self.multiscale_patterns:
                continue

            level_data = self.multiscale_patterns[level]
            if level_data['patterns'].size(0) == 0:
                continue

            # Requête pour ce niveau
            query_level = torch.tensor(query_decompositions[level], dtype=torch.float32, device=self.device)

            # Filtrer les patterns candidats pour ce niveau
            candidate_indices = []
            candidate_labels = []

            for i, label in enumerate(level_data['labels']):
                if label in candidates:
                    candidate_indices.append(i)
                    candidate_labels.append(label)

            if not candidate_indices:
                continue

            # Extraire les patterns candidats
            candidate_patterns = level_data['patterns'][candidate_indices]

            # Calculer les similarités
            if self.normalize_patterns:
                query_norm = F.normalize(query_level.unsqueeze(0), p=2, dim=1)
                candidate_patterns_norm = F.normalize(candidate_patterns, p=2, dim=1)
                similarities = torch.matmul(query_norm, candidate_patterns_norm.t()).squeeze()
            else:
                similarities = torch.matmul(query_level.unsqueeze(0), candidate_patterns.t()).squeeze()

            # Si un seul candidat
            if similarities.dim() == 0:
                similarities = similarities.unsqueeze(0)

            # Mettre à jour les scores
            for i, label in enumerate(candidate_labels):
                candidate_scores[label] = float(similarities[i])

            # Vérifier l'arrêt anticipé
            if len(candidate_scores) > 0:
                max_confidence = max(candidate_scores.values())
                if max_confidence >= threshold and level <= 1:
                    break

            # Filtrer les candidats pour le niveau suivant
            if level > 0:
                # Garder les meilleurs candidats
                num_to_keep = min(max_candidates_per_level, len(candidate_scores))
                sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
                candidates = [label for label, _ in sorted_candidates[:num_to_keep]]

        # Créer les résultats finaux
        results = []
        sorted_results = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

        for i, (label, similarity) in enumerate(sorted_results[:top_k]):
            # Récupérer le pattern original (niveau 0)
            if 0 in self.multiscale_patterns and label in self.multiscale_patterns[0]['labels']:
                pattern_idx = self.multiscale_patterns[0]['labels'].index(label)
                pattern = self.multiscale_patterns[0]['patterns'][pattern_idx].cpu().numpy()
            else:
                pattern = None

            confidence = float((similarity + 1) / 2)  # Normaliser entre 0 et 1

            result = {
                "pattern": pattern,
                "similarity": float(similarity),
                "confidence": confidence,
                "rank": i + 1,
                "label": label
            }

            results.append(result)

        return results

    def save_multiscale(self, filename: str = None) -> str:
        """Sauvegarde le réseau multiscale."""
        if not self.multiscale_mode:
            return self.save(filename)

        if not self.storage_dir:
            raise ValueError("Aucun répertoire de stockage défini")

        if filename is None:
            filename = "hopfield_network_multiscale.pkl"

        file_path = os.path.join(self.storage_dir, filename)

        # Préparer les données à sauvegarder
        multiscale_data = {}
        for level, level_data in self.multiscale_patterns.items():
            multiscale_data[level] = {
                'patterns': level_data['patterns'].cpu().numpy(),
                'labels': level_data['labels']
            }

        data = {
            "beta": self.beta,
            "normalize_patterns": self.normalize_patterns,
            "multiscale_mode": self.multiscale_mode,
            "wavelet_config": self.wavelet_config,
            "multiscale_patterns": multiscale_data
        }

        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        return file_path

    def load_multiscale(self, filename: str = None) -> bool:
        """Charge le réseau multiscale."""
        if not self.storage_dir:
            raise ValueError("Aucun répertoire de stockage défini")

        if filename is None:
            filename = "hopfield_network_multiscale.pkl"

        file_path = os.path.join(self.storage_dir, filename)

        if not os.path.exists(file_path):
            return False

        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            self.beta = data["beta"]
            self.normalize_patterns = data["normalize_patterns"]
            self.multiscale_mode = data.get("multiscale_mode", False)
            self.wavelet_config = data.get("wavelet_config", {'wavelet': 'db3', 'levels': 5})

            if self.multiscale_mode:
                # Réinitialiser le décomposeur
                from ontology.concept_hopfield import WaveletDecomposer
                self.wavelet_decomposer = WaveletDecomposer(**self.wavelet_config)

                # Charger les patterns multiscale
                multiscale_data = data["multiscale_patterns"]
                self.multiscale_patterns = {}

                for level, level_data in multiscale_data.items():
                    patterns_np = level_data['patterns']
                    if patterns_np.size > 0:
                        patterns_tensor = torch.tensor(patterns_np, dtype=torch.float32, device=self.device)
                    else:
                        patterns_tensor = torch.tensor([], dtype=torch.float32, device=self.device)

                    self.multiscale_patterns[int(level)] = {
                        'patterns': patterns_tensor,
                        'labels': level_data['labels']
                    }

            return True
        except Exception as e:
            print(f"Erreur lors du chargement du réseau multiscale: {str(e)}")
            return False

    def store_patterns(self, patterns: np.ndarray, labels: Optional[List[str]] = None) -> None:
        """Stocke un ensemble de patterns (mode classique ou multiscale)."""
        if self.multiscale_mode:
            return self.store_patterns_multiscale(patterns, labels)

        # Code original inchangé
        if patterns.size == 0:
            return

        # Assurer que les patterns sont en 2D
        if patterns.ndim == 1:
            patterns = patterns.reshape(1, -1)

        # Normaliser si demandé
        if self.normalize_patterns:
            patterns = normalize(patterns, axis=1, norm='l2')

        # Convertir en tenseur PyTorch
        patterns_tensor = torch.tensor(patterns, dtype=torch.float32, device=self.device)

        # Stocker les patterns
        if not isinstance(self.stored_patterns, torch.Tensor) or self.stored_patterns.size(0) == 0:
            self.stored_patterns = patterns_tensor
        else:
            self.stored_patterns = torch.cat([self.stored_patterns, patterns_tensor], dim=0)

        # Stocker les labels si fournis
        if labels:
            if not self.pattern_labels:
                self.pattern_labels = labels
            else:
                self.pattern_labels.extend(labels)

    def _compute_state_update(self, state: torch.Tensor) -> torch.Tensor:
        """Calcule la mise à jour de l'état selon la dynamique du réseau de Hopfield moderne."""
        # Calculer la matrice d'attention
        state_norm = F.normalize(state, p=2, dim=1) if self.normalize_patterns else state

        if not isinstance(self.stored_patterns, torch.Tensor) or self.stored_patterns.size(0) == 0:
            return state  # Aucun pattern stocké

        # Calcul des similarités
        similarities = torch.matmul(state_norm, self.stored_patterns.t())

        # Appliquer la fonction softmax avec température
        attention_weights = F.softmax(self.beta * similarities, dim=1)

        # Calculer le nouvel état
        new_state = torch.matmul(attention_weights, self.stored_patterns)

        return new_state

    def _compute_energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calcule l'énergie de l'état actuel par rapport aux patterns stockés.
        Des valeurs d'énergie plus basses indiquent une meilleure correspondance.
        """
        state_norm = F.normalize(state, p=2, dim=1) if self.normalize_patterns else state

        if not isinstance(self.stored_patterns, torch.Tensor) or self.stored_patterns.size(0) == 0:
            return torch.tensor([float('inf')], device=self.device)

        # Calcul des similarités
        similarities = torch.matmul(state_norm, self.stored_patterns.t())

        # Calculer le logsumexp
        lse = torch.logsumexp(self.beta * similarities, dim=1)

        # L'énergie est l'opposé du logsumexp
        return -lse / self.beta

    def retrieve(self, query: np.ndarray, iterations: int = 1) -> Tuple[np.ndarray, List[float]]:
        """Récupère le pattern stocké le plus proche du pattern requête."""
        # Assurer que query est 2D
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Normaliser si demandé
        if self.normalize_patterns:
            query = normalize(query, axis=1, norm='l2')

        # Convertir en tenseur PyTorch
        query_tensor = torch.tensor(query, dtype=torch.float32, device=self.device)

        # État initial
        state = query_tensor
        energies = []

        # Itérer la dynamique du réseau
        for _ in range(iterations):
            # Calculer l'énergie avant mise à jour
            energy = self._compute_energy(state).item()
            energies.append(energy)

            # Mettre à jour l'état
            state = self._compute_state_update(state)

        # Convertir le résultat en numpy
        result = state.detach().cpu().numpy()

        return result, energies

    def get_closest_patterns(self, query: np.ndarray, top_k: int = 1) -> List[Dict[str, Any]]:
        """Récupère les k patterns stockés les plus proches du pattern requête."""
        if not isinstance(self.stored_patterns, torch.Tensor) or self.stored_patterns.size(0) == 0:
            return []

        # Préparer la requête
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if self.normalize_patterns:
            query = normalize(query, axis=1, norm='l2')

        query_tensor = torch.tensor(query, dtype=torch.float32, device=self.device)

        # Normaliser si nécessaire
        query_norm = F.normalize(query_tensor, p=2, dim=1) if self.normalize_patterns else query_tensor

        # Calculer les similarités
        similarities = torch.matmul(query_norm, self.stored_patterns.t()).squeeze()

        # Obtenir les indices des top-k patterns les plus similaires
        if similarities.dim() == 0:  # Si un seul pattern est stocké
            top_indices = [0]
            top_similarities = [similarities.item()]
        else:
            top_similarities, top_indices = torch.topk(similarities, min(top_k, similarities.size(0)))
            top_similarities = top_similarities.cpu().numpy()
            top_indices = top_indices.cpu().numpy()

        # Créer la liste des résultats
        results = []
        for i, (idx, similarity) in enumerate(zip(top_indices, top_similarities)):
            pattern = self.stored_patterns[idx].cpu().numpy()

            # Ajuster entre 0 et 1 pour la confiance (similarité cosinus -1 à 1)
            confidence = float((similarity + 1) / 2)

            result = {
                "pattern": pattern,
                "similarity": float(similarity),
                "confidence": confidence,
                "rank": i + 1
            }

            # Ajouter le label si disponible
            if self.pattern_labels and idx < len(self.pattern_labels):
                result["label"] = self.pattern_labels[idx]

            results.append(result)

        return results

    def evaluate_query(self, query: np.ndarray) -> Dict[str, Any]:
        """Évalue une requête en calculant son énergie et sa proximité aux patterns stockés."""
        # Récupérer le pattern le plus proche
        retrieved, energies = self.retrieve(query)

        # Obtenir les patterns les plus proches
        closest_patterns = self.get_closest_patterns(query, top_k=3)

        # Calculer la confiance globale (sigmoid de -énergie)
        min_energy = min(energies) if energies else float('inf')
        confidence = np.exp(-min_energy) / (1 + np.exp(-min_energy))

        result = {
            "retrieved_pattern": retrieved,
            "closest_patterns": closest_patterns,
            "energy": min_energy,
            "confidence": float(confidence)
        }

        return result

    def save(self, filename: str = None) -> str:
        """Sauvegarde le réseau sur disque."""
        if not self.storage_dir:
            raise ValueError("Aucun répertoire de stockage défini")

        if filename is None:
            filename = "hopfield_network.pkl"

        file_path = os.path.join(self.storage_dir, filename)

        # Convertir les tenseurs en numpy pour la sauvegarde
        if isinstance(self.stored_patterns, torch.Tensor):
            stored_patterns_np = self.stored_patterns.cpu().numpy()
        else:
            stored_patterns_np = np.array([])

        data = {
            "beta": self.beta,
            "normalize_patterns": self.normalize_patterns,
            "stored_patterns": stored_patterns_np,
            "pattern_labels": self.pattern_labels
        }

        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        return file_path

    def load(self, filename: str = None) -> bool:
        """Charge le réseau depuis un fichier."""
        if not self.storage_dir:
            raise ValueError("Aucun répertoire de stockage défini")

        if filename is None:
            filename = "hopfield_network.pkl"

        file_path = os.path.join(self.storage_dir, filename)

        if not os.path.exists(file_path):
            return False

        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            self.beta = data["beta"]
            self.normalize_patterns = data["normalize_patterns"]

            # Charger les patterns en numpy puis convertir en tenseurs
            stored_patterns_np = data["stored_patterns"]
            if stored_patterns_np.size > 0:
                self.stored_patterns = torch.tensor(stored_patterns_np, dtype=torch.float32, device=self.device)
            else:
                self.stored_patterns = torch.tensor([], dtype=torch.float32, device=self.device)

            self.pattern_labels = data["pattern_labels"]

            return True
        except Exception as e:
            print(f"Erreur lors du chargement du réseau de Hopfield: {str(e)}")
            return False


class HopfieldClassifier:
    """
    Classifieur basé sur les réseaux de Hopfield modernes pour classifier
    les documents dans une ontologie multi-niveaux.
    """

    def __init__(
            self,
            rag_engine,  # Moteur RAG complet pour accéder aux embeddings et documents
            ontology_manager,
            storage_dir: str = "hopfield_models",
            beta: float = 22.0
    ):
        """
        Initialise le classifieur Hopfield avec intégration au RAG.

        Args:
            rag_engine: Moteur RAG complet
            ontology_manager: Gestionnaire d'ontologie
            storage_dir: Répertoire pour stocker les modèles
            beta: Paramètre de température inverse pour les réseaux
        """
        self.rag_engine = rag_engine
        self.ontology_manager = ontology_manager
        self.storage_dir = storage_dir
        self.beta = beta

        # Créer le répertoire de stockage
        os.makedirs(storage_dir, exist_ok=True)

        # Dictionnaire des réseaux par domaine
        self.domain_networks = {}

        # Dictionnaire des embeddings moyens par domaine
        self.domain_centroids = {}

    async def train_domain(self, domain_name: str, document_embeddings: List[Tuple[str, np.ndarray]]) -> bool:
        """Entraîne un réseau de Hopfield pour un domaine spécifique."""
        if not document_embeddings:
            return False
        print(f"Training domain {domain_name} !")
        # Créer un répertoire pour ce domaine
        domain_dir = os.path.join(self.storage_dir, domain_name)
        os.makedirs(domain_dir, exist_ok=True)

        # Créer un réseau de Hopfield pour ce domaine
        network = ModernHopfieldNetwork(
            beta=self.beta,
            normalize_patterns=True,
            storage_dir=domain_dir
        )

        # Extraire les embeddings et les IDs
        ids = [doc_id for doc_id, _ in document_embeddings]
        embeddings = np.vstack([emb for _, emb in document_embeddings])

        # Stocker les patterns dans le réseau
        network.store_patterns(embeddings, ids)

        # Calculer le centroïde du domaine (moyenne des embeddings)
        domain_centroid = np.mean(embeddings, axis=0)
        self.domain_centroids[domain_name] = domain_centroid

        # Sauvegarder le réseau
        network.save()

        # Sauvegarder le centroïde
        centroid_path = os.path.join(domain_dir, "centroid.npy")
        np.save(centroid_path, domain_centroid)

        # Stocker le réseau en mémoire
        self.domain_networks[domain_name] = network

        # Associer les documents au domaine dans l'ontologie avec confiance maximale
        for doc_id, _ in document_embeddings:
            self.ontology_manager.associate_document_with_domain(doc_id, domain_name, 1.0)

        return True

    def load_domain_network(self, domain_name: str) -> bool:
        """Charge le réseau de Hopfield pour un domaine spécifique."""
        domain_dir = os.path.join(self.storage_dir, domain_name)

        if not os.path.exists(domain_dir):
            return False

        # Charger le réseau
        network = ModernHopfieldNetwork(
            beta=self.beta,
            normalize_patterns=True,
            storage_dir=domain_dir
        )

        if not network.load():
            return False

        self.domain_networks[domain_name] = network

        # Charger le centroïde
        centroid_path = os.path.join(domain_dir, "centroid.npy")
        if os.path.exists(centroid_path):
            self.domain_centroids[domain_name] = np.load(centroid_path)

        return True

    async def initialize(self) -> None:
        """Initialise le classifieur en chargeant tous les réseaux de domaine existants."""
        # Charger les réseaux pour tous les domaines disponibles
        for domain_name in self.ontology_manager.domains:
            self.load_domain_network(domain_name)

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

    async def classify_document(self, document_embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Classifie un document dans les domaines appropriés."""
        results = []

        # Si aucun domaine n'est défini, retourner une liste vide
        if not self.domain_networks:
            return results

        # Première étape: calculer la similarité avec les centroïdes des domaines
        centroid_similarities = {}
        for domain_name, centroid in self.domain_centroids.items():
            # Normaliser les vecteurs
            norm_embedding = document_embedding / np.linalg.norm(document_embedding)
            norm_centroid = centroid / np.linalg.norm(centroid)

            # Calculer la similarité cosinus
            similarity = np.dot(norm_embedding, norm_centroid)
            centroid_similarities[domain_name] = similarity

        # Trier les domaines par similarité décroissante
        sorted_domains = sorted(centroid_similarities.items(), key=lambda x: x[1], reverse=True)

        # Prendre les top domaines pour évaluation détaillée
        candidate_domains = [domain for domain, _ in sorted_domains[:min(top_k * 2, len(sorted_domains))]]

        # Deuxième étape: évaluation détaillée avec les réseaux de Hopfield
        domain_evaluations = []
        for domain_name in candidate_domains:
            if domain_name in self.domain_networks:
                network = self.domain_networks[domain_name]

                # Évaluer le document avec le réseau
                evaluation = network.evaluate_query(document_embedding)

                domain_evaluations.append({
                    "domain": domain_name,
                    "confidence": float(evaluation["confidence"]),
                    "energy": float(evaluation["energy"]),
                    "closest_patterns": evaluation["closest_patterns"]
                })

        # Trier par confiance décroissante
        domain_evaluations.sort(key=lambda x: x["confidence"], reverse=True)

        # Prendre les top-k résultats
        results = domain_evaluations[:min(top_k, len(domain_evaluations))]

        return results

    async def classify_document_by_id(self, document_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Classifie un document par son ID en utilisant le RAG."""
        # Rechercher l'embedding du document via le RAG
        document_embedding = await self._get_document_embedding(document_id)

        if document_embedding is None:
            return []

        # Classifier le document
        results = await self.classify_document(document_embedding, top_k)

        # Mettre à jour l'ontologie avec les résultats
        for result in results:
            domain_name = result["domain"]
            confidence = result["confidence"]

            # Associer le document au domaine avec le score de confiance
            self.ontology_manager.associate_document_with_domain(
                document_id, domain_name, confidence
            )

        return results

    async def create_domain_from_documents(
            self,
            domain_name: str,
            document_ids: List[str],
            description: str = None
    ) -> bool:
        """Crée un nouveau domaine à partir d'un ensemble de documents."""
        # Créer le domaine dans l'ontologie
        domain = self.ontology_manager.create_domain(domain_name, description)

        # Récupérer les embeddings des documents
        document_embeddings = []
        for doc_id in document_ids:
            embedding = await self._get_document_embedding(doc_id)
            if embedding is not None:
                document_embeddings.append((doc_id, embedding))

        if not document_embeddings:
            print(f"Aucun embedding trouvé pour les documents du domaine {domain_name}")
            return False

        # Entraîner le réseau de Hopfield pour ce domaine
        success = await self.train_domain(domain_name, document_embeddings)

        return success