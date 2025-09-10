"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# wavelet_rag.py
import asyncio
import os
import pickle
import sys

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pywt  # PyWavelets

from scipy.spatial.distance import cosine
from tqdm import tqdm

# Prendre les meilleurs candidats (au moins N*top_k ou P% des candidats)
N = 70 #50 initial 5
P = 0.5 #1.0 initial 0.4
# Mais au plus 100 candidats pour limiter la complexité
MAXC = 400


class SimpleWaveletEmbedding:
    """
    Classe pour gérer les embeddings basés sur les transformations en ondelettes
    permettant une recherche progressive multi-échelle
    """

    def __init__(
            self,
            wavelet: str = 'db1',  # Type d'ondelette (Daubechies 1 = Haar)
            levels: int = 5,  # Nombre de niveaux de décomposition
            threshold: float = 0.7  # Seuil de confiance
    ):
        """
        Initialise le système d'embedding basé sur les ondelettes

        Args:
            wavelet: Type d'ondelette (par défaut: 'db1' = Haar)
            levels: Nombre de niveaux de décomposition
            threshold: Seuil de confiance pour la recherche progressive
        """
        self.wavelet = wavelet
        self.levels = levels
        self.threshold = threshold

        # Dictionnaires pour stocker les décompositions à différents niveaux
        self.wavelet_approx = {}  # Niveau -> {id -> approximation}
        self.wavelet_details = {}  # Niveau -> {id -> détails}

        # Dictionnaire pour stocker les embeddings originaux
        self.raw_embeddings = {}  # id -> embedding

        # Initialiser les dictionnaires pour chaque niveau
        for level in range(levels + 1):  # +1 car le niveau 0 est l'embedding original
            self.wavelet_approx[level] = {}
            self.wavelet_details[level] = {}

    def not_normalized_add_embedding(self, id: str, embedding: List[float]):
        """
        Ajoute un embedding et calcule ses décompositions en ondelettes

        Args:
            id: Identifiant de l'embedding
            embedding: Vecteur d'embedding
        """
        # Vérifier l'embedding
        if embedding is None:
            print(f"Warning: embedding est None pour ID {id}")
            return

        # Convertir en numpy.ndarray si nécessaire
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # Stocker l'embedding original
        self.raw_embeddings[id] = embedding

        # Niveau 0 = embedding original
        self.wavelet_approx[0][id] = embedding
        self.wavelet_details[0][id] = []

        # Calculer les décompositions
        working_embedding = embedding.copy()  # Travailler sur une copie

        for level in range(1, self.levels + 1):
            # S'assurer que le vecteur a une longueur suffisante pour la décomposition
            if len(working_embedding) < 2:
                # Si trop petit, copier les niveaux précédents pour les niveaux restants
                self.wavelet_approx[level][id] = self.wavelet_approx[level - 1][id]
                self.wavelet_details[level][id] = self.wavelet_details[level - 1][id]
                continue

            # Effectuer la décomposition en ondelettes
            try:
                approx, detail = pywt.dwt(working_embedding, self.wavelet)

                # Stocker les résultats
                self.wavelet_approx[level][id] = approx
                self.wavelet_details[level][id] = detail

                # Utiliser l'approximation pour le niveau suivant
                working_embedding = approx
            except Exception as e:
                print(f"Erreur lors de la décomposition en ondelettes pour {id} au niveau {level}: {str(e)}")
                # Utiliser le niveau précédent en cas d'erreur
                self.wavelet_approx[level][id] = self.wavelet_approx[level - 1][id]
                self.wavelet_details[level][id] = self.wavelet_details[level - 1][id]

    def add_embedding(self, id: str, embedding: List[float]):
        """
        Ajoute un embedding et calcule ses décompositions en ondelettes
        avec normalisation à tous les niveaux

        Args:
            id: Identifiant de l'embedding
            embedding: Vecteur d'embedding
        """
        # Vérifier l'embedding
        if embedding is None:
            print(f"Warning: embedding est None pour ID {id}")
            return

        # Convertir en numpy.ndarray si nécessaire
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # Normaliser l'embedding original (niveau 0)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Stocker l'embedding original normalisé
        self.raw_embeddings[id] = embedding

        # Niveau 0 = embedding original (déjà normalisé)
        self.wavelet_approx[0][id] = embedding
        self.wavelet_details[0][id] = []

        # Calculer les décompositions
        working_embedding = embedding.copy()  # Travailler sur une copie (déjà normalisée)

        for level in range(1, self.levels + 1):
            # S'assurer que le vecteur a une longueur suffisante pour la décomposition
            if len(working_embedding) < 2:
                # Si trop petit, copier les niveaux précédents pour les niveaux restants
                self.wavelet_approx[level][id] = self.wavelet_approx[level - 1][id]
                self.wavelet_details[level][id] = self.wavelet_details[level - 1][id]
                continue

            # Effectuer la décomposition en ondelettes
            try:
                approx, detail = pywt.dwt(working_embedding, self.wavelet)

                # Normaliser l'approximation
                if len(approx) > 0:
                    norm = np.linalg.norm(approx)
                    if norm > 0:
                        approx = approx / norm

                # Stocker les résultats
                self.wavelet_approx[level][id] = approx
                self.wavelet_details[level][id] = detail

                # Utiliser l'approximation pour le niveau suivant
                working_embedding = approx
            except Exception as e:
                print(f"Erreur lors de la décomposition en ondelettes pour {id} au niveau {level}: {str(e)}")
                # Utiliser le niveau précédent en cas d'erreur
                self.wavelet_approx[level][id] = self.wavelet_approx[level - 1][id]
                self.wavelet_details[level][id] = self.wavelet_details[level - 1][id]


    def old_good_add_embedding(self, id: str, embedding: List[float]):
        """
        Ajoute un embedding et calcule ses décompositions en ondelettes

        Args:
            id: Identifiant de l'embedding
            embedding: Vecteur d'embedding
        """
        # Vérifier l'embedding
        if embedding is None:
            print(f"Warning: embedding est None pour ID {id}")
            return

        # Convertir en numpy.ndarray si nécessaire
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # Stocker l'embedding original
        self.raw_embeddings[id] = embedding

        # Niveau 0 = embedding original
        self.wavelet_approx[0][id] = embedding
        self.wavelet_details[0][id] = []

        # Calculer les décompositions
        working_embedding = embedding.copy()  # Travailler sur une copie

        for level in range(1, self.levels + 1):
            # S'assurer que le vecteur a une longueur suffisante pour la décomposition
            if len(working_embedding) < 2:
                # Si trop petit, copier les niveaux précédents pour les niveaux restants
                self.wavelet_approx[level][id] = self.wavelet_approx[level - 1][id]
                self.wavelet_details[level][id] = self.wavelet_details[level - 1][id]
                continue

            # Effectuer la décomposition en ondelettes
            try:
                approx, detail = pywt.dwt(working_embedding, self.wavelet)

                # Normaliser l'approximation
                if len(approx) > 0:
                    norm = np.linalg.norm(approx)
                    if norm > 0:
                        approx = approx / norm

                # Stocker les résultats
                self.wavelet_approx[level][id] = approx
                self.wavelet_details[level][id] = detail

                # Utiliser l'approximation pour le niveau suivant
                working_embedding = approx
            except Exception as e:
                print(f"Erreur lors de la décomposition en ondelettes pour {id} au niveau {level}: {str(e)}")
                # Utiliser le niveau précédent en cas d'erreur
                self.wavelet_approx[level][id] = self.wavelet_approx[level - 1][id]
                self.wavelet_details[level][id] = self.wavelet_details[level - 1][id]


    def get_embedding(self, id: str, level: int = 0) -> Optional[List[float]]:
        """
        Récupère l'embedding d'un niveau spécifique

        Args:
            id: Identifiant de l'embedding
            level: Niveau de décomposition (0 = original)

        Returns:
            Embedding au niveau spécifié ou None si non trouvé
        """
        if level not in self.wavelet_approx:
            return None

        return self.wavelet_approx[level].get(id)

    def batch_add_embeddings(self, embeddings: Dict[str, List[float]]):
        """
        Ajoute un lot d'embeddings

        Args:
            embeddings: Dictionnaire id -> embedding
        """
        for id, embedding in tqdm(embeddings.items(), desc="Décomposition en ondelettes"):
            self.add_embedding(id, embedding)

    # Modifiez la méthode similarity dans la classe SimpleWaveletEmbedding
    def similarity(self, embedding1, embedding2) -> float:
        """
        Calcule la similarité entre deux embeddings

        Args:
            embedding1: Premier embedding
            embedding2: Deuxième embedding

        Returns:
            Score de similarité (0-1)
        """
        # Vérification correcte pour les tableaux NumPy
        if embedding1 is None or embedding2 is None:
            return 0.0

        # Vérifier si les tableaux sont vides
        if isinstance(embedding1, np.ndarray):
            if embedding1.size == 0:
                return 0.0
        elif not embedding1:  # Pour les listes ou autres séquences
            return 0.0

        if isinstance(embedding2, np.ndarray):
            if embedding2.size == 0:
                return 0.0
        elif not embedding2:  # Pour les listes ou autres séquences
            return 0.0

        # On ne vérifie pas la taille des embedding, on en est sûre

        # Calculer la similarité cosinus
        try:
            return 1.0 - cosine(embedding1, embedding2)
        except Exception as e:
            print(f"Erreur lors du calcul de similarité: {str(e)}")
            return 0.0

    def save(self, path: str):
        """
        Sauvegarde la structure d'embeddings en ondelettes

        Args:
            path: Chemin du fichier de sauvegarde
        """
        # Convertir tous les tableaux NumPy en listes Python pour une meilleure sérialisation
        raw_embeddings_dict = {}
        wavelet_approx_dict = {}
        wavelet_details_dict = {}

        # Convertir les embeddings bruts
        for id, embedding in self.raw_embeddings.items():
            if isinstance(embedding, np.ndarray):
                raw_embeddings_dict[id] = embedding.tolist()
            else:
                raw_embeddings_dict[id] = embedding

        # Convertir les approximations
        for level in self.wavelet_approx:
            wavelet_approx_dict[level] = {}
            for id, approx in self.wavelet_approx[level].items():
                if isinstance(approx, np.ndarray):
                    wavelet_approx_dict[level][id] = approx.tolist()
                else:
                    wavelet_approx_dict[level][id] = approx

        # Convertir les détails
        for level in self.wavelet_details:
            wavelet_details_dict[level] = {}
            for id, detail in self.wavelet_details[level].items():
                if isinstance(detail, np.ndarray):
                    wavelet_details_dict[level][id] = detail.tolist()
                else:
                    wavelet_details_dict[level][id] = detail

        data = {
            "wavelet": self.wavelet,
            "levels": self.levels,
            "threshold": self.threshold,
            "raw_embeddings": raw_embeddings_dict,
            "wavelet_approx": wavelet_approx_dict,
            "wavelet_details": wavelet_details_dict
        }

        print(f"Sauvegarde de {len(raw_embeddings_dict)} embeddings en ondelettes avec {self.levels} niveaux")
        print(f"Sauvegarde des embeddings en ondelettes...")
        print(f"  Nombre d'embeddings bruts: {len(self.raw_embeddings)}")
        print(f"  Nombre d'approximations niveau 1: {len(self.wavelet_approx[1])}")
        print(f"  Taille des données à sauvegarder: ~{sys.getsizeof(data) / 1024:.2f} KB")

        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=4)  # Utiliser le protocole 4 pour les gros objets

    def load(self, path: str) -> bool:
        """
        Charge la structure d'embeddings en ondelettes

        Args:
            path: Chemin du fichier de sauvegarde

        Returns:
            True si le chargement a réussi, False sinon
        """
        if not os.path.exists(path):
            return False

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            self.wavelet = data["wavelet"]
            self.levels = data["levels"]
            self.threshold = data["threshold"]

            # Charger les embeddings bruts
            self.raw_embeddings = data["raw_embeddings"]

            # Charger les approximations et détails
            self.wavelet_approx = data["wavelet_approx"]
            self.wavelet_details = data["wavelet_details"]

            # Afficher des statistiques pour vérification
            print(f"Chargement des embeddings en ondelettes:")
            print(f"  Embeddings bruts: {len(self.raw_embeddings)}")
            for level in sorted(self.wavelet_approx.keys()):
                print(f"  Niveau {level}: {len(self.wavelet_approx[level])} approximations")

            return True
        except Exception as e:
            print(f"Erreur lors du chargement des embeddings en ondelettes: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


class WaveletRAG:
    """
    Système RAG basé sur les transformations en ondelettes
    permettant une recherche progressive multi-échelle
    """

    def __init__(
            self,
            rag_engine,  # Moteur RAG standard à augmenter
            wavelet: str = 'db3',
            levels: int = 5,
            storage_dir: str = "wavelet_storage",
            force_recompute_WA: bool = False
    ):
        """
        Initialise le système RAG basé sur les ondelettes

        Args:
            rag_engine: Moteur RAG standard à utiliser comme base
            wavelet: Type d'ondelette (par défaut: 'db1' = Haar)
            levels: Nombre de niveaux de décomposition
            storage_dir: Répertoire pour stocker les embeddings en ondelettes
            force_recompute_WA: Si True, force le recalcul des embeddings en ondelettes
        """
        self.rag = rag_engine
        self.storage_dir = storage_dir

        self.force_recompute_WA = force_recompute_WA  # Stocke le paramètre
        self.wavelet_type = wavelet  # Mémoriser le type demandé
        self.wavelet_levels = levels  # Mémoriser le nombre de niveaux demandé

        # Créer le répertoire de stockage
        os.makedirs(storage_dir, exist_ok=True)

        # Initialiser le système d'embeddings en ondelettes
        self.wavelet_embeddings = SimpleWaveletEmbedding(
            wavelet=wavelet,
            levels=levels
        )

        print(f"Fichier d'embeddings en ondelettes: {self.wavelet_embeddings_path}")

    async def initialize(self):
        """Initialise le système WaveletRAG"""
        # D'abord, initialiser le RAG standard
        await self.rag.initialize()

        # Si force_recompute_WA est True et le fichier existe, le supprimer
        # pour forcer un recalcul complet
        if self.force_recompute_WA and os.path.exists(self.wavelet_embeddings_path):
            print(f"Suppression des embeddings en ondelettes existants pour forcer le recalcul...")
            os.remove(self.wavelet_embeddings_path)

        # Vérifier s'il existe déjà un fichier d'embeddings en ondelettes
        if os.path.exists(self.wavelet_embeddings_path):
            print(f"Chargement des embeddings en ondelettes existants...")
            if self.wavelet_embeddings.load(self.wavelet_embeddings_path):
                # Vérifier si les paramètres chargés correspondent aux paramètres demandés
                if self.wavelet_embeddings.wavelet != self.wavelet_type or self.wavelet_embeddings.levels != self.wavelet_levels:
                    print(f"\n⚠️ ATTENTION: Les paramètres d'ondelettes chargés diffèrent des paramètres demandés:")
                    print(
                        f"   - Type d'ondelette: demandé={self.wavelet_type}, chargé={self.wavelet_embeddings.wavelet}")
                    print(f"   - Niveaux: demandé={self.wavelet_levels}, chargé={self.wavelet_embeddings.levels}")
                    print(f"   - Pour recalculer avec les nouveaux paramètres, utilisez force_recompute_WA=True\n")

                print(f"Embeddings en ondelettes chargés pour {len(self.wavelet_embeddings.raw_embeddings)} documents")
                return

        # Si on arrive ici, soit le fichier n'existe pas, soit le chargement a échoué
        print("Création des embeddings en ondelettes...")
        await self._create_wavelet_embeddings()

    async def _create_wavelet_embeddings(self):
        """Crée les embeddings en ondelettes à partir des embeddings du RAG standard"""
        # Récupérer tous les embeddings du RAG standard
        all_embeddings = self.rag.embedding_manager.get_all_embeddings()

        print(f"Récupération de {len(all_embeddings)} embeddings du RAG standard")
        if not all_embeddings:
            print("Aucun embedding trouvé dans le RAG standard")
            return

        # Filtrer pour ne traiter que les nouveaux embeddings si on ne force pas le recalcul
        if not self.force_recompute_WA and self.wavelet_embeddings.raw_embeddings:
            embeddings_to_process = {}
            for chunk_id, embedding in all_embeddings.items():
                if chunk_id not in self.wavelet_embeddings.raw_embeddings:
                    embeddings_to_process[chunk_id] = embedding

            if not embeddings_to_process:
                print("✓ Tous les embeddings sont déjà calculés dans le système d'ondelettes.")
                return

            print(f"Traitement de {len(embeddings_to_process)}/{len(all_embeddings)} nouveaux embeddings")
            # Ajouter seulement les nouveaux embeddings
            self.wavelet_embeddings.batch_add_embeddings(embeddings_to_process)
        else:
            # Recalcul complet si force_recompute_WA est activé ou premier calcul
            print(f"Calcul complet des embeddings en ondelettes pour {len(all_embeddings)} chunks")
            self.wavelet_embeddings.batch_add_embeddings(all_embeddings)

        # Sauvegarder après la mise à jour
        self.wavelet_embeddings.save(self.wavelet_embeddings_path)
        print(f"Embeddings en ondelettes sauvegardés dans {self.wavelet_embeddings_path}")

    @property
    def wavelet_embeddings_path(self):
        """Retourne le chemin standardisé vers le fichier d'embeddings en ondelettes"""
        return os.path.join(self.storage_dir, "wavelet_embeddings.pkl")

    async def add_document(self, filepath: str) -> str:
        """
        Ajoute un document au système

        Args:
            filepath: Chemin vers le document

        Returns:
            ID du document ajouté
        """
        # Ajouter le document au RAG standard
        document_id = await self.rag.add_document(filepath)

        # Récupérer les embeddings du document
        document_embeddings = self.rag.embedding_manager.get_document_embeddings(document_id)

        # Ajouter les embeddings au système d'ondelettes
        self.wavelet_embeddings.batch_add_embeddings(document_embeddings)

        # Sauvegarder les embeddings en ondelettes
        self.wavelet_embeddings.save(self.wavelet_embeddings_path)

        return document_id

    async def add_documents(self, filepaths: List[str]) -> List[str]:
        """
        Ajoute plusieurs documents au système

        Args:
            filepaths: Liste des chemins vers les documents

        Returns:
            Liste des IDs des documents ajoutés
        """
        document_ids = []
        for filepath in filepaths:
            document_id = await self.add_document(filepath)
            document_ids.append(document_id)
        return document_ids

    async def remove_document(self, document_id: str) -> bool:
        """
        Supprime un document du système

        Args:
            document_id: ID du document à supprimer

        Returns:
            True si la suppression a réussi, False sinon
        """
        # Supprimer le document du RAG standard
        result = await self.rag.remove_document(document_id)

        if not result:
            return False

        # Supprimer les embeddings en ondelettes pour ce document
        for level in range(self.wavelet_embeddings.levels + 1):
            # Identifier les embeddings à supprimer
            to_remove = []
            for id in self.wavelet_embeddings.wavelet_approx[level]:
                if id.startswith(f"{document_id}-chunk-"):
                    to_remove.append(id)

            # Supprimer les embeddings
            for id in to_remove:
                if id in self.wavelet_embeddings.wavelet_approx[level]:
                    del self.wavelet_embeddings.wavelet_approx[level][id]
                if id in self.wavelet_embeddings.wavelet_details[level]:
                    del self.wavelet_embeddings.wavelet_details[level][id]

        # Supprimer les embeddings bruts
        to_remove = [id for id in self.wavelet_embeddings.raw_embeddings if id.startswith(f"{document_id}-chunk-")]
        for id in to_remove:
            if id in self.wavelet_embeddings.raw_embeddings:
                del self.wavelet_embeddings.raw_embeddings[id]

        # Sauvegarder les embeddings en ondelettes
        self.wavelet_embeddings.save(self.wavelet_embeddings_path)

        return True

    async def get_all_documents(self) -> Dict[str, Dict[str, Any]]:
        """Récupère tous les documents"""
        return await self.rag.get_all_documents()

    async def progressive_search(
            self,
            query: str,
            start_level: int = 4,
            max_level: int = 0,
            top_k: int = 5,
            threshold: float = 0.7,
            document_id: Optional[str] = None,
            skip_loading: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Effectue une recherche progressive multi-échelle

        Args:
            query: Requête de recherche
            start_level: Niveau de départ (plus grossier)
            max_level: Niveau maximum (plus détaillé, 0 = original)
            top_k: Nombre de passages à retourner
            threshold: Seuil de confiance

        Returns:
            Liste des passages les plus pertinents avec leur score
        """

        # Générer l'embedding de la requête
        query_embedding = (await self.rag.embedding_manager.provider.generate_embeddings([query]))[0]

        # Liste de tous les IDs de chunks
        all_chunk_ids = list(self.wavelet_embeddings.raw_embeddings.keys())

        # Filtrer par document_id si spécifié
        if document_id:
            all_chunk_ids = [chunk_id for chunk_id in all_chunk_ids
                             if chunk_id.startswith(f"{document_id}-chunk-")]

            if not all_chunk_ids:
                print(f"Aucun chunk trouvé pour le document {document_id}")
                return []

        # Précalculer toutes les décompositions de la requête
        query_wavelets = {0: query_embedding}  # Niveau 0 = original
        working_embedding = query_embedding.copy()

        # Ajuster les niveaux pour qu'ils soient dans les limites
        start_level = min(start_level, self.wavelet_embeddings.levels)
        max_level = max(max_level, 0)

        for level in range(1, start_level + 1):
            if len(working_embedding) < 2:
                # Si trop petit, utiliser le niveau précédent
                query_wavelets[level] = query_wavelets[level - 1]
                continue

            approx, _ = pywt.dwt(working_embedding, self.wavelet_embeddings.wavelet)
            query_wavelets[level] = approx
            working_embedding = approx  # Pour le niveau suivant

        if not all_chunk_ids:
            print("Aucun chunk trouvé dans la base de données")
            return []

        # Liste des candidats à considérer à chaque niveau
        candidates = all_chunk_ids

        # Niveau actuel
        level = start_level

        # Score de confiance maximum trouvé
        max_confidence = 0.0

        # Résultats intermédiaires pour les candidats
        candidate_scores = {}

        # Pour le logging
        #print(f"Recherche progressive: {len(candidates)} candidats initiaux")

        # Variables pour l'arrêt anticipé
        previous_max_score = 0
        no_improvement_count = 0
        improvement_threshold = 0.02  # 2% d'amélioration minimum

        # Boucle de recherche progressive
        while level >= max_level:
            #print(f"Niveau {level}: {len(candidates)} candidats")

            # Transformer l'embedding de requête au niveau actuel
            query_wavelet = query_wavelets[level]

            # Calculer les scores pour les candidats à ce niveau
            level_scores = {}
            for chunk_id in candidates:
                chunk_embedding = self.wavelet_embeddings.get_embedding(chunk_id, level)
                if chunk_embedding is not None:
                    score = self.wavelet_embeddings.similarity(query_wavelet, chunk_embedding)
                    level_scores[chunk_id] = score

            # Trouver le score maximum
            if level_scores:
                max_score = max(level_scores.values())
                max_confidence = max(max_confidence, max_score)

                # Vérifier s'il y a une amélioration significative
                improvement = max_score - previous_max_score
                if improvement < improvement_threshold:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0

                previous_max_score = max_score

                # Si pas d'amélioration significative pendant 2 niveaux consécutifs
                if no_improvement_count >= 2 and level <= (start_level - 2):
                    #print(f"Arrêt anticipé au niveau {level}: pas d'amélioration significative")
                    candidate_scores.update(level_scores)
                    break

                # Si on a atteint le seuil de confiance, on peut s'arrêter
                if max_confidence >= threshold and level <= 1:  # Au moins atteindre le niveau 1 ou 0
                    #print(f"Seuil de confiance atteint au niveau {level}: {max_confidence:.4f}")
                    # Mettre à jour les scores des candidats
                    candidate_scores.update(level_scores)
                    break

            # Filtrer les candidats pour le niveau suivant
            if level > max_level:  # Si on n'est pas au dernier niveau
                # Prendre les meilleurs candidats (au moins 3*top_k ou 20% des candidats)
                min_candidates = max(N * top_k, int(P * len(candidates)))
                # Mais au plus 100 candidats pour limiter la complexité
                min_candidates = min(min_candidates, MAXC)

                # Trier les candidats par score
                sorted_candidates = sorted(level_scores.items(), key=lambda x: x[1], reverse=True)

                # Prendre les meilleurs candidats
                candidates = [c[0] for c in sorted_candidates[:min_candidates]]

                # Mettre à jour les scores des candidats
                for chunk_id, score in level_scores.items():
                    if chunk_id in candidates:  # Ne conserver que les scores des candidats sélectionnés
                        candidate_scores[chunk_id] = score
            else:
                # Au dernier niveau, mettre à jour tous les scores
                candidate_scores.update(level_scores)

            # Passer au niveau plus détaillé
            level -= 1

        # Si on a atteint le niveau le plus détaillé, utiliser les scores finaux
        if level < max_level:
            pass
            #print(f"Recherche jusqu'au niveau le plus détaillé: {max_confidence:.4f}")

        # Trier les candidats par score final
        sorted_results = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

        # Extraire les top_k résultats
        top_chunk_ids = [r[0] for r in sorted_results[:top_k]]

        # Récupérer les informations complètes sur ces chunks
        passages = []
        for chunk_id in top_chunk_ids:
            # Extraire l'ID du document à partir de l'ID du chunk (format: "doc_id-chunk-N")
            parts = chunk_id.split("-chunk-")
            if len(parts) != 2:
                continue

            document_id = parts[0]

            # Charger les chunks du document si nécessaire
            if not skip_loading:
                await self.rag.document_store.load_document_chunks(document_id)
            doc_chunks = await self.rag.document_store.get_document_chunks(document_id)

            if not doc_chunks:
                continue

            # Trouver le chunk correspondant
            chunk = None
            for c in doc_chunks:
                if c["id"] == chunk_id:
                    chunk = c
                    break

            if not chunk:
                continue

            # Récupérer les informations sur le document
            document = await self.rag.document_store.get_document(document_id)

            if not document:
                continue

            # Ajouter le passage aux résultats
            passages.append({
                "document_id": document_id,
                "chunk_id": chunk_id,
                "text": chunk["text"],
                "similarity": candidate_scores[chunk_id],
                "start_pos": chunk["start_pos"],
                "end_pos": chunk["end_pos"],
                "metadata": chunk.get("metadata", {}),
                "document_name": document.get("original_filename", ""),
                "document_path": document.get("path", "")
            })

        #print(f"Recherche terminée: {len(passages)} passages trouvés sur {len(candidate_scores)} candidats évalués")

        return passages

    async def search(
            self,
            query: str,
            document_id: Optional[str] = None,
            top_k: int = 5,
            skip_loading: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Interface standard de recherche, compatible avec l'API RAG

        Args:
            query: Requête de recherche
            document_id: ID du document (ignoré dans l'implémentation actuelle)
            top_k: Nombre de passages à retourner

        Returns:
            Liste des passages les plus pertinents avec leur score
        """
        # Utiliser la recherche progressive
        return await self.progressive_search(
            query=query,
            start_level=min(self.wavelet_embeddings.levels-1, 4),  # Niveau max disponible ou 4
            max_level=0,  # Aller jusqu'au niveau le plus détaillé si nécessaire
            top_k=top_k,
            threshold=0.7,  # Seuil de confiance élevé
            document_id=document_id,
            skip_loading=skip_loading
        )

    async def search_with_embedding(
            self,
            query_embedding: np.ndarray,  # Modifié pour accepter directement un numpy array
            start_level: int = None,
            max_level: int = 0,
            top_k: int = 5,
            threshold: float = 0.7,
            document_id: Optional[str] = None,
            skip_loading: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Effectue une recherche progressive multi-échelle, optimisée avec SIMD

        Args:
            query_embedding: Embedding de la requête sous forme de numpy array
            start_level: Niveau de départ (plus grossier)
            max_level: Niveau maximum (plus détaillé, 0 = original)
            top_k: Nombre de passages à retourner
            threshold: Seuil de confiance
            document_id: ID du document (si None, recherche dans tous les documents)
            skip_loading: Si True, suppose que les documents sont déjà chargés

        Returns:
            Liste des passages les plus pertinents avec leur score
        """
        # Liste de tous les IDs de chunks
        all_chunk_ids = list(self.wavelet_embeddings.raw_embeddings.keys())

        # Filtrer par document_id si spécifié
        if document_id:
            all_chunk_ids = [chunk_id for chunk_id in all_chunk_ids
                             if chunk_id.startswith(f"{document_id}-chunk-")]

            if not all_chunk_ids:
                print(f"Aucun chunk trouvé pour le document {document_id}")
                return []

        # Précalculer toutes les décompositions de la requête
        query_wavelets = {0: query_embedding}  # Niveau 0 = original
        working_embedding = query_embedding.copy()

        # Ajuster les niveaux pour qu'ils soient dans les limites
        start_level = self.wavelet_embeddings.levels if start_level is None else start_level
        max_level = max(max_level, 0)

        # Calculer les wavelets de la requête pour tous les niveaux à l'avance
        for level in range(1, start_level + 1):
            if len(working_embedding) < 2:
                # Si trop petit, utiliser le niveau précédent
                query_wavelets[level] = query_wavelets[level - 1]
                continue

            approx, _ = pywt.dwt(working_embedding, self.wavelet_embeddings.wavelet)

            # TODO: Normalisation si nécessaire

            query_wavelets[level] = approx
            working_embedding = approx  # Pour le niveau suivant

        if not all_chunk_ids:
            print("Aucun chunk trouvé dans la base de données")
            return []

        # Liste des candidats à considérer à chaque niveau
        candidates = all_chunk_ids

        # Niveau actuel
        level = start_level

        # Score de confiance maximum trouvé
        max_confidence = 0.0

        # Résultats intermédiaires pour les candidats
        candidate_scores = {}

        # Variables pour l'arrêt anticipé
        previous_max_score = 0
        no_improvement_count = 0
        improvement_threshold = 0.02  # 2% d'amélioration minimum

        # Boucle de recherche progressive
        while level >= max_level:
            # Obtenir l'embedding de requête pour le niveau actuel
            query_wavelet = query_wavelets[level]

            # Collecter tous les embeddings pour le niveau actuel en une seule fois
            valid_embeddings = {}
            valid_indices = []
            valid_chunk_ids = []

            for i, chunk_id in enumerate(candidates):
                embedding = self.wavelet_embeddings.get_embedding(chunk_id, level)
                if embedding is not None:
                    valid_embeddings[chunk_id] = embedding
                    valid_indices.append(i)
                    valid_chunk_ids.append(chunk_id)

            # Optimisation SIMD: calculer toutes les similarités en une seule opération
            if valid_chunk_ids:
                # Créer une matrice d'embeddings
                embeddings_matrix = np.vstack([valid_embeddings[chunk_id] for chunk_id in valid_chunk_ids])

                # Calculer toutes les similarités en une seule opération (dot product si normalisé)

                # Normalisation des vecteurs
                #embeddings_matrix = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
                query_wavelet = query_wavelet / np.linalg.norm(query_wavelet)

                # Sinon, utiliser une autre méthode optimisée
                if hasattr(self.wavelet_embeddings, 'similarity_batch'):
                    # Utiliser une méthode batch si disponible
                    scores = self.wavelet_embeddings.similarity_batch(query_wavelet, embeddings_matrix)
                else:
                    # Implémentation par défaut - produit scalaire si normalisé
                    scores = np.dot(embeddings_matrix, query_wavelet)

                # Associer les scores aux IDs de chunks
                level_scores = {valid_chunk_ids[i]: float(scores[i]) for i in range(len(valid_chunk_ids))}

                # Trouver le score maximum
                max_score = np.max(scores) if len(scores) > 0 else 0
                max_confidence = max(max_confidence, max_score)

                # Vérifier s'il y a une amélioration significative
                improvement = max_score - previous_max_score
                if improvement < improvement_threshold:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0

                previous_max_score = max_score

                # Si pas d'amélioration significative pendant 2 niveaux consécutifs
                if no_improvement_count >= 2 and level <= (start_level - 2):
                    candidate_scores.update(level_scores)
                    break

                # Si on a atteint le seuil de confiance, on peut s'arrêter
                if max_confidence >= threshold and level <= 1:  # Au moins atteindre le niveau 1 ou 0
                    # Mettre à jour les scores des candidats
                    candidate_scores.update(level_scores)
                    break

                # Filtrer les candidats pour le niveau suivant
                if level > max_level:  # Si on n'est pas au dernier niveau
                    # Utiliser argsort pour obtenir les indices des meilleurs scores plus rapidement
                    if len(scores) > 0:
                        # Calcul du nombre de candidats à conserver
                        min_candidates = max(N * top_k, int(P * len(candidates)))
                        min_candidates = min(min_candidates, MAXC)
                        min_candidates = min(min_candidates, len(scores))

                        # Trouver les indices des meilleurs scores
                        top_indices = np.argsort(scores)[-min_candidates:][::-1]

                        # Sélectionner les meilleurs chunks
                        candidates = [valid_chunk_ids[i] for i in top_indices]

                        # Mettre à jour les scores des candidats sélectionnés
                        for i, idx in enumerate(top_indices):
                            chunk_id = valid_chunk_ids[idx]
                            candidate_scores[chunk_id] = float(scores[idx])
                    else:
                        # Aucun embedding valide à ce niveau
                        break
                else:
                    # Au dernier niveau, mettre à jour tous les scores
                    candidate_scores.update(level_scores)
            else:
                # Aucun embedding valide à ce niveau
                pass

            # Passer au niveau plus détaillé
            level -= 1

        # Sélectionner les top_k meilleurs résultats
        if candidate_scores:
            # Tri optimisé pour extraire uniquement les top_k
            chunk_ids = list(candidate_scores.keys())
            scores = np.array([candidate_scores[chunk_id] for chunk_id in chunk_ids])

            # Obtenir les indices des top_k meilleurs scores
            if len(scores) > top_k:
                top_indices = np.argpartition(scores, -top_k)[-top_k:]
                # Trier les top_k par ordre décroissant
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            else:
                top_indices = np.argsort(scores)[::-1]

            # Extraire les top_k chunk IDs
            top_chunk_ids = [chunk_ids[i] for i in top_indices]
        else:
            top_chunk_ids = []

        # Récupérer les informations complètes sur ces chunks
        passages = []
        for chunk_id in top_chunk_ids:
            # Extraire l'ID du document à partir de l'ID du chunk
            parts = chunk_id.split("-chunk-")
            if len(parts) != 2:
                continue

            document_id = parts[0]

            # Charger les chunks du document si nécessaire
            if not skip_loading:
                await self.rag.document_store.load_document_chunks(document_id)
            doc_chunks = await self.rag.document_store.get_document_chunks(document_id)

            if not doc_chunks:
                continue

            # Trouver le chunk correspondant
            chunk = None
            for c in doc_chunks:
                if c["id"] == chunk_id:
                    chunk = c
                    break

            if not chunk:
                continue

            # Récupérer les informations sur le document
            document = await self.rag.document_store.get_document(document_id)

            if not document:
                continue

            # Ajouter le passage aux résultats
            passages.append({
                "document_id": document_id,
                "chunk_id": chunk_id,
                "text": chunk["text"],
                "similarity": candidate_scores[chunk_id],
                "start_pos": chunk["start_pos"],
                "end_pos": chunk["end_pos"],
                "metadata": chunk.get("metadata", {}),
                "document_name": document.get("original_filename", ""),
                "document_path": document.get("path", "")
            })

        return passages


    async def old_search_with_embedding(
            self,
            query_embedding: str,
            start_level: int = None,
            max_level: int = 0,
            top_k: int = 5,
            threshold: float = 0.7,
            document_id: Optional[str] = None,
            skip_loading: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Effectue une recherche progressive multi-échelle

        Args:
            query: Requête de recherche
            start_level: Niveau de départ (plus grossier)
            max_level: Niveau maximum (plus détaillé, 0 = original)
            top_k: Nombre de passages à retourner
            threshold: Seuil de confiance

        Returns:
            Liste des passages les plus pertinents avec leur score
        """

        # Liste de tous les IDs de chunks
        all_chunk_ids = list(self.wavelet_embeddings.raw_embeddings.keys())

        # Filtrer par document_id si spécifié
        if document_id:
            all_chunk_ids = [chunk_id for chunk_id in all_chunk_ids
                             if chunk_id.startswith(f"{document_id}-chunk-")]

            if not all_chunk_ids:
                print(f"Aucun chunk trouvé pour le document {document_id}")
                return []

        # Précalculer toutes les décompositions de la requête
        query_wavelets = {0: query_embedding}  # Niveau 0 = original
        working_embedding = query_embedding.copy()

        # Ajuster les niveaux pour qu'ils soient dans les limites
        start_level = self.wavelet_embeddings.levels if start_level is None else start_level
        max_level = max(max_level, 0)

        for level in range(1, start_level + 1):
            if len(working_embedding) < 2:
                # Si trop petit, utiliser le niveau précédent
                query_wavelets[level] = query_wavelets[level - 1]
                continue

            approx, _ = pywt.dwt(working_embedding, self.wavelet_embeddings.wavelet)

            """
            # TODO Normalisation
            # Normaliser l'approximation
            if len(approx) > 0:
                norm = np.linalg.norm(approx)
                if norm > 0:
                    approx = approx / norm
            """
            query_wavelets[level] = approx
            working_embedding = approx  # Pour le niveau suivant

        if not all_chunk_ids:
            print("Aucun chunk trouvé dans la base de données")
            return []

        # Liste des candidats à considérer à chaque niveau
        candidates = all_chunk_ids

        # Niveau actuel
        level = start_level

        # Score de confiance maximum trouvé
        max_confidence = 0.0

        # Résultats intermédiaires pour les candidats
        candidate_scores = {}

        # Pour le logging
        #print(f"Recherche progressive: {len(candidates)} candidats initiaux")

        # Variables pour l'arrêt anticipé
        previous_max_score = 0
        no_improvement_count = 0
        improvement_threshold = 0.02  # 2% d'amélioration minimum

        # Boucle de recherche progressive
        while level >= max_level:
            #print(f"Niveau {level}: {len(candidates)} candidats")

            # Transformer l'embedding de requête au niveau actuel
            query_wavelet = query_wavelets[level]

            # Calculer les scores pour les candidats à ce niveau
            level_scores = {}
            for chunk_id in candidates:
                chunk_embedding = self.wavelet_embeddings.get_embedding(chunk_id, level)
                if chunk_embedding is not None:
                    score = self.wavelet_embeddings.similarity(query_wavelet, chunk_embedding)
                    level_scores[chunk_id] = score

            # Trouver le score maximum
            if level_scores:
                max_score = max(level_scores.values())
                max_confidence = max(max_confidence, max_score)

                # Vérifier s'il y a une amélioration significative
                improvement = max_score - previous_max_score
                if improvement < improvement_threshold:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0

                previous_max_score = max_score

                # Si pas d'amélioration significative pendant 2 niveaux consécutifs
                if no_improvement_count >= 2 and level <= (start_level - 2):
                    #print(f"Arrêt anticipé au niveau {level}: pas d'amélioration significative")
                    candidate_scores.update(level_scores)
                    break

                # Si on a atteint le seuil de confiance, on peut s'arrêter
                if max_confidence >= threshold and level <= 1:  # Au moins atteindre le niveau 1 ou 0
                    #print(f"Seuil de confiance atteint au niveau {level}: {max_confidence:.4f}")
                    # Mettre à jour les scores des candidats
                    candidate_scores.update(level_scores)
                    break

            # Filtrer les candidats pour le niveau suivant
            if level > max_level:  # Si on n'est pas au dernier niveau
                # Prendre les meilleurs candidats (au moins 3*top_k ou 20% des candidats)
                min_candidates = max(N * top_k, int(P * len(candidates)))
                # Mais au plus 100 candidats pour limiter la complexité
                min_candidates = min(min_candidates, MAXC)

                # Trier les candidats par score
                sorted_candidates = sorted(level_scores.items(), key=lambda x: x[1], reverse=True)

                # Prendre les meilleurs candidats
                candidates = [c[0] for c in sorted_candidates[:min_candidates]]

                # Mettre à jour les scores des candidats
                for chunk_id, score in level_scores.items():
                    if chunk_id in candidates:  # Ne conserver que les scores des candidats sélectionnés
                        candidate_scores[chunk_id] = score
            else:
                # Au dernier niveau, mettre à jour tous les scores
                candidate_scores.update(level_scores)

            # Passer au niveau plus détaillé
            level -= 1

        # Si on a atteint le niveau le plus détaillé, utiliser les scores finaux
        if level < max_level:
            pass
            #print(f"Recherche jusqu'au niveau le plus détaillé: {max_confidence:.4f}")

        # Trier les candidats par score final
        sorted_results = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

        # Extraire les top_k résultats
        top_chunk_ids = [r[0] for r in sorted_results[:top_k]]

        # Récupérer les informations complètes sur ces chunks
        passages = []
        for chunk_id in top_chunk_ids:
            # Extraire l'ID du document à partir de l'ID du chunk (format: "doc_id-chunk-N")
            parts = chunk_id.split("-chunk-")
            if len(parts) != 2:
                continue

            document_id = parts[0]

            # Charger les chunks du document si nécessaire
            if not skip_loading:
                await self.rag.document_store.load_document_chunks(document_id)
            doc_chunks = await self.rag.document_store.get_document_chunks(document_id)

            if not doc_chunks:
                continue

            # Trouver le chunk correspondant
            chunk = None
            for c in doc_chunks:
                if c["id"] == chunk_id:
                    chunk = c
                    break

            if not chunk:
                continue

            # Récupérer les informations sur le document
            document = await self.rag.document_store.get_document(document_id)

            if not document:
                continue

            # Ajouter le passage aux résultats
            passages.append({
                "document_id": document_id,
                "chunk_id": chunk_id,
                "text": chunk["text"],
                "similarity": candidate_scores[chunk_id],
                "start_pos": chunk["start_pos"],
                "end_pos": chunk["end_pos"],
                "metadata": chunk.get("metadata", {}),
                "document_name": document.get("original_filename", ""),
                "document_path": document.get("path", "")
            })

        #print(f"Recherche terminée: {len(passages)} passages trouvés sur {len(candidate_scores)} candidats évalués")

        return passages

    async def chat(
            self,
            query: str,
            top_k: int = 5,
            document_id: Optional[str] = None,
            system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Effectue une recherche et génère une réponse pour le chat

        Args:
            query: Requête de l'utilisateur
            top_k: Nombre de passages à récupérer
            document_id: ID du document (ignoré dans l'implémentation actuelle)
            system_prompt: Prompt système optionnel

        Returns:
            Dictionnaire contenant la réponse et les passages
        """
        # Récupérer les passages pertinents avec la recherche progressive
        passages = await self.search(query, document_id, top_k)

        # Utiliser le RAG standard pour générer une réponse
        answer = await self.rag.generate_answer(query, passages, system_prompt)

        # Vérifier si on peut surligner des passages dans un PDF
        highlighted_pdf = None
        if document_id and all(p["document_id"] == document_id for p in passages):
            document = await self.rag.document_store.get_document(document_id)
            if document and document["path"].lower().endswith('.pdf'):
                highlighted_pdf = await self.rag.highlight_passages(document_id, passages)

        return {
            "answer": answer,
            "passages": passages,
            "highlighted_pdf": highlighted_pdf
        }


# Ajouter cette fonction à la fin du fichier wavelet_rag.py
def inspect_wavelet_storage(file_path):
    """Inspecte et affiche des informations sur le fichier de stockage des ondelettes"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print(f"\n==== Structure du fichier d'embeddings en ondelettes ====")
        print(f"Taille du fichier: {os.path.getsize(file_path) / 1024:.2f} KB")
        print(f"Type d'ondelette: {data['wavelet']}")
        print(f"Nombre de niveaux: {data['levels']}")
        print(f"Seuil de confiance: {data['threshold']}")

        print(f"\nEmbeddings bruts: {len(data['raw_embeddings'])} chunks")
        if data['raw_embeddings']:
            sample_id = next(iter(data['raw_embeddings'].keys()))
            sample_embed = data['raw_embeddings'][sample_id]
            print(f"Dimension d'un embedding brut: {len(sample_embed)}")

        print("\nApproximations par niveau:")
        for level in sorted(data['wavelet_approx'].keys()):
            approx = data['wavelet_approx'].get(level, {})
            print(f"  Niveau {level}: {len(approx)} chunks")
            if approx:
                sample_id = next(iter(approx.keys()))
                sample_approx = approx[sample_id]
                if isinstance(sample_approx, (list, np.ndarray)):
                    print(f"    Dimension: {len(sample_approx)}")

    except Exception as e:
        print(f"Erreur lors de l'inspection du fichier: {str(e)}")
        import traceback
        traceback.print_exc()