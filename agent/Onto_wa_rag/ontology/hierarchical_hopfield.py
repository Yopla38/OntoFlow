"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# ontology/hierarchical_hopfield.py
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from sklearn.preprocessing import normalize
import torch
import torch.nn.functional as F

from ..CONSTANT import BETA, NORMALIZED_PATTERN, TOP_K_CONCEPT
from .hopfield_network import ModernHopfieldNetwork


class HierarchicalHopfieldClassifier:
    """
    Système de classification hiérarchique utilisant des réseaux de Hopfield distincts
    pour chaque niveau de l'ontologie.
    """

    def __init__(
            self,
            rag_engine,
            ontology_manager,
            storage_dir: str = "hopfield_models"
    ):
        """
        Initialise le classifieur Hopfield hiérarchique.

        Args:
            rag_engine: Moteur RAG pour accéder aux embeddings
            ontology_manager: Gestionnaire d'ontologie
            storage_dir: Répertoire pour stocker les modèles
        """
        self.rag_engine = rag_engine
        self.ontology_manager = ontology_manager
        self.storage_dir = storage_dir

        # Créer le répertoire de stockage
        os.makedirs(storage_dir, exist_ok=True)

        # UN SEUL réseau de Hopfield par niveau
        self.level_networks = {}  # niveau -> réseau de Hopfield

        # Structure hiérarchique de l'ontologie
        self.domain_hierarchy = {}  # domaine -> [sous-domaines]
        self.domain_to_level = {}  # domaine -> niveau
        self.level_to_domains = {}  # niveau -> [domaines]
        self.max_level = 1

        # Embeddings représentatifs de chaque domaine
        self.domain_embeddings = {}  # domaine -> embedding

    async def initialize(self):
        """Initialise la structure hiérarchique et charge les réseaux existants."""
        print("Initialisation du classifieur hiérarchique...")

        # 1. Analyser la structure hiérarchique des domaines dans l'ontologie
        await self._build_domain_hierarchy()

        if not self.domain_to_level or not self.level_to_domains:
            print("⚠️ Aucun domaine n'a été détecté dans la hiérarchie. Création forcée...")
            # Créer manuellement un niveau pour chaque domaine
            for name in self.ontology_manager.domains:
                self.domain_to_level[name] = 1
                if 1 not in self.level_to_domains:
                    self.level_to_domains[1] = []
                if name not in self.level_to_domains[1]:
                    self.level_to_domains[1].append(name)
                self.max_level = 1

            print(f"  Domaines forcés au niveau 1: {', '.join(self.level_to_domains[1])}")

        # 2. Charger ou créer les réseaux de Hopfield pour chaque niveau
        for level in range(1, self.max_level + 1):
            level_dir = os.path.join(self.storage_dir, f"level_{level}")
            os.makedirs(level_dir, exist_ok=True)

            # Créer un réseau pour ce niveau
            network = ModernHopfieldNetwork(
                beta=BETA,  # Valeur plus élevée pour une classification plus précise
                normalize_patterns=NORMALIZED_PATTERN,
                storage_dir=level_dir
            )

            # Tenter de charger un réseau existant
            loaded = network.load()
            if loaded:
                print(f"✓ Réseau de niveau {level} chargé")
            else:
                print(f"✓ Nouveau réseau créé pour le niveau {level}")

            self.level_networks[level] = network

        # 3. Charger les embeddings des domaines s'ils existent
        await self._load_domain_embeddings()

        print(f"✓ Classifieur hiérarchique initialisé avec {self.max_level} niveaux")
        for level, domains in self.level_to_domains.items():
            print(f"  - Niveau {level}: {len(domains)} domaines")

    # Dans hierarchical_hopfield.py, nouvelle méthode pour remplacer create_domain_for_level:

    async def add_domain_to_hierarchy(
            self,
            domain_name: str,
            domain_description: str = None,
            parent_domain: str = None
    ) -> bool:
        """
        Ajoute un domaine à la hiérarchie en générant son embedding conceptuel.

        Args:
            domain_name: Nom du domaine à ajouter
            domain_description: Description du domaine (optionnelle)
            parent_domain: Nom du domaine parent (None pour niveau 1)

        Returns:
            True si l'ajout a réussi, False sinon
        """
        # 1. Déterminer le niveau du domaine
        if parent_domain:
            # Récupérer le niveau du parent
            parent_level = self.domain_to_level.get(parent_domain)
            if parent_level is None:
                print(f"⚠️ Domaine parent '{parent_domain}' non trouvé")
                return False

            # Niveau de ce domaine = niveau du parent + 1
            domain_level = parent_level + 1

            # Mettre à jour la hiérarchie
            if parent_domain not in self.domain_hierarchy:
                self.domain_hierarchy[parent_domain] = []
            self.domain_hierarchy[parent_domain].append(domain_name)
        else:
            # Sans parent, c'est un domaine de niveau 1
            domain_level = 1

        # Initialiser l'entrée du domaine dans la hiérarchie
        self.domain_hierarchy[domain_name] = []
        self.domain_to_level[domain_name] = domain_level

        if domain_level not in self.level_to_domains:
            self.level_to_domains[domain_level] = []
        self.level_to_domains[domain_level].append(domain_name)

        self.max_level = max(self.max_level, domain_level)

        # 2. Générer l'embedding conceptuel du domaine
        # Construire un texte riche décrivant ce domaine
        domain_text = f"{domain_name}"
        if domain_description:
            domain_text += f": {domain_description}"

        # Ajouter des informations sur sa place dans la hiérarchie
        if parent_domain:
            domain_text += f" (sous-domaine de {parent_domain})"

        # Ajouter des concepts connexes si disponibles
        domain_obj = self.ontology_manager.domains.get(domain_name)
        if domain_obj and domain_obj.concepts:
            concept_names = [c.label for c in domain_obj.concepts if c.label]
            if concept_names:
                domain_text += f". Concepts associés: {', '.join(concept_names)}"

        # Générer l'embedding du texte représentant le domaine
        domain_embedding = None
        try:
            embeddings = await self.rag_engine.embedding_manager.provider.generate_embeddings([domain_text])
            if embeddings and len(embeddings) > 0:
                domain_embedding = embeddings[0]
                # Normaliser
                domain_embedding = domain_embedding / np.linalg.norm(domain_embedding)
        except Exception as e:
            print(f"⚠️ Erreur lors de la génération de l'embedding pour '{domain_name}': {e}")
            return False

        if domain_embedding is None:
            print(f"⚠️ Impossible de générer l'embedding pour '{domain_name}'")
            return False

        # Stocker l'embedding du domaine
        self.domain_embeddings[domain_name] = domain_embedding

        # 3. S'assurer que le réseau pour ce niveau existe
        if domain_level not in self.level_networks:
            level_dir = os.path.join(self.storage_dir, f"level_{domain_level}")
            os.makedirs(level_dir, exist_ok=True)

            self.level_networks[domain_level] = ModernHopfieldNetwork(
                beta=20.0,
                normalize_patterns=True,
                storage_dir=level_dir
            )

        # 4. Ajouter l'embedding conceptuel du domaine au réseau de son niveau
        network = self.level_networks[domain_level]
        network.store_patterns(
            np.array([domain_embedding]),
            [domain_name]
        )
        network.save()

        # 5. Sauvegarder tous les embeddings de domaines
        await self._save_domain_embeddings()

        print(f"✓ Domaine '{domain_name}' ajouté au niveau {domain_level}")
        if parent_domain:
            print(f"  Parent: {parent_domain}")

        return True

    async def _build_domain_hierarchy(self):
        """Analyse l'ontologie et construit la hiérarchie des domaines."""
        # Réinitialiser les structures
        self.domain_hierarchy = {}
        self.domain_to_level = {}
        self.level_to_domains = {}

        # Parcourir tous les domaines dans l'ontologie
        for name, domain in self.ontology_manager.domains.items():
            # Initialiser les structures pour ce domaine
            if name not in self.domain_hierarchy:
                self.domain_hierarchy[name] = []

            # Traiter les sous-domaines
            for subdomain in domain.subdomains:
                self.domain_hierarchy[name].append(subdomain.name)

                # S'assurer que le sous-domaine a une entrée
                if subdomain.name not in self.domain_hierarchy:
                    self.domain_hierarchy[subdomain.name] = []

        # Déterminer le niveau de chaque domaine
        for name in self.domain_hierarchy:
            level = self._calculate_domain_level(name)
            self.domain_to_level[name] = level

            if level not in self.level_to_domains:
                self.level_to_domains[level] = []

            self.level_to_domains[level].append(name)

            # Mettre à jour le niveau maximum
            self.max_level = max(self.max_level, level)

    def _calculate_domain_level(self, domain_name: str, visited=None) -> int:
        """Calcule le niveau hiérarchique d'un domaine."""
        if visited is None:
            visited = set()

        if domain_name in visited:
            # Cycle détecté, considérer comme niveau 1
            return 1

        visited.add(domain_name)

        # Récupérer le domaine
        domain = self.ontology_manager.domains.get(domain_name)
        if not domain:
            return 1

        # Si pas de parent, c'est un domaine de niveau 1
        if not domain.parent_domain:
            return 1

        # Sinon, c'est le niveau du parent + 1
        parent_level = self._calculate_domain_level(domain.parent_domain.name, visited)
        return parent_level + 1

    async def _load_domain_embeddings(self):
        """Charge les embeddings de domaines existants."""
        embeddings_path = os.path.join(self.storage_dir, "domain_embeddings.pkl")

        if os.path.exists(embeddings_path):
            try:
                with open(embeddings_path, 'rb') as f:
                    self.domain_embeddings = pickle.load(f)
                print(f"✓ Embeddings de {len(self.domain_embeddings)} domaines chargés")
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement des embeddings de domaines: {e}")
                self.domain_embeddings = {}

    async def _save_domain_embeddings(self):
        """Sauvegarde les embeddings des domaines."""
        embeddings_path = os.path.join(self.storage_dir, "domain_embeddings.pkl")

        try:
            with open(embeddings_path, 'wb') as f:
                pickle.dump(self.domain_embeddings, f)
        except Exception as e:
            print(f"⚠️ Erreur lors de la sauvegarde des embeddings de domaines: {e}")

    async def create_domain_for_level(
            self,
            domain_name: str,
            document_embeddings: List[Tuple[str, np.ndarray]],
            parent_domain: str = None
    ) -> bool:
        """
        Crée un nouveau domaine et l'intègre dans la hiérarchie de classification.

        Args:
            domain_name: Nom du domaine à créer
            document_embeddings: Liste de tuples (document_id, embedding)
            parent_domain: Nom du domaine parent (None pour niveau 1)

        Returns:
            True si la création a réussi, False sinon
        """
        if not document_embeddings:
            return False

        # 1. Déterminer le niveau du domaine
        if parent_domain:
            # Récupérer le niveau du parent
            parent_level = self.domain_to_level.get(parent_domain)
            if parent_level is None:
                print(f"⚠️ Domaine parent '{parent_domain}' non trouvé")
                return False

            # Niveau de ce domaine = niveau du parent + 1
            domain_level = parent_level + 1

            # Mettre à jour la hiérarchie
            if parent_domain not in self.domain_hierarchy:
                self.domain_hierarchy[parent_domain] = []
            self.domain_hierarchy[parent_domain].append(domain_name)
        else:
            # Sans parent, c'est un domaine de niveau 1
            domain_level = 1

        # Initialiser l'entrée du domaine dans la hiérarchie
        self.domain_hierarchy[domain_name] = []
        self.domain_to_level[domain_name] = domain_level

        if domain_level not in self.level_to_domains:
            self.level_to_domains[domain_level] = []
        self.level_to_domains[domain_level].append(domain_name)

        self.max_level = max(self.max_level, domain_level)

        # 2. Calculer l'embedding représentatif du domaine
        # Moyenne des embeddings des documents
        domain_embedding = np.mean([emb for _, emb in document_embeddings], axis=0)
        # Normaliser
        domain_embedding = domain_embedding / np.linalg.norm(domain_embedding)

        # Stocker l'embedding du domaine
        self.domain_embeddings[domain_name] = domain_embedding

        # 3. S'assurer que le réseau pour ce niveau existe
        if domain_level not in self.level_networks:
            level_dir = os.path.join(self.storage_dir, f"level_{domain_level}")
            os.makedirs(level_dir, exist_ok=True)

            self.level_networks[domain_level] = ModernHopfieldNetwork(
                beta=BETA,
                normalize_patterns=NORMALIZED_PATTERN,
                storage_dir=level_dir
            )

        # 4. Ajouter l'embedding du domaine au réseau de son niveau
        network = self.level_networks[domain_level]
        network.store_patterns(
            np.array([domain_embedding]),
            [domain_name]
        )
        network.save()

        # 5. Sauvegarder tous les embeddings de domaines
        await self._save_domain_embeddings()

        print(f"✓ Domaine '{domain_name}' créé au niveau {domain_level}")
        if parent_domain:
            print(f"  Parent: {parent_domain}")

        # 6. Associer les documents au domaine dans l'ontologie
        for doc_id, _ in document_embeddings:
            self.ontology_manager.associate_document_with_domain(doc_id, domain_name, 1.0)

        return True

    async def classify_document(self, document_embedding: np.ndarray, top_k: int = TOP_K_CONCEPT) -> List[Dict[str, Any]]:
        """
        Classifie un document en cascade à travers les niveaux hiérarchiques.

        Args:
            document_embedding: Embedding du document
            top_k: Nombre maximal de résultats par niveau

        Returns:
            Liste des résultats de classification avec structure hiérarchique
        """
        if not self.level_networks or 1 not in self.level_networks:
            return []

        # Normaliser l'embedding du document
        if np.linalg.norm(document_embedding) > 0:
            document_embedding = document_embedding / np.linalg.norm(document_embedding)

        # Résultats finaux
        final_results = []

        # Commencer par le niveau 1
        level1_results = await self._classify_at_level(document_embedding, 1, top_k)

        # Pour chaque domaine de niveau 1 détecté
        for domain_result in level1_results:
            domain_name = domain_result["domain"]
            domain_confidence = domain_result["confidence"]

            # Structure pour ce résultat
            hierarchical_result = {
                "domain": domain_name,
                "confidence": domain_confidence,
                "hierarchy": [domain_name],
                "sub_domains": []
            }

            # Récursivement explorer les sous-niveaux
            await self._classify_recursive(
                document_embedding,
                domain_name,
                hierarchical_result,
                level=2,
                top_k=top_k
            )

            final_results.append(hierarchical_result)

        return final_results

    async def _classify_at_level(
            self,
            document_embedding: np.ndarray,
            level: int,
            top_k: int = TOP_K_CONCEPT,
            parent_domain: str = None,
            confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Classifie un document à un niveau spécifique de la hiérarchie.

        Args:
            document_embedding: Embedding du document
            level: Niveau hiérarchique (1, 2, 3, ...)
            top_k: Nombre maximum de résultats
            parent_domain: Domaine parent (pour filtrer les résultats)
            confidence_threshold: Seuil minimal de confiance

        Returns:
            Liste des domaines les plus probables à ce niveau
        """
        # Vérifier que le niveau existe
        if level not in self.level_networks:
            return []

        # Récupérer le réseau pour ce niveau
        network = self.level_networks[level]

        # Évaluer l'embedding avec le réseau
        evaluation = network.evaluate_query(document_embedding)

        # Récupérer les patterns les plus proches
        closest_patterns = evaluation["closest_patterns"]

        # Filtrer les résultats
        results = []

        for pattern in closest_patterns:
            if "label" not in pattern:
                continue

            domain_name = pattern["label"]
            confidence = pattern["confidence"]

            # Vérifier si ce domaine est valide à ce niveau
            if domain_name not in self.domain_to_level or self.domain_to_level[domain_name] != level:
                continue

            # Si un parent est spécifié, vérifier que ce domaine est bien un enfant
            if parent_domain and (parent_domain not in self.domain_hierarchy or
                                  domain_name not in self.domain_hierarchy[parent_domain]):
                continue

            # Vérifier le seuil de confiance
            if confidence < confidence_threshold:
                continue

            results.append({
                "domain": domain_name,
                "confidence": confidence,
                "similarity": pattern["similarity"]
            })

        # Trier par confiance décroissante et limiter au top_k
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k]

    async def _classify_recursive(
            self,
            document_embedding: np.ndarray,
            parent_domain: str,
            parent_result: Dict[str, Any],
            level: int,
            top_k: int = TOP_K_CONCEPT,
            confidence_threshold: float = 0.5
    ):
        """
        Classification récursive à travers les niveaux hiérarchiques.

        Args:
            document_embedding: Embedding du document
            parent_domain: Domaine parent actuel
            parent_result: Dictionnaire du résultat parent à compléter
            level: Niveau actuel
            top_k: Nombre maximum de résultats par niveau
            confidence_threshold: Seuil minimal de confiance
        """
        # Vérifier si ce niveau existe
        if level > self.max_level:
            return

        # Vérifier si le parent a des enfants
        if parent_domain not in self.domain_hierarchy or not self.domain_hierarchy[parent_domain]:
            return

        # Classifier à ce niveau sous ce parent
        level_results = await self._classify_at_level(
            document_embedding,
            level,
            top_k,
            parent_domain,
            confidence_threshold
        )

        # Ajouter les résultats à la structure parent
        for result in level_results:
            domain_name = result["domain"]
            confidence = result["confidence"]

            # Créer la structure pour ce sous-domaine
            sub_result = {
                "domain": domain_name,
                "confidence": confidence,
                "hierarchy": parent_result["hierarchy"] + [domain_name],
                "sub_domains": []
            }

            # Récursivement explorer les niveaux inférieurs
            await self._classify_recursive(
                document_embedding,
                domain_name,
                sub_result,
                level + 1,
                top_k,
                confidence_threshold
            )

            # Ajouter au parent
            parent_result["sub_domains"].append(sub_result)