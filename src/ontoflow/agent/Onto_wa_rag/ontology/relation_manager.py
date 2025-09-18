"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# ontology/relation_manager.py
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import hashlib
from dataclasses import dataclass
from collections import defaultdict

from CONSTANT import RELATION_CONFIDENCE


class RModernHopfield(nn.Module):
    """Réseau de Hopfield moderne avec attention."""

    def __init__(self, input_dim: int, hidden_dim: int = None,
                 beta: float = 1.0, normalize_patterns: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim * 4
        self.beta = beta
        self.normalize_patterns = normalize_patterns

        # Projections pour query, key, value
        self.query_proj = nn.Linear(input_dim, self.hidden_dim)
        self.key_proj = nn.Linear(input_dim, self.hidden_dim)
        self.value_proj = nn.Linear(input_dim, self.hidden_dim)

        # IMPORTANT: Projection de sortie pour ramener à input_dim
        self.output_proj = nn.Linear(self.hidden_dim, input_dim)

        # Mémoire associative
        self.memory_bank = []
        self.memory_keys = None
        self.memory_values = None

    def store(self, patterns: torch.Tensor):
        """Stocke des patterns dans la mémoire."""
        with torch.no_grad():
            # S'assurer que patterns a au moins 2 dimensions
            if patterns.dim() == 1:
                patterns = patterns.unsqueeze(0)

            keys = self.key_proj(patterns)
            values = self.value_proj(patterns)

            if self.normalize_patterns:
                keys = F.normalize(keys, dim=-1)
                values = F.normalize(values, dim=-1)

            self.memory_bank.append((keys, values))

            # Mettre à jour la mémoire consolidée
            all_keys = torch.cat([k for k, _ in self.memory_bank], dim=0)
            all_values = torch.cat([v for _, v in self.memory_bank], dim=0)
            self.memory_keys = all_keys
            self.memory_values = all_values

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """Récupère un pattern depuis la mémoire."""
        if self.memory_keys is None:
            return torch.zeros_like(query)

        # S'assurer que query a au moins 2 dimensions
        if query.dim() == 1:
            query = query.unsqueeze(0)

        # Projeter la requête
        q = self.query_proj(query)
        if self.normalize_patterns:
            q = F.normalize(q, dim=-1)

        # Calculer les scores d'attention
        # Utiliser transpose au lieu de .T pour éviter l'avertissement
        scores = torch.matmul(q, self.memory_keys.transpose(-2, -1)) * self.beta
        attention_weights = F.softmax(scores, dim=-1)

        # Récupérer la valeur pondérée
        retrieved = torch.matmul(attention_weights, self.memory_values)

        # IMPORTANT: Projeter de retour à input_dim
        retrieved = self.output_proj(retrieved)

        # Si l'entrée était 1D, retourner 1D
        if retrieved.size(0) == 1:
            retrieved = retrieved.squeeze(0)

        return retrieved


class NonLinearRelationSpace(nn.Module):
    """Espace de relations non-linéaires avec projections multi-têtes."""

    def __init__(self, concept_dim: int, relation_dim: int, n_heads: int = 8):
        super().__init__()
        self.concept_dim = concept_dim
        self.relation_dim = relation_dim
        self.n_heads = n_heads

        # Projections multi-têtes pour capturer différents aspects
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(concept_dim, relation_dim),
                nn.LayerNorm(relation_dim),
                nn.GELU()  # Non-linéarité
            ) for _ in range(n_heads)
        ])

        # Hopfield pour chaque type de projection
        self.hopfield_heads = nn.ModuleList([
            RModernHopfield(relation_dim, beta=1.0)
            for _ in range(n_heads)
        ])

        # Fusion des têtes
        self.fusion = nn.Sequential(
            nn.Linear(n_heads * relation_dim, relation_dim),
            nn.LayerNorm(relation_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )


class ContextualRelationComposer(nn.Module):
    """Compose S et O en tenant compte du contexte relationnel."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Attention pour pondérer l'importance de S vs O
        self.cross_attention = nn.MultiheadAttention(dim, 8, batch_first=True)

        # Réseau de Hopfield pour le contexte
        self.context_hopfield = RModernHopfield(
            input_dim=dim,
            hidden_dim=4 * dim,  # Plus grande capacité
            beta=0.1  # Plus "soft" pour la généralisation
        )

        # Transformation pour la composition
        self.composition_net = nn.Sequential(
            nn.Linear(4 * dim, 2 * dim),  # 4*dim car on concatène plusieurs features
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, E_s: torch.Tensor, E_o: torch.Tensor, context: Optional[torch.Tensor] = None):
        """Compose les embeddings du sujet et de l'objet."""
        # S'assurer que les tenseurs ont la bonne forme pour l'attention
        if E_s.dim() == 1:
            E_s = E_s.unsqueeze(0).unsqueeze(0)
        elif E_s.dim() == 2:
            E_s = E_s.unsqueeze(0)

        if E_o.dim() == 1:
            E_o = E_o.unsqueeze(0).unsqueeze(0)
        elif E_o.dim() == 2:
            E_o = E_o.unsqueeze(0)

        # Attention croisée entre S et O
        attended_s, _ = self.cross_attention(E_s, E_o, E_o)
        attended_o, _ = self.cross_attention(E_o, E_s, E_s)

        # Aplatir pour la suite - gérer correctement les dimensions
        if attended_s.dim() == 3:
            attended_s = attended_s.squeeze(0)
        if attended_o.dim() == 3:
            attended_o = attended_o.squeeze(0)

        # S'assurer que les tenseurs sont 1D pour la concaténation
        if attended_s.dim() > 1:
            attended_s = attended_s.flatten()
        if attended_o.dim() > 1:
            attended_o = attended_o.flatten()

        # Combinaison non-linéaire
        combined = torch.cat([
            attended_s,
            attended_o,
            attended_s * attended_o,  # Interaction multiplicative
            torch.abs(attended_s - attended_o)  # Distance
        ], dim=-1)

        # Appliquer la transformation de composition
        composed = self.composition_net(combined)

        # Si contexte fourni, l'intégrer via Hopfield
        if context is not None:
            self.context_hopfield.store(context.unsqueeze(0) if context.dim() == 1 else context)
            contextual_memory = self.context_hopfield.retrieve(composed)
            composed = composed + 0.5 * contextual_memory

        return composed


class HierarchicalRelationMemory(nn.Module):
    """Mémoire relationnelle hiérarchique."""

    def __init__(self, base_dim: int = 256):
        super().__init__()
        self.base_dim = base_dim

        # Niveau 1: Relations atomiques - utilise base_dim
        self.atomic_hopfield = RModernHopfield(
            input_dim=base_dim,
            hidden_dim=base_dim * 2,  # Hidden plus petit
            beta=1.0,
            normalize_patterns=True
        )

        # Niveau 2: Méta-relations
        self.meta_hopfield = RModernHopfield(
            input_dim=base_dim * 2,
            hidden_dim=base_dim * 4,
            beta=0.5,
            normalize_patterns=True
        )

        # Niveau 3: Patterns relationnels
        self.pattern_hopfield = RModernHopfield(
            input_dim=base_dim * 4,
            hidden_dim=base_dim * 8,
            beta=0.1,
            normalize_patterns=True
        )

        # Projecteurs entre niveaux
        self.to_meta = nn.Linear(base_dim, base_dim * 2)
        self.to_pattern = nn.Linear(base_dim * 2, base_dim * 4)
        self.from_pattern = nn.Linear(base_dim * 4, base_dim)


class RelationalInference(nn.Module):
    """Système d'inférence complet pour les relations."""

    def __init__(self, concept_dim: int = 768, relation_dim: int = 256):
        super().__init__()
        self.concept_dim = concept_dim
        self.relation_dim = relation_dim

        # Composants de l'architecture
        self.relation_space = NonLinearRelationSpace(concept_dim, relation_dim)
        self.composer = ContextualRelationComposer(relation_dim)
        self.memory = HierarchicalRelationMemory(base_dim=relation_dim)

        # Réseau de raisonnement
        self.reasoning_network = nn.Sequential(
            nn.Linear(relation_dim, relation_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(relation_dim * 2, relation_dim),
            nn.LayerNorm(relation_dim)
        )

        # Tête de confiance
        self.confidence_head = nn.Sequential(
            nn.Linear(relation_dim, relation_dim // 2),
            nn.LayerNorm(relation_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(relation_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Initialisation des poids
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def infer_relation(self, E_s: torch.Tensor, E_o: torch.Tensor,
                       context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Infère la relation entre S et O."""
        # 1. Projection dans l'espace relationnel via les têtes
        projections = []
        projected_s_list = []
        projected_o_list = []

        for i, (proj, hopfield) in enumerate(
                zip(self.relation_space.projections,
                    self.relation_space.hopfield_heads)
        ):
            # Projeter de concept_dim (3072) vers relation_dim (512)
            h_s = proj(E_s)
            h_o = proj(E_o)

            # Sauvegarder les projections pour utilisation ultérieure
            projected_s_list.append(h_s)
            projected_o_list.append(h_o)

            # Requête au Hopfield spécifique
            query = h_s + h_o + (h_s * h_o)
            retrieved = hopfield.retrieve(query)
            projections.append(retrieved)

        # 2. Agrégation des projections
        multi_view = torch.stack(projections, dim=0)
        aggregated = self.relation_space.fusion(multi_view.flatten())

        # 3. Composition contextuelle - utiliser les embeddings projetés (moyennés)
        # Prendre la moyenne des projections pour avoir une représentation unique
        projected_s = torch.mean(torch.stack(projected_s_list), dim=0)
        projected_o = torch.mean(torch.stack(projected_o_list), dim=0)

        # Si contexte fourni, le projeter aussi
        projected_context = None
        if context is not None:
            # Utiliser la première projection pour le contexte
            projected_context = self.relation_space.projections[0](context)

        # Maintenant passer les embeddings projetés (dimension relation_dim) au composer
        composed = self.composer(projected_s, projected_o, projected_context)

        # 4. Requête hiérarchique dans la mémoire
        # D'abord essayer les relations atomiques
        atomic_result = self.memory.atomic_hopfield.retrieve(composed)
        atomic_confidence = self.confidence_head(atomic_result)  # Ne pas appeler .item()

        # Si pas de match fort, essayer les méta-relations
        if atomic_confidence < 0.8:
            meta_query = self.memory.to_meta(composed)
            meta_result = self.memory.meta_hopfield.retrieve(meta_query)

            # Fusionner les résultats
            result = 0.7 * atomic_result + 0.3 * self.memory.from_pattern(
                self.memory.to_pattern(meta_result)
            )
            confidence = 0.7 * atomic_confidence + 0.3
        else:
            result = atomic_result
            confidence = atomic_confidence

        # 5. Raisonnement final
        reasoned = self.reasoning_network(result + aggregated)

        return reasoned, confidence  # confidence est un tenseur


class RelationTransformation:
    """Transformation de relation utilisant l'architecture Hopfield moderne."""

    def __init__(self, relation_uri: str, label: str = None,
                 concept_dim: int = 768, relation_dim: int = 256):
        """Initialise une transformation avec Hopfield moderne."""
        self.uri = relation_uri
        self.label = label or self._extract_label_from_uri(relation_uri)
        self.concept_dim = concept_dim
        self.relation_dim = relation_dim

        # Modèle d'inférence
        self.model = RelationalInference(concept_dim, relation_dim)

        # Stockage des exemples et embeddings
        self.positive_examples = []
        self.negative_examples = []
        self.concept_embeddings = {}

        # Propriétés de l'ontologie
        self.domain_concept_uris = []
        self.range_concept_uris = []
        self.properties = {
            "transitive": False,
            "symmetric": False,
            "asymmetric": False,
            "reflexive": False,
            "irreflexive": False,
            "functional": False,
            "inverse_functional": False
        }

        # Métadonnées d'entraînement
        self.inverse_relation_uri = None
        self.examples_count = 0
        self.examples_hash = None
        self.last_trained = None
        self.training_loss = None
        self.is_trained = False

        # Configuration d'entraînement
        self.num_epochs = 500
        self.learning_rate = 0.0005
        self.batch_size = 8

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _extract_label_from_uri(self, uri: str) -> str:
        """Extrait un label lisible à partir de l'URI."""
        if '#' in uri:
            return uri.split('#')[-1]
        return uri.split('/')[-1]

    def learn_from_triples(self, triples: List[Tuple[str, str, str]],
                           concept_embeddings: Dict[str, np.ndarray],
                           negative_sampling_ratio: float = 1.0,
                           validation_triples: Optional[List[Tuple[str, str, str]]] = None) -> bool:
        """Apprend à partir de triplets avec l'architecture Hopfield."""
        try:
            # Stocker les embeddings de concepts
            self.concept_embeddings.update(concept_embeddings)

            # Préparer les exemples positifs
            self.positive_examples = [(s, o) for s, r, o in triples if r == self.uri]

            if len(self.positive_examples) < 3:
                print(f"⚠️ Pas assez d'exemples pour {self.label}: {len(self.positive_examples)}")
                return False

            # Générer des exemples négatifs
            all_subjects = set(s for s, _ in self.positive_examples)
            all_objects = set(o for _, o in self.positive_examples)

            self.negative_examples = []
            for _ in range(int(len(self.positive_examples) * negative_sampling_ratio)):
                neg_s = np.random.choice(list(all_subjects))
                neg_o = np.random.choice(list(all_objects))
                if (neg_s, neg_o) not in self.positive_examples:
                    self.negative_examples.append((neg_s, neg_o))

            # Entraîner le modèle
            print(f"🔄 Entraînement Hopfield pour {self.label} ({len(self.positive_examples)} exemples positifs)...")

            # Optimiseur avec momentum et weight decay
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01,  # Régularisation L2
                betas=(0.9, 0.999)
            )

            # Scheduler pour ajuster le learning rate
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.num_epochs,
                eta_min=self.learning_rate * 0.01
            )

            losses = []
            self.model.train()

            # Stocker d'abord quelques patterns positifs dans la mémoire
            print("📝 Initialisation de la mémoire avec exemples positifs...")
            for i, (s, o) in enumerate(self.positive_examples[:5]):
                if s in self.concept_embeddings and o in self.concept_embeddings:
                    E_s = torch.tensor(self.concept_embeddings[s]).float().to(self.device)
                    E_o = torch.tensor(self.concept_embeddings[o]).float().to(self.device)
                    with torch.no_grad():
                        rel_pattern, _ = self.model.infer_relation(E_s, E_o)
                        self.model.memory.atomic_hopfield.store(rel_pattern.unsqueeze(0))

            for epoch in range(self.num_epochs):
                epoch_loss = 0.0
                n_correct = 0
                n_total = 0

                # Mélanger les exemples
                positive_batch = np.random.permutation(self.positive_examples)[:self.batch_size]
                negative_batch = np.random.permutation(self.negative_examples)[:self.batch_size]

                for (pos_s, pos_o), (neg_s, neg_o) in zip(positive_batch, negative_batch):
                    # Obtenir les embeddings
                    E_pos_s = torch.tensor(self.concept_embeddings.get(pos_s, np.zeros(self.concept_dim))).float().to(
                        self.device)
                    E_pos_o = torch.tensor(self.concept_embeddings.get(pos_o, np.zeros(self.concept_dim))).float().to(
                        self.device)
                    E_neg_s = torch.tensor(self.concept_embeddings.get(neg_s, np.zeros(self.concept_dim))).float().to(
                        self.device)
                    E_neg_o = torch.tensor(self.concept_embeddings.get(neg_o, np.zeros(self.concept_dim))).float().to(
                        self.device)

                    # Inférer les relations
                    pos_relation, pos_conf = self.model.infer_relation(E_pos_s, E_pos_o)
                    neg_relation, neg_conf = self.model.infer_relation(E_neg_s, E_neg_o)

                    # Loss contrastive corrigée (triplet loss)
                    margin = 0.5
                    loss = torch.relu(margin - (pos_conf - neg_conf))

                    # Ajouter une loss BCE pour forcer les confidences vers 0/1
                    bce_loss = F.binary_cross_entropy(pos_conf, torch.ones_like(pos_conf)) + \
                               F.binary_cross_entropy(neg_conf, torch.zeros_like(neg_conf))

                    # Combiner les losses
                    total_loss = loss + 0.5 * bce_loss

                    # Régularisation pour éviter l'effondrement
                    if epoch % 10 == 0:
                        # Diversité des embeddings
                        div_loss = -torch.log(torch.std(pos_relation) + 1e-8)
                        total_loss += 0.1 * div_loss

                    # Backpropagation
                    optimizer.zero_grad()
                    total_loss.backward()

                    # Gradient clipping pour stabilité
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    optimizer.step()

                    # Stocker occasionnellement les patterns positifs
                    if np.random.rand() < 0.1:  # 10% du temps
                        with torch.no_grad():
                            self.model.memory.atomic_hopfield.store(pos_relation.unsqueeze(0))

                    epoch_loss += total_loss.item()

                    # Calculer l'accuracy
                    with torch.no_grad():
                        n_correct += (pos_conf > 0.5).float().sum().item()
                        n_correct += (neg_conf < 0.5).float().sum().item()
                        n_total += 2

                # Calculer les moyennes pour l'epoch
                avg_loss = epoch_loss / min(len(positive_batch), len(negative_batch))
                accuracy = n_correct / n_total if n_total > 0 else 0
                losses.append(avg_loss)

                # Mise à jour du scheduler (une fois par epoch, pas par batch!)
                scheduler.step()

                if epoch % 20 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"  Epoch {epoch}: Loss = {avg_loss:.4f}, Acc = {accuracy:.2f}, LR = {current_lr:.6f}")

                    # Debug: afficher des exemples de confidences
                    with torch.no_grad():
                        # Prendre le dernier exemple traité
                        print(f"    Dernier exemple - Pos conf: {pos_conf.item():.3f}, Neg conf: {neg_conf.item():.3f}")

            # Sauvegarder les métadonnées
            self.training_loss = float(losses[-1]) if losses else None
            self.last_trained = datetime.now().isoformat()
            self.examples_count = len(self.positive_examples)
            self.is_trained = True

            print(f"✓ Modèle Hopfield entraîné pour {self.label} (loss finale: {self.training_loss:.4f})")
            return True

        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement pour {self.label}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict_objects(self, subject: str, concept_embeddings: Dict[str, np.ndarray],
                        top_k: int = 10, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Prédit les objets les plus probables pour un sujet donné."""
        if not self.is_trained or subject not in concept_embeddings:
            return []

        try:
            self.model.eval()
            results = []

            E_s = torch.tensor(concept_embeddings[subject]).float().to(self.device)

            # Tester tous les objets possibles
            for obj_uri, obj_embedding in concept_embeddings.items():
                if obj_uri != subject or self.properties.get("reflexive", False):
                    E_o = torch.tensor(obj_embedding).float().to(self.device)

                    with torch.no_grad():
                        _, confidence = self.model.infer_relation(E_s, E_o)

                        # Convertir en float si c'est un tenseur
                        if torch.is_tensor(confidence):
                            conf_value = confidence.item()
                        else:
                            conf_value = float(confidence)

                        # Ajouter une pénalité pour les objets qui n'étaient pas dans les exemples positifs
                        # pour éviter que tous aient la même confiance
                        if (subject, obj_uri) not in self.positive_examples:
                            conf_value *= 0.8  # Réduire la confiance pour les paires non vues

                    if conf_value >= threshold:
                        results.append({
                            "concept_uri": obj_uri,
                            "label": obj_uri.split('/')[-1],
                            "similarity": float(conf_value),
                            "confidence": float(conf_value)
                        })

            # Trier par confiance décroissante
            results.sort(key=lambda x: x["confidence"], reverse=True)
            return results[:top_k]

        except Exception as e:
            print(f"⚠️ Erreur lors de la prédiction pour {self.label}: {e}")
            import traceback
            traceback.print_exc()
            return []


class RelationManager:
    """Gestionnaire de relations utilisant l'architecture Hopfield moderne."""

    def __init__(self, ontology_manager, storage_dir: str = "hopfield_models"):
        """Initialise le gestionnaire avec Hopfield moderne."""
        self.ontology_manager = ontology_manager
        self.storage_dir = storage_dir
        self.transformations = {}  # URI -> RelationTransformation
        self.global_inference_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Configuration globale
        self.concept_dim = 768  # Dimension standard des embeddings
        self.relation_dim = 256

        # Créer le répertoire de stockage
        os.makedirs(storage_dir, exist_ok=True)
        print(f"🚀 Architecture Hopfield utilisant le device: {self.device}")

    async def initialize(self, concept_dim: int = 768, relation_dim: int = 256):
        """Initialise le gestionnaire en créant les transformations."""
        print("Initialisation du gestionnaire avec architecture Hopfield moderne...")

        self.concept_dim = concept_dim
        self.relation_dim = relation_dim

        # Charger ou créer les transformations
        if not await self._load_transformations():
            for uri, relation in self.ontology_manager.relations.items():
                transform = RelationTransformation(
                    uri, relation.label, self.concept_dim, self.relation_dim  # Utiliser les dimensions correctes
                )

                # Ajouter les concepts de domaine et de portée
                for domain_concept in relation.domain:
                    if hasattr(domain_concept, 'uri'):
                        transform.domain_concept_uris.append(domain_concept.uri)

                for range_concept in relation.range:
                    if hasattr(range_concept, 'uri'):
                        transform.range_concept_uris.append(range_concept.uri)

                self.transformations[uri] = transform

            print(f"✓ {len(self.transformations)} transformations Hopfield créées")

        # Analyser les propriétés des relations
        self._analyze_relation_properties()

        # Initialiser le modèle global avec les bonnes dimensions
        self.global_inference_model = RelationalInference(
            self.concept_dim, self.relation_dim  # Utiliser les dimensions stockées
        ).to(self.device)

    async def train_global_model(self, all_triples: List[Tuple[str, str, str]],
                                 concept_embeddings: Dict[str, np.ndarray],
                                 num_epochs: int = 200) -> bool:
        """Entraîne un modèle global sur tous les triplets."""
        if not all_triples:
            print("⚠️ Aucun triplet fourni pour l'entraînement global")
            return False

        try:
            print(f"🔄 Entraînement du modèle global Hopfield avec {len(all_triples)} triplets...")

            # S'assurer que le modèle global existe et a les bonnes dimensions
            if self.global_inference_model is None:
                self.global_inference_model = RelationalInference(
                    self.concept_dim, self.relation_dim
                ).to(self.device)

            # Vérifier les dimensions des embeddings
            first_embedding = next(iter(concept_embeddings.values()))
            if first_embedding.shape[0] != self.concept_dim:
                print(f"⚠️ Dimension mismatch: expected {self.concept_dim}, got {first_embedding.shape[0]}")
                # Recréer le modèle avec les bonnes dimensions
                self.concept_dim = first_embedding.shape[0]
                self.global_inference_model = RelationalInference(
                    self.concept_dim, self.relation_dim
                ).to(self.device)

            optimizer = torch.optim.Adam(
                self.global_inference_model.parameters(),
                lr=0.001
            )

            # Organiser les triplets par relation
            triples_by_relation = defaultdict(list)
            for s, r, o in all_triples:
                triples_by_relation[r].append((s, o))

            self.global_inference_model.train()

            for epoch in range(num_epochs):
                epoch_loss = 0.0
                n_batches = 0

                for relation_uri, examples in triples_by_relation.items():
                    if len(examples) < 2:
                        continue

                    # Échantillonner des exemples
                    batch_size = min(32, len(examples))
                    batch = np.random.choice(len(examples), batch_size, replace=False)

                    for idx in batch:
                        s, o = examples[idx]

                        if s not in concept_embeddings or o not in concept_embeddings:
                            continue

                        E_s = torch.tensor(concept_embeddings[s]).float().to(self.device)
                        E_o = torch.tensor(concept_embeddings[o]).float().to(self.device)

                        # Inférer et stocker dans la mémoire globale
                        relation_emb, conf = self.global_inference_model.infer_relation(E_s, E_o)

                        # Stocker le pattern
                        with torch.no_grad():
                            self.global_inference_model.memory.atomic_hopfield.store(
                                relation_emb.unsqueeze(0)
                            )

                        # Loss basée sur la confiance (on veut maximiser la confiance)
                        loss = -torch.log(conf + 1e-8)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        n_batches += 1

                if n_batches > 0 and epoch % 20 == 0:
                    print(f"  Epoch {epoch}: Loss moyenne = {epoch_loss / n_batches:.4f}")

            print(f"✓ Modèle global Hopfield entraîné")
            await self._save_global_model()
            return True

        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement global: {e}")
            return False

    async def learn_relation_transformation(self, relation_uri: str,
                                            examples: List[Tuple[str, str]],
                                            concept_embeddings: Dict[str, np.ndarray],
                                            force_relearn: bool = False,
                                            min_examples_threshold: int = 5) -> bool:
        """Apprend une transformation de relation spécifique."""
        if relation_uri not in self.transformations:
            print(f"⚠️ Relation {relation_uri} non trouvée")
            return False

        transform = self.transformations[relation_uri]

        # Vérifier si déjà appris
        if not force_relearn and transform.is_trained and transform.examples_count >= min_examples_threshold:
            print(f"✓ Transformation pour {transform.label} déjà apprise")
            return True

        if len(examples) < min_examples_threshold:
            print(f"⚠️ Pas assez d'exemples pour {transform.label}: {len(examples)}")
            return False

        # Convertir les exemples en triplets
        triples = [(subj, relation_uri, obj) for subj, obj in examples]

        # Apprendre avec l'architecture Hopfield
        success = transform.learn_from_triples(triples, concept_embeddings)

        if success:
            transform.examples_hash = self._compute_examples_hash(examples)
            await self._save_transformations()

        return success

    def get_related_concepts(self, concept_uri: str, relation_uri: str,
                             concept_embeddings: Dict[str, np.ndarray],
                             top_k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Trouve les concepts liés via l'architecture Hopfield."""
        if relation_uri not in self.transformations:
            return []

        transform = self.transformations[relation_uri]
        return transform.predict_objects(concept_uri, concept_embeddings, top_k, threshold)

    def _analyze_relation_properties(self):
        """Analyse les propriétés logiques des relations dans l'ontologie."""
        properties_found = 0
        for uri, transform in self.transformations.items():
            for axiom_type, source, target in self.ontology_manager.axioms:
                if source == uri:
                    if axiom_type == "transitive_property":
                        transform.properties["transitive"] = True
                        properties_found += 1
                    elif axiom_type == "symmetric_property":
                        transform.properties["symmetric"] = True
                        properties_found += 1
                    elif axiom_type == "asymmetric_property":
                        transform.properties["asymmetric"] = True
                        properties_found += 1
                    elif axiom_type == "reflexive_property":
                        transform.properties["reflexive"] = True
                        properties_found += 1
                    elif axiom_type == "irreflexive_property":
                        transform.properties["irreflexive"] = True
                        properties_found += 1
                    elif axiom_type == "functional_property":
                        transform.properties["functional"] = True
                        properties_found += 1
                    elif axiom_type == "inverse_functional_property":
                        transform.properties["inverse_functional"] = True
                        properties_found += 1
                    elif axiom_type == "inverse_of" and target in self.transformations:
                        transform.inverse_relation_uri = target
                        self.transformations[target].inverse_relation_uri = uri
                        properties_found += 1

        print(f"✓ {properties_found} propriétés logiques de relations identifiées")

    def _compute_examples_hash(self, examples: List[Tuple[str, str]]) -> str:
        """Calcule un hash des exemples."""
        examples_str = str(sorted(examples))
        return hashlib.md5(examples_str.encode()).hexdigest()

    async def _save_transformations(self):
        """Sauvegarde les transformations."""
        transform_path = os.path.join(self.storage_dir, "hopfield_transformations.pkl")
        try:
            # Sauvegarder les métadonnées et les modèles
            serializable_data = {}
            for uri, transform in self.transformations.items():
                # Sauvegarder le modèle PyTorch
                model_path = os.path.join(self.storage_dir, f"model_{hashlib.md5(uri.encode()).hexdigest()}.pt")
                torch.save(transform.model.state_dict(), model_path)

                serializable_data[uri] = {
                    'uri': transform.uri,
                    'label': transform.label,
                    'concept_dim': transform.concept_dim,
                    'relation_dim': transform.relation_dim,
                    'domain_concept_uris': transform.domain_concept_uris,
                    'range_concept_uris': transform.range_concept_uris,
                    'properties': transform.properties,
                    'examples_count': transform.examples_count,
                    'examples_hash': transform.examples_hash,
                    'last_trained': transform.last_trained,
                    'training_loss': transform.training_loss,
                    'is_trained': transform.is_trained,
                    'positive_examples': transform.positive_examples,
                    'model_path': model_path
                }

            with open(transform_path, 'wb') as f:
                pickle.dump(serializable_data, f)

        except Exception as e:
            print(f"⚠️ Erreur lors de la sauvegarde: {e}")

    async def _load_transformations(self) -> bool:
        """Charge les transformations sauvegardées."""
        transform_path = os.path.join(self.storage_dir, "hopfield_transformations.pkl")
        if os.path.exists(transform_path):
            try:
                with open(transform_path, 'rb') as f:
                    data = pickle.load(f)

                for uri, transform_data in data.items():
                    transform = RelationTransformation(
                        transform_data['uri'],
                        transform_data['label'],
                        transform_data.get('concept_dim', self.concept_dim),
                        transform_data.get('relation_dim', self.relation_dim)
                    )

                    # Restaurer les propriétés
                    transform.domain_concept_uris = transform_data['domain_concept_uris']
                    transform.range_concept_uris = transform_data['range_concept_uris']
                    transform.properties = transform_data['properties']
                    transform.examples_count = transform_data['examples_count']
                    transform.examples_hash = transform_data.get('examples_hash')
                    transform.last_trained = transform_data.get('last_trained')
                    transform.training_loss = transform_data.get('training_loss')
                    transform.is_trained = transform_data.get('is_trained', False)
                    transform.positive_examples = transform_data.get('positive_examples', [])

                    # Charger le modèle si disponible
                    model_path = transform_data.get('model_path')
                    if model_path and os.path.exists(model_path):
                        transform.model.load_state_dict(torch.load(model_path, map_location=self.device))

                    self.transformations[uri] = transform

                print(f"✓ {len(self.transformations)} transformations Hopfield chargées")
                return True

            except Exception as e:
                print(f"⚠️ Erreur lors du chargement: {e}")

        return False

    async def _save_global_model(self):
        """Sauvegarde le modèle global."""
        if self.global_inference_model:
            model_path = os.path.join(self.storage_dir, "global_hopfield_model.pt")
            try:
                torch.save(self.global_inference_model.state_dict(), model_path)
            except Exception as e:
                print(f"⚠️ Erreur lors de la sauvegarde du modèle global: {e}")

    async def _load_global_model(self):
        """Charge le modèle global."""
        model_path = os.path.join(self.storage_dir, "global_hopfield_model.pt")
        if os.path.exists(model_path):
            try:
                self.global_inference_model = RelationalInference(
                    self.concept_dim, self.relation_dim
                ).to(self.device)
                self.global_inference_model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                return True
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement du modèle global: {e}")
        return False