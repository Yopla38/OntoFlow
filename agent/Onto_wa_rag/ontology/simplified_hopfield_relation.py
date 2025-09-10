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
import random

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import hashlib
from dataclasses import dataclass
from collections import defaultdict
import json

from CONSTANT import RELATION_CONFIDENCE


class SimplifiedHopfield(nn.Module):
    """Réseau de Hopfield simplifié pour mémoire associative."""

    def __init__(self, dim: int, max_memories: int = 1000, temperature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.max_memories = max_memories
        self.temperature = temperature

        # Mémoire des patterns
        self.register_buffer('memory_bank', torch.zeros(max_memories, dim))
        self.register_buffer('memory_count', torch.tensor(0))
        self.register_buffer('memory_usage', torch.zeros(max_memories, dtype=torch.bool))

    def store(self, pattern: torch.Tensor):
        """Stocke un pattern dans la mémoire."""
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)

        batch_size = pattern.size(0)

        # Trouver les emplacements libres
        free_indices = torch.where(~self.memory_usage)[0]

        if len(free_indices) < batch_size:
            # Remplacer les plus anciens si nécessaire
            n_to_replace = batch_size - len(free_indices)
            indices_to_use = torch.cat([
                free_indices,
                torch.arange(n_to_replace)
            ])
        else:
            indices_to_use = free_indices[:batch_size]

        # Stocker les patterns
        self.memory_bank[indices_to_use] = F.normalize(pattern, dim=-1)
        self.memory_usage[indices_to_use] = True
        self.memory_count = torch.clamp(self.memory_count + batch_size, max=self.max_memories)

    def retrieve(self, query: torch.Tensor, top_k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Récupère les patterns les plus similaires."""
        if self.memory_count == 0:
            return torch.zeros_like(query), torch.zeros(1)

        # S'assurer que query est 2D
        if query.dim() == 1:
            query = query.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Normaliser la requête
        query_norm = F.normalize(query, dim=-1)

        # Calculer les similarités avec les mémoires actives
        active_memories = self.memory_bank[self.memory_usage]
        similarities = torch.matmul(query_norm, active_memories.t()) / self.temperature

        # Récupérer top-k
        k = min(top_k, active_memories.size(0))
        values, indices = torch.topk(similarities, k, dim=-1)

        # Pondérer les mémoires par leur similarité
        weights = F.softmax(values, dim=-1)  # [batch_size, k]

        # Récupérer et pondérer les mémoires
        # Utiliser einsum pour éviter les problèmes de dimensions
        selected_memories = active_memories[indices[0]]  # [k, dim]
        if selected_memories.dim() == 1:
            selected_memories = selected_memories.unsqueeze(0)

        # Calculer la moyenne pondérée
        retrieved = torch.einsum('bk,kd->bd', weights, selected_memories)  # [batch_size, dim]

        # Si l'entrée était 1D, retourner 1D
        if squeeze_output:
            retrieved = retrieved.squeeze(0)
            values = values.squeeze(0)

        return retrieved, values


class RelationTransformationNetwork(nn.Module):
    """Réseau pour apprendre les transformations de relations."""

    def __init__(self, concept_dim: int, relation_dim: int = 256):
        super().__init__()
        self.concept_dim = concept_dim
        self.relation_dim = relation_dim

        # Encodeur de relation (transforme le sujet)
        self.relation_encoder = nn.Sequential(
            nn.Linear(concept_dim, relation_dim * 2),
            nn.LayerNorm(relation_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),  # Augmenter le dropout
            nn.Linear(relation_dim * 2, relation_dim),
            nn.LayerNorm(relation_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # Connexion résiduelle si dimensions compatibles
        self.input_projection = nn.Linear(concept_dim, relation_dim)

        # Décodeur symétrique
        self.relation_decoder = nn.Sequential(
            nn.Linear(relation_dim, relation_dim * 2),
            nn.LayerNorm(relation_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(relation_dim * 2, concept_dim),
            nn.LayerNorm(concept_dim)
        )

        # Tête de confiance avec plus de capacité
        self.confidence_head = nn.Sequential(
            nn.Linear(relation_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Mémoire des prototypes de relation
        self.prototype_memory = SimplifiedHopfield(relation_dim, max_memories=100)

        # Paramètre apprenable pour le biais de la relation
        self.relation_bias = nn.Parameter(torch.zeros(relation_dim))

        # Paramètre de température pour calibrer la confiance
        # self.confidence_temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, subject_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transforme le sujet en objet prédit."""
        h = self.relation_encoder(subject_emb)
        h = h + self.relation_bias

        memory_pattern, similarity = self.prototype_memory.retrieve(h)

        # Fusion adaptative basée sur la similarité
        if similarity > 0.5:  # Si on trouve un pattern similaire en mémoire
            alpha = torch.sigmoid(similarity * 2.0)
            h = alpha * memory_pattern + (1 - alpha) * h

        # Confiance avec température
        confidence_logit = self.confidence_head(h)
        confidence = torch.sigmoid(confidence_logit)

        # Ajouter un biais minimum pour éviter les confidences trop faibles
        # confidence = 0.1 + 0.9 * confidence  # Confiance entre 0.1 et 1.0

        object_pred = self.relation_decoder(h)

        return object_pred, confidence

    def store_example(self, subject_emb: torch.Tensor, object_emb: torch.Tensor):
        """Stocke un exemple dans la mémoire."""
        with torch.no_grad():
            h = self.relation_encoder(subject_emb)
            # Stocker le pattern encodé + biais
            h_with_bias = h + self.relation_bias
            self.prototype_memory.store(h_with_bias)


class RelationTransformation:
    """Transformation simplifiée pour une relation spécifique."""

    def __init__(self, relation_uri: str, label: str = None,
                 concept_dim: int = 768, relation_dim: int = 256):
        self.uri = relation_uri
        self.label = label or self._extract_label_from_uri(relation_uri)
        self.concept_dim = concept_dim
        self.relation_dim = relation_dim

        # Modèle de transformation
        self.model = RelationTransformationNetwork(concept_dim, relation_dim)

        # Stockage des exemples
        self.positive_examples = []
        self.concept_embeddings = {}

        # Propriétés ontologiques
        self.domain_concept_uris = []
        self.range_concept_uris = []
        self.properties = {
            "transitive": False,
            "symmetric": False,
            "functional": False,
            "inverse_functional": False
        }

        # Métadonnées
        self.is_trained = False
        self.training_metrics = {}
        self.last_trained = None

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _extract_label_from_uri(self, uri: str) -> str:
        """Extrait un label lisible."""
        if '#' in uri:
            return uri.split('#')[-1]
        return uri.split('/')[-1]

    def old_learn_from_examples(self, examples: List[Tuple[str, str]],
                            concept_embeddings: Dict[str, np.ndarray],
                            num_epochs: int = None,
                            learning_rate: float = None) -> bool:
        """Apprend la transformation avec exemples négatifs."""

        n_examples = len(examples)

        # Ajustement automatique des hyperparamètres
        if num_epochs is None:
            if n_examples < 5:
                num_epochs = 500
            elif n_examples < 10:
                num_epochs = 400
            elif n_examples < 20:
                num_epochs = 300
            else:
                num_epochs = 200

        if learning_rate is None:
            if n_examples < 5:
                learning_rate = 0.0003
            elif n_examples < 10:
                learning_rate = 0.0005
            else:
                learning_rate = 0.001

        # Régularisation adaptative
        weight_decay = 0.1 if n_examples < 5 else 0.01

        try:
            print(f"🔄 Apprentissage de {self.label} avec {len(examples)} exemples...")
            print(f"   Hyperparamètres: epochs={num_epochs}, lr={learning_rate}, wd={weight_decay}")

            # Stocker les embeddings et exemples
            self.concept_embeddings = concept_embeddings
            self.positive_examples = examples

            # Générer exemples négatifs
            all_concepts = list(concept_embeddings.keys())
            self.negative_examples = self.generate_hard_negative_examples(
                self.positive_examples,
                all_concepts,
                concept_embeddings
            )
            print(f"   Généré {len(self.negative_examples)} exemples négatifs")

            # Préparer les données d'entraînement
            positive_data = []
            for s_uri, o_uri in examples:
                if s_uri in concept_embeddings and o_uri in concept_embeddings:
                    s_emb = torch.tensor(concept_embeddings[s_uri]).float()
                    o_emb = torch.tensor(concept_embeddings[o_uri]).float()
                    positive_data.append((s_emb, o_emb, True))

            negative_data = []
            for s_uri, o_uri in self.negative_examples[:len(examples) * 2]:
                if s_uri in concept_embeddings and o_uri in concept_embeddings:
                    s_emb = torch.tensor(concept_embeddings[s_uri]).float()
                    o_emb = torch.tensor(concept_embeddings[o_uri]).float()
                    negative_data.append((s_emb, o_emb, False))

            if len(positive_data) < 3:
                print(f"⚠️ Pas assez d'embeddings valides")
                return False

            # Combiner toutes les données
            all_data = positive_data + negative_data
            print(f"   Total: {len(positive_data)} positifs, {len(negative_data)} négatifs")

            # Créer l'optimiseur
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )

            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=30,
                min_lr=1e-6
            )

            # Variables pour early stopping
            best_loss = float('inf')
            best_accuracy = 0.0
            best_state = None
            patience = 50
            patience_counter = 0

            # Entraînement
            self.model.train()
            losses = []
            accuracies = []

            for epoch in range(num_epochs):
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0

                # Mélanger les données
                indices = torch.randperm(len(all_data))

                # Taille de batch adaptative
                batch_size = min(8, len(all_data))
                num_batches = 0

                # Traiter par mini-batches
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:min(i + batch_size, len(indices))]
                    batch_loss = 0.0

                    # Accumuler les gradients sur le batch
                    optimizer.zero_grad()

                    for idx in batch_indices:
                        s_emb, o_emb, is_positive = all_data[idx.item()]
                        s_emb = s_emb.to(self.device).unsqueeze(0)
                        o_emb = o_emb.to(self.device).unsqueeze(0)

                        # Forward pass
                        o_pred, confidence = self.model(s_emb)

                        # S'assurer que tout est bien dimensionné
                        o_pred = o_pred.squeeze(0) if o_pred.dim() > 1 else o_pred
                        confidence = confidence.squeeze() if confidence.dim() > 0 else confidence
                        o_emb_squeezed = o_emb.squeeze(0) if o_emb.dim() > 1 else o_emb

                        if is_positive:
                            # Loss pour exemples positifs
                            # 1. Reconstruction loss
                            recon_loss = F.mse_loss(o_pred, o_emb_squeezed)

                            # 2. Confidence loss - on veut une haute confiance
                            # Utiliser une cible adaptative basée sur la qualité de reconstruction
                            with torch.no_grad():
                                # Cible entre 0.7 et 0.95 selon la reconstruction
                                target_conf = 0.7 + 0.25 * torch.exp(-recon_loss * 5)
                                target_conf = torch.clamp(target_conf, 0.7, 0.95)

                            # S'assurer que confidence est un scalaire
                            if confidence.dim() == 0:
                                conf_loss = F.binary_cross_entropy(
                                    confidence.unsqueeze(0),
                                    target_conf.unsqueeze(0)
                                )
                            else:
                                conf_loss = F.binary_cross_entropy(
                                    confidence,
                                    target_conf
                                )

                            # 3. Loss totale (s'assurer que c'est un scalaire)
                            loss = recon_loss.squeeze() + 0.3 * conf_loss.squeeze()

                        else:
                            # Loss pour exemples négatifs
                            # 1. On veut une faible confiance
                            target_conf = torch.tensor(0.1).to(self.device)  # Petite marge au lieu de 0

                            if confidence.dim() == 0:
                                conf_loss = F.binary_cross_entropy(
                                    confidence.unsqueeze(0),
                                    target_conf.unsqueeze(0)
                                )
                            else:
                                conf_loss = F.binary_cross_entropy(
                                    confidence,
                                    target_conf
                                )

                            # 2. Pénalité de similarité (optionnel)
                            with torch.no_grad():
                                pred_similarity = F.cosine_similarity(
                                    o_pred.unsqueeze(0) if o_pred.dim() == 1 else o_pred,
                                    o_emb_squeezed.unsqueeze(0) if o_emb_squeezed.dim() == 1 else o_emb_squeezed,
                                    dim=1
                                )
                                pred_similarity = pred_similarity.squeeze()

                            # Si la prédiction est trop similaire à un mauvais objet, pénaliser
                            similarity_penalty = torch.relu(pred_similarity - 0.5) * 0.5

                            # S'assurer que tout est scalaire
                            loss = conf_loss.squeeze() + similarity_penalty.squeeze()

                        # Ajouter une petite régularisation L2 sur les prédictions
                        l2_reg = 0.001 * torch.norm(o_pred)

                        # S'assurer que la loss finale est un scalaire
                        total_loss = loss.squeeze() + l2_reg.squeeze()
                        if total_loss.dim() > 0:
                            total_loss = total_loss.mean()

                        batch_loss += total_loss

                        # Calculer l'accuracy
                        with torch.no_grad():
                            conf_value = confidence.item() if confidence.dim() == 0 else confidence.squeeze().item()
                            predicted_positive = (conf_value > 0.5)
                            correct = (predicted_positive == is_positive)
                            epoch_correct += int(correct)
                            epoch_total += 1

                    # Backward sur le batch
                    avg_batch_loss = batch_loss / len(batch_indices)
                    avg_batch_loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # Update
                    optimizer.step()

                    epoch_loss += avg_batch_loss.item()
                    num_batches += 1

                # Moyennes pour l'epoch
                avg_loss = epoch_loss / num_batches
                avg_accuracy = epoch_correct / epoch_total

                losses.append(avg_loss)
                accuracies.append(avg_accuracy)

                # Scheduler step
                scheduler.step(avg_loss)

                # Early stopping
                if avg_loss < best_loss or avg_accuracy > best_accuracy:
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                    if avg_accuracy > best_accuracy:
                        best_accuracy = avg_accuracy

                    patience_counter = 0
                    best_state = {
                        'model_state': self.model.state_dict(),
                        'epoch': epoch,
                        'loss': avg_loss,
                        'accuracy': avg_accuracy
                    }
                else:
                    patience_counter += 1

                # Affichage périodique
                if epoch % 20 == 0 or epoch == num_epochs - 1:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"  Epoch {epoch}: Loss = {avg_loss:.4f}, Acc = {avg_accuracy:.2f}, LR = {current_lr:.6f}")

                # Early stopping
                if patience_counter >= patience and epoch > 100:
                    print(f"  Early stopping à l'epoch {epoch} (best acc: {best_accuracy:.2f})")
                    break

                # Stocker périodiquement dans la mémoire Hopfield
                if epoch % 50 == 0 and epoch > 0:
                    with torch.no_grad():
                        for s_emb, o_emb, is_pos in positive_data[:5]:
                            if is_pos:
                                self.model.store_example(
                                    s_emb.to(self.device),
                                    o_emb.to(self.device)
                                )

            # Restaurer le meilleur modèle
            if best_state is not None:
                self.model.load_state_dict(best_state['model_state'])
                print(f"  Modèle restauré de l'epoch {best_state['epoch']} (acc: {best_state['accuracy']:.2f})")

            # Stocker tous les exemples positifs dans la mémoire après entraînement
            print("📝 Stockage des prototypes dans la mémoire...")
            with torch.no_grad():
                for s_emb, o_emb, is_pos in positive_data[:20]:
                    if is_pos:
                        self.model.store_example(
                            s_emb.to(self.device),
                            o_emb.to(self.device)
                        )

            # Métadonnées
            self.is_trained = True
            self.last_trained = datetime.now().isoformat()
            self.training_metrics = {
                'final_loss': float(best_loss),
                'final_accuracy': float(best_accuracy),
                'num_examples': len(examples),
                'num_negatives': len(negative_data),
                'num_epochs': epoch + 1,
                'learning_rate': learning_rate
            }

            print(f"✓ {self.label} appris avec succès (loss: {best_loss:.4f}, acc: {best_accuracy:.2f})")
            return True

        except Exception as e:
            print(f"❌ Erreur lors de l'apprentissage: {e}")
            import traceback
            traceback.print_exc()
            return False

    def learn_from_examples(self, examples: List[Tuple[str, str]],
                            concept_embeddings: Dict[str, np.ndarray],
                            num_epochs: int = None,
                            learning_rate: float = None) -> bool:
        """Apprend la transformation avec une meilleure calibration."""

        n_examples = len(examples)

        # Ajustement automatique des hyperparamètres - PLUS D'EPOCHS
        if num_epochs is None:
            if n_examples < 5:
                num_epochs = 1600
            elif n_examples < 10:
                num_epochs = 1200
            elif n_examples < 20:
                num_epochs = 1000
            else:
                num_epochs = 600

        if learning_rate is None:
            if n_examples < 5:
                learning_rate = 0.001
            elif n_examples < 10:
                learning_rate = 0.001
            else:
                learning_rate = 0.002

        # Régularisation adaptative
        weight_decay = 0.05 if n_examples < 5 else 0.01

        try:
            print(f"🔄 Apprentissage de {self.label} avec {len(examples)} exemples...")
            print(f"   Hyperparamètres: epochs={num_epochs}, lr={learning_rate}, wd={weight_decay}")

            # Stocker les embeddings et exemples
            self.concept_embeddings = concept_embeddings
            self.positive_examples = examples

            # Générer exemples négatifs équilibrés
            all_concepts = list(concept_embeddings.keys())
            self.negative_examples = self.generate_hard_negative_examples(
                self.positive_examples,
                all_concepts,
                concept_embeddings
            )
            self.negative_examples = self.negative_examples[:len(examples)]
            print(f"   Généré {len(self.negative_examples)} exemples négatifs (ratio 1:1)")

            # Préparer les données d'entraînement
            positive_data = []
            for s_uri, o_uri in examples:
                if s_uri in concept_embeddings and o_uri in concept_embeddings:
                    s_emb = torch.tensor(concept_embeddings[s_uri]).float()
                    o_emb = torch.tensor(concept_embeddings[o_uri]).float()
                    positive_data.append((s_emb, o_emb, True))

            negative_data = []
            for s_uri, o_uri in self.negative_examples:
                if s_uri in concept_embeddings and o_uri in concept_embeddings:
                    s_emb = torch.tensor(concept_embeddings[s_uri]).float()
                    o_emb = torch.tensor(concept_embeddings[o_uri]).float()
                    negative_data.append((s_emb, o_emb, False))

            if len(positive_data) < 3:
                print(f"⚠️ Pas assez d'embeddings valides")
                return False

            # Combiner toutes les données
            all_data = positive_data + negative_data
            print(f"   Total: {len(positive_data)} positifs, {len(negative_data)} négatifs")

            # Créer l'optimiseur
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)  # Plus stable
            )

            # Scheduler cosine avec warm restart
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=50,  # Restart tous les 50 epochs
                T_mult=2,  # Doubler la période à chaque restart
                eta_min=learning_rate * 0.01
            )

            # Variables pour early stopping
            best_loss = float('inf')
            best_accuracy = 0.0
            best_f1 = 0.0
            best_state = None
            patience = 100  # Plus de patience
            patience_counter = 0

            # Threshold adaptatif
            confidence_threshold = 0.5

            # Entraînement
            self.model.train()
            losses = []

            for epoch in range(num_epochs):
                epoch_loss = 0.0

                # Métriques détaillées
                true_positives = 0
                false_positives = 0
                true_negatives = 0
                false_negatives = 0

                # Stocker les confidences pour ajuster le seuil
                all_confidences = []
                all_labels = []

                # Mélanger les données
                indices = torch.randperm(len(all_data))

                # Taille de batch adaptative
                batch_size = min(16, len(all_data))  # Batches plus grands
                num_batches = 0

                # Traiter par mini-batches
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:min(i + batch_size, len(indices))]
                    batch_loss = 0.0

                    # Accumuler les gradients sur le batch
                    optimizer.zero_grad()

                    for idx in batch_indices:
                        s_emb, o_emb, is_positive = all_data[idx.item()]
                        s_emb = s_emb.to(self.device).unsqueeze(0)
                        o_emb = o_emb.to(self.device).unsqueeze(0)

                        # Forward pass
                        o_pred, confidence = self.model(s_emb)

                        # S'assurer que tout est bien dimensionné
                        o_pred = o_pred.squeeze(0) if o_pred.dim() > 1 else o_pred
                        confidence = confidence.squeeze() if confidence.dim() > 0 else confidence
                        o_emb_squeezed = o_emb.squeeze(0) if o_emb.dim() > 1 else o_emb

                        # Stocker pour ajuster le seuil
                        all_confidences.append(confidence.item())
                        all_labels.append(is_positive)

                        if is_positive:
                            # Loss pour exemples positifs
                            recon_loss = F.mse_loss(o_pred, o_emb_squeezed)

                            # Cibles de confiance MOINS EXTRÊMES
                            with torch.no_grad():
                                # Cible entre 0.6 et 0.9 selon la reconstruction
                                target_conf = 0.6 + 0.3 * torch.exp(-recon_loss * 2)
                                target_conf = torch.clamp(target_conf, 0.6, 0.9)

                            # Focal loss pour mieux gérer les cas difficiles
                            p = confidence
                            focal_weight = (1 - p) ** 2  # Plus de poids aux erreurs
                            conf_loss = focal_weight * F.binary_cross_entropy(
                                confidence.unsqueeze(0) if confidence.dim() == 0 else confidence,
                                target_conf.unsqueeze(0) if target_conf.dim() == 0 else target_conf
                            )

                            # Loss totale avec pondération équilibrée
                            loss = recon_loss + conf_loss

                        else:
                            # Loss pour exemples négatifs
                            # Cible moins extrême
                            target_conf = torch.tensor(0.1).to(self.device)

                            # Focal loss
                            p = 1 - confidence  # Probabilité d'être correct (négatif)
                            focal_weight = (1 - p) ** 2
                            conf_loss = focal_weight * F.binary_cross_entropy(
                                confidence.unsqueeze(0) if confidence.dim() == 0 else confidence,
                                target_conf.unsqueeze(0) if target_conf.dim() == 0 else target_conf
                            )

                            # Bonus si la prédiction est différente
                            with torch.no_grad():
                                pred_similarity = F.cosine_similarity(
                                    o_pred.unsqueeze(0) if o_pred.dim() == 1 else o_pred,
                                    o_emb_squeezed.unsqueeze(0) if o_emb_squeezed.dim() == 1 else o_emb_squeezed,
                                    dim=1
                                ).squeeze()

                            dissimilarity_bonus = torch.relu(0.7 - pred_similarity) * 0.2

                            loss = conf_loss - dissimilarity_bonus

                        # Régularisation L2 réduite
                        l2_reg = 0.0001 * torch.norm(o_pred)

                        # Loss finale
                        total_loss = loss.squeeze() + l2_reg.squeeze()
                        if total_loss.dim() > 0:
                            total_loss = total_loss.mean()

                        batch_loss += total_loss

                    # Backward sur le batch
                    avg_batch_loss = batch_loss / len(batch_indices)
                    avg_batch_loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                    # Update
                    optimizer.step()

                    epoch_loss += avg_batch_loss.item()
                    num_batches += 1

                # Ajuster le seuil de décision dynamiquement
                if epoch % 10 == 0 and epoch > 0:
                    # Trouver le seuil optimal qui maximise le F1
                    best_threshold = confidence_threshold
                    best_epoch_f1 = 0

                    for threshold in np.linspace(0.2, 0.8, 13):
                        tp = sum(1 for conf, label in zip(all_confidences, all_labels)
                                 if conf > threshold and label)
                        fp = sum(1 for conf, label in zip(all_confidences, all_labels)
                                 if conf > threshold and not label)
                        tn = sum(1 for conf, label in zip(all_confidences, all_labels)
                                 if conf <= threshold and not label)
                        fn = sum(1 for conf, label in zip(all_confidences, all_labels)
                                 if conf <= threshold and label)

                        precision = tp / (tp + fp + 1e-8)
                        recall = tp / (tp + fn + 1e-8)
                        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

                        if f1 > best_epoch_f1:
                            best_epoch_f1 = f1
                            best_threshold = threshold

                    confidence_threshold = best_threshold

                # Calculer les métriques avec le seuil actuel
                for conf, label in zip(all_confidences, all_labels):
                    predicted_positive = (conf > confidence_threshold)

                    if label and predicted_positive:
                        true_positives += 1
                    elif label and not predicted_positive:
                        false_negatives += 1
                    elif not label and not predicted_positive:
                        true_negatives += 1
                    else:
                        false_positives += 1

                # Métriques
                avg_loss = epoch_loss / num_batches
                accuracy = (true_positives + true_negatives) / len(all_data)
                precision = true_positives / (true_positives + false_positives + 1e-8)
                recall = true_positives / (true_positives + false_negatives + 1e-8)
                f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

                losses.append(avg_loss)

                # Scheduler step
                scheduler.step()

                # Early stopping basé sur F1
                if f1_score > best_f1 or (f1_score == best_f1 and avg_loss < best_loss):
                    best_loss = avg_loss
                    best_accuracy = accuracy
                    best_f1 = f1_score
                    patience_counter = 0
                    best_state = {
                        'model_state': self.model.state_dict(),
                        'epoch': epoch,
                        'loss': avg_loss,
                        'accuracy': accuracy,
                        'f1': f1_score,
                        'precision': precision,
                        'recall': recall,
                        'threshold': confidence_threshold
                    }
                else:
                    patience_counter += 1

                # Affichage périodique
                if epoch % 20 == 0 or epoch == num_epochs - 1:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"  Epoch {epoch}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}, "
                          f"P={precision:.2f}, R={recall:.2f}, F1={f1_score:.2f}, "
                          f"Thr={confidence_threshold:.2f}, LR={current_lr:.6f}")

                # Early stopping
                if patience_counter >= patience and epoch > 150:
                    print(f"  Early stopping à l'epoch {epoch} (best F1: {best_f1:.2f})")
                    break

                # Stocker dans la mémoire Hopfield
                if epoch % 100 == 0 and epoch > 0:
                    with torch.no_grad():
                        for s_emb, o_emb, is_pos in positive_data[:5]:
                            if is_pos:
                                self.model.store_example(
                                    s_emb.to(self.device),
                                    o_emb.to(self.device)
                                )

            # Restaurer le meilleur modèle
            if best_state is not None:
                self.model.load_state_dict(best_state['model_state'])
                self.confidence_threshold = best_state['threshold']  # Sauvegarder le seuil optimal
                print(f"  Modèle restauré de l'epoch {best_state['epoch']} "
                      f"(F1: {best_state['f1']:.2f}, Acc: {best_state['accuracy']:.2f}, Threshold: {best_state['threshold']:.2f})")

            # Stocker tous les exemples positifs
            print("📝 Stockage des prototypes dans la mémoire...")
            with torch.no_grad():
                for s_emb, o_emb, is_pos in positive_data[:20]:
                    if is_pos:
                        self.model.store_example(
                            s_emb.to(self.device),
                            o_emb.to(self.device)
                        )

            # Métadonnées
            self.is_trained = True
            self.last_trained = datetime.now().isoformat()
            self.training_metrics = {
                'final_loss': float(best_loss),
                'final_accuracy': float(best_accuracy),
                'final_f1': float(best_f1),
                'final_precision': float(best_state['precision'] if best_state else 0),
                'final_recall': float(best_state['recall'] if best_state else 0),
                'optimal_threshold': float(best_state['threshold'] if best_state else 0.5),
                'num_examples': len(examples),
                'num_negatives': len(negative_data),
                'num_epochs': epoch + 1,
                'learning_rate': learning_rate
            }

            print(f"✓ {self.label} appris avec succès (F1: {best_f1:.2f}, Acc: {best_accuracy:.2f})")
            return True

        except Exception as e:
            print(f"❌ Erreur lors de l'apprentissage: {e}")
            import traceback
            traceback.print_exc()
            return False


    def generate_hard_negative_examples(self, positive_examples: List[Tuple[str, str]],
                                        all_concepts: List[str],
                                        concept_embeddings: Dict[str, np.ndarray]) -> List[Tuple[str, str]]:
        """Génère des exemples négatifs difficiles (hard negatives)."""
        import random
        import numpy as np

        negative_examples = []
        positive_pairs = set(positive_examples)

        # Extraire les sujets et objets des exemples positifs
        positive_subjects = list(set(s for s, o in positive_examples))
        positive_objects = list(set(o for s, o in positive_examples))

        # Stratégie 1: Hard negatives - objets similaires mais incorrects (50%)
        for s, o_correct in positive_examples:
            if s in concept_embeddings:
                s_emb = concept_embeddings[s]

                # Calculer les similarités avec tous les objets positifs
                candidates = []
                for o_candidate in positive_objects:
                    if o_candidate != o_correct and o_candidate in concept_embeddings:
                        o_emb = concept_embeddings[o_candidate]

                        # Similarité cosinus
                        sim = np.dot(s_emb, o_emb) / (np.linalg.norm(s_emb) * np.linalg.norm(o_emb) + 1e-8)
                        candidates.append((o_candidate, sim))

                # Trier par similarité décroissante (les plus similaires sont les plus difficiles)
                candidates.sort(key=lambda x: x[1], reverse=True)

                # Prendre les top-2 plus similaires
                for o_neg, sim in candidates[:2]:
                    if (s, o_neg) not in positive_pairs:
                        negative_examples.append((s, o_neg))

        # Stratégie 2: Corruption de sujets (30%)
        num_subject_corruption = max(1, len(positive_examples) // 3)
        for _ in range(num_subject_corruption):
            if len(positive_subjects) > 1 and positive_objects:
                s_random = random.choice(positive_subjects)
                o_random = random.choice(positive_objects)

                if (s_random, o_random) not in positive_pairs:
                    negative_examples.append((s_random, o_random))

        # Stratégie 3: Semi-hard negatives - concepts hors distribution (20%)
        other_concepts = list(set(all_concepts) - set(positive_subjects) - set(positive_objects))
        if other_concepts and positive_subjects:
            num_random = max(1, len(positive_examples) // 5)
            for _ in range(num_random):
                s = random.choice(positive_subjects)
                o = random.choice(other_concepts[:20])  # Limiter aux 20 premiers pour éviter le bruit

                if (s, o) not in positive_pairs:
                    negative_examples.append((s, o))

        # Dédupliquer et mélanger
        negative_examples = list(set(negative_examples))
        random.shuffle(negative_examples)

        # Retourner au maximum 2x le nombre d'exemples positifs
        return negative_examples[:len(positive_examples) * 2]

    def predict_objects(self, subject_uri: str,
                        all_concept_embeddings: Dict[str, np.ndarray],
                        top_k: int = 10,
                        threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Prédit avec support complet de la transitivité."""
        if not self.is_trained or subject_uri not in all_concept_embeddings:
            return []

        try:
            self.model.eval()
            results = {}  # Utiliser un dict pour éviter les doublons

            # 1. Prédictions directes du modèle
            s_emb = torch.tensor(all_concept_embeddings[subject_uri]).float().to(self.device)

            with torch.no_grad():
                o_pred, confidence = self.model(s_emb)
                conf_value = confidence.item()

                # Calculer les similarités
                o_pred_np = o_pred.cpu().numpy()
                o_pred_norm = o_pred_np / (np.linalg.norm(o_pred_np) + 1e-8)

                for obj_uri, obj_emb in all_concept_embeddings.items():
                    if not self._check_ontological_constraints(subject_uri, obj_uri):
                        continue

                    obj_norm = obj_emb / (np.linalg.norm(obj_emb) + 1e-8)
                    similarity = np.dot(o_pred_norm, obj_norm)

                    # Score de base
                    base_score = similarity * conf_value

                    # Boost si dans les exemples d'entraînement
                    if (subject_uri, obj_uri) in self.positive_examples:
                        score = min(1.0, base_score * 2.0)
                    else:
                        score = base_score

                    if score >= threshold * 0.5:  # Seuil plus permissif
                        results[obj_uri] = {
                            "concept_uri": obj_uri,
                            "label": obj_uri.split('/')[-1],
                            "similarity": float(similarity),
                            "confidence": float(conf_value),
                            "score": float(score),
                            "source": "model"
                        }

            # 2. Ajouter les résultats transitifs si applicable
            if self.properties.get("transitive", False) and hasattr(self, 'transitive_examples'):
                for s, o in self.transitive_examples:
                    if s == subject_uri and o in all_concept_embeddings:
                        if o not in results or results[o]["score"] < 0.8:
                            results[o] = {
                                "concept_uri": o,
                                "label": o.split('/')[-1],
                                "similarity": 0.8,
                                "confidence": 0.8,
                                "score": 0.8,
                                "source": "transitivity"
                            }

            # 3. Vérifier aussi les exemples directs
            for s, o in self.positive_examples:
                if s == subject_uri and o in all_concept_embeddings:
                    if o not in results:
                        results[o] = {
                            "concept_uri": o,
                            "label": o.split('/')[-1],
                            "similarity": 1.0,
                            "confidence": 1.0,
                            "score": 1.0,
                            "source": "training_example"
                        }

            # Convertir en liste et trier
            final_results = list(results.values())
            final_results.sort(key=lambda x: x["score"], reverse=True)

            return final_results[:top_k]

        except Exception as e:
            print(f"⚠️ Erreur lors de la prédiction: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _check_ontological_constraints(self, subject_uri: str, object_uri: str) -> bool:
        """Vérifie les contraintes ontologiques."""
        # Contraintes de domaine/range
        if self.range_concept_uris:
            # Vérifier si l'objet est du bon type
            # (Simplifié ici, devrait vérifier la hiérarchie)
            pass

        # Propriétés logiques
        if self.properties.get("irreflexive", False) and subject_uri == object_uri:
            return False

        return True

    def _adjust_score_by_properties(self, base_score: float,
                                    subject_uri: str, object_uri: str) -> float:
        """Ajuste le score selon les propriétés de la relation."""
        score = base_score

        # Boost si l'exemple existe dans les données d'entraînement
        if (subject_uri, object_uri) in self.positive_examples:
            score *= 1.5

        # Pénalité pour réflexivité non autorisée
        if subject_uri == object_uri and not self.properties.get("reflexive", False):
            score *= 0.5

        return np.clip(score, 0.0, 1.0)

    def predict_objects_with_transitivity(self, subject_uri: str,
                                          all_concept_embeddings: Dict[str, np.ndarray],
                                          all_triples: List[Tuple[str, str, str]],
                                          top_k: int = 10,
                                          threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Prédit avec support de la transitivité."""
        # D'abord les prédictions directes
        direct_predictions = self.predict_objects(
            subject_uri, all_concept_embeddings, top_k * 2, threshold * 0.8
        )

        results = {p['concept_uri']: p for p in direct_predictions}

        # Si relation transitive, chercher les chemins
        if self.properties.get("transitive", False):
            # Trouver tous les triplets de cette relation
            relation_triples = [(s, o) for s, r, o in all_triples if r == self.uri]

            # Construire le graphe
            graph = defaultdict(set)
            for s, o in relation_triples:
                graph[s].add(o)

            # Parcours en largeur pour trouver les chemins transitifs
            visited = set()
            queue = [(subject_uri, 0)]  # (node, distance)

            while queue:
                current, dist = queue.pop(0)
                if current in visited or dist > 2:  # Limiter la profondeur
                    continue

                visited.add(current)

                # Ajouter les voisins directs
                for neighbor in graph.get(current, []):
                    if neighbor not in results and neighbor in all_concept_embeddings:
                        # Calculer un score basé sur la distance
                        transitive_score = 0.8 ** (dist + 1)  # Diminue avec la distance

                        if transitive_score >= threshold * 0.5:
                            results[neighbor] = {
                                "concept_uri": neighbor,
                                "label": neighbor.split('/')[-1],
                                "similarity": transitive_score,
                                "confidence": transitive_score,
                                "score": transitive_score,
                                "transitive": True,
                                "distance": dist + 1
                            }

                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))

        # Trier et retourner
        final_results = list(results.values())
        final_results.sort(key=lambda x: x.get("score", x["confidence"]), reverse=True)
        return final_results[:top_k]

    def generate_negative_examples(self, positive_examples: List[Tuple[str, str]],
                                   all_concepts: List[str]) -> List[Tuple[str, str]]:
        """Génère des exemples négatifs plus difficiles."""
        import random

        negative_examples = []
        positive_pairs = set(positive_examples)

        # Extraire embeddings pour calculer les similarités
        positive_subjects = list(set(s for s, o in positive_examples))
        positive_objects = list(set(o for s, o in positive_examples))

        # Stratégie 1: Hard negatives (objets similaires mais incorrects)
        for s, o_correct in positive_examples:
            if s in self.concept_embeddings:
                s_emb = self.concept_embeddings[s]

                # Trouver des objets similaires mais différents
                similarities = []
                for o_candidate in positive_objects:
                    if o_candidate != o_correct and o_candidate in self.concept_embeddings:
                        o_emb = self.concept_embeddings[o_candidate]
                        sim = np.dot(s_emb, o_emb) / (np.linalg.norm(s_emb) * np.linalg.norm(o_emb) + 1e-8)
                        similarities.append((o_candidate, sim))

                # Prendre les plus similaires (hard negatives)
                similarities.sort(key=lambda x: x[1], reverse=True)
                for o_neg, _ in similarities[:2]:
                    if (s, o_neg) not in positive_pairs:
                        negative_examples.append((s, o_neg))

        # Limiter et mélanger
        random.shuffle(negative_examples)
        return negative_examples[:len(positive_examples) * 2]


class SimplifiedRelationManager:
    """Gestionnaire simplifié des relations."""

    def __init__(self, ontology_manager, storage_dir: str = "relation_models"):
        self.ontology_manager = ontology_manager
        self.storage_dir = storage_dir
        self.transformations = {}

        # Configuration
        self.concept_dim = 768
        self.relation_dim = 256

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Créer le répertoire
        os.makedirs(storage_dir, exist_ok=True)
        print(f"💾 Gestionnaire de relations initialisé (device: {self.device})")

    async def initialize(self, concept_dim: int = 768, relation_dim: int = 256):
        """Initialise le gestionnaire."""
        print("🚀 Initialisation du gestionnaire de relations...")

        self.concept_dim = concept_dim
        self.relation_dim = relation_dim

        # Charger ou créer les transformations
        loaded = await self._load_transformations()

        if not loaded:
            # Créer les transformations pour chaque relation
            for uri, relation in self.ontology_manager.relations.items():
                transform = RelationTransformation(
                    uri, relation.label,
                    self.concept_dim, self.relation_dim
                )

                # Ajouter les métadonnées ontologiques
                for domain in relation.domain:
                    if hasattr(domain, 'uri'):
                        transform.domain_concept_uris.append(domain.uri)

                for range_concept in relation.range:
                    if hasattr(range_concept, 'uri'):
                        transform.range_concept_uris.append(range_concept.uri)

                self.transformations[uri] = transform

        # Analyser les propriétés
        self._analyze_properties()

        print(f"✓ {len(self.transformations)} transformations initialisées")

    def _analyze_properties(self):
        """Analyse les propriétés logiques des relations."""
        for uri, transform in self.transformations.items():
            # Parcourir les axiomes de l'ontologie
            for axiom_type, source, target in self.ontology_manager.axioms:
                if source == uri:
                    if axiom_type in ["transitive_property", "symmetric_property",
                                      "functional_property", "inverse_functional_property"]:
                        prop_name = axiom_type.replace("_property", "")
                        transform.properties[prop_name] = True

    async def learn_relations(self, triples: List[Tuple[str, str, str]],
                              concept_embeddings: Dict[str, np.ndarray],
                              min_examples: int = 3,
                              force_relearn: bool = False) -> Dict[str, bool]:
        """Apprend toutes les relations à partir des triplets."""
        # Organiser par relation
        examples_by_relation = defaultdict(list)
        for s, r, o in triples:
            examples_by_relation[r].append((s, o))

        results = {}

        for relation_uri, examples in examples_by_relation.items():
            if relation_uri not in self.transformations:
                continue

            transform = self.transformations[relation_uri]

            # Vérifier si déjà entraîné
            if transform.is_trained and not force_relearn:
                results[relation_uri] = True
                continue

            # Apprendre si assez d'exemples
            if len(examples) >= min_examples:
                success = transform.learn_from_examples(
                    examples, concept_embeddings
                )
                results[relation_uri] = success
            else:
                results[relation_uri] = False

        # APRÈS l'apprentissage, appliquer la transitivité
        print("\n📊 Application de la transitivité...")
        for relation_uri, transform in self.transformations.items():
            if transform.is_trained and transform.properties.get("transitive", False):
                self.apply_transitivity_for_transform(transform, triples)


        # Sauvegarder
        await self._save_transformations()

        # Résumé
        trained = sum(1 for v in results.values() if v)
        print(f"\n✓ {trained}/{len(results)} relations apprises avec succès")

        return results

    def get_related_concepts(self, concept_uri: str, relation_uri: str,
                             concept_embeddings: Dict[str, np.ndarray],
                             top_k: int = 5,
                             threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Obtient les concepts liés via une relation."""
        if relation_uri not in self.transformations:
            return []

        transform = self.transformations[relation_uri]
        return transform.predict_objects(
            concept_uri, concept_embeddings, top_k, threshold
        )

    def infer_new_relations(self, subject_uri: str,
                            concept_embeddings: Dict[str, np.ndarray],
                            confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:  # Seuil plus bas
        """Infère avec un seuil adaptatif."""
        if subject_uri not in concept_embeddings:
            return []

        results = []

        for relation_uri, transform in self.transformations.items():
            if not transform.is_trained:
                continue

            # Ajuster le seuil selon le nombre d'exemples d'entraînement
            n_examples = len(transform.positive_examples)
            adjusted_threshold = confidence_threshold * (1.0 + n_examples / 50.0)
            adjusted_threshold = min(adjusted_threshold, 0.8)  # Cap maximum

            predictions = transform.predict_objects(
                subject_uri, concept_embeddings,
                top_k=5, threshold=adjusted_threshold
            )

            for pred in predictions:
                results.append({
                    "subject": subject_uri,
                    "relation": relation_uri,
                    "relation_label": transform.label,
                    "object": pred["concept_uri"],
                    "object_label": pred["label"],
                    "confidence": pred["confidence"],
                    "source": pred.get("source", "unknown")
                })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    async def _save_transformations(self):
        """Sauvegarde les transformations."""
        try:
            # Sauvegarder les métadonnées
            metadata = {}

            for uri, transform in self.transformations.items():
                if transform.is_trained:
                    # Sauvegarder le modèle ET les exemples
                    model_filename = f"model_{hashlib.md5(uri.encode()).hexdigest()}.pt"
                    model_path = os.path.join(self.storage_dir, model_filename)

                    torch.save({
                        'model_state_dict': transform.model.state_dict(),
                        'concept_dim': transform.concept_dim,
                        'relation_dim': transform.relation_dim,
                        'positive_examples': transform.positive_examples,
                        'transitive_examples': getattr(transform, 'transitive_examples', [])
                    }, model_path)

                    metadata[uri] = {
                        'uri': transform.uri,
                        'label': transform.label,
                        'concept_dim': transform.concept_dim,
                        'relation_dim': transform.relation_dim,
                        'is_trained': True,
                        'model_file': model_filename,
                        'properties': transform.properties,
                        'domain_uris': transform.domain_concept_uris,
                        'range_uris': transform.range_concept_uris,
                        'training_metrics': transform.training_metrics,
                        'last_trained': transform.last_trained,
                        'num_examples': len(transform.positive_examples),  # ← Correct
                        'positive_examples': transform.positive_examples  # ← Sauvegarder aussi
                    }
                else:
                    metadata[uri] = {
                        'uri': transform.uri,
                        'label': transform.label,
                        'concept_dim': transform.concept_dim,
                        'relation_dim': transform.relation_dim,
                        'is_trained': False,
                        'properties': transform.properties,
                        'num_examples': 0
                    }

            # Sauvegarder les métadonnées
            metadata_path = os.path.join(self.storage_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            print(f"⚠️ Erreur lors de la sauvegarde: {e}")

    async def _load_transformations(self) -> bool:
        """Charge les transformations sauvegardées."""
        metadata_path = os.path.join(self.storage_dir, "metadata.json")

        if not os.path.exists(metadata_path):
            return False

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            for uri, meta in metadata.items():
                transform = RelationTransformation(
                    meta['uri'], meta['label'],
                    meta.get('concept_dim', self.concept_dim),
                    meta.get('relation_dim', self.relation_dim)
                )

                # Restaurer toutes les propriétés
                transform.properties = meta.get('properties', {})
                transform.domain_concept_uris = meta.get('domain_uris', [])
                transform.range_concept_uris = meta.get('range_uris', [])
                transform.training_metrics = meta.get('training_metrics', {})
                transform.last_trained = meta.get('last_trained')
                transform.is_trained = meta.get('is_trained', False)

                # Restaurer les exemples
                transform.positive_examples = meta.get('positive_examples', [])

                # Charger le modèle si entraîné
                if transform.is_trained and meta.get('model_file'):
                    model_path = os.path.join(self.storage_dir, meta['model_file'])
                    if os.path.exists(model_path):
                        checkpoint = torch.load(model_path, map_location=self.device)
                        transform.model.load_state_dict(checkpoint['model_state_dict'])

                        # Restaurer aussi les exemples transitifs si présents
                        if 'transitive_examples' in checkpoint:
                            transform.transitive_examples = checkpoint['transitive_examples']

                self.transformations[uri] = transform

            print(f"✓ {len(self.transformations)} transformations chargées")
            return True

        except Exception as e:
            print(f"⚠️ Erreur lors du chargement: {e}")
            return False

    def apply_transitivity(self, relation_uri: str, max_depth: int = 2):
        """Applique la transitivité à une relation."""
        if relation_uri not in self.transformations:
            return

        transform = self.transformations[relation_uri]
        if not transform.properties.get("transitive", False):
            return

        print(f"\n🔄 Application de la transitivité pour {transform.label}...")

        # Construire le graphe de la relation
        graph = defaultdict(set)
        for s, o in transform.positive_examples:
            graph[s].add(o)

        print(f"  Graphe initial: {len(graph)} nœuds")

        # Calculer la fermeture transitive
        new_edges_total = 0
        for depth in range(max_depth):
            new_edges = []

            for s in list(graph.keys()):
                for o1 in list(graph[s]):
                    if o1 in graph:
                        for o2 in graph[o1]:
                            if o2 not in graph[s]:
                                new_edges.append((s, o2))

            if not new_edges:
                break

            # Ajouter les nouvelles arêtes
            for s, o in new_edges:
                graph[s].add(o)
                new_edges_total += 1

        # Stocker les exemples transitifs
        transform.transitive_examples = []
        for s, objects in graph.items():
            for o in objects:
                if (s, o) not in transform.positive_examples:
                    transform.transitive_examples.append((s, o))

        print(f"  ✓ Ajouté {len(transform.transitive_examples)} exemples transitifs")

        # Afficher quelques exemples
        if transform.transitive_examples:
            print(f"  Exemples: {transform.transitive_examples[:3]}")

    def apply_transitivity_for_transform(self, transform, all_triples):
        """Applique la transitivité pour une transformation."""
        print(f"\n🔄 Calcul de la transitivité pour {transform.label}...")

        # Extraire seulement les triplets de cette relation
        relation_triples = [(s, o) for s, r, o in all_triples if r == transform.uri]

        # Construire le graphe
        graph = defaultdict(set)
        for s, o in relation_triples:
            graph[s].add(o)

        # Fermeture transitive
        changed = True
        iterations = 0
        new_total = 0

        while changed and iterations < 3:  # Max 3 niveaux
            changed = False
            new_edges = []

            for s in list(graph.keys()):
                for o1 in list(graph[s]):
                    if o1 in graph:
                        for o2 in graph[o1]:
                            if o2 not in graph[s]:
                                new_edges.append((s, o2))
                                changed = True

            for s, o in new_edges:
                graph[s].add(o)
                new_total += 1

            iterations += 1

        # Stocker les exemples transitifs
        transform.transitive_examples = []
        for s, objects in graph.items():
            for o in objects:
                if (s, o) not in relation_triples:
                    transform.transitive_examples.append((s, o))

        print(f"  ✓ Trouvé {len(transform.transitive_examples)} relations transitives")
        if transform.transitive_examples[:3]:
            print(f"  Exemples: {transform.transitive_examples[:3]}")

    def get_statistics(self) -> Dict[str, Any]:
        """Retourne des statistiques sur les relations apprises."""
        stats = {
            'total_relations': len(self.transformations),
            'trained_relations': sum(1 for t in self.transformations.values() if t.is_trained),
            'relations_details': []
        }

        for uri, transform in self.transformations.items():
            stats['relations_details'].append({
                'uri': uri,
                'label': transform.label,
                'is_trained': transform.is_trained,
                'num_examples': len(transform.positive_examples) if transform.is_trained else 0,
                'properties': transform.properties,
                'metrics': transform.training_metrics
            })

        return stats

