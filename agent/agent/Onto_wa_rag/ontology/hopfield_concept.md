## Architecture générale du système

Le système utilise une approche hiérarchique avec plusieurs composants clés :

1. **`OntologyClassifier`** : Orchestrateur principal
2. **`ConceptHopfieldClassifier`** : Classifieur de concepts avec réseaux de Hopfield
3. **Structure hiérarchique** : Concepts organisés par niveaux

## Mécanisme d'apprentissage des concepts

### 1. Construction de la hiérarchie

```python
# Dans _build_concept_hierarchy()
self.concept_hierarchy = {}  # URI concept -> [sous-concepts URIs]
self.concept_to_level = {}   # URI concept -> niveau
self.level_to_concepts = {}  # niveau -> [URI concepts]
```

Le système analyse l'ontologie RDF/TTL pour :
- Extraire les relations parent-enfant
- Calculer les niveaux hiérarchiques avec `_calculate_concept_level()`
- Organiser les concepts par niveau

### 2. Génération des embeddings

**Point crucial** : Dans `_generate_concept_embedding()`, l'embedding d'un concept **inclut l'information de ses parents** :

```python
# Récupérer les parents avec vérifications
parent_labels = []
if hasattr(concept, 'parents') and concept.parents:
    for parent in concept.parents:
        parent_label = getattr(parent, 'label', None) or self._extract_concept_label_safe(parent.uri)
        if parent_label:
            parent_labels.append(parent_label)

if parent_labels:
    concept_text += f" (type de: {', '.join(parent_labels)})"
```

**Exemple** : Si on a un concept "Electron" qui hérite de "Particule", l'embedding sera généré à partir du texte :
```
"Electron: Description de l'électron (type de: Particule)"
```

### 3. Stockage par niveau

```python
# Chaque niveau a son propre réseau de Hopfield
self.level_networks = {}  # niveau -> réseau de Hopfield

# Dans add_concept_to_hierarchy()
network = self.level_networks[concept_level]
network.store_patterns(
    np.array([concept_embedding]),
    [concept_uri]
)
```

**Architecture de stockage** :
- Niveau 1 : Concepts racines (ex: "Particule")
- Niveau 2 : Concepts enfants (ex: "Electron", "Photon")
- Niveau 3 : Concepts petits-enfants (ex: "Electron_libre", "Electron_lié")

## Héritage des concepts : Analyse détaillée

### ✅ **Héritage INFORMATIONNEL** : OUI

Les sous-concepts héritent de l'information sémantique de leurs parents :

```python
# L'embedding du concept enfant contient l'information du parent
concept_text = f"{label}"
if description:
    concept_text += f": {description}"
if parent_labels:
    concept_text += f" (type de: {', '.join(parent_labels)})"
```

**Exemple concret** :
- Parent : "Particule" → embedding("Particule: Entité physique élémentaire")
- Enfant : "Electron" → embedding("Electron: Particule chargée négativement (type de: Particule)")

### ❌ **Héritage STRUCTUREL** : NON

Chaque niveau a son propre réseau de Hopfield **indépendant** :

```python
# Les patterns ne sont PAS partagés entre niveaux
self.level_networks[1].store_patterns(...)  # Concepts niveau 1
self.level_networks[2].store_patterns(...)  # Concepts niveau 2 (séparés)
```

**Problème** : Un concept enfant n'utilise pas directement les patterns de reconnaissance de son parent.

### 🔄 **Héritage FONCTIONNEL** : PARTIEL

La classification suit une cascade hiérarchique :

```python
async def _classify_recursive(self, document_embedding, parent_concept_uri, parent_result, level, ...):
    # Si un parent est détecté, on cherche ses enfants
    level_results = await self._classify_at_level(
        document_embedding, level, top_k, parent_concept_uri, threshold
    )
```

**Mécanisme** :
1. Classification au niveau 1 (concepts généraux)
2. Pour chaque concept détecté, classification au niveau 2 (sous-concepts)
3. Récursion jusqu'aux feuilles

## Limitations du système actuel

### 1. **Fragmentation des connaissances**
```python
# Chaque niveau est isolé
level_1_network = self.level_networks[1]  # Particules
level_2_network = self.level_networks[2]  # Electrons, Photons
# ↑ Pas de partage de patterns entre ces réseaux
```

### 2. **Redondance informationnelle**
L'information des parents est répétée dans chaque embedding enfant, ce qui peut créer des biais.

### 3. **Pas de propagation d'activation**
Si "Particule" est fortement activé, cette information ne se propage pas automatiquement vers "Electron".

## Améliorations suggérées

### 1. **Héritage structurel avec pondération**
```python
# Pseudocode pour amélioration
def inherit_parent_patterns(self, child_concept, parent_concept):
    parent_patterns = self.get_parent_patterns(parent_concept)
    child_patterns = self.concept_embeddings[child_concept]
    
    # Combiner avec pondération
    inherited_embedding = 0.7 * child_patterns + 0.3 * parent_patterns
    return inherited_embedding
```

### 2. **Réseau unifié avec masquage hiérarchique**
```python
# Un seul réseau avec masques par niveau
unified_network = ModernHopfieldNetwork(...)
hierarchy_masks = {
    1: mask_level_1,
    2: mask_level_2,
    # ...
}
```

### 3. **Propagation d'activations**
```python
def propagate_activations(self, parent_activations, hierarchy_level):
    # Propager l'activation du parent vers les enfants
    for child in self.get_children(parent_uri):
        child_boost = parent_activations[parent_uri] * 0.5
        child_activations[child] += child_boost
```

## Conclusion

**Le système actuel implémente un héritage informationnel fort mais un héritage structurel faible.** 

- ✅ Les sous-concepts "connaissent" leurs parents via l'embedding
- ❌ Les sous-concepts n'héritent pas des patterns de reconnaissance des parents
- 🔄 La classification suit la hiérarchie mais chaque niveau est indépendant

Cette approche est **correcte mais sous-optimale** pour un système ontologique, car elle ne tire pas pleinement parti de la structure hiérarchique pour la reconnaissance de patterns.