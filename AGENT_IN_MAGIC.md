## Étape 4 : Générer du Code avec le Workflow Agent (Plan & Exécution)

Nous allons maintenant utiliser notre système d'agents avancé. Le processus se déroule en deux temps : d'abord, un agent "Planificateur" vous propose une stratégie. Ensuite, si vous la validez, un agent "Développeur" écrit le code pour vous.

**Cellule 4 : Charger la Magic de l'Agent**

```python
%load_ext Agent_with_magics
```

**Cellule 5 : Demander un Plan d'Action**

Décrivez votre objectif à l'agent. Il va analyser votre demande, consulter la documentation via le RAG et vous proposer un plan.

```python
%%agent
Je veux créer un script pour une simulation de dynamique moléculaire d'une molécule H2.
```

> **Sortie attendue :**
> `🧠 Agent Planificateur : Analyse de la demande et consultation du RAG...`
> ### 📋 Plan de l'Agent
> L'agent propose le plan d'action suivant :
> ### Plan d'action pour la simulation
> 1.  **Importer les bibliothèques** : Importer `System` et `Calculator` de `BigDFT`...
> 2.  **Définir le système moléculaire** : Créer un objet `System` et y ajouter deux atomes d'hydrogène.
>     ... (le reste du plan) ...
>
> **Si ce plan vous convient, exécutez la commande suivante dans une nouvelle cellule :**
> ```
> %agent /execute_plan
> ```

**Cellule 6 : Valider le Plan et Lancer la Génération**

Vous avez examiné le plan et il vous semble correct. Donnez le feu vert à l'agent Développeur.

```python
%agent /execute_plan
```

> **Sortie attendue :**
> `✍️ Agent Développeur : Réception du plan et génération du code...`
> `Code écrit dans le fichier temporaire : /tmp/tmpxxxxx.py`
> ### ✅ Code Généré
> Le plan a été exécuté. Une nouvelle cellule a été créée ci-dessous avec le code.

**Cellule 7 (Générée automatiquement)**

Comme promis, une nouvelle cellule contenant le code final apparaît, prête à être exécutée.

```python
# Code généré par l'Agent Développeur
# Conforme au plan d'action validé.

# Tâche 1: Importer les bibliothèques
from BigDFT.Systems import System
# ... reste du code ...
```