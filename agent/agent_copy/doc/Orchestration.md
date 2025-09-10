# Guide d'Utilisation du Système d'Orchestration d'Agents

## Table des Matières
1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Configuration](#configuration)
4. [Agents et Rôles](#agents-et-rôles)
5. [Workflow](#workflow)
6. [Gestion des Tâches](#gestion-des-tâches)
7. [Exemples d'Utilisation](#exemples-dutilisation)
8. [Bonnes Pratiques](#bonnes-pratiques)
9. [Dépannage](#dépannage)

## Introduction

Le système d'orchestration d'agents est un framework permettant d'automatiser et de coordonner le développement de projets logiciels à l'aide d'agents basés sur des LLM (Large Language Models). Il permet une collaboration efficace entre différents agents spécialisés, chacun ayant un rôle spécifique dans le processus de développement.

### Caractéristiques Principales
- Workflow configurable et flexible
- Gestion des tâches avec dépendances
- Traitement parallèle
- Validation automatique
- Interface HTML interactive

## Architecture

### Composants Principaux
```
orchestrator/
├── agents/          # Définitions des agents
├── components/      # Gestionnaires (fichiers, tâches)
├── workflow/        # Logique d'orchestration
└── interface/       # Interface utilisateur
```

### Flux de Données
1. Description du projet → Architecte
2. Architecte → Tâches
3. Ingénieur → Distribution des tâches
4. Développeurs → Implémentation
5. Testeur → Validation
6. Inspecteur → Révision

## Configuration

### Configuration du Projet
```python
project_config = {
    "name": "nom_projet",
    "description": "Description détaillée du projet",
    "workspace": "/chemin/workspace",
    "requirements": {
        "python_version": "3.10",
        "dependencies": ["package1", "package2"]
    }
}
```

### Configuration des Agents
```python
agent_config = {
    "name": "nom_agent",
    "model": "claude-3-5-sonnet-20241022",  # ou autre modèle
    "prompt": """
    Instructions détaillées pour l'agent...
    """,
    "pydantic_model": {
        # Structure de validation des réponses
    }
}
```

## Agents et Rôles

### Rôles Disponibles

1. **Architecte**
   - Analyse initiale du projet
   - Définition de l'architecture
   - Création des tâches principales

2. **Ingénieur**
   - Décomposition des tâches
   - Attribution aux développeurs
   - Gestion des dépendances

3. **Frontend Developer**
   - Interface utilisateur
   - Interactions client
   - Styles et mise en page

4. **Backend Developer**
   - Logique métier
   - API et services
   - Gestion des données

5. **Database Developer**
   - Modèles de données
   - Migrations
   - Optimisation

## Workflow

### Configuration du Workflow
```python
workflow_config = {
    "initial_state": "analysis",
    "states": {
        "analysis": {
            "type": "processing",
            "agent": "Architecte",
            "transitions": {
                "success": "engineering",
                "error": "error"
            }
        },
        "engineering": {
            "type": "processing",
            "agent": "Ingénieur",
            "transitions": {
                "success": "development",
                "error": "error"
            }
        },
        "development": {
            "type": "parallel_workflow",
            "task_assignment": {
                "source": "engineering",
                "field_mapping": {
                    "tasks": "tasks",
                    "agent": "assigned_role"
                }
            },
            "transitions": {
                "success": "testing",
                "error": "error"
            }
        }
    }
}
```

### Types d'États
- `processing`: Traitement séquentiel par un agent
- `parallel_workflow`: Traitement parallèle de tâches

### Transitions
- `success`: Passage à l'état suivant
- `error`: Gestion des erreurs
- `needs_revision`: Retour pour révision

## Gestion des Tâches

### Structure d'une Tâche
```python
task = {
    "id": 1,
    "title": "Titre de la tâche",
    "description": "Description détaillée",
    "assigned_role": "role_agent",
    "priority": 1,
    "dependencies": [2, 3],  # IDs des tâches dont celle-ci dépend
    "files": ["chemin/fichier1", "chemin/fichier2"],
    "acceptance_criteria": [
        "Critère 1",
        "Critère 2"
    ]
}
```

### États des Tâches
- `pending`: En attente
- `in_progress`: En cours
- `completed`: Terminée
- `failed`: Échouée
- `needs_revision`: Nécessite une révision

## Exemples d'Utilisation

### Initialisation Simple
```python
from orchestrator import WorkflowOrchestrator, FileManager, TaskManager

# Création des managers
file_manager = FileManager("/chemin/projet")
task_manager = TaskManager()

# Configuration du projet
project_description = """
Description détaillée du projet...
"""

# Création de l'orchestrateur
orchestrator = WorkflowOrchestrator(
    workflow_config=workflow_config,
    agents=agents,
    project_description=project_description,
    task_manager=task_manager,
    file_manager=file_manager
)

# Exécution
results = await orchestrator.execute()
```

### Workflow Personnalisé
```python
custom_workflow = {
    "initial_state": "development",
    "states": {
        "development": {
            "type": "processing",
            "agent": "Backend_Developer",
            "transitions": {
                "success": "complete",
                "error": "error"
            }
        }
    }
}
```

## Bonnes Pratiques

### Configuration des Agents
1. Prompts clairs et spécifiques
2. Validation des données avec Pydantic
3. Gestion appropriée des erreurs

### Gestion des Tâches
1. Tâches atomiques et indépendantes
2. Dépendances clairement définies
3. Critères d'acceptation mesurables

### Workflow
1. États clairement définis
2. Transitions logiques
3. Validation à chaque étape

## Dépannage

### Problèmes Courants

1. **Erreur de Configuration**
   ```python
   # Vérification de la configuration
   if not orchestrator.validate_config():
       print(orchestrator.get_config_errors())
   ```

2. **Tâches Bloquées**
   ```python
   # Vérification des dépendances
   blocked_tasks = task_manager.get_blocked_tasks()
   print(f"Tâches bloquées: {blocked_tasks}")
   ```

3. **Erreurs d'Agent**
   ```python
   # Log détaillé des erreurs
   logging.getLogger('orchestrator').setLevel(logging.DEBUG)
   ```

### Solutions Recommandées

1. Vérifier les configurations
2. Valider les dépendances
3. Examiner les logs
4. Tester les agents individuellement

---

Ce guide est un document vivant qui sera mis à jour régulièrement avec les meilleures pratiques et les retours d'expérience.

Pour plus d'informations, consultez la documentation API complète ou contactez l'équipe de support.
