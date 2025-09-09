# Guide d'Utilisation des Tâches d'Agent

## Structure des Tâches

Chaque tâche doit suivre une structure précise avec des mots-clés spécifiques. Ces mots-clés sont essentiels pour le bon fonctionnement du système.

### Mots-clés Obligatoires

```python
{
    "id": int,          # Identifiant unique de la tâche
    "title": str,       # Titre court et descriptif
    "description": str, # Description détaillée
    "files": List[str], # Liste des chemins de fichiers
}
```

### Format Complet d'une Tâche

```python
{
    # Champs d'identification
    "id": int,                    # Ex: 1
    "title": str,                 # Ex: "Création du modèle de calculatrice"
    "description": str,           # Ex: "Implémenter les classes de base..."
    "assigned_role": str,         # Ex: "Codeur", "Designer_Interface", "Testeur"

    # Fichiers et Structure
    "files": List[str],          # Ex: ["app/models/calculator.py", "app/models/operations.py"]
                                # IMPORTANT: Utilisez toujours des chemins relatifs avec '/'

    # Dépendances et Paquets
    "dependencies": List[int],   # Ex: [1, 2] (IDs des tâches dont celle-ci dépend)
    "additional_packages": List[str],  # Ex: ["numpy", "pandas"]

    # Validation
    "acceptance_criteria": List[str]  # Ex: ["Tests unitaires complets", "Documentation complète"]
}
```

## Exemples de Tâches

### Exemple 1 : Tâche de Développement créée par l'agent principal

```python
{
    "id": 1,
    "title": "Implémentation du modèle de calculatrice",
    "description": "Créer les classes de base pour la calculatrice scientifique avec les opérations fondamentales",
    "assigned_role": "Codeur",
    "files": [
        "app/models/calculator.py",
        "app/models/operations.py"
    ],
    "additional_packages": ["math", "decimal"],
    "acceptance_criteria": [
        "Toutes les opérations de base implémentées",
        "Tests unitaires présents",
        "Documentation complète"
    ]
}
```

### Exemple 2 : Tâche d'Interface créée par l'agent principal

```python
{
    "id": 2,
    "title": "Création de l'interface utilisateur",
    "description": "Développer l'interface web avec Flask pour la calculatrice",
    "assigned_role": "Designer_Interface",
    "files": [
        "templates/calculator.html",
        "static/css/style.css",
        "static/js/calculator.js"
    ],
    "dependencies": [1],
    "additional_packages": ["flask", "bootstrap-flask"],
    "acceptance_criteria": [
        "Interface responsive",
        "Thème sombre implémenté",
        "Tous les boutons fonctionnels"
    ]
}
```

## Règles Importantes

1. **Chemins de Fichiers**
   - Utilisez toujours des chemins relatifs
   - Utilisez '/' comme séparateur (même sous Windows)
   - Respectez la structure du projet

2. **Rôles Disponibles**
   - "Architecte"
   - "Codeur"
   - "Designer_Interface"
   - "Testeur"

3. **Dépendances**
   - Utilisez les IDs des tâches dont dépend votre tâche
   - Assurez-vous que les tâches dépendantes existent

4. **Paquets Additionnels**
   - Spécifiez uniquement les paquets Python nécessaires
   - Utilisez les noms exacts des paquets PyPI

## Bonnes Pratiques

1. **Description**
   - Soyez précis et détaillé
   - Incluez le contexte nécessaire
   - Expliquez les interactions avec d'autres composants

2. **Fichiers**
   - Listez tous les fichiers nécessaires
   - Respectez la structure du projet
   - Incluez les fichiers de test si nécessaire

3. **Critères d'Acceptation**
   - Soyez spécifique et mesurable
   - Incluez les critères de qualité
   - Spécifiez les tests requis

## Validation des Tâches

Le système vérifiera automatiquement :
1. La présence de tous les champs obligatoires
2. La validité des chemins de fichiers
3. L'existence des dépendances
4. La cohérence des rôles assignés


Voir fichier improvment_agent_for_MIA.py pour exemple d'agents simples
Voir fichier agent_bigdft.py pour exemple d'agents plus complexe