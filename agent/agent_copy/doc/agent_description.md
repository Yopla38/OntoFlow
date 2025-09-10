# Guide de Configuration des Agents

## Introduction
Ce document décrit les règles et structures nécessaires pour configurer correctement les agents dans le système d'orchestration. La configuration correcte des modèles Pydantic est essentielle pour le bon fonctionnement du système.

## Structure de Base d'un Agent

### Configuration de l'Agent (dans type/role_definition.py)
```python
agent_config = {
    "name": str,           # Nom unique de l'agent
    "model": str,          # Modèle LLM à utiliser (ex: "claude-3-5-sonnet-20241022")
    "prompt": str,         # Instructions système pour l'agent
    "pydantic_model": Dict # Structure de réponse attendue
}
```

## Règles de Structure Pydantic

### 1. Structure de Base
- Chaque modèle Pydantic doit être un dictionnaire définissant la structure de données attendue
- Les types de base supportés sont : "str", "int", "bool", "float"
- Les structures complexes peuvent être des dictionnaires ou des listes

### 2. Champs Spéciaux
Pour que le code soit correctement généré et sauvegardé, certains champs doivent suivre une structure spécifique :

#### 2.1 Fichiers de Code
```python
"files": [
    {
        "filename": "str",
        "code_field": {
            "language": "str",
            "content": "str"
        }
    }
]
```

#### 2.2 Champs Interactifs
Pour les champs nécessitant une interaction utilisateur :
```python
"input_field": {
    "field_name": "str",
    "input_placeholder": "str",
    "is_editable": bool,
    "is_required": bool
}
```

### 3. Règles de Nommage
- Les noms de champs doivent être en snake_case
- Les champs contenant du code doivent toujours avoir un wrapper "code_field"
- Les listes doivent contenir des objets de structure uniforme

## Génération HTML

Le système génère automatiquement des interfaces HTML basées sur la structure Pydantic. Pour une génération correcte :

1. Les champs de code seront affichés avec coloration syntaxique
2. Les champs interactifs seront convertis en formulaires
3. Les structures imbriquées seront présentées hiérarchiquement

## Validation des Réponses

- Les réponses des LLM doivent correspondre exactement à la structure Pydantic définie
- Les champs obligatoires doivent toujours être présents
- Les types de données doivent correspondre aux types déclarés

## Exemple Concret d'Orchestration

Voici un exemple d'orchestration pour un système de développement logiciel :

```python
roles = {
    "AnalysteBesoins": {
        "name": "analyste",
        "model": "claude-3-5-sonnet-20241022",
        "prompt": "Vous êtes un analyste qui décompose les besoins en spécifications détaillées.",
        "pydantic_model": {
            "analysis": {
                "requirements": ["str"],
                "specifications": ["str"]
            },
            "questions": [
                {
                    "input_field": {
                        "field_name": "str",
                        "input_placeholder": "str",
                        "is_editable": True,
                        "is_required": True
                    }
                }
            ]
        }
    },
    "GenerateurCode": {
        "name": "generateur",
        "model": "claude-3-5-sonnet-20241022",
        "prompt": "Vous êtes un générateur de code expert.",
        "pydantic_model": {
            "files": [
                {
                    "filename": "str",
                    "code_field": {
                        "language": "str",
                        "content": "str"
                    }
                }
            ],
            "documentation": {
                "overview": "str",
                "usage": ["str"]
            }
        }
    },
    "Validateur": {
        "name": "validateur",
        "model": "claude-3-5-sonnet-20241022",
        "prompt": "Vous êtes un expert en validation de code.",
        "pydantic_model": {
            "validation_result": {
                "is_valid": "bool",
                "issues": ["str"],
                "suggestions": ["str"]
            },
            "files": [
                {
                    "filename": "str",
                    "code_field": {
                        "language": "str",
                        "content": "str"
                    }
                }
            ]
        }
    }
}

workflow_config = {
    "initial_state": "analysis",
    "states": {
        "analysis": {
            "type": "processing",
            "agent": "AnalysteBesoins",
            "transitions": {
                "success": "generation",
                "error": "error"
            }
        },
        "generation": {
            "type": "processing",
            "agent": "GenerateurCode",
            "transitions": {
                "success": "validation",
                "error": "error"
            }
        },
        "validation": {
            "type": "validation",
            "agent": "Validateur",
            "transitions": {
                "validated": "complete",
                "needs_revision": "generation"
            }
        }
    }
}
```

Cette configuration crée un workflow où :
1. L'analyste décompose les besoins
2. Le générateur crée le code
3. Le validateur vérifie le code et peut demander des révisions

Chaque agent utilise un modèle Pydantic spécifique qui définit exactement la structure de ses réponses et interactions.