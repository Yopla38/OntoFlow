"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# src/models/pydantic_models.py
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, create_model


def get_type(type_str: str) -> Any:
    local_vars = {
        'List': List,
        'Dict': Dict,
        'Optional': Optional,
        'Any': Any,
        'str': str,
        'int': int,
        'bool': bool,
        'float': float,
    }
    try:
        return eval(type_str, {"__builtins__": None}, local_vars)
    except Exception:
        raise ValueError(f"Type non reconnu: {type_str}")


def create_pydantic_model(name: str, fields: Any, definitions: Dict[str, Any] = None) -> BaseModel:
    if definitions is None:
        definitions = {}
    if isinstance(fields, dict):
        annotations = {}
        for field_name, field_type in fields.items():
            if isinstance(field_type, str):
                annotations[field_name] = (get_type(field_type), ...)
            elif isinstance(field_type, dict):
                sub_model = create_pydantic_model(f"{name}_{field_name}", field_type, definitions)
                annotations[field_name] = (sub_model, ...)
            elif isinstance(field_type, list):
                if len(field_type) == 1:
                    list_item_type = field_type[0]
                    if isinstance(list_item_type, str):
                        annotations[field_name] = (List[get_type(list_item_type)], ...)
                    elif isinstance(list_item_type, dict):
                        list_item_model = create_pydantic_model(f"{name}_{field_name}_Item", list_item_type,
                                                                definitions)
                        annotations[field_name] = (List[list_item_model], ...)
                    else:
                        raise ValueError(
                            f"Type d'élément de liste non supporté pour le champ {field_name}: {list_item_type}")
                else:
                    raise ValueError(f"Le champ de liste {field_name} doit contenir exactement une définition de type")
            else:
                raise ValueError(f"Type de champ non supporté pour le champ {field_name}: {field_type}")
        model = create_model(name, **annotations)
        definitions[name] = model
        return model
    else:
        raise ValueError(f"Définition de champ non supportée: {fields}")


class ModelGenerator:
    @staticmethod
    def create_model(name: str, schema: Dict[str, Any]) -> BaseModel:
        return create_pydantic_model(name, schema)

    @staticmethod
    def load_role_models(roles_config: Dict[str, Any]) -> Dict[str, BaseModel]:
        models = {}
        for role, config in roles_config.items():
            if "pydantic_model" in config:
                models[role] = ModelGenerator.create_model(
                    role,
                    config["pydantic_model"]
                )
        return models


# Modèles Pydantic pour les function calls du super agent
class SetupProjectSchema(BaseModel):
    project_path: str
    description: Optional[str] = None


class RecruitAgentSchema(BaseModel):
    project_path: str
    requirement: str
    priority: Optional[str] = "normal"


class WorkflowStepSchema(BaseModel):
    role: str
    task: str
    conditions: Optional[Dict[str, Any]] = None


class CreateWorkflowSchema(BaseModel):
    project_path: str
    workflow_name: str
    description: str
    steps: List[WorkflowStepSchema]


class AgentInteractionSchema(BaseModel):
    agent_id: str
    message: str
    priority: Optional[str] = "normal"


class ProjectStatusSchema(BaseModel):
    project_path: str
    include_agents: Optional[bool] = True
    include_workflows: Optional[bool] = True


if __name__ == "__main__":
    bigdft_simulation_structure = {
        "system": {
            "composition": ["str"],
            "dimensions": ["float"],
            "structure": "str"
        },
        "calculation": {
            "functional": "str",
            "precision": "float",
            "convergence": "float"
        },
        "resources": {
            "cores": "int",
            "memory": "float",
            "time": "float"
        },
        "analysis": {
            "properties": ["str"],
            "visualization": ["str"]
        }
    }

    # Création du modèle Pydantic
    pydantic_model = ModelGenerator.create_model(
        "BigDFTSimulationInfo",
        bigdft_simulation_structure
    )

    json_str = pydantic_model.json()
    print(json_str)