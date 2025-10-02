"""
------------------------------------------
Copyright: CEA Grenoble
Auteur: Louis BEAL
Entité: IRIG
Année: 2025
Description: Agent IA d'Intégration Continue
------------------------------------------
"""

import json
from typing import Any, Union, get_args, get_origin
from pydantic import BaseModel, Field


translate = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


def field_to_string(field: Field) -> str:
    """Converts a pydantic Field object to a string, preserving metadata and descriptions"""
    typing = f"type={translate.get(field.annotation.__name__, field.annotation.__name__)}"

    comment = ["//"]

    if not field.is_required():
        comment.append("(Optional)")

    comment += field.metadata

    if field.description is not None and field.description != "":
        comment.append(field.description)

    if len(comment) > 1:
        return f"{typing}  {' '.join(comment)}"
    return typing


def recursive_convert(model: BaseModel.__class__) -> dict[str, Any]:

    print(f"Model converting {model}, {type(model)}")

    struct = {}
    for field_name, field in model.model_fields.items():
        annotation = field.annotation
        
        # if we hit another nested model, immediately recurse
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            struct[field_name] = recursive_convert(annotation)

        # a Union of objects needs to be expanded
        if get_origin(annotation) is Union:
            tmp = []
            for item in get_args(annotation):
                tmp.append(recursive_convert(item))
            struct[field_name] = tmp

        # otherwise, just store the name
        else:
            struct[field_name] = field_to_string(field=field)

    return struct


def convert_model_to_struct(model: BaseModel.__class__) -> str:
    """
    Convert a Pydantic model to a prompt string for LLMs.

    Args:
        model (BaseModel): The Pydantic model to convert.

    Returns:
        str: The generated prompt string.
    """
    return json.dumps(recursive_convert(model), indent=2)
