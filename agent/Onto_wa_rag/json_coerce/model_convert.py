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
from pydantic import BaseModel
from pydantic.fields import FieldInfo


translate = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}

def translate_type(name: str) -> str:
    if isinstance(name, type):
        name = name.__name__
    return translate.get(name, name)


def field_to_string(field: FieldInfo) -> str:
    """Converts a pydantic Field object to a string, preserving metadata and descriptions"""
    print(f"\t\t\tCasting {field}, {type(field)} to str")

    if hasattr(field, "annotation") and field.annotation is not None:
        typing = f"type={translate_type(field.annotation.__name__)}"
    else:
        typing = str(field)

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

    print(f"Model converting {model.__name__}, {model}")

    struct = {}
    for field_name, field in model.model_fields.items():
        print(f"\tHandling field {field_name}")
        annotation = field.annotation
        
        # if we hit another nested model, immediately recurse
        if isinstance(annotation, type):        
            if issubclass(annotation, BaseModel):
                struct[field_name] = recursive_convert(annotation)
            else:
                struct[field_name] = annotation.__name__

        # a Union of objects needs to be expanded
        elif len(get_args(annotation)) > 0:
            tmp = []
            for item in get_args(annotation):
                if item is None:
                    continue

                if isinstance(item, type):
                    if issubclass(item, BaseModel):
                        tmp.append(recursive_convert(item))
                    else:
                        tmp.append(translate_type(item.__name__))
                else:
                    tmp.append(str(item))

            struct[field_name] = tmp

        elif isinstance(field, FieldInfo):
            struct[field_name] = field_to_string(field)

        # otherwise, just store the name
        else:
            print("\t\tFallback cast annotation to string")
            struct[field_name] = str(annotation)

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
