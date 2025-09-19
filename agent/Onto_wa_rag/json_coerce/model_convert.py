"""
------------------------------------------
Copyright: CEA Grenoble
Auteur: Louis BEAL
Entité: IRIG
Année: 2025
Description: Agent IA d'Intégration Continue
------------------------------------------
"""

from pydantic import BaseModel


def convert_model_to_struct(model: BaseModel.__class__) -> str:
    """
    Convert a Pydantic model to a prompt string for LLMs.

    Args:
        model (BaseModel): The Pydantic model to convert.

    Returns:
        str: The generated prompt string.
    """
    output = ["{"]
    for i, (field_name, field) in enumerate(model.model_fields.items()):
        # first, we convert any metadata into a "comment" for the LLM
        comment = ["  //"]
        if not field.is_required():
            comment.append("(Optional)")
        comment += field.metadata
        if len(comment) > 1:
            output.append(" ".join(comment))

        if field.annotation is None:
            raise ValueError(f"Field {field_name} has no type annotation")
        # now, create the actual field
        content = [f'  "{field_name}": "{field.annotation.__name__}"']
        # add a comma for all but the last entry
        if i < len(model.model_fields) - 1:
            content.append(",")
        # add this entry and continue
        output.append("".join(content))

    output.append("}")
    return "\n".join(output)
