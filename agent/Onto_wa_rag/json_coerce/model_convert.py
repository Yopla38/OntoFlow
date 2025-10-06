"""
------------------------------------------
Copyright: CEA Grenoble
Auteur: Louis BEAL
Entité: IRIG
Année: 2025
Description: Agent IA d'Intégration Continue
------------------------------------------
"""

from typing import Any, Optional
from pydantic import BaseModel


def recursive_convert(
    data: Any,
    defs: Optional[dict[Any, Any]] = None,
    indent: int = 0,
    no_wrap: bool = False,
) -> str:
    def apply_indent(string: str, extra: int = 0) -> str:
        return "  " * (indent + extra) + string

    header = {"title": None, "description": None}
    output = []
    if isinstance(data, dict):
        newdefs = data.pop("$defs", {})
        if defs is None:
            defs = newdefs
        else:
            defs.update(newdefs)

        if "properties" in data:
            data = data.get("properties", {})

        for k, v in data.items():
            if v is None:
                continue

            # add comments and metadata to "header"
            if k in ("title", "description"):
                header[k] = v
                continue

            if k == "type":
                output.append(apply_indent(f"{k}: {v}", extra=1))
                continue

            if k == "$ref":
                def_link = v.split("$defs/")[-1]

                output.append(
                    recursive_convert(
                        defs.get(def_link, {}), indent=indent, no_wrap=True
                    )
                )
            else:
                output.append(apply_indent(f"{k}:", extra=1))
                output.append(recursive_convert(v, defs=defs, indent=indent + 1))

    elif isinstance(data, (list, tuple, set)):
        output.append(apply_indent("[", extra=1))
        for item in data:
            output.append(recursive_convert(item, defs=defs, indent=indent + 2))
        output.append(apply_indent("],", extra=1))

    else:
        output.append(apply_indent(f"{data}", extra=1))

    # Wrap the data
    # first, construct the header
    tmp = [apply_indent("{")]
    if header["title"] is not None:
        tmp.append(f'name="{header["title"]}"')
    if header["description"] is not None:
        tmp.append(f'comment="{header["description"]}"')
    # now insert it at the start of the output

    if not no_wrap:
        output.insert(0, " // ".join(tmp))
        output.append(apply_indent("},"))

    return "\n".join(output)


def convert_model_to_struct(model: BaseModel.__class__) -> str:
    """
    Convert a Pydantic model to a prompt string for LLMs.

    Args:
        model (BaseModel): The Pydantic model to convert.

    Returns:
        str: The generated prompt string.
    """
    return recursive_convert(model.model_json_schema())
