# Define your output structure
"""
------------------------------------------
Copyright: CEA Grenoble
Auteur: Louis BEAL
Entité: IRIG
Année: 2025
Description: Agent IA d'Intégration Continue
------------------------------------------
"""

import ast
from typing import Annotated
from openai import BaseModel
from pydantic import field_validator


class GeneratedFunction(BaseModel):
    name: Annotated[str, "Name of the function"]
    source: Annotated[
        str,
        "Python source code of the function, all imports MUST be within the body, no external dependencies",
    ]

    @field_validator("source")
    def validate_source_code(cls, v: str) -> str:
        try:
            ast.parse(v)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python code: {e}")

        # validate that the imports are within the function body
        body = False
        for line in v.splitlines():
            if "def " in line:
                body = True
            elif "\t" not in line and "    " not in line:
                body = False

            if "import " in line and not body:
                raise ValueError("Imports must be within the body of the function")

        return v
