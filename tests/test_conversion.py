import os
import sys
import inspect
import pytest

from pydantic import BaseModel
from ..agent.Onto_wa_rag.json_coerce.model_convert import convert_model_to_struct

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
    
from . import models


def all_pydantic_models(module) -> list[type[BaseModel]]:
    """
    Return a list of *classes* defined in ``module`` that inherit from
    ``pydantic.BaseModel`` (excluding BaseModel itself).
    """
    return [
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, BaseModel) and cls is not BaseModel
    ]


@pytest.mark.parametrize("model", all_pydantic_models(models))
def test_model_conversion(model: BaseModel.__class__) -> None:
    print(f"Testing model: {model}")
    if not hasattr(model, "decomp"):
        raise ValueError(f"Model {model} is missing a decomp")

    assert convert_model_to_struct(model) == model.decomp(model).strip()
