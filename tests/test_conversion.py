import json
import pytest

from pydantic import BaseModel, Field
from ..agent.Onto_wa_rag.json_coerce.model_convert import convert_model_to_struct


class SimpleModel(BaseModel):
    name: str


class LargeModel(BaseModel):
    name: str
    age: int
    height: float
    is_student: bool
    hobbies: list[str]

class FieldModel(BaseModel):    
    thought: str = Field(..., description="My thoughts on the current situation.")


@pytest.mark.parametrize("model", [SimpleModel, LargeModel])
def test_model_conversion(model: BaseModel.__class__) -> None:
    structure = convert_model_to_struct(model)
    assert structure.startswith("{")
    assert structure.endswith("}")

    # clean = "\n".join(line for line in structure.split("\n") if "//" not in line)

    as_dict = json.loads(structure)

    for name, field in model.model_json_schema().get("properties", {}).items():
        assert name in as_dict
        assert as_dict[name].split("type=")[-1] == field.get("type")


def test_field_model() -> None:
    structure = convert_model_to_struct(FieldModel)

    assert "My thoughts on the current situation." in structure
