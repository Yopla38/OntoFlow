from typing import Union
from pydantic import BaseModel, Field

from ..agent.Onto_wa_rag.json_coerce.model_convert import recursive_convert


class Integer(BaseModel):
    value: int = Field(..., description="stored integer")


class Float(BaseModel):
    value: float = Field(..., description="stored float")


class ModelTest(BaseModel):
    static: str = Field(..., description="static string")

    numbers: Union[
        Integer,
        Float
    ] = Field(..., description="list of stored numbers")


def test_recursive_convert():
    convert = recursive_convert(ModelTest)

    assert isinstance(convert["numbers"], list)
    assert "type=string" in convert["static"]
