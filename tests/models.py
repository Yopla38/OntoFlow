from typing import Literal, Optional, Union
from pydantic import BaseModel, Field


class SimpleModel(BaseModel):
    name: str

    def decomp(self) -> str:
        return """
{
  "name": "str"
}
"""


class LargeModel(BaseModel):
    name: str
    age: int
    height: float
    is_student: bool
    hobbies: list[str]

    def decomp(self) -> str:
        return """
{
  "name": "str",
  "age": "int",
  "height": "float",
  "is_student": "bool",
  "hobbies": List[str]
}
"""


class FieldModel(BaseModel):    
    thought: str = Field(description="My thoughts on the current situation.")

    def decomp(self) -> str:
        return """
{
  "thought": "str"  // My thoughts on the current situation.
}
"""


class Integer(BaseModel):
    value: int = Field(description="stored integer")

    def decomp(self):
        return """
{
  "value": "int"  // stored integer
}
"""


class Float(BaseModel):
    value: float = Field(description="stored float")

    def decomp(self):
        return """
{
  "value": "float"  // stored float
}
"""


class RecursiveTest(BaseModel):
    static: str = Field(..., description="static string")

    numbers: Union[
        Integer,
        Float
    ] = Field(..., description="list of stored numbers")

    def decomp(self):
        return """
{
  "static": "string",  // static string
  "numbers": [  // list of stored numbers
    {
      "value": "int" // stored integer
    },
    {
      "value": "float" // stored float
    }
  ]
}
"""


class LiteralTest(BaseModel):
    food: Literal["apple", "banana", "cucumber"] = Field(description="food item")

    def decomp(self):
        return """
{
  "food": [ Literal  // food item
    "apple",
    "banana",
    "cucumber"
  ]
}
"""


class OptionalLiteralTest(BaseModel):
    food: Optional[Literal["apple", "banana", "cucumber"]] = Field(description="food item")

    def decomp(self):
        return """
{
  "food": [ Literal  // (Optional) food item
    "apple",
    "banana",
    "cucumber"
  ]
}
"""
