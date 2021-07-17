from pydantic import BaseModel
from typing import List, Union


class RequestModel(BaseModel):
    image: str


class ResponseModel(BaseModel):
    bbox: List[int]
    id: Union[int, None]
    name: str
    surname: str
