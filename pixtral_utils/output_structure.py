from pydantic import BaseModel


class Instance(BaseModel):
    name: str
    usage: str


class Part(BaseModel):
    name: str
    purpose: str
    text: str
