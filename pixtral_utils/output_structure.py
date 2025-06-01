from pydantic import BaseModel, Field
from typing import Optional, List


class Instance(BaseModel):
    valid: str
    name: Optional[str] = None
    feature: Optional[List[str]] = Field(default_factory=list)
    usage: Optional[List[str]] = Field(default_factory=list)


class Part(BaseModel):
    name: str
    purpose: str
    text: str
