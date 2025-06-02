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


class KinematicRelationshipType(BaseModel):
    """Enumeration of standard kinematic relationship types between parts."""

    FIXED: str = "fixed/rigid"
    REVOLUTE: str = "revolute"
    PRISMATIC: str = "prismatic"
    CYLINDRICAL: str = "cylindrical"
    SPHERICAL: str = "spherical"
    PLANAR: str = "planar"
    PRESS: str = "press/spring-loaded"
    SUPPORTED: str = "supported"
    UNRELATED: str = "unrelated"
    OTHER: str = "other"


class KinematicRelationship(BaseModel):
    """Model for representing kinematic relationships between two parts."""

    part1_id: str = Field(..., description="ID of the first part")
    part2_id: str = Field(..., description="ID of the second part")
    part1_name: str = Field(..., description="Name of the first part")
    part2_name: str = Field(..., description="Name of the second part")
    relationship_type: KinematicRelationshipType = Field(
        ..., description="Type of kinematic relationship between the parts"
    )
    movement_axis: Optional[str] = Field(
        None, description="Axis or direction of movement, if applicable"
    )
    is_static: Optional[bool] = Field(
        None, description="Whether the connection is designed to be static"
    )
    purpose: Optional[str] = Field(
        None, description="Purpose of this specific connection in the overall function"
    )
    description: str = Field(
        ..., description="Detailed description of the kinematic relationship"
    )
