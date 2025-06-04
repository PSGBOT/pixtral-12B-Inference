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


class KinematicJoint(BaseModel):
    """Enumeration of standard kinematic relationship types between parts."""

    joint_type: str
    joint_movement_axis: str
    is_static: str
    purpose: str


class KinematicRelationship(BaseModel):
    """Model for representing kinematic relationships between two parts."""

    part1_name: str = Field(..., description="Name of the first part")
    part2_name: str = Field(..., description="Name of the second part")
    kinematic_joints: list[KinematicJoint]
    root_part_id: Optional[str] = Field(
        None, description="Name of the part that acts as the kinematic root"
    )
