from pydantic import BaseModel, Field
from typing import Optional, List
import enum


class Instance(BaseModel):
    valid: str
    name: Optional[str] = None
    feature: Optional[List[str]] = Field(default_factory=list)
    usage: Optional[List[str]] = Field(default_factory=list)


class Part(BaseModel):
    name: str
    purpose: str
    text: str


class KinematicJointType(enum.Enum):
    fixed = "fixed"
    revolute = "revolute"
    prismatic = "prismatic"
    cylindrical = "cylindrical"
    spherical = "spherical"
    planar = "planar"
    press = "press"
    supported = "supported"
    unrelated = "unrelated"


class KinematicRoot(enum.Enum):
    PART_0 = "0"
    PART_1 = "1"


class KinematicJoint(BaseModel):
    """Enumeration of standard kinematic relationship types between parts."""

    joint_type: KinematicJointType
    joint_movement_axis: str
    is_static: str
    purpose: str
    root: KinematicRoot


class KinematicRelationship(BaseModel):
    """Model for representing kinematic relationships between two parts."""

    part1_name: str = Field(..., description="Name of the first part")
    part2_name: str = Field(..., description="Name of the second part")
    kinematic_joints: list[KinematicJoint]
