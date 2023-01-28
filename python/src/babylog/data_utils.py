from typing import Dict, TypedDict


class BoundingBoxDict(TypedDict):
    x: int
    y: int
    width: int
    height: int
    confidence: float
    classification: Dict[str, float]
