from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TrackLine:
    line_id: int
    start: Tuple[float, float]
    end: Tuple[float, float]


@dataclass
class Primitive:
    primitive_id: int
    bbox: Tuple[float, float, float, float]
    score: float
    category_id: int
    category_name: str
    keypoints: Optional[List[Tuple[float, float, float]]] = None

