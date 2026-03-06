from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .models import Primitive, TrackLine


class DataLoader:
    @staticmethod
    def load_primitives(path: str | Path) -> List[Primitive]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return DataLoader.load_primitives_from_raw(data)

    @staticmethod
    def load_primitives_from_raw(data: list) -> List[Primitive]:
        primitives: List[Primitive] = []
        for item in data:
            bbox = item.get("bbox") or [0, 0, 0, 0]
            keypoints = item.get("keypoints")
            parsed_keypoints = None
            if keypoints:
                parsed_keypoints = [(kp[0], kp[1], kp[2]) for kp in keypoints]
            primitives.append(
                Primitive(
                    primitive_id=item.get("id", len(primitives) + 1),
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    score=item.get("score", 0.0),
                    category_id=item.get("category_id", -1),
                    category_name=item.get("category_name", "unknown"),
                    keypoints=parsed_keypoints,
                    name=item.get("name"),
                )
            )
        return primitives

    @staticmethod
    def load_track_lines(path: str | Path) -> List[TrackLine]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return DataLoader.load_tracks_from_raw(data)

    @staticmethod
    def load_tracks_from_raw(data: list) -> List[TrackLine]:
        lines: List[TrackLine] = []
        for item in data:
            if item.get("type") != "LineString":
                continue
            coords = item.get("coordinates") or []
            if len(coords) < 2:
                continue
            start = coords[0]
            end = coords[-1]
            line_id = item.get("properties", {}).get("id", len(lines) + 1)
            lines.append(TrackLine(line_id=line_id, start=(start[0], start[1]), end=(end[0], end[1])))
        return lines
