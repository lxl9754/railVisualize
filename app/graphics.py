from __future__ import annotations

from typing import Dict, List

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
)

from .models import Primitive, TrackLine


CATEGORY_STYLES = {
    "xsignal": {"color": QColor(220, 60, 60), "show_keypoints": True},
    "dsignal": {"color": QColor(60, 90, 220), "show_keypoints": True},
    "switch": {"color": QColor(60, 160, 90), "show_keypoints": False},
    "insulation": {"color": QColor(240, 200, 40), "show_keypoints": True},
}

TRACK_STYLE = {"color": QColor(0, 200, 200, 140), "width": 3}


class SceneRenderer:
    def __init__(self, scene: QGraphicsScene) -> None:
        self._scene = scene
        self._track_items: List[QGraphicsLineItem] = []
        self._primitive_items: Dict[str, List[QGraphicsRectItem]] = {
            key: [] for key in CATEGORY_STYLES
        }
        self._keypoint_items: Dict[str, List[QGraphicsEllipseItem]] = {
            key: [] for key in CATEGORY_STYLES
        }

    def clear_tracks(self) -> None:
        for item in self._track_items:
            self._scene.removeItem(item)
        self._track_items.clear()

    def clear_primitives(self) -> None:
        for items in self._primitive_items.values():
            for item in items:
                self._scene.removeItem(item)
        for items in self._keypoint_items.values():
            for item in items:
                self._scene.removeItem(item)
        for key in self._primitive_items:
            self._primitive_items[key].clear()
        for key in self._keypoint_items:
            self._keypoint_items[key].clear()

    def clear(self) -> None:
        self.clear_tracks()
        self.clear_primitives()

    def render_tracks(self, lines: List[TrackLine]) -> None:
        pen = QPen(TRACK_STYLE["color"], TRACK_STYLE["width"], Qt.SolidLine)
        for line in lines:
            item = QGraphicsLineItem(line.start[0], line.start[1], line.end[0], line.end[1])
            item.setPen(pen)
            item.setZValue(1)
            self._scene.addItem(item)
            self._track_items.append(item)

    def render_primitives(self, primitives: List[Primitive]) -> None:
        for primitive in primitives:
            style = CATEGORY_STYLES.get(primitive.category_name)
            if not style:
                continue
            color = style["color"]
            rect_pen = QPen(color, 2, Qt.SolidLine)
            rect = QGraphicsRectItem(
                primitive.bbox[0],
                primitive.bbox[1],
                primitive.bbox[2] - primitive.bbox[0],
                primitive.bbox[3] - primitive.bbox[1],
            )
            rect.setPen(rect_pen)
            rect.setZValue(2)
            self._scene.addItem(rect)
            self._primitive_items[primitive.category_name].append(rect)

            if style["show_keypoints"] and primitive.keypoints:
                brush = QBrush(color)
                for keypoint in primitive.keypoints:
                    if keypoint[2] < 0.5:
                        continue
                    ellipse = QGraphicsEllipseItem(keypoint[0] - 3, keypoint[1] - 3, 6, 6)
                    ellipse.setBrush(brush)
                    ellipse.setPen(QPen(Qt.NoPen))
                    ellipse.setZValue(3)
                    self._scene.addItem(ellipse)
                    self._keypoint_items[primitive.category_name].append(ellipse)

    def clear_category(self, category: str) -> None:
        for item in self._primitive_items.get(category, []):
            self._scene.removeItem(item)
        for item in self._keypoint_items.get(category, []):
            self._scene.removeItem(item)
        if category in self._primitive_items:
            self._primitive_items[category].clear()
        if category in self._keypoint_items:
            self._keypoint_items[category].clear()

    def set_tracks_visible(self, visible: bool) -> None:
        for item in self._track_items:
            item.setVisible(visible)

    def set_category_visible(self, category: str, visible: bool) -> None:
        for item in self._primitive_items.get(category, []):
            item.setVisible(visible)
        for item in self._keypoint_items.get(category, []):
            item.setVisible(visible)

    def has_tracks(self) -> bool:
        return bool(self._track_items)

    def has_category_items(self, category: str) -> bool:
        return bool(self._primitive_items.get(category))
