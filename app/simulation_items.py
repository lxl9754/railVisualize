from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF, QBrush, QFont, QFontMetricsF
from PySide6.QtWidgets import QGraphicsItem, QGraphicsScene

from .models import Primitive, TrackLine

TRACK_COLOR = QColor(0, 100, 255)
POLE_COLOR = QColor(0, 85, 127)  # 信号机柱的深蓝色
PRIMITIVE_SCALE = 2.0


class TrackItem(QGraphicsItem):
    def __init__(self, start: QPointF, end: QPointF, parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self._start = start
        self._end = end
        self.setCacheMode(QGraphicsItem.ItemCoordinateCache)

    def boundingRect(self) -> QRectF:
        pen_width = 2
        left = min(self._start.x(), self._end.x())
        top = min(self._start.y(), self._end.y())
        right = max(self._start.x(), self._end.x())
        bottom = max(self._start.y(), self._end.y())
        return QRectF(left, top, right - left, bottom - top).adjusted(-pen_width, -pen_width, pen_width, pen_width)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        pen = QPen(TRACK_COLOR, 2, Qt.SolidLine)
        painter.setPen(pen)
        painter.drawLine(self._start, self._end)


class InsulationItem(QGraphicsItem):
    def __init__(self, center: QPointF, parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self._center = center
        self._half_height = 5 * PRIMITIVE_SCALE
        self.setCacheMode(QGraphicsItem.ItemCoordinateCache)

    def boundingRect(self) -> QRectF:
        pad = 2 * PRIMITIVE_SCALE
        return QRectF(
            self._center.x() - pad,
            self._center.y() - self._half_height - pad,
            pad * 2,
            self._half_height * 2 + pad * 2,
        )

    def paint(self, painter: QPainter, option, widget=None) -> None:
        pen = QPen(Qt.cyan, 2 * PRIMITIVE_SCALE)
        painter.setPen(pen)
        painter.drawLine(
            QPointF(self._center.x(), self._center.y() - self._half_height),
            QPointF(self._center.x(), self._center.y() + self._half_height),
        )


class SwitchItem(QGraphicsItem):
    def __init__(self, center: QPointF, parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self._center = center
        self._half_base = 4 * PRIMITIVE_SCALE
        self._height = 8 * PRIMITIVE_SCALE
        self.setCacheMode(QGraphicsItem.ItemCoordinateCache)

    def boundingRect(self) -> QRectF:
        pad = 1 * PRIMITIVE_SCALE
        return QRectF(
            self._center.x() - self._half_base - pad,
            self._center.y() - self._height - pad,
            self._half_base * 2 + pad * 2,
            self._height + pad * 2,
        )

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 220, 0))
        points = QPolygonF(
            [
                QPointF(self._center.x() - self._half_base, self._center.y()),
                QPointF(self._center.x() + self._half_base, self._center.y()),
                QPointF(self._center.x(), self._center.y() - self._height),
            ]
        )
        painter.drawPolygon(points)


@dataclass
class SignalStyle:
    # 现在只需要定义内侧圆的颜色，外侧默认透明
    inner_color: QColor


class SignalItem(QGraphicsItem):
    def __init__(
            self,
            pole_top: QPointF,
            pole_bottom: QPointF,
            indicator: QPointF,
            style: SignalStyle,
            parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self._pole_top = pole_top
        self._pole_bottom = pole_bottom
        self._indicator = indicator
        self._style = style

        # === 核心几何参数微调 ===
        self._pole_length = 16 * PRIMITIVE_SCALE  # 机柱的固定长度
        self._radius = self._pole_length / 2  # 圆的半径 = 机柱长度的一半 (直径等于机柱)
        self._pole_width = 2 * PRIMITIVE_SCALE  # 机柱的线宽
        # =========================

        # 计算真正的机柱中心点
        self._center = QPointF(
            (self._pole_top.x() + self._pole_bottom.x()) / 2,
            (self._pole_top.y() + self._pole_bottom.y()) / 2,
        )

        # 判断朝向：指示点(灯位)在机柱中心的左边还是右边
        # 1.0 表示朝右画，-1.0 表示朝左画
        self._direction = 1.0 if self._indicator.x() > self._center.x() else -1.0

        self.setCacheMode(QGraphicsItem.ItemCoordinateCache)

    def boundingRect(self) -> QRectF:
        # 精确计算包围盒，防止残影
        margin = 3 * PRIMITIVE_SCALE  # 边缘冗余量，包容粗边框

        if self._direction > 0:
            left = self._center.x() - self._pole_width / 2 - margin
            right = self._center.x() + self._pole_width / 2 + self._radius * 4 + 2 + margin
        else:
            left = self._center.x() - self._pole_width / 2 - self._radius * 4 - 2 - margin
            right = self._center.x() + self._pole_width / 2 + margin

        top = self._center.y() - self._radius - margin
        bottom = self._center.y() + self._radius + margin

        return QRectF(left, top, right - left, bottom - top)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. 画机柱 (深蓝色垂直线)
        pole_pen = QPen(POLE_COLOR, self._pole_width)
        painter.setPen(pole_pen)
        p1 = QPointF(self._center.x(), self._center.y() - self._pole_length / 2)
        p2 = QPointF(self._center.x(), self._center.y() + self._pole_length / 2)
        painter.drawLine(p1, p2)

        # 2. 计算圆心位置 (解除遮挡的关键)
        # 间距 = 机柱宽度的一半 + 1像素的安全间隙 + 圆的半径
        offset = (self._pole_width / 2) + 1.0 + self._radius

        inner_cx = self._center.x() + self._direction * offset
        outer_cx = inner_cx + self._direction * (self._radius * 2)

        inner_center = QPointF(inner_cx, self._center.y())
        outer_center = QPointF(outer_cx, self._center.y())

        # 3. 画圆
        # 粗一点的浅蓝色边框
        circle_pen = QPen(QColor(100, 200, 255), 2.0 * PRIMITIVE_SCALE)
        painter.setPen(circle_pen)

        # 绘制内侧圆（填充传入的颜色：红或蓝）
        painter.setBrush(QBrush(self._style.inner_color))
        painter.drawEllipse(inner_center, self._radius, self._radius)

        # 绘制外侧圆（绝对不填充：透明透底）
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(outer_center, self._radius, self._radius)


class NameLabelItem(QGraphicsItem):
    def __init__(
        self,
        text: str,
        center: QPointF,
        align: Qt.AlignmentFlag = Qt.AlignCenter,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self._text = text
        self._center = center
        self._align = align
        self._font = QFont("Arial", int(10 * PRIMITIVE_SCALE))
        metrics = QFontMetricsF(self._font)
        self._text_rect = metrics.boundingRect(self._text)
        self.setCacheMode(QGraphicsItem.ItemCoordinateCache)

    def boundingRect(self) -> QRectF:
        return QRectF(
            self._center.x() - self._text_rect.width() / 2,
            self._center.y() - self._text_rect.height() / 2,
            self._text_rect.width(),
            self._text_rect.height(),
        )

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setFont(self._font)
        painter.setPen(QPen(Qt.white))
        rect = QRectF(
            self._center.x() - self._text_rect.width() / 2,
            self._center.y() - self._text_rect.height() / 2,
            self._text_rect.width(),
            self._text_rect.height(),
        )
        painter.drawText(rect, self._align, self._text)


class SimulationRenderer:
    def __init__(self, scene: QGraphicsScene) -> None:
        self._scene = scene
        self._items: List[QGraphicsItem] = []

    def clear(self) -> None:
        for item in self._items:
            self._scene.removeItem(item)
        self._items.clear()

    def render_tracks(self, lines: Iterable[TrackLine]) -> None:
        for line in lines:
            item = TrackItem(QPointF(line.start[0], line.start[1]), QPointF(line.end[0], line.end[1]))
            item.setZValue(1)
            self._scene.addItem(item)
            self._items.append(item)

    def render_primitives(self, primitives: Iterable[Primitive]) -> None:
        for primitive in primitives:
            if primitive.category_name == "insulation" and primitive.keypoints:
                center = QPointF(primitive.keypoints[0][0], primitive.keypoints[0][1])
                item = InsulationItem(center)
                item.setZValue(2)
                self._scene.addItem(item)
                self._items.append(item)
            elif primitive.category_name == "switch":
                bbox = primitive.bbox
                center = QPointF((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                item = SwitchItem(center)
                item.setZValue(2)
                self._scene.addItem(item)
                self._items.append(item)
                if primitive.name:
                    label_pos = QPointF(center.x(), center.y() - 14 * PRIMITIVE_SCALE)
                    label = NameLabelItem(primitive.name, label_pos, Qt.AlignCenter)
                    label.setZValue(4)
                    self._scene.addItem(label)
                    self._items.append(label)
            elif primitive.category_name in ("xsignal", "dsignal") and primitive.keypoints:
                if len(primitive.keypoints) < 3:
                    continue
                pole_top = QPointF(primitive.keypoints[0][0], primitive.keypoints[0][1])
                pole_bottom = QPointF(primitive.keypoints[1][0], primitive.keypoints[1][1])
                indicator = QPointF(primitive.keypoints[2][0], primitive.keypoints[2][1])

                # 配置信号机样式 (只需要传内侧圆的颜色)
                if primitive.category_name == "xsignal":
                    style = SignalStyle(QColor(Qt.red))
                else:
                    style = SignalStyle(QColor(0, 110, 255))

                item = SignalItem(pole_top, pole_bottom, indicator, style)
                item.setZValue(3)
                self._scene.addItem(item)
                self._items.append(item)

                if primitive.name:
                    center = QPointF((pole_top.x() + pole_bottom.x()) / 2, (pole_top.y() + pole_bottom.y()) / 2)
                    direction = 1.0 if indicator.x() > center.x() else -1.0
                    offset = 20 * PRIMITIVE_SCALE if direction < 0 else -20 * PRIMITIVE_SCALE
                    label_pos = QPointF(center.x() + offset, center.y())
                    label = NameLabelItem(primitive.name, label_pos, Qt.AlignCenter)
                    label.setZValue(4)
                    self._scene.addItem(label)
                    self._items.append(label)
