from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QFont, QFontMetricsF
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
    def __init__(
        self,
        junction: QPointF,
        straight_angle: Optional[float],
        diverging_angle: Optional[float],
        length: float,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self._junction = junction
        self._half_length = length / 2
        self._straight_angle = straight_angle
        self._diverging_angle = diverging_angle
        self.setCacheMode(QGraphicsItem.ItemCoordinateCache)

    def boundingRect(self) -> QRectF:
        pad = 2 * PRIMITIVE_SCALE
        return QRectF(
            self._junction.x() - self._half_length - pad,
            self._junction.y() - self._half_length - pad,
            self._half_length * 2 + pad * 2,
            self._half_length * 2 + pad * 2,
        )

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing)

        if self._straight_angle is not None:
            pen_straight = QPen(QColor(80, 220, 80), 2, Qt.SolidLine)
            painter.setPen(pen_straight)
            dx = self._half_length * math.cos(self._straight_angle)
            dy = self._half_length * math.sin(self._straight_angle)
            painter.drawLine(self._junction, QPointF(self._junction.x() + dx, self._junction.y() + dy))

        if self._diverging_angle is not None:
            pen_diverging = QPen(QColor(80, 220, 80), 2, Qt.SolidLine)
            painter.setPen(pen_diverging)
            dx = self._half_length * math.cos(self._diverging_angle)
            dy = self._half_length * math.sin(self._diverging_angle)
            painter.drawLine(self._junction, QPointF(self._junction.x() + dx, self._junction.y() + dy))


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

    def render_primitives(self, primitives: Iterable[Primitive], lines: Optional[Iterable[TrackLine]] = None) -> None:
        primitive_list = list(primitives)
        line_list = list(lines) if lines else []

        switches = [p for p in primitive_list if p.category_name == "switch"]
        avg_switch_width = 16 * PRIMITIVE_SCALE
        if switches:
            avg_switch_width = sum(abs(p.bbox[2] - p.bbox[0]) for p in switches) / len(switches)

        def _project_point(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float]:
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                return x1, y1, 0.0
            t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
            t = max(0.0, min(1.0, t))
            return x1 + t * dx, y1 + t * dy, t

        def _distance_point_to_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
            proj_x, proj_y, _ = _project_point(px, py, x1, y1, x2, y2)
            return math.hypot(px - proj_x, py - proj_y)

        def _angle_diff(a: float, b: float) -> float:
            diff = abs(a - b)
            if diff > math.pi:
                diff = 2 * math.pi - diff
            return diff

        def _line_intersection(
            a1: tuple[float, float],
            a2: tuple[float, float],
            b1: tuple[float, float],
            b2: tuple[float, float],
        ) -> Optional[tuple[float, float]]:
            x1, y1 = a1
            x2, y2 = a2
            x3, y3 = b1
            x4, y4 = b2
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                return None
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            return px, py

        def _angle_to_point(origin: QPointF, point: tuple[float, float]) -> float:
            return math.atan2(point[1] - origin.y(), point[0] - origin.x())

        def _pick_far_endpoint(origin: QPointF, line: TrackLine) -> tuple[float, float]:
            d_start = math.hypot(line.start[0] - origin.x(), line.start[1] - origin.y())
            d_end = math.hypot(line.end[0] - origin.x(), line.end[1] - origin.y())
            return line.start if d_start >= d_end else line.end

        for primitive in primitive_list:
            if primitive.category_name == "insulation" and primitive.keypoints:
                center = QPointF(primitive.keypoints[0][0], primitive.keypoints[0][1])
                item = InsulationItem(center)
                item.setZValue(2)
                self._scene.addItem(item)
                self._items.append(item)
            elif primitive.category_name == "switch":
                bbox = primitive.bbox
                center = QPointF((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

                straight_angle: Optional[float] = None
                diverging_angle: Optional[float] = None
                junction = center

                if line_list:
                    candidates: List[dict] = []
                    width = abs(bbox[2] - bbox[0])
                    height = abs(bbox[3] - bbox[1])
                    search_radius = max(width, height, avg_switch_width) * 1.5

                    for line in line_list:
                        dist = _distance_point_to_segment(
                            center.x(),
                            center.y(),
                            line.start[0],
                            line.start[1],
                            line.end[0],
                            line.end[1],
                        )
                        if dist > search_radius:
                            continue

                        proj_x, proj_y, _ = _project_point(
                            center.x(),
                            center.y(),
                            line.start[0],
                            line.start[1],
                            line.end[0],
                            line.end[1],
                        )
                        d_start = math.hypot(line.start[0] - proj_x, line.start[1] - proj_y)
                        d_end = math.hypot(line.end[0] - proj_x, line.end[1] - proj_y)
                        if d_start >= d_end:
                            angle = math.atan2(line.start[1] - proj_y, line.start[0] - proj_x)
                        else:
                            angle = math.atan2(line.end[1] - proj_y, line.end[0] - proj_x)
                        line_len = math.hypot(line.end[0] - line.start[0], line.end[1] - line.start[1])
                        candidates.append({"angle": angle, "line_len": line_len, "line": line, "proj": (proj_x, proj_y)})

                    if len(candidates) >= 2:
                        # 【修复点 1】：不单纯依靠整体长度，而是看谁“贯穿”了交点
                        # 贯穿交点的线，必然是主线（定位）。计算每条线较短的一端到交点的距离
                        for c in candidates:
                            d1 = math.hypot(c["line"].start[0] - junction.x(), c["line"].start[1] - junction.y())
                            d2 = math.hypot(c["line"].end[0] - junction.x(), c["line"].end[1] - junction.y())
                            c["min_dist_to_junc"] = min(d1, d2)

                        # 优先选“贯穿”交点的线（min_dist_to_junc 大的），其次再看长度
                        candidates.sort(key=lambda c: (c["min_dist_to_junc"], c["line_len"]), reverse=True)
                        straight_candidate = candidates[0]

                        # 【修复点 2】：废弃 max 错误逻辑，改用“最小锐角原理”找侧线
                        # 获取主线向两端延伸的真实角度
                        straight_start_angle = _angle_to_point(junction, straight_candidate["line"].start)
                        straight_end_angle = _angle_to_point(junction, straight_candidate["line"].end)

                        best_diverging = None
                        min_angle_diff = float('inf')

                        for c in candidates[1:]:
                            # 获取候选侧线远离交点方向的角度
                            target = _pick_far_endpoint(junction, c["line"])
                            c_angle = _angle_to_point(junction, target)

                            # 计算这条候选线，与主线两个方向的夹角差
                            diff_s = _angle_diff(c_angle, straight_start_angle)
                            diff_e = _angle_diff(c_angle, straight_end_angle)

                            # 真实的侧线，必然和主线的某一端成极小的锐角（同向分岔）
                            local_min_diff = min(diff_s, diff_e)
                            if local_min_diff < min_angle_diff:
                                min_angle_diff = local_min_diff
                                best_diverging = c

                        diverging_candidate = best_diverging

                        # --- 下面的求交点和 _pick_far_endpoint 等核心代码保持你原来的不变 ---
                        intersect = _line_intersection(
                            straight_candidate["line"].start,
                            straight_candidate["line"].end,
                            diverging_candidate["line"].start,
                            diverging_candidate["line"].end,
                        )
                        if intersect:
                            ix, iy = intersect
                            if math.hypot(ix - center.x(), iy - center.y()) <= search_radius * 2:
                                junction = QPointF(ix, iy)
                        else:
                            junction = QPointF(*straight_candidate["proj"])

                        diverging_target = _pick_far_endpoint(junction, diverging_candidate["line"])
                        diverging_angle = _angle_to_point(junction, diverging_target)

                        straight_start = straight_candidate["line"].start
                        straight_end = straight_candidate["line"].end
                        angle_start = _angle_to_point(junction, straight_start)
                        angle_end = _angle_to_point(junction, straight_end)
                        if diverging_angle is None:
                            straight_angle = angle_end
                        else:
                            diff_start = _angle_diff(angle_start, diverging_angle)
                            diff_end = _angle_diff(angle_end, diverging_angle)
                            straight_angle = angle_start if diff_start <= diff_end else angle_end
                    elif candidates:
                        straight_candidate = candidates[0]
                        junction = QPointF(*straight_candidate["proj"])
                        straight_target = _pick_far_endpoint(junction, straight_candidate["line"])
                        straight_angle = _angle_to_point(junction, straight_target)

                if straight_angle is None:
                    straight_angle = 0.0
                if diverging_angle is None:
                    diverging_angle = math.pi / 4

                item = SwitchItem(junction, straight_angle, diverging_angle, avg_switch_width)
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
