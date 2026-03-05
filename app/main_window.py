from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGroupBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QRadioButton,
    QButtonGroup,
    QCheckBox,
)

from .data_loader import DataLoader
from .graphics import SceneRenderer
from .simulation_items import SimulationRenderer
from .models import Primitive, TrackLine
from .threads import UnetWorker, YoloWorker
from .view import ZoomableGraphicsView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("铁路站场图智能识别与多源信息融合系统")
        self.resize(1400, 900)

        self._yolo_model_path: Optional[str] = None
        self._unet_model_path: Optional[str] = None
        self._image_path: Optional[str] = None

        self._track_lines: List[TrackLine] = []
        self._primitives: List[Primitive] = []

        self._scene = QGraphicsScene()
        self._scene_renderer = SceneRenderer(self._scene)
        self._sim_renderer = SimulationRenderer(self._scene)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None

        self._yolo_worker: Optional[YoloWorker] = None
        self._unet_worker: Optional[UnetWorker] = None

        self._overlay_mode = True
        self._use_sahi = True

        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        splitter = QSplitter(Qt.Horizontal)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        control_layout.addWidget(self._create_mode_group())
        control_layout.addWidget(self._create_model_group())
        control_layout.addWidget(self._create_input_group())
        control_layout.addWidget(self._create_inference_group())
        control_layout.addStretch()

        self._view = ZoomableGraphicsView()
        self._view.setScene(self._scene)
        self._view.setBackgroundBrush(Qt.black)

        splitter.addWidget(control_panel)
        splitter.addWidget(self._view)
        splitter.setStretchFactor(1, 1)

        root_layout = QVBoxLayout(root)
        root_layout.addWidget(splitter)
        self.setCentralWidget(root)

    def _create_mode_group(self) -> QGroupBox:
        group = QGroupBox("视图模式")
        layout = QVBoxLayout(group)

        self._overlay_radio = QRadioButton("原图叠加分析模式 (Overlay)")
        self._overlay_radio.setChecked(True)
        self._sim_radio = QRadioButton("矢量仿真重绘模式 (Simulation)")

        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._overlay_radio)
        self._mode_group.addButton(self._sim_radio)
        self._overlay_radio.toggled.connect(self._on_mode_changed)

        layout.addWidget(self._overlay_radio)
        layout.addWidget(self._sim_radio)
        return group

    def _create_model_group(self) -> QGroupBox:
        group = QGroupBox("模型加载区")
        layout = QVBoxLayout(group)

        yolo_btn = QPushButton("加载 YOLO11 模型 (.pt)")
        yolo_btn.clicked.connect(self._load_yolo_model)
        layout.addWidget(yolo_btn)

        unet_btn = QPushButton("加载 UNet 模型 (.pth)")
        unet_btn.clicked.connect(self._load_unet_model)
        layout.addWidget(unet_btn)

        self._model_status = QLabel("模型状态: 未加载")
        layout.addWidget(self._model_status)
        return group

    def _create_input_group(self) -> QGroupBox:
        group = QGroupBox("数据输入区")
        layout = QVBoxLayout(group)

        image_btn = QPushButton("导入站场图原图 (JPG/PNG)")
        image_btn.clicked.connect(self._load_image)
        layout.addWidget(image_btn)

        primitive_btn = QPushButton("直接导入图元 JSON")
        primitive_btn.clicked.connect(self._load_primitive_json)
        layout.addWidget(primitive_btn)

        track_btn = QPushButton("直接导入轨道线 JSON")
        track_btn.clicked.connect(self._load_track_json)
        layout.addWidget(track_btn)

        return group

    def _create_inference_group(self) -> QGroupBox:
        group = QGroupBox("执行推理区")
        layout = QVBoxLayout(group)

        self._sahi_checkbox = QCheckBox("启用 SAHI 切片推理 (YOLO)")
        self._sahi_checkbox.setChecked(True)
        self._sahi_checkbox.toggled.connect(self._on_sahi_toggled)
        layout.addWidget(self._sahi_checkbox)

        yolo_btn = QPushButton("运行 YOLO 图元检测")
        yolo_btn.clicked.connect(self._run_yolo)
        layout.addWidget(yolo_btn)

        unet_btn = QPushButton("运行 UNet 轨道线提取")
        unet_btn.clicked.connect(self._run_unet)
        layout.addWidget(unet_btn)

        combo_btn = QPushButton("一键综合检测")
        combo_btn.clicked.connect(self._run_both)
        layout.addWidget(combo_btn)

        self._progress = QProgressBar()
        self._progress.setValue(0)
        layout.addWidget(self._progress)
        return group

    def _on_mode_changed(self) -> None:
        self._overlay_mode = self._overlay_radio.isChecked()
        self._render_scene(full_refresh=True)

    def _on_sahi_toggled(self, checked: bool) -> None:
        self._use_sahi = checked

    def _load_yolo_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 YOLO11 模型", "", "Model Files (*.pt)")
        if path:
            self._yolo_model_path = path
            self._update_model_status()

    def _load_unet_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 UNet 模型", "", "Model Files (*.pth)")
        if path:
            self._unet_model_path = path
            self._update_model_status()

    def _update_model_status(self) -> None:
        yolo_status = "已加载" if self._yolo_model_path else "未加载"
        unet_status = "已加载" if self._unet_model_path else "未加载"
        self._model_status.setText(f"模型状态: YOLO={yolo_status} | UNet={unet_status}")

    def _load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择站场图", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        self._image_path = path
        image = QImage(path)
        if image.isNull():
            QMessageBox.warning(self, "错误", "无法加载图像")
            return
        pixmap = QPixmap.fromImage(image)
        if self._pixmap_item:
            self._scene.removeItem(self._pixmap_item)
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._pixmap_item.setZValue(0)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(pixmap.rect())
        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._render_scene(full_refresh=True)

    def _load_primitive_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择图元 JSON", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            self._primitives = DataLoader.load_primitives(path)
            self._refresh_scene(primitives_only=True)
        except Exception as exc:
            QMessageBox.warning(self, "错误", f"加载图元失败: {exc}")

    def _load_track_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择轨道线 JSON", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            self._track_lines = DataLoader.load_track_lines(path)
            self._refresh_scene(tracks_only=True)
        except Exception as exc:
            QMessageBox.warning(self, "错误", f"加载轨道线失败: {exc}")

    def _run_yolo(self) -> None:
        if self._yolo_worker and self._yolo_worker.isRunning():
            QMessageBox.information(self, "提示", "YOLO 推理正在进行")
            return
        self._progress.setValue(0)
        self._yolo_worker = YoloWorker(self._yolo_model_path, self._image_path, use_sahi=self._use_sahi)
        self._yolo_worker.progress_changed.connect(self._progress.setValue)
        self._yolo_worker.result_ready.connect(self._handle_yolo_result)
        self._yolo_worker.error.connect(lambda msg: QMessageBox.warning(self, "错误", msg))
        self._yolo_worker.start()

    def _run_unet(self) -> None:
        if self._unet_worker and self._unet_worker.isRunning():
            QMessageBox.information(self, "提示", "UNet 推理正在进行")
            return
        self._progress.setValue(0)
        self._unet_worker = UnetWorker(self._unet_model_path, self._image_path)
        self._unet_worker.progress_changed.connect(self._progress.setValue)
        self._unet_worker.result_ready.connect(self._handle_unet_result)
        self._unet_worker.error.connect(lambda msg: QMessageBox.warning(self, "错误", msg))
        self._unet_worker.start()

    def _run_both(self) -> None:
        self._run_yolo()
        self._run_unet()

    def _handle_yolo_result(self, raw_result: list) -> None:
        try:
            self._primitives = DataLoader.load_primitives_from_raw(raw_result)
        except Exception:
            self._primitives = []
        self._render_scene(primitives_only=True)

    def _handle_unet_result(self, raw_result: list) -> None:
        try:
            self._track_lines = DataLoader.load_tracks_from_raw(raw_result)
        except Exception:
            self._track_lines = []
        self._render_scene(tracks_only=True)

    def _render_scene(self, primitives_only: bool = False, tracks_only: bool = False, full_refresh: bool = False) -> None:
        if self._pixmap_item:
            self._pixmap_item.setVisible(self._overlay_mode)
        if self._overlay_mode:
            if full_refresh:
                self._scene_renderer.clear()
            if full_refresh or tracks_only or not (primitives_only or tracks_only):
                self._scene_renderer.clear_tracks()
                if self._track_lines:
                    self._scene_renderer.render_tracks(self._track_lines)
            if full_refresh or primitives_only or not (primitives_only or tracks_only):
                self._scene_renderer.clear_primitives()
                if self._primitives:
                    self._scene_renderer.render_primitives(self._primitives)
            if full_refresh:
                self._sim_renderer.clear()
        else:
            if full_refresh:
                self._sim_renderer.clear()
            if full_refresh or tracks_only or not (primitives_only or tracks_only):
                self._sim_renderer.clear()
                if self._track_lines:
                    self._sim_renderer.render_tracks(self._track_lines)
                if self._primitives:
                    self._sim_renderer.render_primitives(self._primitives)
            if full_refresh:
                self._scene_renderer.clear()

    def _refresh_scene(self, primitives_only: bool = False, tracks_only: bool = False) -> None:
        self._render_scene(primitives_only=primitives_only, tracks_only=tracks_only)
