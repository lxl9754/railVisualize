from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import QThread, Signal


class YoloWorker(QThread):
    progress_changed = Signal(int)
    result_ready = Signal(list)
    error = Signal(str)

    def __init__(self, model_path: Optional[str], image_path: Optional[str]) -> None:
        super().__init__()
        self._model_path = model_path
        self._image_path = image_path

    def run(self) -> None:
        try:
            if not self._model_path or not self._image_path:
                raise RuntimeError("未加载模型或图片")
            self.progress_changed.emit(5)
            from ultralytics import YOLO

            model = YOLO(self._model_path)
            self.progress_changed.emit(20)
            results = model.predict(self._image_path, verbose=False)
            if not results:
                raise RuntimeError("YOLO 推理未返回结果")

            result = results[0]
            names = getattr(result, "names", None) or getattr(model, "names", None)
            boxes = getattr(result, "boxes", None)
            keypoints = getattr(result, "keypoints", None)
            kp_xy = keypoints.xy if keypoints is not None else None
            kp_conf = keypoints.conf if keypoints is not None else None

            output = []
            if boxes is not None:
                for idx, box in enumerate(boxes):
                    xyxy = box.xyxy[0].cpu().tolist()
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    cls_id = int(box.cls[0]) if box.cls is not None else -1
                    if isinstance(names, dict):
                        category_name = names.get(cls_id, str(cls_id))
                    elif isinstance(names, (list, tuple)) and cls_id >= 0 and cls_id < len(names):
                        category_name = names[cls_id]
                    else:
                        category_name = "unknown"

                    parsed_keypoints = None
                    if kp_xy is not None and idx < len(kp_xy):
                        parsed_keypoints = []
                        xy_item = kp_xy[idx].cpu().tolist()
                        conf_item = kp_conf[idx].cpu().tolist() if kp_conf is not None else None
                        for kp_index, point in enumerate(xy_item):
                            kp_score = conf_item[kp_index] if conf_item is not None else 1.0
                            parsed_keypoints.append([float(point[0]), float(point[1]), float(kp_score)])

                    output.append(
                        {
                            "id": idx + 1,
                            "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                            "score": conf,
                            "category_id": cls_id,
                            "category_name": category_name,
                            "keypoints": parsed_keypoints,
                        }
                    )

            self.progress_changed.emit(100)
            self.result_ready.emit(output)
        except Exception as exc:  # pragma: no cover - UI feedback
            self.error.emit(str(exc))


class UnetWorker(QThread):
    progress_changed = Signal(int)
    result_ready = Signal(list)
    error = Signal(str)

    def __init__(
        self,
        model_path: Optional[str],
        image_path: Optional[str],
        tile_size: Tuple[int, int] = (800, 1600),
        tile_stride: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self._model_path = model_path
        self._image_path = image_path
        self._tile_size = tile_size
        self._tile_stride = tile_stride

    def run(self) -> None:
        try:
            if not self._model_path or not self._image_path:
                raise RuntimeError("未加载模型或图片")
            self.progress_changed.emit(5)
            from pathlib import Path
            import cv2
            from portable.unet.predict.predict1 import predict_mask_for_image, read_image
            from extract_lines.extract_lines import build_line_features, extract_lines_from_mask

            image = read_image(self._image_path)
            if image is None:
                raise RuntimeError("无法读取原图")

            mask = predict_mask_for_image(self._model_path, self._image_path)
            mask_path = Path(self._image_path).with_name(
                f"{Path(self._image_path).stem}_unet_mask.png"
            )
            cv2.imwrite(str(mask_path), mask)

            self.progress_changed.emit(70)
            merged_lines = extract_lines_from_mask(mask)

            height, width = image.shape[:2]
            mask_h, mask_w = mask.shape[:2]
            if mask_h > 0 and mask_w > 0 and (mask_h != height or mask_w != width):
                scale_x = width / mask_w
                scale_y = height / mask_h
                merged_lines = [
                    [
                        line[0] * scale_x,
                        line[1] * scale_y,
                        line[2] * scale_x,
                        line[3] * scale_y,
                    ]
                    for line in merged_lines
                ]

            features = build_line_features(merged_lines)
            self.progress_changed.emit(100)
            self.result_ready.emit(features)
        except Exception as exc:  # pragma: no cover - UI feedback
            self.error.emit(str(exc))
