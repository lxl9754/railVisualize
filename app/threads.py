from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import QThread, Signal


class YoloWorker(QThread):
    progress_changed = Signal(int)
    result_ready = Signal(list)
    error = Signal(str)

    def __init__(
        self,
        model_path: Optional[str],
        image_path: Optional[str],
        use_sahi: bool = False,
        slice_size: Tuple[int, int] = (900, 1700),
        overlap_ratio: Tuple[float, float] = (0.2, 0.2),
        xsignal_min_conf: float = 0.7,
        conf_threshold: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._model_path = model_path
        self._image_path = image_path
        self._use_sahi = use_sahi
        self._slice_width, self._slice_height = slice_size
        self._overlap_w, self._overlap_h = overlap_ratio
        self._xsignal_min_conf = xsignal_min_conf
        self._conf_threshold = conf_threshold
        self._device = device

    def run(self) -> None:
        try:
            if not self._model_path or not self._image_path:
                raise RuntimeError("未加载模型或图片")
            self.progress_changed.emit(5)
            if self._use_sahi:
                output = self._run_sahi()
            else:
                output = self._run_ultralytics()
            self._save_yolo_results(output)
            self.progress_changed.emit(100)
            self.result_ready.emit(output)
        except Exception as exc:  # pragma: no cover - UI feedback
            self.error.emit(str(exc))

    def _save_yolo_results(self, output: list) -> None:
        import json
        from pathlib import Path

        results_dir = self._build_results_dir()
        results_dir.mkdir(parents=True, exist_ok=True)
        json_path = results_dir / "yolo_primitives.json"
        json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_results_dir(self) -> "Path":
        from pathlib import Path

        image_path = Path(self._image_path) if self._image_path else Path("image")
        base = image_path.stem
        return Path(__file__).resolve().parents[1] / "results" / base

    def _run_ultralytics(self) -> list:
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

        return output

    def _run_sahi(self) -> list:
        from sahi_pose import AutoDetectionModel
        from sahi_pose.predict import get_sliced_prediction
        import torch

        device = self._device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8pose",
            model_path=self._model_path,
            confidence_threshold=self._conf_threshold,
            device=device,
        )
        self.progress_changed.emit(30)
        result = get_sliced_prediction(
            self._image_path,
            detection_model,
            slice_height=self._slice_height,
            slice_width=self._slice_width,
            overlap_height_ratio=self._overlap_h,
            overlap_width_ratio=self._overlap_w,
            postprocess_type="NMS",
            postprocess_match_threshold=0.5,
        )

        predictions = []
        id_counter = 1
        for pred in result.object_prediction_list:
            if pred.category.name == "xsignal" and pred.score.value < self._xsignal_min_conf:
                continue

            keypoints = pred.keypoints or []
            predictions.append(
                {
                    "id": id_counter,
                    "bbox": [float(coord) for coord in pred.bbox.to_xyxy()],
                    "score": float(pred.score.value),
                    "category_id": int(pred.category.id),
                    "category_name": pred.category.name,
                    "keypoints": [[float(x), float(y), float(c)] for x, y, c in keypoints],
                }
            )
            id_counter += 1

        return predictions


class UnetWorker(QThread):
    progress_changed = Signal(int)
    result_ready = Signal(list)
    error = Signal(str)

    def __init__(
        self,
        model_path: Optional[str],
        image_path: Optional[str],
        tile_size: Tuple[int, int] = (800, 1600),
        overlap_ratio: Tuple[float, float] = (0.2, 0.2),
    ) -> None:
        super().__init__()
        self._model_path = model_path
        self._image_path = image_path
        self._tile_size = tile_size
        self._overlap_ratio = overlap_ratio

    def run(self) -> None:
        try:
            if not self._model_path or not self._image_path:
                raise RuntimeError("未加载模型或图片")
            self.progress_changed.emit(5)
            from pathlib import Path
            import cv2
            import numpy as np
            import torch
            import albumentations as A
            from portable.unet.predict.predict1 import (
                load_config_from_model_path,
                load_model_from_path,
                read_image,
            )
            from extract_lines.extract_lines import build_line_features, extract_lines_from_mask

            image = read_image(self._image_path)
            if image is None:
                raise RuntimeError("无法读取原图")

            config = load_config_from_model_path(self._model_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_model_from_path(self._model_path, config, device)

            tile_h, tile_w = self._tile_size
            overlap_h, overlap_w = self._overlap_ratio
            stride_h = self._compute_stride(tile_h, overlap_h)
            stride_w = self._compute_stride(tile_w, overlap_w)
            mask = self._predict_mask_sliding(
                model,
                image,
                config,
                tile_h,
                tile_w,
                stride_h,
                stride_w,
                device,
                A,
                cv2,
                np,
            )

            results_dir = self._build_results_dir()
            results_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(self._image_path).stem
            mask_path = results_dir / f"{stem}_unet_mask.png"
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
            self._save_unet_results(results_dir, features)
            self.progress_changed.emit(100)
            self.result_ready.emit(features)
        except Exception as exc:  # pragma: no cover - UI feedback
            self.error.emit(str(exc))

    def _save_unet_results(self, results_dir: "Path", features: list) -> None:
        import json

        json_path = results_dir / "unet_tracks.json"
        json_path.write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_results_dir(self) -> "Path":
        from pathlib import Path

        image_path = Path(self._image_path) if self._image_path else Path("image")
        base = image_path.stem
        return Path(__file__).resolve().parents[1] / "results" / base

    @staticmethod
    def _compute_stride(tile: int, overlap_ratio: float) -> int:
        ratio = max(0.0, min(float(overlap_ratio), 0.95))
        stride = int(round(tile * (1.0 - ratio)))
        return max(1, stride)

    def _predict_mask_sliding(
        self,
        model,
        image,
        config,
        tile_h,
        tile_w,
        stride_h,
        stride_w,
        device,
        A,
        cv2,
        np,
    ) -> "np.ndarray":
        import torch

        height, width = image.shape[:2]
        positions_y = self._build_positions(height, tile_h, stride_h)
        positions_x = self._build_positions(width, tile_w, stride_w)

        val_transform = A.Compose([
            A.Resize(height=config["input_h"], width=config["input_w"]),
            A.Normalize(),
        ])

        sum_mask = np.zeros((height, width), dtype=np.float32)
        count_mask = np.zeros((height, width), dtype=np.float32)

        total_tiles = len(positions_y) * len(positions_x)
        processed = 0

        with torch.no_grad():
            for y in positions_y:
                for x in positions_x:
                    tile = image[y : y + tile_h, x : x + tile_w]
                    augmented = val_transform(image=tile)
                    img = augmented["image"].astype("float32") / 255
                    img = img.transpose(2, 0, 1)
                    inputs = torch.from_numpy(img).unsqueeze(0).to(device)

                    if config.get("deep_supervision"):
                        output = model(inputs)[-1]
                    else:
                        output = model(inputs)
                    output = torch.sigmoid(output).cpu().numpy()[0, 0]

                    resized_output = cv2.resize(
                        output,
                        (tile.shape[1], tile.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )

                    sum_mask[y : y + tile.shape[0], x : x + tile.shape[1]] += resized_output
                    count_mask[y : y + tile.shape[0], x : x + tile.shape[1]] += 1.0

                    processed += 1
                    if total_tiles > 0:
                        progress = 5 + int(55 * processed / total_tiles)
                        self.progress_changed.emit(progress)

        count_mask[count_mask == 0] = 1.0
        merged = sum_mask / count_mask
        return (merged * 255).astype("uint8")

    @staticmethod
    def _build_positions(length: int, tile: int, stride: int) -> list:
        if length <= tile:
            return [0]
        positions = list(range(0, length - tile + 1, stride))
        last = length - tile
        if positions[-1] != last:
            positions.append(last)
        return positions
