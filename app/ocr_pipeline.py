from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional
import argparse
import json
import sys

import cv2

OCR_ROOT = Path(__file__).resolve().parents[1] / "OCR"
if str(OCR_ROOT) not in sys.path:
    sys.path.insert(0, str(OCR_ROOT))

from ocr_infer import build_args  # type: ignore
from tools.infer import predict_system  # type: ignore

TARGET_CATEGORIES = {"xsignal", "dsignal", "switch"}


@dataclass
class OcrConfig:
    det_model_dir: Optional[str] = None
    rec_model_dir: Optional[str] = None
    det_limit_type: Optional[str] = None
    drop_score: Optional[float] = None
    min_text_score: float = 0.5
    crop_padding: int = 2


def _build_text_system(config: OcrConfig):
    args = build_args([])
    if config.det_model_dir:
        args.det_model_dir = config.det_model_dir
    if config.rec_model_dir:
        args.rec_model_dir = config.rec_model_dir
    if config.det_limit_type:
        args.det_limit_type = config.det_limit_type
    if config.drop_score is not None:
        args.drop_score = config.drop_score
    args.use_mp = False
    args.show_log = False
    return predict_system.TextSystem(args)


def _clamp_bbox(
    bbox: Iterable[float],
    width: int,
    height: int,
    padding: int,
) -> Optional[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(round(x1 - padding)))
    y1 = max(0, int(round(y1 - padding)))
    x2 = min(width, int(round(x2 + padding)))
    y2 = min(height, int(round(y2 + padding)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _pick_text(rec_res: Optional[List[List[object]]], min_text_score: float) -> Optional[str]:
    if not rec_res:
        return None
    best_text, best_score = max(rec_res, key=lambda item: float(item[1]))
    if float(best_score) < min_text_score:
        return None
    text = "".join(str(best_text).split())
    return text or None


def annotate_primitives_with_ocr(
    image_path: str | Path,
    primitives: List[dict],
    config: Optional[OcrConfig] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> List[dict]:
    config = config or OcrConfig()
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError("Failed to read image for OCR")

    text_system = _build_text_system(config)
    height, width = image.shape[:2]
    targets = [item for item in primitives if item.get("category_name") in TARGET_CATEGORIES]

    total = len(targets)
    if total == 0:
        if progress_cb:
            progress_cb(100)
        return primitives

    for index, item in enumerate(targets, start=1):
        bbox = item.get("bbox") or [0, 0, 0, 0]
        clamped = _clamp_bbox(bbox, width, height, config.crop_padding)
        if not clamped:
            continue
        x1, y1, x2, y2 = clamped
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        _, rec_res, _ = text_system(crop)
        name = _pick_text(rec_res, config.min_text_score)
        if name:
            item["name"] = name
        if progress_cb:
            progress = 10 + int(80 * index / total)
            progress_cb(min(progress, 95))

    if progress_cb:
        progress_cb(100)
    return primitives


def load_primitives_json(path: str | Path) -> List[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_primitives_json(path: str | Path, primitives: List[dict]) -> None:
    Path(path).write_text(json.dumps(primitives, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate primitives with OCR names.")
    parser.add_argument("--image", required=True, help="Path to source image")
    parser.add_argument("--primitives", required=True, help="Path to primitives JSON")
    parser.add_argument("--output", help="Output JSON path (default: overwrite primitives)")
    parser.add_argument("--det-model", help="PaddleOCR det model dir")
    parser.add_argument("--rec-model", help="PaddleOCR rec model dir")
    parser.add_argument("--det-limit-type", help="PaddleOCR det_limit_type")
    parser.add_argument("--min-score", type=float, default=0.5, help="Min OCR text score")
    parser.add_argument("--padding", type=int, default=2, help="BBox padding in pixels")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    primitives = load_primitives_json(args.primitives)
    config = OcrConfig(
        det_model_dir=args.det_model,
        rec_model_dir=args.rec_model,
        det_limit_type=args.det_limit_type,
        min_text_score=args.min_score,
        crop_padding=args.padding,
    )
    result = annotate_primitives_with_ocr(args.image, primitives, config)
    output_path = args.output or args.primitives
    save_primitives_json(output_path, result)


if __name__ == "__main__":
    main()
