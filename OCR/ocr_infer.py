"""Run OCR inference using PaddleOCR predict_system with default paths."""

from __future__ import annotations

import os
import sys
import subprocess
from typing import Iterable, List, Optional

from tools.infer import predict_system, utility

DEFAULT_IMAGE_DIR = r"C:\Users\Administrator\Desktop\ocr_train_imgs"
DEFAULT_DET_MODEL_DIR = (
    r"D:\Codes\Python\PaddleOCR-main\inference_model\PP-OCRv5_server_det"
)
DEFAULT_REC_MODEL_DIR = (
    r"D:\Codes\Python\PaddleOCR-main\inference_model\PP-OCRv5_server_rec_infer"
)
DEFAULT_DET_LIMIT_TYPE = "min"


def build_args(cli_args: Optional[Iterable[str]] = None):
    """Build predict_system args with workspace defaults."""
    parser = utility.init_args()
    parser.set_defaults(
        image_dir=DEFAULT_IMAGE_DIR,
        det_model_dir=DEFAULT_DET_MODEL_DIR,
        rec_model_dir=DEFAULT_REC_MODEL_DIR,
        det_limit_type=DEFAULT_DET_LIMIT_TYPE,
    )
    return parser.parse_args(list(cli_args) if cli_args is not None else None)


def _validate_paths(args) -> None:
    missing = []
    if not os.path.exists(args.image_dir):
        missing.append(f"image_dir={args.image_dir}")
    if not os.path.exists(args.det_model_dir):
        missing.append(f"det_model_dir={args.det_model_dir}")
    if not os.path.exists(args.rec_model_dir):
        missing.append(f"rec_model_dir={args.rec_model_dir}")
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Missing required paths: {missing_text}")


def _run_multiprocess(raw_args: List[str], total_process_num: int) -> None:
    processes = []
    script_path = os.path.abspath(__file__)
    for process_id in range(total_process_num):
        cmd = [
            sys.executable,
            "-u",
            script_path,
            *raw_args,
            f"--process_id={process_id}",
            "--use_mp=False",
        ]
        processes.append(subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr))
    for process in processes:
        process.wait()


def main(cli_args: Optional[Iterable[str]] = None) -> None:
    raw_args = list(cli_args) if cli_args is not None else sys.argv[1:]
    args = build_args(raw_args)
    _validate_paths(args)
    if args.use_mp:
        _run_multiprocess(raw_args, args.total_process_num)
    else:
        predict_system.main(args)


if __name__ == "__main__":
    main()
