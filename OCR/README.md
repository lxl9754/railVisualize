# OCR inference (Python runner)

This workspace wraps the PaddleOCR `tools.infer.predict_system` CLI as a Python script.

## Requirements

- Python 3.8+
- PaddlePaddle + PaddleOCR dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run inference (Python)

Defaults match the original absolute paths in `predict.py`:

```bash
python ocr_infer.py
```

Override any PaddleOCR CLI argument, for example:

```bash
python ocr_infer.py --image_dir "C:\\Users\\Administrator\\Desktop\\ocr_train_imgs" --det_model_dir "D:\\Codes\\Python\\PaddleOCR-main\\inference_model\\PP-OCRv5_server_det" --rec_model_dir "D:\\Codes\\Python\\PaddleOCR-main\\inference_model\\PP-OCRv5_server_rec_infer" --det_limit_type min
```

## Smoke test

```bash
python tests\test_infer_smoke.py
```

If model or image directories are missing, the test will be skipped.
