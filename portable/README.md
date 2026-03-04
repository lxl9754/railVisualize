# Portable UNet Package

This folder is a copy-friendly UNet inference bundle. It contains the minimal project structure for running inference in an isolated environment.

## Structure

```
portable/
  unet/
    archs.py
    infer_demo/
      infer_unet.py
      smoke_test.py
    predict/
      predict1.py
    attition/
    model/
    models/
      rail_UNet_woDS_20250617_214524/
        config.yml
        model.pth
```

## Run

```powershell
python unet/infer_demo/infer_unet.py --name rail_UNet_woDS_20250617_214524 --input inputs/test3/images --output outputs/predict1
```

```powershell
python unet/infer_demo/smoke_test.py --name rail_UNet_woDS_20250617_214524 --image inputs/test3/images/example.jpg
```

```powershell
python unet/predict/predict1.py --name rail_UNet_woDS_20250617_214524 --input-dir inputs/test3/images --output-dir outputs/predict1
```

## Dependencies

Install from the original repo `requirements.txt` (or ensure `torch`, `albumentations`, `opencv-python`, `pyyaml`, `numpy` are available).

