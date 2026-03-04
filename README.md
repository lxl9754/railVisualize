# Railway Yard Diagram Intelligent Recognition & Fusion System (Demo)

A PySide6 desktop app prototype for loading models/data, running threaded inference, and visualizing railway yard diagrams.

## Features
- Load YOLO/UNet model paths (placeholder inference)
- Load image and optional JSON results
- Threaded "inference" to keep UI responsive
- Layer toggles for tracks and primitives
- Smooth zoom and pan on large images

## Data Formats
### Track Lines (GeoJSON-like)
```json
[
  {
    "type": "LineString",
    "properties": {"id": 1},
    "coordinates": [[92.0, 613.0], [1670.0, 613.0]]
  }
]
```

### Primitives (YOLO output)
```json
[
  {
    "id": 1,
    "bbox": [5368.0, 3196.0, 5479.0, 3245.0],
    "score": 0.9119,
    "category_id": 0,
    "category_name": "xsignal",
    "keypoints": [
      [5406.9, 3203.7, 0.999],
      [5401.7, 3237.5, 0.999]
    ]
  }
]
```

## Run
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## Notes
- This is a UI-ready scaffold; replace the placeholder inference in `app/threads.py` with your actual YOLO/UNet code.

