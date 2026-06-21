
import os, sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics not installed.  Run: pip install ultralytics onnx onnxsim")

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
OUT = MODELS_DIR / "yolov8m-pose.onnx"

if OUT.exists():
    print(f"[SKIP] {OUT} already exists. Delete it to re-export.")
    sys.exit(0)

print("[INFO] Downloading yolov8m-pose.pt and exporting to ONNX…")
model = YOLO("yolov8m-pose.pt")          # auto-downloads ~52 MB
model.export(
    format   = "onnx",
    opset    = 12,
    simplify = True,
    imgsz    = 640,
    dynamic  = False,                     # static shape = faster + simpler
)


default_out = Path("yolov8m-pose.onnx")
if default_out.exists():
    default_out.rename(OUT)
    print(f"[OK] Moved to {OUT}")
elif (Path(".") / "yolov8m-pose" / "weights" / "best.onnx").exists():
    src = Path("yolov8m-pose") / "weights" / "best.onnx"
    src.rename(OUT)
    print(f"[OK] Moved to {OUT}")
else:
    # Search common locations
    found = list(Path(".").rglob("yolov8m-pose.onnx"))
    if found:
        found[0].rename(OUT)
        print(f"[OK] Moved to {OUT}")
    else:
        sys.exit("[ERROR] ONNX not found after export. Check ultralytics version.")

print(f"\n[DONE] Model ready: {OUT}  ({OUT.stat().st_size // 1024 // 1024} MB)")
print("Next step:  python tools/1_roi_picker.py")
