from math import dist
from pathlib import Path
from tempfile import NamedTemporaryFile

import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageOps

model_path = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"
photo_path = input("Path to your full-body photo: ").strip().strip("'\"")

with Image.open(photo_path) as original:
    photo = ImageOps.exif_transpose(original).convert("RGB")

mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(photo))
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
)

with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
    result = detector.detect(mp_image)

if not result.pose_landmarks:
    raise SystemExit("No pose found. Try a clear photo of one person standing.")

landmarks = result.pose_landmarks[0]
points = [
    ("Left shoulder", 11),
    ("Right shoulder", 12),
    ("Left hip", 23),
    ("Right hip", 24),
]

def pixel_point(index):
    point = landmarks[index]
    return (point.x * photo.width, point.y * photo.height)


shoulder_span = dist(pixel_point(11), pixel_point(12))
hip_span = dist(pixel_point(23), pixel_point(24))

if any((landmarks[index].visibility or 0) < 0.5 for _, index in points):
    print("Ratio skipped: one or more landmarks have low visibility.")
elif hip_span <= 0:
    print("Ratio skipped: the hip points overlap.")
else:
    print(f"Shoulder-to-hip ratio: {shoulder_span / hip_span:.2f}")

draw = ImageDraw.Draw(photo)

for label, index in points:
    point = landmarks[index]
    visibility = point.visibility or 0.0
    print(f"{label}: visibility {visibility:.0%}")

    x = round(point.x * (photo.width - 1))
    y = round(point.y * (photo.height - 1))
    colour = "lime" if visibility >= 0.5 else "orange"
    draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=colour, outline="black")

with NamedTemporaryFile(suffix=".png", delete=False) as preview:
    photo.save(preview, format="PNG")
    print(f"Preview saved to: {preview.name}")