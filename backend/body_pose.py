from math import dist
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"


def measure_body_pose(photo: Image.Image) -> float | None:
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(photo))
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
        result = detector.detect(image)

    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks[0]
    points = [landmarks[index] for index in (11, 12, 23, 24)]

    if any((point.visibility or 0) < 0.5 for point in points):
        return None
    if any(not (0 <= point.x <= 1 and 0 <= point.y <= 1) for point in points):
        return None

    left_shoulder, right_shoulder, left_hip, right_hip = [
        (point.x * photo.width, point.y * photo.height) for point in points
    ]
    hip_span = dist(left_hip, right_hip)
    if hip_span == 0:
        return None

    return round(dist(left_shoulder, right_shoulder) / hip_span, 2)