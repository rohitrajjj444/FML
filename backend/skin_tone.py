
import numpy as np
import cv2
from PIL import Image


def sample_skin_colour(face: Image.Image | None) -> str | None:
    if face is None:
        return None

    pixels = np.asarray(face.convert("RGB"))
    height, width = pixels.shape[:2]

    cheeks = pixels[int(0.55 * height):int(0.68 * height)]

    left = cheeks[:, int(0.12 * width):int(0.30 * width)]
    right = cheeks[:, int(0.70 * width):int(0.88 * width)]

    if left.size == 0 or right.size == 0:
        return None

    samples = np.concatenate([
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ])

    red, green, blue = np.median(samples, axis=0).astype(int)

    return f"#{red:02x}{green:02x}{blue:02x}"


def predict_skin_tone(face: Image.Image | None) -> str | None:
    hex_colour = sample_skin_colour(face)

    if hex_colour is None:
        return None

    rgb = [
        int(hex_colour[i:i+2], 16)
        for i in (1, 3, 5)
    ]

    pixel = np.array([[rgb]], dtype=np.uint8)
    lab = cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)

    lightness = int(lab[0, 0, 0])

    if lightness < 115:
        return "dark"
    elif lightness < 175:
        return "medium"
    else:
        return "light"
