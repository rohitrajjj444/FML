import cv2
import numpy as np
from PIL import Image

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def crop_face(image: Image.Image) -> Image.Image | None:
    padded, _ = crop_face_regions(image)
    return padded


def crop_face_regions(
    image: Image.Image,
) -> tuple[Image.Image | None, Image.Image | None]:
    gray = np.asarray(image.convert("L"))
    faces = detector.detectMultiScale(gray, minSize=(40, 40))

    if len(faces) != 1:
        return None, None

    x, y, width, height = map(int, faces[0])
    padding = int(max(width, height) * 0.20)

    padded = image.crop((
        max(0, x - padding),
        max(0, y - padding),
        min(image.width, x + width + padding),
        min(image.height, y + height + padding),
    ))
    tight = image.crop((x, y, x + width, y + height))
    return padded, tight