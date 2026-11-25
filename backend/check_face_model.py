from getpass import getpass

from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from PIL import Image, ImageOps

from face_crop import crop_face


photo_path = input("Path to a clear face photo: ").strip().strip("'\"")

with Image.open(photo_path) as image:
    face = crop_face(ImageOps.exif_transpose(image))

if face is None:
    raise SystemExit("Could not detect exactly one face. Try a clearer photo.")

api_key = getpass("Roboflow API key (hidden): ").strip()

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key,
).configure(InferenceConfiguration(api_key_transport="header"))

result = client.infer(face.convert("RGB"), model_id="faceshape-atkte/4")
print(result)