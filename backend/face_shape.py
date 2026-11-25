import os

from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from PIL import Image

MODEL_ID = "faceshape-atkte/4"


def predict_face_shape(face: Image.Image) -> dict:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is missing.")

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    ).configure(InferenceConfiguration(api_key_transport="header"))

    result = client.infer(face.convert("RGB"), model_id=MODEL_ID)
    return {
        "label": result["top"],
        "confidence": result["confidence"],
    }