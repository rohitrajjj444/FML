from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from face_shape import predict_face_shape
from starlette.concurrency import run_in_threadpool

from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from face_crop import crop_face_regions
from body_pose import measure_body_pose
from skin_tone import sample_skin_colour
from recommendations import recommend_outfits
from feedback import OutfitFeedback, save_feedback

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_PHOTO_SIZE = 12 * 1024 * 1024  # 12 MB


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/feedback")
def submit_feedback(feedback: OutfitFeedback):
    save_feedback(feedback)
    return {"status": "saved"}

@app.get("/outfits")
def search_outfits(q: str = Query(min_length=1, max_length=500)):
    return {"recommendations": recommend_outfits(q)}

@app.post("/analyze")
async def analyze(
    face_photo: UploadFile = File(...),
    body_photo: UploadFile = File(...),
    clothing_request: str = Form(..., max_length=500),
):
    clothing_request = clothing_request.strip()
    if not clothing_request:
        raise HTTPException(400, "Describe what you want to wear.")

    photo_dimensions = {}
    face_crop = tight_face_crop = None
    body_image = None
    for label, photo in [("Face photo", face_photo), ("Body photo", body_photo)]:
        if photo.content_type not in ALLOWED_TYPES:
            raise HTTPException(400, f"{label} must be a JPEG, PNG, or WebP.")

        content = await photo.read(MAX_PHOTO_SIZE + 1)
        if not content or len(content) > MAX_PHOTO_SIZE:
            raise HTTPException(400, f"{label} must be non-empty and at most 12 MB.")

        try:
            with Image.open(
                BytesIO(content), formats=["JPEG", "PNG", "WEBP"]
            ) as image:
                width, height = image.size
                if width * height > 20_000_000:
                    raise HTTPException(400, f"{label} must be at most 20 megapixels.")
                image.load()
                if label == "Face photo":
                    face_crop, tight_face_crop = crop_face_regions(ImageOps.exif_transpose(image))
                else:
                    body_image = ImageOps.exif_transpose(image).convert("RGB")                
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(400, f"{label} could not be opened as an image.")

        photo_dimensions[label] = {"width": width, "height": height}

    if face_crop is None:
        raise HTTPException(
            400,
            "Could not detect exactly one face. Try a clear, front-facing photo of one person.",
        )
    
    try:
        face_shape = await run_in_threadpool(predict_face_shape, face_crop)
    except Exception:
        raise HTTPException(
            502,
            "Face-shape prediction failed. Check the Roboflow key and connection.",
        )
    body_ratio = await run_in_threadpool(measure_body_pose, body_image)
    return {
        "status": "received",
        "face_photo": face_photo.filename,
        "body_photo": body_photo.filename,
        "clothing_request": clothing_request,
        "photo_dimensions": photo_dimensions,
        "face_shape": face_shape,
        "shoulder_to_hip_landmark_ratio": body_ratio,
        "skin_colour_sample": sample_skin_colour(tight_face_crop),
        "recommendations": recommend_outfits(clothing_request),
        "face_crop_dimensions": {
            "width": face_crop.width,
            "height": face_crop.height,
        },
    }
