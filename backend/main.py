
import asyncio
import logging
from contextlib import asynccontextmanager
from io import BytesIO
from time import perf_counter

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from face_shape import predict_face_shape
from starlette.concurrency import run_in_threadpool

from PIL import Image, ImageOps, UnidentifiedImageError

from face_crop import crop_face_regions
from body_pose import measure_body_pose
from skin_tone import sample_skin_colour, predict_skin_tone
from keyword_recommendations import recommend_outfits as recommend_by_keyword
from ml_recommendations import GENDERS, load_ranking_weights, recommend_outfits
from feedback import OutfitFeedback, save_feedback

from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("uvicorn.error.drape")

@asynccontextmanager
async def lifespan(app: FastAPI):
    started = perf_counter()
    await run_in_threadpool(load_ranking_weights)
    logger.info("Loaded saved recommendation weights in %.2fs", perf_counter() - started)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_PHOTO_SIZE = 12 * 1024 * 1024  # 12 MB


async def run_analysis_step(name, function, *args):
    started = perf_counter()
    try:
        return await run_in_threadpool(function, *args)
    finally:
        logger.info("%s took %.2fs", name, perf_counter() - started)


def predict_face_or_fail(photo: Image.Image) -> dict:
    try:
        return predict_face_shape(photo)
    except Exception as error:
        raise HTTPException(
            502,
            "Face-shape prediction failed. Check the Roboflow key and connection.",
        ) from error


def analyze_body(photo: Image.Image, gender: str) -> tuple[str | None, list[str]]:
    try:
        from body_shape import predict_body_shape_detailed

        result = predict_body_shape_detailed(photo, gender)
        if result.get("error"):
            logger.warning("BMNet inference unavailable: %s", result["error"])
            return None, ["model_unavailable"]
    except ImportError as error:
        logger.warning("BMNet dependencies unavailable: %s", error)
        return None, ["model_unavailable"]

    warnings = result.get("warnings", [])
    if not result.get("ok") or {"feet_cut_off", "head_cut_off"} & set(warnings):
        return None, warnings
    if result.get("borderline"):
        warnings = [*warnings, "borderline_shape"]
    return result["shape"], warnings


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/feedback")
def submit_feedback(feedback: OutfitFeedback):
    save_feedback(feedback)
    return {"status": "saved"}

@app.get("/outfits")
def search_outfits(q: str = Query(min_length=1, max_length=500)):
    return {"recommendations": recommend_by_keyword(q)}

@app.post("/analyze")
async def analyze(
    face_photo: UploadFile = File(...),
    body_photo: UploadFile = File(...),
    clothing_request: str = Form("", max_length=500),
    design_level: str = Form("simple"),
    gender: str | None = Form(None),
):
    started = perf_counter()
    design_level = design_level.strip().lower()
    gender = gender.strip().lower() if gender is not None else None

    if design_level not in {"simple", "fashioned"}:
        raise HTTPException(400, "Invalid design level.")
    if gender is not None and gender not in GENDERS:
        raise HTTPException(400, "Choose Men's or Women's.")

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

    logger.info("Photo preparation took %.2fs", perf_counter() - started)
    body_shape, body_warnings = None, []
    body_ratio = None
    face_job = run_analysis_step("Roboflow face prediction", predict_face_or_fail, face_crop)
    if gender is not None:
        face_shape, (body_shape, body_warnings) = await asyncio.gather(
            face_job,
            run_analysis_step("Local body shape", analyze_body, body_image, gender),
        )
    else:
        face_shape, body_ratio = await asyncio.gather(
            face_job,
            run_analysis_step("Local body pose", measure_body_pose, body_image),
        )
    skin_tone = sample_skin_colour(tight_face_crop)
    skin_category = predict_skin_tone(tight_face_crop)
    if skin_category is None:
        raise HTTPException(400, "Could not detect skin tone.")

    user_features = [skin_category, face_shape["label"], body_shape, design_level, gender]
    recommendations = recommend_outfits(user_features)
    logger.info("Analysis completed in %.2fs", perf_counter() - started)

    return {
        "status": "received",
        "face_photo": face_photo.filename,
        "body_photo": body_photo.filename,
        "clothing_request": clothing_request,
        "photo_dimensions": photo_dimensions,
        "face_shape": face_shape,
        "shoulder_to_hip_landmark_ratio": body_ratio,
        "body_shape": body_shape,
        "body_warnings": body_warnings,
        "gender": gender,
        "skin_colour_sample": skin_tone,
        "skin_tone": skin_category,
        "design_level": design_level,
        "recommendations": recommendations,

        "face_crop_dimensions": {
            "width": face_crop.width,
            "height": face_crop.height,
        },
    }
