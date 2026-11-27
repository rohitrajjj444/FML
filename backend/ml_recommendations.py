import re
from functools import cache
from math import isfinite
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CLASSES_PATH = BASE_DIR / "data/classes_data.csv"
DATASET_PATH = BASE_DIR / "data/final_dataset.csv"
MODEL_PATH = BASE_DIR / "models/compatibility_class_model.joblib"
TOP_N = 5
GENDERS = frozenset({"male", "female"})
CATALOGUE_GENDERS = GENDERS | {"unisex", "children", "unknown"}
SCORE_COLUMNS = ("skin_compatibility", "face_compatibility", "body_compatibility")
PROFILE_COLUMNS = ["skin_color", "face_shape", "body_shape", "design_level"]


def _normalize(value):
    return re.sub(r"[\s-]+", "_", str(value).strip().casefold())


def _get_class_weights(model, class_no):
    """Return the three LogisticRegression coefficients for the requested class."""
    if not hasattr(model, "coef_"):
        raise ValueError(
            "The joblib model must expose coef_ (for example, a trained "
            "sklearn LogisticRegression model)."
        )

    coef = model.coef_
    if getattr(coef, "ndim", 0) == 1:
        weights = coef
    elif coef.shape[0] == 1:
        weights = coef[0]
    else:
        classes = getattr(model, "classes_", None)
        if classes is None:
            raise ValueError(
                "The model has multiple coefficient rows but no classes_ attribute."
            )
        indices = [
            i for i, label in enumerate(classes)
            if str(label).strip() == str(class_no).strip()
        ]
        if not indices:
            raise ValueError(f"Class {class_no} was not found in model.classes_.")
        weights = coef[indices[0]]

    if len(weights) != 3:
        raise ValueError(
            f"Expected 3 weights (skin, face, body), got {len(weights)}. "
            "Check how compatibility_class_model.joblib was trained."
        )
    return tuple(float(x) for x in weights)


@cache
def load_ranking_weights():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Extract compatibility_class_model.joblib from check fml.zip into models/.")
    saved = joblib.load(MODEL_PATH, mmap_mode="r")
    model = saved.get("model") if isinstance(saved, dict) else saved
    if hasattr(model, "coef_"):
        return {str(label): _get_class_weights(model, label) for label in model.classes_}

    weights = saved.get("weights") if isinstance(saved, dict) else None
    if not isinstance(weights, pd.DataFrame) or not {"feature", "normalized_weight"} <= set(weights.columns):
        raise ValueError("The saved bundle must contain a normalized weights table.")
    selected = weights.loc[weights["feature"].isin(SCORE_COLUMNS)].set_index("feature")
    if len(selected) != len(SCORE_COLUMNS) or set(selected.index) != set(SCORE_COLUMNS):
        raise ValueError("Expected one saved weight per ranking feature.")
    values = tuple(float(selected.loc[column, "normalized_weight"]) for column in SCORE_COLUMNS)
    if not all(0 <= value <= 1 for value in values) or sum(values[:2]) <= 0:
        raise ValueError("The saved global ranking weights are invalid.")
    return {None: values}


@cache
def _load_classes():
    classes = pd.read_csv(CLASSES_PATH, dtype={"class": str})
    missing = {"class", *PROFILE_COLUMNS} - set(classes.columns)
    if missing:
        raise ValueError(f"Missing columns in classes_data.csv: {sorted(missing)}")
    classes[PROFILE_COLUMNS] = classes[PROFILE_COLUMNS].apply(lambda column: column.map(_normalize))
    return classes


@cache
def _load_catalogue():
    dataset = pd.read_csv(DATASET_PATH, dtype={"suitable_class": str, "listed_price_inr": str})
    required = {
        "name", "styling_tip", "product_name", "retailer", "product_url",
        "image_url", "listed_price_inr", "gender", "suitable_class", *SCORE_COLUMNS,
    }
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Missing columns in final_dataset.csv: {sorted(missing)}")
    dataset["gender"] = dataset["gender"].str.strip().str.casefold()
    if not dataset["gender"].isin(CATALOGUE_GENDERS).all():
        raise ValueError("Invalid gender label in final_dataset.csv.")
    dataset = dataset[dataset["product_url"].str.startswith("https://", na=False)].copy()
    dataset[list(SCORE_COLUMNS)] = (
        dataset[list(SCORE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        .replace([float("inf"), float("-inf")], float("nan"))
    )
    return dataset


def recommend_outfits(user_features: list | tuple, *, gender: str | None = None) -> list[dict]:
    """Rank the top five for [skin_color, face_shape, body_shape, design_level, gender]."""
    if not isinstance(user_features, (list, tuple)) or len(user_features) not in (4, 5):
        raise ValueError("Pass [skin_color, face_shape, body_shape, design_level] with optional gender as the fifth value.")
    if len(user_features) == 5:
        if gender is not None:
            raise ValueError("Supply gender either in the list or as a keyword argument.")
        gender = user_features[4]
    if gender is not None and gender not in GENDERS:
        raise ValueError("Choose male or female.")

    skin_color, face_shape, body_shape, design_level = user_features[:4]
    skin_color, face_shape, design_level = map(_normalize, (skin_color, face_shape, design_level))
    body_shape = _normalize(body_shape) if body_shape else None

    # 1. Find the matching classes for the supplied profile.
    classes = _load_classes()
    matches = (
        (classes["skin_color"] == skin_color)
        & (classes["face_shape"] == face_shape)
        & (classes["design_level"] == design_level)
    )
    if body_shape is not None:
        matches &= classes["body_shape"] == body_shape
    class_ids = classes.loc[matches, "class"]
    if class_ids.empty:
        return []

    # 2. Keep eligible products and calculate their weighted final scores.
    dataset = _load_catalogue()
    audience = dataset["gender"].isin((gender, "unisex")) if gender else dataset["gender"] != "children"
    columns = SCORE_COLUMNS if body_shape is not None else SCORE_COLUMNS[:2]
    products = dataset[dataset["suitable_class"].isin(class_ids) & audience].dropna(subset=list(columns)).copy()
    if products.empty:
        return []

    saved_weights = load_ranking_weights()
    for class_no, group in products.groupby("suitable_class", sort=False):
        weights = saved_weights[None] if None in saved_weights else saved_weights[class_no]
        active_weights = weights[:len(columns)]
        scores = sum(weight * group[column] for weight, column in zip(active_weights, columns))
        products.loc[group.index, "final_score"] = scores
        weight_scale = sum(abs(weight) for weight in active_weights)
        products.loc[group.index, "compatibility_score"] = 100 * scores / weight_scale if weight_scale else 0

    # 3. Sort, remove duplicate product links, and return the top five.
    products = products.sort_values("final_score", ascending=False, kind="stable")
    products["_url_key"] = (
        products["product_url"].str.split("#").str[0].str.split("?").str[0].str.rstrip("/")
    )
    products = products.drop_duplicates("_url_key").head(TOP_N)
    results = []
    for _, product in products.iterrows():
        results.append({
            "name": product["name"],
            "styling_tip": product["styling_tip"],
            "matched_keywords": [],
            "product_url": product["product_url"],
            "product_name": product["product_name"],
            "retailer": product["retailer"],
            "image_url": product["image_url"] if pd.notna(product["image_url"]) else None,
            "listed_price_inr": product["listed_price_inr"] if pd.notna(product["listed_price_inr"]) else None,
            "design_level": design_level,
            "gender": product["gender"],
            "suitable_class": int(product["suitable_class"]),
            "final_score": round(float(product["final_score"]), 4),
            "compatibility_score": round(float(product["compatibility_score"]), 2),
        })
    return results
