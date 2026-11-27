import joblib
import pandas as pd
from pathlib import Path

# Keep these files in the same directory as new_recommendations.py
BASE_DIR = Path(__file__).resolve().parent
CLASSES_PATH = BASE_DIR / "classes_data.csv"
DATASET_PATH = BASE_DIR / "final_dataset.csv"
MODEL_PATH = BASE_DIR / "compatibility_class_model.joblib"
TOP_N = 5


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


def recommend_outfits(user_features: list) -> list[dict]:
    """
    user_features order: [skin_color, face_shape, body_shape, design_level]

    Returns the top 5 products for the matching class, ranked by:
    w1*skin_compatibility + w2*face_compatibility + w3*body_compatibility
    """
    if not isinstance(user_features, (list, tuple)) or len(user_features) != 4:
        raise ValueError(
            "Pass exactly 4 values: "
            "[skin_color, face_shape, body_shape, design_level]."
        )

    skin_color, face_shape, body_shape, design_level = [
        str(value).strip().casefold() for value in user_features
    ]

    # 1. Find the class number from classes_data.csv.
    classes_df = pd.read_csv(CLASSES_PATH)
    required_class_cols = {
        "class", "skin_color", "face_shape", "body_shape", "design_level"
    }
    missing = required_class_cols - set(classes_df.columns)
    if missing:
        raise ValueError(f"Missing columns in classes_data.csv: {sorted(missing)}")

    matched_class = classes_df[
        (classes_df["skin_color"].astype(str).str.strip().str.casefold() == skin_color)
        & (classes_df["face_shape"].astype(str).str.strip().str.casefold() == face_shape)
        & (classes_df["body_shape"].astype(str).str.strip().str.casefold() == body_shape)
        & (classes_df["design_level"].astype(str).str.strip().str.casefold() == design_level)
    ]
    if matched_class.empty:
        raise ValueError("No class found for the supplied user features.")

    class_no = matched_class.iloc[0]["class"]

    # 2. Load the pretrained model and obtain this class's weights.
    model = joblib.load(MODEL_PATH)
    w1, w2, w3 = _get_class_weights(model, class_no)

    # 3. Keep only rows belonging to this class.
    dataset = pd.read_csv(DATASET_PATH)
    required_data_cols = {
        "name", "styling_tip", "product_name", "retailer", "product_url",
        "image_url", "listed_price_inr", "skin_compatibility",
        "face_compatibility", "body_compatibility", "suitable_class"
    }
    missing = required_data_cols - set(dataset.columns)
    if missing:
        raise ValueError(f"Missing columns in final_dataset.csv: {sorted(missing)}")

    products = dataset[
        pd.to_numeric(dataset["suitable_class"], errors="coerce") == float(class_no)
    ].copy()
    if products.empty:
        return []

    # 4. Calculate final score using only skin, face and body compatibility.
    score_cols = [
        "skin_compatibility", "face_compatibility", "body_compatibility"
    ]
    for col in score_cols:
        products[col] = pd.to_numeric(products[col], errors="coerce")
    products = products.dropna(subset=score_cols)
    if products.empty:
        return []

    products["final_score"] = (
        w1 * products["skin_compatibility"]
        + w2 * products["face_compatibility"]
        + w3 * products["body_compatibility"]
    )

    # 5. Sort, remove duplicate product URLs, and select the top 5.
    products = products[
        products["product_url"].astype(str).str.startswith("https://")
    ].copy()
    products = products.sort_values("final_score", ascending=False)
    products["_url_key"] = (
        products["product_url"].astype(str)
        .str.split("#").str[0]
        .str.split("?").str[0]
        .str.rstrip("/")
    )
    products = products.drop_duplicates("_url_key", keep="first").head(TOP_N)

    # 6. Return the earlier product dictionary structure, plus final_score.
    results = []
    for _, product in products.iterrows():
        results.append({
            "name": product["name"],
            "styling_tip": product["styling_tip"],
            "matched_keywords": [],
            "product_url": product["product_url"],
            "product_name": product["product_name"],
            "retailer": product["retailer"],
            "image_url": product["image_url"],
            "listed_price_inr": product["listed_price_inr"],
            "final_score": round(float(product["final_score"]), 4),
        })
    return results


# Example:
# recommendations = recommend_outfits(["medium", "oval", "rectangle", "simple"])
# print(recommendations)
