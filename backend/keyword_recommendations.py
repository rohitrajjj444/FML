import json
import re
from pathlib import Path

CATALOGUE_PATH = Path(__file__).with_name("catalogue.json")
OUTFITS = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))

GARMENTS = {"kurta", "shirt", "chinos", "blazer", "trousers", "jacket"}
COLOURS = {"black", "white", "ivory", "blue", "red", "green", "brown",
           "grey", "gray", "navy", "beige", "pink", "yellow", "purple"}
MEN_TERMS = {"men", "mens", "male", "man"}
OTHER_SHOPPER_TERMS = {
    "women", "womens", "woman", "ladies", "female", "girl", "girls",
    "boy", "boys", "kid", "kids", "child", "children",
}


def recommend_outfits(clothing_request: str) -> list[dict]:
    words = set(re.findall(r"\w+", clothing_request.casefold()))
    shopping_for_men = bool(words & MEN_TERMS) and not bool(words & OTHER_SHOPPER_TERMS)
    search_words = words - MEN_TERMS if shopping_for_men else words
    garments = words & GARMENTS
    colours = words & COLOURS
    matches = []

    for name, keywords, styling_tip, *details in OUTFITS:
        product = details[0] if details else {}
        url = product.get("product_url")
        if not isinstance(url, str) or not url.startswith("https://"):
            continue
        tags = set(keywords.split())
        if shopping_for_men:
            listing_text = f"{name} {product.get('product_name') or ''} {keywords}"
            listing_words = set(re.findall(r"\w+", listing_text.casefold()))
            if listing_words & OTHER_SHOPPER_TERMS or not listing_words & (MEN_TERMS | {"unisex"}):
                continue
        if garments and not garments.intersection(tags):
            continue
        if colours and not colours.intersection(tags):
            continue
        matching_words = sorted(search_words.intersection(tags))
        if matching_words:
            matches.append({
                "name": name,
                "styling_tip": styling_tip,
                "matched_keywords": matching_words,
                "product_url": url,
                "product_name": product.get("product_name"),
                "retailer": product.get("retailer"),
                "image_url": product.get("image_url"),
                "listed_price_inr": product.get("listed_price_inr"),
            })

    matches.sort(key=lambda outfit: len(outfit["matched_keywords"]), reverse=True)
    unique = {}
    for outfit in matches:
        url_key = outfit["product_url"].split("#", 1)[0].split("?", 1)[0].rstrip("/")
        unique.setdefault(url_key, outfit)
        if len(unique) == 10:
            break
    return list(unique.values())
