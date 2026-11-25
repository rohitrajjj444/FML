"""Merge a retailer CSV into catalogue.json by outfit name."""
from hashlib import sha256
import argparse
import csv
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit

CATALOGUE = Path(__file__).with_name("catalogue.json")
REQUIRED = ("name", "keywords", "styling_tip", "product_name", "retailer", "product_url")


def import_products(source, catalogue=CATALOGUE):
    outfits = {row[0]: row for row in json.loads(catalogue.read_text(encoding="utf-8"))}
    imported = set()
    with source.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not set(REQUIRED) <= set(reader.fieldnames or ()):
            raise ValueError("CSV must include: " + ", ".join(REQUIRED))
        for line, row in enumerate(reader, 2):
            name, keywords, tip, product_name, retailer, url = (
                (row[key] or "").strip() for key in REQUIRED
            )
            if not all((name, keywords, tip, product_name, retailer, url)):
                raise ValueError(f"Row {line}: missing required value.")

            existing = outfits.get(name)
            if existing and (len(existing) < 4 or existing[3].get("product_url") != url):
                suffix = sha256(url.encode("utf-8")).hexdigest()[:10]
                name = f"{name} [{suffix}]"
            
            image = (row.get("image_url") or "").strip()
            for address in filter(None, (url, image)):
                parsed = urlsplit(address)
                if parsed.scheme != "https" or not parsed.hostname:
                    raise ValueError(f"Row {line}: use valid HTTPS URLs.")
            product = {"product_name": product_name, "retailer": retailer, "product_url": url}
            if image:
                product["image_url"] = image
            price = (row.get("listed_price_inr") or "").strip()
            if price:
                if not price.isascii() or not price.isdecimal() or int(price) < 1:
                    raise ValueError(f"Row {line}: price must be a positive whole number in INR.")
                product["listed_price_inr"] = int(price)
            outfits[name] = [name, keywords, tip, product]
            imported.add(name)

    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=catalogue.parent, delete=False) as file:
        pending = Path(file.name)
        try:
            json.dump(list(outfits.values()), file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.close()
            pending.replace(catalogue)
        finally:
            pending.unlink(missing_ok=True)
    return len(imported), len(outfits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    args = parser.parse_args()
    try:
        imported, total = import_products(args.csv_file)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Import failed: {error}\n")
    print(f"Imported {imported} product rows. Catalogue now has {total} outfits.")
