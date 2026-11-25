import csv
from pathlib import Path
from urllib.parse import urlsplit

folder = Path(__file__).with_name("catalogue_data")
retailer_folder = folder / "retailers"
retailer_folder.mkdir(exist_ok=True)
columns = [
    "name", "keywords", "styling_tip", "product_name",
    "retailer", "product_url", "image_url", "listed_price_inr",
]

retailers = {"flipkart", "pantaloons"}
retailers.update(path.name.removesuffix("_products.csv")
                 for path in retailer_folder.glob("*_products.csv"))
for retailer in sorted(retailers):
    master = retailer_folder / f"{retailer}_products.csv"
    batches = sorted((folder / "imports").glob(f"{retailer}_*.csv"))
    if not batches:
        continue
    products = {}

    for source in [master, *batches]:
        if not source.is_file():
            continue

        with source.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                url = (row.get("product_url") or "").strip()
                parts = urlsplit(url)
                if parts.scheme != "https" or not parts.hostname:
                    continue

                key = (parts.hostname, parts.path.rstrip("/"))
                product = products.setdefault(key, {})
                for column in columns:
                    value = (row.get(column) or "").strip()
                    if value:
                        product[column] = value

    if not products:
        continue

    temp = master.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(products.values())
    temp.replace(master)
    print(f"{master.name}: {len(products)} unique products")
