"""Collect public Product JSON-LD into the catalogue importer's CSV format."""

import argparse
import csv
import json
import re
import time
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

AGENT = "DrapeCatalogue/1.0"
MAX_BYTES = 5_000_000
COLUMNS = (
    "name", "keywords", "styling_tip", "product_name", "retailer",
    "product_url", "image_url", "listed_price_inr",
)


def https_url(url):
    parts = urlsplit(url)
    return parts.scheme == "https" and bool(parts.hostname)


@lru_cache
def robots(origin):
    rules = RobotFileParser(origin + "/robots.txt")
    try:
        with urlopen(Request(rules.url, headers={"User-Agent": AGENT}), timeout=12) as response:
            content = response.read(256_001)
        if len(content) > 256_000:
            return None
        rules.parse(content.decode("utf-8", errors="replace").splitlines())
    except OSError:
        return None
    return rules


def fetch(url, delay):
    parts = urlsplit(url)
    rules = robots(f"{parts.scheme}://{parts.netloc}")
    if rules is None or not rules.can_fetch(AGENT, url):
        raise ValueError("robots rules unavailable or URL excluded")
    request_rate = rules.request_rate(AGENT)
    rate_delay = request_rate.seconds / request_rate.requests if request_rate else 0
    time.sleep(max(delay, rules.crawl_delay(AGENT) or 0, rate_delay))
    with urlopen(Request(url, headers={"User-Agent": AGENT}), timeout=15) as response:
        final_url = response.geturl()
        final_parts = urlsplit(final_url)
        final_rules = robots(f"{final_parts.scheme}://{final_parts.netloc}")
        if final_rules is None or not final_rules.can_fetch(AGENT, final_url):
            raise ValueError("redirect destination excluded")
        body = response.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise ValueError("page exceeds size limit")
        return body.decode(response.headers.get_content_charset() or "utf-8", errors="replace")


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.blocks = []
        self.script = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "script" and attributes.get("type") == "application/ld+json":
            self.script = []
        if tag == "a" and attributes.get("href"):
            self.links.append(attributes["href"])

    def handle_data(self, data):
        if self.script is not None:
            self.script.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self.script is not None:
            self.blocks.append("".join(self.script))
            self.script = None


def products(node):
    if isinstance(node, list):
        for child in node:
            yield from products(child)
    elif isinstance(node, dict):
        types = node.get("@type", [])
        types = [types] if isinstance(types, str) else types
        if isinstance(types, list) and "Product" in types:
            yield node
        for key in ("@graph", "mainEntity"):
            yield from products(node.get(key))


def product_row(page, url, retailer):
    for block in page.blocks:
        try:
            data = json.loads(block)
        except ValueError:
            continue
        for product in products(data):
            name = product.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            keywords = set()
            for key in ("name", "category", "color", "material", "pattern"):
                value = product.get(key)
                values = value if isinstance(value, list) else [value]
                for part in values:
                    if isinstance(part, str):
                        keywords.update(re.findall(r"\w+", part.casefold()))
            image = product.get("image", "")
            if isinstance(image, list):
                image = next(iter(image), "")
            if isinstance(image, dict):
                image = image.get("contentUrl") or image.get("url")
            image = urljoin(url, image) if isinstance(image, str) and image else ""
            offers = product.get("offers", [])
            offers = offers if isinstance(offers, list) else [offers]
            price = ""
            for offer in offers:
                if isinstance(offer, dict) and offer.get("priceCurrency") == "INR":
                    raw = str(offer.get("price", ""))
                    if re.fullmatch(r"[1-9]\d*(?:\.0+)?", raw):
                        price = raw.split(".")[0]
                        break
            return {
                "name": f"{name.strip()} at {retailer}",
                "keywords": " ".join(sorted(keywords)),
                "styling_tip": "Check the retailer for size, fit, and availability.",
                "product_name": name.strip(),
                "retailer": retailer,
                "product_url": url,
                "image_url": image if https_url(image) else "",
                "listed_price_inr": price,
            }
    return None


def collect(args, writer):
    pattern = re.compile(args.discover) if args.discover else None
    queue = dict.fromkeys(
        line.strip() for line in args.urls.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    pending = list(queue)
    seen = set()
    names = set()
    stopped_hosts = set()
    saved = 0

    existing = set()
    if args.skip_existing:
        catalogue = Path(__file__).with_name("catalogue.json")
        for row in json.loads(catalogue.read_text(encoding="utf-8")):
            if len(row) > 3 and row[3].get("product_url"):
                parts = urlsplit(row[3]["product_url"])
                existing.add((parts.hostname, parts.path.rstrip("/")))

    for raw in pending:
        if saved >= args.limit or len(seen) >= args.max_pages:
            break
        if not https_url(raw):
            continue
        parts = urlsplit(raw)
        url = urlunsplit(parts._replace(fragment=""))
        if url in seen or parts.hostname in stopped_hosts:
            continue
        if (parts.hostname, parts.path.rstrip("/")) in existing:
            continue

        seen.add(url)
        page = Page()
        try:
            page.feed(fetch(url, args.delay))
        except HTTPError as error:
            print(f"Skipped {url}: HTTP {error.code}", flush=True)
            if error.code in {401, 403, 429}:
                stopped_hosts.add(parts.hostname)
            continue
        except (OSError, ValueError) as error:
            print(f"Skipped {url}: {error}", flush=True)
            continue
        row = product_row(page, url, args.retailer)
        if row:
            if row["name"] in names:
                row["name"] += f" ({parts.path.rstrip('/').rsplit('/', 1)[-1]})"
            if row["name"] in names:
                continue
            names.add(row["name"])
            writer.writerow(row)
            saved += 1
        elif pattern:
            for link in page.links:
                target = urljoin(url, link)
                linked = urlsplit(target)
                target = urlunsplit(linked._replace(query="", fragment=""))
                if linked.hostname == parts.hostname and target not in queue and pattern.search(linked.path):
                    queue[target] = None
                    pending.append(target)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls", type=Path, help="Text file with one public URL per line")
    parser.add_argument("output", type=Path, help="New CSV file")
    parser.add_argument("--retailer", required=True)
    parser.add_argument("--discover", help="Regex identifying product paths on supplied listing pages")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--max-pages", type=int, default=6000)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.limit <= 5000 and args.max_pages >= 1 and args.delay >= 1):
        parser.error("Use a limit of 1–5000, positive max pages, and a delay of at least 1 second.")
    with args.output.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        count = collect(args, writer)
    print(f"Saved {count} products to {args.output}.")
