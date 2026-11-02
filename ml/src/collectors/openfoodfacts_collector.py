"""Open Food Facts Image & Metadata Importer.

Imports packaged foods, beverages, and brand-free reference products from Open Food Facts
into Nutrition AI image structure with nutritional profiling.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger("nutrition_ai.openfoodfacts")

OFF_SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"
USER_AGENT = "NutrifyAI/1.0 (akshkanani01@gmail.com)"


def fetch_openfoodfacts_products(
    category: str,
    page_size: int = 20,
    max_pages: int = 1,
    country: str = "en",
) -> list[dict[str, Any]]:
    """Query Open Food Facts API for products in a specific food category with rate limiting."""
    headers = {"User-Agent": USER_AGENT}
    products = []

    for page in range(1, max_pages + 1):
        params = {
            "action": "process",
            "tagtype_0": "categories",
            "tag_contains_0": "contains",
            "tag_0": category,
            "page_size": page_size,
            "page": page,
            "json": 1,
            "fields": "code,product_name,image_front_url,nutriments,categories_tags",
        }
        url = f"{OFF_SEARCH_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers=headers)

        try:
            time.sleep(1.0)  # Strict 1 request/second rate limit
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                page_products = data.get("products", [])
                products.extend(page_products)
                logger.info(f"Fetched {len(page_products)} products for category '{category}' (page {page})")
        except Exception as e:
            logger.error(f"Error querying Open Food Facts: {e}")
            break

    return products


def import_openfoodfacts_category(
    category_query: str,
    target_label: str,
    dest_images_dir: str | Path = "ml/data/images",
    max_images: int = 30,
) -> int:
    """Download product images for a category and save into label folder."""
    dest_dir = Path(dest_images_dir) / target_label.replace(" ", "_")
    dest_dir.mkdir(parents=True, exist_ok=True)

    products = fetch_openfoodfacts_products(category_query, page_size=max_images)
    downloaded = 0

    headers = {"User-Agent": USER_AGENT}

    for prod in products:
        if downloaded >= max_images:
            break

        img_url = prod.get("image_front_url")
        code = prod.get("code")
        if not img_url or not code:
            continue

        target_file = dest_dir / f"off_{code}.jpg"
        if target_file.exists():
            downloaded += 1
            continue

        try:
            time.sleep(1.0)  # 1 request/second
            req = urllib.request.Request(img_url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                with open(target_file, "wb") as f:
                    f.write(resp.read())
            downloaded += 1
            logger.info(f"Downloaded {target_file.name} for '{target_label}'")
        except Exception as e:
            logger.warning(f"Failed to download image {img_url}: {e}")

    return downloaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Food Images from Open Food Facts")
    parser.add_argument("--category", type=str, required=True, help="Open Food Facts category tag (e.g. 'pizzas')")
    parser.add_argument("--label", type=str, required=True, help="Target normalized label (e.g. 'pizza')")
    parser.add_argument("--max-images", type=int, default=20, help="Max images to collect")
    parser.add_argument("--dest-dir", type=str, default="ml/data/images", help="Destination image folder")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    count = import_openfoodfacts_category(args.category, args.label, args.dest_dir, args.max_images)
    print(f"Imported {count} images for label '{args.label}'.")

