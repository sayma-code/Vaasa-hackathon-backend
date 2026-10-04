"""Download freely licensed background photos from Wikimedia Commons into backgrounds/.

Outside material only (nothing from the scanned site). Writes backgrounds/sources.csv
with the page, author and licence of every photo.
Usage: python download_backgrounds.py [--per-category 30]
"""
import argparse
import csv
import re
import time
from pathlib import Path

import requests

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "veo-hackathon-synthetic-backgrounds/1.0 (educational hackathon project)"}
OUT = Path("backgrounds")
CATEGORIES = [
    "Switchgear", "Electrical switchboards", "Distribution boards", "Electrical cabinets",
    "Control panels", "Electrical rooms", "Electrical substations", "Control rooms",
    "Server rooms", "Factory interiors", "Industrial buildings interiors", "Motor control centers",
    "Circuit breakers", "Electrical enclosures", "Machine rooms", "Warehouses interiors",
]


def category_files(session, category, limit):
    params = {"action": "query", "format": "json", "generator": "categorymembers",
              "gcmtitle": f"Category:{category}", "gcmtype": "file", "gcmlimit": limit,
              "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1600}
    pages = session.get(API, params=params, timeout=30).json().get("query", {}).get("pages", {})
    for page in pages.values():
        info = page.get("imageinfo", [{}])[0]
        if info.get("mime") != "image/jpeg" or min(info.get("width", 0), info.get("height", 0)) < 900:
            continue
        meta = info.get("extmetadata", {})
        clean = lambda key: re.sub(r"<[^>]+>", "", meta.get(key, {}).get("value", "")).strip()
        yield {"title": page["title"], "url": info.get("thumburl") or info["url"], "page": info["descriptionurl"],
               "author": clean("Artist"), "licence": clean("LicenseShortName"), "category": category}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=30)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)
    rows, seen = [], set()
    for category in CATEGORIES:
        n = 0
        for item in category_files(session, category, 200):
            if n >= args.per_category or item["title"] in seen:
                continue
            seen.add(item["title"])
            name = re.sub(r"[^A-Za-z0-9]+", "_", item["title"][5:].rsplit(".", 1)[0])[:80] + ".jpg"
            r = session.get(item["url"], timeout=60)
            if r.status_code != 200 or r.content[:2] != b"\xff\xd8":
                continue
            (OUT / name).write_bytes(r.content)
            rows.append({"file": name, **{k: item[k] for k in ("category", "page", "author", "licence")}})
            n += 1
            time.sleep(0.3)  # be polite to the server
        print(f"{category}: {n}", flush=True)
    with open(OUT / "sources.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "category", "page", "author", "licence"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} photos -> {OUT}/")


if __name__ == "__main__":
    main()
