"""Link each detected asset to its product documents and its asset-register entry.

Documents come from products/<id>/product.json (matched by product name); register rows
from asset_register.csv (matched by nearest position).
Usage: python link_assets.py [--max-distance 0.3]
Writes assets/tags.json - one ready-to-show tag per asset.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ASSETS = Path("assets/assets.json")
REGISTER = Path("asset_register.csv")
PRODUCTS = Path("products")
OUT = Path("assets/tags.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-distance", type=float, default=0.3, help="metres between asset and register entry")
    args = ap.parse_args()

    products = {}
    for f in PRODUCTS.glob("*/product.json"):
        cfg = json.loads(f.read_text())
        products[cfg["name"]] = dict(cfg, id=f.parent.name)
    with open(REGISTER, newline="", encoding="utf-8") as f:
        register = list(csv.DictReader(f))
    assets = json.loads(ASSETS.read_text())["assets"]

    tags, used = [], set()
    for a in assets:
        product = products.get(a["product"], {})
        pos = np.array(a["position_xyz"])
        dist = [np.linalg.norm(pos - [float(r["x"]), float(r["y"]), float(r["z"])]) for r in register]
        i = int(np.argmin(dist)) if dist else -1
        row = register[i] if i >= 0 and dist[i] <= args.max_distance and i not in used else None
        if row:
            used.add(i)
        tags.append(dict(
            id=a["id"],
            title=row["tag_name"] if row else f"Unregistered {a['product']}",
            registered=row is not None,
            panel=row["panel"] if row else None,
            panel_function=row["panel_function"] if row else None,
            cabinet=row["cabinet"] if row else None,
            product=a["product"],
            manufacturer=product.get("manufacturer"),
            description=product.get("description"),
            documents=product.get("documents", []),
            maintenance=dict(last_inspection=row["last_inspection"], next_inspection=row["next_inspection"],
                             note=row["maintenance_note"]) if row else None,
            position_xyz=a["position_xyz"], normal_xyz=a["normal_xyz"],
            detection=dict(sightings=a["sightings"], scan_positions=len(a["sweeps"]),
                           mean_confidence=a["mean_confidence"], best_photo=a["best_photo"]),
            crop=a["crop"]))
        print(f"{a['id']} -> {tags[-1]['title']}" + (f" ({dist[i] * 100:.0f} cm from register position)" if row else ""))

    missing = [r["tag_name"] for k, r in enumerate(register) if k not in used]
    OUT.write_text(json.dumps(dict(tags=tags, register_entries_not_found=missing), indent=1))
    print(f"{len(tags)} tags -> {OUT}; {len(tags[0]['documents']) if tags else 0} documents each; "
          f"register entries with no detected asset: {missing or 'none'}")


if __name__ == "__main__":
    main()
