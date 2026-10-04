"""Turn 2D detections into 3D assets.

Every detection is lifted to a 3D point with the point cloud, sightings of the same spot
from different photos are merged, and only spots seen from several scan positions are kept.
Usage: python locate_assets.py [--conf 0.5] [--min-sweeps 5] [--radius 0.15]
Writes assets/assets.json and assets/crops/<id>.jpg
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import scan3d as s

DETECTIONS = Path("eval/detections.json")
GT = Path("ground_truth/relays.json")
OUT = Path("assets")


def lift(file, box):
    """3D point and distance of the surface in the middle of a detection box."""
    x1, y1, x2, y2 = box
    mx, my = 0.25 * (x2 - x1), 0.25 * (y2 - y1)
    pts, d = s.points_in_box(file, (x1 + mx, y1 + my, x2 - mx, y2 - my))
    if len(pts) < 5:  # tiny box: widen the search a little
        pts, d = s.points_in_box(file, (x1 - 4, y1 - 4, x2 + 4, y2 + 4))
    if len(pts) == 0:
        return None
    near = d <= np.percentile(d, 25) + 0.05  # the front surface, not what is behind it
    return np.median(pts[near], axis=0)


def sightings(dets, conf):
    out = []
    for file in sorted(dets):  # sorted so each sweep's points are loaded once
        for x1, y1, x2, y2, c, cls in dets[file]:
            if c < conf:
                continue
            p = lift(file, (x1, y1, x2, y2))
            if p is not None:
                out.append(dict(photo=file, sweep=s.IMAGES[file]["sweep"], box=[x1, y1, x2, y2],
                                conf=c, cls=int(cls), xyz=p))
    return out


def cluster(items, radius):
    """Group sightings of the same class that fall within `radius` metres of each other."""
    groups = []
    for it in sorted(items, key=lambda i: -i["conf"]):
        best = None
        for g in groups:
            if g["cls"] != it["cls"]:
                continue
            dist = np.linalg.norm(np.median([m["xyz"] for m in g["members"]], axis=0) - it["xyz"])
            if dist <= radius and (best is None or dist < best[0]):
                best = (dist, g)
        if best:
            best[1]["members"].append(it)
        else:
            groups.append(dict(cls=it["cls"], members=[it]))
    return groups


def surface_normal(file, box, toward):
    """Unit normal of the surface inside a box, pointing toward the camera."""
    pts, d = s.points_in_box(file, box)
    if len(pts) < 20:
        return None
    pts = pts[d <= np.percentile(d, 60)]
    n = np.linalg.svd(pts - pts.mean(axis=0))[2][-1]
    return n if n @ (toward - pts.mean(axis=0)) > 0 else -n


def best_view(members):
    """The sighting that shows the asset best: large, confident and fully inside the photo."""
    def quality(m):
        x1, y1, x2, y2 = m["box"]
        im = s.IMAGES[m["photo"]]
        cut = x1 <= 2 or y1 <= 2 or x2 >= im["width"] - 2 or y2 >= im["height"] - 2
        return (x2 - x1) * (y2 - y1) * m["conf"] * (0.2 if cut else 1.0)
    return max(members, key=quality)


def save_crop(view, path, margin=0.25, max_side=900):
    img = cv2.imread(str(s.IMG_DIR / view["photo"]))
    x1, y1, x2, y2 = view["box"]
    m = margin * max(x2 - x1, y2 - y1)
    crop = img[int(max(y1 - m, 0)):int(y2 + m), int(max(x1 - m, 0)):int(x2 + m)]
    k = max_side / max(crop.shape[:2])
    if k < 1:
        crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--min-sweeps", type=int, default=5, help="scan positions that must see a spot")
    ap.add_argument("--radius", type=float, default=0.15, help="metres within which sightings are one asset")
    args = ap.parse_args()

    data = json.loads(DETECTIONS.read_text())
    names = {int(k): v for k, v in data["classes"].items()}
    items = sightings(data["detections"], args.conf)
    groups = cluster(items, args.radius)
    kept = [g for g in groups if len({m["sweep"] for m in g["members"]}) >= args.min_sweeps]
    rejected = [g for g in groups if g not in kept]

    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    for old in (OUT / "crops").glob("*.jpg"):
        old.unlink()
    assets = []
    for g in kept:
        m = g["members"]
        g["xyz"] = np.average([x["xyz"] for x in m], axis=0, weights=[x["conf"] for x in m])
    for i, g in enumerate(sorted(kept, key=lambda g: (g["cls"], *np.round(g["xyz"], 1))), 1):
        m = g["members"]
        view = best_view(m)
        cam_pos = s.cam(s.IMAGES[view["photo"]])[1]
        normal = surface_normal(view["photo"], view["box"], cam_pos)
        aid = f"asset-{i:02d}"
        save_crop(view, OUT / "crops" / f"{aid}.jpg")
        assets.append(dict(
            id=aid, product=names[g["cls"]], class_id=g["cls"],
            position_xyz=[round(float(v), 3) for v in g["xyz"]],
            normal_xyz=[round(float(v), 3) for v in normal] if normal is not None else None,
            sightings=len(m), sweeps=sorted({x["sweep"] for x in m}),
            mean_confidence=round(float(np.mean([x["conf"] for x in m])), 3),
            max_confidence=round(float(max(x["conf"] for x in m)), 3),
            spread_m=round(float(np.max(np.linalg.norm([x["xyz"] - g["xyz"] for x in m], axis=1))), 3),
            best_photo=view["photo"], best_box=[round(v) for v in view["box"]], crop=f"crops/{aid}.jpg",
            seen_in=[dict(photo=x["photo"], box=[round(v) for v in x["box"]], conf=round(x["conf"], 3)) for x in m]))
    (OUT / "assets.json").write_text(json.dumps(
        dict(settings=vars(args), source=str(DETECTIONS), assets=assets), indent=1))

    print(f"{len(items)} detections (conf>={args.conf}) -> {len(groups)} spots -> "
          f"{len(assets)} assets seen from >={args.min_sweeps} scan positions")
    gt = json.loads(GT.read_text())["relays"] if GT.exists() else {}
    used = set()
    for a in assets:
        label = ""
        if gt:  # compare with the hand-made ground truth
            rid, r = min(gt.items(), key=lambda kv: np.linalg.norm(np.array(kv[1]["center_xyz"]) - a["position_xyz"]))
            err = np.linalg.norm(np.array(r["center_xyz"]) - a["position_xyz"])
            label = f"= {rid} (off by {err * 100:.1f} cm)" if err < 0.2 else "= NOT A RELAY"
            used.add(rid) if err < 0.2 else None
        print(f"  {a['id']} at {a['position_xyz']}  {a['sightings']:2d} sightings from {len(a['sweeps']):2d} "
              f"positions, conf {a['mean_confidence']:.2f}  {label}")
    if gt:
        print(f"  relays located: {len(used)}/{len(gt)}; missing: {sorted(set(gt) - used) or 'none'}")
    print(f"rejected spots: {len(rejected)} "
          f"({sum(len(g['members']) for g in rejected)} detections), by scan positions seen from: "
          f"{sorted(len({m['sweep'] for m in g['members']}) for g in rejected)}")


if __name__ == "__main__":
    main()
