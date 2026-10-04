"""Run the trained detector on the real scan photos and score it against the ground truth.

The tiled multi-scale search itself lives in detector.py.
Usage: python evaluate_scan.py [--weights runs/detector/weights/best.pt] [--conf 0.5]
Writes eval/detections.json, eval/report.txt and eval/preview/*.jpg
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import scan3d as s
from detector import detect, iou

GT = Path("ground_truth/relays.json")
OUT = Path("eval")


def all_projections(relays):
    """Every relay's box in every photo, without the visibility filters used for the ground truth.

    Detections that land on these are real relays seen under conditions too poor to demand
    (extreme angle, mostly out of frame), so they are not counted as false alarms.
    """
    out = {}
    for file, im in s.IMAGES.items():
        for r in relays.values():
            u, v, d = s.project(im, np.array(r["corners_xyz"]))
            if (d > 0.05).all():
                out.setdefault(file, []).append([u.min(), v.min(), u.max(), v.max()])
    return out


def score(dets, gt_rows, loose, conf):
    """Match detections to ground truth at one confidence threshold."""
    by_photo = {}
    for g in gt_rows:
        by_photo.setdefault(g["photo"], []).append(g)
    found, false_alarms = set(), []
    for file, rows in dets.items():
        rows = sorted((r for r in rows if r[4] >= conf), key=lambda r: -r[4])
        gts = by_photo.get(file, [])
        for r in rows:
            best = max(gts, key=lambda g: iou(r, [g["x1"], g["y1"], g["x2"], g["y2"]]), default=None)
            if best is not None and iou(r, [best["x1"], best["y1"], best["x2"], best["y2"]]) >= 0.3:
                found.add((file, best["relay"]))
            elif not any(iou(r, b) > 0.1 for b in loose.get(file, [])):
                false_alarms.append((file, r))
    return found, false_alarms


def report(dets, gt, conf):
    rows, relays = gt["appearances"], gt["relays"]
    loose = all_projections(relays)
    lines = ["Confidence sweep (appearances found / false alarms):"]
    for c in (0.25, 0.4, 0.5, 0.6, 0.7, 0.8):
        f, fa = score(dets, rows, loose, c)
        clear = sum((r["photo"], r["relay"]) in f for r in rows if r["difficulty"] == "clear")
        lines.append(f"  conf>={c:.2f}: all {len(f)}/{len(rows)}  clear {clear}/"
                     f"{sum(r['difficulty'] == 'clear' for r in rows)}  false alarms {len(fa)}")
    found, false_alarms = score(dets, rows, loose, conf)
    lines.append(f"\nAt conf>={conf}:")
    for rid, r in relays.items():
        mine = [x for x in rows if x["relay"] == rid]
        clear = [x for x in mine if x["difficulty"] == "clear"]
        hit = lambda xs: sum((x["photo"], x["relay"]) in found for x in xs)
        lines.append(f"  {rid:10s} {r['model']:38s} found in {hit(mine):2d}/{len(mine)} photos "
                     f"(clear {hit(clear)}/{len(clear)})")
    n_assets = sum(any((x["photo"], x["relay"]) in found for x in rows if x["relay"] == rid) for rid in relays)
    lines.append(f"  relays found at least once: {n_assets}/{len(relays)}")
    lines.append(f"  false alarms: {len(false_alarms)} in {len({f for f, _ in false_alarms})} photos")
    for f, r in sorted(false_alarms, key=lambda x: -x[1][4])[:15]:
        lines.append(f"    {f} box {[round(v) for v in r[:4]]} conf {r[4]:.2f}")
    return "\n".join(lines), found, false_alarms


def previews(dets, gt, found, false_alarms, conf):
    (OUT / "preview").mkdir(parents=True, exist_ok=True)
    fa = {(f, tuple(r[:4])) for f, r in false_alarms}
    gt_by_photo = {}
    for g in gt["appearances"]:
        gt_by_photo.setdefault(g["photo"], []).append(g)
    for file in sorted(set(gt_by_photo) | {f for f, _ in false_alarms}):
        img = cv2.imread(str(s.IMG_DIR / file))
        for g in gt_by_photo.get(file, []):
            if (file, g["relay"]) not in found:  # missed: yellow
                cv2.rectangle(img, (g["x1"], g["y1"]), (g["x2"], g["y2"]), (0, 220, 255), 6)
                cv2.putText(img, f"missed {g['relay']}", (g["x1"], g["y2"] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 220, 255), 4)
        for r in dets.get(file, []):
            if r[4] < conf:
                continue
            color = (0, 0, 255) if (file, tuple(r[:4])) in fa else (0, 200, 0)  # false alarm: red, correct: green
            p1, p2 = (int(r[0]), int(r[1])), (int(r[2]), int(r[3]))
            cv2.rectangle(img, p1, p2, color, 6)
            cv2.putText(img, f"{r[4]:.2f}", (p1[0], max(p1[1] - 14, 40)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
        cv2.imwrite(str(OUT / "preview" / file), cv2.resize(img, (1536, 1536), interpolation=cv2.INTER_AREA))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detector/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--reuse", action="store_true", help="score the saved eval/detections.json again")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    if args.reuse:
        dets = json.loads((OUT / "detections.json").read_text())["detections"]
    else:
        model = YOLO(args.weights)
        dets = {}
        for i, file in enumerate(sorted(s.IMAGES), 1):
            dets[file] = [[round(float(v), 3) for v in r] for r in detect(model, cv2.imread(str(s.IMG_DIR / file)))]
            if i % 12 == 0:
                print(f"{i}/{len(s.IMAGES)} photos", flush=True)
        (OUT / "detections.json").write_text(json.dumps(
            {"weights": args.weights, "classes": model.names, "format": "x1,y1,x2,y2,conf,class", "detections": dets}))

    gt = json.loads(GT.read_text())
    text, found, false_alarms = report(dets, gt, args.conf)
    (OUT / "report.txt").write_text(text)
    previews(dets, gt, found, false_alarms, args.conf)
    print(text)


if __name__ == "__main__":
    main()
