"""Build a training dataset from the real photos in test_photos/, plus synthetic variations of them.

Labels come from ground_truth/relays.json for the scan photos (sweepNN_faceK.jpg) and from
photo_labels.json for the other photos. The synthetic images are made from the training photos
only: a random zoomed crop around a relay, a slight change of viewpoint, and changes of
lighting, colour, blur and noise.
Usage: python make_photo_dataset.py [--factor 2] [--val 0.15] [--size 1024] [--seed 0]
Writes dataset_photos/ in YOLO format.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

SRC = Path("test_photos")
OUT = Path("dataset_photos")
GT = Path("ground_truth/relays.json")
WEB_LABELS = Path("photo_labels.json")
CLASS_NAME = "ABB Relion 615 protection relay"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_photos():
    """[(path, boxes in pixels)] for every usable photo; reports what is skipped and why."""
    scan = {}
    for a in json.loads(GT.read_text())["appearances"]:
        scan.setdefault(a["photo"], []).append([a["x1"], a["y1"], a["x2"], a["y2"]])
    web = json.loads(WEB_LABELS.read_text())
    items, seen, skipped = [], set(), []
    for f in sorted(p for p in SRC.iterdir() if p.suffix.lower() in IMG_EXT):
        digest = hashlib.md5(f.read_bytes()).hexdigest()
        if digest in seen:
            skipped.append((f.name, "duplicate of another photo"))
            continue
        seen.add(digest)
        img = cv2.imread(str(f))
        if img is None:
            skipped.append((f.name, "cannot be opened"))
        elif f.name in scan:
            items.append((f, [b for b in scan[f.name] if b[2] - b[0] >= 12 and b[3] - b[1] >= 12]))
        elif f.name.startswith("sweep"):
            items.append((f, []))  # scan photo with no relay in view: a useful empty example
        elif web.get(f.name):
            h, w = img.shape[:2]
            items.append((f, [[x1 * w, y1 * h, x2 * w, y2 * h] for x1, y1, x2, y2 in web[f.name]]))
        else:
            skipped.append((f.name, "left out on purpose" if f.name in web else "no labels: add it to photo_labels.json"))
    return items, skipped


def variation(rng, img, boxes, size):
    """One synthetic image from a real photo: zoomed crop, viewpoint, lighting, blur, noise."""
    h, w = img.shape[:2]
    if boxes:  # a square window around one relay, with the relay 1/8 to 1/2.5 of the window
        x1, y1, x2, y2 = boxes[rng.integers(len(boxes))]
        side = max(x2 - x1, y2 - y1) * rng.uniform(2.5, 8)
        side = float(np.clip(side, 320, min(h, w)))
        cx = rng.uniform(max(x2 - side / 2, side / 2), min(x1 + side / 2, w - side / 2)) if w > side else w / 2
        cy = rng.uniform(max(y2 - side / 2, side / 2), min(y1 + side / 2, h - side / 2)) if h > side else h / 2
        cx, cy = float(np.clip(cx, side / 2, w - side / 2)), float(np.clip(cy, side / 2, h - side / 2))
    else:
        side = min(h, w) * rng.uniform(0.3, 1.0)
        cx, cy = rng.uniform(side / 2, w - side / 2), rng.uniform(side / 2, h - side / 2)
    src = np.float32([[cx - side / 2, cy - side / 2], [cx + side / 2, cy - side / 2],
                      [cx + side / 2, cy + side / 2], [cx - side / 2, cy + side / 2]])
    src += rng.uniform(-0.06, 0.06, (4, 2)).astype(np.float32) * side  # slight change of viewpoint
    dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (size, size), flags=cv2.INTER_AREA if side > size else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    new_boxes = []
    for x1, y1, x2, y2 in boxes:
        corners = cv2.perspectiveTransform(np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]), M)[0]
        bx1, by1, bx2, by2 = corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()
        full = (bx2 - bx1) * (by2 - by1)
        cx1, cy1, cx2, cy2 = max(bx1, 0), max(by1, 0), min(bx2, size), min(by2, size)
        if cx2 - cx1 >= 10 and cy2 - cy1 >= 10 and (cx2 - cx1) * (cy2 - cy1) >= 0.4 * full:
            new_boxes.append([cx1, cy1, cx2, cy2])

    f = out.astype(np.float32) * rng.uniform(0.65, 1.3) * rng.uniform(0.93, 1.07, 3)  # exposure, colour cast
    f = 127 + (f - 127) * rng.uniform(0.7, 1.2)                                          # contrast
    f = 255 * (np.clip(f, 0, 255) / 255) ** rng.uniform(0.75, 1.35)                      # gamma
    out = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 2.0))
    if rng.random() < 0.5:
        out = np.clip(out + rng.normal(0, rng.uniform(2, 10), out.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(45, 95))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), new_boxes


def write(split, name, img, boxes):
    h, w = img.shape[:2]
    cv2.imwrite(str(OUT / "images" / split / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    lines = [f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n"
             for x1, y1, x2, y2 in boxes]
    (OUT / "labels" / split / name).with_suffix(".txt").write_text("".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=float, default=2, help="synthetic images per real photo")
    ap.add_argument("--val", type=float, default=0.15, help="share of real photos held out for validation")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    items, skipped = load_photos()
    shutil.rmtree(OUT, ignore_errors=True)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)

    order = rng.permutation(len(items))
    n_val = round(len(items) * args.val)
    val = {int(i) for i in order[:n_val]}
    train = []
    for i, (path, boxes) in enumerate(items):
        img = cv2.imread(str(path))
        write("val" if i in val else "train", f"real_{path.stem.replace(' ', '_')}.jpg", img, boxes)
        if i not in val:
            train.append((img, boxes))

    n_synth = round(len(items) * args.factor)
    for k in range(n_synth):  # validation photos are never used as a source
        img, boxes = train[k % len(train)]
        out, new_boxes = variation(rng, img, boxes, args.size)
        write("train", f"synth_{k + 1:04d}.jpg", out, new_boxes)

    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: {CLASS_NAME}\n")
    n_boxes = sum(len(b) for _, b in items)
    print(f"real photos: {len(items)} ({len(items) - n_val} train, {n_val} validation), {n_boxes} relays marked")
    print(f"synthetic images: {n_synth}, made from the {len(train)} training photos")
    print(f"dataset: {len(items) + n_synth} images -> {OUT}/")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")


if __name__ == "__main__":
    main()
