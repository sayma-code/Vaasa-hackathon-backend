"""Check the detector with the standard classification metrics: accuracy, precision, recall,
F1-score and ROC-AUC.

The detector outputs boxes, not yes/no answers, so each metric is defined on boxes:
  correct detection  a box that overlaps a marked relay (IoU >= --iou); each relay counts once
  false alarm        a box that matches no marked relay
  missed relay       a marked relay with no matching box
  precision  = correct / (correct + false alarms)        how many of its boxes are right
  recall     = correct / (correct + missed)              how many of the relays it finds
  F1         = 2 * precision * recall / (precision + recall)
  accuracy   = correct / (correct + false alarms + missed)
               (there is no "correctly found nothing" count for boxes, so this is the usual
               detection form of accuracy)
  ROC-AUC    how well the confidence score separates correct detections from false alarms,
               over every box the detector proposes; 1.0 is perfect, 0.5 is no better than chance

Usage: python check_metrics.py [--split val] [--conf 0.5] [--iou 0.5] [--weights runs/detector/weights/best.pt]
  --split val    the photos held out from training (the fair test, default)
  --split train  the photos the detector was trained on
  --split all    both
Writes eval/metrics_<split>.txt
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from detector import detect, iou, scales_for

DATA = Path("dataset_photos")
OUT = Path("eval")


def load_split(split):
    """[(photo path, [marked boxes in pixels])] for the real photos of a split."""
    items = []
    for s in (("train", "val") if split == "all" else (split,)):
        for img_path in sorted((DATA / "images" / s).glob("real_*.jpg")):  # real photos only, not the variations
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            boxes = []
            for line in (DATA / "labels" / s / img_path.name).with_suffix(".txt").read_text().split("\n"):
                if line.strip():
                    _, cx, cy, bw, bh = map(float, line.split())
                    boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            items.append((img_path, boxes))
    return items


def match(dets, boxes, min_iou):
    """For each detection (strongest first): True if it is the first to match a marked relay."""
    taken, correct = set(), []
    for d in dets:
        best = max(((iou(d, b), k) for k, b in enumerate(boxes) if k not in taken), default=(0, None))
        ok = best[0] >= min_iou
        if ok:
            taken.add(best[1])
        correct.append(ok)
    return correct


def roc_auc(scores, labels):
    """Area under the ROC curve from scores and 0/1 labels (rank formula, ties share their rank)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for s in np.unique(scores):  # average the ranks of equal scores
        tie = scores == s
        ranks[tie] = ranks[tie].mean()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def counts_at(scores, correct, n_relays, conf):
    keep = scores >= conf
    tp = int((correct & keep).sum())
    fp = int((~correct & keep).sum())
    fn = n_relays - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return tp, fp, fn, accuracy, precision, recall, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detector/weights/best.pt")
    ap.add_argument("--split", choices=["val", "train", "all"], default="val")
    ap.add_argument("--conf", type=float, default=0.5, help="confidence a box needs to count as a detection")
    ap.add_argument("--iou", type=float, default=0.5, help="overlap a box needs with a marked relay to be correct")
    args = ap.parse_args()

    items = load_split(args.split)
    model = YOLO(args.weights)
    scores, correct, n_relays, per_photo = [], [], 0, []
    for path, boxes in items:
        img = cv2.imread(str(path))
        dets = sorted(detect(model, img, scales=scales_for(img)).tolist(), key=lambda d: -d[4])
        ok = match(dets, boxes, args.iou)
        scores += [d[4] for d in dets]
        correct += ok
        n_relays += len(boxes)
        shown = [(d[4], o) for d, o in zip(dets, ok) if d[4] >= args.conf]
        per_photo.append((path.name, len(boxes), sum(o for _, o in shown), sum(not o for _, o in shown)))
    scores, correct = np.array(scores), np.array(correct, bool)

    tp, fp, fn, accuracy, precision, recall, f1 = counts_at(scores, correct, n_relays, args.conf)
    auc = roc_auc(scores, correct)
    seen = {"val": "photos held out from training (fair test)", "train": "photos the detector was trained on",
            "all": "training and held-out photos together"}[args.split]
    lines = [
        f"Detector: {args.weights}",
        f"Photos: {len(items)} {seen}, showing {n_relays} relays",
        f"A box counts at confidence >= {args.conf} and is correct at overlap (IoU) >= {args.iou}",
        "",
        f"Correct detections   {tp}",
        f"False alarms         {fp}",
        f"Missed relays        {fn}",
        "",
        f"Accuracy             {accuracy:.1%}",
        f"Precision            {precision:.1%}",
        f"Recall               {recall:.1%}",
        f"F1-score             {f1:.1%}",
        "ROC-AUC              " + (f"{auc:.3f}  (over {len(scores)} proposed boxes: {int(correct.sum())} correct, "
                                   f"{int((~correct).sum())} false)" if auc is not None else
                                   "cannot be computed: the proposed boxes are all correct or all false"),
        "",
        "At other confidence levels:",
        "  conf   accuracy  precision  recall   F1     correct  false  missed",
    ]
    for c in (0.25, 0.4, 0.5, 0.6, 0.7, 0.8):
        t, f_, m, a, p, r, f1c = counts_at(scores, correct, n_relays, c)
        lines.append(f"  {c:.2f}   {a:7.1%}  {p:8.1%}  {r:6.1%}  {f1c:6.1%}  {t:6d}  {f_:5d}  {m:6d}")
    lines += ["", "Per photo (relays marked / found / false alarms):"]
    lines += [f"  {name:45s} {n} / {found} / {false}" for name, n, found, false in per_photo]

    text = "\n".join(lines)
    OUT.mkdir(exist_ok=True)
    (OUT / f"metrics_{args.split}.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
