"""Train the product detector on the photo dataset built by make_photo_dataset.py.

The dataset holds the labelled photos of test_photos/ and synthetic variations of them.
Usage: python train.py [--model yolo11s.pt] [--epochs 40] [--imgsz 1024] [--batch 8]
Result: runs/detector/weights/best.pt
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "dataset_photos" / "data.yaml"))
    ap.add_argument("--model", default="yolo11s.pt")  # pretrained weights, downloaded on first use
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)   # lower to 4 if the GPU runs out of memory
    ap.add_argument("--name", default="detector")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run")
    args = ap.parse_args()

    if args.resume:
        YOLO(str(ROOT / "runs" / args.name / "weights" / "last.pt")).train(resume=True)
        return

    YOLO(args.model).train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=4,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        patience=10,          # stop early if validation stops improving
        cos_lr=True,
        close_mosaic=5,
        # The synthetic generator already varies lighting, viewpoint and blur, so keep
        # the built-in augmentation mild. No flips: mirrored text and layout never occur.
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.5,
        scale=0.5,
        translate=0.1,
        degrees=3.0,
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        plots=True,
    )


if __name__ == "__main__":  # required on Windows: the data loader starts worker processes
    main()
