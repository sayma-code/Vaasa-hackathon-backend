"""Prepare the data the 3D viewer needs, in ../frontend-full/data/. Run from the backend folder.

- cloud.bin      downsampled coloured point cloud of the room (float32 xyz + uint8 rgb)
- pano/*.jpg     the scan photos at half size, for the walk-through view
- scene.json     scan positions, photo orientations and the tags
- crops/*.jpg    close-up of every tagged asset
Usage: python build_viewer_data.py [--voxel 0.02] [--skip-cloud] [--skip-pano]
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pye57

SRC = "cloud_0.e57"
IMG_DIR = Path("scan_images")
TAGS = Path("assets/tags.json")
OUT = Path("../frontend-full/data")
# Room extent in scan coordinates (metres); the roof is cut off so the room can be seen from above.
BOUNDS = np.array([[-10.5, -8.5, -0.4], [3.0, 3.0, 2.45]])


def voxel_filter(xyz, rgb, voxel):
    """Keep one point per voxel."""
    keys = np.floor((xyz - BOUNDS[0]) / voxel).astype(np.int64)
    flat = (keys[:, 0] << 42) | (keys[:, 1] << 21) | keys[:, 2]
    _, idx = np.unique(flat, return_index=True)
    return xyz[idx], rgb[idx]


def build_cloud(voxel):
    e57 = pye57.E57(SRC)
    xyz_all, rgb_all = [], []
    for i in range(e57.scan_count):
        d = e57.read_scan(i, colors=True, ignore_missing_fields=True)
        xyz = np.c_[d["cartesianX"], d["cartesianY"], d["cartesianZ"]]
        rgb = np.c_[d["colorRed"], d["colorGreen"], d["colorBlue"]].astype(np.uint8)
        inside = ((xyz >= BOUNDS[0]) & (xyz <= BOUNDS[1])).all(axis=1)
        xyz, rgb = voxel_filter(xyz[inside], rgb[inside], voxel)
        xyz_all.append(xyz.astype(np.float32))
        rgb_all.append(rgb)
        print(f"sweep {i}: {len(xyz)} points", flush=True)
    xyz, rgb = voxel_filter(np.concatenate(xyz_all).astype(np.float64), np.concatenate(rgb_all), voxel)
    with open(OUT / "cloud.bin", "wb") as f:
        f.write(xyz.astype("<f4").tobytes())
        f.write(rgb.tobytes())
    print(f"cloud: {len(xyz)} points, {(OUT / 'cloud.bin').stat().st_size / 1e6:.0f} MB")
    return len(xyz)


def build_panos(size=2048):
    (OUT / "pano").mkdir(exist_ok=True)
    for f in sorted(IMG_DIR.glob("sweep*_face*.jpg")):
        img = cv2.resize(cv2.imread(str(f)), (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / "pano" / f.name), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"panoramas: {len(list((OUT / 'pano').glob('*.jpg')))} photos")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxel", type=float, default=0.02, help="point spacing in metres")
    ap.add_argument("--skip-cloud", action="store_true")
    ap.add_argument("--skip-pano", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    scene_path = OUT / "scene.json"
    n_points = json.loads(scene_path.read_text()).get("points", 0) if scene_path.exists() else 0
    if not args.skip_cloud:
        n_points = build_cloud(args.voxel)
    if not args.skip_pano:
        build_panos()

    meta = json.loads((IMG_DIR / "poses.json").read_text())
    sweeps = []
    for sw in meta["sweeps"]:
        faces = [im for im in meta["images"] if im["sweep"] == sw["index"]]
        w, x, y, z = faces[0]["rotation_wxyz"]
        sweeps.append(dict(index=sw["index"], position=faces[0]["translation_xyz"], faces=[
            dict(file=f"pano/{im['file']}", quaternion_xyzw=im["rotation_wxyz"][1:] + im["rotation_wxyz"][:1])
            for im in faces]))

    tags = json.loads(TAGS.read_text())["tags"]
    shutil.rmtree(OUT / "crops", ignore_errors=True)
    shutil.copytree(TAGS.parent / "crops", OUT / "crops")
    for t in tags:
        t["best_sweep"] = int(t["detection"]["best_photo"][5:7])
    scene_path.write_text(json.dumps(dict(points=n_points, bounds=BOUNDS.tolist(), sweeps=sweeps, tags=tags)))
    print(f"scene: {len(sweeps)} scan positions, {len(tags)} tags -> {OUT}/")


if __name__ == "__main__":
    main()
