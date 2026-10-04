"""Geometry helpers linking the scan photos to the point cloud.

pixel -> 3D (via the sweep's own points) and 3D -> pixel (any photo).
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

SRC = "cloud_0.e57"
IMG_DIR = Path("scan_images")
PTS_DIR = Path("scan_points")

META = json.loads((IMG_DIR / "poses.json").read_text())
IMAGES = {im["file"]: im for im in META["images"]}


def cam(im):
    """Rotation (camera-to-world) and position of a photo."""
    return Quaternion(im["rotation_wxyz"]).rotation_matrix, np.array(im["translation_xyz"])


@lru_cache(maxsize=4)
def sweep_points(sweep):
    """World-space points of one sweep, cached on disk as float32."""
    path = PTS_DIR / f"sweep{sweep:02d}.npy"
    if not path.exists():
        import pye57
        PTS_DIR.mkdir(exist_ok=True)
        d = pye57.E57(SRC).read_scan(sweep, ignore_missing_fields=True)
        np.save(path, np.c_[d["cartesianX"], d["cartesianY"], d["cartesianZ"]].astype(np.float32))
    return np.load(path)


def project(im, pts):
    """World points -> (u, v, depth). depth <= 0 means behind the camera."""
    R, t = cam(im)
    pc = (np.atleast_2d(pts) - t) @ R
    depth = -pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = im["cx"] + im["fx"] * pc[:, 0] / depth
        v = im["cy"] - im["fy"] * pc[:, 1] / depth
    return u, v, depth


@lru_cache(maxsize=8)
def _projected(file):
    im = IMAGES[file]
    pts = sweep_points(im["sweep"])
    u, v, d = project(im, pts)
    ok = (d > 0.05) & (u >= 0) & (u < im["width"]) & (v >= 0) & (v < im["height"])
    return pts[ok], u[ok], v[ok], d[ok]


def points_in_box(file, box):
    """Scan points of the photo's own sweep that fall inside a pixel box (x1, y1, x2, y2)."""
    pts, u, v, d = _projected(file)
    x1, y1, x2, y2 = box
    m = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    return pts[m], d[m]


def pixel_to_world(file, x, y, radius=12):
    """3D point seen at a pixel: median of the nearest-surface scan points around it."""
    pts, d = points_in_box(file, (x - radius, y - radius, x + radius, y + radius))
    if len(pts) == 0:
        return None
    near = d <= np.percentile(d, 25) + 0.05  # keep the front surface
    return np.median(pts[near], axis=0)


def depth_at(file, x, y, radius=12):
    """Range of the nearest surface the photo's sweep measured at a pixel (None if no points)."""
    _, d = points_in_box(file, (x - radius, y - radius, x + radius, y + radius))
    return float(np.percentile(d, 25)) if len(d) else None
