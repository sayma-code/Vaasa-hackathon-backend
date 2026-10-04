"""Build the relay ground truth for every scan photo.

Each physical relay is marked once by hand in a close-up photo, lifted to 3D with the
point cloud, then projected into all photos with an occlusion check.
Writes ground_truth/relays.json, ground_truth/ground_truth.csv and preview images.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import scan3d as s

OUT = Path("ground_truth")
WIDE = "ABB Relion 615, wide HMI (F1-F16)"
COMPACT = "ABB Relion 615, compact HMI (F1-F4)"

# Front plate marked by hand in a close-up: (photo, x1, y1, x2, y2) in full-res pixels.
FULL_BOXES = {
    "H01": ("sweep10_face1.jpg", 712, 521, 1485, 1031, WIDE, "ABB switchgear panel H01"),
    "H04": ("sweep13_face1.jpg", 801, 592, 1517, 1076, WIDE, "ABB switchgear panel H04"),
    "OT1-upper": ("sweep07_face1.jpg", 1821, 628, 2746, 1240, WIDE, "VEO cabinet OT1, upper"),
    "OT1-lower": ("sweep07_face1.jpg", 2000, 1772, 2605, 2375, COMPACT, "VEO cabinet OT1, lower"),
}
# Relays cut by the photo edge in their close-up: one marked plate corner, size copied from H01.
CORNERS = {
    "H02": ("sweep11_face1.jpg", 642, 553, "tl", WIDE, "ABB switchgear panel H02"),
    "H03": ("sweep12_face1.jpg", 592, 524, "tl", WIDE, "ABB switchgear panel H03"),
    "H05": ("sweep14_face1.jpg", 1574, 677, "tr", WIDE, "ABB switchgear panel H05"),
}


def fit_plane(pts):
    c = pts.mean(axis=0)
    n = np.linalg.svd(pts - c)[2][-1]
    return c, n


def ray_plane(file, x, y, c, n):
    """3D point where the viewing ray of a pixel meets a plane."""
    im = s.IMAGES[file]
    R, t = s.cam(im)
    d = R @ np.array([(x - im["cx"]) / im["fx"], -(y - im["cy"]) / im["fy"], -1.0])
    return t + d * ((c - t) @ n) / (d @ n)


def plate_plane(file, box):
    """Plane of a relay front plate from the scan points inside (a shrunken) pixel box."""
    x1, y1, x2, y2 = box
    mx, my = 0.15 * (x2 - x1), 0.15 * (y2 - y1)
    pts, _ = s.points_in_box(file, (x1 + mx, y1 + my, x2 - mx, y2 - my))
    c, n = fit_plane(pts)
    keep = np.abs((pts - c) @ n) < 0.01  # refit without buttons, glare and stray points
    return fit_plane(pts[keep])


def build_relays():
    relays = {}
    for rid, (file, x1, y1, x2, y2, model, where) in FULL_BOXES.items():
        c, n = plate_plane(file, (x1, y1, x2, y2))
        tl, tr, br, bl = (ray_plane(file, x, y, c, n) for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)))
        relays[rid] = dict(model=model, location=where, marked_in=file, corners=[tl, tr, br, bl])
    tl, tr, br, bl = relays["H01"]["corners"]
    right, down = tr - tl, bl - tl
    for rid, (file, x, y, which, model, where) in CORNERS.items():
        sx = 1 if which == "tl" else -1
        c, n = plate_plane(file, (min(x, x + sx * 400), y, max(x, x + sx * 400), y + 400))
        p = ray_plane(file, x, y, c, n)
        p_tl = p if which == "tl" else p - right
        relays[rid] = dict(model=model, location=where, marked_in=file,
                           corners=[p_tl, p_tl + right, p_tl + right + down, p_tl + down])
    for r in relays.values():
        tl, tr, br, bl = r["corners"]
        n = np.cross(tr - tl, bl - tl)
        n /= np.linalg.norm(n)
        r["center"] = (tl + br) / 2
        r["width_m"] = float(np.linalg.norm(tr - tl))
        r["height_m"] = float(np.linalg.norm(bl - tl))
        cam_pos = s.cam(s.IMAGES[r["marked_in"]])[1]
        r["normal"] = n if n @ (cam_pos - r["center"]) > 0 else -n  # points out of the cabinet
    return dict(sorted(relays.items()))


def appearances(relays):
    rows = []
    for file, im in sorted(s.IMAGES.items()):
        W, H = im["width"], im["height"]
        cam_pos = s.cam(im)[1]
        for rid, r in relays.items():
            u, v, d = s.project(im, np.array(r["corners"] + [r["center"]]))
            if (d <= 0.05).any():
                continue
            x1, y1, x2, y2 = u[:4].min(), v[:4].min(), u[:4].max(), v[:4].max()
            cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            in_frame = (cx2 - cx1) * (cy2 - cy1) / ((x2 - x1) * (y2 - y1))
            to_cam = cam_pos - r["center"]
            dist = float(np.linalg.norm(to_cam))
            angle = float(np.degrees(np.arccos(np.clip(r["normal"] @ to_cam / dist, -1, 1))))
            if angle >= 85 or in_frame < 0.2:
                continue
            # occlusion: does the photo's own sweep see a surface clearly in front of the relay?
            seen = s.depth_at(file, (cx1 + cx2) / 2, (cy1 + cy2) / 2, radius=max(6, (cx2 - cx1) / 4))
            if seen is not None and seen < d[4] - 0.25:
                continue
            w_px = float(cx2 - cx1)
            hard = [name for name, bad in (("cut by photo edge", in_frame < 0.95), ("steep angle", angle > 65),
                                           ("small", w_px < 60)) if bad]
            rows.append(dict(photo=file, relay=rid, model=r["model"], location=r["location"],
                             x1=round(cx1), y1=round(cy1), x2=round(cx2), y2=round(cy2),
                             width_px=round(w_px), distance_m=round(dist, 2), view_angle_deg=round(angle),
                             in_frame=round(in_frame, 2), difficulty="hard: " + ", ".join(hard) if hard else "clear"))
    return rows


def previews(rows):
    by_photo = {}
    for row in rows:
        by_photo.setdefault(row["photo"], []).append(row)
    (OUT / "preview").mkdir(parents=True, exist_ok=True)
    for file, items in by_photo.items():
        img = cv2.imread(str(s.IMG_DIR / file))
        for it in items:
            color = (0, 200, 0) if it["difficulty"] == "clear" else (0, 140, 255)
            cv2.rectangle(img, (it["x1"], it["y1"]), (it["x2"], it["y2"]), color, 6)
            cv2.putText(img, it["relay"], (it["x1"], max(it["y1"] - 14, 40)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)
        cv2.imwrite(str(OUT / "preview" / file), cv2.resize(img, (1536, 1536), interpolation=cv2.INTER_AREA))


def main():
    OUT.mkdir(exist_ok=True)
    relays = build_relays()
    rows = appearances(relays)
    as_list = lambda a: [round(float(x), 4) for x in a]
    (OUT / "relays.json").write_text(json.dumps({
        "relays": {rid: dict(model=r["model"], location=r["location"], marked_in=r["marked_in"],
                             center_xyz=as_list(r["center"]), normal_xyz=as_list(r["normal"]),
                             corners_xyz=[as_list(c) for c in r["corners"]],
                             width_m=round(r["width_m"], 3), height_m=round(r["height_m"], 3))
                   for rid, r in relays.items()},
        "appearances": rows}, indent=1))
    with open(OUT / "ground_truth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    previews(rows)
    for rid, r in relays.items():
        n = [x for x in rows if x["relay"] == rid]
        print(f"{rid:10s} {r['model']:38s} centre {np.round(r['center'], 2)} "
              f"{r['width_m'] * 1000:.0f}x{r['height_m'] * 1000:.0f} mm  "
              f"photos {len(n)} (clear {sum(x['difficulty'] == 'clear' for x in n)})")
    print(f"{len(rows)} appearances in {len({x['photo'] for x in rows})} photos")


if __name__ == "__main__":
    main()
