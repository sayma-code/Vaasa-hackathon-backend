"""Find products in a photo of any size.

The photo is searched in overlapping 1024 px tiles at several zoom levels, so both distant
and close-up products are seen at a size the model was trained on.
"""
import cv2
import numpy as np
import torch
from torchvision.ops import nms

TILE = 1024
SCALES = (1.0, 0.5, 0.25)  # photo zoom levels searched
MIN_CONF = 0.10            # weakest detection returned; callers apply their own threshold


def tiles(img, scales=SCALES):
    """Yield (tile, scale, x0, y0) covering the photo at every zoom level."""
    for scale in scales:
        im = img if scale == 1 else cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = im.shape[:2]
        step = TILE * 3 // 4
        xs = sorted({min(x, max(w - TILE, 0)) for x in range(0, max(w - TILE, 0) + step, step)})
        ys = sorted({min(y, max(h - TILE, 0)) for y in range(0, max(h - TILE, 0) + step, step)})
        for y in ys:
            for x in xs:
                yield im[y:y + TILE, x:x + TILE], scale, x, y


def scales_for(img):
    """Zoom levels worth searching: skip the ones that would shrink a small photo to nothing."""
    side = max(img.shape[:2])
    return tuple(s for s in SCALES if s == 1.0 or side * s >= 640)


def area(b):
    return max(b[2] - b[0], 0) * max(b[3] - b[1], 0)


def inter(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def iou(a, b):
    i = inter(a, b)
    return i / (area(a) + area(b) - i + 1e-9)


def inside(a, b):
    return inter(a, b) / (area(a) + 1e-9)


def detect(model, img, batch=16, scales=SCALES):
    """All detections in one photo as rows [x1, y1, x2, y2, conf, class] in full-res pixels."""
    items = list(tiles(img, scales))
    rows = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        results = model.predict([t[0] for t in chunk], imgsz=TILE, conf=MIN_CONF, verbose=False)
        for (tile, scale, x0, y0), r in zip(chunk, results):
            b = r.boxes
            if len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy()
            th, tw = tile.shape[:2]
            # a box touching a tile border is probably a cut-off piece; another tile sees it whole
            edge = 4
            whole = ((xyxy[:, 0] > edge) | (x0 == 0)) & ((xyxy[:, 1] > edge) | (y0 == 0)) \
                & ((xyxy[:, 2] < tw - edge) | (x0 + tw >= img.shape[1] * scale - 1)) \
                & ((xyxy[:, 3] < th - edge) | (y0 + th >= img.shape[0] * scale - 1))
            xyxy = (xyxy + [x0, y0, x0, y0]) / scale
            rows += [[*xyxy[k], float(b.conf[k]), int(b.cls[k])] for k in range(len(xyxy)) if whole[k]]
    if not rows:
        return np.zeros((0, 6))
    rows = np.array(rows)
    keep = nms(torch.tensor(rows[:, :4], dtype=torch.float32), torch.tensor(rows[:, 4], dtype=torch.float32), 0.5).numpy()
    rows = rows[keep]
    final = []  # drop boxes that sit mostly inside a stronger box (a part detected as a whole)
    for r in rows:
        if not any(inside(r, f) > 0.7 for f in final):
            final.append(r)
    return np.array(final)
