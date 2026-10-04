"""Generate a synthetic detection dataset from product reference images.

Every folder in products/ is one class: products/<id>/product.json + references/*.jpg
(clean renders on a white background). Backgrounds are procedural, plus any photos
dropped into backgrounds/ (outside material only - never the scan photos).

Usage: python make_synthetic.py [--n 4000] [--size 1024] [--out dataset] [--seed 0]
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

PRODUCTS = Path("products")
BACKGROUNDS = Path("backgrounds")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------- reference cut-outs

def load_cutout(path, screen_hsv=None, max_side=1100):
    """Cut the product out of a white-background render. Returns BGR, alpha (0..1), screen mask."""
    img = cv2.imread(str(path))
    white = (img.min(axis=2) >= 248).astype(np.uint8)
    n, labels = cv2.connectedComponents(white)
    border = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    bg = np.isin(labels, border[border > 0])  # white regions touching the image border
    alpha = cv2.erode((~bg).astype(np.uint8), np.ones((3, 3), np.uint8))
    ys, xs = np.where(alpha > 0)
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    img, alpha = img[y1:y2, x1:x2], alpha[y1:y2, x1:x2]

    screen = np.zeros(alpha.shape, np.uint8)
    if screen_hsv:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 0] >= screen_hsv["h"][0]) & (hsv[..., 0] <= screen_hsv["h"][1])
             & (hsv[..., 1] >= screen_hsv["s_min"]) & (hsv[..., 2] >= screen_hsv["v_min"])).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:  # the display is the largest coloured blob; fill its hull to cover drawn content
            cv2.fillConvexPoly(screen, cv2.convexHull(max(cnts, key=cv2.contourArea)), 1)

    s = max_side / max(img.shape[:2])
    if s < 1:
        size = (round(img.shape[1] * s), round(img.shape[0] * s))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
        screen = cv2.resize(screen, size, interpolation=cv2.INTER_NEAREST)
    return img, alpha.astype(np.float32), screen


def load_products():
    """[(class_id, name, [cutouts])] for every product folder, in a stable order."""
    out = []
    for cid, folder in enumerate(sorted(p for p in PRODUCTS.iterdir() if (p / "product.json").exists())):
        cfg = json.loads((folder / "product.json").read_text())
        refs = sorted(f for f in (folder / "references").iterdir() if f.suffix.lower() in IMG_EXT)
        out.append((cid, cfg["name"], [load_cutout(f, cfg.get("screen_hsv")) for f in refs]))
    return out


# ---------------------------------------------------------------- object appearance

def smooth_noise(rng, shape, cells, amp):
    """Low-frequency brightness field, shape (h, w, 1)."""
    small = rng.normal(0, amp, (cells, cells)).astype(np.float32)
    return cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)[..., None]


def randomize_screen(rng, img, screen):
    """Displays show different things at different sites: off, lit, tinted, washed out."""
    if not screen.any():
        return img
    r = rng.random()
    if r < 0.25:
        return img
    m = screen.astype(bool)
    out = img.copy()
    if r < 0.75:  # switched off: flat dull LCD colour
        hsv = np.uint8([[[rng.integers(10, 50), rng.integers(0, 90), rng.integers(60, 215)]]])
        color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
        flat = np.clip(color + smooth_noise(rng, img.shape, 4, 8), 0, 255)
        keep = rng.uniform(0, 0.15)  # faint ghost of the content
        out[m] = (keep * img[m] + (1 - keep) * flat[m]).astype(np.uint8)
    else:  # lit with another backlight colour / contrast
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[..., 0] = (hsv[..., 0] + rng.integers(0, 180)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.2, 1.3), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.5, 1.05), 0, 255)
        out[m] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)[m]
    return out


def wear(rng, img, alpha):
    """Label wear: fading, rubbed-off patches, scratches, dirt."""
    out = img.astype(np.float32)
    h, w = alpha.shape
    if rng.random() < 0.5:  # print fades towards the housing colour
        blur = cv2.GaussianBlur(out, (0, 0), max(w, h) / 30)
        out += (blur - out) * rng.uniform(0.2, 0.8)
    for _ in range(rng.integers(0, 4)):  # rubbed-off / smeared patches
        pw, ph = rng.integers(w // 12, w // 3), rng.integers(h // 12, h // 3)
        x, y = rng.integers(0, w - pw), rng.integers(0, h - ph)
        out[y:y + ph, x:x + pw] = cv2.GaussianBlur(out[y:y + ph, x:x + pw], (0, 0), rng.uniform(3, 12))
    for _ in range(rng.integers(0, 5)):  # scratches
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(p1[0] + rng.integers(-w // 3, w // 3)), int(p1[1] + rng.integers(-h // 3, h // 3)))
        cv2.line(out, p1, p2, [float(rng.integers(150, 255))] * 3, int(rng.integers(1, 3)), cv2.LINE_AA)
    if rng.random() < 0.4:  # dirt
        out *= 1 + np.minimum(smooth_noise(rng, out.shape, 6, 0.12), 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def relight(rng, img):
    out = img.astype(np.float32)
    out *= rng.uniform(0.55, 1.3) * rng.uniform(0.92, 1.08, 3)       # exposure and colour cast
    out = 127 + (out - 127) * rng.uniform(0.6, 1.15)                  # contrast
    h, w = img.shape[:2]
    ramp = np.linspace(-1, 1, w, dtype=np.float32)[None, :, None] * rng.uniform(-0.2, 0.2)
    ramp = ramp + np.linspace(-1, 1, h, dtype=np.float32)[:, None, None] * rng.uniform(-0.2, 0.2)
    out *= 1 + ramp                                                   # uneven lighting
    if rng.random() < 0.4:                                            # glare from a lamp
        glare = np.zeros((h, w), np.float32)
        cv2.ellipse(glare, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                    (int(rng.integers(w // 10, w // 2)), int(rng.integers(h // 12, h // 3))),
                    float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        glare = cv2.GaussianBlur(glare, (0, 0), max(w, h) / 12)
        out += glare[..., None] * rng.uniform(40, 170)
    return np.clip(out, 0, 255).astype(np.uint8)


def view_warp(rng, img, alpha, target_w):
    """Random viewpoint: rotate the flat front in 3D, project it, scale to target_w pixels."""
    h, w = alpha.shape
    yaw = np.radians(rng.triangular(-70, 0, 70))
    pitch, roll = np.radians(rng.triangular(-30, 0, 30)), np.radians(rng.normal(0, 3))
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    f = max(w, h) * rng.uniform(1.5, 5)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts = np.c_[src - [w / 2, h / 2], np.zeros(4)] @ (Rz @ Rx @ Ry).T
    dst = pts[:, :2] * (f / (f + pts[:, 2:3]))
    dst = (dst - dst.min(0)) * (target_w / np.ptp(dst[:, 0]))
    size = (max(2, int(np.ceil(dst[:, 0].max()))), max(2, int(np.ceil(dst[:, 1].max()))))
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    interp = cv2.INTER_AREA if target_w < w * 0.7 else cv2.INTER_LINEAR
    if interp == cv2.INTER_AREA:  # avoid aliasing when shrinking a lot
        k = w / target_w
        img, alpha = cv2.GaussianBlur(img, (0, 0), k / 2.5), cv2.GaussianBlur(alpha, (0, 0), k / 2.5)
    return cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR), cv2.warpPerspective(alpha, M, size)


# ---------------------------------------------------------------- backgrounds

def rand_color(rng, neutral=0.6):
    if rng.random() < neutral:  # cabinets and walls: light greys, whites, beiges
        return rng.integers(140, 250) + rng.integers(-12, 13, 3)
    return rng.integers(20, 240, 3)


def rand_text(rng):
    chars = "ABCDEFGHJKLMNPRSTUVXYZ0123456789-/. "
    return "".join(rng.choice(list(chars), rng.integers(2, 12)))


def draw_distractor(rng, canvas):
    """Things that sit on real panels but are not the product."""
    S = canvas.shape[0]
    x, y = int(rng.integers(0, S)), int(rng.integers(0, S))
    s = int(rng.integers(S // 40, S // 5))
    c = [int(v) for v in np.clip(rand_color(rng, 0.3), 0, 255)]
    kind = rng.integers(0, 8)
    if kind == 0:    # dark window / cut-out / display
        cv2.rectangle(canvas, (x, y), (x + s, y + int(s * rng.uniform(0.5, 1.5))), [int(rng.integers(0, 60))] * 3, -1)
    elif kind == 1:  # lamp or push button
        cv2.circle(canvas, (x, y), s // 4, c, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), s // 4, (60, 60, 60), max(1, s // 30), cv2.LINE_AA)
    elif kind == 2:  # label plate with text
        w = int(s * rng.uniform(1.5, 4))
        cv2.rectangle(canvas, (x, y), (x + w, y + s // 2), c, -1)
        cv2.putText(canvas, rand_text(rng), (x + 4, y + s // 3), int(rng.integers(0, 8)), s / 110,
                    [int(rng.integers(0, 90))] * 3, max(1, s // 60), cv2.LINE_AA)
    elif kind == 3:  # warning triangle / sticker
        pts = np.int32([[x, y - s // 2], [x - s // 2, y + s // 3], [x + s // 2, y + s // 3]])
        cv2.fillPoly(canvas, [pts], (0, int(rng.integers(180, 255)), 255), cv2.LINE_AA)
        cv2.polylines(canvas, [pts], True, (0, 0, 0), max(1, s // 25), cv2.LINE_AA)
    elif kind == 4:  # handle / rotary switch
        cv2.rectangle(canvas, (x, y), (x + s // 4, y + s), c, -1)
        cv2.circle(canvas, (x + s // 8, y + s // 8), s // 6, (40, 40, 40), -1, cv2.LINE_AA)
    elif kind == 5:  # ventilation grille
        for i in range(0, s, max(4, s // 12)):
            cv2.line(canvas, (x, y + i), (x + int(s * 1.2), y + i), [int(rng.integers(60, 170))] * 3, 2)
    elif kind == 6:  # other instrument: housing, dark display, a few keys, text
        w, h = int(s * rng.uniform(0.8, 1.8)), s
        cv2.rectangle(canvas, (x, y), (x + w, y + h), c, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), [int(rng.integers(40, 160))] * 3, max(1, s // 40))
        cv2.rectangle(canvas, (x + w // 6, y + h // 6), (x + w * 5 // 6, y + h // 2), [int(rng.integers(0, 120))] * 3, -1)
        for i in range(rng.integers(0, 6)):
            bx = x + w // 6 + i * w // 8
            cv2.rectangle(canvas, (bx, y + h * 2 // 3), (bx + w // 12, y + h * 5 // 6), [int(rng.integers(60, 220))] * 3, -1)
        cv2.putText(canvas, rand_text(rng)[:5], (x + 3, y + h // 8), int(rng.integers(0, 8)), s / 180, (30, 30, 30), 1, cv2.LINE_AA)
    else:            # tape / paper note
        w = int(s * rng.uniform(1, 3))
        cv2.rectangle(canvas, (x, y), (x + w, y + s // 3), c, -1)


def procedural_background(rng, S):
    bg = np.full((S, S, 3), rand_color(rng), np.float32)
    bg += np.linspace(-1, 1, S, dtype=np.float32)[None, :, None] * rng.uniform(-35, 35)
    bg += np.linspace(-1, 1, S, dtype=np.float32)[:, None, None] * rng.uniform(-35, 35)
    bg += smooth_noise(rng, bg.shape, int(rng.integers(3, 12)), rng.uniform(2, 14))
    canvas = np.clip(bg, 0, 255).astype(np.uint8)
    for _ in range(rng.integers(0, 5)):  # doors and panels in slightly different shades
        x1, y1 = int(rng.integers(-S // 4, S)), int(rng.integers(-S // 4, S))
        x2, y2 = x1 + int(rng.integers(S // 5, S)), y1 + int(rng.integers(S // 5, S))
        shade = [int(v) for v in np.clip(rand_color(rng, 0.8), 0, 255)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), shade, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), [int(rng.integers(60, 200))] * 3, int(rng.integers(1, 6)))
    for _ in range(rng.integers(0, 4)):  # seams
        p = int(rng.integers(0, S))
        a, b = ((p, 0), (p, S)) if rng.random() < 0.5 else ((0, p), (S, p))
        cv2.line(canvas, a, b, [int(rng.integers(50, 200))] * 3, int(rng.integers(1, 5)))
    if rng.random() < 0.3:               # perspective: the whole wall seen at an angle
        d = S * 0.35
        src = np.float32([[0, 0], [S, 0], [S, S], [0, S]])
        M = cv2.getPerspectiveTransform(src, (src + rng.uniform(-d, d, (4, 2))).astype(np.float32))
        canvas = cv2.warpPerspective(canvas, M, (S, S), borderMode=cv2.BORDER_REFLECT)
    return canvas


def photo_background(rng, path, S):
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    c = int(min(h, w) * rng.uniform(0.3, 1.0))
    x, y = rng.integers(0, w - c + 1), rng.integers(0, h - c + 1)
    img = cv2.resize(img[y:y + c, x:x + c], (S, S), interpolation=cv2.INTER_AREA)
    return img[:, ::-1].copy() if rng.random() < 0.5 else img


def make_background(rng, S, photos):
    bg = None
    if photos and rng.random() < 0.6:
        bg = photo_background(rng, photos[rng.integers(len(photos))], S)
    if bg is None:
        bg = procedural_background(rng, S)
    for _ in range(rng.integers(0, 9)):
        draw_distractor(rng, bg)
    return bg


# ---------------------------------------------------------------- composition

def paste(rng, canvas, obj, alpha, x, y):
    """Blend obj at (x, y), with a soft shadow. Returns the object's visible mask on the canvas."""
    S = canvas.shape[0]
    x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + obj.shape[1], S), min(y + obj.shape[0], S)
    mask = np.zeros((S, S), np.float32)
    if x2 <= x1 or y2 <= y1:
        return mask
    a = alpha[y1 - y:y2 - y, x1 - x:x2 - x]
    mask[y1:y2, x1:x2] = a
    blur = max(1.0, obj.shape[1] / 25)
    shift = np.float32([[1, 0, rng.normal(0, blur)], [0, 1, abs(rng.normal(0, blur * 2))]])
    shadow = cv2.GaussianBlur(cv2.warpAffine(mask, shift, (S, S)), (0, 0), blur) * rng.uniform(0.1, 0.5)
    canvas[:] = (canvas * (1 - shadow[..., None])).astype(np.uint8)
    soft = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.4, 1.2))[..., None]  # no hard cut-out edge
    region = canvas[y1:y2, x1:x2]
    region[:] = (obj[y1 - y:y2 - y, x1 - x:x2 - x] * soft + region * (1 - soft)).astype(np.uint8)
    return mask


def draw_occluder(rng, S):
    """Cable, strap or sheet in front of the panel. Returns (colour image, mask)."""
    layer, m = np.zeros((S, S, 3), np.uint8), np.zeros((S, S), np.uint8)
    c = [int(v) for v in np.clip(rand_color(rng, 0.5), 0, 255)]
    if rng.random() < 0.7:  # cable: smooth random curve
        t = np.linspace(0, 1, 60)[:, None]
        p = rng.uniform(-0.2 * S, 1.2 * S, (4, 2))
        curve = ((1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3]).astype(np.int32)
        th = int(rng.integers(S // 150, S // 25))
        cv2.polylines(layer, [curve], False, c, th, cv2.LINE_AA)
        cv2.polylines(m, [curve], False, 1, th)
    else:
        x, y = int(rng.integers(0, S)), int(rng.integers(0, S))
        w, h = int(rng.integers(S // 20, S // 3)), int(rng.integers(S // 20, S // 3))
        cv2.rectangle(layer, (x, y), (x + w, y + h), c, -1)
        cv2.rectangle(m, (x, y), (x + w, y + h), 1, -1)
    return layer, m


def camera_effects(rng, img):
    S = img.shape[0]
    out = img
    if rng.random() < 0.6:   # soft panorama stitching: lose detail, scale back up
        s = rng.uniform(0.35, 0.9)
        small = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (S, S), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.4:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 2.5))
    if rng.random() < 0.15:  # motion blur
        k = int(rng.integers(3, 12))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2] = 1 / k
        R = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        out = cv2.filter2D(out, -1, cv2.warpAffine(kernel, R, (k, k)))
    f = out.astype(np.float32) * rng.uniform(0.7, 1.25) * rng.uniform(0.93, 1.07, 3)
    f = 255 * (np.clip(f, 0, 255) / 255) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.6:
        f += rng.normal(0, rng.uniform(1, 12), f.shape)
    out = np.clip(f, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 96))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def make_sample(rng, S, products, photos):
    canvas = make_background(rng, S, photos)
    placed = []  # [class_id, visible mask, unoccluded area]
    n_obj = int(rng.choice([0, 1, 2, 3], p=[0.15, 0.5, 0.25, 0.1]))
    for _ in range(n_obj):
        cid, _, cutouts = products[rng.integers(len(products))]
        img, alpha, screen = cutouts[rng.integers(len(cutouts))]
        img = relight(rng, wear(rng, randomize_screen(rng, img, screen), alpha))
        target_w = float(np.exp(rng.uniform(np.log(0.03 * S), np.log(0.8 * S))))  # tiny to close-up
        obj, a = view_warp(rng, img, alpha, target_w)
        x = int(rng.uniform(-0.3 * obj.shape[1], S - 0.7 * obj.shape[1]))
        y = int(rng.uniform(-0.3 * obj.shape[0], S - 0.7 * obj.shape[0]))
        mask = paste(rng, canvas, obj, a, x, y)
        for p in placed:  # a later object hides the earlier ones
            p[1] = p[1] * (1 - mask)
        placed.append([cid, mask, float(a.sum())])

    for _ in range(rng.integers(0, 4)):
        layer, m = draw_occluder(rng, S)
        if any(mask.sum() > 0 and (mask * m).sum() > 0.35 * mask.sum() for _, mask, _ in placed):
            continue  # would hide too much of a product
        canvas[m > 0] = layer[m > 0]
        for p in placed:
            p[1] = p[1] * (1 - m)

    labels = []
    for cid, mask, total in placed:
        ys, xs = np.where(mask > 0.5)
        if len(xs) < 40 or mask.sum() < 0.3 * total:
            continue  # mostly outside the image or hidden: not a fair target
        x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        labels.append((cid, (x1 + x2) / 2 / S, (y1 + y2) / 2 / S, (x2 - x1) / S, (y2 - y1) / S))
    return camera_effects(rng, canvas), labels


# ---------------------------------------------------------------- dataset writing

_state = {}


def _init(size, seed):
    cv2.setNumThreads(1)
    photos = sorted(f for f in BACKGROUNDS.rglob("*") if f.suffix.lower() in IMG_EXT) if BACKGROUNDS.exists() else []
    _state.update(size=size, seed=seed, products=load_products(), photos=photos)


def _work(job):
    i, img_path, lbl_path = job
    rng = np.random.default_rng([_state["seed"], i])
    img, labels = make_sample(rng, _state["size"], _state["products"], _state["photos"])
    cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    Path(lbl_path).write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in labels))
    return len(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    out = Path(args.out)
    jobs = []
    for i in range(args.n):
        split = "val" if i < args.n * args.val else "train"
        for sub in ("images", "labels"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)
        jobs.append((i, str(out / "images" / split / f"{i:06d}.jpg"), str(out / "labels" / split / f"{i:06d}.txt")))

    _init(args.size, args.seed)
    names = {cid: name for cid, name, _ in _state["products"]}
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {cid}: {name}\n" for cid, name in names.items()))
    print(f"{len(names)} product(s): {list(names.values())}; {len(_state['photos'])} background photo(s)")

    with Pool(args.workers, initializer=_init, initargs=(args.size, args.seed)) as pool:
        total = 0
        for k, n in enumerate(pool.imap_unordered(_work, jobs, chunksize=8), 1):
            total += n
            if k % 500 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} images, {total} boxes", flush=True)


if __name__ == "__main__":
    main()
