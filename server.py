"""Web server for the demo: the 3D viewer plus a page where anyone can submit a photo
and have the trained detector look for products in it.

Usage: python server.py [--open] [--port 8765] [--host 127.0.0.1] [--site ../frontend-full]
Use --host 0.0.0.0 to let other devices on the same network open it.
Photos are processed in memory and never saved.
"""
import argparse
import json
import socket
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from detector import detect, scales_for

ROOT = Path(__file__).resolve().parent
DOC_CACHE = ROOT / "documents_cache"  # local copies of the product documents, so they open instantly and offline
MAX_BYTES = 25 * 1024 * 1024
MAX_SIDE = 6000   # larger photos are scaled down first
OCR_MIN_SIDE = 60  # smaller detections have no readable text

app = FastAPI(title="Product detector")
state = {"model": None, "ocr": None, "products": {}}
gpu_lock = threading.Lock()  # one photo at a time on the GPU


@app.middleware("http")
async def no_stale_pages(request, call_next):
    """Make browsers re-check pages and scripts, so an edited page is never mixed with an old script."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"
    return response


def load_products():
    """Product name -> its settings (documents, text expected on the device)."""
    out = {}
    for f in (ROOT / "products").glob("*/product.json"):
        cfg = json.loads(f.read_text(encoding="utf-8"))
        out[cfg["name"]] = cfg
    return out


def read_text(crop, keywords):
    """Which of the product's expected words can be read on a detection. None if OCR is unavailable."""
    if state["ocr"] is None:
        try:
            import easyocr
            state["ocr"] = easyocr.Reader(["en"], gpu=True, verbose=False)
        except Exception:
            state["ocr"] = False
    if not state["ocr"]:
        return None
    k = 700 / max(crop.shape[:2])  # a consistent size reads best
    crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
    words = " ".join(t.upper() for _, t, c in state["ocr"].readtext(crop) if c >= 0.3)
    return [w for w in keywords if w.upper() in words]


def find_document(doc_id):
    for product in state["products"].values():
        for doc in product.get("documents", []):
            if doc.get("doc_id") == doc_id:
                return doc
    return None


doc_locks = {}


def cached_document(doc):
    """Path of the local copy of a document, downloading it from its source the first time."""
    path = DOC_CACHE / f"{doc['doc_id']}.pdf"
    with doc_locks.setdefault(doc["doc_id"], threading.Lock()):
        if not path.exists():
            DOC_CACHE.mkdir(exist_ok=True)
            part = path.with_suffix(".part")
            with requests.get(doc["url"], stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"}) as r:
                r.raise_for_status()
                with open(part, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            with open(part, "rb") as f:
                is_pdf = f.read(5) == b"%PDF-"
            if not is_pdf:
                part.unlink()
                raise ValueError("the source did not return a PDF")
            part.replace(path)
    return path


def prefetch_documents():
    """Fetch every product document in the background so the first click does not wait."""
    for product in state["products"].values():
        for doc in product.get("documents", []):
            try:
                cached_document(doc)
            except Exception as e:
                print(f"Could not fetch {doc.get('title')}: {e}", flush=True)
    print("All product documents are available offline.", flush=True)


@app.get("/api/document/{doc_id}")
def document(doc_id: str):
    doc = find_document(doc_id)
    if doc is None:
        raise HTTPException(404, "Unknown document.")
    try:
        path = cached_document(doc)
    except Exception:
        raise HTTPException(502, "The document could not be fetched from its source. Check the internet connection.")
    return FileResponse(path, media_type="application/pdf", content_disposition_type="inline",
                        filename=f"{doc['title']}.pdf")


@app.get("/api/info")
def info():
    return {"products": [{"name": p["name"], "documents": p.get("documents", [])} for p in state["products"].values()]}


@app.post("/api/detect")
def detect_photo(file: UploadFile = File(...)):
    raw = file.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "The photo is larger than 25 MB.")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "This file could not be read as a photo. Use a JPG, PNG or WebP image.")
    h, w = img.shape[:2]
    shrink = min(1.0, MAX_SIDE / max(h, w))
    work = img if shrink == 1 else cv2.resize(img, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)

    started = time.time()
    names = state["model"].names
    out = []
    with gpu_lock:
        rows = detect(state["model"], work, scales=scales_for(work))
        for x1, y1, x2, y2, conf, cls in sorted(rows.tolist(), key=lambda r: -r[4]):
            product = state["products"].get(names[int(cls)], {"name": names[int(cls)]})
            text = None
            if conf >= 0.25 and min(x2 - x1, y2 - y1) >= OCR_MIN_SIDE and product.get("ocr_keywords"):
                crop = work[int(max(y1, 0)):int(y2), int(max(x1, 0)):int(x2)]
                text = read_text(crop, product["ocr_keywords"])
            out.append({
                "box": [round(v / shrink) for v in (x1, y1, x2, y2)],
                "confidence": round(conf, 3),
                "product": product["name"],
                "text_expected": product.get("ocr_keywords", []),
                "text_found": text,  # None: too small to read, or OCR not available
                "documents": product.get("documents", []),
            })
    return {"width": w, "height": h, "seconds": round(time.time() - started, 2), "detections": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--weights", default=str(ROOT / "runs" / "detector" / "weights" / "best.pt"))
    ap.add_argument("--site", default=str(ROOT.parent / "frontend-full"), help="folder of the website to serve")
    ap.add_argument("--open", action="store_true", help="open the photo detector in the browser when ready")
    args = ap.parse_args()
    if not (Path(args.site) / "index.html").exists():
        raise SystemExit(f"Website not found in {args.site}. Put the frontend-full folder next to the backend "
                         "folder, or pass its location with --site.")

    print("Loading the detector, this takes about half a minute...", flush=True)

    state["products"] = load_products()
    state["model"] = YOLO(args.weights)
    state["model"].predict(np.zeros((1024, 1024, 3), np.uint8), imgsz=1024, verbose=False)  # warm up
    read_text(np.full((200, 300, 3), 255, np.uint8), [])  # load the text reader now, not on the first photo
    threading.Thread(target=prefetch_documents, daemon=True).start()
    app.mount("/", StaticFiles(directory=args.site, html=True), name="site")
    url = f"http://localhost:{args.port}/detect.html"

    def ready():
        """Announce the address (and open the browser) once the server accepts connections."""
        while True:
            try:
                socket.create_connection(("127.0.0.1", args.port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.3)
        print(f"Ready. Photo detector: {url}   3D twin: http://localhost:{args.port}/", flush=True)
        if args.open:
            webbrowser.open(url)

    threading.Thread(target=ready, daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
