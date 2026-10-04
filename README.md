# Automatic product tagging in a digital twin: algorithm

Hackathon solution for "Product tagging in VEO360 digital twin using object recognition".
From one set of product reference images, it finds every matching device in a Matterport scan,
places a tag on it in 3D, and links the tag to the product's documentation, with no manual tagging.

**Live demo:** https://YOUR-USERNAME.github.io/YOUR-FRONTEND-REPO/

This repository holds the algorithm. The website is in a separate repository: YOUR-FRONTEND-REPO-ADDRESS

## Results on the provided scan

| | |
|---|---|
| Labelled site photos used for training | 0 |
| Relays in the room | 7 |
| Relays found and tagged | 7 |
| Wrong tags | 0 |
| Tag position error | 0.3 to 3.2 cm |
| Clear views detected (confidence 0.8) | 50 of 55 |

The scan photos were used only for testing. One setting (a tag must be seen from at least
5 scan positions) was chosen after looking at this scan.

## How it works

The website the last step feeds is in the separate frontend repository.

1. **Synthetic data** (`make_synthetic.py`): the product is cut out of its reference images and pasted onto
   4,000 backgrounds with random viewpoint, lighting, blur, noise, occlusion, label wear and screen state.
   Backgrounds are computer-drawn or freely licensed photos; none come from the scanned site.
2. **Training** (`train.py`): a YOLO11s detector is trained on the synthetic images only.
3. **Detection on the scan** (`evaluate_scan.py`, `detector.py`): each 4096 px scan photo is searched in
   overlapping tiles at three zoom levels.
4. **2D to 3D** (`locate_assets.py`): each detection is lifted to a 3D point with the point cloud, sightings
   of the same spot are merged, and spots seen from too few scan positions are dropped.
5. **Linking** (`link_assets.py`): each asset gets its product documents and its asset-register entry.
6. **Digital twin** (frontend repository): a 3D overview and a walk-inside view with clickable tags that open the
   asset details and documentation.

## Adding another product

Create `products/<name>/` with a `product.json` and a `references/` folder of images on a white
background, then rerun steps 1 to 6. No code changes are needed.

## Running it

```
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r requirements.txt
```

The demo server and the viewer data step use the website, so place the two repositories side by side:

```
project/
  frontend-full/   the frontend repository
  backend/         this repository
```

To rebuild everything from the scan, put `cloud_0.e57` in this folder (it is not in the repository):

```
python extract_e57.py            # photos and camera poses from cloud_0.e57
python download_backgrounds.py   # outside background photos
python make_synthetic.py         # synthetic training set
python train.py                  # train the detector
python evaluate_scan.py          # detect in the scan photos and score
python locate_assets.py          # 3D assets
python link_assets.py            # tags with documents
python build_viewer_data.py      # data for the 3D viewer, written to ../frontend-full/data
python server.py --open          # local demo, including the photo detector page
```

`start_demo.bat` starts the full local demo. `start_twin.bat` runs the website without the detector,
as it appears online.

## Repository layout

| Path | Content |
|---|---|
| `*.py` | The pipeline scripts and the demo server |
| `products/abb_615/` | Reference images, expected text and documents of the product |
| `runs/detector/weights/best.pt` | The trained detector |
| `assets/` | The 7 located assets, their tags and close-ups |
| `eval/` | Detections on the scan and the score report |
| `ground_truth/` | Hand-checked relay positions used for scoring |
| `asset_register.csv` | Panel names (read from the door plates) and demo maintenance entries |

## Known limits

- The detector knows the front panel shown in the reference images. Other front panels of the same
  product family score low until their reference images are added.
- The maintenance entries are demo data.
- Asset-register entries are matched to assets by position.

## Sources

- Scan and reference images: provided by the hackathon organisers.
- Background photos: Wikimedia Commons, see `backgrounds/sources.csv` for authors and licences.
- Documentation: public ABB 615 series documents, shown from ABB's library.
- Libraries: Ultralytics YOLO, PyTorch, three.js, EasyOCR, pye57, FastAPI.
"# Vaasa-hackathon-backend" 
