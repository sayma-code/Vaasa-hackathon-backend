# Automatic product tagging in a digital twin: algorithm

Hackathon solution for "Product tagging in VEO360 digital twin using object recognition".
It finds every ABB Relion 615 relay in a Matterport scan, places a tag on it in 3D, and links the
tag to the product's documentation, with no manual tagging in the twin.

**Live demo:** https://sayma-code.github.io/Vaasa-hackathon-frontend/

This repository holds the algorithm. The website is in a separate repository: https://github.com/sayma-code/Vaasa-hackathon-frontend

## How it works

### Teaching the detector

1. **Photos** (`test_photos/`, not in this repository): photos of the relay, both scan photos of the
   site and photos from the web.
2. **Labels**: each relay's front plate is marked with a box. Scan photos take their boxes from
   `ground_truth/relays.json`; the other photos from `photo_labels.json`.
3. **Dataset** (`make_photo_dataset.py`): 85% of the photos are used for training and 15% are held out
   for validation. For every photo, two synthetic variations are made from the training photos: a
   zoomed crop around a relay with a slight change of viewpoint and random lighting, colour, blur and noise.
4. **Training** (`train.py`): a YOLO11s detector is fine-tuned on that dataset.

### From the detector to tags in the twin

5. **Scan photos** (`extract_e57.py`): the 108 photos and their camera positions are unpacked from the scan file.
6. **Detection** (`detector.py`, `evaluate_scan.py`): each 4096 px scan photo is searched in overlapping
   1024 px tiles at three zoom levels, and the result is scored against the ground truth.
7. **2D to 3D** (`locate_assets.py`): each detection is lifted to a 3D point with the point cloud, sightings
   within 15 cm are merged into one asset, and assets seen from fewer than 5 scan positions are dropped.
8. **Linking** (`link_assets.py`): each asset gets its product documents and its asset-register entry.
9. **Website data** (`build_viewer_data.py`): the tags and the scan are written into the frontend's `data` folder.

`server.py` runs the local demo: it serves the website, detects the product in submitted photos, reads
the text on each detection ("615", "ABB") as a second check, and serves the product documents.

## Results

Two detectors have been trained.

**First detector: synthetic images only.** Trained on 4,000 images generated from the organisers'
reference renders, pasted onto outside backgrounds; it never saw the site. The tags currently shown in
the twin come from this detector.

| | |
|---|---|
| Relays in the room | 7 |
| Relays found and tagged | 7 |
| Wrong tags | 0 |
| Tag position error | 0.3 to 3.2 cm |
| Clear views detected (confidence 0.8) | 50 of 55 |

It recognised only the front panel shown in the reference renders; a photo of the older blue-key
front of the same relay family scored 12%. The rule that a tag must be seen from at least 5 scan
positions was chosen after looking at this scan.

**Current detector: real photos plus variations.** Trained on 54 photos (33 scan photos of the site,
21 from the web) and 126 synthetic variations of them. `runs/detector/weights/best.pt` is this detector.

| | |
|---|---|
| Held-out validation photos | 9, showing 14 relays |
| Relays found on them | 85% |
| Correct among its detections | 100% |
| Held-out web photos of the blue-key front | 4 of 4 found, at 57% to 78% confidence |

Steps 6 to 9 have not been re-run with the current detector. Its score on the scan would also not be
an independent test, because most of the scan photos that show relays are in its training set.

## Adding another product

Add a `products/<name>/product.json` with the product's name, the text printed on it and its documents,
put photos of it in `test_photos/`, mark them in `photo_labels.json`, and rerun the steps. The dataset
script currently writes one product class.

## Running it

```
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r requirements.txt
```

The demo server and the website data step use the website, so place the two repositories side by side:

```
project/
  frontend-full/   the frontend repository
  backend/         this repository
```

To rebuild from the scan, put `cloud_0.e57` and a `test_photos` folder here (neither is in the repository):

```
python extract_e57.py            # photos and camera poses from cloud_0.e57
python make_photo_dataset.py     # training set from test_photos
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
| `photo_labels.json` | Hand-marked relay boxes for the web photos |
| `products/abb_615/product.json` | The product's name, expected text and documents |
| `runs/detector/weights/best.pt` | The current detector |
| `assets/` | The 7 located assets, their tags and close-ups |
| `eval/` | Detections on the scan and the score report, from the first detector |
| `ground_truth/` | Hand-checked relay positions in the scan |
| `asset_register.csv` | Panel names (read from the door plates) and demo maintenance entries |

## Known limits

- The current detector is trained on few photos and is less confident than the first one.
- It was trained on real photos, including the site's own scan photos. The challenge brief asks for
  training on synthetic data generated from the organisers' reference image, which is what the first
  detector did.
- The maintenance entries are demo data.
- Asset-register entries are matched to assets by position, and those positions come from the ground truth.

## Sources

- Scan and reference images: provided by the hackathon organisers.
- Web photos: collected from an image search for local training and testing; they belong to their
  publishers and are not in this repository.
- Documentation: public ABB 615 series documents, shown from ABB's library.
- Libraries: Ultralytics YOLO, PyTorch, EasyOCR, pye57, FastAPI.
