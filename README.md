**Live demo:** https://sayma-code.github.io/Vaasa-hackathon-frontend/

This repository holds the algorithm. The website is in a separate repository: https://github.com/sayma-code/Vaasa-hackathon-frontend

## 1. Accuracy results

### Current detector

Trained on 54 real photos (33 scan photos of the site, 21 from the web) and 126 synthetic variations
of them. Measured with `check_metrics.py` on the 9 photos held out from training, which show 14 relays.
A detection counts at confidence 0.5 or higher and is correct when it overlaps a marked relay (IoU 0.5 or higher).

| Metric | Result |
|---|---|
| Accuracy | 85.7% |
| Precision | 100% |
| Recall | 85.7% |
| F1-score | 92.3% |
| ROC-AUC | 0.949 |

That is 12 correct detections, 0 false alarms and 2 missed relays. Both misses are small, distant relays
in one scan photo that shows five. All 4 held-out web photos of the older blue-key front were found.

The confidence level changes the result:

| Confidence | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| 0.25 | 75.0% | 85.7% | 85.7% | 85.7% |
| 0.40 | 80.0% | 92.3% | 85.7% | 88.9% |
| 0.50 | 85.7% | 100% | 85.7% | 92.3% |
| 0.60 | 71.4% | 100% | 71.4% | 83.3% |
| 0.70 | 64.3% | 100% | 64.3% | 78.3% |
| 0.80 | 28.6% | 100% | 28.6% | 44.4% |

How the metrics are defined for a detector that draws boxes:

- **Precision**: of the boxes it drew, the share that sit on a real relay.
- **Recall**: of the real relays, the share it found.
- **F1-score**: 2 × precision × recall ÷ (precision + recall).
- **Accuracy**: correct ÷ (correct + false alarms + missed). There is no count of "correctly found nothing" for boxes.
- **ROC-AUC**: how well the confidence score separates correct boxes from false ones, over the 22 boxes
  the detector proposed (13 correct, 9 false). 1.0 is perfect and 0.5 is chance.

Read these numbers with care. Nine photos is a small test: one relay more or less moves recall by
7 points. Five of the nine are scan photos of the same room the detector trained on, seen from other positions.

## 2. How it can be better


1. Collect More images for training. Currently images were doubled from actual images but 
it's possible to make more synthetic variations (10 to 20 per photo instead of 2) | More variety is what a detector trained on 54 photos lacks most | `make_photo_dataset.py --factor 10`, then retrain. But *synthetic created data result is hardly accurate*

2. Add more real photos of installed relays | Only 4 of the web photos show a relay in a panel | Each photo needs a box marked |

3. Use a larger model (YOLO11m) and train longer | A bigger network learns finer detail | About 3 times slower; small gain until 1 to 4 are done |

4.  It's also possible to use different models to and a large dataset then choose which one gives better result but there was no time. So it can be done in future.


## 3. How to run the whole thing

These commands are for Windows (Command Prompt or PowerShell), run from this `backend` folder.

### Quick start

With the set-up below done, this starts the full demo:

```
.venv\Scripts\python.exe server.py --open
```

### Set up once

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

Two inputs are not in the repository and go in this folder: the scan file `cloud_0.e57`, and a
`test_photos` folder with the training photos.

### The whole pipeline, start to finish

```
.venv\Scripts\python.exe extract_e57.py
.venv\Scripts\python.exe make_photo_dataset.py
.venv\Scripts\python.exe train.py
.venv\Scripts\python.exe check_metrics.py
.venv\Scripts\python.exe evaluate_scan.py
.venv\Scripts\python.exe locate_assets.py
.venv\Scripts\python.exe link_assets.py
.venv\Scripts\python.exe build_viewer_data.py
.venv\Scripts\python.exe server.py --open
```

| Command | What it does | Time |
|---|---|---|
| `extract_e57.py` | Unpacks the scan photos and camera positions. Needed once | A minute |
| `make_photo_dataset.py` | Builds the training set from `test_photos` | A minute |
| `train.py` | Trains the detector | About 6 minutes |
| `check_metrics.py` | Prints accuracy, precision, recall, F1-score and ROC-AUC, and saves them in `eval/metrics_val.txt` | Under a minute |
| `evaluate_scan.py` | Finds the relays in the scan photos | A few minutes |
| `locate_assets.py` | Turns the detections into 3D assets | A minute |
| `link_assets.py` | Attaches documents and asset details | Seconds |
| `build_viewer_data.py` | Writes the tags and scan into the website. Add `--skip-cloud --skip-pano` after the first run | A few minutes the first time |
| `server.py --open` | Starts the demo and opens the browser | Half a minute to load |

### Run the website only

Full demo (3D twin plus photo detector). Keep the window open while using it:

```
.venv\Scripts\python.exe server.py --open
```

Twin only, as it appears online. Then open http://localhost:8766:

```
.venv\Scripts\python.exe -m http.server 8766 --directory ..\frontend-full
```

Double-clicking `start_demo.bat` or `start_twin.bat` runs the same two commands. Press Ctrl+C in the
window, or close it, to stop.

### Retrain after adding photos

Put the new photos in `test_photos` and add their boxes to `photo_labels.json` (photos without boxes are
skipped). Stop the website first so the GPU is free:

```
.venv\Scripts\python.exe make_photo_dataset.py
.venv\Scripts\python.exe train.py
.venv\Scripts\python.exe check_metrics.py
```

`check_metrics.py` takes `--split train` to score the training photos instead, and `--conf 0.6` to change
the confidence a detection needs to count.


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

## Adding another product

Add a `products/<name>/product.json` with the product's name, the text printed on it and its documents,
put photos of it in `test_photos/`, mark them in `photo_labels.json`, and rerun the steps. The dataset
script currently writes one product class.

## Repository layout

| Path | Content |
|---|---|
| `*.py` | The pipeline scripts and the demo server |
| `photo_labels.json` | Hand-marked relay boxes for the web photos |
| `products/abb_615/product.json` | The product's name, expected text and documents |
| `runs/detector/weights/best.pt` | The current detector |
| `assets/` | The 7 located assets, their tags and close-ups |
| `eval/` | Detections on the scan and the score report (first detector), and the metrics report (current detector) |
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
