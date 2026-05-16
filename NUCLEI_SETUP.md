# Nuclei / cell counting — local setup

Tools used: **StarDist** (single-channel fluorescence nuclei), **Cellpose 4 / Cellpose-SAM** (generalist cell+nuclei), **InstanSeg** (multi-channel nuclei + cells).

The sandbox where I prototyped this couldn't reach `huggingface.co` / `cellpose.org`, so Cellpose was skipped there. On your local machine Cellpose-SAM will work and is usually the strongest single model for this task.

## 1. Install

```bash
git clone <this repo>
cd EECS486-Ethnicity-Identifier
git checkout claude/cell-nucleus-detection-model-2vQnP

python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-nuclei.txt
```

If you have an NVIDIA GPU, install `torch` from https://pytorch.org/get-started/locally/ **before** running the line above so you get the CUDA build. Apple Silicon users can skip — PyTorch on macOS uses MPS automatically.

First run will download ~700 MB of model weights into:

- `~/.keras/datasets/` — StarDist
- `~/.cellpose/models/` — Cellpose
- `~/.cache/instanseg/` — InstanSeg

## 2. Run

```bash
# Single nuclear-stain image:
python nuclei_count.py --red data/nuclei_00.jpg --gpu

# Two co-registered channels (nuclei + cytoplasm/membrane):
python nuclei_count.py --red data/nuclei_00.jpg --green data/nuclei_01.jpg \
    --pixel-size 0.325 --gpu

# Pick a single model (faster):
python nuclei_count.py --red data/nuclei_00.jpg --only stardist
```

Outputs go to `./results/` — boundaries + colour overlays per model.

## 3. Important tuning knobs

- **`--pixel-size`** (µm per pixel) — InstanSeg internally rescales the image so cells are at the size it was trained for. *Get this right.* It's usually printed in the microscope metadata or ImageJ's "Set Scale".
- **`--diameter`** (pixels) — Cellpose's expected object diameter. `None` lets the model estimate it; usually fine.
- **`--prob-thresh`** — StarDist sensitivity. Default 0.479; lower = more (smaller) detections.

If StarDist and Cellpose disagree by ≥ 2×, the `--pixel-size` / `--diameter` are usually the culprit, not the model.

## 4. What I observed on the demo images

| model | input | nuclei | cells | notes |
|-------|-------|--------|-------|-------|
| StarDist `2D_versatile_fluo` | red | 708 | — | default `prob_thresh`; stable 540–790 across sweeps |
| InstanSeg (1-channel) | red | 305 | 1130 | "cells" output is nonsense without a cytoplasm channel |
| InstanSeg (2-channel) | red+green | 225 | 1157 | `pixel_size` was guessed → likely wrong scale |
| Naive Otsu + CC | red | 305 | — | known to merge touching nuclei → lower bound |

Best single number for the demo image: **~700 nuclei** (StarDist). On your machine I'd run **Cellpose-SAM** as a third opinion — it's typically the strongest model for this kind of image and works zero-shot.
