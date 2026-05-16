"""Run StarDist, Cellpose, and/or InstanSeg on fluorescence microscopy images
and report nucleus / cell counts plus overlay visualisations.

Designed for local use (laptop / desktop with internet + optional GPU). Inside
restricted sandboxes that block huggingface.co you can pass --skip-cellpose
to fall back to StarDist + InstanSeg only.

Examples
--------
# Single nuclei channel image (red), default models:
python nuclei_count.py --red path/to/red.tif

# Two-channel input (nuclei + cytoplasm), known pixel size:
python nuclei_count.py --red red.tif --green green.tif --pixel-size 0.325

# Only StarDist (fastest, no PyTorch needed):
python nuclei_count.py --red red.tif --only stardist
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def percentile_norm(x, lo=1.0, hi=99.8):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / max(b - a, 1e-6), 0, 1)


def load_signal_channel(path: str | Path) -> np.ndarray:
    """Read an image, return a float32 H×W intensity map.

    For RGB / multi-channel inputs we keep the brightest channel.
    For 16-bit TIFFs we preserve dynamic range.
    """
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        # pick dominant channel by mean intensity
        means = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
        return arr[..., int(np.argmax(means))].astype(np.float32)
    raise ValueError(f"Unsupported image shape {arr.shape}")


def save_overlay(rgb: np.ndarray, labels: np.ndarray, out_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.color import label2rgb
    from skimage.segmentation import find_boundaries

    bnd = find_boundaries(labels, mode="outer")
    bound_img = rgb.copy()
    bound_img[bnd] = [255, 255, 0]
    overlay = label2rgb(
        labels, image=rgb, bg_label=0, alpha=0.45, image_alpha=1.0, kind="overlay"
    )

    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    ax[0].imshow(rgb); ax[0].set_title("Input"); ax[0].axis("off")
    ax[1].imshow(bound_img); ax[1].set_title(f"{title} boundaries  N={int(labels.max())}"); ax[1].axis("off")
    ax[2].imshow(overlay); ax[2].set_title(f"{title} overlay  N={int(labels.max())}"); ax[2].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ------------------------- model runners -------------------------

def run_stardist(red_norm: np.ndarray, prob_thresh: float | None) -> np.ndarray:
    from stardist.models import StarDist2D
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    labels, _ = model.predict_instances(red_norm, prob_thresh=prob_thresh)
    return labels


def run_cellpose(red_norm: np.ndarray, green_norm: np.ndarray | None,
                 diameter: float | None, gpu: bool) -> np.ndarray:
    """Use Cellpose 4 (Cellpose-SAM) if available, otherwise fall back to
    Cellpose 3 nuclei/cyto3.
    """
    from cellpose import models  # weights auto-download from HuggingFace
    try:
        # Cellpose 4.x: single generalist model
        model = models.CellposeModel(gpu=gpu)
        if green_norm is not None:
            img = np.stack([red_norm, green_norm], axis=-1)  # H,W,2
        else:
            img = red_norm
        masks, _flows, _styles = model.eval(img, diameter=diameter)
        return masks
    except AttributeError:
        # Cellpose 3.x: pick 'nuclei' or 'cyto3'
        model_type = "cyto3" if green_norm is not None else "nuclei"
        model = models.Cellpose(gpu=gpu, model_type=model_type)
        if green_norm is not None:
            img = np.stack([red_norm, green_norm], axis=-1)
            channels = [1, 2]  # cyto in 1, nuclei in 2 (Cellpose convention)
        else:
            img = red_norm
            channels = [0, 0]
        masks, _flows, _styles, _diams = model.eval(img, diameter=diameter, channels=channels)
        return masks


def run_instanseg(red: np.ndarray, green: np.ndarray | None, pixel_size: float):
    """Returns (nuclei_labels, cells_labels). cells_labels is None if model
    only produced one output."""
    from instanseg import InstanSeg
    model_name = "fluorescence_nuclei_and_cells" if green is not None else "fluorescence_nuclei_and_cells"
    inst = InstanSeg(model_name, verbosity=0)
    if green is not None:
        chw = np.stack([red, green], axis=0).astype(np.float32)  # C,H,W
    else:
        chw = red.astype(np.float32)
    labels, _ = inst.eval_small_image(chw, pixel_size=pixel_size, target="all_outputs")
    arr = labels.cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    nuc = arr[0].astype(np.int32)
    cell = arr[1].astype(np.int32) if arr.shape[0] >= 2 else None
    return nuc, cell


# ----------------------------- main -----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--red", required=True, help="Nuclear-stain image (any common format)")
    p.add_argument("--green", default=None, help="Optional cytoplasm/membrane image (same FOV)")
    p.add_argument("--out-dir", default="results", help="Where to write overlays")
    p.add_argument("--pixel-size", type=float, default=0.5, help="µm per pixel (for InstanSeg)")
    p.add_argument("--diameter", type=float, default=None, help="Cellpose diameter (px); None=auto")
    p.add_argument("--prob-thresh", type=float, default=None, help="StarDist prob threshold")
    p.add_argument("--gpu", action="store_true", help="Use GPU if available")
    p.add_argument("--only", choices=["stardist", "cellpose", "instanseg"], default=None)
    p.add_argument("--skip-cellpose", action="store_true",
                   help="Skip Cellpose (needs HuggingFace access)")
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"=== loading images ===")
    red_raw = load_signal_channel(args.red)
    red_n = percentile_norm(red_raw)
    rgb_red = np.array(Image.open(args.red).convert("RGB"))
    print(f"red    : {red_raw.shape}  range={red_raw.min():.0f}..{red_raw.max():.0f}")

    green_n = None
    rgb_green = None
    if args.green:
        green_raw = load_signal_channel(args.green)
        green_n = percentile_norm(green_raw)
        rgb_green = np.array(Image.open(args.green).convert("RGB"))
        print(f"green  : {green_raw.shape}  range={green_raw.min():.0f}..{green_raw.max():.0f}")

    summary = {}

    def gate(name):
        return args.only in (None, name)

    if gate("stardist"):
        print("\n=== StarDist 2D_versatile_fluo ===")
        t0 = time.time()
        sd_labels = run_stardist(red_n, args.prob_thresh)
        dt = time.time() - t0
        n = int(sd_labels.max())
        summary["stardist"] = n
        print(f"  nuclei={n}  ({dt:.1f}s)")
        save_overlay(rgb_red, sd_labels, out / "stardist.png", "StarDist")

    if gate("cellpose") and not args.skip_cellpose:
        print("\n=== Cellpose ===")
        try:
            t0 = time.time()
            cp_labels = run_cellpose(red_n, green_n, args.diameter, args.gpu)
            dt = time.time() - t0
            n = int(cp_labels.max())
            summary["cellpose"] = n
            print(f"  cells/nuclei={n}  ({dt:.1f}s)")
            save_overlay(rgb_green if rgb_green is not None else rgb_red,
                         cp_labels, out / "cellpose.png", "Cellpose")
        except Exception as e:
            print(f"  Cellpose failed: {e}")

    if gate("instanseg"):
        print("\n=== InstanSeg fluorescence_nuclei_and_cells ===")
        t0 = time.time()
        nuc, cell = run_instanseg(red_n, green_n, args.pixel_size)
        dt = time.time() - t0
        n_nuc = int(nuc.max()); n_cell = int(cell.max()) if cell is not None else None
        summary["instanseg_nuclei"] = n_nuc
        summary["instanseg_cells"] = n_cell
        print(f"  nuclei={n_nuc}  cells={n_cell}  ({dt:.1f}s)")
        save_overlay(rgb_red, nuc, out / "instanseg_nuclei.png", "InstanSeg nuclei")
        if cell is not None:
            save_overlay(rgb_green if rgb_green is not None else rgb_red,
                         cell, out / "instanseg_cells.png", "InstanSeg cells")

    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k:25s} {v}")


if __name__ == "__main__":
    main()
