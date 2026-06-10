"""StarDist + InstanSeg analysis of the CRX-GFP retinal-organoid samples.

For each sample, computes:
  - StarDist nuclei count on c1 (red nuclear stain)
  - InstanSeg nuclei+cells on (c1, c2) two-channel input
  - CRX-GFP+ fraction = (#nuclei with mean green signal > threshold) / #nuclei
"""
import os, time, glob
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops

OUT = "data/crx_gfp/_out"
os.makedirs(OUT, exist_ok=True)

def norm(x, lo=1, hi=99.8):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / max(b - a, 1e-6), 0, 1)

from stardist.models import StarDist2D
from instanseg import InstanSeg
print("loading models...")
sd = StarDist2D.from_pretrained("2D_versatile_fluo")
inst = InstanSeg("fluorescence_nuclei_and_cells", verbosity=0)

samples = sorted(set(os.path.dirname(p) for p in glob.glob("data/crx_gfp/19*/*c1.tif")))
print("samples:", samples)
summary = []

for sample_dir in samples:
    name = os.path.basename(sample_dir)
    print(f"\n=== {name} ===")
    c1_path = os.path.join(sample_dir, f"{name}_c1.tif")  # nuclei
    c2_path = os.path.join(sample_dir, f"{name}_c2.tif")  # CRX-GFP
    merge_path = os.path.join(sample_dir, f"{name}_c1-2.tif")

    c1 = tifffile.imread(c1_path)[..., 0].astype(np.float32)  # red channel only
    c2 = tifffile.imread(c2_path)[..., 1].astype(np.float32)  # green channel only
    merge = tifffile.imread(merge_path)
    print(f"  shape c1={c1.shape}  c2={c2.shape}")

    c1_n = norm(c1); c2_n = norm(c2)

    # ---- StarDist (red channel only) ----
    t0 = time.time()
    sd_labels, _ = sd.predict_instances(c1_n)
    sd_t = time.time() - t0
    sd_count = int(sd_labels.max())
    print(f"  StarDist: {sd_count} nuclei  ({sd_t:.1f}s)")

    # ---- CRX-GFP+ fraction using StarDist masks ----
    # For each StarDist nucleus mask, compute mean intensity in c2.
    # A nucleus is CRX-GFP+ if its mean exceeds an Otsu-like threshold on per-nucleus means.
    from skimage.filters import threshold_otsu
    nuc_means = np.array([c2_n[sd_labels == lid].mean() for lid in range(1, sd_count + 1)])
    if len(nuc_means) > 5:
        try:
            t = max(threshold_otsu(nuc_means), 0.05)  # require at least 5% of dynamic range
        except Exception:
            t = 0.10
    else:
        t = 0.10
    crx_pos = int((nuc_means > t).sum())
    print(f"  CRX-GFP+ threshold={t:.3f}  positive={crx_pos}/{sd_count} ({100*crx_pos/sd_count:.1f}%)")

    # ---- InstanSeg with both channels ----
    chw = np.stack([c1, c2], axis=0).astype(np.float32)
    t0 = time.time()
    is_labels, _ = inst.eval_small_image(chw, pixel_size=0.5, target="all_outputs")
    is_t = time.time() - t0
    arr = is_labels.cpu().numpy()
    if arr.ndim == 4: arr = arr[0]
    is_nuc = arr[0].astype(np.int32)
    is_cell = arr[1].astype(np.int32) if arr.shape[0] >= 2 else None
    is_nuc_n = int(is_nuc.max()); is_cell_n = int(is_cell.max()) if is_cell is not None else None
    print(f"  InstanSeg: nuclei={is_nuc_n} cells={is_cell_n}  ({is_t:.1f}s)")

    # ---- visualisation ----
    fig, ax = plt.subplots(2, 3, figsize=(24, 16))
    ax[0,0].imshow(merge); ax[0,0].set_title(f"{name}  merged (red=nuc, green=CRX-GFP)"); ax[0,0].axis("off")

    # StarDist boundaries on merge, colour CRX+/CRX-
    bnd = find_boundaries(sd_labels, mode="outer")
    pos_mask = np.zeros_like(sd_labels, dtype=bool)
    for lid in range(1, sd_count + 1):
        if nuc_means[lid - 1] > t:
            pos_mask[sd_labels == lid] = True
    viz = merge.copy()
    # CRX+ -> yellow border, CRX- -> magenta border
    pos_bnd = bnd & find_boundaries(pos_mask.astype(np.int32) + 1, mode="outer")
    viz[bnd] = [255, 0, 255]      # magenta = all
    viz[bnd & pos_mask] = [255, 255, 0]  # yellow = CRX+
    ax[0,1].imshow(viz); ax[0,1].set_title(f"StarDist  total={sd_count}  CRX+={crx_pos} ({100*crx_pos/sd_count:.0f}%)"); ax[0,1].axis("off")

    sd_overlay = label2rgb(sd_labels, image=merge, bg_label=0, alpha=0.45, image_alpha=1.0, kind="overlay")
    ax[0,2].imshow(sd_overlay); ax[0,2].set_title("StarDist colour labels"); ax[0,2].axis("off")

    # InstanSeg boundaries
    viz2 = merge.copy()
    viz2[find_boundaries(is_nuc, mode="outer")] = [255, 255, 0]
    ax[1,0].imshow(viz2); ax[1,0].set_title(f"InstanSeg nuclei  N={is_nuc_n}"); ax[1,0].axis("off")

    if is_cell is not None:
        viz3 = merge.copy()
        viz3[find_boundaries(is_cell, mode="outer")] = [0, 255, 255]
        ax[1,1].imshow(viz3); ax[1,1].set_title(f"InstanSeg cells  N={is_cell_n}"); ax[1,1].axis("off")
        cell_overlay = label2rgb(is_cell, image=merge, bg_label=0, alpha=0.45, image_alpha=1.0, kind="overlay")
        ax[1,2].imshow(cell_overlay); ax[1,2].set_title("InstanSeg cells colour"); ax[1,2].axis("off")

    # CRX-GFP+ histogram
    plt.tight_layout()
    out_path = f"{OUT}/{name}_summary.png"
    plt.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")

    # per-nucleus histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(nuc_means, bins=60, color="green", alpha=0.7)
    ax.axvline(t, color="red", linestyle="--", label=f"threshold={t:.3f}")
    ax.set_xlabel("Per-nucleus mean GFP intensity (normalised)")
    ax.set_ylabel("# nuclei")
    ax.set_title(f"{name}  CRX-GFP intensity distribution  (N={sd_count})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}_crx_hist.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    summary.append({
        "sample": name,
        "stardist_nuclei": sd_count,
        "crx_pos": crx_pos,
        "crx_frac": crx_pos / max(sd_count, 1),
        "instanseg_nuclei": is_nuc_n,
        "instanseg_cells": is_cell_n,
        "stardist_time_s": round(sd_t, 1),
        "instanseg_time_s": round(is_t, 1),
    })

print("\n========== SUMMARY ==========")
print(f"{'sample':<25} {'StarDist N':>10} {'CRX+':>6} {'CRX+%':>7} {'InstSeg nuc':>12} {'InstSeg cell':>13}")
for s in summary:
    print(f"{s['sample']:<25} {s['stardist_nuclei']:>10} {s['crx_pos']:>6} "
          f"{100*s['crx_frac']:>6.1f}% {s['instanseg_nuclei']:>12} {s['instanseg_cells']:>13}")

# also write CSV
import csv
with open(f"{OUT}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    for s in summary: w.writerow(s)
print(f"\nwrote {OUT}/summary.csv")
