"""InstanSeg fluorescence_nuclei_and_cells on both images.

Image 1 = the red nuclei picture (data/nuclei_00.jpg)
Image 2 = the green retinal-organoid picture (data/nuclei_01.jpg)
"""
import os, time, glob
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops

OUT_DIR = "data/nuclei_out"
os.makedirs(OUT_DIR, exist_ok=True)

from instanseg import InstanSeg
print("Loading InstanSeg fluorescence model...")
inst = InstanSeg("fluorescence_nuclei_and_cells", verbosity=0)

def process(path, tag, channel_index):
    img = np.array(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    print(f"\n=== {tag}  shape {img.shape} ===")
    # InstanSeg auto-handles channels; feed RGB but it'll detect the signal.
    t0 = time.time()
    # eval_small_image: returns (labels_tensor, image_tensor)
    labels, _img_t = inst.eval_small_image(img, pixel_size=0.5, target="all_outputs")
    dt = time.time() - t0
    # labels shape: [1, C, H, W] where C may be 1 (nuclei only) or 2 (nuclei + cells)
    arr = labels.cpu().numpy()
    print(f"raw labels shape {arr.shape}")
    if arr.ndim == 4:
        arr = arr[0]
    n_channels = arr.shape[0]
    chan_names = ["nuclei", "cells"][:n_channels]
    counts = {}
    for ci, name in enumerate(chan_names):
        counts[name] = int(arr[ci].max())
    print(f"counts: {counts}   ({dt:.1f}s)")

    # Visualise: original | nuclei boundaries | cells boundaries (if available) | colour overlay
    cols = 2 + n_channels
    fig, axes = plt.subplots(1, cols, figsize=(7 * cols, 7))
    axes[0].imshow(img); axes[0].set_title(f"Input ({tag})"); axes[0].axis("off")

    for ci, name in enumerate(chan_names):
        lbl = arr[ci].astype(np.int32)
        bnd = find_boundaries(lbl, mode="outer")
        viz = img.copy()
        viz[bnd] = [255, 255, 0] if name == "nuclei" else [0, 255, 255]
        axes[1 + ci].imshow(viz)
        axes[1 + ci].set_title(f"{name} boundaries  N={counts[name]}")
        axes[1 + ci].axis("off")

    # final panel: colour overlay of cells (or nuclei if only one)
    lbl_final = arr[-1].astype(np.int32)
    overlay = label2rgb(lbl_final, image=img, bg_label=0, alpha=0.45, image_alpha=1.0, kind="overlay")
    axes[-1].imshow(overlay)
    axes[-1].set_title(f"colour overlay ({chan_names[-1]})  N={counts[chan_names[-1]]}")
    axes[-1].axis("off")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"instanseg_{tag}.png")
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)
    return counts, dt

# Use the same red nuclei jpg
process("data/nuclei_00.jpg", "image1_red_nuclei", channel_index=0)
# Try to find the green organoid jpg (may have been extracted)
# Re-extract to be sure (the new green image was sent later)
import json, base64, pathlib
src = "/root/.claude/projects/-home-user-EECS486-Ethnicity-Identifier/93bb37a0-3922-4ed2-b447-20a5a6892e3a.jsonl"
out_dir = pathlib.Path("data")
found = []
with open(src) as f:
    for line in f:
        try: rec = json.loads(line)
        except: continue
        def walk(o, acc):
            if isinstance(o, dict):
                if o.get("type") == "image" and isinstance(o.get("source"), dict):
                    s = o["source"]
                    if s.get("type") == "base64":
                        data = base64.b64decode(s["data"])
                        acc.append(data)
                for v in o.values(): walk(v, acc)
            elif isinstance(o, list):
                for v in o: walk(v, acc)
        walk(rec, found)
print(f"images embedded in session: {len(found)}")
for i, d in enumerate(found):
    p = out_dir / f"nuclei_{i:02d}.jpg"
    p.write_bytes(d)
    print("  wrote", p, len(d), "bytes")

# Second image is the green organoid
if len(found) >= 2:
    process("data/nuclei_01.jpg", "image2_green_organoid", channel_index=1)
else:
    print("WARN: green image not found in session yet")
