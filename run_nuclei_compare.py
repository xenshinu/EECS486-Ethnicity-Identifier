"""Run StarDist (2D_versatile_fluo) and Cellpose-SAM on the nuclei image and
save a side-by-side comparison figure plus per-method overlays.
"""
import os
import time
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.measure import regionprops

IMG_PATH = "data/nuclei_00.jpg"
OUT_DIR = "data/nuclei_out"
os.makedirs(OUT_DIR, exist_ok=True)

img = np.array(Image.open(IMG_PATH).convert("RGB"))
print("image shape:", img.shape, "dtype:", img.dtype)

# The image is fluorescence with signal almost entirely in the red channel.
# Both models want a single-channel intensity image for nuclei.
red = img[..., 0].astype(np.float32)
# Normalize 1-99.8 percentile (StarDist convention)
def norm(x, lo=1, hi=99.8):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    y = (x - a) / max(b - a, 1e-6)
    return np.clip(y, 0, 1)

red_n = norm(red)

# ---------- StarDist ----------
print("\n=== StarDist 2D_versatile_fluo ===")
from stardist.models import StarDist2D
sd_model = StarDist2D.from_pretrained("2D_versatile_fluo")
t0 = time.time()
sd_labels, _ = sd_model.predict_instances(red_n, prob_thresh=None, nms_thresh=None)
sd_t = time.time() - t0
sd_count = int(sd_labels.max())
print(f"StarDist count: {sd_count}  time: {sd_t:.2f}s")

# ---------- Cellpose-SAM ----------
print("\n=== Cellpose-SAM ===")
from cellpose import models as cp_models
cp_model = cp_models.CellposeModel(gpu=False)
t0 = time.time()
# Cellpose 4.x: single-channel image; diameter=None lets the model estimate it
cp_masks, _flows, _styles = cp_model.eval(red_n, diameter=None)
cp_t = time.time() - t0
cp_count = int(cp_masks.max())
print(f"Cellpose-SAM count: {cp_count}  time: {cp_t:.2f}s")

# ---------- visualization ----------
def overlay(image_rgb, labels):
    return label2rgb(
        labels, image=image_rgb, bg_label=0, alpha=0.45, image_alpha=1.0,
        kind="overlay",
    )

sd_overlay = overlay(img, sd_labels)
cp_overlay = overlay(img, cp_masks)

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
axes[0].imshow(img); axes[0].set_title("Input (raw)"); axes[0].axis("off")
axes[1].imshow(sd_overlay)
axes[1].set_title(f"StarDist 2D_versatile_fluo\nN = {sd_count}   ({sd_t:.1f}s)")
axes[1].axis("off")
axes[2].imshow(cp_overlay)
axes[2].set_title(f"Cellpose-SAM\nN = {cp_count}   ({cp_t:.1f}s)")
axes[2].axis("off")
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "comparison.png")
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print("\nsaved", out_path)

# Save individual overlays too
Image.fromarray((sd_overlay * 255).astype(np.uint8)).save(
    os.path.join(OUT_DIR, "stardist_overlay.png"))
Image.fromarray((cp_overlay * 255).astype(np.uint8)).save(
    os.path.join(OUT_DIR, "cellpose_overlay.png"))

# Quick stats
def stats(labels, name):
    if labels.max() == 0:
        print(f"{name}: no objects")
        return
    props = regionprops(labels)
    areas = np.array([p.area for p in props])
    diams = np.array([p.equivalent_diameter for p in props])
    print(f"{name}: count={len(props)}  "
          f"area median={np.median(areas):.0f} px  "
          f"diam median={np.median(diams):.1f} px  "
          f"(min/max diam {diams.min():.1f}/{diams.max():.1f})")

print()
stats(sd_labels, "StarDist")
stats(cp_masks,  "Cellpose-SAM")
