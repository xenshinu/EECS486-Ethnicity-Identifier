"""Check whether the red and green images are the same field of view (co-registered),
then run InstanSeg with the correct 2-channel input (nuclei=red, membrane=green).
Also dump StarDist with multiple thresholds for a fair comparison.
"""
import os, time, numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries

OUT = "data/nuclei_out"
os.makedirs(OUT, exist_ok=True)

img_red   = np.array(Image.open("data/nuclei_00.jpg").convert("RGB"))
img_green = np.array(Image.open("data/nuclei_01.jpg").convert("RGB"))
print("red ", img_red.shape, "green", img_green.shape)

# Extract single-channel signal from each (just dominant channel).
red_ch   = img_red[...,   0].astype(np.float32)
green_ch = img_green[..., 1].astype(np.float32)

# Visual co-registration check: overlay both as a 2-channel false-color
def norm(x, lo=1, hi=99.8):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x-a)/max(b-a,1e-6), 0, 1)

r, g = norm(red_ch), norm(green_ch)
composite = np.stack([r, g, np.zeros_like(r)], axis=-1)

# Save composite
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
axes[0].imshow(img_red);   axes[0].set_title("Red channel (nuclei?)");   axes[0].axis("off")
axes[1].imshow(img_green); axes[1].set_title("Green channel (cyto?)");  axes[1].axis("off")
axes[2].imshow(composite); axes[2].set_title("Overlay (R=red, G=green) — same FOV check"); axes[2].axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/coregistration_check.png", dpi=110, bbox_inches="tight")
plt.close()
print("saved coregistration_check.png")

# Pearson correlation between (low-passed) signals — rough indicator of co-registration
from scipy.ndimage import gaussian_filter
rL = gaussian_filter(r, 8)
gL = gaussian_filter(g, 8)
pearson = float(np.corrcoef(rL.ravel(), gL.ravel())[0, 1])
print(f"Low-passed Pearson r(red,green) = {pearson:.3f}")

# Now run InstanSeg the correct way: 2-channel input (C=2, H, W) with channel 0=nuclei, 1=cyto
from instanseg import InstanSeg
print("\nLoading InstanSeg...")
inst = InstanSeg("fluorescence_nuclei_and_cells", verbosity=0)

# Build CHW float image, channel 0 = red (nuclei), channel 1 = green (cyto)
chw = np.stack([red_ch, green_ch], axis=0).astype(np.float32)  # (2,H,W)
print("input shape", chw.shape)

t0 = time.time()
labels, _ = inst.eval_small_image(chw, pixel_size=0.5, target="all_outputs")
dt = time.time() - t0
arr = labels.cpu().numpy()
if arr.ndim == 4: arr = arr[0]
nuc, cell = arr[0].astype(np.int32), arr[1].astype(np.int32)
print(f"2-channel input -> nuclei={nuc.max()}  cells={cell.max()}  ({dt:.1f}s)")

# Visualise on the green image (since cells are on green)
def boundary_overlay(image, labels, color):
    out = image.copy()
    out[find_boundaries(labels, mode="outer")] = color
    return out

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes[0,0].imshow(composite); axes[0,0].set_title("2-channel composite (R=nuclei, G=cyto)"); axes[0,0].axis("off")
axes[0,1].imshow(boundary_overlay(img_red, nuc, [255,255,0]))
axes[0,1].set_title(f"nuclei boundaries on red img   N={nuc.max()}"); axes[0,1].axis("off")
axes[1,0].imshow(boundary_overlay(img_green, cell, [0,255,255]))
axes[1,0].set_title(f"cell boundaries on green img   N={cell.max()}"); axes[1,0].axis("off")
overlay_cells = label2rgb(cell, image=img_green, bg_label=0, alpha=0.45, image_alpha=1.0, kind="overlay")
axes[1,1].imshow(overlay_cells); axes[1,1].set_title(f"cells colour overlay  N={cell.max()}"); axes[1,1].axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/instanseg_2channel.png", dpi=110, bbox_inches="tight")
plt.close()
print("saved instanseg_2channel.png")

# Compare to StarDist at multiple prob thresholds on the red channel
print("\nStarDist threshold sweep on red channel:")
from stardist.models import StarDist2D
sd = StarDist2D.from_pretrained("2D_versatile_fluo")
red_n = norm(red_ch)
for pt in (0.30, 0.479, 0.60, 0.75):
    lbl, _ = sd.predict_instances(red_n, prob_thresh=pt, nms_thresh=0.3)
    print(f"  prob_thresh={pt}: N={lbl.max()}")

# Also: count just the bright spots in red (a sanity "by-eye" check via simple thresholding)
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, label as sk_label
mask = red_n > max(0.15, threshold_otsu(red_n))
mask = remove_small_objects(mask, 30)
naive = sk_label(mask)
print(f"\nNaive Otsu+CC count on red (sanity check): {naive.max()}")
