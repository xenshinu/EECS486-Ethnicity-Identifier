"""Run StarDist 2D_versatile_fluo on the nuclei image and save overlay + side-by-side."""
import os, time
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

IMG_PATH = "data/nuclei_00.jpg"
OUT_DIR = "data/nuclei_out"
os.makedirs(OUT_DIR, exist_ok=True)

img = np.array(Image.open(IMG_PATH).convert("RGB"))
print("image shape:", img.shape)

red = img[..., 0].astype(np.float32)
def norm(x, lo=1, hi=99.8):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    y = (x - a) / max(b - a, 1e-6)
    return np.clip(y, 0, 1)
red_n = norm(red)

from stardist.models import StarDist2D
model = StarDist2D.from_pretrained("2D_versatile_fluo")
t0 = time.time()
labels, details = model.predict_instances(red_n)
dt = time.time() - t0
count = int(labels.max())
print(f"StarDist count = {count}  ({dt:.2f}s)")

# Coloured overlay (translucent)
overlay = label2rgb(labels, image=img, bg_label=0, alpha=0.45, image_alpha=1.0, kind="overlay")

# Boundaries only on raw image (cleaner visual)
boundaries = find_boundaries(labels, mode="outer")
img_with_bounds = img.copy()
img_with_bounds[boundaries] = [255, 255, 0]  # yellow boundaries

# Side-by-side comparison
fig, axes = plt.subplots(1, 3, figsize=(24, 7))
axes[0].imshow(img); axes[0].set_title("Input"); axes[0].axis("off")
axes[1].imshow(img_with_bounds); axes[1].set_title(f"StarDist boundaries\nN = {count}"); axes[1].axis("off")
axes[2].imshow(overlay); axes[2].set_title(f"StarDist colour overlay\nN = {count}"); axes[2].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "stardist_side_by_side.png"), dpi=110, bbox_inches="tight")
print("saved stardist_side_by_side.png")

# Individual overlays for inspection
Image.fromarray(img_with_bounds).save(os.path.join(OUT_DIR, "stardist_boundaries.png"))
Image.fromarray((overlay * 255).astype(np.uint8)).save(os.path.join(OUT_DIR, "stardist_overlay.png"))

# stats
props = regionprops(labels)
areas = np.array([p.area for p in props])
diams = np.array([p.equivalent_diameter for p in props])
print(f"area median={np.median(areas):.0f} px,  diam median={np.median(diams):.1f} px (min/max {diams.min():.1f}/{diams.max():.1f})")
