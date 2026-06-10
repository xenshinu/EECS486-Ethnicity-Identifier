"""Generate thumbnails of all uploaded TIFFs and detect the dominant colour
of each so we can map c1 / c2 / c1-2 to nuclei / GFP / merge.
"""
import glob, os
import numpy as np
import tifffile
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

paths = sorted(glob.glob("data/crx_gfp/**/*.tif", recursive=True))
os.makedirs("data/crx_gfp/_preview", exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()
for i, p in enumerate(paths):
    img = tifffile.imread(p)
    # downsample for preview
    thumb = img[::6, ::6]
    rgb_mean = img.reshape(-1, 3).mean(axis=0)
    dominant = ["R", "G", "B"][int(np.argmax(rgb_mean))]
    n_red = float(rgb_mean[0]); n_green = float(rgb_mean[1])
    name = os.path.basename(p)
    axes[i].imshow(thumb)
    axes[i].set_title(f"{name}\nmean R={n_red:.1f}  G={n_green:.1f}  dom={dominant}")
    axes[i].axis("off")
plt.tight_layout()
plt.savefig("data/crx_gfp/_preview/all_six.png", dpi=110, bbox_inches="tight")
print("saved data/crx_gfp/_preview/all_six.png")
