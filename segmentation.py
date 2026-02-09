import cv2
import numpy as np
from matplotlib import pyplot as plt

# ===== Change paths here =====
INPUT_PATH = r"C:\Users\Shaof\OneDrive\Desktop\ALHA FIB 02-2.tiff"
OUTPUT_PATH = r"C:\Users\Shaof\OneDrive\Desktop\Olivine_Pyroxene_map_green_blue.png"

# ===== Read image =====
img = cv2.imread(INPUT_PATH)
if img is None:
    raise ValueError(f"Image not loaded: {INPUT_PATH}")

bgr = img
lab_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

# ===== Sample colors =====
olivine_hex = [
    "10c193ff","2e9761ff","23ac6dff","11883eff","0b9d7aff",
    "0f7447ff","0ebd9eff","1ab2a2ff","1c8c78ff","30b183ff"
]

pyroxene_hex = [
    "048c97ff","008db1ff","219792ff","05b2bcff","1cacb6ff",
    "0bb1b8ff","248690ff","0d9fafff","0f7fadff","069490ff"
]

def hex_to_rgb(h):
    h = h.strip().lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return np.array([r, g, b], dtype=np.uint8)

def rgbs_to_lab(rgb_arr):
    bgr_arr = rgb_arr[:, ::-1].reshape(-1, 1, 3)
    lab = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return lab

oli_rgb = np.stack([hex_to_rgb(h) for h in olivine_hex], axis=0)
px_rgb  = np.stack([hex_to_rgb(h) for h in pyroxene_hex], axis=0)

oli_lab_samp = rgbs_to_lab(oli_rgb)
px_lab_samp  = rgbs_to_lab(px_rgb)

# ===== Color centers and thresholds =====
oli_center = oli_lab_samp.mean(axis=0)
px_center  = px_lab_samp.mean(axis=0)

oli_samp_dist = np.linalg.norm(oli_lab_samp - oli_center, axis=1)
px_samp_dist  = np.linalg.norm(px_lab_samp  - px_center,  axis=1)

oli_thr = float(np.percentile(oli_samp_dist, 90))
px_thr  = float(np.percentile(px_samp_dist,  97))

expand = 1.15
oli_thr *= expand
px_thr  *= expand

print("Auto thresholds:")
print("  oli_thr =", round(oli_thr, 2), "px_thr =", round(px_thr, 2))

# ===== Distance maps =====
d_oli = np.linalg.norm(lab_img - oli_center, axis=2).astype(np.float32)
d_px  = np.linalg.norm(lab_img - px_center,  axis=2).astype(np.float32)

candidate = (d_oli <= oli_thr) | (d_px <= px_thr)

# ===== Initial classification =====
diff_raw = d_px - d_oli
sigma = 0.8
diff_s = cv2.GaussianBlur(diff_raw, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)

margin_oli = 2.5
margin_px  = 2.0

oli_mask = candidate & (d_oli <= oli_thr) & (diff_s >  margin_oli)
px_mask  = candidate & (d_px  <= px_thr)  & (diff_s < -margin_px)

protect_margin = 5.0
pyx_protect = candidate & (d_px <= px_thr) & ((d_oli - d_px) <= protect_margin)
px_mask = px_mask | pyx_protect
oli_mask = oli_mask & (~px_mask)

undecided = candidate & (~oli_mask) & (~px_mask)
px_mask[undecided]  = d_px[undecided] < d_oli[undecided]
oli_mask[undecided] = ~px_mask[undecided]

# ===== Connected-component voting =====
cand_u8 = candidate.astype(np.uint8)
num_labels, labels = cv2.connectedComponents(cand_u8, connectivity=8)

oli_final = np.zeros_like(oli_mask, dtype=bool)
px_final  = np.zeros_like(px_mask, dtype=bool)

very_small = 3

for lab_id in range(1, num_labels):
    region = (labels == lab_id)
    area = int(region.sum())

    if area <= very_small:
        if float(d_oli[region].mean()) <= float(d_px[region].mean()):
            oli_final |= region
        else:
            px_final  |= region
        continue

    oli_count = int((region & oli_mask).sum())
    px_count  = int((region & px_mask).sum())

    if oli_count > px_count:
        oli_final |= region
    elif px_count > oli_count:
        px_final  |= region
    else:
        if float(d_oli[region].mean()) <= float(d_px[region].mean()):
            oli_final |= region
        else:
            px_final  |= region

px_final = px_final & (~oli_final)
oli_mask, px_mask = oli_final, px_final

# ===== Morphological cleanup =====
kernel = np.ones((3, 3), np.uint8)
oli_mask = cv2.morphologyEx(oli_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)
px_mask  = cv2.morphologyEx(px_mask.astype(np.uint8),  cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)
px_mask = px_mask & (~oli_mask)

# ===== Area statistics =====
total = bgr.shape[0] * bgr.shape[1]
oli_area = int(oli_mask.sum())
px_area  = int(px_mask.sum())

print("Relative to full image:", "Oli", oli_area/total, "Px", px_area/total)
den = max(oli_area + px_area, 1)
print("Within (Oli+Px):", "Oli", oli_area/den, "Px", px_area/den)

# ===== Output colors =====
oli_bgr_out = (0, 180, 0)
px_bgr_out  = (180, 0, 0)

out = np.full_like(bgr, 255)
out[oli_mask] = oli_bgr_out
out[px_mask]  = px_bgr_out

ok = cv2.imwrite(OUTPUT_PATH, out)
print("Saved:", ok, OUTPUT_PATH)

plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
