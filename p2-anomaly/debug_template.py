"""Visual/quantitative sanity check for the template baseline on one category."""
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import p2common as C

root = "/tmp/slv176/data/VisA_pytorch/1cls"
cat = sys.argv[1] if len(sys.argv) > 1 else "candle"
calib, pool = C.ref_pool(root, cat, 20)
ref_path = os.path.join(C.train_dir(root, cat), pool[0])
ref = C.prep_square(C.load_rgb(ref_path))
print("ref:", ref_path)

gr = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255


def dm(img):
    gi = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    d_abs = np.abs(gi - gr)
    _, s = structural_similarity(gr, gi, data_range=1.0, full=True, win_size=11)
    d = 0.5 * d_abs + 0.5 * np.clip(1 - s, 0, 1)
    return gaussian_filter(d, sigma=4)


rows = []
items = C.test_items(root, cat)
for label, p, _ in items[:]:
    img = C.prep_square(C.load_rgb(p))
    d = dm(img)
    rows.append((label, C.top1p_mean(d), d.mean(), float(np.abs(
        cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255 - gr).mean()),
        os.path.basename(p)))

rows.sort(key=lambda r: r[1])
good = [r for r in rows if r[0] == 0]
bad = [r for r in rows if r[0] == 1]
print("n good=%d bad=%d" % (len(good), len(bad)))
for name, rs in (("good", good), ("bad", bad)):
    a = np.array([r[1] for r in rs])
    b = np.array([r[3] for r in rs])
    print(f"{name:5s} top1p mean={a.mean():.4f} med={np.median(a):.4f} | grayMAE mean={b.mean():.4f}")
print("\nlowest 5:", [(r[4], r[0], round(r[1], 4)) for r in rows[:5]])
print("highest 5:", [(r[4], r[0], round(r[1], 4)) for r in rows[-5:]])

# montage: ref + 3 good + 3 bad with their maps
sel = [("ref", ref, None)] + [(f"good {r[4]}", C.prep_square(C.load_rgb(
    [i[1] for i in items if os.path.basename(i[1]) == r[4]][0])), dm(C.prep_square(C.load_rgb(
        [i[1] for i in items if os.path.basename(i[1]) == r[4]][0])))) for r in good[:3]] + [
    (f"bad {r[4]}", C.prep_square(C.load_rgb(
        [i[1] for i in items if os.path.basename(i[1]) == r[4]][0])), dm(C.prep_square(C.load_rgb(
            [i[1] for i in items if os.path.basename(i[1]) == r[4]][0])))) for r in bad[:3]]

tiles = []
for name, img, d in sel:
    t = cv2.resize(img, (224, 224))
    if d is not None:
        h = cv2.applyColorMap((np.clip(d / max(d.max(), 1e-6), 0, 1) * 255).astype(np.uint8),
                              cv2.COLORMAP_JET)
        t = cv2.hconcat([t, cv2.resize(h, (224, 224))])
    else:
        t = cv2.hconcat([t, np.zeros((224, 224, 3), np.uint8)])
    cv2.putText(t, name[:28], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    tiles.append(t)
out = cv2.vconcat(tiles)
os.makedirs("/tmp/slv176/debug", exist_ok=True)
cv2.imwrite(f"/tmp/slv176/debug/{cat}_check.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
print("wrote /tmp/slv176/debug/%s_check.png" % cat)
