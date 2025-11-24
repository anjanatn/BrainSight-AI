import os
import io
import time
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")  # server-friendly backend
import matplotlib.pyplot as plt

# try to import scipy for better filters; fallback if missing
try:
    from scipy.ndimage import gaussian_filter, sobel, distance_transform_edt
    SCIPY = True
except Exception:
    SCIPY = False

# --- Config ---
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "results"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ALLOWED_EXT = {"png", "jpg", "jpeg", "tif", "tiff"}

# DEMO image path (the file you uploaded earlier in this environment)
DEMO_IMAGE = "/mnt/data/op.PNG"

OUT_COMPOSITE = UPLOAD_FOLDER / "generated_6panel.png"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.secret_key = "dev-key"

# ---------------------
# Utility functions
# ---------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def normalize01(x):
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def load_image_to_array(path_or_file, size=(512,512)):
    """path_or_file may be a filesystem path or a file-like object returned by Flask."""
    if hasattr(path_or_file, "read"):
        img = Image.open(io.BytesIO(path_or_file.read())).convert("L")
    else:
        img = Image.open(path_or_file).convert("L")
    img = img.resize(size, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)
    return arr, img

def synthetic_heatmaps(img_arr):
    """Return six arrays: placeholder (normalized), tumor, lesion, atrophy, hemorrhage, volumetric mask (prob)."""
    imgn = normalize01(img_arr)
    H, W = imgn.shape
    yy, xx = np.mgrid[0:H, 0:W]

    # Placeholder CT-like base (radial)
    cy, cx = H//2, W//2
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    base = np.exp(-(dist/(0.48*max(H,W)))**2)
    base = normalize01(base)

    # Tumor: smooth bright central-lobe
    if SCIPY:
        smooth = gaussian_filter(imgn, sigma=6)
    else:
        smooth = imgn.copy()
    center_weight = np.exp(-(dist/(0.40*max(H,W)))**2)
    tumor = normalize01(smooth * center_weight)

    # Lesion: gradient-based
    if SCIPY:
        gx = sobel(imgn, axis=1); gy = sobel(imgn, axis=0)
        grad = np.hypot(gx, gy)
    else:
        gx = np.abs(np.gradient(imgn, axis=1)); gy = np.abs(np.gradient(imgn, axis=0))
        grad = np.hypot(gx, gy)
    lesion = normalize01(grad)
    if SCIPY:
        lesion = gaussian_filter(lesion, sigma=3)

    # Atrophy: diffuse low frequency (larger-scale darker regions)
    if SCIPY:
        low = gaussian_filter(imgn, sigma=12)
    else:
        low = imgn.copy()
    atrophy = normalize01(1.0 - low)
    if SCIPY:
        atrophy = gaussian_filter(atrophy, sigma=4)

    # Hemorrhage: bright-spot detection
    th = imgn.mean() + 0.12 * imgn.std()
    bright = (imgn > th).astype(float)
    if SCIPY:
        distmap = distance_transform_edt(~(bright.astype(bool)))
        hem = gaussian_filter(bright, sigma=2) * (1.0 - distmap/(distmap.max()+1e-12))
    else:
        hem = bright
    hemorrhage = normalize01(hem)

    # Volumetric proxy: tumor mask probability scaled
    vol = normalize01(tumor)

    return base, tumor, lesion, atrophy, hemorrhage, vol

def compute_probabilities(maps):
    """
    maps: tuple/list of 5 probability maps (tumor, lesion, atrophy, hemorrhage, vol)
    Return normalized scalar probabilities [0..100] for each abnormality.
    Simple heuristic: mean probability scaled and normalized.
    """
    tumor, lesion, atrophy, hemorrhage, vol = maps
    def score(x):
        return float(np.mean(x))
    raw = {
        "Tumor": score(tumor),
        "Lesion": score(lesion),
        "Atrophy": score(atrophy),
        "Hemorrhage": score(hemorrhage),
        "VolumeProxy": score(vol)
    }
    # convert to percentage (0-100) and calibrate mildly
    probs = {k: min(99.9, round(v*100.0, 1)) for k, v in raw.items()}
    return probs

def save_individual_maps(base, tumor, lesion, atrophy, hemorrhage, vol, prefix="result"):
    # save greyscale + heatmaps to app static folder
    out_dir = Path(app.config["UPLOAD_FOLDER"])
    timestamp = int(time.time())
    names = {}
    # base
    base_path = out_dir / f"{prefix}_{timestamp}_base.png"
    Image.fromarray((normalize01(base)*255).astype("uint8")).save(base_path)
    names["base"] = base_path.name
    # others as color heatmaps using matplotlib
    cmap = plt.get_cmap("inferno")
    def save_map(arr, name):
        rgb = (cmap(normalize01(arr))[:, :, :3] * 255).astype("uint8")
        p = out_dir / f"{prefix}_{timestamp}_{name}.png"
        Image.fromarray(rgb).save(p)
        return p.name
    names["tumor"] = save_map(tumor, "tumor")
    names["lesion"] = save_map(lesion, "lesion")
    names["atrophy"] = save_map(atrophy, "atrophy")
    names["hemorrhage"] = save_map(hemorrhage, "hem")
    names["vol"] = save_map(vol, "vol")
    return names

def compose_6panel(base, tumor, lesion, atrophy, hemorrhage, vol, outpath):
    # create a composite 2x3 figure and save
    fig = plt.figure(figsize=(14,8))
    ax1 = fig.add_subplot(2,3,1)
    ax1.imshow(normalize01(base), cmap="gray"); ax1.set_title("Input / Placeholder"); ax1.axis("off")
    ax2 = fig.add_subplot(2,3,2)
    ax2.imshow(normalize01(tumor), cmap="inferno"); ax2.set_title("Tumor heatmap"); ax2.axis("off")
    ax3 = fig.add_subplot(2,3,3)
    ax3.imshow(normalize01(lesion), cmap="inferno"); ax3.set_title("Lesion heatmap"); ax3.axis("off")
    ax4 = fig.add_subplot(2,3,4)
    ax4.imshow(normalize01(atrophy), cmap="inferno"); ax4.set_title("Atrophy heatmap"); ax4.axis("off")
    ax5 = fig.add_subplot(2,3,5)
    ax5.imshow(normalize01(hemorrhage), cmap="inferno"); ax5.set_title("Hemorrhage heatmap"); ax5.axis("off")
    from mpl_toolkits.mplot3d import Axes3D
    ax6 = fig.add_subplot(2,3,6, projection="3d")
    ys, xs = np.where(normalize01(vol) > 0.4)
    zs = (normalize01(vol)[ys, xs] * 50.0) if len(xs)>0 else np.array([0])
    if len(xs) > 0:
        ax6.scatter(xs, ys, zs, c="red", s=3, alpha=0.8)
        ax6.set_xlim(0, base.shape[1]); ax6.set_ylim(0, base.shape[0])
        ax6.set_zlim(0, max(50.0, float(np.max(zs)+1)))
    ax6.set_title("Volumetric map")
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------------
# Flask views
# ---------------------
@app.route("/", methods=["GET", "POST"])
def upload():
    return render_template("upload.html", demo_path=DEMO_IMAGE)

@app.route("/predict", methods=["POST"])
def predict():
    # if user clicked "use demo"
    if request.form.get("use_demo"):
        if not os.path.exists(DEMO_IMAGE):
            flash(f"Demo image not found at {DEMO_IMAGE}", "error")
            return redirect(url_for("upload"))
        arr, pil = load_image_to_array(DEMO_IMAGE, size=(512,512))
    else:
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("No file selected", "error")
            return redirect(url_for("upload"))
        if not allowed_file(file.filename):
            flash("File type not allowed", "error")
            return redirect(url_for("upload"))
        # reset file stream pointer for reading inside load_image_to_array
        stream = file.stream.read()
        arr, pil = load_image_to_array(io.BytesIO(stream), size=(512,512))

    # generate maps
    base, tumor, lesion, atrophy, hemorrhage, vol = synthetic_heatmaps(arr)
    probs = compute_probabilities((tumor, lesion, atrophy, hemorrhage, vol))

    # save maps and composite
    names = save_individual_maps(base, tumor, lesion, atrophy, hemorrhage, vol, prefix="result")
    composite_filename = f"composite_{int(time.time())}.png"
    composite_path = Path(app.config["UPLOAD_FOLDER"]) / composite_filename
    compose_6panel(base, tumor, lesion, atrophy, hemorrhage, vol, composite_path)

    # build urls for template
    # We also include the literal local demo path as requested
    urls = {
        "base": url_for("static", filename=f"results/{names['base']}"),
        "tumor": url_for("static", filename=f"results/{names['tumor']}"),
        "lesion": url_for("static", filename=f"results/{names['lesion']}"),
        "atrophy": url_for("static", filename=f"results/{names['atrophy']}"),
        "hemorrhage": url_for("static", filename=f"results/{names['hemorrhage']}"),
        "vol": url_for("static", filename=f"results/{names['vol']}"),
        "composite": url_for("static", filename=f"results/{composite_filename}"),
        # ALSO include the actual local uploaded demo path so you can see it (developer asked to expose)
        "demo_local_path": DEMO_IMAGE
    }

    return render_template("result.html", maps=urls, probs=probs)

# Serve static results are served via Flask's static folder automatically

if __name__ == "__main__":
    # Run dev server
    app.run(host="0.0.0.0", port=5000, debug=True)
