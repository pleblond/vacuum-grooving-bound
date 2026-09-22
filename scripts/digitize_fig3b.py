"""Digitize Fig. 3b of Lee et al., PRL 136, 033801 (2026), arXiv:2509.13503v1.

Fetches the arXiv source tarball, extracts the drift-rate figure
(``figures/drift_07102025.jpg''), and digitizes the four cavity traces by
color segmentation against a hand-calibrated axis mapping.

Axis calibration (pixels of figures/drift_07102025.jpg cropped at 45% height):
  x: log10(days) = 2 + (x - 362) / 1470 * 2      (left/right spines)
  top half (+): log10(v) fit on tick labels 1000/100/10  at rows 18/212/418
  bottom half (-): log10|v| fit on tick labels 10/100/1000 at rows 706/902/1085
Contamination control (all verified against upscaled crops of the figure):
  per-trace x-ranges cut empty zones (dark vertical minor-grid dashes),
  spine break-slash zones, and annotation arrows/text past the trace ends;
  red/blue halos removed from the black mask by 3px dilation;
  red/blue pixels above row 640 are label text + arrow tops;
  the black "Si2 -50 uHz/s" label sits inside the plot and is boxed out.
  Thin arrow shafts crossing trace zones are median-robust in log-x bins.

Output: data/lee2026_fig3b.csv + data/lee2026_fig3b_meta.json
Provenance is recorded in the meta file. Recent-epoch values are better taken
from the paper text (-50/-15/-20/+10 uHz/s); the digitized curves supply the
decay history, which is what the text does not tabulate.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

ARXIV_ID = "2509.13503v1"
FIGURE_NAME = "figures/drift_07102025.jpg"

# Plot interior (cropped-figure pixels).
X0, X1 = 362, 1832
YT0, YT1 = 12, 556
YB0, YB1 = 573, 1094

# Tick-label calibration: (pixel row, value).
TOP_CAL = [(18, 1e3), (212, 1e2), (418, 1e1)]
BOT_CAL = [(706, 1e1), (902, 1e2), (1085, 1e3)]

TEXT_BOXES = {  # kept for metadata only; end-trimming is by X_CUTS/Y_ANNOT_CUT
    "Si6": [(975, 150, 1200, 360)],
    "Si5": [],
    "Si3": [],
    "Si2": [],
}

# Per-trace x range (cropped-figure px): each trace only spans part of the
# axis; cutting empty zones removes grid-dash/artefact bogus bins. Also kills
# annotation arrows/text past the trace ends and spine break-slash zones.
X_RANGES = {
    "Si6": (420, 960),
    "Si3": (600, 1560),
    "Si5": (940, 1460),
    "Si2": (660, 1530),
}
# The black "Si2 -50 uHz/s" label sits inside the plot area (right side).
BLACK_TEXT_BOX = (1490, 925, 1800, 1095)
# Red/blue pixels above this row are label text + arrow tops (trace is deeper).
Y_ANNOT_CUT = 640

N_BINS = 55


def fetch_figure(cache: Path) -> np.ndarray:
    from PIL import Image

    cache.mkdir(parents=True, exist_ok=True)
    tar_path = cache / f"{ARXIV_ID}.tar.gz"
    if not tar_path.exists():
        url = f"https://arxiv.org/e-print/{ARXIV_ID}"
        with urllib.request.urlopen(url, timeout=60) as r, open(tar_path, "wb") as f:
            f.write(r.read())
    with tarfile.open(tar_path) as tar:
        member = tar.getmember(FIGURE_NAME)
        raw = tar.extractfile(member).read()
    full = np.array(Image.open(io.BytesIO(raw)).convert("RGB")).astype(int)
    return full[int(full.shape[0] * 0.45):]


def _fit_log(rows_vals):
    rows = np.array([r for r, _ in rows_vals], float)
    vals = np.array([v for _, v in rows_vals], float)
    b, a = np.polyfit(rows, np.log10(vals), 1)
    return a, b


def digitize(pb: np.ndarray) -> tuple[dict, dict]:
    from scipy.ndimage import binary_dilation

    R, G, B = pb[..., 0], pb[..., 1], pb[..., 2]
    # Interior excludes both spine break-slash zones (x<400, x>1800).
    top = np.zeros(pb.shape[:2], bool)
    bot = np.zeros(pb.shape[:2], bool)
    top[YT0:YT1, X_RANGES["Si6"][0]:X_RANGES["Si6"][1]] = True
    bot[YB0:YB1, X_RANGES["Si2"][0]:X_RANGES["Si2"][1]] = True

    red = bot & (R > 120) & ((R - G) > 40) & ((R - B) > 40)
    blue = bot & (B > 120) & ((B - R) > 40) & ((B - G) > 30)
    black = bot & ((R + G + B) < 250)
    # Dark antialiased halos of the red/blue traces read as "black":
    # drop black pixels adjacent to any red/blue pixel.
    black = black & ~binary_dilation(red | blue, iterations=3)
    purple = top & (R > 90) & (B > 90) & ((R - G) > 25) & ((B - G) > 25)
    rows = np.arange(pb.shape[0])[:, None]
    cols = np.arange(pb.shape[1])[None, :]
    # X-axis minor-tick marks and label tops ("2".."9" per decade) intrude a
    # few px above the axis (y > 1050) as black pixels. Blank them in the
    # black mask only (ticks are black, never red/blue/purple).
    for decade in (2, 3):
        for k in range(2, 10):
            xt = X0 + (np.log10(k) + decade - 2.0) / 2.0 * (X1 - X0)
            black[(rows > 1050) & (np.abs(cols - xt) <= 8)] = False
    red = red & (rows >= Y_ANNOT_CUT) & (cols >= X_RANGES["Si3"][0]) & (cols <= X_RANGES["Si3"][1])
    blue = blue & (rows >= Y_ANNOT_CUT) & (cols >= X_RANGES["Si5"][0]) & (cols <= X_RANGES["Si5"][1])
    x0, y0, x1, y1 = BLACK_TEXT_BOX
    black[y0:y1, x0:x1] = False
    raw_masks = {"Si6": purple, "Si3": red, "Si5": blue, "Si2": black}
    for name, boxes in TEXT_BOXES.items():
        for x0, y0, x1, y1 in boxes:
            raw_masks[name][y0:y1, x0:x1] = False

    a_top, b_top = _fit_log(TOP_CAL)
    a_bot, b_bot = _fit_log(BOT_CAL)

    out: dict[str, list[tuple[float, float, int]]] = {}
    for name, m in raw_masks.items():
        ys, xs = np.where(m)
        if name == "Si6":
            logv = a_top + b_top * ys
            drift = 10.0 ** logv
        else:
            logv = a_bot + b_bot * ys
            drift = -(10.0 ** logv)
        days = 10.0 ** (2.0 + (xs - X0) / (X1 - X0) * 2.0)
        lo, hi = np.log10(days.min()), np.log10(days.max())
        edges = np.logspace(lo, hi, N_BINS + 1)
        pts = []
        for k in range(N_BINS):
            sel = (days >= edges[k]) & (days < edges[k + 1])
            if sel.sum() >= 8:
                pts.append((float(np.median(days[sel])), float(np.median(drift[sel])), int(sel.sum())))
        out[name] = pts
    meta = {
        "arxiv": ARXIV_ID,
        "figure": FIGURE_NAME,
        "method": "color segmentation vs hand-calibrated broken-log axes",
        "x_map": {"x0": X0, "x1": X1, "day0": 100.0, "day1": 10000.0},
        "top_cal_rows_vals": TOP_CAL,
        "bot_cal_rows_vals": BOT_CAL,
        "text_boxes_excluded": {k: v for k, v in TEXT_BOXES.items() if v},
        "bins": N_BINS,
        "min_pts_per_bin": 8,
        "stated_digitization_error": "~15% per point; endpoints better taken from paper text",
    }
    return out, meta


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / "data"
    data_dir.mkdir(exist_ok=True)
    pb = fetch_figure(Path("/tmp/lee2026_cache"))
    out, meta = digitize(pb)
    with open(data_dir / "lee2026_fig3b.csv", "w") as f:
        f.write("cavity,days_since_contacting,drift_rate_uHz_s,n_pixels\n")
        for name in ("Si2", "Si3", "Si5", "Si6"):
            for days, drift, n in out[name]:
                f.write(f"{name},{days:.1f},{drift:.3f},{n}\n")
    with open(data_dir / "lee2026_fig3b_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    for name, pts in out.items():
        arr = np.array(pts)
        print(f"{name}: {len(pts)} bins, days {arr[:,0].min():.0f}-{arr[:,0].max():.0f}, "
              f"last {arr[-1,1]:.1f} uHz/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
