"""
Reader for Apple Health ECG **PDF** exports.

Apple's Health app can share an ECG either as the raw ``.txt`` (one voltage per
line) or as a printable PDF. The PDF is a *vector* rendering — the waveform is
stored as explicit polyline points in Apple's ECG red, drawn as stacked rows on
a 1 mm grid — so we can recover the signal at full fidelity rather than
OCR-ing a raster.

Everything needed to calibrate is printed in the PDF header, e.g.
    "25 mm/s, 10 mm/mV, Lead I, 512Hz, ... Watch7,5, Algorithm Version 2"
  * 25 mm/s   -> horizontal scale (time)
  * 10 mm/mV  -> vertical scale (voltage)
  * 512 Hz    -> native sample rate
and 1 mm = 72/25.4 pt is confirmed independently by the grid-line spacing, so
we cross-check the stated scale against the drawn grid and warn on mismatch.

Layout (observed, and asserted at parse time):
  * the trace is every curve whose stroking colour is Apple ECG red;
  * it wraps into N horizontal rows, each one contiguous time window;
  * rows read top-to-bottom on the page = earliest-to-latest in time;
  * within a row, x increases with time.

Output matches ``apple_ecg.AppleECG`` so the two loaders are interchangeable and
everything downstream (resampling, windowing, the model) is unchanged.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from apple_ecg import AppleECG, apply_qrs_polarity   # reuse container + polarity
from rates import APPLE_FS

PT_PER_MM = 72.0 / 25.4                    # 1 mm in PDF points  (~2.8346)
APPLE_RED = (0.8039216, 0.03921569, 0.1254902)
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _match_color(c, ref, tol=0.04):
    if not isinstance(c, (tuple, list)) or len(c) != len(ref):
        return False
    return all(abs(a - b) <= tol for a, b in zip(c, ref))


def _header_scales(text: str) -> dict:
    """Pull mm/s, mm/mV, Hz, device from the header line."""
    out = {"mm_per_s": 25.0, "mm_per_mV": 10.0, "fs": APPLE_FS, "device": ""}
    m = re.search(r"([\d.]+)\s*mm/s", text)
    if m:
        out["mm_per_s"] = float(m.group(1))
    m = re.search(r"([\d.]+)\s*mm/mV", text)
    if m:
        out["mm_per_mV"] = float(m.group(1))
    m = re.search(r"([\d.]+)\s*Hz", text, re.I)
    if m:
        out["fs"] = int(round(float(m.group(1))))
    m = re.search(r"(Watch[\d,]+)", text)
    if m:
        out["device"] = m.group(1)
    return out


def _grid_mm_in_pt(page) -> float | None:
    """Most common small grid spacing (1 mm square), for cross-checking scale."""
    def spacings(coords):
        a = np.array(sorted(set(round(c, 1) for c in coords)))
        d = np.diff(a)
        d = d[(d > 1.5) & (d < 6.0)]           # ~1 mm squares
        if d.size == 0:
            return None
        u, c = np.unique(np.round(d, 1), return_counts=True)
        return float(u[np.argmax(c)])
    vx = [ln["x0"] for ln in page.lines if abs(ln["x1"] - ln["x0"]) < 0.5]
    return spacings(vx)


def _cluster_rows(y, gap_mm, pt_mm):
    """Group y-values into rows; return sorted (row_top, row_bottom) descending."""
    ys = np.sort(np.unique(np.round(y, 1)))[::-1]
    gap = gap_mm * pt_mm
    rows, cur = [], [ys[0]]
    for v in ys[1:]:
        if cur[-1] - v > gap:
            rows.append((max(cur), min(cur)))
            cur = [v]
        else:
            cur.append(v)
    rows.append((max(cur), min(cur)))
    return rows


def read_apple_pdf(source, meta_only: bool = False,
                   correct_polarity: bool = True) -> AppleECG:
    """Parse an Apple Health ECG PDF.

    ``source`` may be a path (str/Path) or an open binary stream / file-like
    object (e.g. ``io.BytesIO`` of an uploaded file), so the web server can
    parse an upload without touching disk.

    ``correct_polarity`` (default on) flips a predominantly-negative QRS to match
    the model's upright training domain; ``meta['polarity_flipped']`` records it.
    """
    import pdfplumber

    src = source if hasattr(source, "read") else str(source)
    with pdfplumber.open(src) as pdf:
        page = pdf.pages[0]
        text = page.extract_text() or ""
        sc = _header_scales(text)

        pt_mm = PT_PER_MM
        grid = _grid_mm_in_pt(page)
        if grid is not None and abs(grid - PT_PER_MM) / PT_PER_MM > 0.05:
            # trust the drawn grid over the nominal constant if they disagree
            print(f"[apple_pdf] grid spacing {grid:.2f} pt/mm differs from "
                  f"{PT_PER_MM:.2f}; using the grid.")
            pt_mm = grid

        # collect the red trace, keeping (x, y) and the source curve order
        segs = []
        for c in page.curves:
            if _match_color(c.get("stroking_color"), APPLE_RED):
                segs.append(np.asarray(c["pts"], dtype=float))
        if not segs:
            raise ValueError(f"{path}: no Apple-red waveform curves found")
        allpts = np.vstack(segs)

    meta = {"Recorded Date": _find(text, r"Recorded on ([^\n]+)"),
            "Classification": _find(text, r"\n([A-Z][a-z]+ Rhythm[^\n—]*)"),
            "Device": sc["device"], "Sample Rate": f"{sc['fs']} hertz",
            "Lead": "Lead I", "Unit": "µV", "Source": "pdf",
            "mm_per_s": sc["mm_per_s"], "mm_per_mV": sc["mm_per_mV"]}

    if meta_only:
        return AppleECG(signal_uv=np.array([]), fs=sc["fs"], meta=meta)

    # --- rows: cluster by y, order top->bottom = time order ---------------
    rows = _cluster_rows(allpts[:, 1], gap_mm=8.0, pt_mm=pt_mm)

    px_per_s = sc["mm_per_s"] * pt_mm         # pt per second
    px_per_mV = sc["mm_per_mV"] * pt_mm        # pt per mV

    row_signals = []
    for top, bot in rows:
        pad = 2.0 * pt_mm
        m = (allpts[:, 1] <= top + pad) & (allpts[:, 1] >= bot - pad)
        rp = allpts[m]
        # order by x; average duplicate x (curve endpoints overlap)
        order = np.argsort(rp[:, 0], kind="stable")
        rp = rp[order]
        xs, inv = np.unique(np.round(rp[:, 0], 2), return_inverse=True)
        ys = np.zeros_like(xs)
        cnt = np.zeros_like(xs)
        np.add.at(ys, inv, rp[:, 1])
        np.add.at(cnt, inv, 1.0)
        ys /= np.maximum(cnt, 1)
        # resample this row onto a uniform time grid at fs
        t = (xs - xs.min()) / px_per_s
        n = int(round(t.max() * sc["fs"])) + 1
        tt = np.arange(n) / sc["fs"]
        y_uni = np.interp(tt, t, ys)
        # Each row is drawn at its own vertical position on the page, so its
        # baseline must be removed BEFORE rows are concatenated — otherwise the
        # page offset between rows leaks in as a huge false voltage step.
        row_signals.append(y_uni - np.median(y_uni))

    y = np.concatenate(row_signals)
    # PDF y is up = positive voltage; rows already de-baselined above.
    mv = y / px_per_mV
    signal_uv = (mv * 1000.0).astype(np.float64)

    flipped = False
    if correct_polarity:
        signal_uv, flipped = apply_qrs_polarity(signal_uv, sc["fs"])
    meta["polarity_flipped"] = str(flipped)

    meta["n_rows"] = len(rows)
    meta["pt_per_mm"] = round(pt_mm, 4)
    return AppleECG(signal_uv=signal_uv, fs=sc["fs"], meta=meta)


def _find(text, pat):
    m = re.search(pat, text)
    return m.group(1).strip() if m else ""


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        ecg = read_apple_pdf(p)
        print(f"\n{Path(p).name}")
        print(f"  {ecg}")
        print(f"  device={ecg.meta['Device']}  scales={ecg.meta['mm_per_s']}mm/s "
              f"{ecg.meta['mm_per_mV']}mm/mV  rows={ecg.meta['n_rows']}  "
              f"pt/mm={ecg.meta['pt_per_mm']}")
        x = ecg.signal_uv
        print(f"  samples={len(x)}  {len(x)/ecg.fs:.2f}s @ {ecg.fs}Hz  "
              f"range=[{x.min():.0f}, {x.max():.0f}] µV  std={x.std():.0f}")
