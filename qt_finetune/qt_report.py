"""
Apple Watch ECG -> user-facing QT report.

End to end: parse the export, gate on signal quality, run the model over the
three 10 s windows, derive QTc from the measured RR, place the result in a
calibrated tier, and render text a non-clinician can read.

Everyone gets a number. What changes across tiers is the *interval*, which is
fitted against cardiologist annotations (see calibrate_display.py):

    normal   +-18 ms   the model is accurate here (9 ms mean error, and no
                       truly-prolonged record landed in this tier on LUDB)
    monitor  +-54 ms   wider, because the model understates once readings climb
    refer    +-61 ms   all five LUDB records with QTc>500 land in this tier

So the uncertainty is disclosed by the width of the interval the user is shown,
rather than by one global error figure that would be 2x wrong for half of them.

QTc is reported with Bazett, which is the convention the 450/500 ms thresholds
are defined against, and is what the tier calibration was fitted in.

Usage:
    python qt_finetune/qt_report.py ~/Downloads/ecg_2025-01-15.txt
    python qt_finetune/qt_report.py ecg.txt --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import signal as sg

_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (_SCRIPT_DIR.parent / "models", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from apple_ecg import read_apple_ecg, to_windows
from rates import resample_signal, window_len
from qt_objective import Objective
from ecg_resnet1d import ECGResNet1d

DEFAULT_CKPT = _SCRIPT_DIR / "checkpoints" / "best_resnet_lead1_128hz_pinball_tailloss.pt"
DEFAULT_CALIB = _SCRIPT_DIR / "qt_calibration.json"

# Quality gates. The 2025-01-12 "High Heart Rate" export sits far outside both:
# RR coefficient of variation 0.67 against 0.07 for a clean recording.
MAX_RR_CV = 0.25          # beat-to-beat instability => motion/EMG artefact
MAX_WINDOW_SPREAD = 25.0  # ms disagreement between the three windows
MIN_BEATS = 15


def detect_r(x, fs):
    b, a = sg.butter(3, [5 / (fs / 2), 15 / (fs / 2)], btype="band")
    f = sg.filtfilt(b, a, x)
    d = np.diff(f, prepend=f[0])
    w = max(1, int(0.12 * fs))
    integ = np.convolve(d ** 2, np.ones(w) / w, mode="same")
    pk, _ = sg.find_peaks(integ, height=np.percentile(integ, 98) * 0.3,
                          distance=int(0.28 * fs))
    out, hw, med = [], int(0.05 * fs), np.median(x)
    for p in pk:
        lo, hi = max(0, p - hw), min(len(x), p + hw)
        if hi > lo:
            out.append(lo + int(np.argmax(np.abs(x[lo:hi] - med))))
    return np.array(sorted(set(out)))


def pick_tier(qtc95, calib):
    for t in calib["tiers"]:
        lo, hi = t["q95_lo"], t["q95_hi"]
        if (lo is None or qtc95 > lo) and (hi is None or qtc95 <= hi):
            return t
    return calib["tiers"][-1]


def analyse(path, ckpt=DEFAULT_CKPT, calib_path=DEFAULT_CALIB):
    calib = json.loads(Path(calib_path).read_text())
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = ECGResNet1d(in_channels=ck.get("in_channels", 1),
                        num_outputs=ck.get("num_outputs", 1))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    obj = Objective(ck.get("loss", "mse"),
                    tuple(ck.get("quantiles", (0.5,)))).to(torch.device("cpu"))
    fs_t = ck["target_fs"]

    ecg = read_apple_ecg(path)
    res = {"file": Path(path).name, "recorded": ecg.meta.get("Recorded Date", ""),
           "device": ecg.meta.get("Device", ""), "fs_native": ecg.fs,
           "duration_s": round(ecg.duration_s, 1),
           "apple_classification": ecg.classification, "model_fs": fs_t}

    # --- rhythm + quality, measured on the native-rate signal ---------------
    x_uv = ecg.signal_uv
    r = detect_r(x_uv, ecg.fs)
    if len(r) < MIN_BEATS:
        res.update(quality="poor", reason=f"only {len(r)} beats detected")
        return res, calib
    rr_s = np.diff(r) / ecg.fs
    rr_med = float(np.median(rr_s))
    rr_cv = float(np.std(rr_s) / max(np.mean(rr_s), 1e-9))
    res.update(hr_bpm=round(60.0 / rr_med, 1), rr_cv=round(rr_cv, 3),
               n_beats=int(len(r)))

    # --- model over the three windows ---------------------------------------
    wins = to_windows(ecg, fs_t)                       # (3, 1, L) mV
    L = window_len(fs_t)
    p50, p95 = [], []
    with torch.no_grad():
        for w in wins:
            lg = model(torch.from_numpy(w[None, :, :L]))["logits"]
            p50.append(float(obj.point(lg).item()))
            _lo, hi = obj.interval(lg)
            p95.append(float(hi.item()) if hi is not None else float("nan"))
    p50, p95 = np.asarray(p50), np.asarray(p95)
    spread = float(p50.max() - p50.min())
    res.update(n_windows=len(p50), window_spread_ms=round(spread, 1))

    if rr_cv > MAX_RR_CV:
        res.update(quality="poor", reason=f"unstable rhythm/artefact (RR CV "
                                          f"{rr_cv:.2f} > {MAX_RR_CV})")
        return res, calib
    if spread > MAX_WINDOW_SPREAD:
        res.update(quality="poor", reason=f"windows disagree by {spread:.0f} ms")
        return res, calib
    res["quality"] = "good"

    # --- calibrated QTc ------------------------------------------------------
    g = np.sqrt(rr_med)
    qt50 = float(np.median(p50)) - calib["global_bias_ms"]
    qt95 = float(np.median(p95)) - calib["global_bias_ms"]
    qtc50, qtc95 = qt50 / g, qt95 / g
    tier = pick_tier(qtc95, calib)
    shown = qtc50 - tier["bias_ms"]
    res.update(qt_ms=round(qt50, 1), qtc_bazett=round(shown, 1),
               qtc_half_width=round(tier["half_width_ms"], 1),
               qtc_low=round(shown - tier["half_width_ms"], 1),
               qtc_high=round(shown + tier["half_width_ms"], 1),
               tier=tier["name"], score_q95=round(qtc95, 1))
    return res, calib


MESSAGES = {
    "normal": ("NORMAL RANGE",
               "Your QT interval looks normal in this recording."),
    "monitor": ("SLIGHTLY ELEVATED",
                "This reading sits a little above the usual range. Single readings "
                "vary, so repeat it a few times over the next few days. If you take "
                "medication known to affect QT, or you have had fainting spells or "
                "palpitations, mention this reading to your doctor."),
    "refer": ("POSSIBLE QT PROLONGATION",
              "This reading suggests your QT interval may be prolonged. We recommend "
              "a clinical 12-lead ECG to check properly. This is a screening result, "
              "not a diagnosis."),
}
POOR = ("RECORDING QUALITY TOO LOW",
        "We could not get a reliable measurement from this recording. Sit still, "
        "rest your forearms on a table, keep your finger steady on the crown, and "
        "record again.")


def render(res, calib) -> str:
    L = []
    L.append(f"QT ANALYSIS — {res['file']}")
    meta = " · ".join(p for p in [res.get("recorded") or None, res.get("device"),
                                  f"Lead I · {res['duration_s']:.0f}s @ "
                                  f"{res['fs_native']} Hz"] if p)
    L.append(meta)
    if res.get("apple_classification"):
        L.append(f"Apple's own classification: {res['apple_classification']}")
    L.append("")

    if res.get("quality") != "good":
        head, body = POOR
        L.append(f"  [!] {head}")
        for line in _wrap(body, 74):
            L.append(f"  {line}")
        L.append(f"  (reason: {res.get('reason','unknown')})")
        return "\n".join(L)

    L.append(f"Heart rate {res['hr_bpm']:.0f} bpm · {res['n_beats']} beats · "
             f"quality good ({res['n_windows']} windows, spread "
             f"{res['window_spread_ms']:.0f} ms)")
    L.append("")
    L.append(f"  QTc (Bazett)   {res['qtc_bazett']:.0f} +/- "
             f"{res['qtc_half_width']:.0f} ms"
             f"      [{res['qtc_low']:.0f} - {res['qtc_high']:.0f}]")
    L.append(f"  QT (uncorrected)  {res['qt_ms']:.0f} ms")
    L.append("")
    head, body = MESSAGES[res["tier"]]
    L.append(f"  {head}")
    for line in _wrap(body, 74):
        L.append(f"  {line}")
    L.append("")
    L.append("-" * 78)
    L.append("How accurate is this?")
    cov = int(calib["coverage"] * 100)
    widths = {t["name"]: t["half_width_ms"] for t in calib["tiers"]}
    L.append(_indent(
        f"This is a screening estimate from a single-lead recording, not a "
        f"diagnosis. The range shown above is calibrated against cardiologist "
        f"measurements and contains the true value about {cov}% of the time."))
    L.append(_indent(
        f"Accuracy depends on the reading: about +/-{widths.get('normal',0):.0f} ms "
        f"in the normal range, but only +/-{widths.get('monitor',0):.0f} to "
        f"+/-{widths.get('refer',0):.0f} ms once readings are elevated — which is "
        f"why an elevated result asks you to repeat it or get a clinical ECG "
        f"rather than relying on the number itself."))
    L.append(_indent(
        "A single-lead reading cannot rule out QT prolongation. A 12-lead ECG "
        "is the definitive test."))
    return "\n".join(L)


def _wrap(s, w):
    out, line = [], ""
    for word in s.split():
        if len(line) + len(word) + 1 > w:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def _indent(s, w=74):
    return "\n".join("  " + l for l in _wrap(s, w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--calib", default=str(DEFAULT_CALIB))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    for f in args.files:
        res, calib = analyse(f, args.ckpt, args.calib)
        print(json.dumps(res, indent=2) if args.json else render(res, calib))
        print()


if __name__ == "__main__":
    main()
