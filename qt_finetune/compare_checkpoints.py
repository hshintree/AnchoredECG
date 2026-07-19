"""
Score several QT checkpoints against the same human-annotated LUDB pass.

``validate_human_qt.py`` gives the full report for one checkpoint. This runs the
grid: the expensive part (reading records, parsing per-lead cardiologist
annotations) happens ONCE, then every model is evaluated on the identical
windows, so the comparison has no data-order or subset confound. Each model is
resampled to its own target_fs from the same 500 Hz source.

Two tables:
  accuracy  — MAE, bias, the bias-corrected MAE that actually matters, the
              pred~ref slope, and the error inside the two prolonged bins where
              the MSE baselines collapsed.
  clinical  — sensitivity at the QTc thresholds, from the median and (for
              quantile models) from the upper quantile, with false alarms.

Usage:
    python qt_finetune/compare_checkpoints.py --local-dir <ludb>
    python qt_finetune/compare_checkpoints.py --local-dir <ludb> \
        --ckpts qt_finetune/checkpoints/best_resnet_lead1_128hz*.pt
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (_REPO_ROOT / "models", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from rates import resample_signal, window_len
from validate_human_qt import ludb_cases
from qt_objective import Objective
from ecg_resnet1d import ECGResNet1d


def load_ckpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    n_out = ck.get("num_outputs", 1)
    m = ECGResNet1d(in_channels=ck.get("in_channels", 1), num_outputs=n_out)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    obj = Objective(ck.get("loss", "mse"),
                    tuple(ck.get("quantiles", (0.5,)))).to(torch.device("cpu"))
    label = Path(path).stem.replace("best_resnet_lead1_", "")
    return dict(label=label, model=m, obj=obj, fs=ck["target_fs"],
                quantiles=tuple(ck.get("quantiles", (0.5,))),
                val_mae=ck.get("val_mae"), loss=ck.get("loss", "mse"))


def auc(pos_scores, neg_scores):
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    a, b = np.asarray(pos_scores)[:, None], np.asarray(neg_scores)[None, :]
    return float((a > b).mean() + 0.5 * (a == b).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=None,
                    help="checkpoint paths or globs (default: all lead1 resnets)")
    ap.add_argument("--local-dir", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thresholds", default="450,500")
    args = ap.parse_args()

    pats = args.ckpts or [str(_SCRIPT_DIR / "checkpoints" / "best_resnet_lead1_*.pt")]
    paths = sorted({p for pat in pats for p in glob.glob(pat)})
    if not paths:
        raise SystemExit(f"no checkpoints matched {pats}")
    models = [load_ckpt(p) for p in paths]
    print(f"Scoring {len(models)} checkpoint(s) on one LUDB pass:")
    for m in models:
        print(f"  {m['label']:<34} fs={m['fs']:>3} loss={m['loss']:<7} "
              f"val_mae(12SL)={m['val_mae']:.2f} ms")

    # ---- single data pass -------------------------------------------------
    sigs, refs, rrs = [], [], []
    for rec, x, fs, ref, rr, _ in ludb_cases(args.limit, args.local_dir, "lead1"):
        sigs.append((x, fs))
        refs.append(ref)
        rrs.append(rr)
    ref = np.asarray(refs, float)
    rr = np.asarray(rrs, float)
    print(f"\nLUDB: {len(ref)} records   reference QT {ref.mean():.1f} "
          f"+- {ref.std():.1f} ms  [{ref.min():.0f}, {ref.max():.0f}]")

    # ---- inference --------------------------------------------------------
    for m in models:
        pt, lo, hi = [], [], []
        L = window_len(m["fs"])
        for x, fs in sigs:
            xr = resample_signal(x, int(fs), m["fs"])[..., :L]
            with torch.no_grad():
                lg = m["model"](torch.from_numpy(xr)[None])["logits"]
                pt.append(float(m["obj"].point(lg).item()))
                a, b = m["obj"].interval(lg)
                lo.append(float(a.item()) if a is not None else np.nan)
                hi.append(float(b.item()) if b is not None else np.nan)
        m["pred"] = np.asarray(pt)
        m["lo"], m["hi"] = np.asarray(lo), np.asarray(hi)

    # ---- accuracy table ---------------------------------------------------
    print(f"\n{'='*104}\nACCURACY vs cardiologist (LUDB lead I)   "
          f"[human lead-to-lead noise floor ~16.4 ms]\n{'='*104}")
    print(f"{'checkpoint':<34} {'MAE':>7} {'bias':>7} {'bcMAE':>7} {'r':>6} "
          f"{'slope':>6} {'450-500':>12} {'500+':>12} {'cov':>6}")
    n45 = ((ref >= 450) & (ref < 500)).sum()
    n50 = (ref >= 500).sum()
    for m in sorted(models, key=lambda z: z["label"]):
        p = m["pred"]
        bias = np.median(p - ref)
        dc = (p - bias) - ref
        slope = float(np.polyfit(ref, p, 1)[0])
        b45 = (ref >= 450) & (ref < 500)
        b50 = ref >= 500
        cov = ("-" if not np.isfinite(m["hi"]).any() else
               f"{((ref>=m['lo'])&(ref<=m['hi'])).mean()*100:.0f}%")
        print(f"{m['label']:<34} {np.abs(p-ref).mean():>7.2f} {bias:>+7.2f} "
              f"{np.abs(dc).mean():>7.2f} {np.corrcoef(p,ref)[0,1]:>6.3f} "
              f"{slope:>6.2f} {dc[b45].mean():>+7.1f}/{np.abs(dc[b45]).mean():>4.0f} "
              f"{dc[b50].mean():>+7.1f}/{np.abs(dc[b50]).mean():>4.0f} {cov:>6}")
    print(f"{'':<34} {'':>7} {'':>7} {'':>7} {'':>6} {'1.00':>6} "
          f"{'n='+str(n45):>12} {'n='+str(n50):>12}")
    print("  bcMAE = bias-corrected MAE.  450-500 / 500+ columns are "
          "mean error / MAE inside that reference-QT band.")

    # ---- clinical table ---------------------------------------------------
    thrs = [int(t) for t in args.thresholds.split(",")]
    qtc_ref = ref / np.sqrt(rr)
    print(f"\n{'='*104}\nCLINICAL FLAGGING (QTc Bazett, RR measured from the ECG)"
          f"\n{'='*104}")
    hdr = f"{'checkpoint':<34}"
    for t in thrs:
        hdr += f" {'QTc>'+str(t)+' median':>18} {'QTc>'+str(t)+' q-upper':>18}"
    print(hdr)
    for t in thrs:
        print(f"{'':<34}  (n positive = {int((qtc_ref>t).sum())})", end="")
        break
    print()
    for m in sorted(models, key=lambda z: z["label"]):
        p = m["pred"]
        bias = np.median(p - ref)
        qp = (p - bias) / np.sqrt(rr)
        qu = ((m["hi"] - bias) / np.sqrt(rr)) if np.isfinite(m["hi"]).any() else None
        row = f"{m['label']:<34}"
        for t in thrs:
            pos = qtc_ref > t
            tp = int((pos & (qp > t)).sum()); fp = int((~pos & (qp > t)).sum())
            row += f" {tp:>3}/{int(pos.sum()):<3}({fp:>2}FP) {'':>4}"
            if qu is None:
                row += f" {'n/a':>18}"
            else:
                tu = int((pos & (qu > t)).sum()); fu = int((~pos & (qu > t)).sum())
                row += f" {tu:>3}/{int(pos.sum()):<3}({fu:>2}FP) {'':>4}"
        print(row)
    print("\n  ranking AUC (threshold-free, unchanged by any linear recalibration):")
    for m in sorted(models, key=lambda z: z["label"]):
        s = m["pred"] / np.sqrt(rr)
        a = " ".join(f"QTc>{t}: {auc(s[qtc_ref>t], s[qtc_ref<=t]):.3f}" for t in thrs)
        print(f"    {m['label']:<34} {a}")


if __name__ == "__main__":
    main()
