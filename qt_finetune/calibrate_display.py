"""
Fit the user-facing display calibration against human annotations.

Everything the user sees is derived from constants computed here, on LUDB's
cardiologist-annotated lead I. Three tiers, split by the model's upper quantile:

    normal   q95 <= 435          no action, show a tight interval
    monitor  435 < q95 <= 445     repeat the reading, mention it if symptomatic
    refer    q95 > 445            recommend a clinical 12-lead ECG

Per tier we fit two constants:

  bias         the median signed error of the q50 estimate inside that tier, so
               the displayed number is unbiased *for the tier the user is in*.
               (A single global bias is wrong: the model understates by ~31 ms
               among the prolonged and is near-unbiased among the normal.)

  half_width   split-conformal: the `coverage` quantile of the absolute residual
               after bias correction. The displayed interval is number +- this.
               It is NOT the model's own q05/q95 spread, which covers only ~62%
               out of domain; it is measured against human labels.

The resulting widths (~18 ms normal, ~54-61 ms elevated) *are* the error
disclosure — the interval communicates the uncertainty in place of a footnote
that would have to quote a single misleading number.

Fit on one half of LUDB, coverage verified on the held-out half, then refit on
all 200 for the shipped constants.

Usage:
    python qt_finetune/calibrate_display.py --local-dir <ludb>
    python qt_finetune/calibrate_display.py --local-dir <ludb> --coverage 0.95
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (_SCRIPT_DIR.parent / "models", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from rates import resample_signal, window_len
from validate_human_qt import ludb_cases
from qt_objective import Objective
from ecg_resnet1d import ECGResNet1d

DEFAULT_CKPT = _SCRIPT_DIR / "checkpoints" / "best_resnet_lead1_128hz_pinball_tailloss.pt"
DEFAULT_OUT = _SCRIPT_DIR / "qt_calibration.json"

# Tier boundaries on the model's upper quantile (QTc, Bazett).
#
# normal/monitor at 435: on LUDB no truly-prolonged record has q95 <= 435, so
#   the "normal" bucket is clean. Do not raise this without re-checking that.
# monitor/refer at 445: the five severe (QTc>500) records sit at q95 =
#   448, 468, 482, 499, 549 — a 445 boundary puts all five in "refer".
#   Raising it to 470 drops two of them into "monitor" and only reduces the
#   referral rate from 30% to 14%. Not worth it.
NORMAL_MAX = 435.0
MONITOR_MAX = 445.0


def predict_all(ckpt, local_dir, limit=0):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = ECGResNet1d(in_channels=ck.get("in_channels", 1),
                        num_outputs=ck.get("num_outputs", 1))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    obj = Objective(ck.get("loss", "mse"),
                    tuple(ck.get("quantiles", (0.5,)))).to(torch.device("cpu"))
    fs_t, L = ck["target_fs"], window_len(ck["target_fs"])

    ref, rr, p50, p95 = [], [], [], []
    for _rec, x, fs, r_, rr_, _ in ludb_cases(limit, local_dir, "lead1"):
        xr = resample_signal(x, int(fs), fs_t)[..., :L]
        with torch.no_grad():
            lg = model(torch.from_numpy(xr)[None])["logits"]
            p50.append(float(obj.point(lg).item()))
            _lo, hi = obj.interval(lg)
            p95.append(float(hi.item()) if hi is not None else float("nan"))
        ref.append(r_)
        rr.append(rr_)
    return (np.asarray(ref), np.asarray(rr), np.asarray(p50), np.asarray(p95),
            ck["target_fs"])


def fit(qtc_true, qtc50, qtc95, coverage, idx=None):
    """Return per-tier {bias, half_width} fitted on `idx` (default: all)."""
    idx = np.arange(len(qtc_true)) if idx is None else idx
    tiers = []
    for name, lo, hi in TIERS:
        m = np.ones(len(idx), bool)
        if lo is not None:
            m &= qtc95[idx] > lo
        if hi is not None:
            m &= qtc95[idx] <= hi
        if m.sum() < 5:
            tiers.append(dict(name=name, q95_lo=lo, q95_hi=hi, bias_ms=0.0,
                              half_width_ms=float("nan"), n_fit=int(m.sum())))
            continue
        sel = idx[m]
        bias = float(np.median(qtc50[sel] - qtc_true[sel]))
        resid = np.abs((qtc50[sel] - bias) - qtc_true[sel])
        tiers.append(dict(name=name, q95_lo=lo, q95_hi=hi,
                          bias_ms=bias,
                          half_width_ms=float(np.quantile(resid, coverage)),
                          n_fit=int(m.sum())))
    return tiers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--local-dir", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--coverage", type=float, default=0.90)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--normal-max", type=float, default=NORMAL_MAX,
                    help="q95 QTc at/below which a reading is called normal")
    ap.add_argument("--monitor-max", type=float, default=MONITOR_MAX,
                    help="q95 QTc above which we recommend a clinical ECG")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    global TIERS
    TIERS = [("normal", None, args.normal_max),
             ("monitor", args.normal_max, args.monitor_max),
             ("refer", args.monitor_max, None)]

    ref, rr, p50, p95, fs_t = predict_all(args.ckpt, args.local_dir, args.limit)
    g = np.sqrt(rr)
    gbias = float(np.median(p50 - ref))          # global 12SL -> single-lead offset
    qtc_true = ref / g
    qtc50 = (p50 - gbias) / g
    qtc95 = (p95 - gbias) / g

    # --- honest check: fit on half, measure coverage on the other half -------
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ref))
    cal, tst = perm[:len(perm) // 2], perm[len(perm) // 2:]
    held = fit(qtc_true, qtc50, qtc95, args.coverage, cal)
    print(f"Split-half check (fit n={len(cal)}, verify n={len(tst)}), "
          f"target coverage {args.coverage*100:.0f}%")
    print(f"  {'tier':>8} {'bias':>7} {'half-width':>11} {'held-out cov':>13} "
          f"{'mean err':>9} {'true>450':>10} {'>500':>5}")
    for t in held:
        m = np.ones(len(tst), bool)
        if t["q95_lo"] is not None:
            m &= qtc95[tst] > t["q95_lo"]
        if t["q95_hi"] is not None:
            m &= qtc95[tst] <= t["q95_hi"]
        if m.sum() == 0:
            continue
        sel = tst[m]
        shown = qtc50[sel] - t["bias_ms"]
        cov = ((qtc_true[sel] >= shown - t["half_width_ms"]) &
               (qtc_true[sel] <= shown + t["half_width_ms"])).mean()
        print(f"  {t['name']:>8} {t['bias_ms']:>+7.1f} {t['half_width_ms']:>10.1f} "
              f"{cov*100:>12.0f}% {np.abs(shown-qtc_true[sel]).mean():>9.1f} "
              f"{int((qtc_true[sel]>450).sum()):>6}/{m.sum():<3} "
              f"{int((qtc_true[sel]>500).sum()):>5}")

    # --- shipped constants: refit on everything ------------------------------
    tiers = fit(qtc_true, qtc50, qtc95, args.coverage)
    for t in tiers:
        m = np.ones(len(ref), bool)
        if t["q95_lo"] is not None:
            m &= qtc95 > t["q95_lo"]
        if t["q95_hi"] is not None:
            m &= qtc95 <= t["q95_hi"]
        t["n_total"] = int(m.sum())
        t["frac_true_over_450"] = float((qtc_true[m] > 450).mean()) if m.sum() else 0.0
        t["n_true_over_500"] = int((qtc_true[m] > 500).sum())

    cfg = dict(checkpoint=Path(args.ckpt).name, target_fs=int(fs_t),
               normal_max=args.normal_max, monitor_max=args.monitor_max,
               global_bias_ms=gbias, coverage=args.coverage,
               n_calibration=int(len(ref)), source="LUDB lead I (.i annotations)",
               tiers=tiers)
    Path(args.out).write_text(json.dumps(cfg, indent=2))
    print(f"\nShipped constants (refit on all {len(ref)}) -> {args.out}")
    for t in tiers:
        print(f"  {t['name']:>8}: bias {t['bias_ms']:+6.1f} ms, "
              f"+-{t['half_width_ms']:.0f} ms, n={t['n_total']}, "
              f"{t['frac_true_over_450']*100:.0f}% truly >450, "
              f"{t['n_true_over_500']} severe")


if __name__ == "__main__":
    main()
