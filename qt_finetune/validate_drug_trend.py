"""
Does the single-lead model track *drug-induced QT change*, not just resting state?

ECGRDVQ (PhysioNet) is a randomised, double-blind, five-period crossover in 22
healthy subjects. Each subject received, in separate periods with a 7-day
washout: dofetilide 500 ug, quinidine sulphate 400 mg, ranolazine 1500 mg,
verapamil 120 mg, and placebo. Triplicate 10-second 12-lead ECGs were extracted
at 16 timepoints per period — one pre-dose (-0.5 h) and 15 post-dose out to 24 h.
5,232 records, each with an expert-adjudicated QT and RR in the clinical table.

This is the closest available proxy for the deployment scenario, and it tests the
claim the absolute-accuracy numbers cannot: that a *personal baseline* removes
most of the model's error, because much of that error is a fixed per-subject
offset that cancels under differencing.

Three questions, in order of importance:

  1. Does the model reproduce the known pharmacological ordering
     (quinidine > dofetilide > ranolazine > verapamil ~ placebo) from lead I alone?
  2. Does baseline-differencing shrink the error, as the offset argument predicts?
  3. Does it recover the time course — onset, peak, washout — per drug?

Endpoint follows thorough-QT convention: QTcF (Fridericia), triplicate-averaged,
expressed as change from each subject's own pre-dose baseline within the same
period (dQTcF), then as placebo-corrected change (ddQTcF).

Usage:
    python qt_finetune/validate_drug_trend.py --data-dir <ecgrdvq> --predict
    python qt_finetune/validate_drug_trend.py --data-dir <ecgrdvq>   # analyse cache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (_SCRIPT_DIR.parent / "models", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from rates import resample_signal, window_len
from qt_objective import Objective
from ecg_resnet1d import ECGResNet1d

DEFAULT_CKPT = _SCRIPT_DIR / "checkpoints" / "best_resnet_lead1_128hz_pinball_tailloss.pt"
PLACEBO = "Placebo"
# expected direction from the trial literature, for a ranking check
EXPECTED_ORDER = ["Quinidine Sulph", "Dofetilide", "Ranolazine", "Verapamil HCL", PLACEBO]


def qtcf(qt_ms, rr_ms):
    """Fridericia correction — the thorough-QT convention, and more stable than
    Bazett at the heart-rate extremes a drug study produces."""
    return qt_ms / np.cbrt(rr_ms / 1000.0)


# --------------------------------------------------------------------------
# stage 1 — inference
# --------------------------------------------------------------------------
def run_predictions(data_dir: Path, ckpt: Path, out_csv: Path, limit=0):
    import wfdb
    meta = pd.read_csv(data_dir / "SCR-002.Clinical.Data.csv")
    meta = meta.dropna(subset=["QT", "RR"])
    if limit:
        keep = sorted(meta.RANDID.unique())[:limit]
        meta = meta[meta.RANDID.isin(keep)]
    print(f"{len(meta)} records, {meta.RANDID.nunique()} subjects, "
          f"{meta.EXTRT.nunique()} treatments")

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = ECGResNet1d(in_channels=ck.get("in_channels", 1),
                        num_outputs=ck.get("num_outputs", 1))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    obj = Objective(ck.get("loss", "mse"),
                    tuple(ck.get("quantiles", (0.5,)))).to(torch.device("cpu"))
    fs_t, L = ck["target_fs"], window_len(ck["target_fs"])
    print(f"model: target_fs={fs_t} Hz, window {L} samples, loss={ck.get('loss')}")

    rows, miss = [], 0
    for n, r in enumerate(meta.itertuples(), 1):
        p = data_dir / "raw" / str(r.RANDID) / str(r.EGREFID)
        if not p.with_suffix(".hea").exists():
            miss += 1
            continue
        try:
            rec = wfdb.rdrecord(str(p))
            names = [s.upper() for s in rec.sig_name]
            ch = names.index("I") if "I" in names else 0
            x = np.nan_to_num(rec.p_signal[:, ch].astype(np.float32))[None, :]  # mV
            xr = resample_signal(x, int(rec.fs), fs_t)
            if xr.shape[-1] < L:
                miss += 1
                continue
            with torch.no_grad():
                lg = model(torch.from_numpy(xr[..., :L])[None])["logits"]
                p50 = float(obj.point(lg).item())
                _lo, hi = obj.interval(lg)
                p95 = float(hi.item()) if hi is not None else np.nan
        except Exception:
            miss += 1
            continue
        rows.append(dict(egrefid=r.EGREFID, subject=r.RANDID, drug=r.EXTRT,
                         tpt=r.TPT, baseline=r.BASELINE,
                         qt_true=r.QT, rr=r.RR, qt_pred=p50, qt_pred_hi=p95))
        if n % 400 == 0:
            print(f"  {n}/{len(meta)} …")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({len(df)} records, {miss} skipped)")
    return df


# --------------------------------------------------------------------------
# stage 2 — analysis
# --------------------------------------------------------------------------
def analyse(df: pd.DataFrame):
    df = df.copy()
    df["qtcf_true"] = qtcf(df.qt_true, df.rr)
    df["qtcf_pred"] = qtcf(df.qt_pred, df.rr)

    # triplicate averaging: the trial's own unit of analysis
    g = (df.groupby(["subject", "drug", "tpt"], as_index=False)
           .agg(qtcf_true=("qtcf_true", "mean"), qtcf_pred=("qtcf_pred", "mean"),
                n=("qtcf_true", "size")))

    # change from each subject's own pre-dose baseline, within the same period
    base = (g[g.tpt < 0].groupby(["subject", "drug"], as_index=False)
              .agg(b_true=("qtcf_true", "mean"), b_pred=("qtcf_pred", "mean")))
    g = g.merge(base, on=["subject", "drug"], how="inner")
    g["d_true"] = g.qtcf_true - g.b_true
    g["d_pred"] = g.qtcf_pred - g.b_pred
    post = g[g.tpt > 0].copy()

    # ---- Q2: what does a personal baseline actually buy? -------------------
    # Three references, weakest to strongest:
    #   (a) none            — absolute QTcF
    #   (b) single pre-dose ECG in the same period
    #   (c) pooled personal baseline: every pre-dose reading the subject has
    #       (one per period). This is the wearable scenario — a baseline
    #       established over days rather than from one recording.
    pool = (g[g.tpt < 0].groupby("subject", as_index=False)
              .agg(pb_true=("qtcf_true", "mean"), pb_pred=("qtcf_pred", "mean"),
                   n_base=("qtcf_true", "size")))
    post = post.merge(pool, on="subject", how="left")
    sig = {
        "absolute QTcF (no baseline)": post.qtcf_pred - post.qtcf_true,
        "minus single pre-dose ECG": post.d_pred - post.d_true,
        "minus pooled personal baseline":
            (post.qtcf_pred - post.pb_pred) - (post.qtcf_true - post.pb_true),
    }
    print("=" * 78)
    print("Q2  What does a personal baseline actually buy?")
    print("=" * 78)
    print(f"  {'reference used':<34} {'MAE':>7} {'bias':>7} {'SD':>7} {'vs abs':>8}")
    base = None
    for lab, s in sig.items():
        mae = s.abs().mean()
        base = mae if base is None else base
        rel = "—" if lab.startswith("absolute") else f"{(1-mae/base)*100:+.0f}%"
        print(f"  {lab:<34} {mae:>7.2f} {s.median():>+7.2f} {s.std():>7.2f} {rel:>8}")

    # The diagnostic that matters: if the error were a fixed per-person offset,
    # differencing would drive the per-subject spread toward zero. Check it.
    off_abs = sig["absolute QTcF (no baseline)"].groupby(post.subject).median()
    off_pool = sig["minus pooled personal baseline"].groupby(post.subject).median()
    print(f"\n  per-subject offset SD: {off_abs.std():.2f} ms absolute  ->  "
          f"{off_pool.std():.2f} ms after pooled differencing "
          f"({post.subject.nunique()} subjects, median "
          f"{pool.n_base.median():.0f} baseline readings each)")
    if off_pool.std() > 0.6 * off_abs.std():
        print("  NOTE: the per-subject component barely moves, so the error is NOT"
              "\n        mainly a fixed personal offset. The MAE gain above comes from"
              "\n        removing a GLOBAL bias, which a single recalibration constant"
              "\n        would also achieve — a personal baseline is not required for it.")

    # ---- Q1: placebo-corrected effect, and the drug ordering ---------------
    pl = (post[post.drug == PLACEBO][["subject", "tpt", "d_true", "d_pred"]]
          .rename(columns={"d_true": "p_true", "d_pred": "p_pred"}))
    dd = post[post.drug != PLACEBO].merge(pl, on=["subject", "tpt"], how="inner")
    dd["dd_true"] = dd.d_true - dd.p_true
    dd["dd_pred"] = dd.d_pred - dd.p_pred

    print("\n" + "=" * 78)
    print("Q1  Placebo-corrected effect (ddQTcF) — mean over subjects, peak timepoint")
    print("=" * 78)
    print(f"  {'drug':<18} {'true peak':>11} {'pred peak':>11} {'at h':>6} "
          f"{'true mean':>10} {'pred mean':>10}")
    summ = []
    for drug in [d for d in EXPECTED_ORDER if d != PLACEBO]:
        s = dd[dd.drug == drug]
        if s.empty:
            continue
        by_t = s.groupby("tpt").agg(t=("dd_true", "mean"), p=("dd_pred", "mean"))
        pk = by_t.t.idxmax()
        summ.append((drug, by_t.t.max(), by_t.p.loc[pk], pk,
                     s.dd_true.mean(), s.dd_pred.mean()))
        print(f"  {drug:<18} {by_t.t.max():>10.1f} {by_t.p.loc[pk]:>10.1f} "
              f"{pk:>6.1f} {s.dd_true.mean():>10.1f} {s.dd_pred.mean():>10.1f}")

    order_true = [x[0] for x in sorted(summ, key=lambda z: -z[1])]
    order_pred = [x[0] for x in sorted(summ, key=lambda z: -z[2])]
    print(f"\n  ranking by true peak: {' > '.join(order_true)}")
    print(f"  ranking by pred peak: {' > '.join(order_pred)}")
    print(f"  ordering reproduced: {'YES' if order_true == order_pred else 'NO'}")

    r = np.corrcoef(dd.dd_pred, dd.dd_true)[0, 1]
    print(f"\n  per-subject-timepoint ddQTcF correlation: r = {r:.3f} "
          f"(n={len(dd)})")

    # ---- can it separate a real effect from placebo at all? ---------------
    print("\n" + "=" * 78)
    print("Effect detection: is each drug distinguishable from placebo?")
    print("=" * 78)
    from scipy import stats
    for drug in [d for d in EXPECTED_ORDER if d != PLACEBO]:
        s = dd[dd.drug == drug]
        if s.empty:
            continue
        # paired over subject-timepoint: is ddQTcF > 0?
        tt = stats.ttest_1samp(s.dd_true, 0)
        tp = stats.ttest_1samp(s.dd_pred, 0)
        print(f"  {drug:<18} true t={tt.statistic:>7.2f} p={tt.pvalue:.2e} | "
              f"pred t={tp.statistic:>7.2f} p={tp.pvalue:.2e}")
    return post, dd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="ECGRDVQ root (contains raw/ and SCR-002.Clinical.Data.csv)")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--cache", default=None, help="predictions CSV")
    ap.add_argument("--predict", action="store_true", help="(re)run inference")
    ap.add_argument("--limit", type=int, default=0, help="first N subjects only")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    cache = Path(args.cache) if args.cache else data_dir / "qt_predictions.csv"
    if args.predict or not cache.exists():
        df = run_predictions(data_dir, Path(args.ckpt), cache, args.limit)
    else:
        df = pd.read_csv(cache)
        print(f"loaded {len(df)} cached predictions from {cache}")
    analyse(df)


if __name__ == "__main__":
    main()
