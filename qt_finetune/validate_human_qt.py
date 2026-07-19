"""
Validate a trained QT ResNet against HUMAN (cardiologist) annotations.

Everything else in this pipeline is trained and scored against GE 12SL algorithm
labels on PTB-XL. That measures how well we mimic 12SL, not how well we measure
QT. This script is the independent check.

Two benchmarks, in order of relevance to a Lead-I model:

  ludb  Lobachevsky University Database — 200 records, 12-lead, 500 Hz, 10 s,
        mV, with cardiologist P/QRS/T delineation **for each lead separately**.
        The `.i` annotation is Lead I, so this is lead-matched, and the record
        geometry (500 Hz / 10 s / mV) is identical to a PTB-XL training window.
        THIS IS THE PRIMARY BENCHMARK.

  qtdb  PhysioNet QT Database — 105 records, 250 Hz, 15 min, ~30-100 manually
        annotated beats each (.q1c). The classic QT benchmark, but be aware:
        **no record in qtdb contains Lead I** (channel 0 is ECG1 x57, MLII x12,
        CM5 x12, D3 x9, V4 x7, V5 x4, ...). Scoring a Lead-I model here is a
        cross-lead robustness probe, not a like-for-like validation.

Annotation grammar (both databases):
    ( p )  ( N )  ( t )        onset / peak / offset for P, QRS, T
Some qtdb beats omit the T onset, giving  ( N ) t )  — handled either way.
    QT = (offset ')' following the 't' peak)  -  (onset '(' preceding the beat)

Reported metrics:
    MAE / bias / bias-corrected MAE / RMSE / Pearson r / limits of agreement.

**Read the bias and the bias-corrected MAE together.** PTB-XL's QT_Int_Global is
a *global* 12SL measurement (earliest onset to latest offset across all leads),
whereas a single-lead human QT is measured within one lead. Global QT is
systematically longer, so a constant positive bias here is expected and is
correctable by recalibration; it is the residual scatter, not the offset, that
says whether the model learned to measure QT.

Usage:
    # data path only, no checkpoint needed — works before training finishes
    python qt_finetune/validate_human_qt.py --dry-run --limit 40

    # score a trained checkpoint (rate is read from the checkpoint)
    python qt_finetune/validate_human_qt.py \
        --ckpt qt_finetune/checkpoints/best_resnet_lead1_512hz.pt

    python qt_finetune/validate_human_qt.py --db qtdb \
        --ckpt qt_finetune/checkpoints/best_resnet_lead1_128hz.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (_REPO_ROOT / "models", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from rates import resample_signal, window_len, WINDOW_SECONDS

# Beat annotation symbols (MIT/AAMI beat codes).
BEAT_SYMS = set("NLRBAaJSVrFejnE/fQ?")
QT_MIN_MS, QT_MAX_MS = 200.0, 700.0
QRS_ONSET_MAX_MS = 120.0     # '(' must be this close before the beat marker
T_OFFSET_MAX_MS = 600.0      # ')' must be this close after the beat marker

DBS = {
    "ludb": dict(pn_dir="ludb/1.0.1/data", fs=500, lead_matched=True),
    "qtdb": dict(pn_dir="qtdb", fs=250, lead_matched=False),
}


# --------------------------------------------------------------------------
# annotation -> per-beat QT
# --------------------------------------------------------------------------
def beat_qts(ann, fs: float) -> list[tuple[int, int, float]]:
    """Extract (qrs_onset, t_offset, qt_ms) for every fully-delineated beat.

    Walks the symbol stream: for each beat marker, the QRS onset is the nearest
    preceding '(' and the T offset is the first ')' after the following 't'.
    """
    sym, samp = list(ann.symbol), np.asarray(ann.sample)
    out = []
    for i, s in enumerate(sym):
        if s not in BEAT_SYMS:
            continue

        # QRS onset: nearest preceding '('
        j = i - 1
        while j >= 0 and sym[j] not in "(":
            if sym[j] in BEAT_SYMS:      # ran into the previous beat
                j = -1
                break
            j -= 1
        if j < 0:
            continue
        qon = int(samp[j])
        if (samp[i] - qon) / fs * 1000.0 > QRS_ONSET_MAX_MS:
            continue

        # T peak after this beat, then its closing ')'
        k = i + 1
        t_peak = -1
        while k < len(sym):
            if sym[k] in BEAT_SYMS:      # hit the next beat first
                break
            if sym[k] == "t":
                t_peak = k
                break
            k += 1
        if t_peak < 0:
            continue
        m = t_peak + 1
        while m < len(sym) and sym[m] != ")":
            if sym[m] in BEAT_SYMS:
                m = len(sym)
                break
            m += 1
        if m >= len(sym):
            continue
        toff = int(samp[m])

        qt = (toff - qon) / fs * 1000.0
        if not (QT_MIN_MS <= qt <= QT_MAX_MS):
            continue
        if (toff - samp[i]) / fs * 1000.0 > T_OFFSET_MAX_MS:
            continue
        out.append((qon, toff, qt))
    return out


# --------------------------------------------------------------------------
# record loading
# --------------------------------------------------------------------------
def _read(rec, pn_dir, local_dir, ext=None, sampfrom=0, sampto=None):
    import wfdb
    if local_dir:
        p = str(Path(local_dir) / rec)
        return (wfdb.rdann(p, ext) if ext
                else wfdb.rdrecord(p, sampfrom=sampfrom, sampto=sampto))
    return (wfdb.rdann(rec, ext, pn_dir=pn_dir) if ext
            else wfdb.rdrecord(rec, pn_dir=pn_dir, sampfrom=sampfrom, sampto=sampto))


def _rr_seconds(ann, fs):
    """Median RR interval (s) from the beat markers of an annotation."""
    b = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMS])
    if len(b) < 3:
        return float("nan")
    return float(np.median(np.diff(b)) / fs)


def ludb_cases(limit, local_dir, leads_cfg):
    """Yield (rec, signal_mV, fs, ref_qt_ms, rr_s, per_lead_qts)."""
    import wfdb
    recs = [r.split("/")[-1] for r in wfdb.get_record_list("ludb/1.0.1")]
    if limit:
        recs = recs[:limit]
    pn = DBS["ludb"]["pn_dir"]
    for rec in recs:
        try:
            r = _read(rec, pn, local_dir)
            names = [n.lower() for n in r.sig_name]
            # QT from every lead the cardiologist delineated (annotation noise floor)
            per_lead, ann_i = {}, None
            for nm in names:
                try:
                    a = _read(rec, pn, local_dir, ext=nm)
                except Exception:
                    continue
                if nm == "i":
                    ann_i = a
                q = beat_qts(a, r.fs)
                if q:
                    per_lead[nm] = float(np.median([x[2] for x in q]))
            if "i" not in per_lead or ann_i is None:
                continue
            rr = _rr_seconds(ann_i, r.fs)
            sig = r.p_signal.T.astype(np.float32)          # (12, 5000) mV
            if leads_cfg == "all12":
                x, ref = sig, float(np.median(list(per_lead.values())))
            else:
                x, ref = sig[[0]], per_lead["i"]           # Lead I
            yield rec, np.nan_to_num(x), r.fs, ref, rr, per_lead
        except Exception as e:
            print(f"  [skip] {rec}: {type(e).__name__}: {e}")


def qtdb_cases(limit, local_dir, leads_cfg):
    """Yield 10 s windows around the manually annotated segment of each record."""
    import wfdb
    recs = wfdb.get_record_list("qtdb")
    if limit:
        recs = recs[:limit]
    pn, fs = DBS["qtdb"]["pn_dir"], DBS["qtdb"]["fs"]
    W = int(WINDOW_SECONDS * fs)
    for rec in recs:
        try:
            ann = _read(rec, pn, local_dir, ext="q1c")
            beats = beat_qts(ann, fs)
            if len(beats) < 3:
                continue
            lo, hi = beats[0][0], beats[-1][1]
            # clamp to the record length — the annotated span can end at the
            # very last sample, and wfdb rejects sampto > sig_len
            sig_len = (wfdb.rdheader(str(Path(local_dir) / rec)).sig_len
                       if local_dir else wfdb.rdheader(rec, pn_dir=pn).sig_len)
            off = max(0, lo - W)
            r = _read(rec, pn, local_dir, sampfrom=off,
                      sampto=min(hi + W, sig_len))
            sig = np.nan_to_num(r.p_signal.T.astype(np.float32))
            for start in range(lo - off, max(lo - off + 1, hi - off - W + 1), W):
                if start < 0 or start + W > sig.shape[1]:
                    continue
                inwin = [b[2] for b in beats
                         if start <= b[0] - off and b[1] - off <= start + W]
                if len(inwin) < 2:
                    continue
                x = sig[[0], start:start + W]              # annotated channel
                if leads_cfg == "all12":
                    x = np.repeat(x, 12, axis=0)           # crude channel fill
                yield (f"{rec}@{start}", x, fs, float(np.median(inwin)),
                       _rr_seconds(ann, fs), {})
        except Exception as e:
            print(f"  [skip] {rec}: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def report(pred, ref, label):
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    d = pred - ref
    bias = float(np.median(d))
    dc = d - bias
    r = float(np.corrcoef(pred, ref)[0, 1]) if len(pred) > 2 else float("nan")
    print(f"\n=== {label}  (n={len(pred)}) ===")
    print(f"  reference QT   : {ref.mean():7.1f} +- {ref.std():5.1f} ms  "
          f"[{ref.min():.0f}, {ref.max():.0f}]")
    print(f"  predicted QT   : {pred.mean():7.1f} +- {pred.std():5.1f} ms  "
          f"[{pred.min():.0f}, {pred.max():.0f}]")
    print(f"  MAE            : {np.abs(d).mean():7.2f} ms")
    print(f"  bias (median)  : {bias:+7.2f} ms   <- constant offset, recalibratable")
    print(f"  bias-corr. MAE : {np.abs(dc).mean():7.2f} ms   <- the honest number")
    print(f"  RMSE           : {np.sqrt((d**2).mean()):7.2f} ms")
    print(f"  Pearson r      : {r:7.3f}")
    print(f"  95% LoA        : [{np.percentile(dc,2.5):+.1f}, "
          f"{np.percentile(dc,97.5):+.1f}] ms (bias-corrected)")
    print(f"  within 10/20 ms: {(np.abs(dc)<=10).mean()*100:.1f}% / "
          f"{(np.abs(dc)<=20).mean()*100:.1f}%")
    # a model must beat predicting the population mean
    base = np.abs(ref - np.median(ref)).mean()
    print(f"  [baseline] predict-the-median MAE : {base:.2f} ms  "
          f"-> model is {base/max(np.abs(dc).mean(),1e-9):.2f}x better")

    # --- where do the errors live? MSE regressors shrink toward the mean, and
    #     the long-QT tail is exactly the clinically important part. ---
    print(f"\n  error by reference QT (bias removed):")
    print(f"  {'bin (ms)':>13} {'n':>5} {'mean err':>9} {'MAE':>7}")
    for lo, hi in ((0, 350), (350, 400), (400, 450), (450, 500), (500, 9999)):
        m = (ref >= lo) & (ref < hi)
        if m.sum() == 0:
            continue
        flag = "  <-- shrunk toward the mean" if abs(dc[m].mean()) > 15 else ""
        print(f"  {lo:>5}-{hi:<7} {m.sum():>5} {dc[m].mean():>+9.1f} "
              f"{np.abs(dc[m]).mean():>7.1f}{flag}")
    slope = float(np.polyfit(ref, pred, 1)[0])
    print(f"  dispersion: predicted std {pred.std():.1f} / reference std "
          f"{ref.std():.1f} = {pred.std()/ref.std()*100:.0f}%")
    print(f"  slope pred~ref = {slope:.2f}  (1.00 = faithful, <1 = compressed range)")


def report_qtc(pred, ref, rr, corrections=("bazett", "fridericia")):
    """QTc derived from the MEASURED RR — never a predicted one.

    RR is trivially and accurately recoverable from the ECG, so composing
    QTc = f(QT_pred, RR_measured) keeps the error dominated by the QT model and
    leaves the choice of correction formula open. Training directly on QTc would
    instead force the network to learn heart rate implicitly and would bake in
    one formula.
    """
    pred, ref, rr = (np.asarray(a, float) for a in (pred, ref, rr))
    ok = np.isfinite(rr) & (rr > 0)
    pred, ref, rr = pred[ok], ref[ok], rr[ok]
    if len(ref) < 5:
        return
    bias = np.median(pred - ref)
    pc = pred - bias
    hr = 60.0 / rr

    F = {"bazett": lambda q, r: q / np.sqrt(r),
         "fridericia": lambda q, r: q / np.cbrt(r),
         "framingham": lambda q, r: q + 154.0 * (1.0 - r)}

    print(f"\n=== QTc from measured RR  (HR {hr.min():.0f}-{hr.max():.0f} bpm, "
          f"median {np.median(hr):.0f}) ===")
    qt_mae = np.abs(pc - ref).mean()
    for name in corrections:
        f = F[name]
        qr, qp = f(ref, rr), f(pc, rr)
        mae = np.abs(qp - qr).mean()
        print(f"  {name:>11}: QTc MAE {mae:6.2f} ms  "
              f"(QT MAE {qt_mae:.2f} -> amplified {mae/qt_mae:.2f}x)")
        for lo, hi in ((0, 70), (70, 90), (90, 300)):
            m = (hr >= lo) & (hr < hi)
            if m.sum() < 3:
                continue
            print(f"      HR {lo:>3}-{hi:<3} (n={m.sum():>3}): "
                  f"QT {np.abs(pc-ref)[m].mean():5.1f} -> "
                  f"QTc {np.abs(qp-qr)[m].mean():5.1f} ms")

    # clinical flagging — the number that decides whether this is deployable
    qr, qp = F["bazett"](ref, rr), F["bazett"](pc, rr)
    print(f"\n  clinical flagging (Bazett):")
    for thr in (450, 500):
        pos = qr > thr
        if pos.sum() == 0:
            continue
        tp = int((pos & (qp > thr)).sum())
        fp = int((~pos & (qp > thr)).sum())
        sens = tp / pos.sum() * 100
        warn = "   <-- MISSES THE CASES THAT MATTER" if sens < 60 else ""
        print(f"    QTc>{thr}: caught {tp}/{int(pos.sum())} ({sens:.0f}% sens), "
              f"{fp} false alarms{warn}")
        # threshold-free: is the information there even if the scale is off?
        s = qp
        p_, n_ = s[pos], s[~pos]
        auc = float((p_[:, None] > n_[None, :]).mean()
                    + 0.5 * (p_[:, None] == n_[None, :]).mean())
        print(f"             ranking AUC {auc:.3f} — discrimination is "
              f"{'usable' if auc > 0.75 else 'weak'}, so a lower operating "
              f"threshold trades false alarms for sensitivity")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", choices=list(DBS), default="ludb")
    ap.add_argument("--ckpt", default=None, help="trained checkpoint (.pt)")
    ap.add_argument("--local-dir", default=None,
                    help="local copy of the database; omit to stream from PhysioNet")
    ap.add_argument("--limit", type=int, default=0, help="first N records only")
    ap.add_argument("--target-fs", type=int, default=None,
                    help="override; normally read from the checkpoint")
    ap.add_argument("--leads", default=None,
                    help="override; normally read from the checkpoint")
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise the data path and report the human annotation "
                         "noise floor, without running a model")
    ap.add_argument("--save-csv", default=None)
    args = ap.parse_args()

    target_fs, leads, model = args.target_fs, args.leads, None
    if not args.dry_run:
        if not args.ckpt:
            ap.error("--ckpt is required unless --dry-run")
        import torch
        from ecg_resnet1d import ECGResNet1d
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        target_fs = target_fs or ck.get("target_fs")
        leads = leads or ck.get("leads", "lead1")
        in_ch = ck.get("in_channels", 1)
        if target_fs is None:
            ap.error("checkpoint has no target_fs; pass --target-fs")
        model = ECGResNet1d(in_channels=in_ch)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        print(f"Loaded {args.ckpt}")
        print(f"  leads={leads}  in_channels={in_ch}  target_fs={target_fs} Hz  "
              f"(trained val MAE {ck.get('val_mae', float('nan')):.2f} ms vs 12SL)")
    else:
        target_fs = target_fs or 512
        leads = leads or "lead1"
        print(f"DRY RUN — data path only. leads={leads} target_fs={target_fs} Hz")

    db = DBS[args.db]
    if not db["lead_matched"]:
        print(f"\n!! {args.db} contains NO Lead I records. Treat this as a "
              f"cross-lead robustness probe, not a like-for-like validation.")
    gen = (ludb_cases if args.db == "ludb" else qtdb_cases)(
        args.limit, args.local_dir, leads)

    ids, preds, refs, rrs, spreads = [], [], [], [], []
    L = window_len(target_fs)
    import torch
    for rec, x, fs, ref, rr, per_lead in gen:
        xr = resample_signal(x, int(fs), int(target_fs))
        if xr.shape[-1] < L:
            continue
        xr = xr[..., :L]
        if model is not None:
            with torch.no_grad():
                p = float(model(torch.from_numpy(xr)[None])["logits"].item())
            preds.append(p)
        ids.append(rec)
        refs.append(ref)
        rrs.append(rr)
        if len(per_lead) >= 3:
            spreads.append(float(np.std(list(per_lead.values()))))
        print(f"  {rec:>12}  ref={ref:6.1f} ms" +
              (f"  pred={preds[-1]:6.1f}  err={preds[-1]-ref:+6.1f}" if model else ""))

    if not refs:
        print("\nNo usable records — check --local-dir / network access.")
        return

    if spreads:
        print(f"\n--- human annotation noise floor (LUDB) ---")
        print(f"  QT std across the 12 separately-annotated leads: "
              f"median {np.median(spreads):.1f} ms")
        print(f"  No single-lead model should be expected to beat this by much.")

    if model is None:
        refs = np.asarray(refs)
        print(f"\n=== DRY RUN: reference QT over {len(refs)} records ===")
        print(f"  mean {refs.mean():.1f} +- {refs.std():.1f} ms  "
              f"[{refs.min():.0f}, {refs.max():.0f}]")
        print("  Data path OK. Re-run with --ckpt to score a model.")
    else:
        report(preds, refs, f"{args.db} / leads={leads} / {target_fs} Hz")
        report_qtc(preds, refs, rrs)

    if args.save_csv:
        import csv
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["record", "ref_qt_ms", "pred_qt_ms"])
            for i, rec in enumerate(ids):
                w.writerow([rec, refs[i], preds[i] if model else ""])
        print(f"\nWrote {args.save_csv}")


if __name__ == "__main__":
    main()
