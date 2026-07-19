"""Reader for Apple Watch ECG exports (the inference-side input).

Apple's Health export writes one ``.txt`` per recording: a short ``key,value``
metadata header, then a single column of Lead I voltages — one sample per line,
512 Hz, 30 s, in µV.

Two things about this format bite hard if you use a plain ``read_csv``:

1.  **Locale.** Exports from a European locale use the *decimal comma*, so
    ``-137,098`` is one value, -137.098 µV, not two columns. A naive parse
    silently yields two integer columns of nonsense. We detect this from the
    data lines themselves: a data line holds exactly one value, so a comma
    appearing in one can only be a decimal separator.

2.  **Units.** Apple exports µV; PTB-XL (and therefore the trained model) is in
    mV. That is a factor of 1000 — feeding raw µV to a model trained on mV is
    ~1400x off-scale and produces garbage, not merely degraded, predictions.

Everything here is unit/format correctness, not augmentation: the waveform is
resampled and rescaled, never corrupted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rates import APPLE_FS, resample_signal, window_len

_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


@dataclass
class AppleECG:
    """One Apple Watch recording, as exported."""
    signal_uv: np.ndarray                  # (L,) Lead I, µV, at ``fs``
    fs: int                                # native sample rate (512 Hz)
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return len(self.signal_uv) / self.fs

    @property
    def classification(self) -> str:
        return self.meta.get("Classification", "")

    def __repr__(self) -> str:
        return (f"AppleECG({len(self.signal_uv)} samples, {self.duration_s:.2f}s "
                f"@ {self.fs}Hz, classification={self.classification!r})")


def _is_metadata(line: str) -> bool:
    """Metadata lines have a non-numeric key before the first comma."""
    key = line.split(",", 1)[0]
    return any(c.isalpha() for c in key)


def read_apple_ecg(path: str | Path) -> AppleECG:
    """Parse an Apple Watch ECG export.

    Handles both decimal-comma and decimal-dot locales, and tolerates localised
    metadata keys (the sample rate is recovered by pattern, not by key name).
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    meta, raw_lines = {}, []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if _is_metadata(s):
            k, _, v = s.partition(",")
            meta[k.strip()] = v.strip().strip('"')
        else:
            raw_lines.append(s)

    if not raw_lines:
        raise ValueError(f"{path}: no data lines found")

    # --- locale: a data line holds one value, so a comma must be the decimal ---
    decimal_comma = any("," in s for s in raw_lines)
    vals = []
    for s in raw_lines:
        t = s.replace(",", ".") if decimal_comma else s.replace(",", "")
        try:
            vals.append(float(t))
        except ValueError:
            continue
    signal = np.asarray(vals, dtype=np.float64)
    if signal.size == 0:
        raise ValueError(f"{path}: data lines present but none parsed as numbers")

    # --- sample rate: find the metadata value that looks like "512 hertz" ---
    fs = APPLE_FS
    for v in meta.values():
        if re.search(r"\b(hertz|hz)\b", v, re.I):
            m = _NUM.search(v)
            if m:
                fs = int(round(float(m.group().replace(",", "."))))
                break

    # --- unit: normalise to µV (Apple uses µV; accept mV defensively) ---
    unit = next((v for k, v in meta.items() if re.fullmatch(r"\s*[µumM]?V\s*", v)), "µV")
    if unit.strip().lower() == "mv":
        signal = signal * 1000.0

    meta.setdefault("_decimal_comma", str(decimal_comma))
    return AppleECG(signal_uv=signal, fs=fs, meta=meta)


def to_windows(ecg: AppleECG, target_fs: int, seconds: float = 10.0,
               stride_s: float | None = None) -> np.ndarray:
    """Resample to ``target_fs``, convert µV -> mV, and cut analysis windows.

    Args:
        ecg       : parsed recording.
        target_fs : model input rate (512 or 128).
        seconds   : window length; must match the training window (10 s).
        stride_s  : hop between windows; defaults to non-overlapping.

    Returns:
        (n_windows, 1, L) float32 in **mV**, ready for ``ECGResNet1d``.
        A 30 s recording gives exactly 3 windows at both supported rates.
    """
    x_mv = ecg.signal_uv / 1000.0                                  # µV -> mV
    x = resample_signal(x_mv, ecg.fs, target_fs)                   # (L,)
    L = window_len(target_fs, seconds)
    hop = L if stride_s is None else int(round(stride_s * target_fs))
    if x.size < L:
        raise ValueError(f"recording is {x.size/target_fs:.1f}s at {target_fs}Hz, "
                         f"shorter than the {seconds}s window")
    starts = range(0, x.size - L + 1, hop)
    return np.stack([x[s:s + L] for s in starts])[:, None, :].astype(np.float32)


if __name__ == "__main__":
    import sys
    from rates import SUPPORTED_RATES
    for p in sys.argv[1:]:
        ecg = read_apple_ecg(p)
        print(f"\n{Path(p).name}\n  {ecg}")
        print(f"  decimal_comma={ecg.meta.get('_decimal_comma')}  "
              f"range=[{ecg.signal_uv.min():.1f}, {ecg.signal_uv.max():.1f}] µV")
        for fs in SUPPORTED_RATES:
            w = to_windows(ecg, fs)
            print(f"  -> {fs:>4} Hz: windows {w.shape}  "
                  f"std={w.std():.4f} mV  absmax={np.abs(w).max():.3f} mV")
