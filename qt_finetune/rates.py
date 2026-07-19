"""Sampling-rate configuration for the Apple-Watch-oriented QT pipeline.

The Apple Watch ECG app emits a fixed signal contract: Lead I, 512 Hz, 30 s,
µV — unchanged across every ECG-capable watch since Series 4. We are optimising
for that input, so the training rates are chosen to be rates the *watch* reaches
with the least possible processing, and the awkward resampling is pushed onto
PTB-XL, which we convert once, offline, under our own control.

    rate     Apple 512 Hz ->        PTB-XL 500 Hz ->     10 s window
    ------   --------------------   ------------------   -----------
    512 Hz   no-op (native)         up   128/125          5120
    128 Hz   exact /4 decimation    down  32/125          1280
    500 Hz   down 125/128           no-op (native)        5000   [legacy baseline]

Both Apple-oriented rates divide a 30 s recording into exactly three 10 s
windows, which is what the inference path uses to produce a per-window spread.

500 Hz is retained only to reproduce the original PTB-XL-native baseline; it is
the one rate that requires resampling the user's watch data at inference time.
"""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

# --- native rates of the two domains ---------------------------------------
PTBXL_FS = 500          # PTB-XL records500/
APPLE_FS = 512          # Apple Watch ECG app, all models Series 4 onward

# --- rates the training pipeline supports ----------------------------------
# 512 : native watch rate, zero resampling of user data  (primary)
# 128 : exact /4 of the watch rate, 4x shorter sequences (ablation)
# 500 : PTB-XL native, legacy baseline only
SUPPORTED_RATES = (512, 128, 500)
DEFAULT_RATE = 512

WINDOW_SECONDS = 10.0   # PTB-XL record length; also the inference window


def window_len(fs: int, seconds: float = WINDOW_SECONDS) -> int:
    """Samples in one analysis window at ``fs``.  500->5000, 512->5120, 128->1280."""
    return int(round(seconds * fs))


def resample_signal(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Polyphase resample along the last axis.

    ``resample_poly`` applies a Kaiser-windowed FIR anti-alias filter sized to
    the lower of the two rates, so downsampling is band-limited correctly and
    upsampling is interpolated without imaging. A no-op when the rates match.

    Args:
        x       : (..., L) signal.
        fs_in   : source rate (Hz).
        fs_out  : target rate (Hz).

    Returns:
        (..., L') array, float32, where L' = round(L * fs_out / fs_in).
    """
    if fs_in == fs_out:
        return np.ascontiguousarray(x, dtype=np.float32)
    g = gcd(int(fs_in), int(fs_out))
    up, down = int(fs_out) // g, int(fs_in) // g
    y = resample_poly(x, up, down, axis=-1)
    return np.ascontiguousarray(y, dtype=np.float32)


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Trim or edge-pad the last axis to exactly ``n`` samples.

    Defensive only — every supported (rate, window) pair divides exactly, so
    this is a no-op in the normal path.
    """
    L = x.shape[-1]
    if L == n:
        return x
    if L > n:
        return np.ascontiguousarray(x[..., :n])
    pad = [(0, 0)] * x.ndim
    pad[-1] = (0, n - L)
    return np.ascontiguousarray(np.pad(x, pad, mode="edge"))


def describe(fs_out: int) -> str:
    """One-line summary of what this target rate implies for both domains."""
    def path(fs_in):
        if fs_in == fs_out:
            return "native (no-op)"
        g = gcd(fs_in, fs_out)
        return f"{fs_out // g}/{fs_in // g}"
    return (f"target {fs_out} Hz | window {window_len(fs_out)} samples "
            f"| PTB-XL {PTBXL_FS}->{fs_out}: {path(PTBXL_FS)} "
            f"| Apple {APPLE_FS}->{fs_out}: {path(APPLE_FS)}")


if __name__ == "__main__":
    for r in SUPPORTED_RATES:
        print(describe(r))
