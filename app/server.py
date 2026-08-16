#!/usr/bin/env python3
"""Local AnchoredECG app server with real QT model inference.

Run from the repo root:

    python3 app/server.py

Then open:

    http://127.0.0.1:8000

The inference endpoint uses:

    qt_finetune/checkpoints/best_resnet_lead1_128hz_pinball_tailloss.pt
"""
from __future__ import annotations

import cgi
import json
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error, request
from urllib.parse import unquote, urlparse

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "app"
MODEL_ROOT = REPO_ROOT / "models"
QT_ROOT = REPO_ROOT / "qt_finetune"
CHECKPOINT = QT_ROOT / "checkpoints" / "best_resnet_lead1_128hz_pinball_tailloss.pt"

for path in (MODEL_ROOT, QT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

TARGET_FS = 128
APPLE_FS = 512
WINDOW_SECONDS = 10.0
WINDOW_LEN = int(TARGET_FS * WINDOW_SECONDS)
DEFAULT_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")

_MODEL = None
_OBJECTIVE = None
_CKPT_META = None


def _json_error(handler, status: int, message: str) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(json.dumps({"error": message}).encode("utf-8"))


def _json_ok(handler, payload: dict) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(json.dumps(payload, allow_nan=False).encode("utf-8"))


def _is_metadata(line: str) -> bool:
    key = line.split(",", 1)[0]
    return any(c.isalpha() for c in key)


def _looks_numeric(value: str) -> bool:
    try:
        float(value.replace(",", "."))
        return True
    except ValueError:
        return False


def parse_apple_or_numeric_text(text: str) -> tuple[np.ndarray, int, dict]:
    """Parse Apple Watch text exports or simple numeric CSV/TXT into uV samples."""
    meta: dict[str, str] = {}
    raw_lines: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if _is_metadata(s):
            key, _, value = s.partition(",")
            meta[key.strip()] = value.strip().strip('"')
        else:
            raw_lines.append(s)

    if not raw_lines:
        raise ValueError("No ECG sample lines found.")

    fs = APPLE_FS
    for value in meta.values():
        if re.search(r"\b(hertz|hz)\b", value, re.I):
            match = _NUM.search(value)
            if match:
                fs = int(round(float(match.group().replace(",", "."))))
                break

    decimal_comma = any(re.fullmatch(r"\s*-?\d+,\d+\s*", s) for s in raw_lines[:30])
    values = []
    for line in raw_lines:
        if decimal_comma and re.fullmatch(r"\s*-?\d+,\d+\s*", line):
            token = line.replace(",", ".")
            values.append(float(token))
            continue
        parts = re.split(r"[\t,; ]+", line)
        nums = [float(part.replace(",", ".")) for part in parts if _looks_numeric(part)]
        if nums:
            values.append(nums[-1])

    if len(values) < 200:
        raise ValueError("At least 200 ECG samples are needed for model inference.")

    signal = np.asarray(values, dtype=np.float64)
    unit = next((v for _, v in meta.items() if re.fullmatch(r"\s*(?:u|U|µ|m|M)?[vV]\s*", v)), "uV")
    if unit.strip().lower() == "mv":
        signal = signal * 1000.0
    elif np.percentile(signal, 95) - np.percentile(signal, 5) < 20:
        # Numeric CSVs are often already mV. The server normalizes to uV here.
        signal = signal * 1000.0

    meta["_decimal_comma"] = str(decimal_comma)
    return signal, fs, meta


def to_128hz_mv(signal_uv: np.ndarray, fs: int) -> np.ndarray:
    x_mv = signal_uv.astype(np.float64) / 1000.0
    if fs == TARGET_FS:
        return x_mv.astype(np.float32)
    if fs == APPLE_FS:
        usable = (len(x_mv) // 4) * 4
        return x_mv[:usable].reshape(-1, 4).mean(axis=1).astype(np.float32)

    duration = len(x_mv) / float(fs)
    old_t = np.linspace(0.0, duration, len(x_mv), endpoint=False)
    new_len = max(1, int(round(duration * TARGET_FS)))
    new_t = np.linspace(0.0, duration, new_len, endpoint=False)
    return np.interp(new_t, old_t, x_mv).astype(np.float32)


def windows_10s(x_mv_128: np.ndarray) -> np.ndarray:
    if len(x_mv_128) < WINDOW_LEN:
        raise ValueError("Recording is shorter than one 10-second model window.")
    n = min(3, len(x_mv_128) // WINDOW_LEN)
    return np.stack([x_mv_128[i * WINDOW_LEN:(i + 1) * WINDOW_LEN] for i in range(n)])[:, None, :]


def load_model():
    global _MODEL, _OBJECTIVE, _CKPT_META
    if _MODEL is not None:
        return _MODEL, _OBJECTIVE, _CKPT_META
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")

    import torch
    from ecg_resnet1d import ECGResNet1d
    from qt_objective import Objective

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    n_out = ckpt.get("num_outputs", 1)
    model = ECGResNet1d(in_channels=ckpt.get("in_channels", 1), num_outputs=n_out)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    objective = Objective(ckpt.get("loss", "mse"), tuple(ckpt.get("quantiles", DEFAULT_QUANTILES))).to(
        torch.device("cpu")
    )

    _MODEL, _OBJECTIVE, _CKPT_META = model, objective, ckpt
    return _MODEL, _OBJECTIVE, _CKPT_META


def parse_apple_pdf_bytes(data: bytes) -> tuple[np.ndarray, int, dict]:
    """Recover the waveform from an Apple Health ECG PDF (vector trace)."""
    import io

    try:
        from apple_pdf import read_apple_pdf
    except ImportError as exc:  # pdfplumber missing
        raise ValueError(
            "PDF support needs pdfplumber. Install it with "
            "`pip install pdfplumber` and restart the server."
        ) from exc

    # Polarity is corrected uniformly in _run_inference (covers text uploads
    # too), so the loader must not also flip — that would double-correct.
    ecg = read_apple_pdf(io.BytesIO(data), correct_polarity=False)
    if ecg.signal_uv.size < 200:
        raise ValueError("Could not recover a waveform from this PDF.")
    meta = {str(k): str(v) for k, v in ecg.meta.items()}
    return ecg.signal_uv.astype(np.float64), int(ecg.fs), meta


def infer_qt(text: str, filename: str) -> dict:
    signal_uv, native_fs, meta = parse_apple_or_numeric_text(text)
    return _run_inference(signal_uv, native_fs, meta, filename)


def infer_qt_pdf(data: bytes, filename: str) -> dict:
    signal_uv, native_fs, meta = parse_apple_pdf_bytes(data)
    return _run_inference(signal_uv, native_fs, meta, filename)


def _dominant_beats(x: np.ndarray, fs: int) -> np.ndarray:
    absx = np.abs(x)
    thr = np.percentile(absx, 90)
    refr = max(1, int(0.28 * fs))
    peaks, i, n = [], 0, len(x)
    while i < n:
        if absx[i] >= thr:
            j = min(n, i + refr)
            peaks.append(i + int(np.argmax(absx[i:j])))
            i = peaks[-1] + refr
        else:
            i += 1
    return np.asarray(peaks, dtype=int)


def qrs_is_upright(signal: np.ndarray, fs: int) -> bool:
    x = np.asarray(signal, dtype=float)
    x = x - np.median(x)
    if len(x) < int(0.5 * fs):
        return True
    beats = _dominant_beats(x, fs)
    if len(beats) >= 3:
        return float(np.median(x[beats])) >= 0.0
    return np.percentile(x, 99.5) >= -np.percentile(x, 0.5)


def apply_qrs_polarity(signal: np.ndarray, fs: int) -> tuple[np.ndarray, bool]:
    if qrs_is_upright(signal, fs):
        return signal, False
    med = float(np.median(signal))
    return (2.0 * med - signal), True


def _run_inference(signal_uv: np.ndarray, native_fs: int, meta: dict,
                   filename: str) -> dict:
    import torch

    # Match the model's upright training domain: an Apple Lead I recorded on the
    # other wrist (or held reversed) has an inverted QRS, which is out of
    # distribution for the amplitude features. Applies to both text and PDF.
    signal_uv, polarity_flipped = apply_qrs_polarity(signal_uv, native_fs)
    meta["polarity_flipped"] = str(polarity_flipped)

    x_mv = to_128hz_mv(signal_uv, native_fs)
    windows = windows_10s(x_mv)
    model, objective, ckpt = load_model()

    with torch.no_grad():
        logits = model(torch.from_numpy(windows))["logits"]
        quantile_values = objective.transform(logits).cpu().numpy()
        point = objective.point(logits).cpu().numpy().reshape(-1)

    quantiles = tuple(float(v) for v in ckpt.get("quantiles", DEFAULT_QUANTILES))
    q50_index = int(np.argmin(np.abs(np.asarray(quantiles) - 0.5)))
    qt_ms = float(np.median(point))
    rr_sec, hr, peaks = estimate_rr(x_mv, TARGET_FS)

    payload = {
        "source": "ml",
        "filename": filename,
        "polarityFlipped": bool(polarity_flipped),
        "model": CHECKPOINT.name,
        "targetFs": TARGET_FS,
        "nativeFs": int(native_fs),
        "sampleCount": int(len(x_mv)),
        "duration": float(len(x_mv) / TARGET_FS),
        "qtMs": qt_ms,
        "qtLowerMs": float(np.median(quantile_values[:, 0])),
        "qtUpperMs": float(np.median(quantile_values[:, -1])),
        "rrSec": rr_sec,
        "hr": hr,
        "quantiles": list(quantiles),
        "pointQuantile": quantiles[q50_index],
        "windows": [
            {
                "startSec": i * WINDOW_SECONDS,
                "qtMs": float(point[i]),
                "quantilesMs": [float(v) for v in quantile_values[i]],
            }
            for i in range(len(point))
        ],
        "waveform": {
            "fs": TARGET_FS,
            "unit": "mV",
            "values": [round(float(v), 6) for v in x_mv[:3 * WINDOW_LEN]],
        },
        "peaks": peaks[:80],
        "meta": {k: str(v) for k, v in meta.items()},
    }
    payload["geminiSummary"] = gemini_clinical_summary(payload)
    return payload


def corrected_qt_fridericia(qt_ms: float, rr_sec: float | None) -> float | None:
    if not rr_sec or rr_sec <= 0:
        return None
    return float(qt_ms / (rr_sec ** (1.0 / 3.0)))


def qtc_tier(qtc_upper_ms: float | None) -> str:
    if qtc_upper_ms is None:
        return "unknown"
    if qtc_upper_ms <= 435:
        return "normal"
    if qtc_upper_ms <= 445:
        return "monitor"
    return "refer"


def fallback_clinical_summary(payload: dict) -> str:
    qtc = corrected_qt_fridericia(payload["qtMs"], payload.get("rrSec"))
    upper = corrected_qt_fridericia(payload["qtUpperMs"], payload.get("rrSec"))
    if qtc is None:
        return "Gemini summary unavailable because a stable RR interval could not be estimated."
    tier = qtc_tier(upper)
    tier_text = {"normal": "normal-range", "monitor": "monitor-range", "refer": "referral-range"}.get(
        tier, "unclassified"
    )
    hr_text = f", average HR {payload['hr']:.0f} bpm" if payload.get("hr") else ""
    return (
        f"Single-lead Apple Watch ECG, {payload['duration']:.0f} seconds{hr_text}. "
        f"Estimated Fridericia QTc is {qtc:.0f} ms with a {tier_text} flag. "
        "Research preview only; not a diagnostic QT measurement."
    )


def gemini_clinical_summary(payload: dict) -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    fallback = fallback_clinical_summary(payload)
    if not api_key:
        return fallback

    qtc = corrected_qt_fridericia(payload["qtMs"], payload.get("rrSec"))
    lower = corrected_qt_fridericia(payload["qtLowerMs"], payload.get("rrSec"))
    upper = corrected_qt_fridericia(payload["qtUpperMs"], payload.get("rrSec"))
    tier = qtc_tier(upper)
    hr_text = f"{payload['hr']:.0f} bpm" if payload.get("hr") else "not available"
    qtc_text = f"{qtc:.0f} ms" if qtc is not None else "not available"
    range_text = f"{lower:.0f}-{upper:.0f} ms" if lower is not None and upper is not None else "not available"
    prompt = (
        "Write one concise, non-diagnostic clinical-style summary for a wearable ECG research prototype. "
        "Do not give medical advice. Include that this is a single-lead Apple Watch ECG research preview. "
        "Use cautious language and mention clinician review for monitor or referral range. "
        f"Duration: {payload['duration']:.1f} seconds. "
        f"Heart rate: {hr_text}. "
        f"QT interval: {payload['qtMs']:.0f} ms. "
        f"Fridericia QTc: {qtc_text}. "
        f"Model interval: {range_text}. "
        f"Risk tier: {tier}. "
        f"Model: {payload['model']}."
    )
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 120},
        }
    ).encode("utf-8")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        parts = data["candidates"][0]["content"]["parts"]
        text = " ".join(part.get("text", "") for part in parts).strip()
        return text or fallback
    except (KeyError, IndexError, TimeoutError, OSError, error.URLError, error.HTTPError, json.JSONDecodeError):
        return fallback


def estimate_rr(x: np.ndarray, fs: int) -> tuple[float | None, float | None, list[int]]:
    y = x - moving_average(x, max(3, int(fs * 0.25)))
    y = moving_average(y, max(3, int(fs * 0.02)))
    abs_y = np.abs(y)
    threshold = float(np.percentile(abs_y, 88))
    refractory = int(fs * 0.28)
    peaks: list[int] = []
    for i in range(1, len(y) - 1):
        if abs_y[i] >= threshold and abs_y[i] >= abs_y[i - 1] and abs_y[i] >= abs_y[i + 1]:
            if not peaks or i - peaks[-1] > refractory:
                peaks.append(i)
            elif abs_y[i] > abs_y[peaks[-1]]:
                peaks[-1] = i
    peaks = [p for p in peaks if fs * 0.35 < p < len(y) - fs * 0.5]
    if len(peaks) < 3:
        return None, None, peaks
    rr = float(np.median(np.diff(peaks)) / fs)
    return rr, float(60.0 / rr), peaks


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, kernel, mode="same")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("", "/"):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/infer":
            _json_error(self, 404, "Unknown endpoint.")
            return
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                },
            )
            field = form["file"] if "file" in form else None
            if field is None or not getattr(field, "file", None):
                _json_error(self, 400, "Missing uploaded file.")
                return
            raw = field.file.read()
            if isinstance(raw, str):
                raw = raw.encode("utf-8", errors="replace")
            filename = unquote(getattr(field, "filename", "") or "uploaded ECG")
            # Route PDFs to the vector-trace parser; everything else is text.
            is_pdf = raw[:5] == b"%PDF-" or filename.lower().endswith(".pdf")
            if is_pdf:
                _json_ok(self, infer_qt_pdf(raw, filename))
            else:
                _json_ok(self, infer_qt(raw.decode("utf-8", errors="replace"),
                                        filename))
        except Exception as exc:
            _json_error(self, 500, str(exc))


def main() -> None:
    host, port = parse_host_port()
    httpd = bind_server(host, port)
    bound_host, bound_port = httpd.server_address
    print(f"AnchoredECG ML app: http://{bound_host}:{bound_port}")
    print(f"Checkpoint: {CHECKPOINT}")
    with httpd:
        httpd.serve_forever()


def parse_host_port() -> tuple[str, int]:
    host, port = "127.0.0.1", int(os.environ.get("PORT", "8000"))
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--host" and i + 1 < len(args):
            host = args[i + 1]
        elif arg == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
        elif arg.startswith("--port="):
            port = int(arg.split("=", 1)[1])
        elif arg.startswith("--host="):
            host = arg.split("=", 1)[1]
    return host, port


def bind_server(host: str, port: int) -> ThreadingHTTPServer:
    last_error = None
    for candidate in range(port, port + 20):
        try:
            return ThreadingHTTPServer((host, candidate), Handler)
        except OSError as exc:
            last_error = exc
            if exc.errno != 48:
                raise
            print(f"Port {candidate} is in use, trying {candidate + 1}...")
    raise last_error or OSError("No available port found.")


if __name__ == "__main__":
    main()
