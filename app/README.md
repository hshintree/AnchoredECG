# AnchoredECG QT App

Prototype for estimating QT/QTc from an uploaded single-lead ECG waveform.

For the real ML path, run the local server from the repo root:

```bash
python3 app/server.py
```

Then open:

```text
http://127.0.0.1:8000
```

The server uses `qt_finetune/checkpoints/best_resnet_lead1_128hz_pinball_tailloss.pt`.
Opening `index.html` directly still works, but it uses the browser-only heuristic fallback unless the local server is running.

The waveform display is formatted as three 10-second ECG strips, matching the current recording format used in this repo.

QTc interpretation uses the selected adult range:

- Men: normal <= 450 ms, borderline 451-480 ms.
- Women: normal <= 460 ms, borderline 461-480 ms.
- Both: abnormally short < 390 ms; QTc above 500 ms is flagged as prolonged.

## Supported Uploads

The importer accepts `.csv` or `.txt` files with numeric ECG samples. It detects common time and voltage columns automatically.

Recommended shape:

```csv
time,voltage_mV
0.00000,0.012
0.00195,0.014
0.00391,0.019
```

If no time column is present, the app assumes 512 Hz sampling.

## Apple Watch Path

Without Apple HealthKit access through an Apple developer account, the practical path is user-initiated export/import. Apple Watch ECGs are most commonly shared from Health as PDFs, while true waveform CSV access usually comes from a third-party export workflow or a full Health export. A production version should support:

- CSV upload for raw waveform exports.
- PDF upload with waveform extraction as a convenience layer.
- Later HealthKit import once a developer account and user consent flow are available.

The current prototype implements the CSV/TXT path and a demo signal.

## Clinical Boundary

The QT estimate is heuristic and intended for research UI validation. It is not a diagnostic measurement and should not be used for clinical decisions.
