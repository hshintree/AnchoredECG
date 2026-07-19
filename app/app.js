const els = {
  fileInput: document.getElementById("fileInput"),
  dropzone: document.getElementById("dropzone"),
  demoButton: document.getElementById("demoButton"),
  clearButton: document.getElementById("clearButton"),
  formulaSelect: document.getElementById("formulaSelect"),
  sexSelect: document.getElementById("sexSelect"),
  fileName: document.getElementById("fileName"),
  fileMeta: document.getElementById("fileMeta"),
  modelName: document.getElementById("modelName"),
  modelMeta: document.getElementById("modelMeta"),
  statusTitle: document.getElementById("statusTitle"),
  qualityPill: document.getElementById("qualityPill"),
  qtValue: document.getElementById("qtValue"),
  qtSub: document.getElementById("qtSub"),
  qtcValue: document.getElementById("qtcValue"),
  qtcSub: document.getElementById("qtcSub"),
  qtcCard: document.getElementById("qtcCard"),
  hrValue: document.getElementById("hrValue"),
  intervalCard: document.getElementById("intervalCard"),
  intervalTitle: document.getElementById("intervalTitle"),
  intervalText: document.getElementById("intervalText"),
  qualityTitle: document.getElementById("qualityTitle"),
  qualityText: document.getElementById("qualityText"),
  rPeaks: document.getElementById("rPeaks"),
  qOnset: document.getElementById("qOnset"),
  tEnd: document.getElementById("tEnd"),
  rrInterval: document.getElementById("rrInterval"),
  canvas: document.getElementById("waveCanvas"),
};

let currentAnalysis = null;

function parseDelimited(text) {
  const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (lines.length < 8) {
    throw new Error("The file does not contain enough ECG samples.");
  }

  const delimiter = detectDelimiter(lines.slice(0, 10));
  const rows = lines.map((line) => splitLine(line, delimiter));
  const firstNumeric = rows.findIndex((row) => row.some((cell) => Number.isFinite(parseNumber(cell))));
  const header = firstNumeric > 0 ? rows[firstNumeric - 1].map((h) => h.toLowerCase()) : [];
  const dataRows = rows.slice(firstNumeric).map((row) => row.map(parseNumber));
  const columnCount = Math.max(...dataRows.map((row) => row.length));

  const columns = Array.from({ length: columnCount }, (_, columnIndex) =>
    dataRows.map((row) => row[columnIndex]).filter(Number.isFinite)
  );

  const timeIndex = pickTimeColumn(columns, header);
  const voltageIndex = pickVoltageColumn(columns, timeIndex, header);
  if (voltageIndex === -1) {
    throw new Error("A numeric voltage column could not be identified.");
  }

  const samples = [];
  const sourceRows = dataRows.filter((row) => Number.isFinite(row[voltageIndex]));
  for (let i = 0; i < sourceRows.length; i += 1) {
    const y = sourceRows[i][voltageIndex];
    const t = timeIndex >= 0 && Number.isFinite(sourceRows[i][timeIndex]) ? sourceRows[i][timeIndex] : i;
    samples.push({ t, y });
  }

  if (samples.length < 200) {
    throw new Error("At least 200 ECG samples are needed for a QT estimate.");
  }

  return normalizeTiming(samples);
}

function detectDelimiter(lines) {
  const candidates = [",", "\t", ";"];
  return candidates
    .map((delimiter) => ({
      delimiter,
      score: lines.reduce((sum, line) => sum + splitLine(line, delimiter).length, 0),
    }))
    .sort((a, b) => b.score - a.score)[0].delimiter;
}

function splitLine(line, delimiter) {
  if (delimiter === "\t") return line.split("\t");
  const cells = [];
  let value = "";
  let quoted = false;
  for (const char of line) {
    if (char === "\"") quoted = !quoted;
    else if (char === delimiter && !quoted) {
      cells.push(value.trim());
      value = "";
    } else value += char;
  }
  cells.push(value.trim());
  return cells;
}

function parseNumber(value) {
  if (value == null) return NaN;
  const cleaned = String(value).replace(/"/g, "").replace(/[^\d.+\-eE]/g, "");
  if (!cleaned || cleaned === "." || cleaned === "-" || cleaned === "+") return NaN;
  return Number(cleaned);
}

function pickTimeColumn(columns, header) {
  const headerIndex = header.findIndex((h) => /\b(time|timestamp|seconds?|sec|ms)\b/.test(h));
  if (headerIndex >= 0) return headerIndex;

  let best = -1;
  let bestScore = 0;
  columns.forEach((col, index) => {
    if (col.length < 20) return;
    const diffs = col.slice(1).map((value, i) => value - col[i]).filter(Number.isFinite);
    const positive = diffs.filter((d) => d > 0).length / diffs.length;
    const medianStep = median(diffs.filter((d) => d > 0));
    const score = positive * (medianStep > 0 ? 1 : 0);
    if (score > bestScore) {
      best = index;
      bestScore = score;
    }
  });
  return bestScore > 0.92 ? best : -1;
}

function pickVoltageColumn(columns, timeIndex, header) {
  const voltageHeader = header.findIndex((h, i) => i !== timeIndex && /\b(voltage|mv|uv|ecg|lead|sample)\b/.test(h));
  if (voltageHeader >= 0) return voltageHeader;

  let best = -1;
  let bestScore = -Infinity;
  columns.forEach((col, index) => {
    if (index === timeIndex || col.length < 200) return;
    const spread = percentile(col, 95) - percentile(col, 5);
    const absMedian = Math.abs(median(col));
    const score = spread - absMedian * 0.03;
    if (spread > 0 && score > bestScore) {
      best = index;
      bestScore = score;
    }
  });
  return best;
}

function normalizeTiming(samples) {
  const times = samples.map((s) => s.t);
  const diffs = times.slice(1).map((value, i) => value - times[i]).filter((d) => Number.isFinite(d) && d > 0);
  let dt = median(diffs);
  let fs = 512;

  if (Number.isFinite(dt) && dt > 0) {
    if (dt > 2) fs = 1000 / dt;
    else fs = 1 / dt;
  }

  if (!Number.isFinite(fs) || fs < 40 || fs > 2000) {
    fs = 512;
    dt = 1 / fs;
  }

  return {
    fs,
    y: samples.map((s) => s.y),
    duration: samples.length / fs,
  };
}

function analyzeSignal(raw, name) {
  const fs = raw.fs;
  let y = raw.y.slice();
  const spread = percentile(y, 95) - percentile(y, 5);
  if (spread > 20) y = y.map((value) => value / 1000);

  const cleaned = bandShape(y, fs);
  const peaks = detectRPeaks(cleaned, fs);
  if (peaks.length < 3) {
    throw new Error("R peaks were not stable enough for QT estimation.");
  }

  const rrSamples = median(peaks.slice(1).map((peak, i) => peak - peaks[i]));
  const hr = 60 * fs / rrSamples;
  const beat = medianBeat(cleaned, peaks, fs);
  const landmarks = detectQtLandmarks(beat, fs);
  const qtMs = (landmarks.tEnd - landmarks.qOnset) * 1000 / fs;
  const rrSec = rrSamples / fs;
  const qtc = correctQt(qtMs, rrSec, els.formulaSelect.value);
  const quality = qualityGrade(peaks, cleaned, fs, landmarks, qtMs);

  return {
    name,
    source: "heuristic",
    fs,
    duration: raw.duration,
    sampleCount: raw.y.length,
    cleaned,
    peaks,
    rrSamples,
    hr,
    beat,
    landmarks,
    qtMs,
    qtc,
    rrSec,
    quality,
  };
}

function bandShape(values, fs) {
  const baselineWindow = Math.max(5, Math.round(fs * 0.22));
  const smoothWindow = Math.max(3, Math.round(fs * 0.012));
  const baseline = movingAverage(values, baselineWindow);
  const centered = values.map((value, index) => value - baseline[index]);
  return movingAverage(centered, smoothWindow);
}

function detectRPeaks(values, fs) {
  const abs = values.map(Math.abs);
  const threshold = percentile(abs, 88);
  const refractory = Math.round(fs * 0.28);
  const candidates = [];

  for (let i = 1; i < values.length - 1; i += 1) {
    if (abs[i] >= threshold && abs[i] >= abs[i - 1] && abs[i] >= abs[i + 1]) {
      candidates.push(i);
    }
  }

  const peaks = [];
  for (const candidate of candidates) {
    const last = peaks[peaks.length - 1];
    if (last == null || candidate - last > refractory) {
      peaks.push(candidate);
    } else if (abs[candidate] > abs[last]) {
      peaks[peaks.length - 1] = candidate;
    }
  }

  return peaks.filter((peak) => peak > fs * 0.35 && peak < values.length - fs * 0.5);
}

function medianBeat(values, peaks, fs) {
  const before = Math.round(fs * 0.28);
  const after = Math.round(fs * 0.62);
  const beats = peaks
    .filter((peak) => peak - before >= 0 && peak + after < values.length)
    .map((peak) => values.slice(peak - before, peak + after));

  const length = before + after;
  const medianValues = Array.from({ length }, (_, index) => median(beats.map((beat) => beat[index])));
  const rIndex = before;
  const rSign = medianValues[rIndex] >= 0 ? 1 : -1;
  return {
    y: medianValues.map((value) => value * rSign),
    rIndex,
    beatCount: beats.length,
  };
}

function detectQtLandmarks(beat, fs) {
  const y = beat.y;
  const r = beat.rIndex;
  const d = derivative(y);
  const noise = Math.max(0.0001, percentile(d.slice(0, Math.max(4, r - Math.round(fs * 0.14))).map(Math.abs), 80));

  let qOnset = Math.max(0, r - Math.round(fs * 0.07));
  for (let i = r - Math.round(fs * 0.015); i > r - Math.round(fs * 0.16) && i > 2; i -= 1) {
    if (Math.abs(d[i]) < noise * 1.25 && Math.abs(d[i - 1]) < noise * 1.25) {
      qOnset = i;
      break;
    }
  }

  const searchStart = r + Math.round(fs * 0.12);
  const searchEnd = Math.min(y.length - 2, r + Math.round(fs * 0.52));
  let tPeak = searchStart;
  for (let i = searchStart; i < searchEnd; i += 1) {
    if (y[i] > y[tPeak]) tPeak = i;
  }

  const tailBaseline = median(y.slice(Math.min(y.length - 1, r + Math.round(fs * 0.5))));
  const tAmplitude = Math.max(0.0001, y[tPeak] - tailBaseline);
  let tEnd = Math.min(y.length - 1, tPeak + Math.round(fs * 0.18));
  for (let i = tPeak + Math.round(fs * 0.04); i < Math.min(y.length - 2, r + Math.round(fs * 0.58)); i += 1) {
    const closeToBaseline = Math.abs(y[i] - tailBaseline) < tAmplitude * 0.12;
    const flat = Math.abs(d[i]) < noise * 1.5;
    if (closeToBaseline && flat) {
      tEnd = i;
      break;
    }
  }

  return { qOnset, r, tPeak, tEnd };
}

function correctQt(qtMs, rrSec, formula) {
  if (formula === "bazett") return qtMs / Math.sqrt(rrSec);
  if (formula === "framingham") return qtMs + 154 * (1 - rrSec);
  return qtMs / Math.cbrt(rrSec);
}

function interpretQtc(qtcMs, sex) {
  const rounded = Math.round(qtcMs);
  const normalUpper = sex === "female" ? 460 : 450;
  const borderlineLower = normalUpper + 1;
  const sexLabel = sex === "female" ? "women" : "men";

  if (rounded < 390) {
    return {
      level: "abnormal",
      title: "Abnormally short QTc",
      text: `${rounded} ms is below 390 ms, which is outside the reference range for both adult groups. This is not a diagnosis; confirm with a clinician-grade ECG.`,
    };
  }

  if (rounded <= normalUpper) {
    return {
      level: "normal",
      title: "QTc in normal range",
      text: `${rounded} ms is within the selected adult ${sexLabel} range of <= ${normalUpper} ms.`,
    };
  }

  if (rounded <= 480) {
    return {
      level: "abnormal",
      title: "Borderline QTc",
      text: `${rounded} ms is in the borderline range for adult ${sexLabel}: ${borderlineLower}-480 ms.`,
    };
  }

  if (rounded > 500) {
    return {
      level: "abnormal",
      title: "Prolonged QTc",
      text: `${rounded} ms is above 500 ms, a range associated with higher risk and needing clinical review.`,
    };
  }

  return {
    level: "abnormal",
    title: "Elevated QTc",
    text: `${rounded} ms is above the borderline range and below 500 ms. This should be reviewed on a clinician-grade ECG.`,
  };
}

function qualityGrade(peaks, values, fs, landmarks, qtMs) {
  const rr = peaks.slice(1).map((peak, i) => (peak - peaks[i]) / fs);
  const rrCv = std(rr) / Math.max(0.001, mean(rr));
  const plausibleQt = qtMs >= 240 && qtMs <= 560;
  const enoughBeats = peaks.length >= 5;
  const ordered = landmarks.qOnset < landmarks.r && landmarks.r < landmarks.tEnd;
  const score = Number(enoughBeats) + Number(rrCv < 0.12) + Number(plausibleQt) + Number(ordered);

  if (score >= 4) {
    return {
      level: "good",
      title: "Readable single-lead waveform",
      text: `Beat-to-beat timing was consistent across ${peaks.length} detected complexes.`,
    };
  }
  if (score >= 2) {
    return {
      level: "caution",
      title: "Manual review recommended",
      text: `The estimate used ${peaks.length} complexes; motion noise or T-wave shape may affect the endpoint.`,
    };
  }
  return {
    level: "poor",
    title: "Low confidence estimate",
    text: "The waveform does not provide a stable QT endpoint.",
  };
}

function renderAnalysis(analysis) {
  currentAnalysis = analysis;
  const formulaName = els.formulaSelect.selectedOptions[0].textContent;
  const interpretation = interpretQtc(analysis.qtc, els.sexSelect.value);
  els.fileName.textContent = analysis.name;
  els.fileMeta.textContent = `${analysis.sampleCount.toLocaleString()} samples · ${analysis.fs.toFixed(0)} Hz · ${analysis.duration.toFixed(1)} s`;
  els.statusTitle.textContent = analysis.source === "ml" ? "ML QT interval estimated" : "QT interval estimated";
  els.qualityPill.textContent = analysis.quality.level === "good" ? "Readable" : analysis.quality.level === "caution" ? "Review" : "Low confidence";
  els.qualityPill.className = `quality-pill ${analysis.quality.level}`;
  els.qtValue.textContent = Math.round(analysis.qtMs);
  els.qtSub.textContent = analysis.source === "ml" ? "ML median, milliseconds" : "milliseconds";
  els.qtcValue.textContent = Math.round(analysis.qtc);
  els.qtcSub.textContent = `${formulaName} QTc`;
  els.qtcCard.className = `metric-card accent-qtc ${interpretation.level}`;
  els.hrValue.textContent = Math.round(analysis.hr);
  els.modelName.textContent = analysis.source === "ml" ? analysis.modelName : "Heuristic fallback";
  els.modelMeta.textContent = analysis.source === "ml"
    ? `${analysis.modelMeta} · QT interval ${Math.round(analysis.qtLowerMs)}-${Math.round(analysis.qtUpperMs)} ms`
    : "Browser-only estimate; start the local ML server for checkpoint inference.";
  els.intervalCard.className = `interval-card ${interpretation.level}`;
  els.intervalTitle.textContent = interpretation.title;
  els.intervalText.textContent = interpretation.text;
  els.qualityTitle.textContent = analysis.quality.title;
  els.qualityText.textContent = analysis.quality.text;
  els.rPeaks.textContent = analysis.peaks.length;
  els.qOnset.textContent = formatRelativeMs((analysis.landmarks.qOnset - analysis.landmarks.r) * 1000 / analysis.fs);
  els.tEnd.textContent = formatRelativeMs((analysis.landmarks.tEnd - analysis.landmarks.r) * 1000 / analysis.fs);
  els.rrInterval.textContent = `${Math.round(analysis.rrSec * 1000)} ms`;
  drawWaveform(analysis);
}

function drawWaveform(analysis) {
  const canvas = els.canvas;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  if (!analysis || !analysis.cleaned.length) {
    drawGrid(ctx, w, h);
    ctx.fillStyle = "#5d6f6c";
    ctx.font = "700 24px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Upload a waveform to view three 10-second ECG strips", w / 2, h / 2);
    return;
  }

  drawGrid(ctx, w, h);
  const strips = 3;
  const secondsPerStrip = 10;
  const padX = 54;
  const padTop = 28;
  const stripGap = 18;
  const stripHeight = (h - padTop * 2 - stripGap * (strips - 1)) / strips;
  const samplesPerStrip = Math.round(analysis.fs * secondsPerStrip);
  const globalLow = percentile(analysis.cleaned, 2);
  const globalHigh = percentile(analysis.cleaned, 98);
  const globalAmp = Math.max(0.001, globalHigh - globalLow);
  const markerOffsets = {
    q: analysis.landmarks.qOnset - analysis.landmarks.r,
    t: analysis.landmarks.tEnd - analysis.landmarks.r,
  };

  for (let strip = 0; strip < strips; strip += 1) {
    const start = strip * samplesPerStrip;
    const end = Math.min(analysis.cleaned.length, start + samplesPerStrip);
    const top = padTop + strip * (stripHeight + stripGap);
    const bottom = top + stripHeight;
    const mid = top + stripHeight / 2;
    const usableHeight = stripHeight * 0.72;
    const xAt = (sampleIndex) => padX + ((sampleIndex - start) / samplesPerStrip) * (w - padX * 2);
    const yAt = (value) => mid - ((value - (globalLow + globalHigh) / 2) / globalAmp) * usableHeight;

    drawStripBackground(ctx, padX, top, w - padX * 2, stripHeight, strip + 1);

    if (start >= analysis.cleaned.length) {
      ctx.fillStyle = "#8aa09b";
      ctx.font = "700 18px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No samples in this interval", w / 2, mid + 6);
      continue;
    }

    ctx.strokeStyle = "#087f75";
    ctx.lineWidth = 2.6;
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let index = start; index < end; index += 1) {
      const x = xAt(index);
      const y = yAt(analysis.cleaned[index]);
      if (index === start) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    const stripPeaks = analysis.peaks.filter((peak) => peak >= start && peak < end);
    stripPeaks.forEach((peak) => {
      ctx.fillStyle = "#2d6cdf";
      ctx.beginPath();
      ctx.arc(xAt(peak), yAt(analysis.cleaned[peak]), 3.8, 0, Math.PI * 2);
      ctx.fill();
    });

    const markerPeak = stripPeaks[Math.floor(stripPeaks.length / 2)];
    if (markerPeak != null) {
      const q = markerPeak + markerOffsets.q;
      const t = markerPeak + markerOffsets.t;
      if (q >= start && t < end) {
        drawStripMarker(ctx, xAt(q), top + 12, bottom - 12, "#d85b4c", "Q");
        drawStripMarker(ctx, xAt(t), top + 12, bottom - 12, "#087f75", "T");
        drawQtBracket(ctx, xAt(q), xAt(t), bottom - 20, `${Math.round(analysis.qtMs)} ms`);
      }
    }
  }
}

function drawGrid(ctx, w, h) {
  ctx.fillStyle = "#fffefe";
  ctx.fillRect(0, 0, w, h);
  for (let x = 0; x <= w; x += 24) {
    ctx.strokeStyle = x % 120 === 0 ? "#f0bbb2" : "#f8ded9";
    ctx.lineWidth = x % 120 === 0 ? 1.3 : 0.8;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y <= h; y += 24) {
    ctx.strokeStyle = y % 120 === 0 ? "#f0bbb2" : "#f8ded9";
    ctx.lineWidth = y % 120 === 0 ? 1.3 : 0.8;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
}

function drawStripBackground(ctx, x, y, width, height, stripNumber) {
  const ctxY = Math.round(y) + 0.5;
  ctx.strokeStyle = "#d9e7e2";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(x, ctxY);
  ctx.lineTo(x + width, ctxY);
  ctx.moveTo(x, y + height + 0.5);
  ctx.lineTo(x + width, y + height + 0.5);
  ctx.stroke();

  ctx.fillStyle = "#5d6f6c";
  ctx.font = "800 15px Inter, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(`${(stripNumber - 1) * 10}-${stripNumber * 10}s`, x + 4, y + 20);
}

function drawStripMarker(ctx, x, top, bottom, color, label) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 7]);
  ctx.beginPath();
  ctx.moveTo(x, top);
  ctx.lineTo(x, bottom);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = color;
  ctx.font = "800 14px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(label, x, top + 14);
}

function drawQtBracket(ctx, qX, tX, y, label) {
  ctx.strokeStyle = "#132322";
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  ctx.moveTo(qX, y);
  ctx.lineTo(tX, y);
  ctx.moveTo(qX, y - 8);
  ctx.lineTo(qX, y + 8);
  ctx.moveTo(tX, y - 8);
  ctx.lineTo(tX, y + 8);
  ctx.stroke();

  ctx.fillStyle = "#132322";
  ctx.font = "800 17px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(label, (qX + tX) / 2, y - 12);
}

function loadText(text, name) {
  els.modelName.textContent = "Checking ML server...";
  els.modelMeta.textContent = "Trying the 128 Hz checkpoint path.";
  els.statusTitle.textContent = "Running ML inference";
  els.qualityPill.textContent = "ML pending";
  els.qualityPill.className = "quality-pill";
  els.fileName.textContent = name;
  els.fileMeta.textContent = "Waiting for the local checkpoint server.";

  requestMlInference(text, name)
    .then((payload) => renderAnalysis(analysisFromMl(payload, name)))
    .catch((error) => showError(`ML server unavailable: ${error.message}`));
}

function showError(message) {
  currentAnalysis = null;
  els.statusTitle.textContent = "ML inference failed";
  els.qualityPill.textContent = "Check server";
  els.qualityPill.className = "quality-pill poor";
  els.fileMeta.textContent = message;
  els.modelName.textContent = "Inference unavailable";
  els.modelMeta.textContent = "Start the local server and retry the upload.";
  els.qualityTitle.textContent = "No ML result";
  els.qualityText.textContent = message;
  els.qtValue.textContent = "--";
  els.qtcCard.className = "metric-card accent-qtc";
  els.qtcValue.textContent = "--";
  els.hrValue.textContent = "--";
  els.intervalCard.className = "interval-card";
  els.intervalTitle.textContent = "No QTc available";
  els.intervalText.textContent = "Upload an ECG to compare QTc with the selected adult range.";
  els.rPeaks.textContent = "--";
  els.qOnset.textContent = "--";
  els.tEnd.textContent = "--";
  els.rrInterval.textContent = "--";
  drawWaveform(null);
}

function clearAnalysis() {
  currentAnalysis = null;
  els.fileInput.value = "";
  els.fileName.textContent = "No waveform loaded";
  els.fileMeta.textContent = "Awaiting ECG samples";
  els.modelName.textContent = "128 Hz checkpoint";
  els.modelMeta.textContent = "Run the local ML server before uploading.";
  els.statusTitle.textContent = "Ready for ECG upload";
  els.qualityPill.textContent = "No signal";
  els.qualityPill.className = "quality-pill";
  els.qtValue.textContent = "--";
  els.qtcCard.className = "metric-card accent-qtc";
  els.qtcValue.textContent = "--";
  els.hrValue.textContent = "--";
  els.intervalCard.className = "interval-card";
  els.intervalTitle.textContent = "No QTc available";
  els.intervalText.textContent = "Upload an ECG to compare QTc with the selected adult range.";
  els.qualityTitle.textContent = "No ECG loaded";
  els.qualityText.textContent = "Upload a waveform or load the demo ECG.";
  els.rPeaks.textContent = "--";
  els.qOnset.textContent = "--";
  els.tEnd.textContent = "--";
  els.rrInterval.textContent = "--";
  drawWaveform(null);
}

function demoCsv() {
  const fs = 512;
  const duration = 30;
  const rows = ["time,voltage_mV"];
  for (let i = 0; i < fs * duration; i += 1) {
    const t = i / fs;
    const phase = t % 0.92;
    const value =
      gaussian(phase, 0.18, 0.025, 0.08) -
      gaussian(phase, 0.34, 0.012, 0.16) +
      gaussian(phase, 0.37, 0.014, 1.05) -
      gaussian(phase, 0.405, 0.012, 0.24) +
      gaussian(phase, 0.65, 0.075, 0.26) +
      Math.sin(t * Math.PI * 2 * 0.23) * 0.025 +
      (Math.random() - 0.5) * 0.018;
    rows.push(`${t.toFixed(5)},${value.toFixed(5)}`);
  }
  return rows.join("\n");
}

function gaussian(x, mu, sigma, amplitude) {
  return amplitude * Math.exp(-0.5 * ((x - mu) / sigma) ** 2);
}

function movingAverage(values, windowSize) {
  const half = Math.floor(windowSize / 2);
  const result = new Array(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    sum += values[i];
    if (i - windowSize >= 0) sum -= values[i - windowSize];
    const start = Math.max(0, i - windowSize + 1);
    result[i - half] = sum / (i - start + 1);
  }
  for (let i = 0; i < half; i += 1) result[i] = result[half] ?? values[i];
  for (let i = values.length - half; i < values.length; i += 1) result[i] = result[values.length - half - 1] ?? values[i];
  return result;
}

function percentile(values, p) {
  const sorted = values.filter(Number.isFinite).slice().sort((a, b) => a - b);
  if (!sorted.length) return NaN;
  const index = (sorted.length - 1) * (p / 100);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function median(values) {
  return percentile(values, 50);
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function std(values) {
  const avg = mean(values);
  return Math.sqrt(mean(values.map((value) => (value - avg) ** 2)));
}

function derivative(values) {
  return values.map((value, index) => (index === 0 ? 0 : value - values[index - 1]));
}

function formatRelativeMs(value) {
  const rounded = Math.round(value);
  if (rounded < 0) return `${Math.abs(rounded)} ms before R`;
  if (rounded > 0) return `${rounded} ms after R`;
  return "at R";
}

async function requestMlInference(text, name) {
  const endpoints = endpointCandidates();
  let lastError = "";
  for (const endpoint of endpoints) {
    const form = new FormData();
    form.append("file", new Blob([text], { type: "text/plain" }), name);
    try {
      const response = await fetch(endpoint, { method: "POST", body: form });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      return payload;
    } catch (error) {
      lastError = error.message;
    }
  }
  throw new Error(lastError || "ML server unavailable");
}

function endpointCandidates() {
  const localPorts = Array.from({ length: 20 }, (_, index) => `http://127.0.0.1:${8000 + index}/api/infer`);
  if (location.protocol === "file:") return localPorts;
  return ["/api/infer", ...localPorts];
}

function analysisFromMl(payload, fallbackName) {
  const fs = payload.waveform.fs;
  const cleaned = payload.waveform.values.map(Number);
  const rrSec = Number.isFinite(payload.rrSec) && payload.rrSec > 0 ? payload.rrSec : 1;
  const hr = Number.isFinite(payload.hr) && payload.hr > 0 ? payload.hr : 60 / rrSec;
  const qOffset = Math.round(fs * 0.04);
  const tOffset = Math.max(qOffset + 1, Math.round((payload.qtMs / 1000 - 0.04) * fs));
  const firstPeak = payload.peaks && payload.peaks.length ? payload.peaks[0] : Math.round(fs * 0.5);
  const rIndex = Math.min(Math.max(firstPeak, qOffset), Math.max(qOffset, cleaned.length - tOffset - 1));
  const qOnset = Math.max(0, rIndex - qOffset);
  const tEnd = Math.min(cleaned.length - 1, rIndex + tOffset);

  return {
    name: payload.filename || fallbackName,
    source: "ml",
    modelName: payload.model,
    modelMeta: `128 Hz ResNet pinball tail-loss · ${payload.windows.length} model windows`,
    fs,
    duration: payload.duration,
    sampleCount: payload.sampleCount,
    cleaned,
    peaks: payload.peaks || [],
    rrSamples: rrSec * fs,
    hr,
    beat: {
      y: cleaned.slice(Math.max(0, rIndex - Math.round(fs * 0.28)), rIndex + Math.round(fs * 0.62)),
      rIndex: Math.round(fs * 0.28),
      beatCount: payload.windows.length,
    },
    landmarks: { qOnset, r: rIndex, tPeak: Math.round((qOnset + tEnd) / 2), tEnd },
    qtMs: payload.qtMs,
    qtLowerMs: payload.qtLowerMs,
    qtUpperMs: payload.qtUpperMs,
    qtc: correctQt(payload.qtMs, rrSec, els.formulaSelect.value),
    rrSec,
    quality: {
      level: "good",
      title: "ML checkpoint inference",
      text: `QT was predicted by the 128 Hz lead-I ResNet checkpoint across ${payload.windows.length} ten-second windows.`,
    },
    mlWindows: payload.windows,
  };
}

els.fileInput.addEventListener("change", () => {
  const file = els.fileInput.files[0];
  if (!file) return;
  els.fileName.textContent = file.name;
  file.text().then((text) => loadText(text, file.name));
});

els.dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  els.dropzone.classList.add("is-dragging");
});

els.dropzone.addEventListener("dragleave", () => {
  els.dropzone.classList.remove("is-dragging");
});

els.dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  els.dropzone.classList.remove("is-dragging");
  const file = event.dataTransfer.files[0];
  if (!file) return;
  els.fileName.textContent = file.name;
  file.text().then((text) => loadText(text, file.name));
});

els.demoButton.addEventListener("click", () => loadText(demoCsv(), "Demo synthetic ECG.csv"));
els.clearButton.addEventListener("click", clearAnalysis);
els.formulaSelect.addEventListener("change", () => {
  if (!currentAnalysis) return;
  currentAnalysis.qtc = correctQt(currentAnalysis.qtMs, currentAnalysis.rrSec, els.formulaSelect.value);
  renderAnalysis(currentAnalysis);
});
els.sexSelect.addEventListener("change", () => {
  if (!currentAnalysis) return;
  renderAnalysis(currentAnalysis);
});

drawWaveform(null);
