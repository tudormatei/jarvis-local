const NUM_MARKS = 60;
const MIN_TRANSLATE = 125;
const MAX_TRANSLATE = 225;

const marksList = document.getElementById("marks");
const marks = marksList.children;

let BLOCK_SIZE = 256;
let SAMPLE_RATE = 24000;

const HISTORY_SIZE = 4096;
const sampleRing = new Float32Array(HISTORY_SIZE);
let writeHead = 0; // total samples written (never wraps)

function pushSamples(float32) {
  for (let i = 0; i < float32.length; i++) {
    sampleRing[writeHead % HISTORY_SIZE] = float32[i];
    writeHead++;
  }
}

function readSample(absoluteIndex) {
  if (absoluteIndex < 0 || absoluteIndex >= writeHead) return 0;
  return sampleRing[
    ((absoluteIndex % HISTORY_SIZE) + HISTORY_SIZE) % HISTORY_SIZE
  ];
}

const smoothed = new Float32Array(NUM_MARKS).fill(0);
const ATTACK = 0.55; // rise speed
const RELEASE = 0.18; // fall speed

let smoothedRMS = 0;
const SILENCE_RMS = 0.004;

function blockRMS(float32) {
  let s = 0;
  for (let i = 0; i < float32.length; i++) s += float32[i] * float32[i];
  return Math.sqrt(s / float32.length);
}

let idleTime = 0;

function idleAmp(markIndex, t) {
  const a = (2 * Math.PI * markIndex) / NUM_MARKS;
  return (
    0.03 *
    (0.5 +
      0.4 * Math.sin(a * 2 + t * 1.0) +
      0.3 * Math.sin(a * 3 - t * 0.65) +
      0.15 * Math.sin(a * 5 + t * 0.4))
  );
}

let gotNewFrame = false;

function startAudioStream() {
  const ws = new WebSocket("ws://localhost:8765");
  ws.binaryType = "arraybuffer";

  ws.onopen = () => console.log("[waveform] connected");

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      try {
        const cfg = JSON.parse(event.data);
        if (cfg.type === "config") {
          SAMPLE_RATE = cfg.sample_rate;
          BLOCK_SIZE = cfg.block_size;
          console.log(`[waveform] ${SAMPLE_RATE} Hz, block=${BLOCK_SIZE}`);
        }
      } catch (_) {}
      return;
    }
    if (!(event.data instanceof ArrayBuffer)) return;

    const samples = new Float32Array(event.data);
    pushSamples(samples);

    const rms = blockRMS(samples);
    smoothedRMS = smoothedRMS * 0.75 + rms * 0.25;
    gotNewFrame = true;
  };

  ws.onerror = (e) => console.error("[waveform] error", e);
  ws.onclose = () => {
    console.log("[waveform] closed, retry 1s");
    setTimeout(startAudioStream, 1000);
  };
}

let lastNow = performance.now();

function updateWaveform(now) {
  const dt = Math.min((now - lastNow) / 1000, 0.05); // cap at 50ms
  lastNow = now;
  idleTime += dt;

  const isSilent = smoothedRMS < SILENCE_RMS;

  if (isSilent) {
    for (let i = 0; i < NUM_MARKS; i++) {
      const target = idleAmp(i, idleTime);
      smoothed[i] += (target - smoothed[i]) * 0.06;
    }
  } else {
    const window = Math.min(BLOCK_SIZE * 2, HISTORY_SIZE - 1);
    const step = window / NUM_MARKS;
    const kernel = Math.max(1, Math.round(step * 0.6));

    const raw = new Float32Array(NUM_MARKS);
    let maxRaw = 1e-7;

    for (let m = 0; m < NUM_MARKS; m++) {
      const center = writeHead - window + m * step;

      let sum = 0;
      for (let k = -kernel; k <= kernel; k++) {
        sum += Math.abs(readSample(Math.round(center + k)));
      }
      raw[m] = sum / (2 * kernel + 1);
      if (raw[m] > maxRaw) maxRaw = raw[m];
    }

    const loudness = Math.min(1.0, smoothedRMS / 0.08);
    const minShow = 0.1;

    for (let i = 0; i < NUM_MARKS; i++) {
      const norm = raw[i] / maxRaw; // 0..1 shape
      const scaled = minShow + (1 - minShow) * norm * loudness;
      const alpha = scaled > smoothed[i] ? ATTACK : RELEASE;
      smoothed[i] += (scaled - smoothed[i]) * alpha;
    }
  }

  gotNewFrame = false;

  for (let i = 0; i < NUM_MARKS; i++) {
    const t = Math.max(0, Math.min(1, smoothed[i]));
    const ty = MIN_TRANSLATE + (MAX_TRANSLATE - MIN_TRANSLATE) * t;
    const r = (360 / NUM_MARKS) * i;
    marks[i].style.transform = `rotate(${r}deg) translateY(-${ty}px)`;
  }

  requestAnimationFrame(updateWaveform);
}

document.addEventListener("DOMContentLoaded", () => {
  startAudioStream();
  requestAnimationFrame(updateWaveform);
});
