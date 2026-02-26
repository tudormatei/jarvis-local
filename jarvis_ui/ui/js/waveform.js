let audioSocket;

const NUM_MARKS = 60;
const fftSize = 256;
const marksList = document.getElementById("marks");
const marks = marksList.children;
const minTranslateY = 125;
const maxTranslateY = 170;

const fft = new FFT(fftSize);
const input = new Float32Array(fftSize);
const output = fft.createComplexArray();

// Smoothed amplitudes for each mark
let smoothed = new Float32Array(NUM_MARKS).fill(0);
const SMOOTH_FACTOR = 0.3; // 0 = no smoothing, 1 = instant

function computeMirroredMagnitudes() {
  const half = fftSize / 2;
  const mags = new Float32Array(half);

  let maxMag = 1e-6;
  for (let i = 0; i < half; i++) {
    const re = output[2 * i];
    const im = output[2 * i + 1];
    mags[i] = Math.sqrt(re * re + im * im);
    if (mags[i] > maxMag) maxMag = mags[i];
  }

  // Normalize
  for (let i = 0; i < half; i++) {
    mags[i] /= maxMag;
  }

  // Mirror FFT bins across NUM_MARKS marks symmetrically
  // First half of marks: bins 0..half-1 mapped to marks 0..NUM_MARKS/2-1
  // Second half of marks: mirror of first half
  const result = new Float32Array(NUM_MARKS);
  const halfMarks = NUM_MARKS / 2;
  for (let i = 0; i < halfMarks; i++) {
    // Map mark index to FFT bin (skip DC bin 0, start from 1)
    const binIndex = 1 + Math.floor((i / halfMarks) * (half / 4));
    result[i] = mags[binIndex];
    result[NUM_MARKS - 1 - i] = mags[binIndex]; // mirror
  }
  return result;
}

function startAudioStream() {
  audioSocket = new WebSocket("ws://localhost:8765");
  audioSocket.binaryType = "arraybuffer";

  audioSocket.onopen = () => console.log("WebSocket opened");

  audioSocket.onmessage = (event) => {
    if (!(event.data instanceof ArrayBuffer)) return;
    const float32 = new Float32Array(event.data);

    for (let i = 0; i < fftSize; i++) {
      input[i] = i < float32.length ? float32[i] : 0;
    }

    fft.realTransform(output, input);
    fft.completeSpectrum(output);

    const mags = computeMirroredMagnitudes();

    // Apply temporal smoothing
    for (let i = 0; i < NUM_MARKS; i++) {
      smoothed[i] = smoothed[i] * (1 - SMOOTH_FACTOR) + mags[i] * SMOOTH_FACTOR;
    }
  };

  audioSocket.onerror = (err) => console.error("WebSocket error:", err);

  audioSocket.onclose = () => {
    console.log("WebSocket closed, retrying in 1s...");
    // Decay to silence on close
    for (let i = 0; i < NUM_MARKS; i++) smoothed[i] *= 0.95;
    setTimeout(startAudioStream, 1000);
  };
}

function updateWaveform() {
  for (let i = 0; i < marks.length; i++) {
    const translateY =
      minTranslateY + (maxTranslateY - minTranslateY) * smoothed[i];
    const rotateValue = (360 / NUM_MARKS) * i;
    marks[i].style.transform =
      `rotate(${rotateValue}deg) translateY(-${translateY}px)`;
  }

  // Decay smoothed values toward silence when no audio
  for (let i = 0; i < NUM_MARKS; i++) {
    smoothed[i] *= 0.97;
  }

  requestAnimationFrame(updateWaveform);
}

document.addEventListener("DOMContentLoaded", () => {
  startAudioStream();
  requestAnimationFrame(updateWaveform);
});
