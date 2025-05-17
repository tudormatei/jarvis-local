let audioSocket;

// --- Waveform animation setup ---
const bufferLength = 60;
const fftSize = 256;
const marksList = document.getElementById("marks");
const marks = marksList.children;
const minTranslateY = 125;
const maxTranslateY = 170;

// --- FFT setup ---
const fft = new FFT(fftSize);
const input = new Float32Array(fftSize);
const output = fft.createComplexArray();
let latestDataArray = new Uint8Array(bufferLength); // Store latest data for animation

// --- WebSocket PCM data stream setup ---
function startAudioStream() {
  audioSocket = new WebSocket("ws://localhost:8765");
  audioSocket.binaryType = "arraybuffer";
  audioSocket.onopen = () => {
    console.log("WebSocket connection opened");
  };
  audioSocket.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      const float32 = new Float32Array(event.data);

      // Fill input buffer (zero-pad if needed)
      for (let i = 0; i < fftSize; i++) {
        input[i] = float32[i] || 0;
      }

      // Perform FFT
      fft.realTransform(output, input);
      fft.completeSpectrum(output);

      // Compute magnitudes and fill dataArray
      let dataArray = new Uint8Array(bufferLength);
      for (let i = 0; i < bufferLength; i++) {
        const re = output[2 * i];
        const im = output[2 * i + 1];
        const mag = Math.sqrt(re * re + im * im);
        dataArray[i] = Math.min(255, Math.floor(mag * 20));
      }

      // --- Your original waveform post-processing ---
      const firstZeroIndex = dataArray.indexOf(0);
      const length = dataArray.length - 1;
      const threshold = 4;
      for (let i = length; i >= firstZeroIndex + threshold; i--) {
        dataArray[i] = dataArray[length - i];
      }

      let x =
        dataArray[firstZeroIndex + threshold] - dataArray[firstZeroIndex - 1];
      x = Math.floor(x / 5);
      for (let i = firstZeroIndex + threshold - 1; i >= firstZeroIndex; i--) {
        dataArray[i] = dataArray[i + 1] - x;
      }

      // Store for animation
      latestDataArray = dataArray;
    }
  };
  audioSocket.onerror = (err) => {
    console.error("WebSocket error:", err);
  };
  audioSocket.onclose = () => {
    console.log("WebSocket connection closed");
  };
}

// --- Animation loop at 60 FPS ---
function updateWaveform() {
  for (let i = 0; i < marks.length; i++) {
    const mark = marks[i];
    const normalizedValue = latestDataArray[i] / 255;
    const translateY =
      minTranslateY + (maxTranslateY - minTranslateY) * normalizedValue;
    const rotateValue = 6 * i;
    mark.style.transform = `rotate(${rotateValue}deg) translateY(${translateY}px)`;
  }
  requestAnimationFrame(updateWaveform);
}

document.addEventListener("DOMContentLoaded", () => {
  startAudioStream();
  requestAnimationFrame(updateWaveform);
});
