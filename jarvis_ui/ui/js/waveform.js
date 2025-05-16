let audioSocket;

// --- Waveform animation setup ---
const bufferLength = 60;
const fftSize = 256; // Should be a power of 2 and <= PCM block size
const marksList = document.getElementById("marks");
const marks = marksList.children;
const minTranslateY = 125;
const maxTranslateY = 170;

// --- FFT setup ---
const fft = new FFT(fftSize);
const input = new Float32Array(fftSize);
const output = fft.createComplexArray();
const dataArray = new Uint8Array(bufferLength);

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
      for (let i = 0; i < bufferLength; i++) {
        const re = output[2 * i];
        const im = output[2 * i + 1];
        const mag = Math.sqrt(re * re + im * im);
        // Scale to 0-255 for visualization (adjust multiplier for sensitivity)
        dataArray[i] = Math.min(255, Math.floor(mag * 16));
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

      // --- Animate the waveform ---
      for (let i = 0; i < marks.length; i++) {
        const mark = marks[i];
        // Nonlinear scaling for more dynamic movement
        let normalizedValue = dataArray[i] / 255;
        normalizedValue = Math.pow(normalizedValue, 0.7);

        const translateY =
          minTranslateY + (maxTranslateY - minTranslateY) * normalizedValue;
        const rotateValue = (360 / marks.length) * i;

        // Optional: color gradient
        const hue = Math.floor(200 + 100 * normalizedValue); // blue to cyan
        mark.style.background = `hsl(${hue}, 80%, 60%)`;

        mark.style.transform = `rotate(${rotateValue}deg) translateY(${translateY}px)`;
      }
    }
  };
  audioSocket.onerror = (err) => {
    console.error("WebSocket error:", err);
  };
  audioSocket.onclose = () => {
    console.log("WebSocket connection closed");
  };
}

document.addEventListener("DOMContentLoaded", () => {
  startAudioStream();
});
