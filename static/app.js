const video = document.querySelector("#camera");
const canvas = document.querySelector("#frame-canvas");
const stage = document.querySelector("#camera-stage");
const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const message = document.querySelector("#camera-message");
const livePill = document.querySelector("#live-pill");
const statusBanner = document.querySelector("#status-banner");
const overallStatus = document.querySelector("#overall-status");
const statusIcon = document.querySelector("#status-icon");
const eyeStatus = document.querySelector("#eye-status");
const mouthStatus = document.querySelector("#mouth-status");
const eyeConfidence = document.querySelector("#eye-confidence");
const mouthConfidence = document.querySelector("#mouth-confidence");
const sessionTime = document.querySelector("#session-time");

let stream;
let timer;
let sessionStartedAt;
let requestInProgress = false;
let closedEyeFrames = 0;
let yawnFrames = 0;
let audioContext;
let alarmTimer;
let currentAlarm = "";

function setStatus(state, title, icon) {
  statusBanner.dataset.state = state;
  overallStatus.textContent = title;
  statusIcon.textContent = icon;
}

function formatTime(totalSeconds) {
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function playBeep() {
  if (!audioContext || audioContext.state !== "running") return;
  const oscillator = audioContext.createOscillator();
  const gain = audioContext.createGain();
  oscillator.type = "sine";
  oscillator.frequency.value = currentAlarm === "sleep" ? 880 : 620;
  gain.gain.setValueAtTime(0.0001, audioContext.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.08, audioContext.currentTime + 0.025);
  gain.gain.exponentialRampToValueAtTime(0.0001, audioContext.currentTime + 0.22);
  oscillator.connect(gain);
  gain.connect(audioContext.destination);
  oscillator.start();
  oscillator.stop(audioContext.currentTime + 0.23);
}

function setAlarm(type) {
  if (currentAlarm === type) return;
  clearInterval(alarmTimer);
  currentAlarm = type;
  if (type) {
    playBeep();
    alarmTimer = setInterval(playBeep, 900);
  }
}

function resetMetrics() {
  closedEyeFrames = 0;
  yawnFrames = 0;
  eyeStatus.textContent = "Waiting";
  mouthStatus.textContent = "Waiting";
  eyeConfidence.textContent = "—";
  mouthConfidence.textContent = "—";
  setStatus("idle", "Ready when you are", "—");
  setAlarm("");
}

async function analyzeFrame() {
  if (!stream || requestInProgress || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  requestInProgress = true;
  canvas.width = Math.min(video.videoWidth, 640);
  canvas.height = Math.round(video.videoHeight * (canvas.width / video.videoWidth));
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  try {
    const frame = await new Promise((resolve, reject) => {
      canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Could not capture a camera frame.")), "image/jpeg", 0.78);
    });
    const response = await fetch("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
      body: frame,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Frame analysis failed.");
    updateInsights(result);
  } catch (error) {
    message.textContent = error.message;
    message.classList.add("error-text");
  } finally {
    requestInProgress = false;
  }
}

function updateInsights(result) {
  message.classList.remove("error-text");
  message.textContent = "Camera active · analysis stays on this device.";
  if (!result.face_detected) {
    eyeStatus.textContent = "No face detected";
    mouthStatus.textContent = "—";
    eyeConfidence.textContent = "—";
    mouthConfidence.textContent = "—";
    closedEyeFrames = 0;
    yawnFrames = 0;
    setAlarm("");
    setStatus("warning", "Looking for a face", "⌕");
    return;
  }

  const eyes = result.eyes;
  const mouth = result.mouth;
  eyeStatus.textContent = eyes?.label ?? "Not detected";
  mouthStatus.textContent = mouth?.label === "no_yawn" ? "Normal" : (mouth?.label ?? "Not detected");
  eyeConfidence.textContent = eyes ? `${eyes.confidence}%` : "—";
  mouthConfidence.textContent = mouth ? `${mouth.confidence}%` : "—";

  closedEyeFrames = eyes?.label === "Closed" ? closedEyeFrames + 1 : 0;
  yawnFrames = mouth?.label === "yawn" ? yawnFrames + 1 : 0;

  if (closedEyeFrames >= 4) {
    setStatus("alert", "Eyes closed — wake up!", "!");
    setAlarm("sleep");
  } else if (yawnFrames >= 3) {
    setStatus("warning", "Signs of drowsiness", "!");
    setAlarm("yawn");
  } else if (eyes?.label === "Closed" || mouth?.label === "yawn") {
    setStatus("warning", "Stay focused on the road", "!");
    setAlarm("");
  } else {
    setStatus("safe", "You look alert", "✓");
    setAlarm("");
  }
}

function updateSessionClock() {
  if (!sessionStartedAt) return;
  const elapsed = Math.floor((Date.now() - sessionStartedAt) / 1000);
  sessionTime.textContent = formatTime(elapsed);
}

async function startMonitoring() {
  if (!navigator.mediaDevices?.getUserMedia) {
    message.textContent = "Camera access requires HTTPS or http://localhost.";
    message.classList.add("error-text");
    return;
  }
  startButton.disabled = true;
  message.classList.remove("error-text");
  message.textContent = "Requesting camera access…";
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    });
    video.srcObject = stream;
    await video.play();
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      audioContext = new AudioContextClass();
      audioContext.resume().catch(() => {
        message.textContent = "Sound alerts unavailable; visual warnings remain active.";
      });
    } else {
      message.textContent = "Sound alerts are not supported by this browser.";
    }
    stage.dataset.active = "true";
    livePill.dataset.live = "true";
    livePill.innerHTML = "<span></span> LIVE";
    sessionStartedAt = Date.now();
    resetMetrics();
    startButton.disabled = true;
    stopButton.disabled = false;
    timer = setInterval(() => {
      updateSessionClock();
      analyzeFrame();
    }, 400);
    updateSessionClock();
    analyzeFrame();
  } catch (error) {
    stopMonitoring();
    startButton.disabled = false;
    message.classList.add("error-text");
    message.textContent = error.name === "NotAllowedError"
      ? "Camera permission was denied. Allow camera access and try again."
      : `Could not start the camera: ${error.message}`;
  }
}

function stopMonitoring() {
  clearInterval(timer);
  clearInterval(alarmTimer);
  currentAlarm = "";
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = undefined;
  video.srcObject = null;
  stage.dataset.active = "false";
  livePill.dataset.live = "false";
  livePill.innerHTML = "<span></span> STANDBY";
  startButton.disabled = false;
  stopButton.disabled = true;
  sessionStartedAt = undefined;
  sessionTime.textContent = "00:00";
  message.classList.remove("error-text");
  message.textContent = "Monitoring stopped. Your camera has been released.";
  resetMetrics();
}

startButton.addEventListener("click", startMonitoring);
stopButton.addEventListener("click", stopMonitoring);
window.addEventListener("beforeunload", () => {
  if (stream) stream.getTracks().forEach((track) => track.stop());
});
