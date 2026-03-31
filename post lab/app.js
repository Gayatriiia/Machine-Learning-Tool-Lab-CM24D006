// =============================
// CONFIGURATION
// =============================

// TODO: Replace with your model URL (Teachable Machine export)
const MODEL_URL = "https://teachablemachine.withgoogle.com/models/pTB8wq_Qj/model.json";
// If labels not auto-detected, define manually
let LABELS = ["Happy", "Sad"];

let model;
let chart;

// =============================
// LOAD MODEL (supports both types)
// =============================
async function loadModel() {
  try {
    model = await tf.loadLayersModel(MODEL_URL);
    console.log("Loaded Layers Model");
  } catch (e) {
    model = await tf.loadGraphModel(MODEL_URL);
    console.log("Loaded Graph Model");
  }
}

loadModel();

// =============================
// CAMERA
// =============================
const video = document.getElementById("webcam");

document.getElementById("startCam").onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    console.log("Camera started");
  } catch (err) {
    console.error("Camera error:", err);
    alert("Camera error: " + err.message);
  }
};

// =============================
// IMAGE PREPROCESSING
// =============================
function preprocessImage(img) {
  return tf.tidy(() => {
    let tensor = tf.browser.fromPixels(img)
      .resizeNearestNeighbor([224, 224]) // match model input
      .toFloat()
      .div(255.0)
      .expandDims();

    return tensor;
  });
}

// =============================
// PREDICTION
// =============================
async function predict(img) {
  const tensor = preprocessImage(img);
  const predictions = await model.predict(tensor).data();

  // Convert to array with labels
  let results = LABELS.map((label, i) => ({
    label,
    prob: predictions[i]
  }));

  results.sort((a, b) => b.prob - a.prob);

  displayResults(results);
}

// =============================
// DISPLAY RESULTS
// =============================
function displayResults(results) {
  const top = results[0];
  document.getElementById("finalEmotion").innerText =
    `${top.label} (${(top.prob * 100).toFixed(2)}%)`;

  updateChart(results.slice(0, 3));
  addHistory(top);
}

// =============================
// CHART
// =============================
function updateChart(data) {
  const ctx = document.getElementById("chart").getContext("2d");

  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: data.map(d => d.label),
      datasets: [{
        label: "Confidence",
        data: data.map(d => d.prob * 100)
      }]
    }
  });
}

// =============================
// HISTORY
// =============================
function addHistory(result) {
  const li = document.createElement("li");
  const time = new Date().toLocaleTimeString();
  li.innerText = `${time} → ${result.label} (${(result.prob*100).toFixed(1)}%)`;
  document.getElementById("history").prepend(li);
}

// =============================
// CAPTURE FROM WEBCAM
// =============================
document.getElementById("capture").onclick = () => {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);

  predict(canvas);
};

// =============================
// UPLOAD IMAGE
// =============================
const upload = document.getElementById("upload");
const preview = document.getElementById("preview");

upload.onchange = () => {
  const file = upload.files[0];
  preview.src = URL.createObjectURL(file);
};

document.getElementById("predictUpload").onclick = () => {
  predict(preview);
};