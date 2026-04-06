const selectFields = ["traffic_level", "weather_condition", "road_condition", "vehicle_type"];
const statusText = document.getElementById("statusText");
const statusIndicator = document.getElementById("statusIndicator");
const form = document.getElementById("predictForm");
const submitBtn = document.getElementById("submitBtn");
const randomBtn = document.getElementById("randomBtn");
const resultPanel = document.getElementById("resultPanel");
const predictionResult = document.getElementById("predictionResult");

// Store category options for random generation
let categoryOptions = {};

function setSelectOptions(selectId, values) {
  const select = document.getElementById(selectId);
  select.innerHTML = "";
  categoryOptions[selectId] = values;
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value.charAt(0).toUpperCase() + value.slice(1);
    select.appendChild(option);
  });
}

function updateMetrics(metrics) {
  if (!metrics) return;
  
  document.getElementById("mae").textContent = metrics.mae?.toFixed(3) ?? "—";
  document.getElementById("rmse").textContent = metrics.rmse?.toFixed(3) ?? "—";
  document.getElementById("r2").textContent = ((metrics.r2 ?? 0) * 100).toFixed(2) + "%" ?? "—";
  document.getElementById("train_r2").textContent = ((metrics.train_r2 ?? 0) * 100).toFixed(2) + "%" ?? "—";
}

function generateRandomData() {
  // Generate random numeric values
  document.getElementById("distance_km").value = (Math.random() * 50 + 5).toFixed(2); // 5-55 km
  document.getElementById("stops").value = Math.floor(Math.random() * 10); // 0-9 stops
  document.getElementById("parcel_weight_kg").value = (Math.random() * 25 + 1).toFixed(2); // 1-26 kg
  document.getElementById("pickup_delay_min").value = (Math.random() * 30).toFixed(2); // 0-30 min
  
  // Generate random selections for category fields
  selectFields.forEach((field) => {
    if (categoryOptions[field] && categoryOptions[field].length > 0) {
      const randomIndex = Math.floor(Math.random() * categoryOptions[field].length);
      document.getElementById(field).value = categoryOptions[field][randomIndex];
    }
  });
  
  // Random peak hour
  document.getElementById("is_peak_hour").value = Math.random() > 0.5 ? "1" : "0";
}

async function fetchStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();

    // Update status indicator
    if (data.training) {
      statusIndicator.className = "status-indicator loading";
      statusText.textContent = "Training model...";
      submitBtn.disabled = true;
    } else if (data.trained) {
      statusIndicator.className = "status-indicator ready";
      statusText.textContent = "Model ready";
      submitBtn.disabled = false;
    } else if (data.error) {
      statusIndicator.className = "status-indicator error";
      statusText.textContent = `Error: ${data.error.substring(0, 30)}...`;
      submitBtn.disabled = true;
    } else {
      statusIndicator.className = "status-indicator loading";
      statusText.textContent = "Initializing...";
      submitBtn.disabled = true;
    }

    // Update metrics
    updateMetrics(data.metrics);

    // Update category selects
    if (data.categories) {
      selectFields.forEach((field) => {
        if (Array.isArray(data.categories[field]) && data.categories[field].length > 0) {
          setSelectOptions(field, data.categories[field]);
        }
      });
    }
  } catch (error) {
    console.error("Status fetch error:", error);
    statusIndicator.className = "status-indicator error";
    statusText.textContent = "Connection error";
  }
}

// Random data button event listener
randomBtn.addEventListener("click", (event) => {
  event.preventDefault();
  generateRandomData();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  
  submitBtn.disabled = true;
  submitBtn.textContent = "⏳ Predicting...";
  resultPanel.classList.add("hidden");

  const formData = new FormData(form);
  const payload = {
    distance_km: parseFloat(formData.get("distance_km")),
    traffic_level: String(formData.get("traffic_level")).toLowerCase(),
    weather_condition: String(formData.get("weather_condition")).toLowerCase(),
    road_condition: String(formData.get("road_condition")).toLowerCase(),
    vehicle_type: String(formData.get("vehicle_type")).toLowerCase(),
    is_peak_hour: parseInt(String(formData.get("is_peak_hour")), 10),
    stops: parseInt(String(formData.get("stops")), 10),
    parcel_weight_kg: parseFloat(formData.get("parcel_weight_kg")),
    pickup_delay_min: parseFloat(formData.get("pickup_delay_min")),
  };

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    
    if (!response.ok) {
      alert("Error: " + (data.detail || "Prediction failed"));
      submitBtn.disabled = false;
      submitBtn.innerHTML = "<i class='fas fa-magic'></i> Predict Delivery Time";
      return;
    }

    predictionResult.textContent = `${data.predicted_delivery_time_min.toFixed(2)}`;
    resultPanel.classList.remove("hidden");
    resultPanel.scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    console.error("Prediction error:", error);
    alert("An error occurred. Please try again.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = "<i class='fas fa-magic'></i> Predict Delivery Time";
  }
});

// Initial load and periodic updates
fetchStatus();
setInterval(fetchStatus, 2000);
