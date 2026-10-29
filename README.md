# HealthSense AI - Disease Detection System

**B.E/CSE PROJECT**  
*Ensemble Machine Learning with SHAP (Explainable AI) & Live Optical PPG Pulse Sensing*

---

## 📌 Project Overview
**HealthSense AI** is an advanced medical diagnostic web application that predicts disease risk probabilities across 5 major clinical domains using Machine Learning (Ensemble Random Forest with SHAP explainability) and includes a real-time, non-contact camera-based heart rate monitor (rPPG).

### Key Features
1. **5 Clinical ML Diagnostic Modules**:
   - **Diabetes Risk**: Evaluates Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree, and Age.
   - **Heart Disease Risk**: Evaluates Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Angina, ST Depression, ST Slope, Major Vessels, and Thalassemia.
   - **Chronic Kidney Disease (CKD)**: Evaluates 15 physiological & lab metrics (Albumin, Sugar, Serum Creatinine, Blood Urea, Hemoglobin, Specific Gravity, etc.).
   - **Malaria Risk**: Evaluates CBC panel (WBC, RBC, Platelets, Hemoglobin), Fever duration, Body Temperature, Chills/Rigor, and Splenomegaly.
   - **Dengue Risk**: Evaluates Hematocrit, WBC, Platelets, NS1 Antigen, IgG/IgM Antibodies, Fever duration, and Temperature.
2. **SHAP (Explainable AI) Visualizations**: Every single assessment generates a horizontal bar chart powered by Chart.js showing exactly which features drove the patient's risk calculation up or down.
3. **Live Camera-Based Heart Rate Monitor (rPPG)**: Remote Photoplethysmography using browser webcam color intensity variations (Green channel) to track pulse waveforms in real-time, compute Beats Per Minute (BPM), and classify output into Normal / Bradycardia / Tachycardia.
4. **SQLite Prediction Logging & Real-Time Dashboard**: Automatically logs every prediction to `predictions.db` and updates the Dashboard Overview page with aggregated cards, bar charts, risk donuts, time series, and auto-refreshing recent activity tables.

---

## 🛠️ Tech Stack
- **Backend Framework**: Python Flask
- **Machine Learning Engine**: Custom Decision Tree / Random Forest Ensemble Engine, StandardScaler normalization, SHAP explainability calculation
- **Database**: SQLite (`predictions.db` with `predictions` table)
- **Frontend**: HTML5, Vanilla CSS3 (Custom Glassmorphic Dark Navy Sidebar Design System), Vanilla JavaScript (AJAX fetch, canvas processing, camera stream)
- **Data Visualizations**: Chart.js 4.4 (via CDN) & FontAwesome 6.4 (via CDN)
- **Model Storage**: Pickle `.pkl` files

---

## 📁 Directory Structure
```
c:\Users\ELCOT\Downloads\Disease detection\
├── app.py                      # Flask Server (Routes, Prediction endpoints, SQLite logging)
├── seed_data.py                # Database seeder script for dashboard testing
├── requirements.txt            # Package dependencies
├── README.md                   # Documentation
├── backend/
│   ├── database.py             # SQLite helper (init, log_prediction, get_dashboard_stats)
│   ├── ml_engine.py             # Pure Python StandardScaler, RandomForest & SHAP calculator
│   ├── predict_diabetes.py     # Diabetes prediction module
│   ├── predict_heart.py        # Heart disease prediction module
│   ├── predict_kidney.py       # Chronic kidney disease prediction module
│   ├── predict_malaria.py      # Malaria risk prediction module
│   └── predict_dengue.py       # Dengue risk prediction module
├── datasets/
│   ├── generate_datasets.py    # Generator script for clinical & synthetic datasets
│   ├── diabetes.csv
│   ├── heart.csv
│   ├── kidney.csv
│   ├── malaria.csv
│   └── dengue.csv
├── models/                     # Saved trained models & scalers (.pkl)
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── heart_model.pkl
│   ├── heart_scaler.pkl
│   ├── kidney_model.pkl
│   ├── kidney_scaler.pkl
│   ├── malaria_model.pkl
│   ├── malaria_scaler.pkl
│   ├── dengue_model.pkl
│   └── dengue_scaler.pkl
├── training/
│   ├── train_all.py            # Master script to train all 5 models
│   ├── train_diabetes.py
│   ├── train_heart.py
│   ├── train_kidney.py
│   ├── train_malaria.py
│   └── train_dengue.py
├── templates/                  # HTML templates
│   ├── layout.html             # Base layout (Sidebar, Header, CSS/JS dependencies)
│   ├── diabetes.html           # Diabetes assessment page
│   ├── heart.html              # Heart disease page
│   ├── kidney.html             # Kidney disease page
│   ├── malaria.html            # Malaria risk page
│   ├── dengue.html             # Dengue risk page
│   ├── heart_rate.html         # Live camera rPPG heart rate monitor page
│   └── dashboard.html          # Dashboard analytics page
└── static/
    ├── css/
    │   └── style.css           # Master CSS design system
    └── js/
        ├── assessment.js       # Form AJAX handler & SHAP chart renderer
        ├── heart_rate.js       # Camera rPPG signal processing & waveform renderer
        └── dashboard.js        # Dashboard statistics polling & Chart.js renderer
```

---

## 🚀 Quick Setup & Installation

### Step 1: Environment Setup
```bash
# Navigate to project folder
cd "c:\Users\ELCOT\Downloads\Disease detection"

# Create a Python virtual environment (optional)
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### Step 2: Generate Datasets & Train Models
To generate all 5 datasets and train the ML models:
```bash
# Generate datasets
python datasets/generate_datasets.py

# Train all 5 disease models (saves into models/*.pkl)
python training/train_all.py
```

### Step 3: Seed Sample Predictions (Optional for Dashboard Preview)
To seed 50+ sample records into `predictions.db` for instant dashboard testing:
```bash
python seed_data.py
```

### Step 4: Run the Flask Web Application
```bash
python app.py
```

Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

---

## 🌐 Web Routes & API Endpoints

| Route | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Redirects to `/diabetes-risk` |
| `/diabetes-risk` | `GET` / `POST` | Diabetes Risk form & AJAX prediction |
| `/heart-risk` | `GET` / `POST` | Heart Disease Risk form & AJAX prediction |
| `/kidney-risk` | `GET` / `POST` | Chronic Kidney Disease Risk form & AJAX prediction |
| `/malaria-risk` | `GET` / `POST` | Malaria Risk form & AJAX prediction |
| `/dengue-risk` | `GET` / `POST` | Dengue Risk form & AJAX prediction |
| `/heart-rate-monitor` | `GET` | Camera rPPG Heart Rate Monitor page |
| `/log-heart-rate` | `POST` | Logs heart rate calculation to `predictions.db` |
| `/dashboard` | `GET` | Dashboard Overview Analytics page |
| `/api/dashboard-stats` | `GET` | Returns dynamic aggregated JSON stats from SQLite |

---

## 🎓 Academic Project Credit
Developed as a **B.E/CSE Project** by **Dharani**.
