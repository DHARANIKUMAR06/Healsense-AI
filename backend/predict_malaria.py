import pickle
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from backend.ml_engine import calculate_shap_values, CustomStandardScaler, CustomRandomForestClassifier

MODEL_PATH = 'models/malaria_model.pkl'
SCALER_PATH = 'models/malaria_scaler.pkl'

FEATURE_NAMES = [
    'WBC Count', 'RBC Count', 'Platelet Count', 'Hemoglobin',
    'Fever Duration', 'Body Temp (°F)', 'Chills / Rigor', 'Spleen Enlargement'
]

def predict_malaria(input_data):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Malaria model or scaler file not found. Train model first.")

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    raw_features = [
        float(input_data.get('wbc_count', 7000)),
        float(input_data.get('rbc_count', 4.5)),
        float(input_data.get('platelet_count', 200000)),
        float(input_data.get('hemoglobin', 13.5)),
        float(input_data.get('fever_duration', 3)),
        float(input_data.get('body_temperature', 99.5)),
        float(input_data.get('chills_rigor', 0)),
        float(input_data.get('spleen_enlargement', 0))
    ]

    scaled_features = scaler.transform(np.array(raw_features).reshape(1, -1))
    probs = model.predict_proba(scaled_features)[0]
    pos_prob = float(probs[1])
    risk_percentage = round(pos_prob * 100, 1)

    result = "Positive" if pos_prob >= 0.5 else "Negative"
    shap_values = calculate_shap_values(model, scaler, FEATURE_NAMES, raw_features)

    return {
        'result': result,
        'risk_percentage': risk_percentage,
        'shap_values': shap_values
    }
