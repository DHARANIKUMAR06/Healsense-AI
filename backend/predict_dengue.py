import pickle
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from backend.ml_engine import calculate_shap_values, CustomStandardScaler, CustomRandomForestClassifier

MODEL_PATH = 'models/dengue_model.pkl'
SCALER_PATH = 'models/dengue_scaler.pkl'

FEATURE_NAMES = [
    'WBC Count', 'Platelet Count', 'Hematocrit', 'NS1 Antigen',
    'IgG Antibody', 'IgM Antibody', 'Fever Duration', 'Body Temp (°F)'
]

def predict_dengue(input_data):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Dengue model or scaler file not found. Train model first.")

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    raw_features = [
        float(input_data.get('wbc_count', 6000)),
        float(input_data.get('platelet_count', 180000)),
        float(input_data.get('hematocrit', 40.0)),
        float(input_data.get('ns1_antigen', 0)),
        float(input_data.get('igg', 0)),
        float(input_data.get('igm', 0)),
        float(input_data.get('fever_duration', 3)),
        float(input_data.get('body_temperature', 99.5))
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
