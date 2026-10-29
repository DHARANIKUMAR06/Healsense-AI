import pickle
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from backend.ml_engine import calculate_shap_values, CustomStandardScaler, CustomRandomForestClassifier

MODEL_PATH = 'models/kidney_model.pkl'
SCALER_PATH = 'models/kidney_scaler.pkl'

FEATURE_NAMES = [
    'Age', 'Blood Pressure', 'Specific Gravity', 'Albumin', 'Sugar',
    'Red Blood Cells', 'Pus Cells', 'Pus Cell Clumps', 'Bacteria',
    'Blood Glucose Random', 'Blood Urea', 'Serum Creatinine',
    'Sodium', 'Potassium', 'Hemoglobin'
]

def predict_kidney(input_data):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Kidney model or scaler file not found. Train model first.")

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    raw_features = [
        float(input_data.get('age', 40)),
        float(input_data.get('bp', 80)),
        float(input_data.get('sg', 1.020)),
        float(input_data.get('al', 0)),
        float(input_data.get('su', 0)),
        float(input_data.get('rbc', 0)),
        float(input_data.get('pc', 0)),
        float(input_data.get('pcc', 0)),
        float(input_data.get('ba', 0)),
        float(input_data.get('bgr', 120)),
        float(input_data.get('bu', 30)),
        float(input_data.get('sc', 1.0)),
        float(input_data.get('sod', 138)),
        float(input_data.get('pot', 4.5)),
        float(input_data.get('hemo', 13.5))
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
