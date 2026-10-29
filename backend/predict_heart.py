import pickle
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from backend.ml_engine import calculate_shap_values, CustomStandardScaler, CustomRandomForestClassifier

MODEL_PATH = 'models/heart_model.pkl'
SCALER_PATH = 'models/heart_scaler.pkl'

FEATURE_NAMES = [
    'Age', 'Sex', 'Chest Pain Type', 'Resting BP',
    'Serum Cholesterol', 'Fasting Sugar > 120', 'Resting ECG',
    'Max Heart Rate', 'Exercise Angina', 'ST Depression',
    'ST Segment Slope', 'Major Vessels (0-4)', 'Thalassemia Type'
]

def predict_heart(input_data):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Heart model or scaler file not found. Train model first.")

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    raw_features = [
        float(input_data.get('age', 50)),
        float(input_data.get('sex', 1)),
        float(input_data.get('cp', 0)),
        float(input_data.get('trestbps', 120)),
        float(input_data.get('chol', 200)),
        float(input_data.get('fbs', 0)),
        float(input_data.get('restecg', 0)),
        float(input_data.get('thalach', 150)),
        float(input_data.get('exang', 0)),
        float(input_data.get('oldpeak', 1.0)),
        float(input_data.get('slope', 1)),
        float(input_data.get('ca', 0)),
        float(input_data.get('thal', 2))
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
