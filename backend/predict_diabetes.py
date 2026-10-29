import pickle
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from backend.ml_engine import calculate_shap_values, CustomStandardScaler, CustomRandomForestClassifier

MODEL_PATH = 'models/diabetes_model.pkl'
SCALER_PATH = 'models/diabetes_scaler.pkl'

FEATURE_NAMES = [
    'Pregnancies', 'Glucose Level', 'Blood Pressure', 
    'Skin Thickness', 'Insulin Level', 'BMI', 
    'Diabetes Pedigree', 'Age'
]

def predict_diabetes(input_data):
    """
    input_data: dict with keys matching fields:
    {pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age}
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Diabetes model or scaler file not found. Train model first.")
        
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    raw_features = [
        float(input_data.get('pregnancies', 0)),
        float(input_data.get('glucose', 100)),
        float(input_data.get('blood_pressure', 70)),
        float(input_data.get('skin_thickness', 20)),
        float(input_data.get('insulin', 80)),
        float(input_data.get('bmi', 25)),
        float(input_data.get('pedigree', 0.5)),
        float(input_data.get('age', 30))
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
