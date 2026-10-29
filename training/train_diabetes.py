import os
import pickle
import pandas as pd
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from backend.ml_engine import CustomStandardScaler, CustomRandomForestClassifier

def train_diabetes():
    print("--- Training Diabetes Model ---")
    data_path = 'datasets/diabetes.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found.")
    
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Outcome']).values
    y = df['Outcome'].values
    feature_names = [col for col in df.columns if col != 'Outcome']
    
    # 80/20 train/test split
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_idx = int(n_samples * 0.8)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    scaler = CustomStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = CustomRandomForestClassifier(n_estimators=40, max_depth=6)
    model.fit(X_train_scaled, y_train, feature_names=feature_names)
    
    preds = model.predict(X_test_scaled)
    acc = np.mean(preds == y_test)
    
    tp = np.sum((preds == 1) & (y_test == 1))
    fp = np.sum((preds == 1) & (y_test == 0))
    fn = np.sum((preds == 0) & (y_test == 1))
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    print(f"Diabetes Model Metrics -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    
    os.makedirs('models', exist_ok=True)
    with open('models/diabetes_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/diabetes_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print("Saved models/diabetes_model.pkl and models/diabetes_scaler.pkl")
    return acc, prec, rec

if __name__ == '__main__':
    train_diabetes()
