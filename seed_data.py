import random
import json
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from backend.database import init_db, get_db_connection

def seed_sample_predictions():
    print("Seeding sample predictions into predictions.db...")
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    # Clear previous seed records if any
    cursor.execute("DELETE FROM predictions")
    
    diseases = ['diabetes', 'heart', 'kidney', 'malaria', 'dengue']
    now = datetime.now()
    
    sample_data_count = 50
    for i in range(sample_data_count):
        d_type = random.choice(diseases)
        days_back = random.randint(0, 14)
        hours_back = random.randint(0, 23)
        mins_back = random.randint(0, 59)
        record_time = (now - timedelta(days=days_back, hours=hours_back, minutes=mins_back)).strftime("%Y-%m-%d %H:%M:%S")

        # Create realistic synthetic input values
        if d_type == 'diabetes':
            glucose = random.randint(70, 195)
            bmi = round(random.uniform(18.5, 45.0), 1)
            age = random.randint(22, 75)
            risk_pct = round(min(100.0, max(5.0, (glucose - 80)*0.6 + (bmi - 20)*1.1 + (age - 20)*0.4)), 1)
            result = "Positive" if risk_pct >= 50.0 else "Negative"
            input_feat = {"glucose": glucose, "bmi": bmi, "age": age, "pregnancies": random.randint(0, 8)}

        elif d_type == 'heart':
            chol = random.randint(150, 340)
            bp = random.randint(110, 180)
            risk_pct = round(min(100.0, max(5.0, (chol - 160)*0.35 + (bp - 110)*0.5 + random.randint(0, 30))), 1)
            result = "Positive" if risk_pct >= 50.0 else "Negative"
            input_feat = {"chol": chol, "trestbps": bp, "age": random.randint(35, 75), "sex": random.choice([0, 1])}

        elif d_type == 'kidney':
            sc = round(random.uniform(0.6, 6.5), 2)
            hemo = round(random.uniform(7.0, 16.0), 1)
            risk_pct = round(min(100.0, max(5.0, sc * 16.0 + (15.0 - hemo) * 4.5)), 1)
            result = "Positive" if risk_pct >= 50.0 else "Negative"
            input_feat = {"sc": sc, "hemo": hemo, "bp": random.randint(60, 110), "age": random.randint(25, 80)}

        elif d_type == 'malaria':
            fever_dur = random.randint(1, 10)
            temp = round(random.uniform(98.0, 104.0), 1)
            plt = random.randint(30000, 350000)
            risk_pct = round(min(100.0, max(5.0, fever_dur * 6.5 + (temp - 98.6) * 12.0 - (plt / 10000.0))), 1)
            result = "Positive" if risk_pct >= 50.0 else "Negative"
            input_feat = {"fever_duration": fever_dur, "body_temperature": temp, "platelet_count": plt}

        elif d_type == 'dengue':
            plt = random.randint(20000, 300000)
            hct = round(random.uniform(32.0, 54.0), 1)
            ns1 = random.choice([0, 1])
            risk_pct = round(min(100.0, max(5.0, (300000 - plt)/3200.0 + ns1 * 40.0 + (hct - 38.0)*1.8)), 1)
            result = "Positive" if risk_pct >= 50.0 else "Negative"
            input_feat = {"platelet_count": plt, "hematocrit": hct, "ns1_antigen": ns1}

        cursor.execute('''
            INSERT INTO predictions (timestamp, disease_type, input_features, result, risk_percentage)
            VALUES (?, ?, ?, ?, ?)
        ''', (record_time, d_type, json.dumps(input_feat), result, risk_pct))

    # Add a few heart rate readings too
    for i in range(5):
        days_back = random.randint(0, 5)
        record_time = (now - timedelta(days=days_back, minutes=random.randint(0, 1400))).strftime("%Y-%m-%d %H:%M:%S")
        bpm = random.randint(55, 115)
        if bpm < 60:
            classification = "Bradycardia"
        elif bpm > 100:
            classification = "Tachycardia"
        else:
            classification = "Normal"
        cursor.execute('''
            INSERT INTO predictions (timestamp, disease_type, input_features, result, risk_percentage)
            VALUES (?, ?, ?, ?, ?)
        ''', (record_time, "heart_rate", json.dumps({"bpm": bpm}), classification, None))

    conn.commit()
    conn.close()
    print("Successfully seeded 50 predictions and 5 heart rate logs into predictions.db!")

if __name__ == '__main__':
    seed_sample_predictions()
