import sqlite3
import json
import os
from datetime import datetime, timedelta

DB_PATH = 'predictions.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize SQLite database and create predictions table if not exists."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            disease_type TEXT NOT NULL,
            input_features TEXT NOT NULL,
            result TEXT NOT NULL,
            risk_percentage REAL
        )
    ''')
    conn.commit()
    conn.close()
    print("Database predictions.db initialized successfully.")

def log_prediction(disease_type, input_data, result, risk_percentage):
    """Log a prediction entry into SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    input_json = json.dumps(input_data)
    
    cursor.execute('''
        INSERT INTO predictions (timestamp, disease_type, input_features, result, risk_percentage)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp, disease_type, input_json, result, risk_percentage))
    
    conn.commit()
    conn.close()
    return True

def get_dashboard_stats():
    """
    Computes aggregated JSON statistics from predictions.db for the dashboard endpoint.
    If database has no records, returns is_empty: True with zeroed structure.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_count = cursor.fetchone()[0]
    
    diseases = ['diabetes', 'heart', 'kidney', 'malaria', 'dengue']
    disease_display_names = {
        'diabetes': 'Diabetes Risk',
        'heart': 'Heart Disease',
        'kidney': 'Chronic Kidney',
        'malaria': 'Malaria Risk',
        'dengue': 'Dengue Risk',
        'heart_rate': 'Live Heart Rate'
    }

    if total_count == 0:
        conn.close()
        summary_cards = []
        for d in diseases:
            summary_cards.append({
                'disease': disease_display_names[d],
                'disease_key': d,
                'total_assessments': 0,
                'positive_percentage': 0.0,
                'trend': 'neutral'
            })
        return {
            'is_empty': True,
            'summary_cards': summary_cards,
            'assessments_by_disease': {
                'labels': [disease_display_names[d] for d in diseases],
                'counts': [0, 0, 0, 0, 0]
            },
            'risk_distribution': {
                'positive_pct': 0.0,
                'negative_pct': 0.0
            },
            'assessments_over_time': {
                'dates': [],
                'series': {d: [] for d in diseases}
            },
            'recent_activity': []
        }

    # Summary cards per disease
    summary_cards = []
    disease_counts = {}
    for d in diseases:
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE disease_type = ?", (d,))
        count = cursor.fetchone()[0]
        disease_counts[d] = count

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE disease_type = ? AND (result = 'Positive' OR result LIKE 'High%')", (d,))
        pos_count = cursor.fetchone()[0]
        pos_pct = round((pos_count / count * 100), 1) if count > 0 else 0.0

        # Calculate simple trend (compare last 7 days vs previous 7 days)
        now = datetime.now()
        seven_days_ago = (now - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        fourteen_days_ago = (now - timedelta(days=14)).strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE disease_type = ? AND timestamp >= ?", (d, seven_days_ago))
        recent_period = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE disease_type = ? AND timestamp >= ? AND timestamp < ?", (d, fourteen_days_ago, seven_days_ago))
        prior_period = cursor.fetchone()[0]

        if recent_period > prior_period:
            trend = 'up'
        elif recent_period < prior_period:
            trend = 'down'
        else:
            trend = 'stable'

        summary_cards.append({
            'disease': disease_display_names[d],
            'disease_key': d,
            'total_assessments': count,
            'positive_percentage': pos_pct,
            'trend': trend
        })

    # Overall Risk Distribution
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE result = 'Positive' OR result LIKE 'High%' OR result = 'Tachycardia'")
    total_positive = cursor.fetchone()[0]
    pos_dist_pct = round((total_positive / total_count * 100), 1) if total_count > 0 else 0.0
    neg_dist_pct = round(100.0 - pos_dist_pct, 1) if total_count > 0 else 0.0

    # Assessments over time (grouped by date)
    cursor.execute("SELECT DISTINCT date(timestamp) as log_date FROM predictions ORDER BY log_date ASC")
    dates_rows = cursor.fetchall()
    dates = [row['log_date'] for row in dates_rows if row['log_date']]

    series = {d: [] for d in diseases}
    for date_str in dates:
        for d in diseases:
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE disease_type = ? AND date(timestamp) = ?", (d, date_str))
            series[d].append(cursor.fetchone()[0])

    # Recent activity table (last 15 records)
    cursor.execute("SELECT timestamp, disease_type, result, risk_percentage FROM predictions ORDER BY id DESC LIMIT 15")
    recent_rows = cursor.fetchall()
    recent_activity = []
    for r in recent_rows:
        d_type = r['disease_type']
        recent_activity.append({
            'timestamp': r['timestamp'],
            'disease_type': disease_display_names.get(d_type, d_type.replace('_', ' ').title()),
            'result': r['result'],
            'risk_percentage': round(r['risk_percentage'], 1) if r['risk_percentage'] is not None else None
        })

    conn.close()

    return {
        'is_empty': False,
        'summary_cards': summary_cards,
        'assessments_by_disease': {
            'labels': [disease_display_names[d] for d in diseases],
            'counts': [disease_counts[d] for d in diseases]
        },
        'risk_distribution': {
            'positive_pct': pos_dist_pct,
            'negative_pct': neg_dist_pct
        },
        'assessments_over_time': {
            'dates': dates,
            'series': series
        },
        'recent_activity': recent_activity
    }
