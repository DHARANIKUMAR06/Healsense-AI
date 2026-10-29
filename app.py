from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import sys

from backend.database import init_db, log_prediction, get_dashboard_stats
from backend.predict_diabetes import predict_diabetes
from backend.predict_heart import predict_heart
from backend.predict_kidney import predict_kidney
from backend.predict_malaria import predict_malaria
from backend.predict_dengue import predict_dengue

app = Flask(__name__)

# Ensure DB is initialized on startup
init_db()

@app.route('/')
def index():
    return redirect(url_for('diabetes_risk'))

# -------------------------------------------------------------
# 1. DIABETES RISK MODULE
# -------------------------------------------------------------
@app.route('/diabetes-risk', methods=['GET', 'POST'])
def diabetes_risk():
    if request.method == 'POST':
        try:
            data = request.get_json(force=True) if request.is_json else request.form.to_dict()
            res = predict_diabetes(data)
            log_prediction('diabetes', data, res['result'], res['risk_percentage'])
            return jsonify({
                'status': 'success',
                'result': res['result'],
                'risk_percentage': res['risk_percentage'],
                'shap_values': res['shap_values']
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400
    return render_template('diabetes.html', active_page='diabetes')

# -------------------------------------------------------------
# 2. HEART DISEASE RISK MODULE
# -------------------------------------------------------------
@app.route('/heart-risk', methods=['GET', 'POST'])
def heart_risk():
    if request.method == 'POST':
        try:
            data = request.get_json(force=True) if request.is_json else request.form.to_dict()
            res = predict_heart(data)
            log_prediction('heart', data, res['result'], res['risk_percentage'])
            return jsonify({
                'status': 'success',
                'result': res['result'],
                'risk_percentage': res['risk_percentage'],
                'shap_values': res['shap_values']
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400
    return render_template('heart.html', active_page='heart')

# -------------------------------------------------------------
# 3. CHRONIC KIDNEY DISEASE RISK MODULE
# -------------------------------------------------------------
@app.route('/kidney-risk', methods=['GET', 'POST'])
def kidney_risk():
    if request.method == 'POST':
        try:
            data = request.get_json(force=True) if request.is_json else request.form.to_dict()
            res = predict_kidney(data)
            log_prediction('kidney', data, res['result'], res['risk_percentage'])
            return jsonify({
                'status': 'success',
                'result': res['result'],
                'risk_percentage': res['risk_percentage'],
                'shap_values': res['shap_values']
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400
    return render_template('kidney.html', active_page='kidney')

# -------------------------------------------------------------
# 4. MALARIA RISK MODULE
# -------------------------------------------------------------
@app.route('/malaria-risk', methods=['GET', 'POST'])
def malaria_risk():
    if request.method == 'POST':
        try:
            data = request.get_json(force=True) if request.is_json else request.form.to_dict()
            res = predict_malaria(data)
            log_prediction('malaria', data, res['result'], res['risk_percentage'])
            return jsonify({
                'status': 'success',
                'result': res['result'],
                'risk_percentage': res['risk_percentage'],
                'shap_values': res['shap_values']
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400
    return render_template('malaria.html', active_page='malaria')

# -------------------------------------------------------------
# 5. DENGUE RISK MODULE
# -------------------------------------------------------------
@app.route('/dengue-risk', methods=['GET', 'POST'])
def dengue_risk():
    if request.method == 'POST':
        try:
            data = request.get_json(force=True) if request.is_json else request.form.to_dict()
            res = predict_dengue(data)
            log_prediction('dengue', data, res['result'], res['risk_percentage'])
            return jsonify({
                'status': 'success',
                'result': res['result'],
                'risk_percentage': res['risk_percentage'],
                'shap_values': res['shap_values']
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400
    return render_template('dengue.html', active_page='dengue')

# -------------------------------------------------------------
# 6. LIVE HEART RATE MONITOR MODULE
# -------------------------------------------------------------
@app.route('/heart-rate-monitor')
def heart_rate_monitor():
    return render_template('heart_rate.html', active_page='heart_rate')

@app.route('/log-heart-rate', methods=['POST'])
def log_heart_rate():
    try:
        data = request.get_json(force=True)
        bpm = data.get('bpm')
        classification = data.get('classification', 'Normal')
        log_prediction('heart_rate', {'bpm': bpm}, classification, None)
        return jsonify({'status': 'success', 'message': 'Heart rate logged successfully.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# -------------------------------------------------------------
# 7. DASHBOARD OVERVIEW & API STATS
# -------------------------------------------------------------
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', active_page='dashboard')

@app.route('/api/dashboard-stats', methods=['GET'])
def dashboard_stats():
    stats = get_dashboard_stats()
    return jsonify(stats)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting HealthSense AI Flask Server on port {port}...")
    app.run(host='0.0.0.0', port=port)

