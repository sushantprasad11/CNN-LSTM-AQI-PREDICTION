"""
AQI Prediction Flask Web Application
=======================================
A comprehensive dashboard for AQI prediction results, model comparison,
ablation study, meteorological analysis, and early warning system.
"""

import sys
import os

# Configure local Keras and Matplotlib paths to avoid sandbox permission errors
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
keras_dir = os.path.join(workspace_dir, '.keras')
mpl_dir = os.path.join(workspace_dir, '.matplotlib')
os.makedirs(keras_dir, exist_ok=True)
os.makedirs(mpl_dir, exist_ok=True)
os.environ.setdefault('KERAS_HOME', keras_dir)
os.environ.setdefault('MPLCONFIGDIR', mpl_dir)

sys.path.insert(0, workspace_dir)

from flask import Flask, render_template, jsonify, send_from_directory, request
import json
from src.prediction.latest_result import load_latest_prediction, save_latest_prediction

app = Flask(__name__)

# AQI category information
AQI_INFO = {
    0: {"label": "Good", "color": "#009966", "risk": "Minimal", "range": "0–50",
        "advisory": "Air quality is satisfactory. Enjoy outdoor activities."},
    1: {"label": "Satisfactory", "color": "#FFDE33", "risk": "Low", "range": "51–100",
        "advisory": "Sensitive individuals should consider limiting prolonged outdoor exertion."},
    2: {"label": "Moderate", "color": "#FF9933", "risk": "Moderate", "range": "101–200",
        "advisory": "People with respiratory conditions should limit outdoor exertion."},
    3: {"label": "Poor", "color": "#CC0033", "risk": "High", "range": "201–300",
        "advisory": "Avoid prolonged outdoor activities. Use masks if going outside."},
    4: {"label": "Very Poor", "color": "#660099", "risk": "Very High", "range": "301–400",
        "advisory": "Minimize outdoor activities. Use N95 masks. Keep windows closed."},
    5: {"label": "Severe", "color": "#7E0023", "risk": "Severe", "range": "401–500",
        "advisory": "AVOID all outdoor activities. Health emergency for entire population."},
}


def load_results():
    """Load all result files."""
    results = {}
    base = os.path.join(os.path.dirname(__file__), '..')

    # Full summary
    summary_path = os.path.join(base, 'results', 'reports', 'full_results_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            results['summary'] = json.load(f)

    # Ablation results
    ablation_path = os.path.join(base, 'results', 'ablation', 'ablation_results.csv')
    if os.path.exists(ablation_path):
        import csv
        with open(ablation_path, 'r') as f:
            reader = csv.DictReader(f)
            results['ablation'] = list(reader)

    # Meteorological results
    meteo_path = os.path.join(base, 'results', 'meteorological', 'weather_impact.csv')
    if os.path.exists(meteo_path):
        import csv
        with open(meteo_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                # Normalize Unicode Δ in keys to 'Delta' for Jinja2 compatibility
                clean_row = {}
                for k, v in row.items():
                    clean_key = k.replace('Δ', 'Delta')
                    clean_row[clean_key] = v
                rows.append(clean_row)
            results['meteorological'] = rows

    # McNemar tests
    mcnemar_path = os.path.join(base, 'results', 'reports', 'mcnemar_tests.csv')
    if os.path.exists(mcnemar_path):
        import csv
        with open(mcnemar_path, 'r') as f:
            reader = csv.DictReader(f)
            results['mcnemar'] = list(reader)

    # Training histories
    histories = {}
    for name in ['cnn', 'lstm', 'hybrid']:
        hist_path = os.path.join(base, 'results', 'metrics', f'{name}_history.json')
        if os.path.exists(hist_path):
            with open(hist_path, 'r') as f:
                histories[name] = json.load(f)
    results['histories'] = histories

    # Latest live prediction
    results['latest_prediction'] = load_latest_prediction()

    return results


@app.route('/')
def index():
    """Render the main dashboard page."""
    results = load_results()
    return render_template('index.html', aqi_info=AQI_INFO, results=results)


@app.route('/results-plots/<path:filename>')
def serve_plot(filename):
    """Serve plot images from results directory."""
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    # Allow serving from subdirectories
    directory = os.path.join(results_dir, os.path.dirname(filename))
    basename = os.path.basename(filename)
    return send_from_directory(directory, basename)


@app.route('/api/aqi-categories')
def api_aqi_categories():
    """Return AQI category information as JSON."""
    return jsonify(AQI_INFO)


@app.route('/api/results')
def api_results():
    """Return all results as JSON."""
    return jsonify(load_results())


@app.route('/api/project-info')
def api_project_info():
    """Return project metadata."""
    return jsonify({
        "project": "AQI Prediction using Hybrid CNN-LSTM",
        "city": "Delhi, India",
        "coordinates": {"lat": 28.6139, "lon": 77.2090},
        "models": ["CNN", "LSTM", "Hybrid CNN-LSTM"],
        "features": {
            "pollutants": ["PM2.5", "PM10", "NO₂", "SO₂", "CO", "O₃"],
            "weather": ["Temperature", "Humidity", "Wind Speed", "Pressure"],
            "engineered": ["Lag", "Rolling Average", "Cyclic", "Ratio"],
        },
        "aqi_standard": "Indian CPCB",
        "data_source": "Open-Meteo API (Free)",
    })


@app.route('/api/latest-prediction')
def api_latest_prediction():
    """Return the latest prediction result as JSON for live dashboard polling."""
    latest = load_latest_prediction()
    if latest is None:
        return jsonify({"status": "empty", "message": "No predictions available yet."})
    return jsonify({"status": "success", "prediction": latest})


@app.route('/api/run-prediction', methods=['POST'])
def api_run_prediction():
    """Trigger an on-demand date & time prediction directly from the dashboard."""
    data = request.get_json() or {}
    target_dt = data.get('datetime')
    if not target_dt:
        return jsonify({"status": "error", "message": "Please select a valid date and time."}), 400

    try:
        target_dt_clean = str(target_dt).strip().replace('T', ' ')
        if len(target_dt_clean) > 16:
            target_dt_clean = target_dt_clean[:16]
        elif len(target_dt_clean) == 10:
            target_dt_clean += " 12:00"

        from predict_any_datetime import predict_for_datetime
        result = predict_for_datetime(target_dt_clean)
        save_latest_prediction(
            result,
            pred_type="single_datetime",
            command_str=f"Dashboard Run: Predicted for {target_dt_clean}"
        )
        return jsonify({"status": "success", "prediction": load_latest_prediction()})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
