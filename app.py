import os
import json
from flask import Flask, render_template, request, jsonify
from database.db import init_db, add_detection, get_history, delete_detection, clear_history, get_statistics
from predict import predictor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cyberbullying-detection-project-key-2026'

# Ensure SQLite DB is initialized on application startup
with app.app_context():
    init_db()

@app.route('/')
def index():
    """Home / Detection Dashboard."""
    return render_template('index.html')

@app.route('/history')
def history():
    """Detection History Page."""
    return render_template('history.html')

@app.route('/statistics')
def statistics():
    """Statistics & Visual Analytics Dashboard Page."""
    return render_template('statistics.html')

@app.route('/model-info')
def model_info():
    """Model Information & Evaluation Metrics Page."""
    return render_template('model_info.html')

@app.route('/about')
def about():
    """About & Technical Architecture Page."""
    return render_template('about.html')

# ==================== API ENDPOINTS ====================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint to handle cyberbullying classification.
    Processes input text, classifies with ML model, and saves to database.
    """
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()

    if not text:
        return jsonify({
            'status': 'error',
            'message': 'Please enter a valid message or comment to analyze.'
        }), 400

    result = predictor.predict(text)

    if result.get('status') == 'error':
        return jsonify(result), 400

    # Save to SQLite database
    try:
        record_id = add_detection(
            text=result['raw_text'],
            cleaned_text=result['cleaned_text'],
            prediction=result['prediction'],
            confidence=result['confidence'],
            severity=result['severity'],
            category=result['category']
        )
        result['record_id'] = record_id
    except Exception as e:
        print(f"[!] Database logging warning: {e}")

    return jsonify(result)

@app.route('/api/history', methods=['GET'])
def api_get_history():
    """API endpoint to fetch detection history with optional search filtering."""
    query = request.args.get('q', '').strip()
    records = get_history(search_query=query if query else None)
    return jsonify({
        'status': 'success',
        'count': len(records),
        'history': records
    })

@app.route('/api/history/<int:record_id>', methods=['DELETE'])
def api_delete_history(record_id):
    """API endpoint to delete a specific history record."""
    success = delete_detection(record_id)
    if success:
        return jsonify({'status': 'success', 'message': f'Record #{record_id} deleted.'})
    return jsonify({'status': 'error', 'message': 'Record not found.'}), 404

@app.route('/api/history/clear', methods=['POST'])
def api_clear_history():
    """API endpoint to wipe all history records."""
    clear_history()
    return jsonify({'status': 'success', 'message': 'All detection history cleared.'})

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """API endpoint for visual dashboard aggregate statistics."""
    stats = get_statistics()
    return jsonify({'status': 'success', 'data': stats})

@app.route('/api/model-metrics', methods=['GET'])
def api_model_metrics():
    """API endpoint for raw model metrics and comparison data."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(base_dir, 'model', 'model_metrics.json')

    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
        return jsonify({'status': 'success', 'metrics': metrics_data})
    else:
        return jsonify({
            'status': 'error',
            'message': 'Model metrics file not found. Run python train_model.py first.'
        }), 444

if __name__ == '__main__':
    print("[*] Starting Cyberbullying Detection Flask Application...")
    print("[*] Open your browser and navigate to: http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)
