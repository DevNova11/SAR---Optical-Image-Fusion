from flask import Flask, jsonify
from config import Config
from backend.api.routes import api

app = Flask(__name__)
app.config.from_object(Config)
app.register_blueprint(api)

@app.route('/')
def index():
    # This Flask app is the API only -- the actual UI is the Streamlit
    # dashboard (CRCD-Net/dashboard/dashboard.py), not a page served here.
    return jsonify({
        'service': 'CRCD-Net backend API',
        'ui': 'Run the Streamlit dashboard separately: cd dashboard && streamlit run dashboard.py',
        'endpoints': ['/api/health', '/api/analysis', '/api/analysis/<id>', '/api/analysis/history'],
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
