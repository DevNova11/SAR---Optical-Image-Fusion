from pathlib import Path

root = Path(r"c:\Users\Shanmukha\Desktop\crc\CRCD-Net")
root.mkdir(parents=True, exist_ok=True)

files = {
    "README.md": """# CRCD-Net

CRCD-Net is a starter project structure for a remote sensing change detection workflow using Earth Engine, preprocessing, fusion, segmentation, and change detection components.

## Structure
- Flask/FastAPI entry point in app.py
- Frontend templates and static assets in frontend/
- Backend services and API routes in backend/
- Remote sensing processing modules in google_earth_engine/, preprocessing/, fusion/, segmentation/, change_detection/, statistics/, and gis/

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `python app.py`
3. Open http://127.0.0.1:5000/
""",
    "requirements.txt": """Flask>=3.0.0\npython-dotenv>=1.0.0\nrequests>=2.31.0\nnumpy>=1.26.0\npandas>=2.2.0\nmatplotlib>=3.8.0\nscikit-learn>=1.4.0\nPillow>=10.0.0\n""",
    "environment.yml": """name: crcd-net\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.11\n  - flask\n  - numpy\n  - pandas\n  - scikit-learn\n  - matplotlib\n  - pip\n  - pip:\n      - python-dotenv\n      - requests\n""",
    ".gitignore": """__pycache__/\n*.py[cod]\n*.db\n*.log\n.env\n.venv/\ninstance/\noutputs/\n""",
    "LICENSE": """MIT License\n\nCopyright (c) 2026\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the \"Software\"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n""",
    "Dockerfile": """FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt ./\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nEXPOSE 5000\nCMD [\"python\", \"app.py\"]\n""",
    "docker-compose.yml": """version: '3.8'\nservices:\n  web:\n    build: .\n    ports:\n      - \"5000:5000\"\n    environment:\n      FLASK_ENV: development\n""",
    "app.py": """from flask import Flask, render_template\nfrom config import Config\n\napp = Flask(__name__)\napp.config.from_object(Config)\n\n@app.route('/')\ndef index():\n    return render_template('index.html')\n\n\nif __name__ == '__main__':\n    app.run(host='0.0.0.0', port=5000, debug=True)\n""",
    "config.py": """import os\n\n\nclass Config:\n    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret')\n    DEBUG = os.getenv('FLASK_ENV', 'development') == 'development'\n""",
    "frontend/templates/base.html": """<!doctype html>\n<html lang=\"en\">\n  <head>\n    <meta charset=\"utf-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n    <title>CRCD-Net</title>\n    <link rel=\"stylesheet\" href=\"{{ url_for('static', filename='css/style.css') }}\">\n  </head>\n  <body>\n    <nav>\n      <h1>CRCD-Net</h1>\n    </nav>\n    <main>\n      {% block content %}{% endblock %}\n    </main>\n    <script src=\"{{ url_for('static', filename='js/map.js') }}\"></script>\n  </body>\n</html>\n""",
    "frontend/templates/index.html": """{% extends 'base.html' %}\n{% block content %}\n  <h2>Welcome to CRCD-Net</h2>\n  <p>This starter UI is ready for integration with your remote sensing workflow.</p>\n{% endblock %}\n""",
    "frontend/templates/loading.html": """{% extends 'base.html' %}\n{% block content %}\n  <div class=\"panel\">\n    <h2>Processing request...</h2>\n    <p>Please wait while the workflow runs.</p>\n  </div>\n{% endblock %}\n""",
    "frontend/templates/dashboard.html": """{% extends 'base.html' %}\n{% block content %}\n  <div class=\"panel\">\n    <h2>Dashboard</h2>\n    <div id=\"dashboard\"></div>\n  </div>\n{% endblock %}\n""",
    "frontend/templates/results.html": """{% extends 'base.html' %}\n{% block content %}\n  <div class=\"panel\">\n    <h2>Results</h2>\n    <p>The workflow output will appear here.</p>\n  </div>\n{% endblock %}\n""",
    "frontend/static/css/style.css": """body { font-family: Arial, sans-serif; margin: 0; background: #f4f7fb; color: #1f2937; }\nnav { background: #0f172a; color: white; padding: 1rem 2rem; }\nmain { padding: 2rem; }\n.panel { background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }\n""",
    "frontend/static/js/map.js": """console.log('CRCD-Net map module loaded');\n""",
    "frontend/static/js/dashboard.js": """console.log('CRCD-Net dashboard module loaded');\n""",
    "backend/api/routes.py": """from flask import Blueprint, jsonify\n\napi = Blueprint('api', __name__, url_prefix='/api')\n\n@api.route('/health')\ndef health():\n    return jsonify({'status': 'ok'})\n""",
    "backend/api/download_api.py": """def download_data():\n    return {'message': 'Download API placeholder'}\n""",
    "backend/api/inference_api.py": """def run_inference():\n    return {'message': 'Inference API placeholder'}\n""",
    "backend/controllers/workflow_controller.py": """class WorkflowController:\n    def run(self):\n        return {'status': 'workflow started'}\n""",
    "backend/services/gee_service.py": """def fetch_gee_data():\n    return {'source': 'Google Earth Engine'}\n""",
    "backend/services/preprocessing_service.py": """def preprocess():\n    return {'status': 'preprocessing placeholder'}\n""",
    "backend/services/fusion_service.py": """def fuse_inputs():\n    return {'status': 'fusion placeholder'}\n""",
    "backend/services/segmentation_service.py": """def segment():\n    return {'status': 'segmentation placeholder'}\n""",
    "backend/services/change_service.py": """def detect_change():\n    return {'status': 'change detection placeholder'}\n""",
    "backend/services/export_service.py": """def export_results():\n    return {'status': 'export placeholder'}\n""",
    "backend/database/history.db": "",
    "google_earth_engine/authenticate.py": """def authenticate():\n    return {'message': 'GEE authentication placeholder'}\n""",
    "google_earth_engine/gee_utils.py": """def get_region():\n    return {'region': 'AOI'}\n""",
    "google_earth_engine/sentinel1.py": """def get_sentinel1_data():\n    return {'source': 'Sentinel-1'}\n""",
    "google_earth_engine/sentinel2.py": """def get_sentinel2_data():\n    return {'source': 'Sentinel-2'}\n""",
    "google_earth_engine/download_images.py": """def download_images():\n    return {'status': 'download placeholder'}\n""",
    "google_earth_engine/cloudmask.py": """def apply_cloud_mask():\n    return {'status': 'cloud mask placeholder'}\n""",
    "google_earth_engine/aoi.py": """def define_aoi():\n    return {'aoi': 'user-defined'}\n""",
    "google_earth_engine/date_filter.py": """def filter_by_date():\n    return {'status': 'date filter placeholder'}\n""",
    "preprocessing/registration.py": """def register_images():\n    return {'status': 'registration placeholder'}\n""",
    "preprocessing/normalization.py": """def normalize():\n    return {'status': 'normalization placeholder'}\n""",
    "preprocessing/resize.py": """def resize_images():\n    return {'status': 'resize placeholder'}\n""",
    "preprocessing/radiometric.py": """def correct_radiometry():\n    return {'status': 'radiometric correction placeholder'}\n""",
    "preprocessing/speckle.py": """def apply_speckle_filter():\n    return {'status': 'speckle filter placeholder'}\n""",
    "preprocessing/cloud_removal.py": """def remove_clouds():\n    return {'status': 'cloud removal placeholder'}\n""",
    "preprocessing/preprocess_pipeline.py": """def run_preprocess_pipeline():\n    return {'status': 'pipeline placeholder'}\n""",
    "fusion/optical_encoder.py": """def encode_optical():\n    return {'status': 'optical encoder placeholder'}\n""",
    "fusion/sar_encoder.py": """def encode_sar():\n    return {'status': 'SAR encoder placeholder'}\n""",
    "fusion/attention.py": """def attention_block():\n    return {'status': 'attention placeholder'}\n""",
    "fusion/fusion_network.py": """def build_fusion_network():\n    return {'status': 'fusion network placeholder'}\n""",
    "fusion/fusion_model.py": """def load_fusion_model():\n    return {'status': 'fusion model placeholder'}\n""",
    "fusion/inference.py": """def run_fusion_inference():\n    return {'status': 'fusion inference placeholder'}\n""",
    "segmentation/segformer.py": """def build_segformer():\n    return {'status': 'segformer placeholder'}\n""",
    "segmentation/classifier.py": """def build_classifier():\n    return {'status': 'classifier placeholder'}\n""",
    "segmentation/predict.py": """def predict_segments():\n    return {'status': 'prediction placeholder'}\n""",
    "segmentation/labels.py": """def prepare_labels():\n    return {'status': 'labels placeholder'}\n""",
    "change_detection/changeformer.py": """def build_changeformer():\n    return {'status': 'changeformer placeholder'}\n""",
    "change_detection/siamese_unet.py": """def build_siamese_unet():\n    return {'status': 'siamese unet placeholder'}\n""",
    "change_detection/compare.py": """def compare_maps():\n    return {'status': 'comparison placeholder'}\n""",
    "change_detection/confidence.py": """def compute_confidence():\n    return {'status': 'confidence placeholder'}\n""",
    "change_detection/postprocess.py": """def postprocess_change_map():\n    return {'status': 'postprocessing placeholder'}\n""",
    "statistics/forest_loss.py": """def compute_forest_loss():\n    return {'status': 'forest loss placeholder'}\n""",
    "statistics/urban_growth.py": """def compute_urban_growth():\n    return {'status': 'urban growth placeholder'}\n""",
    "statistics/water_change.py": """def compute_water_change():\n    return {'status': 'water change placeholder'}\n""",
    "statistics/area.py": """def compute_area():\n    return {'status': 'area placeholder'}\n""",
    "statistics/report.py": """def build_report():\n    return {'status': 'report placeholder'}\n""",
    "gis/geojson_export.py": """def export_geojson():\n    return {'status': 'geojson export placeholder'}\n""",
    "gis/shapefile_export.py": """def export_shapefile():\n    return {'status': 'shapefile export placeholder'}\n""",
    "gis/raster_export.py": """def export_raster():\n    return {'status': 'raster export placeholder'}\n""",
    "gis/map_overlay.py": """def overlay_map():\n    return {'status': 'map overlay placeholder'}\n""",
    "dashboard/dashboard.py": """def build_dashboard():\n    return {'status': 'dashboard placeholder'}\n""",
    "dashboard/charts.py": """def build_chart():\n    return {'status': 'chart placeholder'}\n""",
    "dashboard/leaflet_map.py": """def build_leaflet_map():\n    return {'status': 'leaflet map placeholder'}\n""",
    "dashboard/download.py": """def download_dashboard_data():\n    return {'status': 'download placeholder'}\n""",
    "models/fusion_weights.pth": "placeholder",
    "models/segformer_weights.pth": "placeholder",
    "models/changeformer_weights.pth": "placeholder",
    "datasets/raw/.gitkeep": "",
    "datasets/processed/.gitkeep": "",
    "datasets/labels/.gitkeep": "",
    "datasets/cache/.gitkeep": "",
    "outputs/fused/.gitkeep": "",
    "outputs/classified/.gitkeep": "",
    "outputs/changed/.gitkeep": "",
    "outputs/geojson/.gitkeep": "",
    "outputs/shapefiles/.gitkeep": "",
    "outputs/reports/.gitkeep": "",
    "outputs/screenshots/.gitkeep": "",
    "utils/logger.py": """import logging\n\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger(__name__)\n""",
    "utils/helpers.py": """def get_project_name():\n    return 'CRCD-Net'\n""",
    "utils/visualization.py": """def plot_result():\n    return {'status': 'visualization placeholder'}\n""",
    "utils/constants.py": """DATASET_DIR = 'datasets'\nOUTPUT_DIR = 'outputs'\n""",
    "notebooks/experiments.ipynb": '{"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# CRCD-Net experiments\n", "Use this notebook to prototype workflows."]}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}',
    "docs/architecture.png": "placeholder image file",
    "docs/workflow.png": "placeholder image file",
    "docs/ui_design.png": "placeholder image file",
    "docs/report.pdf": "placeholder report file",
}

for relative_path, content in files.items():
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')

print(f'Created {len(files)} files under {root}')
