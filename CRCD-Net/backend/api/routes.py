from flask import Blueprint, jsonify

api = Blueprint('api', __name__, url_prefix='/api')

@api.route('/health')
def health():
    return jsonify({'status': 'ok'})
