from flask import Blueprint, jsonify, request

from backend.controllers.workflow_controller import WorkflowController
from backend.services.analysis_history_service import AnalysisExecutionError

api = Blueprint('api', __name__, url_prefix='/api')
controller = WorkflowController()

@api.route('/health')
def health():
    return jsonify({'status': 'ok'})


@api.route('/analysis', methods=['POST'])
def create_analysis():
    payload = request.get_json(silent=True) or {}
    try:
        result = controller.start_analysis(payload)
        return jsonify(result), 201
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except AnalysisExecutionError as exc:
        return jsonify({'analysis_id': exc.analysis_id, 'error': str(exc)}), 500
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 500


@api.route('/analysis/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id: str):
    result = controller.get_analysis(analysis_id)
    if result is None:
        return jsonify({'error': 'analysis not found'}), 404
    return jsonify(result)


@api.route('/analysis/history', methods=['GET'])
def get_history():
    limit = request.args.get('limit', default=100, type=int)
    return jsonify({'items': controller.get_history(limit=limit)})
