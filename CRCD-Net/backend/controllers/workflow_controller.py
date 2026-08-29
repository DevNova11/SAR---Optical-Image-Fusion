from backend.services.analysis_history_service import AnalysisHistoryService


class WorkflowController:
    def __init__(self, history_service: AnalysisHistoryService | None = None):
        self.history_service = history_service or AnalysisHistoryService()

    def run(self):
        return {'status': 'workflow started'}

    def start_analysis(self, payload: dict) -> dict:
        return self.history_service.create_analysis(payload)

    def get_analysis(self, analysis_id: str):
        return self.history_service.get_analysis(analysis_id)

    def get_history(self, limit: int = 100):
        return self.history_service.list_history(limit=limit)
