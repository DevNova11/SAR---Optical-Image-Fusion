from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from backend.database.database import get_connection, initialize_database
from change_detection import compare
from change_detection.visualization import save_change_map, save_difference_map, save_image_visualization
from fusion.baseline import fuse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / 'outputs'


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_slug(value: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9._-]+', '_', value.strip())
    return cleaned.strip('_') or 'analysis'


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


class AnalysisExecutionError(RuntimeError):
    def __init__(self, analysis_id: str, message: str):
        super().__init__(message)
        self.analysis_id = analysis_id


class AnalysisHistoryService:
    def __init__(self) -> None:
        initialize_database()

    def _create_analysis_id(self) -> str:
        return uuid.uuid4().hex

    def _build_aoi(self, latitude: float, longitude: float, buffer_meters: float = 1000.0):
        try:
            import ee
        except ImportError as exc:
            raise RuntimeError('earthengine-api is required for coordinate-based analysis runs') from exc

        point = ee.Geometry.Point([float(longitude), float(latitude)])
        return point.buffer(buffer_meters).bounds()

    def _insert_analysis_run(
        self,
        analysis_id: str,
        payload: dict[str, Any],
        status: str,
        created_at: str,
        completed_at: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO analysis_runs
                (analysis_id, date_1, date_2, latitude, longitude, location_name, status, created_at, completed_at, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    payload['date_1'],
                    payload['date_2'],
                    float(payload['latitude']),
                    float(payload['longitude']),
                    payload.get('location_name'),
                    status,
                    created_at,
                    completed_at,
                    error_message,
                ),
            )
            connection.commit()

    def _update_analysis_run(
        self,
        analysis_id: str,
        status: str,
        completed_at: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        with get_connection() as connection:
            connection.execute(
                """
                UPDATE analysis_runs
                SET status = ?, completed_at = ?, error_message = ?
                WHERE analysis_id = ?
                """,
                (status, completed_at, error_message, analysis_id),
            )
            connection.commit()

    def _upsert_outputs(
        self,
        analysis_id: str,
        fused_image_date_1_path: str,
        fused_image_date_2_path: str,
        change_map_path: str,
        difference_map_path: str,
        created_at: str,
    ) -> None:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO analysis_outputs
                (analysis_id, fused_image_date_1_path, fused_image_date_2_path, change_map_path, difference_map_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    fused_image_date_1_path,
                    fused_image_date_2_path,
                    change_map_path,
                    difference_map_path,
                    created_at,
                ),
            )
            connection.commit()

    def _upsert_metrics(
        self,
        analysis_id: str,
        deforestation: dict[str, Any],
        urbanisation: dict[str, Any],
        created_at: str,
    ) -> None:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO deforestation_metrics
                (analysis_id, forest_loss_area, forest_loss_percentage, changed_regions, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    deforestation.get('forest_loss_area'),
                    deforestation.get('forest_loss_percentage'),
                    deforestation.get('changed_regions'),
                    created_at,
                ),
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO urbanisation_metrics
                (analysis_id, urban_growth_area, urban_growth_percentage, changed_regions, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    urbanisation.get('urban_growth_area'),
                    urbanisation.get('urban_growth_percentage'),
                    urbanisation.get('changed_regions'),
                    created_at,
                ),
            )
            connection.commit()

    def _fetch_one(self, query: str, analysis_id: str) -> Optional[dict[str, Any]]:
        with get_connection() as connection:
            row = connection.execute(query, (analysis_id,)).fetchone()
        return dict(row) if row else None

    def _fetch_many(self, query: str, limit: int) -> list[dict[str, Any]]:
        with get_connection() as connection:
            rows = connection.execute(query, (limit,)).fetchall()
        return [dict(row) for row in rows]

    def _build_metric_payload(self, result, analysis_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        stats = dict(result.statistics)
        metadata = dict(result.metadata)
        direction = metadata.get('direction_heuristics') or {}
        label = str(direction.get('label', '')).lower()
        changed_regions = json.dumps(
            {
                'count': stats.get('num_change_regions'),
                'label': direction.get('label'),
                'confidence': direction.get('confidence'),
                'note': direction.get('note'),
            },
            default=_json_default,
        )

        changed_area_hectares = stats.get('changed_area_hectares')
        change_percentage = stats.get('change_percentage')

        deforestation_area = 0.0
        urban_area = 0.0
        if 'vegetation loss' in label or 'forest' in label or 'deforestation' in label:
            deforestation_area = float(changed_area_hectares or 0.0)
        elif 'urban' in label:
            urban_area = float(changed_area_hectares or 0.0)

        deforestation = {
            'analysis_id': analysis_id,
            'forest_loss_area': deforestation_area,
            'forest_loss_percentage': float(change_percentage or 0.0),
            'changed_regions': changed_regions,
        }
        urbanisation = {
            'analysis_id': analysis_id,
            'urban_growth_area': urban_area,
            'urban_growth_percentage': float(change_percentage or 0.0),
            'changed_regions': changed_regions,
        }
        return deforestation, urbanisation

    def create_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        required_fields = ('latitude', 'longitude', 'date_1', 'date_2')
        missing_fields = [field for field in required_fields if field not in payload]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        latitude = float(payload['latitude'])
        longitude = float(payload['longitude'])
        date_1 = str(payload['date_1'])
        date_2 = str(payload['date_2'])

        try:
            datetime.fromisoformat(date_1)
            datetime.fromisoformat(date_2)
        except ValueError as exc:
            raise ValueError('date_1 and date_2 must be valid ISO-8601 dates in YYYY-MM-DD format') from exc

        location_name = payload.get('location_name')
        if not location_name:
            location_name = _sanitize_slug(f'{latitude:.5f}_{longitude:.5f}')

        analysis_id = self._create_analysis_id()
        created_at = _utcnow_iso()
        base_payload = {
            'date_1': date_1,
            'date_2': date_2,
            'latitude': latitude,
            'longitude': longitude,
            'location_name': location_name,
        }
        self._insert_analysis_run(analysis_id, base_payload, status='pending', created_at=created_at)
        self._update_analysis_run(analysis_id, status='processing')

        try:
            from handoff import get_training_pair

            # Cache-first: cached demo AOIs must never require a live GEE call
            # (this is the whole point of pre-exporting them -- see DATA_CONTRACT.md).
            # Only build an ee.Geometry, which itself requires ee.Initialize(),
            # if get_training_pair actually reports a cache miss.
            try:
                s1_date1, s2_date1, s1_date2, s2_date2 = get_training_pair(
                    None, date_1, date_2, location_name
                )
            except FileNotFoundError:
                aoi = self._build_aoi(latitude, longitude)
                s1_date1, s2_date1, s1_date2, s2_date2 = get_training_pair(
                    aoi, date_1, date_2, location_name
                )
            fused_1 = fuse(s1_date1, s2_date1, method='weighted', data_layout='HWC')
            fused_2 = fuse(s1_date2, s2_date2, method='weighted', data_layout='HWC')

            result = compare(
                fused_1,
                fused_2,
                metadata={
                    'aoi': location_name,
                    'date1': date_1,
                    'date2': date_2,
                    'latitude': latitude,
                    'longitude': longitude,
                    'pixel_size': 10.0,
                    'analysis_id': analysis_id,
                },
                config={'enable_direction_heuristics': True},
            )

            analysis_dir = OUTPUT_ROOT / f'analysis_{analysis_id}'
            fused_dir = analysis_dir / 'fused'
            changed_dir = analysis_dir / 'changed'
            reports_dir = analysis_dir / 'reports'
            fused_dir.mkdir(parents=True, exist_ok=True)
            changed_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(parents=True, exist_ok=True)

            fused_1_path = save_image_visualization(fused_1, 'Fused Image - Date 1', fused_dir / 'fused_date_1.png')
            fused_2_path = save_image_visualization(fused_2, 'Fused Image - Date 2', fused_dir / 'fused_date_2.png')
            change_map_path = save_change_map(result.change_map, changed_dir / 'change_map.png')
            difference_map_path = save_difference_map(result.difference_map, changed_dir / 'difference_map.png')

            report_path = reports_dir / 'analysis_report.json'
            report_payload = {
                'analysis_id': analysis_id,
                'status': 'completed',
                'inputs': {
                    'latitude': latitude,
                    'longitude': longitude,
                    'location_name': location_name,
                    'date_1': date_1,
                    'date_2': date_2,
                },
                'outputs': {
                    'fused_image_date_1_path': fused_1_path,
                    'fused_image_date_2_path': fused_2_path,
                    'change_map_path': change_map_path,
                    'difference_map_path': difference_map_path,
                },
                'statistics': result.statistics,
                'metadata': result.metadata,
            }
            report_path.write_text(json.dumps(report_payload, indent=2, default=_json_default))

            deforestation, urbanisation = self._build_metric_payload(result, analysis_id)
            metric_created_at = _utcnow_iso()
            self._upsert_outputs(
                analysis_id,
                fused_1_path,
                fused_2_path,
                change_map_path,
                difference_map_path,
                metric_created_at,
            )
            self._upsert_metrics(analysis_id, deforestation, urbanisation, metric_created_at)
            completed_at = _utcnow_iso()
            self._update_analysis_run(analysis_id, status='completed', completed_at=completed_at)
            return self.get_analysis(analysis_id) or {
                'analysis_id': analysis_id,
                'status': 'completed',
            }
        except Exception as exc:
            completed_at = _utcnow_iso()
            self._update_analysis_run(analysis_id, status='failed', completed_at=completed_at, error_message=str(exc))
            raise AnalysisExecutionError(analysis_id, f'Analysis {analysis_id} failed: {exc}') from exc

    def get_analysis(self, analysis_id: str) -> Optional[dict[str, Any]]:
        run = self._fetch_one('SELECT * FROM analysis_runs WHERE analysis_id = ?', analysis_id)
        if run is None:
            return None
        outputs = self._fetch_one('SELECT * FROM analysis_outputs WHERE analysis_id = ?', analysis_id)
        deforestation = self._fetch_one('SELECT * FROM deforestation_metrics WHERE analysis_id = ?', analysis_id)
        urbanisation = self._fetch_one('SELECT * FROM urbanisation_metrics WHERE analysis_id = ?', analysis_id)
        return {
            'run': run,
            'outputs': outputs,
            'deforestation_metrics': deforestation,
            'urbanisation_metrics': urbanisation,
        }

    def list_history(self, limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 500))
        return self._fetch_many(
            'SELECT * FROM analysis_runs ORDER BY created_at DESC LIMIT ?',
            safe_limit,
        )
