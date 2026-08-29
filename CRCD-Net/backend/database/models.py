from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class AnalysisRunRecord:
    analysis_id: str
    date_1: str
    date_2: str
    latitude: float
    longitude: float
    location_name: Optional[str]
    status: str
    created_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    id: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisOutputsRecord:
    analysis_id: str
    fused_image_date_1_path: Optional[str]
    fused_image_date_2_path: Optional[str]
    change_map_path: Optional[str]
    difference_map_path: Optional[str]
    created_at: str
    id: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricRecord:
    analysis_id: str
    created_at: str
    id: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
