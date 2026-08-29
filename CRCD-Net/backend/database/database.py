from __future__ import annotations

import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / 'history.db'


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS analysis_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id TEXT NOT NULL UNIQUE,
        date_1 TEXT NOT NULL,
        date_2 TEXT NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        location_name TEXT,
        status TEXT NOT NULL CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
        created_at TEXT NOT NULL,
        completed_at TEXT,
        error_message TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS analysis_outputs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id TEXT NOT NULL UNIQUE,
        fused_image_date_1_path TEXT,
        fused_image_date_2_path TEXT,
        change_map_path TEXT,
        difference_map_path TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (analysis_id) REFERENCES analysis_runs (analysis_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS deforestation_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id TEXT NOT NULL UNIQUE,
        forest_loss_area REAL,
        forest_loss_percentage REAL,
        changed_regions TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (analysis_id) REFERENCES analysis_runs (analysis_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS urbanisation_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id TEXT NOT NULL UNIQUE,
        urban_growth_area REAL,
        urban_growth_percentage REAL,
        changed_regions TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (analysis_id) REFERENCES analysis_runs (analysis_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
]


def get_connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA foreign_keys = ON')
    return connection


def initialize_database() -> None:
    with get_connection() as connection:
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.commit()
