
import sqlite3
from datetime import datetime, timezone

DATABASE_NAME = "firegridnet.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_NAME)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_name TEXT,
                detection_count INTEGER NOT NULL DEFAULT 0,
                detections_json TEXT NOT NULL DEFAULT '[]',
                latitude REAL,
                longitude REAL,
                location_source TEXT,
                temperature REAL,
                humidity REAL,
                wind_speed REAL,
                rainfall_24h REAL,
                risk_score REAL,
                risk_category TEXT,
                status TEXT NOT NULL DEFAULT 'New'
            )
        """)


def save_incident(
    image_name,
    detections,
    latitude=None,
    longitude=None,
    location_source=None,
    temperature=None,
    humidity=None,
    wind_speed=None,
    rainfall_24h=None,
    risk_score=None,
    risk_category=None,
):
    import json

    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO incidents (
                timestamp, image_name, detection_count,
                detections_json, latitude, longitude, location_source,
                temperature, humidity, wind_speed, rainfall_24h,
                risk_score, risk_category
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            image_name,
            len(detections),
            json.dumps(detections),
            latitude,
            longitude,
            location_source,
            temperature,
            humidity,
            wind_speed,
            rainfall_24h,
            risk_score,
            risk_category,
        ))
        return cursor.lastrowid


def get_incidents(limit=500):
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT *
            FROM incidents
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

        return [dict(row) for row in rows]
def update_incident_status(incident_id, status):
    allowed_statuses = ["New", "Under Review", "Resolved"]

    if status not in allowed_statuses:
        raise ValueError("Invalid incident status")

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE incidents
            SET status = ?
            WHERE id = ?
            """,
            (status, incident_id),
        )
