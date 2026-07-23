import sqlite3
from contextlib import contextmanager
import os


class PatientMemoryManager:
    """
    Dedicated Electronic Health Record (EHR) memory engine.
    Manages SQLite storage, normalized schemas, and patient demographics.
    Stores ONLY name and age of the patient. No medical data is stored.
    """

    def __init__(self, db_path="database/patient_memory.db"):
        self.db_path = db_path
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Enable Write-Ahead Logging (WAL) and busy timeout to avoid database locks
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA busy_timeout=30000;")

            # Patients Table (Name & Age ONLY)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                age INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Ensure default patient 1 exists
            cursor.execute("SELECT patient_id FROM patients WHERE patient_id = 1")
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO patients (patient_id, name, age) VALUES (1, NULL, NULL)"
                )
            conn.commit()

    def update_patient_demographics(self, patient_id=1, name=None, age=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            updates = []
            params = []

            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if age is not None:
                updates.append("age = ?")
                params.append(age)

            if updates:
                updates.append("updated_at = CURRENT_TIMESTAMP")
                sql = f"UPDATE patients SET {', '.join(updates)} WHERE patient_id = ?"
                params.append(patient_id)
                cursor.execute(sql, params)
                conn.commit()

    def get_patient_snapshot(self, patient_id=1):
        """
        Reconstructs the patient profile snapshot dictionary (name and age only).
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT name, age FROM patients WHERE patient_id = ?", (patient_id,))
            p_row = cursor.fetchone()
            name = p_row["name"] if p_row else None
            age = p_row["age"] if p_row else None

            return {
                "name": name,
                "age": age
            }

    def clear_all_data(self):
        """
        Clears patient records and resets default row.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM patients;")
            cursor.execute("INSERT INTO patients (patient_id, name, age) VALUES (1, NULL, NULL)")
            try:
                cursor.execute("DELETE FROM sqlite_sequence;")
            except Exception:
                pass
            conn.commit()
