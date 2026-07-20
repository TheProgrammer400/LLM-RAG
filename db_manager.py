import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime


class PatientMemoryManager:
    """
    Dedicated Electronic Health Record (EHR) memory engine.
    Manages SQLite storage, normalized schemas, medical timelines,
    and prompt snapshot generation for the AI Physician.
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

            # 1. Patients Demographics Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                age INTEGER,
                gender TEXT,
                height REAL,
                weight REAL,
                blood_group TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # 2. Allergies Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS allergies (
                allergy_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                allergy_name TEXT NOT NULL,
                severity TEXT,
                reaction TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(patient_id, allergy_name),
                FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
            )
            """)

            # 3. Medications Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS medications (
                medication_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                medication_name TEXT NOT NULL,
                dosage TEXT,
                frequency TEXT,
                reason TEXT,
                prescribed_by TEXT,
                start_date TEXT,
                end_date TEXT,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(patient_id, medication_name),
                FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
            )
            """)

            # 4. Surgeries Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS surgeries (
                surgery_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                surgery_name TEXT NOT NULL,
                surgery_date TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(patient_id, surgery_name),
                FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
            )
            """)

            # 5. Family History Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS family_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                detail TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(patient_id, detail),
                FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
            )
            """)

            # 6. Active Conditions Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS active_conditions (
                condition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                condition_name TEXT NOT NULL,
                severity TEXT,
                confidence TEXT,
                source TEXT,
                onset_date TEXT,
                status TEXT DEFAULT 'active',
                metadata_json TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(patient_id, condition_name),
                FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
            )
            """)

            # 7. Suspected Conditions Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS suspected_conditions (
                condition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                condition_name TEXT NOT NULL,
                severity TEXT,
                confidence TEXT,
                source TEXT,
                onset_date TEXT,
                status TEXT DEFAULT 'suspected',
                metadata_json TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(patient_id, condition_name),
                FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
            )
            """)

            # 8. Resolved Conditions Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS resolved_conditions (
                condition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                condition_name TEXT NOT NULL,
                severity TEXT,
                resolved_date TEXT,
                duration TEXT,
                notes TEXT,
                metadata_json TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(patient_id, condition_name),
                FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
            )
            """)

            # 9. Chronic Conditions Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS chronic_conditions (
                condition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                condition_name TEXT NOT NULL,
                severity TEXT,
                metadata_json TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(patient_id, condition_name),
                FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
            )
            """)

            # 10. Consultations Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS consultations (
                consultation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                chief_complaint TEXT,
                assessment TEXT,
                treatment_plan TEXT,
                summary TEXT,
                FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
            )
            """)

            # 11. Conversation Summaries Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                summary TEXT NOT NULL,
                FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
            )
            """)

            # 12. Medical Timeline Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS medical_timeline (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_name TEXT NOT NULL,
                description TEXT,
                source TEXT,
                FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
            )
            """)

            # Ensure default patient 1 exists
            cursor.execute("SELECT patient_id FROM patients WHERE patient_id = 1")
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO patients (patient_id, name, age, gender) VALUES (1, NULL, NULL, NULL)"
                )
            conn.commit()

    def update_patient_demographics(self, patient_id=1, name=None, age=None, gender=None, height=None, weight=None, blood_group=None):
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
            if gender is not None:
                updates.append("gender = ?")
                params.append(gender)
            if height is not None:
                updates.append("height = ?")
                params.append(height)
            if weight is not None:
                updates.append("weight = ?")
                params.append(weight)
            if blood_group is not None:
                updates.append("blood_group = ?")
                params.append(blood_group)

            if updates:
                updates.append("updated_at = CURRENT_TIMESTAMP")
                sql = f"UPDATE patients SET {', '.join(updates)} WHERE patient_id = ?"
                params.append(patient_id)
                cursor.execute(sql, params)
                conn.commit()

    def add_allergy(self, patient_id=1, allergy_name=None, severity=None, reaction=None):
        if not allergy_name:
            return
        clean_name = str(allergy_name).strip().lower()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO allergies (patient_id, allergy_name, severity, reaction) VALUES (?, ?, ?, ?)",
                (patient_id, clean_name, severity, reaction)
            )
            self.record_timeline_event(
                patient_id=patient_id,
                event_type="ALLERGY_ADDED",
                entity_type="allergy",
                entity_name=clean_name,
                description=f"Allergy '{clean_name}' recorded.",
                source="user",
                conn=conn
            )
            conn.commit()

    def add_medication(self, patient_id=1, medication_name=None, dosage=None, frequency=None, reason=None, status="active"):
        if not medication_name:
            return
        clean_name = str(medication_name).strip().lower()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO medications (patient_id, medication_name, dosage, frequency, reason, status) VALUES (?, ?, ?, ?, ?, ?)",
                (patient_id, clean_name, dosage, frequency, reason, status)
            )
            self.record_timeline_event(
                patient_id=patient_id,
                event_type="MEDICATION_STARTED",
                entity_type="medication",
                entity_name=clean_name,
                description=f"Medication '{clean_name}' recorded.",
                source="user",
                conn=conn
            )
            conn.commit()

    def add_surgery(self, patient_id=1, surgery_name=None, surgery_date=None, notes=None):
        if not surgery_name:
            return
        clean_name = str(surgery_name).strip().lower()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO surgeries (patient_id, surgery_name, surgery_date, notes) VALUES (?, ?, ?, ?)",
                (patient_id, clean_name, surgery_date, notes)
            )
            self.record_timeline_event(
                patient_id=patient_id,
                event_type="SURGERY_RECORDED",
                entity_type="surgery",
                entity_name=clean_name,
                description=f"Surgery '{clean_name}' recorded.",
                source="user",
                conn=conn
            )
            conn.commit()

    def add_family_history(self, patient_id=1, detail=None):
        if not detail:
            return
        clean_detail = str(detail).strip().lower()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO family_history (patient_id, detail) VALUES (?, ?)",
                (patient_id, clean_detail)
            )
            self.record_timeline_event(
                patient_id=patient_id,
                event_type="FAMILY_HISTORY_ADDED",
                entity_type="family_history",
                entity_name=clean_detail,
                description=f"Family history item '{clean_detail}' recorded.",
                source="user",
                conn=conn
            )
            conn.commit()

    def add_condition(self, patient_id=1, condition_name=None, category="active", metadata=None, confidence=None, severity=None, source="user"):
        if not condition_name:
            return
        
        c_name = str(condition_name).strip().lower()
        meta_str = json.dumps(metadata) if isinstance(metadata, dict) else (str(metadata) if metadata else None)

        with self.get_connection() as conn:
            cursor = conn.cursor()

            if category == "active":
                # Remove from suspected if promoted
                cursor.execute("DELETE FROM suspected_conditions WHERE patient_id = ? AND condition_name = ?", (patient_id, c_name))
                # Remove from resolved if flare-up
                cursor.execute("DELETE FROM resolved_conditions WHERE patient_id = ? AND condition_name = ?", (patient_id, c_name))
                
                cursor.execute(
                    "INSERT OR REPLACE INTO active_conditions (patient_id, condition_name, severity, confidence, source, metadata_json, last_updated) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                    (patient_id, c_name, severity, confidence, source, meta_str)
                )
                self.record_timeline_event(
                    patient_id=patient_id,
                    event_type="DIAGNOSIS_CONFIRMED" if confidence == "confirmed" else "CONDITION_ACTIVE",
                    entity_type="condition",
                    entity_name=c_name,
                    description=f"Active condition '{c_name}' updated.",
                    source=source,
                    conn=conn
                )

            elif category == "suspected":
                # Do not demote a confirmed diagnosis in active_conditions
                cursor.execute("SELECT condition_id FROM active_conditions WHERE patient_id = ? AND condition_name = ?", (patient_id, c_name))
                if cursor.fetchone():
                    return
                
                cursor.execute("DELETE FROM resolved_conditions WHERE patient_id = ? AND condition_name = ?", (patient_id, c_name))
                cursor.execute(
                    "INSERT OR REPLACE INTO suspected_conditions (patient_id, condition_name, severity, confidence, source, metadata_json, last_updated) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                    (patient_id, c_name, severity, confidence or "suspected", source, meta_str)
                )
                self.record_timeline_event(
                    patient_id=patient_id,
                    event_type="CONDITION_SUSPECTED",
                    entity_type="condition",
                    entity_name=c_name,
                    description=f"Suspected condition '{c_name}' recorded.",
                    source=source,
                    conn=conn
                )

            elif category == "chronic":
                cursor.execute(
                    "INSERT OR REPLACE INTO chronic_conditions (patient_id, condition_name, severity, metadata_json, last_updated) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                    (patient_id, c_name, severity, meta_str)
                )
                self.record_timeline_event(
                    patient_id=patient_id,
                    event_type="CHRONIC_CONDITION_ADDED",
                    entity_type="condition",
                    entity_name=c_name,
                    description=f"Chronic condition '{c_name}' recorded.",
                    source=source,
                    conn=conn
                )

            conn.commit()

    def resolve_condition(self, patient_id=1, condition_name=None, notes=None, metadata=None):
        if not condition_name:
            return
        
        c_name = str(condition_name).strip().lower()
        meta_str = json.dumps(metadata) if isinstance(metadata, dict) else (str(metadata) if metadata else None)

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Remove from active and suspected
            cursor.execute("DELETE FROM active_conditions WHERE patient_id = ? AND condition_name = ?", (patient_id, c_name))
            cursor.execute("DELETE FROM suspected_conditions WHERE patient_id = ? AND condition_name = ?", (patient_id, c_name))

            cursor.execute(
                "INSERT OR REPLACE INTO resolved_conditions (patient_id, condition_name, notes, metadata_json, resolved_date, last_updated) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                (patient_id, c_name, notes, meta_str)
            )
            self.record_timeline_event(
                patient_id=patient_id,
                event_type="CONDITION_RESOLVED",
                entity_type="condition",
                entity_name=c_name,
                description=f"Condition '{c_name}' resolved.",
                source="user",
                conn=conn
            )
            conn.commit()

    def record_consultation(self, patient_id=1, chief_complaint=None, assessment=None, treatment_plan=None, summary=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO consultations (patient_id, chief_complaint, assessment, treatment_plan, summary) VALUES (?, ?, ?, ?, ?)",
                (patient_id, chief_complaint, assessment, treatment_plan, summary)
            )
            conn.commit()

    def save_summary(self, patient_id=1, summary=None):
        if not summary:
            return
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO conversation_summaries (patient_id, summary) VALUES (?, ?)",
                (patient_id, summary)
            )
            conn.commit()

    def get_latest_summary(self, patient_id=1):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT summary FROM conversation_summaries WHERE patient_id = ? ORDER BY summary_id DESC LIMIT 1",
                (patient_id,)
            )
            row = cursor.fetchone()
            return row["summary"] if row else ""

    def record_timeline_event(self, patient_id=1, event_type="EVENT", entity_type="general", entity_name="", description="", source="system", conn=None):
        if conn is not None:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO medical_timeline (patient_id, event_type, entity_type, entity_name, description, source) VALUES (?, ?, ?, ?, ?, ?)",
                (patient_id, event_type, entity_type, entity_name, description, source)
            )
        else:
            with self.get_connection() as c:
                cursor = c.cursor()
                cursor.execute(
                    "INSERT INTO medical_timeline (patient_id, event_type, entity_type, entity_name, description, source) VALUES (?, ?, ?, ?, ?, ?)",
                    (patient_id, event_type, entity_type, entity_name, description, source)
                )

    def get_timeline(self, patient_id=1, limit=50):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM medical_timeline WHERE patient_id = ? ORDER BY event_id DESC LIMIT ?",
                (patient_id, limit)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_patient_snapshot(self, patient_id=1):
        """
        Reconstructs the prompt-ready patient profile snapshot dictionary
        dynamically from the SQLite database tables.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Demographics
            cursor.execute("SELECT name, age, gender FROM patients WHERE patient_id = ?", (patient_id,))
            p_row = cursor.fetchone()
            name = p_row["name"] if p_row else None
            age = p_row["age"] if p_row else None
            gender = p_row["gender"] if p_row else None

            # Lists
            cursor.execute("SELECT allergy_name FROM allergies WHERE patient_id = ?", (patient_id,))
            allergies = [r["allergy_name"] for r in cursor.fetchall()]

            cursor.execute("SELECT medication_name, dosage, frequency FROM medications WHERE patient_id = ?", (patient_id,))
            medications = []
            for r in cursor.fetchall():
                med_str = f"{r['medication_name']} ({r['dosage']})" if r['dosage'] else r['medication_name']
                medications.append(med_str)

            cursor.execute("SELECT surgery_name FROM surgeries WHERE patient_id = ?", (patient_id,))
            surgeries = [r["surgery_name"] for r in cursor.fetchall()]

            cursor.execute("SELECT detail FROM family_history WHERE patient_id = ?", (patient_id,))
            family_history = [r["detail"] for r in cursor.fetchall()]

            # Condition tables
            def parse_condition_rows(sql):
                cursor.execute(sql, (patient_id,))
                result = []
                for row in cursor.fetchall():
                    c_dict = {"name": row["condition_name"]}
                    if row["metadata_json"]:
                        try:
                            meta = json.loads(row["metadata_json"])
                            if isinstance(meta, dict):
                                for k, v in meta.items():
                                    if k != "name":
                                        c_dict[k] = v
                        except Exception:
                            pass
                    result.append(c_dict)
                return result

            active_conditions = parse_condition_rows("SELECT condition_name, metadata_json FROM active_conditions WHERE patient_id = ?")
            suspected_conditions = parse_condition_rows("SELECT condition_name, metadata_json FROM suspected_conditions WHERE patient_id = ?")
            resolved_conditions = parse_condition_rows("SELECT condition_name, metadata_json FROM resolved_conditions WHERE patient_id = ?")
            chronic_conditions = parse_condition_rows("SELECT condition_name, metadata_json FROM chronic_conditions WHERE patient_id = ?")

            return {
                "name": name,
                "age": age,
                "gender": gender,
                "allergies": allergies,
                "chronic_conditions": chronic_conditions,
                "active_conditions": active_conditions,
                "suspected_conditions": suspected_conditions,
                "resolved_conditions": resolved_conditions,
                "medications": medications,
                "surgeries": surgeries,
                "family_history": family_history
            }

    def clear_all_data(self):
        """
        Clears all records across all database tables while preserving table schemas.
        """
        tables = [
            "patients", "allergies", "medications", "surgeries", "family_history",
            "active_conditions", "suspected_conditions", "resolved_conditions",
            "chronic_conditions", "consultations", "conversation_summaries", "medical_timeline"
        ]
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for t in tables:
                cursor.execute(f"DELETE FROM {t};")
            try:
                cursor.execute("DELETE FROM sqlite_sequence;")
            except Exception:
                pass
            conn.commit()
