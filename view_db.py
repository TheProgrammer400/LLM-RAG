import json
import sqlite3
from db_manager import PatientMemoryManager


def main():
    pm = PatientMemoryManager()
    
    print("=" * 60)
    print("PATIENT PROFILE SNAPSHOT")
    print("=" * 60)
    print(json.dumps(pm.get_patient_snapshot(), indent=4))
    
    print("\n" + "=" * 60)
    print("MEDICAL TIMELINE EVENTS")
    print("=" * 60)
    timeline = pm.get_timeline(limit=10)
    if not timeline:
        print("No timeline events recorded yet.")
    else:
        for event in timeline:
            print(f"[{event['timestamp']}] {event['event_type']} -> {event['entity_name']} ({event['description']})")

    print("\n" + "=" * 60)
    print("DATABASE TABLES SUMMARY")
    print("=" * 60)
    with pm.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cursor.fetchall() if r[0] != "sqlite_sequence"]
        for t in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {t}")
            count = cursor.fetchone()[0]
            print(f" - {t:<25} : {count} rows")


if __name__ == "__main__":
    main()
