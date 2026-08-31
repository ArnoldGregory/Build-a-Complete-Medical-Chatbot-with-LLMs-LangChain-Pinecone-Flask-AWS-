import sqlite3
import os
from langchain_core.tools import tool

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "appointments.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS availability (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_name TEXT NOT NULL,
            available_time TEXT NOT NULL,
            is_booked INTEGER DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            doctor_name TEXT NOT NULL,
            appointment_time TEXT NOT NULL
        )
        """
    )
    conn.commit()

    seed_availability(conn)
    conn.close()


def seed_availability(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS count FROM availability")
    if cur.fetchone()["count"] > 0:
        return

    slots = [
        ("Dr. Kamau", "2026-09-01 09:00:00"),
        ("Dr. Kamau", "2026-09-01 10:00:00"),
        ("Dr. Kamau", "2026-09-02 09:00:00"),
        ("Dr. Kamau", "2026-09-02 14:00:00"),
        ("Dr. Omondi", "2026-09-01 11:00:00"),
        ("Dr. Omondi", "2026-09-01 15:00:00"),
        ("Dr. Omondi", "2026-09-03 10:00:00"),
        ("Dr. Omondi", "2026-09-03 13:00:00"),
    ]
    cur.executemany(
        "INSERT INTO availability (doctor_name, available_time, is_booked) VALUES (?, ?, 0)",
        slots,
    )
    conn.commit()


@tool
def check_clinic_schedule() -> str:
    """Retrieve the currently open (unbooked) appointment slots across all doctors. Use this when a patient asks about available appointment times or the clinic schedule."""
    init_db()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT doctor_name, available_time FROM availability WHERE is_booked = 0 ORDER BY available_time"
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return "There are currently no open appointment slots available."

    return "\n".join(f"{row['doctor_name']}: {row['available_time']}" for row in rows)


@tool
def book_appointment(patient_name: str, doctor_name: str, time_slot: str) -> str:
    """Book an appointment for a patient. Provide the patient name, the doctor name (e.g. 'Dr. Kamau' or 'Dr. Omondi'), and the exact time slot (e.g. '2026-09-01 09:00:00') that was returned by check_clinic_schedule. The chosen slot becomes unavailable and a booking log is recorded."""
    init_db()
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "UPDATE availability SET is_booked = 1 WHERE doctor_name = ? AND available_time = ? AND is_booked = 0",
        (doctor_name, time_slot),
    )
    if cur.rowcount == 0:
        conn.close()
        return f"Sorry, the slot for {doctor_name} at {time_slot} is no longer available or does not exist. Please check the clinic schedule for an open slot."

    cur.execute(
        "INSERT INTO bookings (patient_name, doctor_name, appointment_time) VALUES (?, ?, ?)",
        (patient_name, doctor_name, time_slot),
    )
    conn.commit()
    conn.close()
    return f"Appointment confirmed for {patient_name} with {doctor_name} at {time_slot}."


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
