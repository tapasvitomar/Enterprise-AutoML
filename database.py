import sqlite3

DB_NAME = "automl.db"

def create_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS projects(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_name TEXT,
        algorithm TEXT,
        accuracy REAL,
        train_time TEXT,
        date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def save_project(project_name, algorithm, accuracy, train_time):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO projects
    (project_name, algorithm, accuracy, train_time)
    VALUES (?, ?, ?, ?)
    """, (project_name, algorithm, accuracy, train_time))

    conn.commit()
    conn.close()


def load_projects():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM projects ORDER BY id DESC")

    data = cursor.fetchall()

    conn.close()

    return data