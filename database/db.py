import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'database', 'cyberbullying.db')

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes SQLite database tables."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            cleaned_text TEXT,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT NOT NULL,
            category TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_detection(text, cleaned_text, prediction, confidence, severity, category):
    """Inserts a new detection record into database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO detections (text, cleaned_text, prediction, confidence, severity, category)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (text, cleaned_text, prediction, float(confidence), severity, category))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def get_history(search_query=None, limit=100):
    """Fetches list of detection records from SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if search_query:
        query = '''
            SELECT * FROM detections
            WHERE text LIKE ? OR category LIKE ? OR prediction LIKE ?
            ORDER BY timestamp DESC LIMIT ?
        '''
        term = f"%{search_query}%"
        cursor.execute(query, (term, term, term, limit))
    else:
        query = 'SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?'
        cursor.execute(query, (limit,))
        
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_detection(record_id):
    """Deletes a specific detection record by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM detections WHERE id = ?', (record_id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0

def clear_history():
    """Clears all detection records from database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM detections')
    conn.commit()
    conn.close()
    return True

def get_statistics():
    """Computes aggregate analytical metrics from stored detections."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM detections')
    total_messages = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM detections WHERE prediction = 'Cyberbullying'")
    cyberbullying_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM detections WHERE prediction = 'Not Cyberbullying'")
    clean_count = cursor.fetchone()[0]

    cyberbullying_percentage = round((cyberbullying_count / total_messages * 100), 1) if total_messages > 0 else 0.0

    # Severity distribution
    cursor.execute('''
        SELECT severity, COUNT(*) FROM detections
        WHERE prediction = 'Cyberbullying'
        GROUP BY severity
    ''')
    severity_dist = {row[0]: row[1] for row in cursor.fetchall()}

    # Category distribution
    cursor.execute('''
        SELECT category, COUNT(*) FROM detections
        GROUP BY category
    ''')
    category_dist = {row[0]: row[1] for row in cursor.fetchall()}

    # Recent activity timeline (last 7 entries)
    cursor.execute('''
        SELECT id, text, prediction, confidence, timestamp FROM detections
        ORDER BY timestamp DESC LIMIT 7
    ''')
    recent_entries = [dict(row) for row in cursor.fetchall()]

    conn.close()

    return {
        'total_messages': total_messages,
        'cyberbullying_count': cyberbullying_count,
        'clean_count': clean_count,
        'cyberbullying_percentage': cyberbullying_percentage,
        'severity_distribution': severity_dist,
        'category_distribution': category_dist,
        'recent_entries': recent_entries
    }
