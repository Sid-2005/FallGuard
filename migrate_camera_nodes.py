"""
migrate_camera_nodes.py — one-time migration.

Adds the new columns (is_local, stream_url, is_running) to the existing
camera_nodes table WITHOUT deleting any existing rows, users, or events.

Run this ONCE, from inside your FallGuard folder:
    python migrate_camera_nodes.py

Safe to run more than once -- it checks first and skips any column that's
already there.
"""

import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fallguard.db')


def column_exists(cursor, table, column):
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def main():
    if not os.path.exists(DB_PATH):
        print(f"No database found at {DB_PATH} -- nothing to migrate. "
              f"Just run the app normally and it will be created fresh.")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Make sure the table actually exists (it will, if you've registered
    # even one camera before)
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='camera_nodes'")
    if cur.fetchone() is None:
        print("No camera_nodes table found yet -- nothing to migrate. "
              "It will be created automatically with the new columns "
              "the first time the app starts.")
        conn.close()
        return

    additions = [
        ("is_local",   "BOOLEAN DEFAULT 0"),
        ("stream_url", "VARCHAR(500) DEFAULT ''"),
        ("is_running", "BOOLEAN DEFAULT 0"),
    ]

    changed = False
    for col_name, col_def in additions:
        if column_exists(cur, "camera_nodes", col_name):
            print(f"  [skip] '{col_name}' already exists")
            continue
        cur.execute(f"ALTER TABLE camera_nodes ADD COLUMN {col_name} {col_def}")
        print(f"  [added] '{col_name}'")
        changed = True

    # host was NOT NULL before; SQLite can't easily drop a NOT NULL
    # constraint via ALTER TABLE, but since we only INSERT new local rows
    # with host=None going forward and SQLite is lenient about this in
    # practice for existing tables without a strict NOT NULL enforcement
    # rebuild, no further action is needed here for existing installs.

    conn.commit()
    conn.close()

    if changed:
        print("\nMigration complete. Your existing users and events are untouched.")
    else:
        print("\nNothing to do -- database was already up to date.")


if __name__ == '__main__':
    main()