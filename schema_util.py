# schema_util.py
import sqlite3
from typing import List
def get_schema_description(db_path: str) -> str:
   """
   Inspect the SQLite database at db_path and return a Markdown description
   of each non-internal table, including row counts and column info.
   """
   conn = sqlite3.connect(db_path)
   cursor = conn.cursor()
   # 1) find all user tables (ignore sqlite_internal)
   cursor.execute(
       "SELECT name FROM sqlite_master "
       "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
       "ORDER BY name;"
   )
   tables: List[str] = [row[0] for row in cursor.fetchall()]
   if not tables:
       conn.close()
       return "No tables found in database."
   md_lines: List[str] = ["# Database Schema\n"]
   for table in tables:
       md_lines.append(f"## `{table}`\n")
       # 2) row count
       try:
           cursor.execute(f"SELECT COUNT(*) FROM `{table}`;")
           count = cursor.fetchone()[0]
           md_lines.append(f"**Rows:** {count}\n")
       except sqlite3.Error:
           md_lines.append("**Rows:** _(could not fetch)_\n")
       # 3) column details
       md_lines.append("| Column | Type | Not Null | Primary Key |")
       md_lines.append("|---|---|:---:|:---:|")
       cursor.execute(f"PRAGMA table_info(`{table}`);")
       for cid, name, coltype, notnull, default_value, pk in cursor.fetchall():
           nn = "✅" if notnull else ""
           key = "✅" if pk else ""
           md_lines.append(f"| {name} | {coltype} | {nn} | {key} |")
       md_lines.append("")  # blank line
   conn.close()
   return "\n".join(md_lines)