import csv
import sqlite3
from contextlib import closing

from feedback import DB_PATH

if not DB_PATH.is_file():
    raise SystemExit("No feedback saved yet. Rate an outfit in Drape first.")

output = DB_PATH.with_name("outfit_feedback.csv")

with closing(sqlite3.connect(DB_PATH)) as db:
    cursor = db.execute("""
        SELECT clothing_request, outfit, face_shape, body_shape, skin_depth, rating
        FROM outfit_feedback
        ORDER BY id
    """)
    columns = [column[0] for column in cursor.description]
    rows = cursor.fetchall()

with output.open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(columns)
    writer.writerows(rows)

print(f"Exported {len(rows)} ratings to: {output}")
print(f"Likes: {sum(row[-1] == 'like' for row in rows)}")
print(f"Dislikes: {sum(row[-1] == 'dislike' for row in rows)}")
print(f"Distinct requests: {len({row[0].strip().casefold() for row in rows})}")
print(f"Distinct outfits: {len({row[1] for row in rows})}")