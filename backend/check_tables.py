import os
import psycopg

db_url = os.getenv("DATABASE_URL", "postgresql://postgres:change_me_in_env@localhost:5432/marine_debris_db")
conn = psycopg.connect(db_url)
cur = conn.cursor()
cur.execute("SELECT indexname FROM pg_indexes WHERE schemaname='public'")
print([row[0] for row in cur.fetchall()])
conn.close()
