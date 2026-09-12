from app.config import settings
import psycopg

conn_url = settings.DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
conn = psycopg.connect(conn_url)
cur = conn.cursor()
cur.execute("SELECT indexname FROM pg_indexes WHERE schemaname='public'")
print([row[0] for row in cur.fetchall()])
conn.close()
