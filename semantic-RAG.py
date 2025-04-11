import json
import os
import sqlite3

import sqlite_vss
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Open your database connection.
conn = sqlite3.connect("un_speeches.db")
c = conn.cursor()

# Load the vector-index extension.
print(sqlite_vss.vss_loadable_path())

try:
    conn.enable_load_extension(True)
    sqlite_vss.load(conn)
    print("Vector extension loaded.")
    print(conn.execute("select vss_version()").fetchone()[0])
except Exception as e:
    print("Could not load vector extension:", e)

# Create the virtual table for vector embeddings.
# Here we assume the model returns vectors of dimension 1536.
c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS speeches_vss USING vss0(vector(1536));")
conn.commit()

# Retrieve rows from the main table (including the implicit rowid).
c.execute("SELECT rowid, country, session, year, text FROM speeches LIMIT 3000")
rows = c.fetchall()

for row in rows:
    rowid, country, session, year, text = row
    print(f"Processing rowid {rowid} ({country}, {session}, {year})...")
    # Check if an embedding already exists for this rowid.
    c.execute("SELECT rowid FROM speeches_vss WHERE rowid = ?", (rowid,))
    if c.fetchone():
        print(f"Embedding for rowid {rowid} exists; skipping.")
        continue
    try:
        response = client.embeddings.create(input=text, model="text-embedding-3-small")
        embedding = response.data[0].embedding  # this is a list
        embedding_str = json.dumps(embedding)  # convert list to JSON string
        print(f"Received embedding for rowid {rowid}: {embedding_str[:60]}...")

        c.execute(
            "INSERT INTO speeches_vss(rowid, vector) VALUES (?, ?)",
            (rowid, embedding_str),
        )
        conn.commit()
        print(f"Inserted embedding for rowid {rowid}.")
    except Exception as e:
        print(f"Error for rowid {rowid}: {e}")

conn.close()
print("Done.")
