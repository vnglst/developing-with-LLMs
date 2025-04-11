import json
import sqlite3

import sqlite_vss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the vector-index extension.
print(sqlite_vss.vss_loadable_path())


def setup_db_connection():
    """Set up the database connection."""
    try:
        conn = sqlite3.connect("un_speeches.db")

        # Load the vector-index extension
        conn.enable_load_extension(True)
        sqlite_vss.load(conn)
        print("Vector extension loaded successfully.")

        # Verify VSS version
        try:
            version = conn.execute("SELECT vss_version()").fetchone()[0]
            print(f"SQLite VSS version: {version}")
        except sqlite3.OperationalError as e:
            print(f"Could not check VSS version: {e}")

        print("Database connection established successfully.")
        return conn
    except Exception as e:
        print(f"Error setting up database connection: {e}")
        return None


def get_speech_embeddings(conn, limit=None):
    """Get speech embeddings from the database."""
    cursor = conn.cursor()

    limit_clause = f"LIMIT {limit}" if limit else ""

    query = f"""
    SELECT s.rowid, s.country, s.country_name, s.session, s.year, 
           COALESCE(s.speaker, 'Unknown') as speaker, 
           v.vector
    FROM speeches s
    JOIN speeches_vss v ON s.rowid = v.rowid
    {limit_clause}
    """

    try:
        cursor.execute(query)
        rows = cursor.fetchall()

        embeddings_dict = {}
        for row in rows:
            rowid, country, country_name, session, year, speaker, vector_str = row
            metadata = {
                "country": country,
                "country_name": country_name,
                "session": session,
                "year": year,
                "speaker": speaker,
            }

            # Try different methods to parse the vector data
            try:
                # First try to parse as JSON string
                embedding = json.loads(vector_str)
            except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                try:
                    # If that fails, try to interpret as binary data
                    if isinstance(vector_str, bytes):
                        # Attempt to convert binary to string representation
                        import struct

                        # Assuming it's stored as array of 32-bit floats
                        vector_size = len(vector_str) // 4  # 4 bytes per float
                        embedding = list(struct.unpack(f"{vector_size}f", vector_str))
                    else:
                        print(
                            f"Unknown vector format for rowid {rowid}: {type(vector_str)}"
                        )
                        continue
                except Exception as inner_e:
                    print(f"Could not parse vector for rowid {rowid}: {inner_e}")
                    continue

            embeddings_dict[rowid] = (metadata, embedding)

        print(f"Retrieved {len(embeddings_dict)} speech embeddings")
        return embeddings_dict
    except Exception as e:
        print(f"Error retrieving speech embeddings: {e}")
        import traceback

        traceback.print_exc()
        return {}


def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = sum(a * a for a in v1) ** 0.5
    norm_v2 = sum(b * b for b in v2) ** 0.5

    if norm_v1 == 0 or norm_v2 == 0:
        return 0

    return dot_product / (norm_v1 * norm_v2)


def find_similar_speeches(embeddings_dict, threshold=0.8):
    """Find groups of similar speeches based on cosine similarity."""
    speech_ids = list(embeddings_dict.keys())
    similar_groups = []

    # Track processed speeches to avoid duplicates
    processed = set()

    for i, id1 in enumerate(speech_ids):
        if id1 in processed:
            continue

        metadata1, embedding1 = embeddings_dict[id1]

        # Find all speeches similar to this one
        similar_speeches = []
        for id2 in speech_ids[i + 1 :]:
            metadata2, embedding2 = embeddings_dict[id2]
            similarity = compute_cosine_similarity(embedding1, embedding2)

            if similarity >= threshold:
                similar_speeches.append((id2, similarity, metadata2))

        # If we found similar speeches, create a group
        if similar_speeches:
            group = [(id1, 1.0, metadata1)] + similar_speeches
            similar_groups.append(group)
            processed.add(id1)
            processed.update(id2 for id2, _, _ in similar_speeches)

    return similar_groups


def main():
    conn = setup_db_connection()
    if not conn:
        print("Could not connect to database. Exiting.")
        return

    # Get embeddings
    limit = int(
        input("Enter maximum number of speeches to analyze (or 0 for all): ") or "100"
    )
    embeddings_dict = get_speech_embeddings(conn, limit if limit > 0 else None)

    if not embeddings_dict:
        print("No speech embeddings found.")
        return

    # Find similar speech groups
    similarity_threshold = float(
        input("Enter similarity threshold (0.0-1.0): ") or "0.8"
    )
    similar_groups = find_similar_speeches(embeddings_dict, similarity_threshold)

    # Display results
    print(f"\nFound {len(similar_groups)} groups of similar speeches:")
    print("=" * 50)

    for i, group in enumerate(similar_groups, 1):
        print(f"Group {i} - {len(group)} similar speeches:")
        for speech_id, similarity, metadata in sorted(
            group, key=lambda x: x[1], reverse=True
        ):
            print(
                f"   {metadata['country_name']} ({metadata['year']}, Session {metadata['session']}) - Similarity: {similarity:.4f}"
            )
        print("-" * 50)

    conn.close()


if __name__ == "__main__":
    main()
