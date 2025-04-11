import json
import os
import sqlite3
from collections import defaultdict

import sqlite_vss
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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


def get_ukraine_russia_speeches(conn):
    """Get speeches from Ukraine and Russia/USSR."""
    cursor = conn.cursor()

    # Query to get all speeches from Ukraine and Russia/USSR with their embeddings
    query = """
    SELECT s.rowid, s.country, s.country_name, s.session, s.year, 
           COALESCE(s.speaker, 'Unknown') as speaker, 
           v.vector
    FROM speeches s
    JOIN speeches_vss v ON s.rowid = v.rowid
    WHERE s.country_name IN ('Ukraine', 'Russia', 'USSR', 'Russian Federation', 'Soviet Union')
    ORDER BY s.year
    """

    try:
        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            print("No speeches found for Ukraine or Russia/USSR.")
            return {}

        print(f"Found {len(rows)} speeches from Ukraine and Russia/USSR.")

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

        return embeddings_dict
    except Exception as e:
        print(f"Error retrieving speeches: {e}")
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


def compare_speeches_by_year(embeddings_dict):
    """Compare speeches between Ukraine and Russia for each year."""
    # Group speeches by year and country
    speeches_by_year_country = defaultdict(lambda: defaultdict(list))

    for rowid, (metadata, embedding) in embeddings_dict.items():
        year = metadata["year"]
        country_name = metadata["country_name"]

        # Group Russia, USSR, Russian Federation, Soviet Union together
        if country_name in ["Russia", "USSR", "Russian Federation", "Soviet Union"]:
            country_group = "Russia/USSR"
        else:
            country_group = "Ukraine"

        speeches_by_year_country[year][country_group].append(
            (rowid, metadata, embedding)
        )

    # Compare speeches for each year where both countries have speeches
    results = []
    for year in sorted(speeches_by_year_country.keys()):
        countries_data = speeches_by_year_country[year]

        if "Russia/USSR" in countries_data and "Ukraine" in countries_data:
            russia_speeches = countries_data["Russia/USSR"]
            ukraine_speeches = countries_data["Ukraine"]

            # Compare each Ukraine speech with each Russia speech for this year
            for u_rowid, u_metadata, u_embedding in ukraine_speeches:
                for r_rowid, r_metadata, r_embedding in russia_speeches:
                    similarity = compute_cosine_similarity(u_embedding, r_embedding)

                    results.append(
                        {
                            "year": year,
                            "similarity": similarity,
                            "ukraine_speaker": u_metadata["speaker"],
                            "russia_speaker": r_metadata["speaker"],
                            "ukraine_session": u_metadata["session"],
                            "russia_session": r_metadata["session"],
                        }
                    )

    return results


def main():
    conn = setup_db_connection()
    if not conn:
        print("Could not connect to database. Exiting.")
        return

    # Get Ukraine and Russia speeches
    embeddings_dict = get_ukraine_russia_speeches(conn)

    if not embeddings_dict:
        print("No speech embeddings found.")
        return

    # Compare speeches by year
    results = compare_speeches_by_year(embeddings_dict)

    if not results:
        print("No years with speeches from both Ukraine and Russia/USSR.")
        return

    # Group results by year for display
    results_by_year = defaultdict(list)
    for result in results:
        results_by_year[result["year"]].append(result)

    # Display results
    print("\nSimilarity between Ukraine and Russia/USSR speeches by year:")
    print("=" * 70)

    for year in sorted(results_by_year.keys()):
        year_results = results_by_year[year]
        avg_similarity = sum(r["similarity"] for r in year_results) / len(year_results)

        print(f"\nYear: {year} - Average similarity: {avg_similarity:.4f}")
        print("-" * 70)

        for i, result in enumerate(
            sorted(year_results, key=lambda x: x["similarity"], reverse=True)
        ):
            print(f"  Comparison {i + 1}: Similarity: {result['similarity']:.4f}")
            print(
                f"    Ukraine: {result['ukraine_speaker']} (Session {result['ukraine_session']})"
            )
            print(
                f"    Russia/USSR: {result['russia_speaker']} (Session {result['russia_session']})"
            )

    # Calculate overall trend
    years = sorted(results_by_year.keys())
    if len(years) > 1:
        avg_similarities = [
            sum(r["similarity"] for r in results_by_year[y]) / len(results_by_year[y])
            for y in years
        ]
        trend_description = (
            "increasing" if avg_similarities[-1] > avg_similarities[0] else "decreasing"
        )
        print("\nOverall trend in speech similarity: " + trend_description)

        # Identify significant changes between consecutive years
        for i in range(1, len(years)):
            prev_avg = sum(
                r["similarity"] for r in results_by_year[years[i - 1]]
            ) / len(results_by_year[years[i - 1]])
            curr_avg = sum(r["similarity"] for r in results_by_year[years[i]]) / len(
                results_by_year[years[i]]
            )
            change = curr_avg - prev_avg

            if abs(change) > 0.1:  # Threshold for significant change
                direction = "increased" if change > 0 else "decreased"
                print(
                    f"Significant change from {years[i - 1]} to {years[i]}: Similarity {direction} by {abs(change):.4f}"
                )

    conn.close()


if __name__ == "__main__":
    main()
