import json
import os
import sqlite3

import dotenv
import sqlite_vss
from openai import OpenAI

# Load environment variables
dotenv.load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def setup_db_connection():
    """Set up the database connection with vector search capabilities."""
    try:
        conn = sqlite3.connect("un_speeches.db")

        # Load the vector-index extension
        conn.enable_load_extension(True)
        sqlite_vss.load(conn)
        print("Vector extension loaded successfully.")

        # Initialize vector tables if needed
        setup_vector_tables(conn)

        return conn
    except Exception as e:
        print(f"Error setting up database connection: {e}")
        return None


def setup_vector_tables(conn):
    """Set up vector search tables if they don't exist."""
    try:
        cursor = conn.cursor()

        # Check if speeches_vss table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='speeches_vss'"
        )
        if not cursor.fetchone():
            print("Creating vector search table speeches_vss...")
            # Get available functions to diagnose
            cursor.execute("SELECT * FROM sqlite_master WHERE type='function'")
            functions = cursor.fetchall()
            print(f"Available SQLite functions: {[f[1] for f in functions]}")

            # Create the vector search table
            cursor.execute("""
            CREATE VIRTUAL TABLE speeches_vss USING vss0(
                vector(1536),
                rowid INTEGER PRIMARY KEY
            );
            """)

            # Get all speeches without embeddings
            cursor.execute("SELECT rowid, text FROM speeches")
            speeches = cursor.fetchall()

            print(f"Generating embeddings for {len(speeches)} speeches...")
            for rowid, text in speeches:
                # Check if embedding already exists
                cursor.execute("SELECT rowid FROM speeches_vss WHERE rowid=?", (rowid,))
                if cursor.fetchone():
                    continue

                # Generate embedding and store it
                embedding = generate_embedding(text)
                if embedding:
                    embedding_json = json.dumps(embedding)
                    cursor.execute(
                        "INSERT INTO speeches_vss (rowid, vector) VALUES (?, ?)",
                        (rowid, embedding_json),
                    )

            conn.commit()
            print("Vector search table setup complete.")

        else:
            print("Vector search table already exists.")

        # List available vector functions for debugging
        try:
            cursor.execute("SELECT vss_version()")
            version = cursor.fetchone()
            print(f"SQLite VSS version: {version[0]}")
        except sqlite3.OperationalError as e:
            print(f"Could not check VSS version: {e}")

    except Exception as e:
        print(f"Error setting up vector tables: {e}")


def generate_embedding(text):
    """Generate an embedding for the given text using OpenAI API."""
    try:
        response = client.embeddings.create(input=text, model="text-embedding-3-small")
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def search_similar_speeches(conn, query_embedding, limit=2):
    """Search for speeches similar to the query embedding."""
    try:
        # Convert embedding to JSON string for SQLite
        query_embedding_str = json.dumps(query_embedding)

        # Execute vector similarity search
        cursor = conn.cursor()

        # Try to get available vector functions
        try:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='function' AND name LIKE '%vss%' OR name LIKE '%vector%'"
            )
            functions = cursor.fetchall()
            print(f"Available vector functions: {[f[0] for f in functions]}")
        except:
            pass

        # Join with the main speeches table to get the full text and metadata
        # Use vector_distance instead of vss_distance
        query = """
        SELECT s.rowid, s.country, s.session, s.year, s.speaker, s.text, 
               vector_distance(v.vector, ?) as distance
        FROM speeches s
        JOIN speeches_vss v ON s.rowid = v.rowid
        ORDER BY distance ASC
        LIMIT ?
        """

        try:
            cursor.execute(query, (query_embedding_str, limit))
            results = cursor.fetchall()
            return results
        except sqlite3.OperationalError as e:
            if "no such function" in str(e):
                # Try alternative function name
                alternative_query = """
                SELECT s.rowid, s.country, s.session, s.year, s.speaker, s.text, 
                       vss_search(v.vector, ?) as distance
                FROM speeches s
                JOIN speeches_vss v ON s.rowid = v.rowid
                ORDER BY distance ASC
                LIMIT ?
                """
                cursor.execute(alternative_query, (query_embedding_str, limit))
                results = cursor.fetchall()
                return results
            else:
                raise e

    except Exception as e:
        print(f"Error searching similar speeches: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        return []


def format_sources(sources):
    """Format source information for display."""
    formatted = []
    for i, source in enumerate(sources):
        rowid, country, session, year, speaker = source[0:5]
        formatted.append(
            f"{i + 1}. Speech by {speaker} ({country}, {year}, Session {session}) [ID: {rowid}]"
        )
    return "\n".join(formatted)


def generate_answer(query, context_texts):
    """Generate an answer based on the query and context using OpenAI."""
    try:
        # Prepare system message and context
        system_message = "You are a helpful assistant that answers questions about UN speeches based solely on the provided context. If you don't know the answer based on the context, say so."

        # Combine context texts
        combined_context = "\n\n".join(
            [f"Context {i + 1}:\n{text}" for i, text in enumerate(context_texts)]
        )

        # Create the messages for the chat
        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Context information is below.\n\n{combined_context}\n\nQuestion: {query}\n\nAnswer the question based only on the provided context:",
            },
        ]

        # Generate completion
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=0
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "I encountered an issue while processing your question."


def chat_loop():
    """Main chat loop for interacting with the UN speeches database."""
    print("Welcome to UN Speeches Chat!")
    print("Ask questions about UN speeches and get answers based on their content.")
    print("Type 'exit' or 'quit' to end the chat.\n")

    # Set up database connection
    conn = setup_db_connection()
    if not conn:
        print("Failed to set up database connection. Exiting.")
        return

    chat_history = []

    while True:
        # Get user input
        user_input = input("You: ")

        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit"]:
            print("Thank you for using UN Speeches Chat. Goodbye!")
            break

        try:
            print("Processing your question...")

            # Generate embedding for the query
            query_embedding = generate_embedding(user_input)
            if not query_embedding:
                print("Failed to generate embedding for your question.")
                continue

            # Search for similar speeches
            similar_speeches = search_similar_speeches(conn, query_embedding)
            if not similar_speeches:
                print("No relevant speeches found.")
                continue

            context = "<speeches>\n"
            for speech in similar_speeches:
                _rowid, country, session, year, speaker, text = speech[0:6]
                context += "<speech>\n"
                context += f"<country>{country}</country>\n"
                context += f"<session>{session}</session>\n"
                context += f"<year>{year}</year>\n"
                context += f"<speaker>{speaker}</speaker>\n"
                context += f"<text>{text}</text>\n"
                context += "</speech>\n"
            context += "</speeches>"

            print("\nContext:")
            print(context)

            # Generate answer based on query and context
            answer = generate_answer(user_input, context)

            # Display the answer and sources
            print("\nAnswer:")
            print(answer)

            if similar_speeches:
                print("\nSources:")
                sources_text = format_sources(similar_speeches)
                print(sources_text)

            # Add to chat history
            chat_history.append({"question": user_input, "answer": answer})

            print("\n" + "-" * 50 + "\n")

        except Exception as e:
            print(f"Error processing query: {e}")
            print("An error occurred while processing your question.")

    # Close the database connection
    conn.close()


if __name__ == "__main__":
    chat_loop()
