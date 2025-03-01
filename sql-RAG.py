import os
import re
import sqlite3

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

conn = sqlite3.connect("un_speeches.db")


def read_schema(conn):
    """Retrieve database schema information."""
    cur = conn.cursor()
    cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    return cur.fetchall()


def execute_sql(conn, query):
    """Execute a SQL query and return results."""
    print(f"\n\033[90m[DEBUG] SQL Query:\n{query}\033[0m")
    cur = conn.cursor()
    try:
        cur.execute(query)
        result = cur.fetchall()
        print(f"\n\033[90m[DEBUG] SQL Result:\n{result}\033[0m")
        return result
    except Exception as e:
        return f"Error: {e}"


def extract_sql(text):
    """Extract SQL queries from markdown-style blocks."""
    pattern = r"```sql(.*?)```"
    return [match.strip() for match in re.findall(pattern, text, re.DOTALL)]


def call_gpt(messages):
    """Call GPT-4o to generate a response based on conversation history."""
    response = client.chat.completions.create(model="gpt-4o", messages=messages)
    return response.choices[0].message.content


def main():
    schema = read_schema(conn)
    schema_info = "\n".join([f"{name}: {sql}" for name, sql in schema])

    system_message = (
        "You are a SQL analysis assistant with access to a database of UN speeches."
        "Your primary task is to answer user questions by first exploring the data through a series of SQL queries."
        "Before executing any queries, generate a detailed plan outlining your approach—list the necessary steps, including inspecting table names, column distributions, "
        "and performing fuzzy text searches to match relevant terms even if the input text is imprecise."
        "During the exploratory phase, use SQL queries (enclosed in markdown code blocks labeled 'sql') to gather evidence from the database. "
        "Base your reasoning on this evidence and refine your queries iteratively."
        "Only when you have enough verified evidence, execute the final SQL query to extract the answer."
        "Once you are confident in your answer, provide the final response in plain text by starting your message with 'FINAL ANSWER:' and do not include any SQL code."
        "If, after exploring, no answer can be found, clearly inform the user that the data does not support an answer and explain that no evidence was found."
        "\n\nDatabase schema:\n" + schema_info
    )

    messages = [{"role": "system", "content": system_message}]

    print("Welcome! Type your question below. Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        messages.append({"role": "user", "content": user_input})

        total_queries = 0

        # Begin an inner loop to let the assistant refine its answer.
        while True:
            total_queries += 1

            if total_queries > 5:
                print("\nAssistant: I'm sorry, but I couldn't find an answer.")
                break

            assistant_reply = call_gpt(messages)
            # print("\n[DEBUG] Assistant:", assistant_reply)
            print(".", end="", flush=True)

            # Check if the assistant indicates a final answer.
            if assistant_reply.strip().startswith("FINAL ANSWER:"):
                final_answer = (
                    assistant_reply.strip().replace("FINAL ANSWER:", "", 1).strip()
                )
                print("\nAssistant:", final_answer)
                messages.append({"role": "assistant", "content": assistant_reply})
                # Exit the inner loop as final answer was reached.
                break

            # Otherwise, check if there are any SQL queries in the response.
            sql_queries = extract_sql(assistant_reply)
            # print("\n[DEBUG] Extracted SQL queries:", sql_queries)

            if sql_queries:
                for query in sql_queries:
                    result = execute_sql(conn, query)
                    # Append SQL execution result as a function call (hidden from user).
                    messages.append(
                        {"role": "function", "name": "sql", "content": str(result)}
                    )
                # Record the assistant's message and continue prompting.
                messages.append({"role": "assistant", "content": assistant_reply})
            else:
                # No SQL queries and no final answer marker—continue refining.
                messages.append({"role": "assistant", "content": assistant_reply})
                # print("\n[DEBUG] No SQL queries detected; continuing to refine...")

    conn.close()


if __name__ == "__main__":
    main()
