import os
import time
import chromadb

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv

# ১. অ্যাপ চালুর আগেই .env ফাইল লোড করতে হবে
load_dotenv()

app = Flask(__name__)
CORS(app)

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# .env ফাইল থেকে API Key লোড করা
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# ChromaDB Initialization
chroma_client = chromadb.PersistentClient(
    path="./isu_chroma_db"
)

collection = chroma_client.get_collection(
    name="isu_info"
)


@app.route("/")
def home():
    return render_template("index.html")


def retrieve_documents(question):
    """Retrieve relevant ISU records from ChromaDB dynamically with department awareness."""
    if collection.count() == 0:
        return ""

    q = question.lower().strip()
    retrieved_docs = []
    seen = set()

    def add_records(records):
        for record in records:
            doc = record["document"]
            source = record["source"]

            if not doc:
                continue

            entry = (
                f"Source URL: {source}\n"
                f"Information:\n{doc}"
            )

            if entry not in seen:
                seen.add(entry)
                retrieved_docs.append(entry)

    def add_results(query, n=7, filter_url=None):
        count = collection.count()
        if count == 0:
            return

        search_args = {
            "query_texts": [query],
            "n_results": min(n, count)
        }

        # যদি নির্দিষ্ট কোনো URL ফিল্টার করতে হয়
        if filter_url:
            search_args["where"] = {"source_url": filter_url}

        try:
            results = collection.query(**search_args)
            documents = results.get("documents") or [[]]
            metadatas = results.get("metadatas") or [[]]

            docs = documents[0] if documents else []
            metas = metadatas[0] if metadatas else []

            records = []
            for doc, metadata in zip(docs, metas):
                metadata = metadata or {}
                records.append({
                    "document": doc,
                    "source": metadata.get(
                        "source_url", "Unknown source"
                    )
                })

            add_records(records)

        except Exception as error:
            print("ChromaDB retrieval error:", error)

    # ইউজার যদি CSE বা কম্পিউটার সায়েন্স নিয়ে প্রশ্ন করে
    if "cse" in q or "computer science" in q or "কম্পিউটার" in q:
        add_results(q, n=7, filter_url="https://admission.isu.ac.bd/faculty-members")
        add_results(q, n=5, filter_url="https://www.isu.ac.bd/faculties/department-of-computer-science-and-engineering")

    # ইউজার যদি AI বা Data Science নিয়ে প্রশ্ন করে
    elif "ai" in q or "data science" in q or "এআই" in q:
        add_results(q, n=7, filter_url="https://admission.isu.ac.bd/faculty-members")
        add_results(q, n=5, filter_url="https://isu.ac.bd/faculties/department-of-ai-and-data-science")

    # সাধারণ প্রশ্নের জন্য নরমাল সেমানটিক সার্চ
    else:
        add_results(q, n=10)

    return "\n\n".join(retrieved_docs)


@app.route("/api/chat", methods=["POST"])
def chat():
    user_message = request.form.get("message", "").strip()
    uploaded_file = request.files.get("file")

    if not user_message and not uploaded_file:
        return jsonify({
            "response": "Please enter a message or upload a file."
        }), 400

    try:
        retrieved_context = ""

        if user_message:
            retrieved_context = retrieve_documents(user_message)

            print("\n--- RETRIEVED CONTEXT ---")
            print(retrieved_context)
            print("--- END CONTEXT ---\n")

        system_instruction = f"""
You are ISU AI Assistant, an intelligent university chatbot.

Use the following retrieved information from ISU sources to answer the user's query:

{retrieved_context}

Guidelines:
1. Answer ISU-related questions using the provided information accurately.
2. Never invent names, designations, email addresses, or facts.
3. If information is unavailable in the retrieved context, state clearly that it was not found in the available ISU records.
4. If the user asks in Bengali, reply in polite, natural, and accurate Bengali.
5. If the user asks in English, reply in English.
6. Keep answers structured, short, and clear.
7. Pay strict attention to the department requested by the user. Do not mix up information or department heads between different departments (e.g., CSE vs AI & Data Science).
8. Use relevant and engaging emojis naturally in your responses (e.g., 👋, 🎓, 💻, 📌) to make the chat lively and friendly.
9. For general questions unrelated to ISU, answer using your general knowledge.
"""

        contents = [system_instruction]

        # Process uploaded file if provided.
        if uploaded_file and uploaded_file.filename:
            filename = os.path.basename(uploaded_file.filename)

            if filename in ("", ".", ".."):
                return jsonify({
                    "response": "Please upload a valid filename."
                }), 400

            file_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            uploaded_file.save(file_path)

            if filename.lower().endswith(
                (".txt", ".json", ".csv", ".md")
            ):
                with open(
                    file_path,
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as tf:
                    file_content = tf.read()

                contents.append(
                    "[Uploaded File Context]\n" + file_content
                )

            else:
                gemini_file = client.files.upload(file=file_path)
                contents.append(gemini_file)

        if user_message:
            contents.append("User Question:\n" + user_message)

        config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=300
        )

        models = [
            "gemini-flash-lite-latest",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash-lite"
        ]

        last_error = None

        for model in models:
            for attempt in range(3):
                try:
                    print(
                        f"Trying {model} - Attempt {attempt + 1}"
                    )

                    response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config
                    )

                    answer = response.text

                    if not answer:
                        answer = (
                            "I could not generate a response. "
                            "Please try again."
                        )

                    return jsonify({"response": answer})

                except Exception as error:
                    last_error = error
                    error_message = str(error)

                    print(f"{model} failed: {error_message}")

                    if (
                        "503" in error_message
                        or "UNAVAILABLE" in error_message
                        or "429" in error_message
                        or "RESOURCE_EXHAUSTED" in error_message
                    ):
                        if attempt < 2:
                            time.sleep(2)
                        else:
                            break
                    else:
                        break

        print("All Gemini models failed:", last_error)

        return jsonify({
            "response": (
                "All available Gemini models failed. "
                "Please try again later."
            )
        }), 503

    except Exception as error:
        print("Detailed Error:", str(error))

        return jsonify({
            "response": "An error occurred while processing your request."
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)