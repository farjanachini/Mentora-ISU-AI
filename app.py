import os
import time
import chromadb

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv  # ১. এই লাইনটি ইমপোর্ট করো

# ২. অ্যাপ চালুর আগেই .env ফাইল লোড করতে হবে
load_dotenv()

app = Flask(__name__)
CORS(app)

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# এখন .env ফাইল থেকে সঠিক উপায়ে কি (Key) চলে আসবে
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)



# ChromaDB
chroma_client = chromadb.PersistentClient(
    path="./isu_chroma_db"
)

collection = chroma_client.get_collection(
    name="isu_info"
)


@app.route("/")
def home():
    return render_template("index.html")


def get_department(question):
    """Identify a department mentioned in the question."""

    q = question.lower()

    cse_terms = [
        "cse",
        "computer science",
        "কম্পিউটার সায়েন্স",
        "কম্পিউটার সায়েন্স"
    ]

    ai_ds_terms = [
        "ai and data science",
        "artificial intelligence",
        "data science",
        "এআই",
        "ডেটা সায়েন্স",
        "ডেটা সায়েন্স"
    ]

    is_cse = any(term in q for term in cse_terms)
    is_ai_ds = any(term in q for term in ai_ds_terms)

    if is_cse and not is_ai_ds:
        return "cse"

    if is_ai_ds and not is_cse:
        return "ai_ds"

    return None


def is_head_question(question):
    """Check whether the user asks about a department head."""

    q = question.lower()

    terms = [
        "head",
        "chairman",
        "chairperson",
        "department chief",
        "বিভাগীয় প্রধান",
        "বিভাগীয় প্রধান",
        "বিভাগের প্রধান",
        "প্রধান কে"
    ]

    return any(term in q for term in terms)

def find_cse_head_record():
    """Find the specific CSE Head record in ChromaDB."""

    results = collection.get(
        where={
            "source_url": "https://admission.isu.ac.bd/faculty-members"
        },
        include=["documents", "metadatas"]
    )

    for doc in results.get("documents", []):
        text = doc.lower()

        if (
            "dr. md. hakikur rahman" in text
            and "professor and head" in text
            and "department of computer science and engineering" in text
        ):
            return doc

    return None  



def retrieve_documents(question, department=None):
    """Retrieve relevant ISU records from ChromaDB."""

    if collection.count() == 0:
        return ""

    q = question.lower().strip()

    # Exact source URLs
    all_departments_url = (
        "https://www.isu.ac.bd/faculties/all-departments"
    )
    cse_url = (
        "https://www.isu.ac.bd/faculties/"
        "department-of-computer-science-and-engineering"
    )
    ai_ds_url = (
        "https://isu.ac.bd/faculties/"
        "department-of-ai-and-data-science"
    )
    faculty_url = (
        "https://admission.isu.ac.bd/faculty-members"
    )

    retrieved_docs = []
    seen = set()

    # Retrieve documents by source URL.
    def get_source_records(source_url):
        try:
            results = collection.get(
                where={"source_url": source_url},
                include=["documents", "metadatas"]
            )
            documents = results.get("documents") or []
            metadatas = results.get("metadatas") or []

            records = []
            for doc, metadata in zip(documents, metadatas):
                metadata = metadata or {}
                records.append({
                    "document": doc,
                    "source": metadata.get(
                        "source_url", source_url
                    )
                })
            return records

        except Exception as error:
            print("ChromaDB source retrieval error:", error)
            return []

    # Add records without duplicates.
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

    # Semantic search, optionally limited to one source.
    def add_results(query, source_url=None, n=5):
        count = collection.count()
        if count == 0:
            return

        search_args = {
            "query_texts": [query],
            "n_results": min(n, count)
        }

        if source_url:
            search_args["where"] = {
                "source_url": source_url
            }

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

    # Retrieve a source's complete records.
    def add_all_from_source(source_url):
        add_records(get_source_records(source_url))

    # Recognize common question types.
    department_terms = [
        "what departments",
        "which departments",
        "list of departments",
        "available departments",
        "department list",
        "all departments",
        "কী কী বিভাগ",
        "কোন কোন বিভাগ",
        "বিভাগগুলোর নাম",
        "বিভাগের তালিকা"
    ]

    faculty_terms = [
        "faculty", "faculties", "faculty members",
        "teacher", "teachers", "professor",
        "lecturer", "instructor", "who teaches",
        "designation", "academic staff",
        "শিক্ষক", "শিক্ষকবৃন্দ", "শিক্ষকদের নাম"
    ]

    head_terms = [
        "head", "department head", "chairperson",
        "chairman", "chairperson of",
        "বিভাগের প্রধান", "বিভাগীয় প্রধান"
    ]

    asks_departments = any(
        term in q for term in department_terms
    )

    asks_faculty = any(
        term in q for term in faculty_terms
    )

    asks_head = (
        any(term in q for term in head_terms)
        or is_head_question(question)
    )

    # 1. Department-list questions.
    if asks_departments:
        add_all_from_source(all_departments_url)

        if not retrieved_docs:
            add_results(
                "ISU academic departments and department names",
                n=10
            )

        return "\n\n".join(retrieved_docs)

    # 2. Identify the department when possible.
    if department is None:
        if any(term in q for term in [
            "computer science", "cse",
            "কম্পিউটার সায়েন্স"
        ]):
            department = "cse"

        elif any(term in q for term in [
            "ai and data science",
            "artificial intelligence",
            "data science", "ai & ds",
            "ai ds", "এআই"
        ]):
            department = "ai_ds"

    # 3. Department-head questions.
    if asks_head and department == "cse":
        add_results(
            "Professor and Head Department of Computer Science "
            "and Engineering",
            faculty_url,
            10
        )
        add_results(
            "Professor and Head Department of Computer Science "
            "and Engineering",
            cse_url,
            10
        )
        add_results(
            "Head of Computer Science and Engineering",
            all_departments_url,
            5
        )

        return "\n\n".join(retrieved_docs)

    if asks_head and department == "ai_ds":
        add_results(
            "Professor and Head Department of AI and Data Science",
            faculty_url,
            10
        )
        add_results(
            "Professor and Head Department of AI and Data Science",
            ai_ds_url,
            10
        )
        add_results(
            "Head of AI and Data Science Department",
            all_departments_url,
            5
        )

        return "\n\n".join(retrieved_docs)

    # 4. Faculty questions: retrieve relevant faculty records
    # from the general faculty source and relevant department.
    if asks_faculty:
        add_results(question, faculty_url, 15)

        if department == "cse":
            add_all_from_source(cse_url)
        elif department == "ai_ds":
            add_all_from_source(ai_ds_url)
        else:
            add_results(
                "ISU faculty members names and designations",
                cse_url,
                10
            )
            add_results(
                "ISU faculty members names and designations",
                ai_ds_url,
                10
            )

        add_results(question, n=5)
        return "\n\n".join(retrieved_docs)

    # Find a specific faculty member by name.
    faculty_names = [
        "g.m. fahim tazwar",
        "gm fahim tazwar",
        "fahim tazwar"
    ]

    if any(name in q for name in faculty_names):
        for source_url in [
            faculty_url,
            ai_ds_url,
            all_departments_url
        ]:
            add_all_from_source(source_url)

        matching_docs = [
            doc for doc in retrieved_docs
            if any(name in doc.lower() for name in faculty_names)
        ]

        if matching_docs:
            return "\n\n".join(matching_docs)

    # 5. General questions: search semantically first.
    add_results(question, n=7)

    # Use additional source-specific searches when relevant.
    if department == "cse":
        add_results(question, cse_url, 5)

    elif department == "ai_ds":
        add_results(question, ai_ds_url, 5)

    # Search faculty source for a named person or staff-related query.
    if any(term in q for term in [
        "who is", "about", "name", "email",
        "contact", "profile", "designation",
        "কে", "নাম", "ইমেইল", "পরিচয়"
    ]):
        add_results(question, faculty_url, 5)

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
            department = get_department(user_message)

            retrieved_context = retrieve_documents(
                user_message,
                department
            )

            print("\n--- RETRIEVED CONTEXT ---")
            print(retrieved_context)
            print("--- END CONTEXT ---\n")

        system_instruction = f"""
You are ISU AI Assistant.

Use the following retrieved information from ISU sources:

{retrieved_context}

Guidelines:
1. Answer ISU-related questions using the provided information.
2. Never invent names, designations, email addresses, or facts.
3. If information is unavailable, say that it was not found in the
   available ISU information.
4. If the user asks in Bengali, reply in polite Bengali.
5. If the user asks in English, reply in English.
6. Keep answers short and clear.
7. Do not assume an advisor, dean, or professor is a department head
   unless the source explicitly identifies them as the head.
8. Pay attention to the source URL and department.
9. Do not use information about one department as the answer for
   another department.
10. If sources give conflicting information about a department head,
    clearly explain that the available sources conflict. Do not
    present either claim as definitely correct without verification.
11. If no relevant retrieved information is available, say so.
12. When asked to list departments, list every department name
    supported by the retrieved information. Do not return only one
    department if more are available.
    13. For general questions unrelated to ISU, answer using your general knowledge.
    
14. When listing faculty members, include only people explicitly
identified as affiliated with the department requested by the user.

15. Do not list the same person more than once. If a person has
    multiple roles, clearly distinguish their roles and institutions.
"""

        contents = [system_instruction]

        # Process uploaded file.
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
            max_output_tokens=250
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
                        # Try the next model for model-specific errors.
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
