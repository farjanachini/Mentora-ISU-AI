
import json
import chromadb

# 1. ChromaDB client
chroma_client = chromadb.PersistentClient(path="./isu_chroma_db")

# 2. পুরোনো collection মুছে নতুন collection তৈরি
try:
    chroma_client.delete_collection(name="isu_info")
except Exception:
    pass

collection = chroma_client.create_collection(name="isu_info")

# 3. JSON data load
with open("isu_data.json", "r", encoding="utf-8") as f:
    isu_data = json.load(f)

documents = []
ids = []
metadatas = []

# 4. প্রতিটি record সংরক্ষণ
for i, item in enumerate(isu_data):
    content = str(item.get("content", "")).strip()

    if not content:
        continue

    category = str(item.get("category", "general"))
    source_url = str(item.get("source_url", ""))

    documents.append(
        f"Category: {category}\n"
        f"Source: {source_url}\n"
        f"Information: {content}"
    )

    ids.append(f"isu_{i}")
    metadatas.append({
        "category": category,
        "source_url": source_url
    })

# 5. Batch করে database-এ যোগ
batch_size = 100

for start in range(0, len(documents), batch_size):
    end = start + batch_size

    collection.add(
        documents=documents[start:end],
        ids=ids[start:end],
        metadatas=metadatas[start:end]
    )

print(f"Done! Added {len(documents)} records.")
print(f"ChromaDB count: {collection.count()}")
