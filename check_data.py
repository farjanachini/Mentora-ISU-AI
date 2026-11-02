
import chromadb

client = chromadb.PersistentClient(path="./isu_chroma_db")
collection = client.get_collection(name="isu_info")

results = collection.get(
    include=["documents", "metadatas"]
)

print("\nTotal ChromaDB records:", collection.count())

found = False

for doc, metadata in zip(
    results.get("documents", []),
    results.get("metadatas", [])
):
    if "fahim tazwar" in doc.lower():
        found = True
        print("\n--- MATCH FOUND ---")
        print("Source:", (metadata or {}).get("source_url"))
        print("Information:", doc)

if not found:
    print("\nG.M. Fahim Tazwar was NOT found in ChromaDB.")
