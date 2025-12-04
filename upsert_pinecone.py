import json
from pinecone import Pinecone, ServerlessSpec

# ----------------------------
# CONFIGURATION
# ----------------------------
CHUNKS_JSON = "chunks_with_embeddings.json"
PINECONE_API_KEY = "****************************************"
INDEX_NAME = "unabot1"
VECTOR_DIM = 1024  # Ensure this matches your embedding size
BATCH_SIZE = 100

# ----------------------------
# LOAD CHUNKS WITH EMBEDDINGS
# ----------------------------
with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

# ----------------------------
# INITIALIZE PINECONE CLIENT
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create the index if it doesn't exist
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# ----------------------------
# UPSERT CHUNKS INTO PINECONE
# ----------------------------
index = pc.Index(INDEX_NAME)


def chunk_to_pinecone_item(c):
    vector_id = f"{c['doc_id']}_{c['chunk_index']}"
    metadata = {
        "doc_id": c["doc_id"],
        "chunk_index": c["chunk_index"],
        "heading_path": c.get("heading_path", []),
        "token_count": c.get("token_count", 0),
        "content": c["content"],  # optional
    }
    return (vector_id, c["embedding"], metadata)


# Upsert in batches
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i : i + BATCH_SIZE]
    vectors = [chunk_to_pinecone_item(c) for c in batch]
    index.upsert(vectors=vectors)
    print(f"Upserted batch {i} to {i + len(batch)}")

print("\n✅ All chunks upserted into Pinecone index!")
