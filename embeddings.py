# import os
# import json
# from tqdm import tqdm
# import torch
# from transformers import AutoTokenizer, AutoModel

# # ----------------------------
# # CONFIGURATION
# # ----------------------------
# CHUNKS_JSON = "markdown_chunks.json"  # input chunks
# OUTPUT_JSON = "markdown_chunks_with_embeddings.json"  # output with embeddings
# MODEL_NAME = "BAAI/bge-base-en-v1.5"  # your chosen embedding model
# BATCH_SIZE = 8  # adjust according to GPU memory
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ----------------------------
# # LOAD MODEL
# # ----------------------------
# print("Loading model and tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModel.from_pretrained(MODEL_NAME)
# model.to(DEVICE)
# model.eval()
# print(f"Model loaded on {DEVICE}")


# # ----------------------------
# # HELPER: Get embedding
# # ----------------------------
# def get_embedding(texts):
#     """
#     texts: list of strings
#     returns: list of embeddings (torch tensors)
#     """
#     inputs = tokenizer(
#         texts, padding=True, truncation=True, return_tensors="pt", max_length=2048
#     )
#     inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
#     with torch.no_grad():
#         outputs = model(**inputs)
#         # The usual practice for BGE: use CLS token embedding
#         embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]
#         embeddings = embeddings.cpu()
#     return embeddings


# # ----------------------------
# # MAIN: Generate embeddings
# # ----------------------------
# # Load your chunked markdown JSON
# with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
#     chunks = json.load(f)

# all_chunks_with_embeddings = []

# for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding batches"):
#     batch = chunks[i : i + BATCH_SIZE]
#     texts = [c["content"] for c in batch]
#     embeddings = get_embedding(texts)

#     # attach embeddings to each chunk
#     for j, c in enumerate(batch):
#         c["embedding"] = embeddings[j].tolist()  # convert tensor to list for JSON
#         all_chunks_with_embeddings.append(c)

# # Save the output JSON
# with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
#     json.dump(all_chunks_with_embeddings, f, ensure_ascii=False, indent=2)

# print(f"\n✅ All embeddings generated and saved to {OUTPUT_JSON}")


import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

# ----------------------------
# CONFIG
# ----------------------------
CHUNKS_JSON = "markdown_chunks.json"
OUTPUT_JSON = "chunks_with_embeddings.json"
MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  # Adjust based on GPU/CPU memory

# ----------------------------
# LOAD CHUNKS
# ----------------------------
with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

# ----------------------------
# LOAD MODEL
# ----------------------------
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()
print("Model loaded on", DEVICE)


# ----------------------------
# FUNCTION TO GET EMBEDDINGS
# ----------------------------
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        # Take the mean of last hidden state as embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()
    return embedding


# ----------------------------
# EMBED ALL CHUNKS
# ----------------------------
all_chunks_with_embeddings = []
for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding batches"):
    batch = chunks[i : i + BATCH_SIZE]
    for c in batch:
        emb = get_embedding(c["content"])
        c["embedding"] = emb
        all_chunks_with_embeddings.append(c)

# ----------------------------
# SAVE OUTPUT
# ----------------------------
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(all_chunks_with_embeddings, f, ensure_ascii=False, indent=2)

print(f"\n✅ All embeddings created and saved to {OUTPUT_JSON}")
