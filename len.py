import json
import tiktoken

# ----------------------------
# CONFIG
# ----------------------------
CHUNKS_JSON = "markdown_chunks.json"
MAX_TOKENS = 400  # threshold to flag

# ----------------------------
# LOAD CHUNKS
# ----------------------------
with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)


# ----------------------------
# HELPER: Count tokens
# ----------------------------
def count_tokens(text, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


# ----------------------------
# PROCESS CHUNKS
# ----------------------------
print(f"{'Doc_ID':<20} {'Chunk_Index':<12} {'Token_Count':<12} {'Flag>400':<10}")
print("-" * 60)

over_limit_chunks = []

for c in chunks:
    token_len = count_tokens(c["content"])
    flag = "YES" if token_len > MAX_TOKENS else ""
    print(f"{c['doc_id']:<20} {c['chunk_index']:<12} {token_len:<12} {flag:<10}")

    if token_len > MAX_TOKENS:
        over_limit_chunks.append(
            {
                "doc_id": c["doc_id"],
                "chunk_index": c["chunk_index"],
                "token_count": token_len,
            }
        )

# ----------------------------
# SUMMARY
# ----------------------------
print("\nChunks exceeding", MAX_TOKENS, "tokens:")
for c in over_limit_chunks:
    print(
        f"- Doc: {c['doc_id']}, Chunk: {c['chunk_index']}, Tokens: {c['token_count']}"
    )

print(f"\nTotal chunks over {MAX_TOKENS} tokens: {len(over_limit_chunks)}")
