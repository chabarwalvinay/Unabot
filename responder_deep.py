#!/usr/bin/env python3
"""
rag_chatbot.py

RAG-based chatbot using:
1. Local BGE-M3 embeddings.
2. Pinecone vector DB for retrieval.
3. Gemini LLM for generating answers.
4. Conversation memory for context.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------
# Configuration
# ---------------------------
# Pinecone
PINECONE_API_KEY = (
    "****************************************"
)
INDEX_NAME = "unabot1"
VECTOR_DIM = 1024
TOP_K = 5

# Gemini
GEMINI_API_KEY = "****************************************"

# Local embeddings
MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CONTEXT_TOKENS = 3000  # truncate context if too long

# ---------------------------
# Load local BGE-M3 model
# ---------------------------
print("Loading BGE-M3 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()
print("Model loaded on", DEVICE)


def get_embedding(text: str):
    """Get embedding from local BGE-M3."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()
    return embedding


def get_query_embedding(query: str):
    return get_embedding(query)


# ---------------------------
# Initialize Pinecone
# ---------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


# ---------------------------
# Pinecone retrieval
# ---------------------------
def retrieve_top_chunks(query: str, top_k: int = TOP_K):
    try:
        query_emb = get_query_embedding(query)

        # Use new Pinecone query syntax
        result = index.query(vector=query_emb, top_k=top_k, include_metadata=True)

        # Access matches directly
        matches = result.get("matches", [])
        return matches

    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return []


# ---------------------------
# Construct prompt for Gemini
# ---------------------------
def construct_prompt(query: str, conversation_history: str, retrieved_context: str):
    template = (
        "You are a professional virtual assistant. Based on the following FAQ context and conversation history, "
        "generate a clear, informative, and natural answer. Do not repeat the context verbatim; synthesize it. "
        "If context is not relevant, respond: 'I'm sorry, I don't have enough information. Could you please rephrase?'\n\n"
        "FAQ Context:\n{context}\n\n"
        "Conversation History:\n{history}\n\n"
        "User Query:\n{query}\n\n"
        "Answer:"
    )
    prompt_template = PromptTemplate(
        template=template, input_variables=["history", "context", "query"]
    )
    return prompt_template.format(
        history=conversation_history, context=retrieved_context, query=query
    )


# ---------------------------
# Conversation memory
# ---------------------------
def get_conversation_history(memory: ConversationBufferMemory) -> str:
    if memory.buffer:
        return "\n".join([msg.content for msg in memory.buffer])
    return ""


# ---------------------------
# Main chatbot loop
# ---------------------------
def main():
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    # Initialize Gemini LLM via LangChain
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Or "gemini-2.0flash"
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=GEMINI_API_KEY,
    )

    print("Chatbot started. Type 'exit' to quit.")
    while True:
        query = input("\nUser: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            print("Empty query. Please try again.")
            continue

        # Retrieve top chunks from Pinecone
        top_chunks = retrieve_top_chunks(query)
        if not top_chunks:
            print("No relevant context found. Try rephrasing your query.")
            continue

        # CORRECTED: Handle potential missing metadata safely
        context_parts = []
        for chunk in top_chunks:
            if "metadata" in chunk and "content" in chunk["metadata"]:
                context_parts.append(chunk["metadata"]["content"])

        context_text = "\n".join(context_parts)
        if not context_text:
            print("No content found in retrieved chunks.")
            continue

        context_text = context_text[:MAX_CONTEXT_TOKENS]  # Truncate if needed

        conversation_history = get_conversation_history(memory)
        prompt = construct_prompt(query, conversation_history, context_text)

        # Generate response via Gemini
        response = llm.invoke([HumanMessage(content=prompt)])
        final_answer = response.content.strip()

        # Print and save to memory
        print("\nVA:", final_answer)
        memory.save_context({"input": query}, {"output": final_answer})


if __name__ == "__main__":
    main()
