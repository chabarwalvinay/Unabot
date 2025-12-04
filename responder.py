#!/usr/bin/env python3
"""
rag_chatbot_enhanced.py

Enhanced RAG-based chatbot using:
1. Local BGE-M3 embeddings + BM25 hybrid search
2. Pinecone vector DB for retrieval
3. Gemini LLM for generating answers
4. Conversation memory for context
5. Query decomposition for complex queries
6. Enhanced prompt engineering
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from rank_bm25 import BM25Okapi
import re
from typing import List, Dict, Any

# ---------------------------
# Configuration
# ---------------------------
# Pinecone
PINECONE_API_KEY = "****************************************"
INDEX_NAME = "unabot1"
VECTOR_DIM = 1024
TOP_K = 5

# Gemini
GEMINI_API_KEY = "****************************************"
# Local embeddings
MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CONTEXT_TOKENS = 3000  # truncate context if too long

# BM25 configuration
BM25_TOP_K = 5
HYBRID_TOP_K = 5

# ---------------------------
# Load local BGE-M3 model
# ---------------------------
print("Loading BGE-M3 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()
print("Model loaded on", DEVICE)

# ---------------------------
# BM25 Setup (In-memory for demo - in production you'd want persistent storage)
# ---------------------------
class BM25Retriever:
    def __init__(self):
        self.corpus = []
        self.doc_ids = []
        self.bm25 = None
        self.is_initialized = False

    def initialize_from_pinecone(self, index, sample_size=1000):
        """Initialize BM25 with a sample of documents from Pinecone"""
        print("Initializing BM25 index from Pinecone...")
        try:
            # Sample some vectors to build the corpus
            sample_result = index.query(
                vector=[0.0] * VECTOR_DIM,  # Dummy vector
                top_k=sample_size,
                include_metadata=True
            )

            self.corpus = []
            self.doc_ids = []

            for match in sample_result.get('matches', []):
                if 'metadata' in match and 'content' in match['metadata']:
                    content = match['metadata']['content']
                    # Simple tokenization for BM25
                    tokens = self._tokenize(content)
                    self.corpus.append(tokens)
                    self.doc_ids.append(match['id'])

            if self.corpus:
                self.bm25 = BM25Okapi(self.corpus)
                self.is_initialized = True
                print(f"BM25 initialized with {len(self.corpus)} documents")
            else:
                print("Warning: No documents found for BM25 initialization")

        except Exception as e:
            print(f"Error initializing BM25: {e}")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def search(self, query: str, top_k: int = BM25_TOP_K) -> List[Dict[str, Any]]:
        """Search using BM25"""
        if not self.is_initialized or not self.bm25:
            return []

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get top_k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if idx < len(self.doc_ids) and scores[idx] > 0:
                results.append({
                    'id': self.doc_ids[idx],
                    'score': scores[idx],
                    'content': ' '.join(self.corpus[idx])  # Reconstruct text from tokens
                })

        return results

# ---------------------------
# Initialize Pinecone and BM25
# ---------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
bm25_retriever = BM25Retriever()
bm25_retriever.initialize_from_pinecone(index)

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
# Enhanced Retrieval with Hybrid Search
# ---------------------------
def reciprocal_rank_fusion(dense_results: List[Dict], sparse_results: List[Dict], k: int = 60):
    """Combine results using Reciprocal Rank Fusion"""
    fused_scores = {}

    # Score dense results
    for rank, doc in enumerate(dense_results):
        doc_id = doc['id']
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (rank + k)

    # Score sparse results
    for rank, doc in enumerate(sparse_results):
        doc_id = doc['id']
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (rank + k)

    # Sort by fused score
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in sorted_docs]

def hybrid_retrieval(query: str, top_k: int = HYBRID_TOP_K):
    """Hybrid retrieval combining dense and sparse search"""
    try:
        # Dense vector search
        query_emb = get_query_embedding(query)
        dense_result = index.query(
            vector=query_emb,
            top_k=top_k,
            include_metadata=True
        )
        dense_matches = dense_result.get("matches", [])

        # Sparse BM25 search
        sparse_matches = bm25_retriever.search(query, top_k=top_k)

        # If BM25 fails, fall back to dense search only
        if not sparse_matches:
            return dense_matches

        # Combine using Reciprocal Rank Fusion
        fused_ids = reciprocal_rank_fusion(dense_matches, sparse_matches)

        # Create a mapping of id to document for easy lookup
        doc_map = {}
        for doc in dense_matches + sparse_matches:
            doc_id = doc['id']
            if doc_id not in doc_map:
                # Prefer documents with full metadata from Pinecone
                if 'metadata' in doc:
                    doc_map[doc_id] = doc
                else:
                    # For BM25 results, create a minimal structure
                    doc_map[doc_id] = {
                        'id': doc_id,
                        'metadata': {'content': doc.get('content', '')}
                    }

        # Return documents in RRF order
        final_results = []
        for doc_id in fused_ids[:top_k]:
            if doc_id in doc_map:
                final_results.append(doc_map[doc_id])

        return final_results if final_results else dense_matches

    except Exception as e:
        print(f"Error in hybrid retrieval: {e}")
        # Fallback to dense search
        query_emb = get_query_embedding(query)
        result = index.query(vector=query_emb, top_k=top_k, include_metadata=True)
        return result.get("matches", [])

# ---------------------------
# Query Decomposition
# ---------------------------
def decompose_complex_query(query: str, llm) -> List[str]:
    """Decompose complex queries into simpler sub-questions"""

    decomposition_prompt = f"""
    Analyze the following user query and break it down into simpler, independent sub-questions
    that can be answered separately. Focus on extracting the key information needs.

    User Query: "{query}"

    If this is a simple query that doesn't need decomposition, just return the original query.
    If it's complex (involving comparisons, multiple aspects, or requiring information from different contexts),
    break it down into 2-4 simpler questions.

    Return each sub-question on a new line. Be specific and clear.

    Sub-questions:
    """

    try:
        response = llm.invoke([HumanMessage(content=decomposition_prompt)])
        sub_questions = response.content.strip().split('\n')

        # Clean and filter sub-questions
        cleaned_questions = []
        for q in sub_questions:
            q = q.strip()
            if q and not q.startswith(('Sub-questions:', '```')) and len(q) > 5:
                # Remove numbering if present
                q = re.sub(r'^\d+[\.\)]\s*', '', q)
                cleaned_questions.append(q)

        return cleaned_questions if cleaned_questions else [query]

    except Exception as e:
        print(f"Error in query decomposition: {e}")
        return [query]

# ---------------------------
# Enhanced Prompt Engineering
# ---------------------------
def construct_enhanced_prompt(query: str, conversation_history: str, retrieved_context: str, is_complex: bool = False):
    """Construct enhanced prompt with better instructions"""

    system_prompt = """
    You are a professional virtual assistant for IIIT Una. Strictly use the provided context to answer the user's question.
    The context contains official academic documents, policies, and curriculum information.

    IMPORTANT INSTRUCTIONS:
    1. Base your answer ONLY on the provided context. Do not use external knowledge.
    2. If the context provides a direct reference (like page numbers) but not full details, state the reference clearly.
    3. If information is completely absent from context, say: "I'm sorry, this specific information is not available in my current knowledge base."
    4. For curriculum comparisons, analyze the provided documents point by point.
    5. Be precise, factual, and maintain academic tone.
    6. Structure complex answers with clear sections when appropriate.

    Context Information:
    {context}

    Conversation History:
    {history}

    User Query: {query}

    Answer:
    """

    if is_complex:
        system_prompt = system_prompt.replace("Answer:",
            "This appears to be a complex query. Provide a comprehensive, well-structured answer:\n\nAnswer:")

    prompt_template = PromptTemplate(
        template=system_prompt,
        input_variables=["history", "context", "query"]
    )

    return prompt_template.format(
        history=conversation_history,
        context=retrieved_context,
        query=query
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
        model="gemini-2.0-flash",  # Using a more reliable model
        temperature=0.1,  # Slightly increased for better creativity in complex answers
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=GEMINI_API_KEY,
    )

    print("Chatbot started. Type 'exit' to quit.")
    print("Enhanced with hybrid search and query decomposition!")

    while True:
        query = input("\nUser: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            print("Empty query. Please try again.")
            continue

        # Detect complex queries for decomposition
        complex_indicators = ['compare', 'difference', 'versus', 'vs', 'advantages', 'disadvantages',
                             'multiple', 'various', 'comprehensive', 'detailed analysis']
        is_complex_query = any(indicator in query.lower() for indicator in complex_indicators)

        all_context_parts = []

        if is_complex_query:
            print("Detected complex query - decomposing...")
            sub_queries = decompose_complex_query(query, llm)

            for i, sub_query in enumerate(sub_queries):
                if i >= 3:  # Limit to 3 sub-queries to avoid too many API calls
                    break
                print(f"  Sub-query {i+1}: {sub_query}")
                top_chunks = hybrid_retrieval(sub_query)

                for chunk in top_chunks:
                    if "metadata" in chunk and "content" in chunk["metadata"]:
                        content = chunk["metadata"]["content"]
                        if content not in all_context_parts:  # Avoid duplicates
                            all_context_parts.append(content)
        else:
            # Standard retrieval for simple queries
            top_chunks = hybrid_retrieval(query)
            for chunk in top_chunks:
                if "metadata" in chunk and "content" in chunk["metadata"]:
                    all_context_parts.append(chunk["metadata"]["content"])

        # Combine context
        context_text = "\n".join(all_context_parts)
        if not context_text:
            print("No relevant context found. Try rephrasing your query.")
            continue

        context_text = context_text[:MAX_CONTEXT_TOKENS]  # Truncate if needed

        conversation_history = get_conversation_history(memory)
        prompt = construct_enhanced_prompt(query, conversation_history, context_text, is_complex_query)

        # Generate response via Gemini
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            final_answer = response.content.strip()

            # Print and save to memory
            print("\nVA:", final_answer)
            memory.save_context({"input": query}, {"output": final_answer})

        except Exception as e:
            print(f"\nError generating response: {e}")
            print("VA: I encountered an error while processing your query. Please try again.")

if __name__ == "__main__":
    main()
