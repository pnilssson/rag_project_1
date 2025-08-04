from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np
import openai

COLLECTION_NAME = "rag_chunks"
MODEL_NAME = "qwen/qwen3-8b"

client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")

openai.api_base = "http://localhost:1234/v1"
openai.api_key = "sk-local"

def retrieve_relevant_chunks(query, top_k=3):
    embedding = model.encode(query).tolist()
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=top_k
    )
    return [hit.payload["text"] for hit in hits]

def ask_local_model(question, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""
Du är en assistent. Svara endast baserat på följande information:

{context}

Fråga: {question}
"""

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message["content"]

if __name__ == "__main__":
    while True:
        question = input("\nStäll en fråga (eller 'q' för att avsluta): ")
        if question.lower() == "q":
            break
        chunks = retrieve_relevant_chunks(question)
        answer = ask_local_model(question, chunks)
        print("\nSvar:\n", answer)
