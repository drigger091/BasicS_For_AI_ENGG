import numpy as np  
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Load the FAISS index and the chunks globally so they are available to the function
index = faiss.read_index("wiki_base.faiss")
with open("embedded_chunks.pkl", "rb") as f:
    embedded_chunks = pickle.load(f)
def faiss_retrive(query, k=5):



    embedding_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device="cuda"  )

# Create query embedding
    query_embedding = embedding_model.encode(
        f"query: {query}",
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Search FAISS
    scores, indices = index.search(
        np.array([query_embedding]).astype("float32"),
        k
    )

    results = []

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        results.append({
            "rank": rank,
            "chunk_id": embedded_chunks[idx]["chunk_id"],
            "title": embedded_chunks[idx]["title"],
            "score": float(score),
            "content": embedded_chunks[idx]["content"]
        })

    return results

