import pickle
import numpy as np

with open("bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)

with open("embedded_chunks.pkl", "rb") as f:
    embedded_chunks = pickle.load(f)

def bm25_retrieve(
    query,
    k=5
):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(
        tokenized_query
    )

    top_indices = np.argsort(scores)[::-1][:k]

    results = []

    for rank, idx in enumerate(
        top_indices,
        start=1
    ):

        results.append({

            "rank": rank,

            "chunk_id":
                embedded_chunks[idx]["chunk_id"],

            "title":
                embedded_chunks[idx]["title"],

            "score":
                float(scores[idx]),

            "content":
                embedded_chunks[idx]["content"]

        })

    return results