from faiss_ret import faiss_retrieve
from bm25_ret import bm25_retrieve
def rrf_fusion(
    query,
    faiss_k=20,
    bm25_k=20,
    rrf_k=60
):

    faiss_results = faiss_retrieve(
        query=query,
        k=faiss_k
    )

    bm25_results = bm25_retrieve(
        query=query,
        k=bm25_k
    )

    fused_scores = {}

    # FAISS contribution
    for rank, doc in enumerate(
        faiss_results,
        start=1
    ):

        chunk_id = doc["chunk_id"]

        fused_scores.setdefault(
            chunk_id,
            {
                "title": doc["title"],
                "content": doc["content"],
                "score": 0
            }
        )

        fused_scores[chunk_id]["score"] += (
            1 / (rrf_k + rank)
        )

    # BM25 contribution
    for rank, doc in enumerate(
        bm25_results,
        start=1
    ):

        chunk_id = doc["chunk_id"]

        fused_scores.setdefault(
            chunk_id,
            {
                "title": doc["title"],
                "content": doc["content"],
                "score": 0
            }
        )

        fused_scores[chunk_id]["score"] += (
            1 / (rrf_k + rank)
        )

    results = []

    for chunk_id, data in fused_scores.items():

        results.append({

            "chunk_id": chunk_id,

            "title": data["title"],

            "content": data["content"],

            "rrf_score": data["score"]

        })

    results.sort(
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return results