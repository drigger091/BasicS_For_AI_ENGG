from sentence_transformers import CrossEncoder

# Load reranker model once
reranker = CrossEncoder(
    "BAAI/bge-reranker-base",
    device="cuda"
)


def rerank_results(
    query,
    candidates,
    top_k=5
):
    """
    query: str

    candidates: list of dicts
    [
        {
            "chunk_id": ...,
            "title": ...,
            "content": ...
        }
    ]

    top_k: number of results to return
    """

    # Create query-document pairs
    pairs = [
        (query, doc["content"])
        for doc in candidates
    ]

    # Predict relevance scores
    scores = reranker.predict(pairs)

    reranked = []

    for doc, score in zip(candidates, scores):

        result = doc.copy()

        result["rerank_score"] = float(score)

        reranked.append(result)

    # Sort by reranker score
    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked[:top_k]