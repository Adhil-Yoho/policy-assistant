import re
import time
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from retrieval import retrieve_chunks


def get_all_chunks():
    """
    Read every stored chunk from Qdrant.

    We reuse the existing retrieval module's client.
    """

    from retrieval import client

    all_chunks = []
    offset = None

    while True:
        result, offset = client.scroll(
            collection_name="company_documents",
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        for point in result:
            payload = point.payload

            all_chunks.append({
                "id": str(point.id),
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
                "page": payload.get("page", 1),
                "department": payload.get("department", ""),
                "year": payload.get("year", 0),
                "doc_type": payload.get("doc_type", "")
            })

        if offset is None:
            break

    return all_chunks


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


class BM25Search:

    def __init__(self, chunks):
        self.chunks = chunks

        tokenized_documents = [
            tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.model = BM25Okapi(tokenized_documents)

    def search(self, question, k=4):

        query_tokens = tokenize(question)

        scores = self.model.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []

        for index in ranked_indices[:k]:

            chunk = self.chunks[index].copy()
            chunk["score"] = float(scores[index])

            results.append(chunk)

        return results


def reciprocal_rank_fusion(
    ranked_lists,
    k=4,
    constant=60
):
    """
    RRF score = sum(1 / (constant + rank))

    Ranks begin at 1.
    """

    scores = {}
    chunk_lookup = {}

    for ranked_list in ranked_lists:

        for rank, chunk in enumerate(ranked_list, start=1):

            chunk_id = chunk["id"]

            scores[chunk_id] = (
                scores.get(chunk_id, 0)
                + 1 / (constant + rank)
            )

            chunk_lookup[chunk_id] = chunk

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    results = []

    for chunk_id in ranked_ids[:k]:

        chunk = chunk_lookup[chunk_id].copy()
        chunk["rrf_score"] = scores[chunk_id]

        results.append(chunk)

    return results


def hybrid_search(question, k=4, bm25_model=None):

    if bm25_model is None:
        chunks = get_all_chunks()
        bm25_model = BM25Search(chunks)

    bm25_results = bm25_model.search(question, k=50)

    vector_results = retrieve_chunks(question, k=50)

    hybrid_results = reciprocal_rank_fusion(
        [bm25_results, vector_results],
        k=k
    )

    return hybrid_results

RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank_chunks(question, chunks, top_n=5):

    if not chunks:
        return []

    model = CrossEncoder(RERANKER_NAME)

    pairs = [
        (question, chunk["text"])
        for chunk in chunks
    ]

    scores = model.predict(pairs)

    ranked = []

    for chunk, score in zip(chunks, scores):

        item = chunk.copy()
        item["rerank_score"] = float(score)

        ranked.append(item)

    ranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return ranked[:top_n]


def generate_hypothetical_document(question, llm_client, model):

    prompt = f"""
Write a short hypothetical company policy passage
that could contain the answer to this question.

Question:
{question}

Write only the hypothetical passage.
Do not explain your reasoning.
"""

    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


def hyde_search(
    question,
    llm_client,
    model,
    k=3
):

    hypothetical_document = generate_hypothetical_document(
        question,
        llm_client,
        model
    )

    results = retrieve_chunks(
        hypothetical_document,
        k=k
    )

    return results


def metadata_filter(
    chunks,
    department=None,
    year=None
):

    filtered = []

    for chunk in chunks:

        if department is not None:
            if chunk.get("department") != department:
                continue

        if year is not None:
            if chunk.get("year") != year:
                continue

        filtered.append(chunk)

    return filtered


def hit_at_k(results, relevant_source, k=3):

    top_results = results[:k]

    for chunk in top_results:

        if chunk.get("source") == relevant_source:
            return 1

    return 0


def measure_latency(function, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - start

    return result, elapsed
