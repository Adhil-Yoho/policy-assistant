from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import tiktoken

from config import (DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL, SCORE_THRESHOLD, FALLBACK)

model = SentenceTransformer(EMBEDDING_MODEL)

client = QdrantClient(path=str(DB_DIR))

encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    return len(encoding.encode(text))


def retrieve_chunks(
    question,
    k=4,
    department=None,
    year=None
):

    question_vector = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    conditions = []

    if department is not None:
        conditions.append(
            FieldCondition(
                key="department",
                match=MatchValue(value=department)
            )
        )

    if year is not None:
        conditions.append(
            FieldCondition(
                key="year",
                match=MatchValue(value=year)
            )
        )

    query_filter = None

    if conditions:
        query_filter = Filter(must=conditions)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=question_vector,
        query_filter=query_filter,
        limit=k,
        with_payload=True
    )

    chunks = []

    for point in results.points:

        payload = point.payload

        chunks.append({
            "id": str(point.id),
            "chunk_id": payload["chunk_id"],
            "text": payload["text"],
            "source": payload["source"],
            "page": payload["page"],
            "department": payload["department"],
            "year": payload["year"],
            "doc_type": payload["doc_type"],
            "score": point.score
        })

    return chunks


def format_context(chunks):

    formatted_chunks = []

    for chunk in chunks:

        text = (
            f"[Source: {chunk['source']}, "
            f"Page {chunk['page']}]\n"
            f"Department: {chunk['department']}\n"
            f"Year: {chunk['year']}\n"
            f"Text: {chunk['text']}"
        )

        formatted_chunks.append(text)

    return "\n\n".join(formatted_chunks)


def fit_context(chunks, budget):

    selected_chunks = []

    for chunk in chunks:

        candidate = selected_chunks + [chunk]

        context = format_context(candidate)

        tokens = count_tokens(context)

        if tokens <= budget:
            selected_chunks.append(chunk)
        else:
            continue

    return selected_chunks
