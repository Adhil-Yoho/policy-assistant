import json
import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, VectorParams, PointStruct)
from config import (DOCUMENTS_PATH, DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP)
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

print("Loading embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL)

print("Embedding model loaded")

client = QdrantClient(path=str(DB_DIR))


def load_documents():

    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as file:
        documents = json.load(file)

    return documents


def parse_documents(documents):

    parsed_documents = []

    for document in documents:

        parsed_documents.append({
            "id": document["id"],
            "source": document["source"],
            "title": document["title"],
            "text": document["text"],
            "page": document["page"],
            "department": document["department"],
            "year": document["year"],
            "doc_type": document["doc_type"]
        })

    return parsed_documents


def chunk_text(text):

    tokens = encoding.encode(text)

    chunks = []

    start = 0

    while start < len(tokens):

        end = start + CHUNK_SIZE

        chunk_tokens = tokens[start:end]

        chunk = encoding.decode(chunk_tokens)

        if chunk.strip():
            chunks.append(chunk.strip())

        if end >= len(tokens):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def create_chunks(documents):

    all_chunks = []

    for document in documents:
        text_chunks = chunk_text(document["text"])

        for index, chunk in enumerate(text_chunks):

            unique_text = (
                f"{document['source']}:"
                f"{document['page']}:"
                f"{index}:"
                f"{chunk}"
            )

            stable_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    unique_text
                )
            )

            all_chunks.append({
                "chunk_id": stable_id,
                "document_id": document["id"],
                "text": chunk,
                "source": document["source"],
                "page": document["page"],
                "department": document["department"],
                "year": document["year"],
                "doc_type": document["doc_type"],
                "chunk_index": index
            })

    return all_chunks


def store_chunks(chunks):

    vector_size = model.get_sentence_embedding_dimension()

    if not client.collection_exists(COLLECTION_NAME):

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

    points = []

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    for chunk, embedding in zip(chunks, embeddings):

        points.append(
            PointStruct(
                id=chunk["chunk_id"],
                vector=embedding.tolist(),
                payload=chunk
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )

    return client.count(
        collection_name=COLLECTION_NAME,
        exact=True
    ).count


def run_indexing():

    print("\nSTEP 1: LOADING DOCUMENTS")

    documents = load_documents()

    print("Documents loaded:", len(documents))

    print("\nSTEP 2: PARSING DOCUMENTS")

    parsed_documents = parse_documents(documents)

    print("Documents parsed:", len(parsed_documents))

    print("\nSTEP 3: CHUNKING")

    chunks = create_chunks(parsed_documents)

    print("Total chunks:", len(chunks))

    print("\nSTEP 4: EMBEDDING AND STORAGE")

    total_stored = store_chunks(chunks)

    print("Total records in Qdrant:", total_stored)

    print("\nINDEXING COMPLETED SUCCESSFULLY")

    return total_stored


if __name__ == "__main__":

    run_indexing()

    client.close()
