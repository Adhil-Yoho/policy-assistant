import json
from pathlib import Path
import pandas as pd
from sentence_transformers import CrossEncoder
from advanced_retrieval import (hybrid_search, get_all_chunks, BM25Search, RERANKER_NAME)

TOP_CANDIDATES = 50 
TOP_FINAL = 5 

with open("data/retrieval_questions.json", "r", encoding="utf-8") as file:
    questions = json.load(file)

chunks = get_all_chunks()
bm25_model = BM25Search(chunks)

reranker = CrossEncoder(RERANKER_NAME)


def rank_of(results, source):
    for position, chunk in enumerate(results, start=1):
        if chunk["source"] == source:
            return position
    return None


def hit(rank, k):
    return int(rank is not None and rank <= k)


rows = []

for item in questions:

    question = item["question"]
    relevant_source = item["relevant_source"]

    candidates = hybrid_search(
        question,
        k=TOP_CANDIDATES,
        bm25_model=bm25_model
    )

    rank_before = rank_of(candidates, relevant_source)

    pairs = [(question, chunk["text"]) for chunk in candidates]
    scores = reranker.predict(pairs)

    reranked = []
    for chunk, score in zip(candidates, scores):
        new_chunk = chunk.copy()
        new_chunk["rerank_score"] = float(score)
        reranked.append(new_chunk)

    reranked.sort(key=lambda c: c["rerank_score"], reverse=True)

    rank_after = rank_of(reranked, relevant_source)
    final_top = reranked[:TOP_FINAL]

    rows.append({
        "question": question,
        "relevant_source": relevant_source,
        "rank_before": rank_before,
        "rank_after": rank_after,
        "hybrid_hit@3": hit(rank_before, 3),
        "rerank_hit@3": hit(rank_after, 3),
        "hit@5_before": hit(rank_before, 5),
        "hit@5_after": int(
            rank_of(final_top, relevant_source) is not None
        ),
    })


df = pd.DataFrame(rows)

Path("reports").mkdir(exist_ok=True)
df.to_csv("reports/Task2_reranking_comparison.csv", index=False)

print("\nRank of the correct chunk (1 = best)")
print(
    df[["question", "relevant_source", "rank_before", "rank_after"]]
    .to_string(index=False)
)

print("\nAverage Hit@3")
print("Hybrid :", round(df["hybrid_hit@3"].mean(), 2))
print("Hybrid + rerank :", round(df["rerank_hit@3"].mean(), 2))

print("\nAverage Hit@5")
print("Before rerank :", round(df["hit@5_before"].mean(), 2))
print("After rerank :", round(df["hit@5_after"].mean(), 2))

print("\nSaved to: reports/Task2_reranking_comparison.csv")
