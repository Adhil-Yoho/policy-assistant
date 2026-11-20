import json
from pathlib import Path
import pandas as pd
from advanced_retrieval import (get_all_chunks, BM25Search, hybrid_search, hit_at_k, measure_latency)
from retrieval import retrieve_chunks


DATA_PATH = Path("data/retrieval_questions.json")
REPORT_PATH = Path("reports")

REPORT_PATH.mkdir(exist_ok=True)


def main():

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        questions = json.load(file)

    chunks = get_all_chunks()
    bm25_model = BM25Search(chunks)

    rows = []

    for item in questions:

        question = item["question"]
        relevant_source = item["relevant_source"]

        bm25_results, bm25_time = measure_latency(
            bm25_model.search,
            question,
            3
        )

        vector_results, vector_time = measure_latency(
            retrieve_chunks,
            question,
            3
        )

        hybrid_results, hybrid_time = measure_latency(
            hybrid_search,
            question,
            3,
            bm25_model
        )

        rows.append({
            "question": question,
            "type": item["type"],
            "bm25_hit@3": hit_at_k(
                bm25_results,
                relevant_source
            ),
            "vector_hit@3": hit_at_k(
                vector_results,
                relevant_source
            ),
            "hybrid_hit@3": hit_at_k(
                hybrid_results,
                relevant_source
            ),
            "bm25_latency": bm25_time,
            "vector_latency": vector_time,
            "hybrid_latency": hybrid_time
        })

    df = pd.DataFrame(rows)

    output_file = REPORT_PATH / "Task2_retrieval_comparison.csv"

    df.to_csv(output_file, index=False)

    print("\nIndividual results:")
    print(df.to_string(index=False))

    print("\nAverage Hit@3:")
    print(
        df[
            [
                "bm25_hit@3",
                "vector_hit@3",
                "hybrid_hit@3"
            ]
        ].mean()
    )

    print("\nResults saved to:", output_file)


if __name__ == "__main__":
    main()
