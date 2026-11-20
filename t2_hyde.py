import json
import time
import pandas as pd
from pathlib import Path
from openai import OpenAI
from config import (OPENROUTER_API_KEY, OPENROUTER_MODEL)
from retrieval import retrieve_chunks
from advanced_retrieval import (hyde_search, hit_at_k)


client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


with open(
    "data/retrieval_questions.json",
    "r",
    encoding="utf-8"
) as file:
    questions = json.load(file)


rows = []

for item in questions:

    question = item["question"]
    relevant_source = item["relevant_source"]

    start = time.perf_counter()

    normal_results = retrieve_chunks(
        question,
        k=3
    )

    normal_latency = time.perf_counter() - start

    start = time.perf_counter()

    hyde_results = hyde_search(
        question,
        client,
        OPENROUTER_MODEL,
        k=3
    )

    hyde_latency = time.perf_counter() - start

    rows.append({
        "question": question,
        "normal_hit@3": hit_at_k(
            normal_results,
            relevant_source
        ),
        "hyde_hit@3": hit_at_k(
            hyde_results,
            relevant_source
        ),
        "normal_latency_seconds": normal_latency,
        "hyde_latency_seconds": hyde_latency
    })


df = pd.DataFrame(rows)

Path("reports").mkdir(exist_ok=True)

df.to_csv(
    "reports/Task2_hyde_comparison.csv",
    index=False
)

print(df.to_string(index=False))

print("\nNormal Hit@3:")
print(df["normal_hit@3"].mean())

print("\nHyDE Hit@3:")
print(df["hyde_hit@3"].mean())

print("\nAverage normal latency:")
print(df["normal_latency_seconds"].mean())

print("\nAverage HyDE latency:")
print(df["hyde_latency_seconds"].mean())