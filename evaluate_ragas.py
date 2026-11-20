import json
import asyncio
import sys
import pandas as pd
from pathlib import Path
from openai import AsyncOpenAI
import instructor
from config import (NVIDIA_API_KEY, NVIDIA_MODEL, NVIDIA_BASE_URL, FALLBACK)
from main_nvidia import answer
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall)


LABEL = sys.argv[1] if len(sys.argv) > 1 else "before"

if LABEL not in {"before", "after"}:
    raise ValueError("Label must be either 'before' or 'after'.")

DATA_PATH = Path("data/golden.jsonl")
REPORT_PATH = Path("reports")

REPORT_PATH.mkdir(exist_ok=True)

if not NVIDIA_API_KEY or not NVIDIA_MODEL:
    raise ValueError(
        "Configure NVIDIA_API_KEY and NVIDIA_MODEL in .env"
    )


async_client = AsyncOpenAI(
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL,
    timeout=120.0,
    max_retries=2
)

evaluator_llm = llm_factory(
    NVIDIA_MODEL,
    provider="openai",
    client=async_client,
    mode=instructor.Mode.MD_JSON
)

evaluator_embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)


faithfulness_metric = Faithfulness(
    llm=evaluator_llm
)

answer_relevancy_metric = AnswerRelevancy(
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

context_precision_metric = ContextPrecision(
    llm=evaluator_llm
)

context_recall_metric = ContextRecall(
    llm=evaluator_llm
)


golden_questions = []

with open(DATA_PATH, "r", encoding="utf-8") as file:

    for line in file:

        if line.strip():
            golden_questions.append(json.loads(line))


def get_unique_sources(chunks):

    sources = []

    for chunk in chunks:

        source = chunk.get("source")

        if source and source not in sources:
            sources.append(source)

    return sources


def calculate_source_recall(
    retrieved_sources,
    reference_sources
):

    if not reference_sources:
        return None

    matched_sources = set(retrieved_sources).intersection(
        set(reference_sources)
    )

    return len(matched_sources) / len(reference_sources)


async def run_evaluation():

    rows = []

    for item in golden_questions:

        question = item["question"]

        print("\n" + "=" * 65)
        print("QUESTION:", question)
        print("TYPE:", item["type"])
        print("=" * 65)

        result = answer(
            question,
            k=6
        )

        generated_answer = result["answer"]

        retrieved_chunks = result.get("chunks", [])

        retrieved_contexts = [
            chunk["text"]
            for chunk in retrieved_chunks
            if chunk.get("text")
        ]

        retrieved_sources = get_unique_sources(
            retrieved_chunks
        )

        reference_sources = item.get(
            "reference_sources",
            []
        )

        source_recall = calculate_source_recall(
            retrieved_sources,
            reference_sources
        )

        citation_validation = result.get(
            "citation_validation",
            {}
        )

        row = {
            "id": item["id"],
            "type": item["type"],
            "question": question,
            "reference_answer": item["reference_answer"],
            "generated_answer": generated_answer,
            "answerable": item["answerable"],
            "called_llm": result["called_llm"],
            "retrieved_context_count": len(retrieved_contexts),
            "reference_sources": json.dumps(reference_sources),
            "retrieved_sources": json.dumps(retrieved_sources),
            "cited_sources": json.dumps(result.get("sources", [])),
            "source_recall": source_recall,
            "citation_count": citation_validation.get(
                "citation_count", 0
            ),
            "all_citations_valid": citation_validation.get(
                "all_citations_valid"
            ),
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
            "correct_abstention": None,
            "evaluation_status": None
        }


        if item["answerable"]:

            if result["called_llm"]:

                faith_result = await faithfulness_metric.ascore(
                    user_input=question,
                    response=generated_answer,
                    retrieved_contexts=retrieved_contexts
                )

                row["faithfulness"] = faith_result.value

                relevancy_result = await answer_relevancy_metric.ascore(
                    user_input=question,
                    response=generated_answer
                )

                row["answer_relevancy"] = relevancy_result.value

            if retrieved_contexts:

                precision_result = await context_precision_metric.ascore(
                    user_input=question,
                    reference=item["reference_answer"],
                    retrieved_contexts=retrieved_contexts
                )

                row["context_precision"] = precision_result.value

                recall_result = await context_recall_metric.ascore(
                    user_input=question,
                    reference=item["reference_answer"],
                    retrieved_contexts=retrieved_contexts
                )

                row["context_recall"] = recall_result.value

            else:

                row["context_recall"] = 0.0

            if result["called_llm"]:
                row["evaluation_status"] = "completed"
            else:
                row["evaluation_status"] = "retrieval_fallback_no_llm"

        else:

            row["correct_abstention"] = (
                generated_answer.strip() == FALLBACK
            )

            row["evaluation_status"] = "abstention_checked"

        rows.append(row)

        print("\nANSWER:")
        print(generated_answer)

        print("\nLLM CALLED:", result["called_llm"])

        print("RETRIEVED CONTEXTS:", len(retrieved_contexts))

        print("RETRIEVED SOURCES:", retrieved_sources)

        print("SOURCE RECALL:", source_recall)

        if item["answerable"]:

            print("FAITHFULNESS:", row["faithfulness"])
            print("ANSWER RELEVANCY:", row["answer_relevancy"])
            print("CONTEXT PRECISION:", row["context_precision"])
            print("CONTEXT RECALL:", row["context_recall"])

        else:

            print(
                "CORRECT ABSTENTION:",
                row["correct_abstention"]
            )


    df = pd.DataFrame(rows)

    output_file = REPORT_PATH / f"task4_{LABEL}_results.csv"

    df.to_csv(output_file, index=False)


    answerable_df = df[df["answerable"] == True]

    unanswerable_df = df[df["answerable"] == False]

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "source_recall"
    ]

    summary = {}

    for metric in metric_names:

        values = pd.to_numeric(
            answerable_df[metric],
            errors="coerce"
        )

        summary[metric] = (
            float(values.mean())
            if values.notna().any()
            else None
        )

        summary[f"{metric}_evaluated_count"] = int(
            values.notna().sum()
        )

    summary["correct_abstention_count"] = int(
        unanswerable_df["correct_abstention"].sum()
    )

    summary["unanswerable_question_count"] = len(
        unanswerable_df
    )

    summary["answerable_question_count"] = len(
        answerable_df
    )

    summary["llm_called_count"] = int(
        df["called_llm"].sum()
    )

    summary["evaluation_label"] = LABEL

    summary_file = REPORT_PATH / f"task4_{LABEL}_summary.json"

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )


    print("\n" + "=" * 65)
    print("RAGAS SUMMARY:", LABEL.upper())
    print("=" * 65)

    for metric, value in summary.items():

        print(f"{metric}: {value}")

    print("\nDetailed results:", output_file)
    print("Summary:", summary_file)


if __name__ == "__main__":
    asyncio.run(run_evaluation())