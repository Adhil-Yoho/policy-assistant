import json
import pandas as pd
from pathlib import Path

REPORT_PATH = Path("reports")

before_path = REPORT_PATH / "task4_before_summary.json"
after_path = REPORT_PATH / "task4_after_summary.json"

if not before_path.exists() or not after_path.exists():
    raise FileNotFoundError(
        "Run both before and after evaluations first."
    )


with open(before_path, "r", encoding="utf-8") as file:
    before = json.load(file)

with open(after_path, "r", encoding="utf-8") as file:
    after = json.load(file)


metrics = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "source_recall"
]

rows = []

for metric in metrics:

    before_value = before.get(metric)
    after_value = after.get(metric)

    if before_value is None or after_value is None:
        change = None
    else:
        change = after_value - before_value

    rows.append({
        "metric": metric,
        "before": before_value,
        "after": after_value,
        "change": change
    })


comparison = pd.DataFrame(rows)

print("\nMETRIC COMPARISON")
print(comparison.to_string(index=False))


print("\nABSTENTION COMPARISON")

print(
    "Before:",
    before["correct_abstention_count"],
    "/",
    before["unanswerable_question_count"]
)

print(
    "After:",
    after["correct_abstention_count"],
    "/",
    after["unanswerable_question_count"]
)


print("\nLLM CALL COUNTS")

print("Before:", before["llm_called_count"])
print("After:", after["llm_called_count"])


output_path = REPORT_PATH / "task4_before_after_comparison.csv"

comparison.to_csv(output_path, index=False)

print("\nComparison saved to:", output_path)
