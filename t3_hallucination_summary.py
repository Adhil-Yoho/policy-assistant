import pandas as pd

df = pd.read_csv(
    "reports/task3_prompt_comparison.csv"
)

df["manual_hallucination_flag"] = pd.to_numeric(
    df["manual_hallucination_flag"],
    errors="coerce"
)

if df["manual_hallucination_flag"].isna().any():
    print("Please review every answer and fill the hallucination flag.")
else:

    summary = df.groupby("prompt_type")[
        "manual_hallucination_flag"
    ].sum()

    print("\nHallucination count by prompt:")
    print(summary)

    summary.to_csv(
        "reports/Task3_hallucination_summary.csv"
    )
