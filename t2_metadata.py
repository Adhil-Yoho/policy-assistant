from advanced_retrieval import (get_all_chunks, metadata_filter)
from retrieval import retrieve_chunks

chunks = get_all_chunks()


hr_chunks = metadata_filter(
    chunks,
    department="HR"
)

print("HR chunks:", len(hr_chunks))


current_chunks = metadata_filter(
    chunks,
    year=2026
)

print("2026 chunks:", len(current_chunks))


current_hr_chunks = metadata_filter(
    chunks,
    department="HR",
    year=2026
)

print("2026 HR chunks:", len(current_hr_chunks))


question = "How many annual PTO days are provided?"

results = retrieve_chunks(
    question,
    k=10
)

print("\nBefore filtering:")

for result in results:

    if result["source"] in [
        "employee_policy.pdf",
        "employee_policy_2024.pdf"
    ]:

        print(
            result["source"],
            result["year"],
            result["text"][:250]
        )


print("\nAfter filtering to 2026:")

filtered_results = metadata_filter(
    results,
    year=2026
)

for result in filtered_results:

    if result["source"] in [
        "employee_policy.pdf",
        "employee_policy_2024.pdf"
    ]:

        print(
            result["source"],
            result["year"],
            result["text"][:250]
        )