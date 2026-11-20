import json
import pandas as pd
from pathlib import Path
from openai import OpenAI
from config import (OPENROUTER_API_KEY, OPENROUTER_MODEL)
from retrieval import retrieve_chunks, format_context


client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


PROMPTS = {

    "basic": """
Answer the following question.

Context:
{context}

Question:
{question}

Answer:
""",

    "context_only": """
Answer ONLY using the supplied context.
Do not use outside knowledge.
Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
""",

    "context_citation_abstention": """
Answer ONLY using the supplied context.

Do not use outside knowledge.
Do not invent facts.

Cite every factual claim using:
[Source: filename, Page number]

Use only sources included in the context.

If the answer is not available, reply exactly:
I don't have information on that

Context:
{context}

Question:
{question}

Answer:
"""
}


def generate_answer(prompt):

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


with open(
    "data/prompt_test_questions.json",
    "r",
    encoding="utf-8"
) as file:
    questions = json.load(file)


rows = []

for item in questions:

    question = item["question"]

    chunks = retrieve_chunks(
        question,
        k=4
    )

    context = format_context(chunks)

    for prompt_name, template in PROMPTS.items():

        prompt = template.format(
            context=context,
            question=question
        )

        answer = generate_answer(prompt)

        rows.append({
            "question": question,
            "answerable": item["answerable"],
            "prompt_type": prompt_name,
            "retrieved_context_count": len(chunks),
            "answer": answer,
            "manual_hallucination_flag": ""
        })

        print("\n", prompt_name, ":", question)
        print(answer)


df = pd.DataFrame(rows)

Path("reports").mkdir(exist_ok=True)

df.to_csv(
    "reports/task3_prompt_comparison.csv",
    index=False
)

print("\nPrompt comparison completed")
