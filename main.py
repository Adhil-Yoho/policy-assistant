main.py
from openai import OpenAI

from config import (OPENROUTER_API_KEY,OPENROUTER_MODEL,SCORE_THRESHOLD, CONTEXT_LIMIT, RESERVED_OUTPUT_TOKENS, FALLBACK)

from retrieval import (retrieve_chunks, format_context, fit_context, count_tokens)


SYSTEM_PROMPT = """
You are a company policy Q&A assistant.

Answer ONLY using the context provided.

Do not invent facts or use outside knowledge.

If the answer is not available in the context,
reply exactly:

I don't have information on that
"""


PROMPT_TEMPLATE = """
Answer ONLY using the context below.

Context:
{context}

Question:
{question}

Answer:
"""


def call_llm(prompt):

    if not OPENROUTER_API_KEY:
        raise ValueError("OpenRouter API key is missing")

    if not OPENROUTER_MODEL:
        raise ValueError("OpenRouter model is missing")

    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=RESERVED_OUTPUT_TOKENS
    )

    answer_text = response.choices[0].message.content

    if not answer_text:
        raise ValueError("LLM returned an empty answer.")

    return answer_text.strip()


def answer(question, k=4):

    chunks = retrieve_chunks(question, k)

    if not chunks:

        return {
            "answer": FALLBACK,
            "sources": [],
            "chunks": [],
            "called_llm": False
        }

    best_score = chunks[0]["score"]

    print("\nBest retrieval score:", round(best_score, 4))

    if best_score < SCORE_THRESHOLD:

        print("Fallback activated. LLM was not called.")

        return {
            "answer": FALLBACK,
            "sources": [],
            "chunks": chunks,
            "called_llm": False
        }

    empty_prompt = PROMPT_TEMPLATE.format(
        context="",
        question=question
    )

    fixed_tokens = (
        count_tokens(SYSTEM_PROMPT)
        + count_tokens(empty_prompt)
        + RESERVED_OUTPUT_TOKENS
    )

    available_budget = CONTEXT_LIMIT - fixed_tokens

    if available_budget <= 0:

        return {
            "answer": FALLBACK,
            "sources": [],
            "chunks": [],
            "called_llm": False
        }

    selected_chunks = fit_context(
        chunks,
        available_budget
    )

    if not selected_chunks:

        return {
            "answer": FALLBACK,
            "sources": [],
            "chunks": [],
            "called_llm": False
        }

    context = format_context(selected_chunks)

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    total_tokens = (
        count_tokens(SYSTEM_PROMPT)
        + count_tokens(prompt)
        + RESERVED_OUTPUT_TOKENS
    )

    if total_tokens > CONTEXT_LIMIT:

        return {
            "answer": FALLBACK,
            "sources": [],
            "chunks": [],
            "called_llm": False
        }

    print("Context tokens:", count_tokens(context))
    print("Total tokens including reserved output:", total_tokens)

    answer_text = call_llm(prompt)

    sources = []

    for chunk in selected_chunks:

        source = {
            "source": chunk["source"],
            "page": chunk["page"]
        }

        if source not in sources:
            sources.append(source)

    return {
        "answer": answer_text,
        "sources": sources,
        "chunks": selected_chunks,
        "called_llm": True
    }


def run_question(question, k=4):

    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("=" * 60)

    result = answer(question, k)

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    print(result["sources"])

    print("\nLLM CALLED:", result["called_llm"])

    return result


answerable_questions = [
    "How many casual leaves per year?",
    "How many sick leave days are provided per year?",
    "How many unused PTO days can employees carry forward?",
    "What security practices are included in the company guidelines?",
    "What milestones are included in the employee onboarding plan?",
    "What topics are covered in the technical notes?"
]


unanswerable_questions = [
    "Who won the FIFA World Cup?",
    "What is the current price of Bitcoin?",
    "Give me a chicken biryani recipe.",
    "What is the capital of Japan?"
]


def test_answerable_questions():

    print("\nTASK 1: ANSWERABLE QUESTIONS")

    for question in answerable_questions:

        run_question(question, k=4)


def test_unanswerable_questions():

    print("\nTASK 1: UNANSWERABLE QUESTIONS")

    for question in unanswerable_questions:

        run_question(question, k=4)


def test_context_budget():

    print("\nTASK 1: CONTEXT BUDGET TEST")

    question = "How many PTO days are provided each year?"

    print("\nk | Retrieved chunks | Used context tokens | LLM called")

    for k in [2, 5, 10]:

        result = answer(question, k=k)

        context_tokens = count_tokens(
            format_context(result["chunks"])
        ) if result["chunks"] else 0

        print(
            k,
            "|",
            len(result["chunks"]),
            "|",
            context_tokens,
            "|",
            result["called_llm"]
        )


if __name__ == "__main__":

    test_answerable_questions()

    test_unanswerable_questions()

    test_context_budget()