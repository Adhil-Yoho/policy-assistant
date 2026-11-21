from openai import OpenAI
from config import (NVIDIA_API_KEY, NVIDIA_MODEL, NVIDIA_BASE_URL, SCORE_THRESHOLD, CONTEXT_LIMIT, RESERVED_OUTPUT_TOKENS, FALLBACK)
from retrieval import (retrieve_chunks, format_context, fit_context, count_tokens)
from citations import validate_citations


SYSTEM_PROMPT = """
You are a company policy Q&A assistant.

Answer ONLY using the supplied context.

Do not use outside knowledge.
Do not invent facts, dates, policies or numbers.

Every factual statement must have a citation
using this exact format:

[Source: filename, Page number]

Use only source filenames and page numbers
that appear in the supplied context.

If the answer is not available in the context,
reply exactly:

I don't have information on that
"""


PROMPT_TEMPLATE = """
Answer ONLY using the context below.

Cite every factual claim using the source
filename and page number shown in the context.

Context:
{context}

Question:
{question}

Answer:
"""


def call_llm(prompt):

    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA API key is missing")

    if not NVIDIA_MODEL:
        raise ValueError("NVIDIA model is missing")

    client = OpenAI(
        api_key=NVIDIA_API_KEY,
        base_url=NVIDIA_BASE_URL
    )

    response = client.chat.completions.create(
        model=NVIDIA_MODEL,
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
        raise ValueError("LLM returned an empty answer")

    return answer_text.strip()

def fallback_result(
    chunks=None,
    best_score=0.0,
    context_tokens=0,
    total_tokens=0
):
    return {
        "answer": FALLBACK,
        "sources": [],
        "chunks": chunks or [],
        "called_llm": False,
        "best_score": best_score,
        "context_tokens": context_tokens,
        "total_tokens": total_tokens,
        "citation_validation": {
            "citations": [],
            "invalid_citations": [],
            "citation_count": 0,
            "all_citations_valid": None
        }
    }


def retrieve_question_chunks(

    question,

    k=4,

    department=None,

    year=None

):

    """

    Retrieve compound questions in separate parts.


    This helps retrieve evidence for both parts of a

    multi-part question instead of relying on one query.

    """


    if " and " in question.lower():


        parts = question.split(" and ", 1)


        queries = [

            parts[0].strip(),

            parts[1].strip()

        ]


    else:

        queries = [question]


    retrieved_lists = []


    for query in queries:


        chunks = retrieve_chunks(

            query,

            k=k,

            department=department,

            year=year

        )


        retrieved_lists.append(chunks)


    merged = {}


    for chunks in retrieved_lists:


        for chunk in chunks:


            chunk_id = chunk["chunk_id"]


            if chunk_id not in merged:

                merged[chunk_id] = chunk


            elif chunk["score"] > merged[chunk_id]["score"]:

                merged[chunk_id] = chunk


    selected = []

    selected_ids = set()


    for chunks in retrieved_lists:


        for chunk in chunks[:2]:


            chunk_id = chunk["chunk_id"]


            if chunk_id not in selected_ids:

                selected.append(chunk)

                selected_ids.add(chunk_id)


    remaining = sorted(

        [

            chunk

            for chunk_id, chunk in merged.items()

            if chunk_id not in selected_ids

        ],

        key=lambda item: item["score"],

        reverse=True

    )


    for chunk in remaining:


        if len(selected) >= k:

            break


        selected.append(chunk)


    selected.sort(

        key=lambda item: item["score"],

        reverse=True

    )


    return selected[:k]


def answer(
    question,
    k=4,
    department=None,
    year=None
):

    chunks = retrieve_question_chunks(
        question,
        k=k,
        department=department,
        year=year
    )

    if not chunks:
        return fallback_result()

    best_score = chunks[0]["score"]

    print("\nBest retrieval score:", round(best_score, 4))

    if best_score < SCORE_THRESHOLD:
        print("Fallback activated. LLM was not called")

        return fallback_result(
            chunks=chunks,
            best_score=best_score
        )

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
        return fallback_result(
            chunks=chunks,
            best_score=best_score
        )

    selected_chunks = fit_context(
        chunks,
        available_budget
    )

    if not selected_chunks:
        return fallback_result(
            chunks=chunks,
            best_score=best_score
        )

    context = format_context(selected_chunks)

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    context_tokens = count_tokens(context)

    total_tokens = (
        count_tokens(SYSTEM_PROMPT)
        + count_tokens(prompt)
        + RESERVED_OUTPUT_TOKENS
    )

    if total_tokens > CONTEXT_LIMIT:
        return fallback_result(
            chunks=selected_chunks,
            best_score=best_score,
            context_tokens=context_tokens,
            total_tokens=total_tokens
        )

    print("Context tokens:", context_tokens)
    print("Total tokens:", total_tokens)

    answer_text = call_llm(prompt)

    sources = []

    for chunk in selected_chunks:

        source = {
            "source": chunk["source"],
            "page": chunk["page"]
        }

        if source not in sources:
            sources.append(source)

    citation_validation = validate_citations(
        answer_text,
        selected_chunks
    )
    sources = citation_validation["citations"]

    retrieved_sources =[]

    for chunk in selected_chunks:
        source = {
            "source": chunk["source"],
            "page": chunk["page"]
        }

        if source not in retrieved_sources:
            retrieved_sources.append(source)

    return {
        "answer": answer_text,
        "sources": sources,
        "retrieved_sources": retrieved_sources,
        "chunks": selected_chunks,
        "called_llm": True,
        "best_score": best_score,
        "context_tokens": context_tokens,
        "total_tokens": total_tokens,
        "citation_validation": citation_validation
    }


def run_question(
    question,
    k=4,
    department=None,
    year=None
):

    print("\n" + "=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)

    result = answer(
        question=question,
        k=k,
        department=department,
        year=year
    )

    print(
        f"\nBest retrieval score: "
        f"{result['best_score']:.4f}"
    )

    print(
        f"Context tokens: "
        f"{result['context_tokens']}"
    )

    print(
        f"Total tokens: "
        f"{result['total_tokens']}"
    )

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    print(result["sources"])

    print("\nCITATION VALIDATION:")
    print(result["citation_validation"])

    print(f"\nLLM CALLED: {result['called_llm']}")

    return result


if __name__ == "__main__":

    run_question(
        "How many annual PTO days are provided?"
    )

    run_question(
        "What is the annual PTO entitlement?",
        department="HR",
        year=2026
    )

    run_question(
        "What is the current price of Bitcoin?"
    )