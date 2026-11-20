# Policy Q&A Assistant

A policy Q&A assistant that answers only from company documents, cites its sources, and refuses to answer when the documents do not contain the information.

Built as the Week 7 individual assessment on Retrieval-Augmented Generation (RAG).

---

## Table of contents

- [About](#about)
- [Features](#features)
- [Tech stack](#tech-stack)
- [How it works](#how-it-works)
- [Getting started](#getting-started)
- [Usage](#usage)
- [API reference](#api-reference)
- [Project structure](#project-structure)
- [Design decisions](#design-decisions)
- [Results](#results)
- [Limitations](#limitations)

---

## About

Employees often ask questions that policy documents already answer, such as how many PTO days they get or what to do after a security incident. A general chatbot may answer from memory and get it wrong. PolicyLens-RAG retrieves the relevant policy text first, asks the LLM to answer only from that text, and returns the file and page for every fact.

The sample company is "ABC Technologies". The corpus has 9 documents: an employee policy, handbook, onboarding guide, operational guidelines, technical notes, three short sample policies (access control, incident response, travel) and an outdated 2024 HR policy that exists to test metadata filtering.

---

## Features

- Indexing pipeline with metadata and stable chunk ids. Running it twice keeps the record count unchanged.
- Fallback that returns "I don't have information on that" without calling the LLM when retrieval confidence is low.
- Context token budget (`fit_context`) so the prompt never exceeds the limit.
- Hybrid search (BM25 + vector) merged with Reciprocal Rank Fusion, cross-encoder reranking, HyDE and metadata filters.
- Citations in `[Source: file, Page n]` format, checked with a regular expression against the retrieved chunks.
- REST API (`POST /ask`) protected by an API key.
- Evaluation with a 15-question golden dataset and RAGAS.

---

## Tech stack

| Area | Tools |
|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector database | Qdrant (local file mode) |
| Keyword search | `rank-bm25` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM access | OpenAI-compatible client with OpenRouter and NVIDIA endpoints |
| Token counting | `tiktoken` |
| API | FastAPI, Uvicorn, Pydantic |
| Evaluation | RAGAS, pandas |

Advanced retrieval (BM25, Reciprocal Rank Fusion, cross-encoder reranking, HyDE) is built and evaluated in the Task 2 scripts. The live `answer()` function uses vector retrieval with optional department and year filters.

---

## Getting started

### Prerequisites

- Python 3.11
- An OpenRouter API key (Task 1, HyDE, prompt comparison)
- An NVIDIA API key (grounded answers, the API and the RAGAS judge)

### Install

```bash
git clone https://github.com/Adhil-Yoho/policy-assistant.git
cd policylens-rag
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

### Configure

Open `.env` and fill in your keys. The file is in `.gitignore` and must never be committed.

```env
OPENROUTER_API_KEY=your_api_key
OPENROUTER_MODEL=your_model_name

NVIDIA_API_KEY=your_api_key
NVIDIA_MODEL=your_model_name
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

SCORE_THRESHOLD=0.35
CONTEXT_LIMIT=4000
RESERVED_OUTPUT_TOKENS=500
API_KEY=your_secret_key
```

| Variable | Meaning |
|---|---|
| `SCORE_THRESHOLD` | Minimum similarity score. Below it the assistant refuses without calling the LLM. |
| `CONTEXT_LIMIT` | Token limit for the system prompt, the context and the reserved output. |
| `RESERVED_OUTPUT_TOKENS` | Tokens kept free for the LLM's answer. |
| `API_KEY` | Secret that clients send in the `x-api-key` header. |

### Build the index

```bash
py indexing.py
py indexing.py
```

Run it twice. Both runs should report 9 records, which shows the pipeline is idempotent. The database is created in `qdrant_db/`, which is not committed.

---

## Usage

Each task has its own scripts. Results are saved in `reports/`.

| Task | Command | What it shows |
|---|---|---|
| 1 | `py main.py` | Answerable and unanswerable questions, fallback, context budget at k = 2, 5, 10 |
| 2 | `py t2_evaluation.py` | Hit@3 for BM25, vector and hybrid |
| 2 | `py t2_reranking.py` | Rank of the correct chunk before and after reranking |
| 2 | `py t2_hyde.py` | HyDE hit@3 and latency |
| 2 | `py t2_metadata.py` | Metadata filter removing the 2024 policy |
| 3 | `py main_nvidia.py` | Grounded answers with citation validation |
| 3 | `py t3_prompt_comparison.py` | Three prompt versions on 10 questions |
| 3 | `py t3_hallucination_summary.py` | Hallucination count per prompt |
| 4 | `py evaluate_ragas.py before` | RAGAS on the golden dataset |
| 4 | `py evaluate_ragas.py after` | RAGAS after the improvement |
| 4 | `py t4_compare_results.py` | Before and after table |

Start the API:

```bash
uvicorn api:app --reload
```

Interactive docs are at `http://127.0.0.1:8000/docs`.

---

## API reference

### POST /ask

Header: `x-api-key: <your API_KEY>`

Request body:

```json
{
  "question": "How many annual PTO days are provided?",
  "k": 4,
  "department": "HR",
  "year": 2026
}
```

Only `question` is required. `k` accepts 1 to 10. `department` and `year` are optional filters.

Successful response (200):

```json
{
  "question": "How many annual PTO days are provided?",
  "answer": "Full-time employees accrue 12 days of paid time off per calendar year [Source: employee_policy.pdf, Page 1].",
  "sources": [{"source": "employee_policy.pdf", "page": 1}],
  "citation_validation": {
    "citations": [{"source": "employee_policy.pdf", "page": 1}],
    "invalid_citations": [],
    "citation_count": 1,
    "all_citations_valid": true
  }
}
```

Example call:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "x-api-key: YOUR_API_KEY" -H "Content-Type: application/json" \
  -d '{"question": "How many annual PTO days are provided?"}'
```

---

## Project structure

```
policy-assistant/
├── config.py # paths, model names, thresholds, env loading
├── indexing.py # Task 1 indexing pipeline (idempotent)
├── retrieval.py # vector search, context format, fit_context, budget
├── main.py # Task 1 answer() using OpenRouter
├── main_nvidia.py # Task 3 answer() with citations, NVIDIA model
├── advanced_retrieval.py # BM25, RRF, hybrid, rerank, HyDE, metadata filter
├── citations.py # regex citation parser + validator
├── api.py # FastAPI POST /ask with API key
├── t2_evaluation.py # Task 2 hit@3 BM25 / vector / hybrid
├── t2_reranking.py # Task 2 cross-encoder before/after ranks
├── t2_hyde.py # Task 2 HyDE vs normal (hit@3 + latency)
├── t2_metadata.py # Task 2 metadata filter demo (removes 2024 doc)
├── t3_prompt_comparison.py # Task 3 prompt v1/v2/v3 comparison
├── t3_hallucination_summary.py
├── t4_compare_results.py
├── evaluate_ragas.py # Task 4 RAGAS run (before / after)
├── data/
│   ├── documents.json # 9 source documents
│   ├── retrieval_questions.json # 12 Qs (6 exact, 6 meaning)
│   ├── prompt_test_questions.json # 10 Qs for prompt comparison
│   └── golden.jsonl # 15 golden Qs
├── reports/ # CSV / JSON outputs from every run
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Design decisions

### Storage and indexing

- Qdrant runs in local file mode, so there is no server to install.
- Each chunk id is a UUID made with `uuid5` from the source, page, chunk index and text. The same input always gives the same id, so re-running `indexing.py` overwrites the same points instead of adding duplicates.
- Chunks are 500 tokens with 64 tokens of overlap. Every document is shorter than 500 tokens, so each became one chunk (9 documents, 9 chunks). Longer documents would be split.
- Every chunk stores source, page, department, year and document type. Filters and citations depend on these fields.

### Fallback and context budget

- The threshold is 0.35. It refused all four unanswerable test questions before any LLM call. The trade-off is that a vague but valid question with a low score is also refused.
- `fit_context(chunks, budget)` counts tokens with `tiktoken` and skips any chunk that would go over the budget. The budget is `CONTEXT_LIMIT` minus the system prompt, the question template and `RESERVED_OUTPUT_TOKENS`.

### Retrieval

- BM25 runs over the same chunks stored in Qdrant, so both methods search identical text.
- Reciprocal Rank Fusion uses `1 / (60 + rank)` to merge the BM25 and vector lists.
- The reranker rescores the hybrid candidates and keeps the top 5.
- HyDE was chosen over query rewriting. It improves hit@3 but needs an LLM call before every search, so it is not in the live pipeline.
- Metadata filters are Qdrant payload conditions on `department` and `year`. The 2024 policy stays in the index on purpose, so the filter has something to remove.

### Prompts, citations and API

- Prompt version 3 (answer only from context, cite every fact, say you do not know) is the production prompt. It is the only version that returns the exact fallback sentence and adds citations.
- The citation pattern is `[Source: file, Page n]`. The `sources` field in the response lists citations found in the answer, not every retrieved chunk.
- `/ask` compares the `x-api-key` header with `API_KEY`. An assistant refusal becomes a 404, so a client can tell "no answer" from "answer".

## Results

### Task 1: pipeline and fallback

All 4 unanswerable questions returned "I don't have information on that" with no LLM call. Context used at k = 2, 5 and 10 was 390, 963 and 1666 tokens. At k = 10 only 9 chunks exist, so 9 were used.

### Task 2: advanced retrieval

| Method | Hit@3 (12 questions) |
|---|---|
| BM25 | 0.92 |
| Vector | 0.83 |
| Hybrid (RRF) | 0.92 |
| Hybrid + rerank | 0.92 |

The reranker did not change hit@3 but improved the rank of the correct chunk:

- "What is the annual PTO entitlement?" moved from rank 3 to rank 1.
- "How does the company handle a serious production service incident?" moved from rank 8 to rank 5.
- The other 10 questions were already at rank 1.

Hit@5 rose from 0.92 to 1.00 after reranking. The corpus has only 9 chunks, so the top 50 candidates are all 9 chunks.

HyDE raised hit@3 from 0.83 to 0.92, and average latency went from 0.02 seconds to 11.0 seconds per question.

Metadata filtering with `year=2026` removed `employee_policy_2024.pdf` (15 PTO days), so the answer uses the current 12 days.

### Task 3: citations and hallucination control

| Prompt | Hallucinated answers (10 questions, 3 unanswerable) |
|---|---|
| 1. Basic | 0 |
| 2. Only from context | 0 |
| 3. Only from context, cite, say you don't know | 0 |

The model handled the unanswerable questions safely even with the basic prompt, so the versions tie on hallucination. They differ in format: only version 3 returns the exact fallback sentence and cites sources.

Manual faithfulness: 5 answers were split into 9 claims, and all 9 were supported by the retrieved context.

API checks: a valid request returned 200 with sources, an out-of-scope question returned 404, and a wrong key returned 401.

### Task 4: golden dataset and RAGAS

The golden set has 15 questions (8 simple, 4 multi-chunk, 3 unanswerable). Each has a reference answer, source file and page.

| Metric | Score |
|---|---|
| Faithfulness | 0.99 |
| Answer relevancy | 0.79 |
| Context precision | 0.63 |
| Context recall | 0.98 |
| Source recall | 0.96 |
| Correct abstentions | 3 of 3 |

Faithfulness is above the 0.85 target. The before and after runs gave the same scores, so the comparison shows no improvement yet.

Failure analysis of the weakest questions:

| Question | Failure type | What happened |
|---|---|---|
| G011, incident steps and SEV targets | Retrieval | `technical_notes.txt` was not retrieved (source recall 0.5). The answer said it had no information on SEV targets while still citing a source. |
| G009, first-day activities and security controls | Generation | Both documents were retrieved, but the answer listed only multi-factor authentication from the security controls. |
| G010, travel approvals and reimbursement | Retrieval | Both documents were retrieved among six chunks, but most of the context was unrelated, so context precision was 0. |

The next fix to try is hybrid search with reranking inside `answer()`. G011 should benefit because "SEV" is an exact term that BM25 matches and the vector search does not.

---

## Limitations

- The corpus is small and synthetic. Every document is one page, so every citation says Page 1.
- RAGAS scores come from an LLM judge and vary slightly between runs.
- The live pipeline uses vector search only. BM25, RRF and reranking are evaluated but not yet connected to `answer()`.
- The 0.35 threshold was tuned by hand on a few questions and may need adjusting for a larger corpus.
