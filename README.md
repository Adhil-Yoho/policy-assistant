# Policy Q&A Assistant – RAG Assessment

## Project Overview
This project implements a Policy Q&A Assistant using Retrieval-Augmented Generation (RAG).

The assistant retrieves relevant information from company documents and generates answers using an LLM. It is designed to answer only from the available documents, provide source references, and abstain when the required information is unavailable.

The project covers four main tasks:

1. RAG Pipeline with Fallback
2. Advanced Retrieval
3. Citations and Hallucination Control
4. Golden Dataset and RAGAS Evaluation

### Main Objectives

- Build a complete document indexing and retrieval pipeline.
- Generate embeddings and store document chunks in Qdrant.
- Implement semantic search and advanced retrieval techniques.
- Prevent unsupported answers through fallback handling and grounded prompts.
- Provide source citations with generated answers.
- Evaluate retrieval and generation quality using a golden dataset and RAGAS.

---

## Project Structure

```
policy-assistant/
│
├── data/
│   ├── documents.json
│   ├── retrieval_questions.json
│   ├── prompt_test_questions.json
│   └── golden.jsonl
│
├── reports/
│   ├── Task2_retrieval_comparison.csv
│   ├── Task2_reranking_comparison.csv
│   ├── Task2_hyde_comparison.csv
│   ├── Task3_prompt_comparison.csv
│   ├── Task3_hallucination_summary.csv
│   ├── task3_manual_faithfulness.csv
│   ├── task4_before_results.csv
│   ├── task4_after_results.csv
│   ├── task4_before_summary.json
│   ├── task4_after_summary.json
│   └── task4_before_after_comparison.csv
│
├── config.py
├── indexing.py
├── retrieval.py
├── main.py
│
├── advanced_retrieval.py
├── t2_evaluation.py
├── t2_reranking.py
├── t2_hyde.py
├── t2_metadata.py
│
├── citations.py
├── main_nvidia.py
├── t3_prompt_comparison.py
├── t3_hallucination_summary.py
├── api.py
│
├── evaluate_ragas.py
├── t4_compare_results.py
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

The local qdrant_db/ directory is created when indexing runs. It stores the persistent Qdrant database and is excluded from git

documents.json contains normalized document text records with source filenames, page numbers and metadata. The indexing pipeline uses these prepared records; it does not perform fresh PDF or DOCX extraction from the original files.

## Installation and Configuration

Step 1: Clone the repository
```
git clone https://github.com/Adhil-Yoho/policy-assistant.git
cd policy-assistant
```

Step 2: Create and activate a virtual environment
```
py -m venv venv
.\venv\Scripts\Activate.ps1
```

Step 3: Install dependencies
```
py -m pip install -r requirements.txt
```

Step 4: Configure environment variables

Create the local `.env` file from the example:
```
Copy-Item .env.example .env
```
Configure the required API keys and model settings in `.env`.

The project uses OpenRouter for the Task 1 and Task 2 HyDE experiments, and NVIDIA API for the Task 3 grounded-answer pipeline.

Step 5: Configure environemnt variables

```
OPENROUTER_API_KEY=your_api_key
OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free

NVIDIA_API_KEY=your_api_key
NVIDIA_MODEL=nvidia/nemotron-3.5-bolt-e4b 
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

SCORE_THRESHOLD=0.35
CONTEXT_LIMIT=4000
RESERVED_OUTPUT_TOKENS=800

API_KEY=your_secret_key
```

Never commit `.env`, API keys or passwords to Git.

## Running the Project
Run commands from the project root with the virtual environment activated.

Task 1 – Index documents
```
py indexing.py
```
Run it twice to verify idempotency.

Task 1 – Run the RAG pipeline
```
py main.py
```

Task 2 – Compare retrieval methods
```
py t2_evaluation.py
```

Task 2 – Run reranking
```
py t2_reranking.py
```

Task 2 – Run HyDE
```
py t2_hyde.py
```

Task 2 – Test metadata filtering
```
py t2_metadata.py
```

Task 3 – Run citation-aware answers
```
py main_nvidia.py
```

Task 3 – Compare prompts
```
py t3_prompt_comparison.py
```

After reviewing and filling the manual hallucination flags:
```
py t3_hallucination_summary.py
```

Task 3 – Start the API
```
py -m uvicorn api:app --reload
```

Open the FastAPI documentation:
`http://127.0.0.1:8000/docs`

Task 4 – Run the before evaluation
```
py evaluate_ragas.py before
```

Task 4 – Run the after evaluation
```
py evaluate_ragas.py after
```

Compare the evaluation results
```
py t4_compare_results.py
```

---

## Task 1 – RAG Pipeline with Fallback

### Objective
Build a complete RAG pipeline that loads documents, processes them into chunks, generates embeddings, stores them in Qdrant, retrieves relevant information, and generates grounded answers.

The task also requires a fallback mechanism and context token budgeting.

**Answerable questions**

Question | Expected result| Observed result |
|---|---|---|
| Annual PTO | 12 days | 12 days |
| Sick leave | 10 days | 10 days |
| PTO carry-forward | Up to 5 days | Up to 5 days |
| Security practices | Company security requirements | Detailed answer returned |
| Onboarding milestones | Before Day 1 through Day 90 | Complete answer returned |
| Kubernetes version | Kubernetes 1.30 | AWS EKS Kubernetes 1.30 |

All six questions returned answers, and the LLM was called for each one.

The onboarding response was complete, covering the pre-joining activities, Days 1–5 and the 30-, 60- and 90-day milestones.

**Unanswerable questions**

| Question | Retrieval score | LLM called | Result |
|---|---|---|---|
| FIFA World Cup winner | 0.0790 | No | Correct fallback |
| Current Bitcoin price | 0.1228 | No | Correct fallback |
| Chicken biryani recipe | 0.0272 | No | Correct fallback |
| Capital of Japan | 0.0145 | No | Correct fallback |

All four questions returned the exact fallback:

`I don't have information on that`

The LLM was not called because the retrieval scores were below the configured threshold of 0.35.

### Context-budget experiment

The context-budget test was executed with k values of 2, 5 and 10.

| k | Retrieved chunks | Context tokens | Total tokens including reserved output |
|---|---|---|---|
| 2 | 2 | 390 | 1,261 |
| 5 | 5 | 963 | 1,834 |
| 10 | 9 | 1,666 | 2,537 |

All three tests remained within the configured 4,000-token context limit.

The collection contained nine chunks, so the k=10 request returned the nine available chunks.

### Conclusion

The corrected Task 1 implementation successfully answered all six answerable test questions and correctly abstained on all four unanswerable questions. The fallback mechanism prevented unnecessary LLM calls for unrelated questions.

The context-budget experiment also confirmed that the selected context and reserved output tokens remained within the configured token limit for all three tested k values.

---

## Task 2 – Advanced Retrieval

### Objective

Compare keyword-based and semantic retrieval, combine their results, improve ranking using a cross-encoder, experiment with HyDE and apply metadata filtering.

**BM25 Retrieval**
BM25 uses keyword matching to retrieve documents relevant to the user's query.

The `rank_bm25` library was used to implement keyword-based retrieval.

**Vector Retrieval**
Vector retrieval converts questions and document chunks into embeddings.

Qdrant retrieves semantically similar chunks using vector similarity.

This allows the system to retrieve relevant content even when the wording of the question differs from the document.

**Hybrid Search using RRF**
Hybrid search combines BM25 and vector retrieval.

Reciprocal Rank Fusion combines the rankings using:
```
RRF Score = 1 / (60 + rank)
``

Here:

- `rank` is the position of a chunk in a result list.
- `60` is the rank constant.
- A chunk appearing near the top of multiple lists receives a stronger combined score.
- RRF combines rankings without requiring the raw BM25 and vector scores to be on the same scale.

The implementation uses:

```reciprocal_rank_fusion()
```

### Retrieval Evaluation – t2_evaluation.py

The evaluation dataset contains:

- 6 exact-term questions.
- 6 meaning-based questions.
- 12 questions in total.

The script compares BM25, vector retrieval and hybrid retrieval using Hit@3.

Hit@3 measures whether the correct source appears within the first three results.

**Actual Hit@3 Results**

| Retrieval method | Hit@3 |
|---|---|
| BM25 | 0.9167 |
| Vector | 0.8333 |
| Hybrid | 0.9167 |

Hybrid retrieval performed better than vector retrieval in this dataset, while matching BM25's overall Hit@3.

### Cross-Encoder Reranking – t2_reranking.py

A cross-encoder is a model that evaluates a question and a candidate text together.

Unlike a bi-encoder, which separately creates embeddings for the question and document, a cross-encoder directly evaluates their relevance as a pair.

The implementation:

- Retrieves the top 50 hybrid-search candidates.
- Creates question-and-chunk pairs.
- Scores each pair using the cross-encoder.
- Sorts the candidates by reranking score.
- Keeps the top 5 results.
Model:

```
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The script also records the correct source's rank before and after reranking.

**Reranking Results**

| Metric | Before | After |
|---|---|---|
| Hit@3	| 0.92 | 0.92 |
| Hit@5	| 0.92 | 1.00 |

Reranking improved Hit@5 from 11/12 to 12/12.

It did not improve Hit@3 in this experiment.

### HyDE – Hypothetical Document Embeddings – t2_hyde.py

HyDE is a retrieval technique that generates a hypothetical passage related to the user's question.

**HyDE Results**

| Metric | Normal retrieval | HyDE |
|---|---|---|
| Hit@3	| 0.8333 | 0.9167 |
| Average latency	| 0.0232 seconds | 11.0031 seconds |

HyDE improved Hit@3 by approximately 8.33 percentage points

However, it increased latency substantially because an additional LLM request is needed to generate the hypothetical passage. One query took approximately 51.63 secs

This demonstrates the trade-off between retrieval performance and response time.

### Metadata Filtering – t2_metadata.py

Metadata filtering restricts retrieval using document attributes.

The project uses:

- Department
- Year

For example, the dataset contains:

- `employee_policy.pdf` – 2026
- `employee_policy_2024.pdf` – 2024

The historical policy states 15 days of annual PTO, while the current policy states 12 days.

Filtering to year 2026 removes the outdated policy from the result set.

This helps prevent historical documents from being used to answer current policy questions.

---

## Task 3 – Citations and Hallucination Control

### Objective

Require the model to cite factual statements, validate citations, compare prompt strategies, manually inspect faithfulness and expose the answer pipeline through an authenticated API.

### Citation Formatting
The retrieved context includes source information in this format:
```
[Source: employee_policy.pdf, Page 1]
```

The prompt instructs the model to cite factual claims using the same filename and page number.

Example:
```
Full-time employees accrue 12 days of paid time off
per calendar year [Source: employee_policy.pdf, Page 1].
```

### Citation Extraction and Validation – citations.py

The module uses a regular expression to extract citations.

The pattern recognises citations containing:
- Source filename
- Page number

`extract_citations()` extracts the citations from an answer

`validate_citations()` compares each extracted filename/page combination with the retrieved chunks.

It returns:

- Extracted citations 
- Invalid citations
- Citation count
- Whether all extracted citations are valid

This checks whether a cited source was actually retrieved.

It does not independently prove that the source semantically supports the associated claim.

### Grounded Answer Generation – main_nvidia.py

This module implements the citation-aware RAG answer pipeline using the NVIDIA API.

It reuses the Task 1 retrieval and context-budget functions.

The main differences are:

- Stronger context-only instructions.
- Citation requirements for factual statements.
- Citation validation after generation.
- Support for department and year filters.
- Additional retrieval and token information in the result.

The function returns the answer, cited sources, retrieved chunks, retrieval score, token counts and citation-validation details.

### Prompt Comparison – t3_prompt_comparison.py

Three prompt versions were tested:

**Prompt 1 – Basic**

Provides the question and retrieved context without strict grounding instructions.

**Prompt 2 – Context Only**

Instructs the model to answer only from the supplied context and not invent facts.

**Prompt 3 – Context + Citation + Abstention**

Instructs the model to:

- Answer only from the supplied context.
- Avoid outside knowledge.
- Cite every factual claim.
- Use only source filenames and pages included in the context.
- Return the exact fallback when the answer is unavailable.

The evaluation dataset contains 10 questions:

- 7 answerable questions.
- 3 unanswerable questions.

Each question is tested with all three prompts, producing 30 outputs.

**Prompt Comparison Results**

| Prompt | Questions tested | Manually flagged hallucinations |
|---|---|---|
| Basic | 10 | 0 |
| Context Only | 10 | 0 |
| Context + Citation + Abstention | 10 | 0 |

All 30 outputs have a recorded manual hallucination flag of zero.

The strictest prompt also returned the exact fallback for all 3 unanswerable questions.

The hallucination flags are based on the recorded manual review, not an automated hallucination detector.

### Manual Faithfulness – task3_manual_faithfulness.csv

Faithfulness measures whether the generated answer is supported by the supplied context.

The manual review process:

- Select an answer.
- Split it into individual factual claims.
- Locate the relevant document evidence.
- Check whether the evidence supports each claim.
- Record the result and supporting reason.

The submitted manual review contains 9 claims across 5 question groups.

All 9 claims were marked as supported by the corresponding source documents.

### FastAPI – api.py

The application exposes a POST endpoint:

```
POST /ask
```

The endpoint accepts:

- `question`
- `k`
- Optional `department`
- Optional `year`

It also requires an API key through the `x-api-key` request header.

The endpoint:

1. Validates the API key.
2. Calls the grounded-answer function.
3. Handles LLM provider errors.
4. Returns the answer, sources and citation-validation information.
5. Returns a clear error when no relevant information is found.

FastAPI documentation is available at:
```
http://127.0.0.1:8000/docs
```

The port may differ depending on the local server command.

## Task 4 – Golden Dataset and RAGAS Evaluation

### Objective

Evaluate the RAG pipeline using a fixed set of questions and reference answers, identify weaknesses and compare the results before and after an attempted improvement.

**Golden Dataset** – `data/golden.jsonl`
The golden dataset contains 15 questions

| Question type	| Count |
|---|---|
| Simple | 8 |
| Multi-chunk	| 4 |
| Unanswerable | 3 |
| Total | 15 |

### RAGAS Evaluation – evaluate_ragas.py

RAGAS is used to evaluate the quality of the RAG pipeline.

**Faithfulness:**
- The score remained at approximately 0.9924. This is above the assessment target of 0.85.
- It indicates that the generated answers were highly consistent with the retrieved context in the evaluated dataset.

**Answer Relevancy:**
- The score increased slightly from approximately 0.7941 to 0.7953.
- This is a very small improvement and should not be interpreted as a substantial performance gain.

**Context Precision:**
- The score remained at 0.6250.
- This indicates that irrelevant retrieved chunks remain an area for improvement.

**Context Recall:**
- The score remained at approximately 0.9792.
- The system retrieved most of the required information for the evaluated questions.

**Source Recall:**
- The score remained at approximately 0.9583.
- The system retrieved the expected source information for most of the questions.

The three unanswerable questions are evaluated separately for correct fallback behaviour. The RAGAS averages shown below are calculated over the 12 answerable questions.

**Actual RAGAS Results**

| Metric | Before | After |
|---|---|---|
| Faithfulness | 0.9924	| 0.9924 |
| Answer Relevancy | 0.7941 | 0.7953 |
| Context Precision	| 0.6250 | 0.6250 |
| Context Recall | 0.9792 | 0.9792 |
| Source Recall | 0.9583 | 0.9583 |

The threshold adjustment did not materially change the results. Answer relevancy increased only slightly, while faithfulness, context precision, context recall and source recall remained unchanged.

This indicates that lowering the score threshold alone was not sufficient to address the identified retrieval and generation issues.

---

## Business Summary

The Policy Q&A Assistant successfully demonstrates how company policy information can be retrieved and presented through a grounded question-answering system. The evaluation used 15 questions covering simple queries, questions requiring information from multiple documents, and questions that should be refused.

The final recorded faithfulness score was approximately 0.9924, exceeding the assessment target of 0.85. This indicates that generated answers were highly consistent with the retrieved information during the evaluation. Context recall was also strong at approximately 0.9792, showing that the system retrieved most of the required information.

However, context precision remained at approximately 0.6250. This means that some irrelevant information is still included in the retrieved context. The failure analysis also identified cases where the system either omitted important details during generation or failed to retrieve all required supporting documents.

The threshold adjustment produced no meaningful change in the main retrieval metrics. Therefore, further improvement should focus on more accurate retrieval, reducing irrelevant chunks and ensuring that generated answers cover all parts of multi-part questions.

Overall, the project demonstrates a functional RAG workflow with grounded generation, source references, fallback handling and systematic evaluation. It also highlights the importance of evaluating retrieval and generation separately rather than relying on a single quality score.

---

## Limitations

- The current dataset contains prepared text records rather than a complete automated ingestion process from original PDF and DOCX files.
- Retrieval quality depends on the embedding model, available document content and score threshold.
- A valid citation check confirms that the cited source/page was retrieved; it does not independently establish semantic support for every claim.
- HyDE increases retrieval latency because it requires an additional LLM call.

---

## Conclusion

The Week 7 assessment implements a complete Retrieval-Augmented Generation workflow for a company Policy Q&A Assistant.

The project covers document indexing, embedding generation, vector storage, semantic retrieval, hybrid retrieval, reranking, HyDE, metadata filtering, grounded generation, citation validation, fallback handling, FastAPI integration and RAGAS evaluation.

The final evaluation achieved a faithfulness score of approximately 0.9924, exceeding the required target of 0.85.

The experiments also demonstrated that retrieval quality, answer completeness and response latency involve trade-offs. The failure analysis provides clear directions for future improvements.

This project demonstrates the practical application of RAG concepts and the importance of evaluating both retrieval quality and generated-answer quality.
