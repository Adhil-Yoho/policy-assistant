import re

CITATION_PATTERN = re.compile(
    r"\[Source:\s*([^,\]]+),\s*Page\s+(\d+)\]"
)


def extract_citations(text):

    citations = []

    matches = CITATION_PATTERN.findall(text)

    for source, page in matches:
        citation = {
            "source": source.strip(),
            "page": int(page)
        }

        if citation not in citations:
            citations.append(citation)

    return citations


def validate_citations(answer, retrieved_chunks):
    cited_sources = extract_citations(answer)

    retrieved_sources = {
        (
            chunk["source"],
            int(chunk["page"])
        )
        for chunk in retrieved_chunks
    }

    invalid_citations = []

    for citation in cited_sources:
        key = (
            citation["source"],
            citation["page"]
        )

        if key not in retrieved_sources:
            invalid_citations.append(citation)

    return {
        "citations": cited_sources,
        "invalid_citations": invalid_citations,
        "citation_count": len(cited_sources),
        "all_citations_valid": (
            len(cited_sources) > 0
            and len(invalid_citations)== 0
        )
    }
