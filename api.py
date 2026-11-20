from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from openai import (RateLimitError, APIConnectionError, APIStatusError)
from config import API_KEY, FALLBACK
from main_nvidia import answer

app = FastAPI(
    title="Company Policy Q&A API",
    description="A retrieval-augmented company policy assistant",
    version="1.0.0"
)


class AskRequest(BaseModel):

    question: str = Field(
        min_length=3,
        max_length=1000
    )

    k: int = Field(
        default=4,
        ge=1,
        le=10
    )

    department: str | None = None

    year: int | None = None


@app.get("/")
def home():

    return {
        "message": "Company Policy Q&A API is running"
    }


@app.post("/ask")
def ask_question(
    request: AskRequest,
    x_api_key: str | None = Header(default=None)
):

    if not API_KEY:

        raise HTTPException(
            status_code=500,
            detail="API key is not configured on the server"
        )

    if x_api_key != API_KEY:

        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )

    try:

        result = answer(
            question=request.question,
            k=request.k,
            department=request.department,
            year=request.year
        )

    except RateLimitError:

        raise HTTPException(
            status_code=503,
            detail="LLM provider rate limit reached. Please retry later."
        )

    except APIConnectionError:

        raise HTTPException(
            status_code=503,
            detail="Unable to connect to the LLM provider."
        )

    except APIStatusError:

        raise HTTPException(
            status_code=502,
            detail="The LLM provider returned an error."
        )

    if result["answer"] == FALLBACK:

        raise HTTPException(
            status_code=404,
            detail="No relevant information was found in the company documents"
        )

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"],
        "citation_validation": result["citation_validation"]
    }
