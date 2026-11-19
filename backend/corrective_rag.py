import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from backend.retriever import (
    load_retriever,
    retrieve_documents
)

# Load .env from CARAG root
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found in .env"
    )

groq_client = Groq(api_key=GROQ_API_KEY)

# Set this to an available Groq chat model.
# Notebook's model name must be verified first.
MODEL_NAME = os.getenv("GROQ_MODEL")

if not MODEL_NAME:
    raise RuntimeError(
        "Add GROQ_MODEL to your .env file"
    )


def grade_document(query, document):
    """
    Use Groq to judge document relevance.
    Returns YES or NO.
    """

    prompt = f"""
You are a retrieval relevance grader.

Question:
{query}

Retrieved document:
{document}

Determine whether the document contains information
that is useful for answering the question.

Return ONLY one word:
YES or NO
"""

    response = groq_client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0,
    max_completion_tokens=100
)

    result = (
        response.choices[0].message.content or ""
    ).strip().upper()

    if result not in ("YES", "NO"):
        raise ValueError(
            f"Unexpected relevance grade: {result[:100]}"
        )

    return result


def corrective_retrieval(query, collection, model, k=5):
    """
    Retrieve chunks and filter them
    using the LLM relevance grader.
    """

    documents = retrieve_documents(
        query,
        collection,
        model,
        k=k
    )

    relevant_docs = []

    for i, document in enumerate(documents, 1):

        grade = grade_document(
            query,
            document["text"]
        )

        print(f"Document {i}: {grade}")

        if grade == "YES":
            relevant_docs.append(document)

    return relevant_docs


if __name__ == "__main__":

    collection, model = load_retriever()

    question = "What is an intelligent agent?"

    print("\nQuestion:", question)

    relevant_docs = corrective_retrieval(
        question,
        collection,
        model,
        k=5
    )

    print("\nRelevant Documents:", len(relevant_docs))

    for i, doc in enumerate(relevant_docs, 1):
        print("\n" + "=" * 50)
        print("Relevant Document:", i)
        print("Page:", doc["metadata"].get("page", "N/A"))
        print("Text:", doc["text"][:400])