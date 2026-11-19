from backend.retriever import load_retriever
from backend.corrective_rag import (
    corrective_retrieval,
    groq_client,
    MODEL_NAME
)


def rewrite_query(query):
    """
    Improve the search query using Groq LLM.
    """

    prompt = f"""
You are an expert search query optimizer.

Rewrite the following question to improve
semantic document retrieval.

Preserve the original meaning.

Return ONLY the rewritten query.

Original Question:
{query}
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

    rewritten = (
        response.choices[0].message.content or ""
    ).strip()

    return rewritten or query


def adaptive_retrieval(
    query,
    collection,
    model,
    k=5,
    min_relevant=2,
    max_retries=2
):
    """
    Retrieve and grade documents.
    Rewrite the query when relevant
    evidence is insufficient.
    """

    current_query = query
    best_documents = []

    for attempt in range(max_retries + 1):

        print(f"\nRetrieval Attempt: {attempt + 1}")
        print("Search Query:", current_query)

        relevant_docs = corrective_retrieval(
            current_query,
            collection,
            model,
            k=k
        )

        print("Relevant Documents:", len(relevant_docs))

        if len(relevant_docs) > len(best_documents):
            best_documents = relevant_docs

        if len(relevant_docs) >= min_relevant:
            print("Sufficient relevant documents found!")
            return relevant_docs

        if attempt < max_retries:
            print("Insufficient evidence. Rewriting query...")

            current_query = rewrite_query(current_query)

            print("Rewritten Query:", current_query)

    print("Maximum retries reached.")

    return best_documents


if __name__ == "__main__":

    collection, model = load_retriever()

    question = "What is an intelligent agent?"

    print("\nOriginal Question:", question)

    documents = adaptive_retrieval(
        question,
        collection,
        model,
        k=5,
        min_relevant=2,
        max_retries=2
    )

    print("\nFinal Retrieved Documents:", len(documents))

    for i, doc in enumerate(documents, 1):
        print("\n" + "=" * 50)
        print("Document:", i)
        print("Page:", doc["metadata"].get("page", "N/A"))
        print("Text:", doc["text"][:300])