from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# CARAG root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Existing ChromaDB
CHROMA_PATH = BASE_DIR / "chroma_db"

# Existing collection
COLLECTION_NAME = "ai_modern_approach"

# Same BGE model used for document embeddings
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_retriever():
    # Connect to existing ChromaDB
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print("ChromaDB connected successfully!")
    print("Stored chunks:", collection.count())

    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    print("BGE embedding model loaded successfully!")

    return collection, model


def retrieve_documents(query, collection, model, k=5):

    if not query.strip():
        return []

    # Convert question to embedding
    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0].tolist()

    # Search similar documents
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(k, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = []

    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        documents.append({
            "text": text,
            "metadata": metadata or {},
            "distance": distance
        })

    return documents


if __name__ == "__main__":

    collection, model = load_retriever()

    question = "What is an intelligent agent?"

    print("\nQuestion:", question)

    documents = retrieve_documents(
        question,
        collection,
        model,
        k=5
    )

    print("\nRetrieved Documents:", len(documents))

    for i, doc in enumerate(documents, 1):

        print("\n" + "=" * 50)
        print("Document:", i)
        print("Page:", doc["metadata"].get("page", "N/A"))
        print("Distance:", round(doc["distance"], 4))
        print("Text:", doc["text"][:500])