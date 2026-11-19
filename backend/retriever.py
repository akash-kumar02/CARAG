import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from transformers import AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "carag-docs"
ONNX_PATH = BASE_DIR / "onnx_model"


class ONNXEmbeddingModel:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(ONNX_PATH)
        )

        self.session = ort.InferenceSession(
            str(ONNX_PATH / "model.onnx"),
            providers=["CPUExecutionProvider"]
        )

        self.input_names = {
            item.name for item in self.session.get_inputs()
        }

        self.output_names = [
            item.name for item in self.session.get_outputs()
        ]

    def encode(self, texts, normalize_embeddings=True):

        if isinstance(texts, str):
            texts = [texts]

        embeddings = []

        for text in texts:

            inputs = self.tokenizer(
                text,
                return_tensors="np",
                truncation=True,
                max_length=512
            )

            onnx_inputs = {
                key: value.astype(np.int64)
                for key, value in inputs.items()
                if key in self.input_names
            }

            outputs = self.session.run(
                None,
                onnx_inputs
            )

            output_map = dict(
                zip(self.output_names, outputs)
            )

            if "sentence_embedding" in output_map:
                embedding = output_map[
                    "sentence_embedding"
                ][0]
            else:
                embedding = output_map[
                    "token_embeddings"
                ][0, 0, :]

            if normalize_embeddings:
                norm = np.linalg.norm(embedding)

                if norm > 0:
                    embedding = embedding / norm

            embeddings.append(embedding)

        return np.asarray(embeddings)


def load_retriever():

    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError(
            "QDRANT_URL and QDRANT_API_KEY are missing."
        )

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60
    )

    collection_info = client.get_collection(
        collection_name=COLLECTION_NAME
    )

    print("Qdrant Cloud connected successfully!")
    print("Stored chunks:", collection_info.points_count)

    model = ONNXEmbeddingModel()

    print("ONNX embedding model loaded successfully!")

    return client, model


def retrieve_documents(query, collection, model, k=5):

    if not query.strip() or k <= 0:
        return []

    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0].tolist()

    results = collection.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=k,
        with_payload=True,
        with_vectors=False
    ).points

    documents = []

    for point in results:

        payload = point.payload or {}

        documents.append({
            "text": payload.get("text", ""),
            "metadata": payload.get("metadata") or {},
            "distance": 1.0 - point.score
        })

    return documents


if __name__ == "__main__":

    collection, model = load_retriever()

    question = "What is an intelligent agent?"

    documents = retrieve_documents(
        question,
        collection,
        model,
        k=5
    )

    print("\nQuestion:", question)
    print("Retrieved Documents:", len(documents))

    for i, doc in enumerate(documents, 1):
        print(f"\nDocument {i}")
        print("Distance:", round(doc["distance"], 4))
        print("Text:", doc["text"][:300])