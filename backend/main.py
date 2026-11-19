from groq import RateLimitError
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from backend.retriever import load_retriever
from backend.rag_pipeline import run_carag


# Load models once when server starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    collection, model = load_retriever()
    app.state.collection = collection
    app.state.embedding_model = model
    yield


app = FastAPI(
    title="CARAG API",
    description="Corrective & Adaptive Agentic RAG",
    version="1.0.0",
    lifespan=lifespan
)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class ChatRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {
        "message": "CARAG Backend is running",
        "status": "success"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "rag": "ready"
    }


@app.post("/chat")
async def chat(request: ChatRequest):

    question = request.query.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        result = await run_in_threadpool(
            run_carag,
            question,
            app.state.collection,
            app.state.embedding_model
        )

        return {
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except RateLimitError as e:
        print("Groq rate limit exceeded:", repr(e))

        raise HTTPException(
            status_code=429,
            detail=(
                "Groq API rate limit exceeded. "
                "Please wait a moment and try again."
            )
        )

    except Exception as e:
        print("CARAG pipeline error:", repr(e))

        raise HTTPException(
            status_code=500,
            detail="Unable to generate answer"
        )