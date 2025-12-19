from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from logger import logger
from middleware.exception_handler import catch_exception_middleware
from routes.ask_questions import router as ask_questions_router
from routes.upload_files import router as upload_files_router

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing application resources")

    app.state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    app.state.pinecone_index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

    logger.info("Resources initialized")
    yield
    logger.info("Application shutdown")

app = FastAPI(
    title="Medical Assistant API",
    description="Medical Assistant Chatbot",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(catch_exception_middleware)

app.include_router(upload_files_router, prefix="/files")
app.include_router(ask_questions_router, prefix="/ask")

@app.get("/")
def health():
    return {"status": "ok"}
