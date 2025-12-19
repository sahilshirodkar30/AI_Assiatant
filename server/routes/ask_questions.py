from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pydantic import Field
from typing import List, Optional
from logger import logger
from modules.llm import get_llm_chain
from modules.query_handler import query_chain
import os

router = APIRouter()

# -------------------------------------------------
# 🔥 INITIALIZE HEAVY OBJECTS ONCE (VERY IMPORTANT)
# -------------------------------------------------

logger.info("Loading embedding model...")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

logger.info("Connecting to Pinecone...")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

# -------------------------------------------------
# Retriever
# -------------------------------------------------

class SimpleRetriever(BaseRetriever):
    tags: Optional[List[str]] = Field(default_factory=list)
    metadata: Optional[dict] = Field(default_factory=dict)

    def __init__(self, documents: List[Document]):
        super().__init__()
        self._docs = documents

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self._docs

# -------------------------------------------------
# API Endpoint
# -------------------------------------------------

@router.post("/ask")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"User query: {question}")

        embedded_query = EMBED_MODEL.encode(question).tolist()

        res = index.query(
            vector=embedded_query,
            top_k=3,
            include_metadata=True
        )

        docs = [
            Document(
                page_content=match["metadata"].get("page_content", ""),
                metadata=match["metadata"]
            )
            for match in res["matches"]
        ]

        retriever = SimpleRetriever(docs)
        chain = get_llm_chain(retriever)

        result = query_chain(chain, question)

        logger.info("Query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
