from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List, Optional
from pydantic import Field
from logger import logger
from modules.llm import get_llm_chain
from modules.query_handler import query_chain

router = APIRouter()

class SimpleRetriever(BaseRetriever):
    tags: Optional[List[str]] = Field(default_factory=list)
    metadata: Optional[dict] = Field(default_factory=dict)

    def __init__(self, documents: List[Document]):
        super().__init__()
        self._docs = documents

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self._docs

@router.post("/ask")
async def ask_question(request: Request, question: str = Form(...)):
    try:
        embed_model = request.app.state.embed_model
        index = request.app.state.pinecone_index

        embedded_query = embed_model.encode(question).tolist()

        res = index.query(
            vector=embedded_query,
            top_k=3,
            include_metadata=True
        )

        docs = [
            Document(
                page_content=m["metadata"].get("page_content", ""),
                metadata=m["metadata"]
            )
            for m in res["matches"]
        ]

        retriever = SimpleRetriever(docs)
        chain = get_llm_chain(retriever)

        return query_chain(chain, question)

    except Exception as e:
        logger.exception("Query failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
