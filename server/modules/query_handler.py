from logger import logger

def query_chain(chain, user_input: str):
    try:
        result = chain.invoke({"query": user_input})

        return {
            "response": result["result"],
            "sources": [
                {
                    "file": d.metadata.get("source", ""),
                    "page": d.metadata.get("page", ""),
                    "content": d.page_content
                }
                for d in result["source_documents"]
            ]
        }

    except Exception:
        logger.exception("Query chain failed")
        raise
