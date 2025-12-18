from logger import logger

def query_chain(chain, user_input: str):
    try:
        logger.debug(f"Running chain for input: {user_input}")

        result = chain.invoke({"query": user_input})

        response = {
            "response": result["result"],
            "sources": [
                {
                    "file": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("page", ""),
                    "content": doc.page_content
                }
                for doc in result["source_documents"]
            ]
        }

        logger.debug(f"Chain response: {response}")
        return response

    except Exception:
        logger.exception("Error on query chain")
        raise
