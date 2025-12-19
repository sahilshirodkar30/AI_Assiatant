from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from logger import logger


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

def get_llm_chain(retriever):
    """
    Returns optimized RetrievalQA chain
    """

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",  # ✅ lighter model
        temperature=0
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are MediBot, an AI-powered assistant trained to help users understand medical documents.

Use ONLY the provided context to answer the question.

Context:
{context}

Question:
{question}

Answer Rules:
- Be concise and factual
- If answer not found, say:
  "I'm sorry, but I couldn't find relevant information in the provided documents."
- Do NOT give medical advice
"""
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return chain
