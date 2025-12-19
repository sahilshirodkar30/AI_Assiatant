import os
import time
import shutil
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import logger
from modules.embedding_model import embedding_model

# -------------------------------------------------
# ENV
# -------------------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-index")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))  # MiniLM default

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------------------------
# Pinecone (lazy init)
# -------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

def get_or_create_index():
    existing = [i["name"] for i in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing:
        logger.info("Creating Pinecone index...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENV
            ),
        )

        while not pc.describe_index(PINECONE_INDEX_NAME).status.ready:
            time.sleep(1)

        logger.info("Pinecone index ready")

    return pc.Index(PINECONE_INDEX_NAME)

# -------------------------------------------------
# Utility
# -------------------------------------------------
def batch_data(data, size=32):
    for i in range(0, len(data), size):
        yield data[i:i + size]

# -------------------------------------------------
# Main loader
# -------------------------------------------------
def load_vectorstore(uploaded_files):
    """
    Loads PDFs, splits text, embeds in batches,
    and uploads vectors to Pinecone safely.
    """

    index = get_or_create_index()
    model = embedding_model

    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename

        logger.info(f"Saving {file.filename}")

        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        loader = PyPDFLoader(str(save_path))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [f"{save_path.stem}-{i}" for i in range(len(chunks))]

        vectors = []

        logger.info(f"Embedding {len(texts)} chunks from {file.filename}")

        for text_batch, id_batch, meta_batch in zip(
            batch_data(texts),
            batch_data(ids),
            batch_data(metadatas)
        ):
            embeddings = model.encode(
                text_batch,
                convert_to_numpy=True
            )

            for i in range(len(embeddings)):
                vectors.append(
                    (
                        id_batch[i],
                        embeddings[i].tolist(),
                        meta_batch[i]
                    )
                )

        logger.info("Uploading vectors to Pinecone")
        index.upsert(vectors=vectors)

        logger.info(f"Upload complete for {file.filename}")
