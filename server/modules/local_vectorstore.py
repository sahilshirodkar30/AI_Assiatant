import os
import time
import shutil
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import logger
from modules.embedding_model import get_embedding_model

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_or_create_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [i["name"] for i in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENV
            )
        )

        while not pc.describe_index(PINECONE_INDEX_NAME).status.ready:
            time.sleep(1)

    return pc.Index(PINECONE_INDEX_NAME)

def batch_data(data, size=32):
    for i in range(0, len(data), size):
        yield data[i:i + size]

def load_vectorstore(uploaded_files):
    index = get_or_create_index()
    model = get_embedding_model()

    for file in uploaded_files:
        path = Path(UPLOAD_DIR) / file.filename

        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        docs = PyPDFLoader(str(path)).load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(docs)

        texts = [c.page_content for c in chunks]
        metas = [c.metadata for c in chunks]
        ids = [f"{path.stem}-{i}" for i in range(len(chunks))]

        vectors = []

        for tb, ib, mb in zip(
            batch_data(texts),
            batch_data(ids),
            batch_data(metas)
        ):
            embeds = model.encode(tb, convert_to_numpy=True)
            for i in range(len(embeds)):
                vectors.append((ib[i], embeds[i].tolist(), mb[i]))

        index.upsert(vectors=vectors)
