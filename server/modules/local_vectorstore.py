import os
import time
import shutil
from pathlib import Path
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from modules.embedding_model import embedding_model


# ────────────────────────────────
# ENV SETUP
# ────────────────────────────────
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medical-index"

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ────────────────────────────────
# PINECONE INIT (SAFE)
# ────────────────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY)

spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)

existing_indexes = [i["name"] for i in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=spec,
    )

    while not pc.describe_index(PINECONE_INDEX_NAME).status.ready:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)

# ────────────────────────────────
# UTILITY: BATCHING
# ────────────────────────────────
def batch_data(data, size=32):
    for i in range(0, len(data), size):
        yield data[i:i + size]

# ────────────────────────────────
# MAIN VECTORSTORE LOADER
# ────────────────────────────────
def load_vectorstore(uploaded_files):
    """
    Loads PDFs, splits text, embeds in batches,
    and uploads vectors to Pinecone safely.
    """

    model = embedding_model  # 🔒 reuse loaded model

    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename

        # ✅ STREAM FILE (no RAM spike)
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

        print(f"🔍 Embedding {len(texts)} chunks from {file.filename}")

        # ✅ EMBED IN SMALL BATCHES
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

        print("📤 Uploading vectors to Pinecone...")
        index.upsert(vectors=vectors)

        print(f"✅ Upload complete for {file.filename}")
