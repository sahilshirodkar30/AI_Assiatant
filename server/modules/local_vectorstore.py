import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medical-index"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

UPLOAD_DIR= './uploaded_docs'
os.makedirs(UPLOAD_DIR, exist_ok=True)


#initize pincone instanse

pc = Pinecone(api_key=PINECONE_API_KEY)

spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)

if PINECONE_INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=spec,
    )

    while not pc.describe_index(PINECONE_INDEX_NAME).status.ready:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)


def load_vectorstore(uploaded_files):
    model = SentenceTransformer("all-mpnet-base-v2")

    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())

        loader = PyPDFLoader(str(save_path))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        texts = [c.page_content for c in chunks]
        metadatas = [
        {
        **c.metadata,
        "page_content": c.page_content
        }
            for c in chunks
        ]
        ids = [f"{save_path.stem}-{i}" for i in range(len(chunks))]

        print(f"🔍 Embedding {len(texts)} chunks...")
        embeddings = model.encode(texts, convert_to_numpy=True)

        # ✅ FIX: convert ndarray → list
        vectors = [
            (ids[i], embeddings[i].tolist(), metadatas[i])
            for i in range(len(ids))
        ]

        print("📤 Uploading to Pinecone...")
        index.upsert(vectors=vectors)

        print(f"✅ Upload complete for {save_path.name}")
