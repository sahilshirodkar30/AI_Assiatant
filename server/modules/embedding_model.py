from sentence_transformers import SentenceTransformer

# 🔒 Load ONCE at startup (critical for memory)
embedding_model = SentenceTransformer("all-mpnet-base-v2")
