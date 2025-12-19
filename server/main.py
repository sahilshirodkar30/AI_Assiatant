from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from middleware.exception_handler import catch_exception_middleware
from routes.ask_questions import router as ask_questions_router
from routes.upload_files import router as upload_files_router
from logger import logger   # ✅ FIX 1: import logger

app = FastAPI(
    title="Medical Assistant API",
    description="Medical Assistant Chatbot"
)

# ✅ Log safely (no startup hook needed)
logger.info("🚀 Medical Assistant API initialized")

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Global Exception Middleware
# -----------------------------
app.middleware("http")(catch_exception_middleware)

# -----------------------------
# Routers
# -----------------------------
app.include_router(upload_files_router, prefix="/files")
app.include_router(ask_questions_router, prefix="/ask")

# -----------------------------
# Health Check (IMPORTANT for Render)
# -----------------------------
@app.get("/")
def health():
    return {"status": "ok"}
